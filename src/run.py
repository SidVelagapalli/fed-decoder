"""Run the pipeline on chosen statements: diff -> extract (LLM) -> score (rules) -> explain (LLM).

Default provider is Groq (free tier, openai/gpt-oss-20b); --provider anthropic uses Anthropic Haiku (paid).
Dry run by default: prints the plan and estimate, builds no API client. Add --go to make the calls.
  uv run python -m src.run --dates 2000-02-02 2015-12-16          # dry run
  uv run python -m src.run --dates 2000-02-02 2015-12-16 --go     # 2 calls per statement
Each statement's result is appended to logs/run-<timestamp>.jsonl as soon as it's done, so a crash loses nothing.
Dates that already have an ok result from the same model are skipped (resume); --redo runs them again.
Readers should use the newest ok record per date and model. On Groq, hitting the free daily limit stops the run
cleanly; run the same command the next day to continue.
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import httpx

from src import extract as extract_mod
from src import groq_client
from src.diff import redline
from src.explain import explain
from src.extract import extract
from src.score import score

STATEMENTS = Path("data/statements.csv")
LOG_DIR = Path("logs")
PRICE = {"input": 1.00, "output": 5.00}  # Haiku 4.5, $ per million tokens; Groq free tier is $0
# Per statement (extract + explain), measured on the 5-statement pilots (Haiku 2026-11-01, gpt-oss-20b 2026-11-06).
EST_TOKENS = {"anthropic": {"input": 3700, "output": 470}, "groq": {"input": 3100, "output": 550}}
GROQ_TOKENS_PER_DAY = 200_000  # free tier, gpt-oss-20b; shared with every other project using the key


class UsageMeter:
    """Wraps the client so every messages.create call adds its real token usage to the totals."""

    def __init__(self, client):
        self.client, self.messages = client, self
        self.input = self.output = 0

    def create(self, **kwargs):
        response = self.client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input += usage.input_tokens
            self.output += usage.output_tokens
        return response


def cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE["input"] + output_tokens * PRICE["output"]) / 1_000_000


def done_dates(model: str) -> set[str]:
    """Dates with an ok result from this model. Failed, errored, pre-status or other-model records don't count."""
    done = set()
    for log in LOG_DIR.glob("run-*.jsonl"):
        for n, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("status") == "ok" and record.get("model") == model:
                    done.add(record["date"])
            except (json.JSONDecodeError, KeyError):  # e.g. a line cut off by a crash mid-write
                print(f"WARNING: unreadable line {n} in {log}; that date will be re-run")
    return done


def run_one(client, prev_text: str, row: dict) -> dict:
    diff = redline(prev_text, row["text"])
    signals = extract(client, row["text"], diff)
    invalid = [f for f, s in signals.items() if "invalid" in s]
    base = {"date": row["date"], "url": row["url"], "signals": signals, "invalid_fields": invalid}
    if len(invalid) == len(signals):
        # Nothing usable came back; a 0.0 here would look like a real neutral statement. Skip the explain call too.
        return {**base, "status": "extract_failed", "score": None, "rules_fired": [], "explanation": ""}
    result = score(signals)  # code decides; the LLM only gathered the evidence above
    explanation = explain(client, result, signals, diff)
    return {**base, **result, "explanation": explanation, "status": "ok" if explanation else "explain_failed"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dates", nargs="+", required=True, help="statement dates, YYYY-MM-DD")
    parser.add_argument("--provider", choices=["groq", "anthropic"], default="groq")
    parser.add_argument("--go", action="store_true", help="make the API calls")
    parser.add_argument("--redo", action="store_true", help="re-run dates that already have an ok result in logs/")
    args = parser.parse_args(argv)
    dates = list(dict.fromkeys(args.dates))  # a repeated date would be run twice
    model = groq_client.MODEL if args.provider == "groq" else extract_mod.MODEL

    rows = list(csv.DictReader(STATEMENTS.open(encoding="utf-8")))
    index = {r["date"]: i for i, r in enumerate(rows)}
    missing = [d for d in dates if d not in index]
    if missing:
        raise SystemExit(f"not in {STATEMENTS}: {missing}")

    if not args.redo:
        # Resume: don't redo dates this model already handled after a crash, a daily limit or a rerun.
        done = done_dates(model)
        skipped = [d for d in dates if d in done]
        dates = [d for d in dates if d not in done]
        if skipped:
            print(f"Skipping {len(skipped)} dates already done by {model} in {LOG_DIR}/ (use --redo to re-run them)")

    n = len(dates)
    est = EST_TOKENS[args.provider]
    tokens = n * (est["input"] + est["output"])
    if args.provider == "groq":
        days = math.ceil(tokens / GROQ_TOKENS_PER_DAY) if n else 0
        print(f"{n} statements -> {2 * n} calls to {model} (free), ~{tokens:,} tokens; "
              f"free tier {GROQ_TOKENS_PER_DAY:,}/day -> about {days} day(s) with resume")
    else:
        print(f"{n} statements -> {2 * n} calls to {model}, estimated ~${cost(n * est['input'], n * est['output']):.3f}")
    if not args.go:
        print("Dry run: no API calls made. Add --go to run.")
        return

    import anthropic  # only for the anthropic client and its error class; importing builds nothing

    if args.provider == "groq":
        client = UsageMeter(groq_client.GroqClient())
    else:
        client = UsageMeter(anthropic.Anthropic(timeout=60.0))  # a hang shouldn't look like a slow run
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    not_ok, stopped = [], None
    try:
        for date in dates:
            i = index[date]
            try:
                record = run_one(client, rows[i - 1]["text"] if i else "", rows[i])
            except groq_client.DailyLimitReached as e:
                stopped = str(e)  # not this date's fault: stop, and resume tomorrow
                break
            except (anthropic.APIError, httpx.HTTPError) as e:
                # One failed date (after retries) shouldn't end the run; it's logged and retried next time.
                record = {"date": date, "status": "error", "error": f"{type(e).__name__}: {e}"}
            record["model"] = model
            record["tokens_so_far"] = {"input": client.input, "output": client.output}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if record["status"] != "ok":
                not_ok.append(date)
            score_txt = "  n/a " if record.get("score") is None else f"{record['score']:+.3f}"
            print(f"{date}  {record['status']:14}  score {score_txt}  invalid {record.get('invalid_fields') or '-'}")
    finally:  # print usage even if the run is interrupted
        spent = "free" if args.provider == "groq" else f"actual cost ${cost(client.input, client.output):.4f}"
        print(f"Tokens in/out: {client.input}/{client.output}  {spent}")
        print(f"Log: {log_path}")
    if stopped:
        raise SystemExit(f"{stopped}. Run the same command tomorrow; finished dates are skipped.")
    if not_ok:
        raise SystemExit(f"{len(not_ok)} dates not ok (re-run the same command to retry them): {not_ok}")


if __name__ == "__main__":
    main()
