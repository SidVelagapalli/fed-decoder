"""Run the pipeline on chosen statements: diff -> extract (Haiku) -> score (rules) -> explain (Haiku).

Dry run by default: prints the plan and estimated cost, builds no API client. Add --go to make the paid calls.
  uv run python -m src.run --dates 2000-02-02 2015-12-16          # dry run, free
  uv run python -m src.run --dates 2000-02-02 2015-12-16 --go     # 2 calls per statement, paid
Each statement's result is appended to logs/run-<timestamp>.jsonl as soon as it's done, so a crash loses nothing.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from src.diff import redline
from src.explain import explain
from src.extract import MODEL, extract
from src.score import score

STATEMENTS = Path("data/statements.csv")
LOG_DIR = Path("logs")
PRICE = {"input": 1.00, "output": 5.00}  # Haiku 4.5, $ per million tokens
# Per statement (extract + explain), from the Checkpoint 2 estimate over all 223 real prompts.
EST_TOKENS = {"input": 3300, "output": 550}


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


def run_one(client, prev_text: str, row: dict) -> dict:
    diff = redline(prev_text, row["text"])
    signals = extract(client, row["text"], diff)
    result = score(signals)  # code decides; the LLM only gathered the evidence above
    return {"date": row["date"], "url": row["url"], **result, "signals": signals,
            "explanation": explain(client, result, signals, diff)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dates", nargs="+", required=True, help="statement dates, YYYY-MM-DD")
    parser.add_argument("--go", action="store_true", help="make the paid API calls")
    args = parser.parse_args(argv)
    dates = list(dict.fromkeys(args.dates))  # a repeated date would be paid for twice

    rows = list(csv.DictReader(STATEMENTS.open(encoding="utf-8")))
    index = {r["date"]: i for i, r in enumerate(rows)}
    missing = [d for d in dates if d not in index]
    if missing:
        raise SystemExit(f"not in {STATEMENTS}: {missing}")

    n = len(dates)
    est = cost(n * EST_TOKENS["input"], n * EST_TOKENS["output"])
    print(f"{n} statements -> {2 * n} calls to {MODEL}, estimated ~${est:.3f}")
    if not args.go:
        print("Dry run: no API calls made. Add --go to run.")
        return

    import anthropic  # only here, so a dry run never builds a client

    client = UsageMeter(anthropic.Anthropic())
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    for date in dates:
        i = index[date]
        record = run_one(client, rows[i - 1]["text"] if i else "", rows[i])
        invalid = [f for f, s in record["signals"].items() if "invalid" in s]
        record["tokens_so_far"] = {"input": client.input, "output": client.output}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{date}  score {record['score']:+.3f}  rules {len(record['rules_fired'])}  invalid {invalid or '-'}")
    print(f"Tokens in/out: {client.input}/{client.output}  actual cost ${cost(client.input, client.output):.4f}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
