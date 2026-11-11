"""Rung 2 eval: does the stance predict the Fed's NEXT decision and the same-day 2-year yield move?
All four methods are scored on the same meetings (those with a next decision AND an ok decoder record).
  uv run python -m src.eval [--model openai/gpt-oss-20b]
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from src.extract import MODEL

LABELS = Path("data/labels.csv")
STATEMENTS = Path("data/statements.csv")
LOG_DIR = Path("logs")
CLASSES = ("hike", "hold", "cut")
SIGN = {"hike": 1, "hold": 0, "cut": -1}
# Chosen by Siddarth before seeing results, not tuned: stance beyond +-0.2 counts as a call.
THRESHOLD = 0.2
MIN_N = 30  # below this, the table is noise, not a result
# Fixed before seeing results. No phrase is a substring of one in the other list.
HAWK_WORDS = (
    "inflation pressures", "upside risks to inflation", "firming", "raise the target", "increase the target",
    "tightening", "reduce its holdings", "elevated inflation", "strong", "robust", "solid",
    "heightened inflation", "higher inflation", "overheating", "expanding rapidly",
)
DOVE_WORDS = (
    "accommodative", "accommodation", "lower the target", "reduce the target", "decrease the target", "slack",
    "downside risks", "weak", "subdued", "purchases", "low inflation", "easing", "moderated", "deteriorat",
    "softened",
)


def load_scores(log_dir: Path, model: str) -> dict[str, float]:
    """Newest ok score per date for this model: later log file wins, then later line in the file."""
    scores = {}
    for log in sorted(log_dir.glob("run-*.jsonl")):
        for n, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("status") == "ok" and record.get("model") == model:
                    scores[record["date"]] = record["score"]
            except (json.JSONDecodeError, KeyError):
                print(f"WARNING: unreadable line {n} in {log}; skipped", file=sys.stderr)  # keep stdout a clean table
    return {d: s for d, s in scores.items() if s is not None}


def word_stance(text: str) -> float:
    text = text.lower()
    hawk = sum(text.count(w) for w in HAWK_WORDS)
    dove = sum(text.count(w) for w in DOVE_WORDS)
    return (hawk - dove) / (hawk + dove) if hawk + dove else 0.0


def predict(stance: float) -> str:
    return "hike" if stance > THRESHOLD else "cut" if stance < -THRESHOLD else "hold"


def macro_f1(true: list[str], pred: list[str]) -> float:
    f1s = []
    for c in CLASSES:
        tp = sum(t == c and p == c for t, p in zip(true, pred))
        n_pred, n_true = pred.count(c), true.count(c)
        if n_pred == 0 and n_true == 0:
            continue  # class never appears: skip rather than count a free 0
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def spearman(x: list[float], y: list[float]) -> float | None:
    try:
        return statistics.correlation(x, y, method="ranked")
    except statistics.StatisticsError:  # constant stance (or <2 points): correlation undefined
        return None


def evaluate(labels: list[dict], texts: dict[str, str], scores: dict[str, float]):
    """Returns (meetings used, meetings with a yield move, rows of (method, accuracy, macro-F1, spearman, calls))."""
    meetings = [r for r in labels if r["next_decision"] and r["date"] in scores]
    true = [r["next_decision"] for r in meetings]
    stances = {
        "Always hold": [0.0 for _ in meetings],
        "Persistence": [float(SIGN[r["decision"]]) for r in meetings],
        "Word list (no LLM)": [word_stance(texts[r["date"]]) for r in meetings],
        "Fed Speak Decoder": [scores[r["date"]] for r in meetings],
    }
    preds = {
        "Always hold": ["hold"] * len(meetings),
        "Persistence": [r["decision"] for r in meetings],
    }
    with_yield = [i for i, r in enumerate(meetings) if r["dgs2_change_bp"]]
    moves = [float(meetings[i]["dgs2_change_bp"]) for i in with_yield]
    rows = []
    for name, stance in stances.items():
        pred = preds.get(name) or [predict(s) for s in stance]
        acc = sum(t == p for t, p in zip(true, pred)) / len(true)
        # calls = hike/hold/cut counts: the same threshold can mean very different call rates per method
        calls = "/".join(str(pred.count(c)) for c in CLASSES)
        rows.append((name, acc, macro_f1(true, pred), spearman([stance[i] for i in with_yield], moves), calls))
    return len(meetings), len(with_yield), rows


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL, help="whose log records to evaluate")
    args = parser.parse_args(argv)

    scores = load_scores(LOG_DIR, args.model)
    if not scores:
        raise SystemExit(f"No ok records for {args.model} in {LOG_DIR}/; nothing to evaluate.")
    labels = list(csv.DictReader(LABELS.open(encoding="utf-8")))
    texts = {r["date"]: r["text"] for r in csv.DictReader(STATEMENTS.open(encoding="utf-8"))}
    n, n_yield, rows = evaluate(labels, texts, scores)
    if n == 0:
        raise SystemExit("No meeting has both a next decision and an ok record.")

    print(f"Model: {args.model}")
    print(f"Meetings scored: {n} (yield correlation uses {n_yield})")
    if n < MIN_N:
        print(f"WARNING: only {n} meetings; too few to report these numbers", file=sys.stderr)
    print(f"Statements without an ok record: {len(set(texts) - set(scores))} of {len(texts)}")
    print()
    print("| Method | Next-decision accuracy | Macro-F1 | Spearman vs 2y yield move (same day) | Calls hike/hold/cut |")
    print("|---|---|---|---|---|")
    for name, acc, f1, rho, calls in rows:
        print(f"| {name} | {fmt(acc)} | {fmt(f1)} | {fmt(rho)} | {calls} |")


if __name__ == "__main__":
    main()
