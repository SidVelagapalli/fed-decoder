"""Turn extracted stance signals into a hawk/dove score in [-1, 1] plus the rules that fired.

Code decides, not the LLM: plain rules, no randomness, no I/O, so the score can't be prompt-injected and is unit-tested.
"""

# Level -> +1 hawkish / -1 dovish. Any other level (moderate, neutral, not_mentioned, typos) counts 0.
DIRECTION = {
    "inflation_concern": {"high": 1, "low": -1},
    "growth_view": {"strong": 1, "weak": -1},
    "labor_view": {"strong": 1, "weak": -1},
    "guidance_direction": {"tightening": 1, "easing": -1},
    "balance_sheet": {"shrinking": 1, "expanding": -1},
    "dissents": {"hawkish": 1, "dovish": -1},
}
# Draft weights: forward guidance matters most, then inflation.
WEIGHTS = {
    "inflation_concern": 1.5,
    "growth_view": 1.0,
    "labor_view": 1.0,
    "guidance_direction": 2.0,
    "balance_sheet": 1.0,
    "dissents": 1.0,
}
TOTAL = sum(WEIGHTS.values())  # 7.5: dividing by the max possible keeps the score in [-1, 1]


def score(signals: dict) -> dict:
    total = 0.0
    rules_fired = []
    for field, weight in WEIGHTS.items():  # schema order
        signal = signals.get(field)
        # Missing, malformed or validation-flagged fields count 0.
        if not isinstance(signal, dict) or "invalid" in signal:
            continue
        level = signal.get("level")
        d = DIRECTION[field].get(level, 0) if isinstance(level, str) else 0  # a list level must not crash
        if d:
            total += d * weight
            rules_fired.append(f"{'HAWK' if d > 0 else 'DOVE'} {field}={level} ({d * weight:+.1f})")
    return {"score": round(total / TOTAL, 3), "rules_fired": rules_fired}
