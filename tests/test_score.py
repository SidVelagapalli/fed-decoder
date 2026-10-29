# Weights are a design choice; if you change them in score.py, update the exact-value tests.
from src.score import score

HAWK = {"inflation_concern": "high", "growth_view": "strong", "labor_view": "strong",
        "guidance_direction": "tightening", "balance_sheet": "shrinking", "dissents": "hawkish"}
DOVE = {"inflation_concern": "low", "growth_view": "weak", "labor_view": "weak",
        "guidance_direction": "easing", "balance_sheet": "expanding", "dissents": "dovish"}
NEUTRAL = {"inflation_concern": "moderate", "growth_view": "moderate", "labor_view": "moderate",
           "guidance_direction": "neutral", "balance_sheet": "steady", "dissents": "none"}


def sig(base=NEUTRAL, **levels):
    """Build a signals dict from a base of levels, overriding some fields."""
    return {f: {"level": l, "evidence_quote": "q"} for f, l in {**base, **levels}.items()}


def test_all_neutral_scores_zero():
    assert score(sig())["score"] == 0.0


def test_all_hawkish_scores_plus_one():
    assert score(sig(HAWK))["score"] == 1.0


def test_all_dovish_scores_minus_one():
    assert score(sig(DOVE))["score"] == -1.0


def test_score_always_in_range():
    for base in (HAWK, DOVE, NEUTRAL):
        assert -1 <= score(sig(base))["score"] <= 1
    mixed = sig(guidance_direction="tightening", inflation_concern="low")
    assert score(mixed)["score"] == 0.067


def test_not_mentioned_counts_zero():
    s = sig(NEUTRAL)
    s["guidance_direction"] = {"level": "not_mentioned", "evidence_quote": ""}
    assert score(s) == {"score": 0.0, "rules_fired": []}


def test_missing_field_does_not_crash():
    s = sig(HAWK)
    del s["guidance_direction"]
    assert score(s)["score"] == round(5.5 / 7.5, 3)
    assert score({})["score"] == 0.0


def test_unknown_level_counts_zero():
    assert score(sig(growth_view="banana"))["score"] == 0.0


def test_invalid_signal_ignored():
    s = sig(HAWK)
    s["guidance_direction"] = {"level": "not_mentioned", "evidence_quote": "", "invalid": "bad quote"}
    s["growth_view"] = "not a dict"
    assert score(s)["score"] == round(4.5 / 7.5, 3)


def test_guidance_weighs_more_than_growth():
    g = score(sig(guidance_direction="tightening"))["score"]
    gr = score(sig(growth_view="strong"))["score"]
    assert g > gr > 0


def test_rules_fired_lists_only_nonzero_fields():
    r = score(sig(growth_view="strong", labor_view="weak"))["rules_fired"]
    assert len(r) == 2
    assert all("inflation_concern" not in x for x in r)


def test_rules_fired_order_and_format():
    r = score(sig(guidance_direction="tightening", inflation_concern="low"))["rules_fired"]
    assert r == ["DOVE inflation_concern=low (-1.5)", "HAWK guidance_direction=tightening (+2.0)"]


def test_deterministic():
    s = sig(HAWK, inflation_concern="low")
    assert score(s) == score(s)
