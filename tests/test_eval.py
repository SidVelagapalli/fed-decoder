import json

import pytest

from src.eval import (DOVE_WORDS, HAWK_WORDS, THRESHOLD, evaluate, load_scores, macro_f1, predict, spearman,
                      word_stance)


def write_log(path, *records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def rec(date, score, status="ok", model="m"):
    return {"date": date, "score": score, "status": status, "model": model}


def test_newest_ok_record_wins(tmp_path):
    write_log(tmp_path / "run-1.jsonl", rec("2020-01-01", 0.1), rec("2020-02-01", 0.5))
    write_log(tmp_path / "run-2.jsonl", rec("2020-01-01", 0.9, status="explain_failed"), rec("2020-01-01", 0.3))
    assert load_scores(tmp_path, "m") == {"2020-01-01": 0.3, "2020-02-01": 0.5}


def test_failed_newer_record_does_not_replace_older_ok(tmp_path):
    write_log(tmp_path / "run-1.jsonl", rec("2020-01-01", 0.1))
    write_log(tmp_path / "run-2.jsonl", rec("2020-01-01", None, status="error"))
    assert load_scores(tmp_path, "m") == {"2020-01-01": 0.1}


def test_other_model_ignored_and_bad_line_warns(tmp_path, capsys):
    (tmp_path / "run-1.jsonl").write_text(
        json.dumps(rec("2020-01-01", 0.1, model="other")) + "\n{cut off\n", encoding="utf-8")
    assert load_scores(tmp_path, "m") == {}
    assert "WARNING" in capsys.readouterr().err


def test_predict_threshold_boundaries():
    assert THRESHOLD == 0.2
    assert predict(0.2) == "hold" and predict(-0.2) == "hold"
    assert predict(0.201) == "hike" and predict(-0.201) == "cut"


def test_word_stance_counts_and_zero_hits():
    # 2 hawk hits (strong x2), 1 dove hit (weak) -> (2-1)/3
    assert word_stance("Strong output, STRONG jobs, weak prices.") == pytest.approx(1 / 3)
    assert word_stance("Nothing relevant here.") == 0.0


def test_word_lists_do_not_overlap():
    assert not [(h, d) for h in HAWK_WORDS for d in DOVE_WORDS if h in d or d in h]


def test_macro_f1_hand_checked():
    true = ["hike", "hike", "hold", "cut"]
    pred = ["hike", "hold", "hold", "hold"]
    # hike: p=1, r=.5 -> 2/3; hold: p=1/3, r=1 -> .5; cut: no correct, f1=0
    assert macro_f1(true, pred) == pytest.approx((2 / 3 + 0.5 + 0) / 3)
    assert macro_f1(["hold", "hold"], ["hold", "hold"]) == 1.0  # hike and cut skipped


def test_constant_stance_correlation_is_na():
    assert spearman([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) is None


def labels():
    return [
        {"date": "d1", "decision": "hike", "next_decision": "hold", "dgs2_change_bp": "3.0"},
        {"date": "d2", "decision": "hold", "next_decision": "cut", "dgs2_change_bp": ""},
        {"date": "d3", "decision": "cut", "next_decision": "", "dgs2_change_bp": "1.0"},
        {"date": "d4", "decision": "cut", "next_decision": "cut", "dgs2_change_bp": "-4.0"},
    ]


def test_evaluate_excludes_blank_next_decision_and_uses_this_meetings_decision():
    texts = {d: "strong" for d in ("d1", "d2", "d3", "d4")}
    scores = {"d1": 0.5, "d2": -0.5, "d3": 0.9, "d4": -0.5}
    n, n_yield, rows = evaluate(labels(), texts, scores)
    assert (n, n_yield) == (3, 2)  # d3 has no next decision; d2 has no yield move
    by_name = {r[0]: r for r in rows}
    assert by_name["Always hold"][1] == pytest.approx(1 / 3)
    assert by_name["Always hold"][3] is None
    # persistence predicts hike, hold, cut vs true hold, cut, cut -> only d4 right
    assert by_name["Persistence"][1] == pytest.approx(1 / 3)
    assert by_name["Fed Speak Decoder"][1] == pytest.approx(2 / 3)  # hike miss, cut, cut
    assert by_name["Fed Speak Decoder"][3] == pytest.approx(1.0)  # d1 high/+3bp, d4 low/-4bp
    assert by_name["Fed Speak Decoder"][4] == "1/0/2"  # hike/hold/cut call counts


def test_rescore_rechecks_rejected_answers_offline(tmp_path):
    from src.extract import FIELDS
    signals = {f: {"level": "not_mentioned", "evidence_quote": ""} for f in FIELDS}
    signals["guidance_direction"] = {"level": "not_mentioned", "evidence_quote": "", "invalid": "quote not in statement",
                                     "rejected": {"level": "tightening", "evidence_quote": "Further Firming"}}
    write_log(tmp_path / "run-1.jsonl", {**rec("d1", 0.0), "signals": signals})
    assert load_scores(tmp_path, "m") == {"d1": 0.0}  # logged score untouched
    rescored = load_scores(tmp_path, "m", {"d1": "Some further firming may be needed."})
    assert rescored == {"d1": pytest.approx(2.0 / 7.5, abs=1e-3)}  # guidance tightening now counts
