import json
from types import SimpleNamespace

import pytest

from src import run
from src.extract import FIELDS
from tests.conftest import FakeClient

ALL_NOT_MENTIONED = json.dumps({f: {"level": "not_mentioned", "evidence_quote": ""} for f in FIELDS})


def test_run_one_extracts_scores_and_explains():
    client = FakeClient(ALL_NOT_MENTIONED)
    rec = run.run_one(client, "old text", {"date": "2020-01-29", "url": "u", "text": "new text"})
    assert len(client.calls) == 2  # one extract, one explain
    assert rec["score"] == 0.0 and rec["rules_fired"] == []
    assert set(rec["signals"]) == set(FIELDS)


def test_dry_run_makes_no_calls_and_writes_no_log(tmp_path, monkeypatch, capsys):
    statements = tmp_path / "statements.csv"
    statements.write_text("date,url,text\n2020-01-29,u,Some text.\n", encoding="utf-8")
    monkeypatch.setattr(run, "STATEMENTS", statements)
    monkeypatch.setattr(run, "LOG_DIR", tmp_path / "logs")
    run.main(["--dates", "2020-01-29"])  # the conftest guard fails this if a real client is built
    assert "Dry run" in capsys.readouterr().out
    assert not (tmp_path / "logs").exists()


def test_unknown_date_stops_before_any_call(tmp_path, monkeypatch):
    statements = tmp_path / "statements.csv"
    statements.write_text("date,url,text\n2020-01-29,u,Some text.\n", encoding="utf-8")
    monkeypatch.setattr(run, "STATEMENTS", statements)
    with pytest.raises(SystemExit, match="1999-01-01"):
        run.main(["--dates", "1999-01-01", "--go"])


class MeteredFakeClient(FakeClient):
    def _create(self, **kwargs):
        response = super()._create(**kwargs)
        response.usage = SimpleNamespace(input_tokens=10, output_tokens=2)
        return response


def test_go_logs_each_statement_with_previous_text_and_tokens(tmp_path, monkeypatch, capsys):
    import anthropic

    statements = tmp_path / "statements.csv"
    statements.write_text("date,url,text\n2020-01-29,u1,Alpha text.\n2020-03-03,u2,Beta text.\n", encoding="utf-8")
    monkeypatch.setattr(run, "STATEMENTS", statements)
    monkeypatch.setattr(run, "LOG_DIR", tmp_path / "logs")
    fake = MeteredFakeClient(ALL_NOT_MENTIONED)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake)

    run.main(["--dates", "2020-01-29", "2020-03-03", "2020-03-03", "--go", "--provider", "anthropic"])  # duplicate date runs once

    (log,) = (tmp_path / "logs").iterdir()
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["date"] for r in records] == ["2020-01-29", "2020-03-03"]
    assert len(fake.calls) == 4  # extract + explain per statement
    assert "[-Alpha-] {+Beta+}" in fake.calls[2]["messages"][0]["content"]  # 2nd extract diffs against row 1
    assert records[-1]["tokens_so_far"] == {"input": 40, "output": 8}
    assert "actual cost $0.0001" in capsys.readouterr().out

    # Resume: a second identical run pays for nothing; --redo runs the date again.
    run.main(["--dates", "2020-01-29", "2020-03-03", "--go", "--provider", "anthropic"])
    assert len(fake.calls) == 4
    assert "Skipping 2 dates" in capsys.readouterr().out
    run.main(["--dates", "2020-03-03", "--go", "--redo", "--provider", "anthropic"])
    assert len(fake.calls) == 6


def test_failed_extract_is_not_a_neutral_score():
    client = FakeClient("not json")
    rec = run.run_one(client, "", {"date": "2020-01-29", "url": "u", "text": "new text"})
    assert rec["status"] == "extract_failed" and rec["score"] is None
    assert len(client.calls) == 1  # no paid explain call for nothing


def test_done_dates_counts_only_ok_records_and_survives_a_cut_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "LOG_DIR", tmp_path)
    (tmp_path / "run-1.jsonl").write_text(
        '{"date": "2020-01-29", "status": "ok", "model": "m"}\n'
        '{"date": "2020-03-03", "status": "extract_failed", "model": "m"}\n'
        '{"date": "2020-03-15"}\n'  # pilot record from before status/model existed
        '{"date": "2020-03-18", "status": "ok", "model": "other-model"}\n'
        '{"date": "2020-04-29", "sta', encoding="utf-8")
    assert run.done_dates("m") == {"2020-01-29"}
    assert "unreadable line 5" in capsys.readouterr().out


class FakeAPIError(__import__("anthropic").APIError):
    def __init__(self):
        Exception.__init__(self, "overloaded")


def test_api_error_logs_date_and_keeps_going(tmp_path, monkeypatch, capsys):
    import anthropic

    statements = tmp_path / "statements.csv"
    statements.write_text("date,url,text\n2020-01-29,u1,Alpha text.\n2020-03-03,u2,Beta text.\n", encoding="utf-8")
    monkeypatch.setattr(run, "STATEMENTS", statements)
    monkeypatch.setattr(run, "LOG_DIR", tmp_path / "logs")
    fake = MeteredFakeClient(ALL_NOT_MENTIONED)
    first = True

    def flaky(**kwargs):
        nonlocal first
        if first:
            first = False
            raise FakeAPIError()
        return MeteredFakeClient._create(fake, **kwargs)

    fake.messages.create = flaky
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake)

    with pytest.raises(SystemExit, match="1 dates not ok"):
        run.main(["--dates", "2020-01-29", "2020-03-03", "--go", "--provider", "anthropic"])
    (log,) = (tmp_path / "logs").iterdir()
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in records] == ["error", "ok"]
    assert "actual cost" in capsys.readouterr().out
    assert run.done_dates(run.extract_mod.MODEL) == {"2020-03-03"}  # the errored date is retried next run


def test_groq_is_default_and_daily_limit_stops_cleanly(tmp_path, monkeypatch, capsys):
    statements = tmp_path / "statements.csv"
    statements.write_text("date,url,text\n2020-01-29,u1,Alpha text.\n2020-03-03,u2,Beta text.\n", encoding="utf-8")
    monkeypatch.setattr(run, "STATEMENTS", statements)
    monkeypatch.setattr(run, "LOG_DIR", tmp_path / "logs")
    fake = MeteredFakeClient(ALL_NOT_MENTIONED)
    real_create = fake.messages.create

    def limited(**kwargs):
        if len(fake.calls) == 2:  # first statement done, then the free daily quota runs out
            raise run.groq_client.DailyLimitReached("Groq daily limit reached")
        return real_create(**kwargs)

    fake.messages.create = limited
    monkeypatch.setattr(run.groq_client, "GroqClient", lambda: fake)

    with pytest.raises(SystemExit, match="Run the same command tomorrow"):
        run.main(["--dates", "2020-01-29", "2020-03-03", "--go"])  # no --provider: Groq
    out = capsys.readouterr().out
    assert "(free)" in out and "free" in out.splitlines()[-2]
    (log,) = (tmp_path / "logs").iterdir()
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [(r["date"], r["status"], r["model"]) for r in records] == [("2020-01-29", "ok", "openai/gpt-oss-20b")]
    assert run.done_dates("openai/gpt-oss-20b") == {"2020-01-29"}  # tomorrow's run starts at 2020-03-03
