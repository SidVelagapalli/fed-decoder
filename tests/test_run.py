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
    monkeypatch.setattr(anthropic, "Anthropic", lambda: fake)

    run.main(["--dates", "2020-01-29", "2020-03-03", "2020-03-03", "--go"])  # duplicate date runs once

    (log,) = (tmp_path / "logs").iterdir()
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["date"] for r in records] == ["2020-01-29", "2020-03-03"]
    assert len(fake.calls) == 4  # extract + explain per statement
    assert "[-Alpha-] {+Beta+}" in fake.calls[2]["messages"][0]["content"]  # 2nd extract diffs against row 1
    assert records[-1]["tokens_so_far"] == {"input": 40, "output": 8}
    assert "actual cost $0.0001" in capsys.readouterr().out
