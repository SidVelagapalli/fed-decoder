from pathlib import Path
from types import SimpleNamespace

import pytest

from src import fetch

FIX = Path(__file__).parent / "fixtures"
NEW_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260128a.htm"
OLD_URL = "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050202/"


def read(name):
    return (FIX / name).read_text()


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "RAW_DIR", tmp_path)

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(fetch.httpx, "get", boom)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    return tmp_path


def test_statement_links_keeps_only_statement_pages():
    links = fetch.statement_links(read("index.html"))
    assert links == [
        NEW_URL,
        OLD_URL,
        "https://www.federalreserve.gov/boarddocs/press/general/2001/20010103/default.htm",
    ]


def test_statement_links_absolute_and_deduped():
    links = fetch.statement_links(read("index.html"))
    assert all(u.startswith("https://www.federalreserve.gov/") for u in links)
    assert links.count(NEW_URL) == 1


def test_statement_links_calendar_html_and_2006_2010_format():
    html = """
    <p>Statement: <a href="/monetarypolicy/files/monetary20260128a1.pdf">PDF</a> |
       <a href="/newsevents/pressreleases/monetary20260128a.htm">HTML</a></p>
    <p>Minutes: <a href="/monetarypolicy/fomcminutes20260128.htm">HTML</a></p>
    <a href="/newsevents/press/monetary/20080122b.htm">Statement</a>
    <a href="/newsevents/pressreleases/monetary20190130b.htm">Statement on Longer-Run Goals</a>
    """
    assert fetch.statement_links(html) == [
        NEW_URL,
        "https://www.federalreserve.gov/newsevents/press/monetary/20080122b.htm",
    ]


def test_parse_new_format_text_and_date():
    r = fetch.parse_statement(read("statement_new.html"), NEW_URL)
    assert r["date"] == "2026-01-28"
    assert r["url"] == NEW_URL
    paras = r["text"].split("\n\n")
    assert paras[0].startswith("The Committee judges that economic activity has continued")
    assert "  " not in r["text"] and "\n " not in r["text"]
    assert "maintain the target range" in r["text"]


def test_parse_old_boarddocs_format():
    r = fetch.parse_statement(read("statement_old.html"), OLD_URL)
    assert r["date"] == "2005-02-02"
    assert r["text"].startswith("The Committee decided today to keep its target")
    assert "measured." in r["text"]


@pytest.mark.parametrize("name,url", [("statement_new.html", NEW_URL), ("statement_old.html", OLD_URL)])
def test_parse_keeps_vote_paragraph(name, url):
    text = fetch.parse_statement(read(name), url)["text"]
    assert "Voting for the FOMC monetary policy action were" in text
    assert "Voting against the action was Dan Placeholder" in text


@pytest.mark.parametrize("name,url", [("statement_new.html", NEW_URL), ("statement_old.html", OLD_URL)])
def test_parse_drops_boilerplate(name, url):
    text = fetch.parse_statement(read(name), url)["text"]
    for junk in ["Release Date", "For immediate release", "Home |", "News and events",
                 "News & Events", "Last update", "Last Update", "Contact Us",
                 "Implementation Note", "For release at", "Press Releases"]:
        assert junk not in text


def test_parse_drops_link_only_paragraphs():
    html = """<div id="article"><p>Real statement text.</p><p><a href="/faq.htm">Current FAQs</a></p>
    <p>Text with <a href="/x.htm">a link</a> inside.</p>
    <p><a href="/a.htm">Bank of Canada</a> <a href="/b.htm">Bank of England</a></p>
    <p><a href="/j.pdf">Bank of Japan</a> (79 KB PDF)</p>
    <p>1. A footnote. Return to text</p></div>"""
    text = fetch.parse_statement(html, NEW_URL)["text"]
    assert text == "Real statement text.\n\nText with a link inside."


def test_parse_keeps_loose_vote_text_after_empty_p():
    html = """<div id="article"><div><P>Policy text.</P><P></P>Voting for the FOMC monetary policy action
    were: A Person.<!-- a comment --></div></div>"""
    text = fetch.parse_statement(html, NEW_URL)["text"]
    assert text == "Policy text.\n\nVoting for the FOMC monetary policy action were: A Person."


def test_parse_unclosed_p_inside_article_not_duplicated():
    html = """<div id="article"><div><p>Joint statement.<p>Federal Reserve actions.
    <p><a href="/j.pdf">Bank of Japan (57 KB PDF)</a></div></div>"""
    text = fetch.parse_statement(html, NEW_URL)["text"]
    assert text == "Joint statement.\n\nFederal Reserve actions."


def test_parse_old_unclosed_p_no_duplicate_or_footer():
    html = """<body><p>First para.<p>Second para.<p>Voting for the FOMC monetary policy action were: A.
    <p><font>2003 Monetary policy <a href="/">Home</a> | News and events Last update: May 1, 2003</font></body>"""
    text = fetch.parse_statement(html, OLD_URL)["text"]
    assert text == "First para.\n\nSecond para.\n\nVoting for the FOMC monetary policy action were: A."


def test_get_uses_cache_without_network(offline):
    p = offline / "x.htm"
    p.write_text("cached")
    assert fetch.get(NEW_URL, p) == "cached"


def test_get_refuses_other_hosts(offline):
    with pytest.raises(ValueError):
        fetch.get("https://example.com/a.htm", offline / "a.htm")
    assert not (offline / "a.htm").exists()


def test_get_refuses_redirect_to_other_host(offline, monkeypatch):
    redirected = SimpleNamespace(url="https://example.com/x.htm", text="evil", raise_for_status=lambda: None)
    monkeypatch.setattr(fetch.httpx, "get", lambda *a, **k: redirected)
    with pytest.raises(ValueError):
        fetch.get(NEW_URL, offline / "x.htm")
    assert not (offline / "x.htm").exists()


def test_read_fred_handles_dots():
    csv = "observation_date,DGS2\n2020-01-01,1.5\n2020-01-02,.\n2020-01-03,\n"
    assert fetch.read_fred(csv) == {"2020-01-01": 1.5, "2020-01-02": None, "2020-01-03": None}
    assert fetch.read_fred("DATE,DGS2\n2020-01-01,1.5\n") == {"2020-01-01": 1.5}


def test_splice_target_switches_after_2008_12_15():
    old = {"2008-12-14": 1.0, "2008-12-15": 1.0, "2008-12-16": 9.9}
    new = {"2008-12-15": 9.9, "2008-12-16": 0.25, "2008-12-17": 0.25}
    out = fetch.splice_target(old, new)
    assert out["2008-12-15"] == 1.0
    assert out["2008-12-14"] == 1.0
    assert out["2008-12-16"] == 0.25
    assert out["2008-12-17"] == 0.25


def labels_by_date(rows):
    return {r["date"]: r for r in rows}


def test_labels_hike_hold_cut():
    target = {"2020-01-01": 1.0, "2020-01-10": 1.25,   # hike around 01-05
              "2020-02-01": 1.25, "2020-02-10": 1.25,  # hold around 02-05
              "2020-03-01": 1.25, "2020-03-10": 1.0}   # cut around 03-05
    rows = fetch.build_labels(["2020-03-05", "2020-01-05", "2020-02-05"], target, {})
    assert [r["date"] for r in rows] == ["2020-01-05", "2020-02-05", "2020-03-05"]
    assert [r["decision"] for r in rows] == ["hike", "hold", "cut"]
    assert [r["next_decision"] for r in rows] == ["hold", "cut", ""]
    assert set(rows[0]) == {"date", "decision", "next_decision", "dgs2_change_bp"}


def test_labels_next_day_effective_move_counts():
    target = {"2020-01-04": 1.0, "2020-01-06": 1.25}  # nothing on the statement date itself
    rows = fetch.build_labels(["2020-01-05"], target, {})
    assert rows[0]["decision"] == "hike"


def test_labels_last_next_decision_blank():
    target = {"2020-01-01": 1.0, "2020-01-09": 1.0}
    rows = fetch.build_labels(["2020-01-05"], target, {})
    assert rows[0]["next_decision"] == ""


def test_labels_dgs2_change_skips_missing():
    target = {"2020-01-01": 1.0, "2020-01-09": 1.0}
    dgs2 = {"2020-01-02": 1.50, "2020-01-03": None, "2020-01-05": 1.62}
    rows = fetch.build_labels(["2020-01-05"], target, dgs2)
    assert rows[0]["dgs2_change_bp"] == 12.0
    rows = fetch.build_labels(["2020-01-05"], target, {"2020-01-02": 1.5, "2020-01-05": None})
    assert rows[0]["dgs2_change_bp"] == ""
