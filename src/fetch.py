"""Download FOMC statements and FRED series, parse them, and write statements.csv and labels.csv.

Run: uv run python -m src.fetch
"""

import csv
import re
import time
from bisect import bisect_left, bisect_right
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
HEADERS = {"User-Agent": "FedSpeakDecoder/0.1 (+https://github.com/SidVelagapalli/fed-decoder)"}
TIMEOUT = 20.0
DELAY = 1.0
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FRED_DIR = DATA_DIR / "fred"

ALLOWED_PREFIXES = (BASE + "/", "https://fred.stlouisfed.org/")
STATEMENT_PATH = re.compile(
    r"^/boarddocs/press/(monetary|general)/\d{4}/\d{8}/(default\.htm)?$"
    r"|^/newsevents/pressreleases/monetary\d{8}a\.htm$"
)

# Paragraphs starting with any of these (case-insensitive) are page chrome, not statement text.
# Tune this list against real pages.
SKIP_PREFIXES = (
    "release date",
    "for immediate release",
    "for release at",
    "home | news",
    "home |",
    "last update",
    "implementation note",
    "for media inquiries",
    "accessibility",
    "contact us",
    "return to top",
    "skip to",
)
# A paragraph that is only a date, e.g. "January 28, 2026" (the release-date heading).
DATE_ONLY = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


def get(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text()
    # Trailing "/" in the allowed prefixes stops look-alike hosts like www.federalreserve.gov.evil.com.
    if not url.startswith(ALLOWED_PREFIXES):
        raise ValueError(f"refusing to fetch from non-allowed host: {url}")
    response = httpx.get(url, headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text)
    time.sleep(DELAY)  # be polite to the servers
    return response.text


def raw_path(url: str) -> Path:
    return RAW_DIR / urlparse(url).path.replace("/", "_")


def statement_links(index_html: str) -> list[str]:
    links = []
    for a in BeautifulSoup(index_html, "html.parser").find_all("a", href=True):
        if a.get_text().strip() != "Statement":
            continue
        href = a["href"].strip()
        path = href.removeprefix(BASE)
        if not STATEMENT_PATH.match(path):
            continue
        url = BASE + path
        if url not in links:
            links.append(url)
    return links


def index_urls(this_year: int) -> list[str]:
    # Years up to 2020 have their own historical page; 2021+ meetings are all on the calendars page.
    years = range(2000, min(this_year, 2020) + 1)
    urls = [f"{BASE}/monetarypolicy/fomchistorical{y}.htm" for y in years]
    return urls + [f"{BASE}/monetarypolicy/fomccalendars.htm"]


def _keep(paragraph: str) -> bool:
    if not paragraph or DATE_ONLY.match(paragraph):
        return False
    return not paragraph.lower().startswith(SKIP_PREFIXES)


def parse_statement(html: str, url: str) -> dict:
    digits = re.search(r"\d{8}", url).group()
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("div#article")
    if article is not None:
        raw = [p.get_text() for p in article.find_all("p")]
    else:
        # Old boarddocs pages: table layout, sometimes with <p> tags, sometimes plain text with <br>.
        raw = [p.get_text() for p in soup.find_all("p")]
        if not raw:
            body = soup.body or soup
            raw = body.get_text("\n").split("\n")
    paragraphs = [" ".join(p.split()) for p in raw]
    text = "\n\n".join(p for p in paragraphs if _keep(p))
    return {"date": f"{digits[:4]}-{digits[4:6]}-{digits[6:]}", "url": url, "text": text}


def read_fred(csv_text: str) -> dict[str, float | None]:
    rows = list(csv.reader(csv_text.splitlines()))[1:]  # skip header, whatever it's called
    values = {}
    for row in rows:
        if len(row) < 2:
            continue
        value = row[1].strip()
        values[row[0].strip()] = None if value in ("", ".") else float(value)
    return values


def splice_target(dfedtar: dict, dfedtaru: dict) -> dict:
    # DFEDTAR (single target) ends 2008-12-15; after that the Fed sets a range, so use its upper bound.
    spliced = {d: v for d, v in dfedtar.items() if d <= "2008-12-15"}
    spliced.update({d: v for d, v in dfedtaru.items() if d > "2008-12-15"})
    return spliced


def _known(series: dict) -> tuple[list[str], list[float]]:
    items = sorted((d, v) for d, v in series.items() if v is not None)
    return [d for d, _ in items], [v for _, v in items]


def build_labels(dates: list[str], target: dict, dgs2: dict) -> list[dict]:
    t_dates, t_vals = _known(target)
    y_dates, y_vals = _known(dgs2)
    rows = []
    for d in sorted(dates):
        i = bisect_left(t_dates, d)  # t_dates[i - 1] is the last date strictly before d
        j = bisect_right(t_dates, d)  # t_dates[j] is the first date strictly after d
        before = t_vals[i - 1] if i > 0 else None
        after = t_vals[j] if j < len(t_dates) else None
        if before is None or after is None:
            decision = ""
        elif after > before:
            decision = "hike"
        elif after < before:
            decision = "cut"
        else:
            decision = "hold"

        change = ""
        if dgs2.get(d) is not None:
            k = bisect_left(y_dates, d)
            if k > 0:
                change = round((dgs2[d] - y_vals[k - 1]) * 100, 1)

        rows.append({"date": d, "decision": decision, "next_decision": "", "dgs2_change_bp": change})
    for row, nxt in zip(rows, rows[1:]):
        row["next_decision"] = nxt["decision"]
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    by_date = {}
    for index_url in index_urls(date.today().year):
        for link in statement_links(get(index_url, raw_path(index_url))):
            by_date.setdefault(re.search(r"\d{8}", link).group(), link)  # keep first link per date

    statements = [parse_statement(get(url, raw_path(url)), url) for url in by_date.values()]
    statements.sort(key=lambda s: s["date"])
    _write_csv(DATA_DIR / "statements.csv", ["date", "url", "text"], statements)

    series = {}
    for name in ("DFEDTAR", "DFEDTARU", "DGS2"):
        series[name] = read_fred(get(FRED_URL.format(series=name), FRED_DIR / f"{name}.csv"))
    target = splice_target(series["DFEDTAR"], series["DFEDTARU"])
    labels = build_labels([s["date"] for s in statements], target, series["DGS2"])
    _write_csv(DATA_DIR / "labels.csv", ["date", "decision", "next_decision", "dgs2_change_bp"], labels)

    print(f"{len(statements)} statements, {len(labels)} label rows")
    print("FRED points: " + ", ".join(f"{k}={len(v)}" for k, v in series.items()))


if __name__ == "__main__":
    main()
