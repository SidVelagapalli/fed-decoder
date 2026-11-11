# Fed Speak Decoder

Reads every US Federal Reserve FOMC statement since 2000, highlights what changed since the last one, and turns the
wording into a hawk/dove score in [-1, 1]. An LLM only gathers evidence (verbatim quotes, checked in code); fixed,
tested rules set the score; the LLM then explains it. The score is tested against what the Fed did next and how the
2-year Treasury yield moved that day, next to two simple baselines.

**Short version of the result:** the decoder clearly beats a no-LLM word list, but it does not beat the simplest
baseline of all, "the Fed's next move is the same as its last one." The reasons are below.

## Results
223 statements (2000-02-02 to 2026-10-28), 446 LLM calls (`claude-haiku-4-5`), 773k input / 105k output tokens,
**$1.30** total. All 223 statements scored. 222 have a known next decision (148 hold, 40 hike, 34 cut).

| Method | Next-decision accuracy | Macro-F1 | Spearman vs same-day 2y yield move | Calls hike/hold/cut |
|---|---|---|---|---|
| Always hold | 0.667 | 0.267 | n/a | 0/222/0 |
| Persistence (next move = last move) | **0.797** | **0.750** | 0.077 | 41/147/34 |
| Hawk/dove word list (no LLM) | 0.288 | 0.300 | 0.021 | 78/35/109 |
| Fed Speak Decoder | 0.437 | 0.443 | 0.054 | 69/68/85 |
| Fed Speak Decoder, quote-check fix (see below) | 0.437 | 0.444 | 0.047 | 70/66/86 |

Made by `uv run python -m src.eval` (last row: `--rescore`).

- Macro-F1 averages F1 over hike, hold and cut. Two-thirds of meetings are holds, so "always hold" gets 0.667 accuracy
  but only 0.267 macro-F1; macro-F1 is the fairer number.
- The threshold (score above +0.2 = hike, below -0.2 = cut, else hold) and the word list were fixed **before** any
  results were seen and were not tuned. Score weights in `src/score.py` are a first draft, also untuned.
- The result is stable over time: decoder accuracy 0.473 on 2000-2012 and 0.402 on 2013-2026; persistence 0.882
  and 0.714.
- **Quote-check fix (made after seeing results).** 78 extracted quotes (in 61 statements) were rejected as "not in
  the statement." About half were not real misquotes: 31 differed only in capitalization, 7 in curly vs straight
  punctuation, and 5 were empty. 35 were genuine paraphrases. The check now ignores case and typographic punctuation
  (the words must still match). Re-scoring the logged answers offline ($0) changed 20 of 223 scores and left the
  results essentially unchanged.

### Why persistence wins
The score measures **how hawkish the Fed sounds**. The label asks **whether it will change rates next time**. In a
long hiking cycle (2005, 2022-23) the Fed sounds hawkish at every meeting, but the decoder can't tell which meeting
will be the last hike before a pause. Persistence gets those for free, because rate cycles are long and the Fed
moves in steps.

Same-day yield moves barely line up with any method (all correlations near 0). Markets move on the *surprise*
relative to what was expected. A stance that everyone already anticipated doesn't move yields, and this project has
no data on market expectations.

## How it works
```
federalreserve.gov ─> statements ─> diff vs previous statement
                                          │
                                          v
              LLM: six stance signals, each with a verbatim quote as evidence
              (statement text is wrapped and labelled as untrusted data in the prompt)
                                          │
                                          v
              code: every quote must appear in the statement, or the signal is dropped
                                          │
                                          v
              score.py: fixed weighted rules -> score in [-1, 1] + rules_fired
                                          │
                                          v
              LLM: plain-English explanation that may only cite rules_fired
                                          │
                                          v
              eval.py: vs the Fed's next decision (FRED target rate) and same-day 2y yield (FRED DGS2)
```
Signals: inflation concern, growth view, labor view, forward-guidance direction, balance sheet, dissents. The LLM never
decides the score, so a prompt-injected statement can at most change the evidence, and evidence must be a real quote.

Each statement's result is appended to `logs/` as soon as it finishes. Every record has a status (`ok`,
`extract_failed`, `explain_failed`, `error`), and re-running the same command retries only dates that aren't `ok`.

## How to run
```bash
uv sync
uv run pytest                 # 80 offline tests, no network or API key
uv run python -m src.fetch    # statements + FRED data into data/ (free, ~5 min)
uv run python -m src.run --dates 2015-12-16 2020-03-15                          # dry run: plan + cost estimate
uv run python -m src.run --dates 2015-12-16 2020-03-15 --provider anthropic --go  # real calls (needs ANTHROPIC_API_KEY)
uv run python -m src.eval     # results table from logs/
```
`--provider groq` (the default) uses a free Groq model instead; its free daily token limit makes a full run take
several days.

## Limitations
- Measures stance level, not the change the market didn't expect. Next steps that would test the idea properly:
  use the change in score since the last statement, and compare against market-implied expectations
  (e.g. fed funds futures), decided before looking at results.
- One model, one run. LLM answers vary between runs, and that variance wasn't measured.
- The quote check proves a quote exists, not that it supports the level the model chose. It accepts very short quotes.
- The word list is small and hand-made; a published dictionary could be a stronger baseline.
- Four non-policy announcements are excluded. The Sunday 2020-03-15 emergency cut has no same-day yield move.

## Data sources
- FOMC statements: federalreserve.gov (public domain)
- FRED: DFEDTAR, DFEDTARU (fed funds target), DGS2 (2-year Treasury yield)

## Acknowledgments
Built with AI-assisted development tools; the design, decisions, evaluation and review are my own.
