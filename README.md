# Fed Speak Decoder

Reads every US Federal Reserve FOMC statement since 2000, highlights what changed since the last one, and turns the
wording into a hawk/dove score in [-1, 1]. An LLM only gathers evidence (quoted sentences); fixed, tested rules set the
score; the LLM then explains it. The score is checked against what the Fed did next and how the 2-year Treasury
yield moved that day.

## How it works
```
FOMC statements ─> diff vs previous ─> LLM: stance signals + verbatim quotes (checked in code)
                                              │
                                              v
                          score.py: fixed rules -> score + rules_fired
                                              │
                                              v
                          LLM: plain-English explanation (cites rules only)
```

## Results
_Not run yet (Rung 2)._

| Method | Next decision accuracy | Corr. with 2y yield move |
|---|---|---|
| Persistence (next move = last move) | - | - |
| Hawk/dove word list (no LLM) | - | - |
| Fed Speak Decoder | - | - |

## How to run
```bash
uv sync
uv run pytest                 # offline tests
uv run python -m src.fetch    # download statements + FRED data into data/ (free, ~5 min)
```

## Data sources
- FOMC statements: federalreserve.gov (public domain)
- FRED: DFEDTAR, DFEDTARU (fed funds target), DGS2 (2-year Treasury yield)

## Limitations
_To fill in after the eval._

## Acknowledgments
Built with AI-assisted development tools; the design, decisions, evaluation and review are my own.
