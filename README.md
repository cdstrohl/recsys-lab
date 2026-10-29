# recsys-lab

**Status: `in-progress`.** Offline evaluation lab for news recommendation on
[Microsoft MIND-small](https://msnews.github.io/). There are no results yet. The
parser, data inventory, split policy and ranking metrics are in place and tested on
synthetic fixtures. No models have been run.

The question: which ranking approach is worth shipping for a news feed, across
cold-start and history-rich users, once quality, complexity and inference cost are
all weighed? Results will be offline ranking metrics only. They aren't engagement,
CTR or business-impact claims.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
```

## Tests and checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

All tests run on small, hand-authored **synthetic** fixtures in `tests/fixtures/`
(labeled in each folder's `PROVENANCE.json`). They aren't evidence of real user
behavior.

## Data

See [data/README.md](data/README.md) for manual download steps and the license note.
No MIND data is committed. Once the files are in place:

```bash
uv run python scripts/inspect_data.py --data data/raw
# or on fixtures:
uv run python scripts/inspect_data.py --data tests/fixtures/mind_synthetic
```

## Evaluation design

**Splits**

| Split | Source | Use |
|---|---|---|
| `fit` | MIND-small train, before midnight of the last calendar day | Fit models, compute statistics |
| `tune` | MIND-small train, last calendar day | Model selection and debugging |
| `holdout` | MIND-small dev | Locked; headline numbers only |

**Metrics.** Computed per impression, then averaged:
- nDCG@10 (headline)
- nDCG@5
- MIND-style MRR: the mean of 1/rank over *all* clicked items, not just the first
- AUC: tied pairs count 0.5

**Which impressions count.**
- No-click impressions are excluded from every metric.
- All-click impressions are excluded from AUC only.
- Every exclusion is counted and reported.

**Short lists.** With fewer than k candidates, nDCG@k's ideal DCG uses min(k, n).

The optional parity test against MIND's official `evaluate.py` runs only if you
supply a local copy (it isn't committed):

```bash
MIND_EVAL_PY=/path/to/evaluate.py uv run --with scikit-learn pytest -k parity
```

**Leakage rules.**
- Popularity is as-of: it counts only clicks strictly before the impression.
- Statistics come from `fit` only.
- Score ties are broken by a seeded hash of the candidate ID, never by the order candidates appear in the impression.

Conventions and their consequences are recorded in [DECISIONS.md](DECISIONS.md).

## Layout

```
src/recsys/   schemas, loaders, inventory, splits, metrics
scripts/      inspect_data.py
tests/        unit tests + synthetic fixtures
data/         README only; data/raw/ is gitignored
```

## Limitations

- MIND is general news from 2019. Results don't show sports-specific behavior.
- MIND-small numbers aren't comparable to the public MIND leaderboard, which is scored on MIND-large test.
- Offline ranking metrics don't predict online engagement.
