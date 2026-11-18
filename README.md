# recsys-lab

**Status: `in-progress`.** Offline evaluation lab for news recommendation on
[Microsoft MIND-small](https://msnews.github.io/). Four rankers (random, popularity,
category affinity, TF-IDF) are evaluated on the provided MIND-small candidate sets.
Results are in [reports/baseline_results.md](reports/baseline_results.md) and
`results/`, including a figure of nDCG@10 by History length and labeled sports slices.
A leave-one-day-out hybrid has development results on train days (no holdout run yet).

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

## Run the benchmark

Every model is scored on the same impressions, candidates and seed:

```bash
# once, before the first holdout run; commit eval/manifest.json afterwards
uv run python scripts/evaluate.py --config configs/baseline.yaml --lock-manifest
# tune + holdout; writes results/experiments.json, results/summary.json, reports/baseline_results.md
uv run python scripts/evaluate.py --config configs/baseline.yaml
# development only: tune split, no manifest needed
uv run python scripts/evaluate.py --config configs/baseline.yaml --splits tune
# rebuild the figure and report from results/experiments.json (no data read, nothing re-scored)
uv run python scripts/evaluate.py --config configs/baseline.yaml --render-only
```

Use `--render-only` after editing `reports/interpretation.md`. It leaves the JSON results,
including the holdout numbers, untouched.

A holdout run stops if `eval/manifest.json` doesn't match the data files. To try the
pipeline without MIND, run it on the synthetic fixtures. The outputs are labeled
`synthetic` and go to the gitignored `results/synthetic/`:

```bash
uv run python scripts/evaluate.py --config configs/synthetic.yaml --lock-manifest
uv run python scripts/evaluate.py --config configs/synthetic.yaml
```

**Rankers** (`src/recsys/rankers.py`). Each one scores a label-free request:
- `random`: seeded random order (sanity floor).
- `popularity`: as-of item clicks, with ties broken by category clicks. Non-personalized.
- `category_affinity`: share of the user's History in the candidate's category and subcategory.
- `tfidf`: cosine between title+abstract TF-IDF vectors of the candidate and the user's History.

The two personalized rankers break ties by the popularity order. Users with no usable
History (cold start) fall back to popularity, and the number of fallbacks is reported.

**What's reported:**
- nDCG@10 (headline), nDCG@5, MIND-style MRR and AUC
- eligible-impression counts
- user-grouped bootstrap intervals and paired differences vs popularity
- breakdowns by History length (0, 1-5, 6-20, 21-50, 51+)
- coverage@10, scoring latency, and tie and fallback counts
- labeled sports slices (impressions with a sports candidate, users with sports History)
- deltas against both `popularity` and `category_affinity`
- the figure `results/figures/ndcg_by_history.png`

The report includes a hand-written interpretation, `reports/interpretation.md`, verbatim.
By rule it holds no numbers, and a test checks that, so every figure in the report
comes from the generated JSON.

### Tune-only diagnostics

```bash
uv run python scripts/diagnose.py --config configs/diagnostics.yaml
```

This reads `MINDsmall_train` only; the holdout is never loaded. It scores each train
day with rankers fitted on the earlier days (a rolling-origin stability check) and
tests ablation variants on the tune protocol. It writes
`results/tune_diagnostics.json` and
[reports/tune_diagnostics.md](reports/tune_diagnostics.md). The decision rules are
recorded in DECISIONS.md D-020, and the findings in D-021.

### Baseline diagnostics and the tune/holdout reversal

```bash
uv run python scripts/baseline_diagnostics.py --config configs/baseline_diagnostics.yaml
```

This also reads `MINDsmall_train` only. Holdout numbers come from the committed
`results/experiments.json`, used as a descriptive reference, and are never re-scored.
It reports:
- train/tune data coverage and score-tie profiles
- History-length and History-concentration segments
- returning vs new users
- a fixed-target corpus-size experiment (trailing windows and random subsamples)
- how much of the reversal each explanation can account for

Outputs are `results/baseline_diagnostics.json` and
[reports/baseline_diagnostics.md](reports/baseline_diagnostics.md). Rules are in D-023,
findings in D-024, and the follow-up proposal in
[reports/reversal_memo.md](reports/reversal_memo.md).

### E2: leave-one-day-out hybrid (development only)

```bash
uv run python scripts/hybrid_development.py --config configs/hybrid.yaml
```

This fits a logistic-regression hybrid over out-of-time base-ranker scores on four train
days (11-11 to 11-14). For each day, the base rankers see only earlier data, and the
weights are fitted on the other days. No holdout data is loaded or scored. Outputs are
`results/hybrid_development.json`, which includes the frozen weights, and
[reports/hybrid_development.md](reports/hybrid_development.md). The protocol is in
DECISIONS.md D-022 and D-025, the findings in D-026. These are development results, not
holdout results.

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
supply a local copy. It isn't committed. The script is in Microsoft's
[msnews/MIND](https://github.com/msnews/MIND) repository, and the MIND competition page
on CodaLab offers it too. Keep it outside the repo, or under the gitignored
`data/archives/`. Parity passed on 2026-11-03 (D-019):

```bash
MIND_EVAL_PY=data/archives/mind_evaluate.py uv run pytest -k parity
```

**Leakage rules.**
- Popularity is as-of: it counts only clicks strictly before the impression.
- Tune is scored with models fitted on `fit`. Holdout is scored with models refitted on `fit` + `tune`. Dev clicks are never features.
- Score ties are broken by a seeded hash of the candidate ID, never by the order candidates appear in the impression.

Conventions and their consequences are recorded in [DECISIONS.md](DECISIONS.md).

## Layout

```
src/recsys/   schemas, loaders, inventory, splits, metrics, rankers, evaluate, diagnostics, hybrid
scripts/      inspect_data.py, evaluate.py, diagnose.py, baseline_diagnostics.py, hybrid_development.py
configs/      baseline.yaml (real data), synthetic.yaml (fixtures), diagnostics configs
eval/         manifest.json, locked before the first holdout run
results/      experiments.json, summary.json, figures/ (generated)
reports/      generated reports; interpretation.md and reversal_memo.md are hand-written
tests/        unit tests + synthetic fixtures
data/         README only; data/raw/ is gitignored
```

## Limitations

- MIND is general news from 2019. Results don't show sports-specific behavior.
- MIND-small numbers aren't comparable to the public MIND leaderboard, which is scored on MIND-large test.
- Offline ranking metrics don't predict online engagement.
- Labels reflect what Microsoft News's own 2019 system chose to show (exposure bias).
- Holdout popularity uses train clicks only, so it's up to a day stale, and items first shown in dev have no click history.
- Rankers are simple, fixed-parameter baselines. Nothing was tuned.
