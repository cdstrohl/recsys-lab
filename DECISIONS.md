# Decisions

Non-obvious choices, newest last. Any change to a metric, candidate set or
eligibility rule needs a new entry here and a re-run of every model.

## D-001 Tooling (REC-01)
Python 3.11, `uv` with committed `uv.lock`, pytest, ruff, mypy (strict on `src`).
numpy is the only runtime dependency. pandas waits until a ticket needs it (likely
REC-04).

## D-002 Data acquisition is manual (REC-01)
The repo ships written acquisition steps (`data/README.md`) and **no download
code**, by owner instruction. The shared data standard says repos "ship download
scripts, not data". We read its intent as "ship the means to get the data, never
the data itself", and manual steps meet that more strictly. The owner downloads
MIND-small after reviewing the Microsoft Research License Terms. Raw files,
archives and article text are never committed.

## D-003 Parser policy (REC-01)
- Original MIND TSVs only, no header row. `news.tsv` must have 8 columns and `behaviors.tsv` 5.
- **Strict by default:** the first malformed row raises `ParseError` with file:line and a reason. `strict=False` skips malformed rows and counts them by reason. The inventory uses lenient mode, so one run shows every problem in the real files.
- Blank lines are skipped and counted, never treated as errors. A `\r\n` line ending is accepted. Only the line terminator is stripped, so trailing empty columns (empty abstract or entities) are kept.
- **Duplicate IDs:** the first row is kept and later rows are reported as `duplicate_id`.
- **Duplicate news ID within one impression:** rejected as `duplicate_candidate`. Tie-breaking (D-008) keys on news ID, so duplicates would make it ambiguous.
- **Timestamps:** `M/D/YYYY h:MM:SS AM|PM` is parsed without depending on locale, into a naive `datetime`. MIND states no time zone and none is assumed.
- **Fields:** history and candidates split on whitespace, and an empty history becomes `()`. Entity columns stay raw JSON strings until a ticket needs them.
- Error details name IDs and timestamps only, never article text.

## D-004 Repo contents (REC-01)
The shared agent rules (`AGENTS.md`, `CLAUDE.md`) and planning docs (`docs/`) stay
local and gitignored, because they reference other projects and private context.
The public repo is self-describing through `README.md`, `data/README.md` and this
file.

## D-005 Split boundary (REC-02)
- **`fit` / `tune`:** the cut is midnight at the start of the **calendar date** of the latest `MINDsmall_train` impression, not the last 24 hours before it. Owner confirmed the calendar day.
- **Half-open intervals:** `fit` is `t < cutoff` and `tune` is `t >= cutoff`, so an impression at exactly 00:00:00 on the last day is `tune`.
- **Holdout:** all of `MINDsmall_dev`.
- **Time zone:** timestamps are naive and nothing is converted.
- **Labels never affect the split:** the cut depends on timestamps only.
- **Fallback:** if the inventory (`splits` section) shows the last day is a poor cut, `assign_splits(..., cutoff=...)` takes another time-based cut. That choice gets its own entry here before any model comparison.
- **Inventory reporting:** the inventory reports a cut that can't be made instead of raising.

## D-006 Holdout guard (REC-02)
`SplitAssignment.holdout()` raises `HoldoutLockedError` unless the caller passes
`allow_holdout=True`. Only the headline evaluation path should pass it.
`statistics_source()` returns `fit` (or `fit` + `tune` for the documented final
refit) and never includes holdout. The guard prevents accidents. It isn't access
control.

## D-007 As-of popularity and statistics source (REC-02)
- A click event is a clicked candidate, stamped with its impression's timestamp.
- `History` clicks carry no timestamp, so they're **not** click events and never count toward as-of popularity.
- `count_before(item, t)` counts events with `t_event < t`, strictly before. An impression therefore never sees its own labels, or those of another impression with the same timestamp.
- The optional `since` window is inclusive at its lower bound.
- The index is built only from `statistics_source()`. When scoring `tune` or `holdout`, popularity therefore reflects `fit` clicks only, not earlier clicks in the scored split.
- The inventory reports `n_impression_ids_shared` between train and dev. If MIND restarts IDs per file, any key that spans splits must include the split name.

## D-008 Ties (REC-03)
- **MRR and nDCG:** these need a strict order. Candidates are sorted by score descending, then by a 64-bit key `blake2b(seed, impression_id, news_id)` ascending.
- **Properties of the key:** it's deterministic for a given seed and independent of the order candidates appear in the row. Row order may carry information, so it's never used.
- **What a tied MRR or nDCG means:** each is one seeded draw of the tie order, not the expectation over all tie orders. A model that ties often can move with the seed, which is why `n_tied` is reported.
- **AUC:** a tied (clicked, non-clicked) pair counts 0.5, as in Mann-Whitney and sklearn. That equals the expected AUC under a uniformly random tie order, so AUC needs no key.
- **Parity:** the official MIND script breaks ties by array position, so parity tests use tie-free cases only.
- **Finite scores:** scores must be finite. NaN or inf raises an error, so a broken model can't silently produce a ranking.

## D-009 MRR and nDCG definitions (REC-03)
- **MRR is MIND-style:** the mean of `1/rank` over **all** clicked candidates in the impression, then averaged across impressions. With `c` clicks, even a perfect ranking scores `H_c / c` (for example 0.75 for two clicks), not 1. This isn't "first relevant item" MRR, and reports label it as MIND-style.
- **nDCG@k:** gain `2^rel - 1` (1 for a click) and discount `log2(rank + 1)`. With `n < k` candidates, DCG and ideal DCG both run over min(k, n) positions, so a short list isn't penalized for positions it doesn't have.

## D-010 Eligibility: no-click and all-click impressions (REC-03)
- **No-click impressions** (no clicked candidate) are excluded from **every** metric and counted (`n_no_click`). Every metric is undefined on them.
- **All-click impressions** (every candidate clicked, including single-candidate impressions with a click) are excluded from **AUC only** and counted (`n_all_click`), by owner decision. MRR and nDCG still include them. Each metric reports its own `n_eligible`.

Consequences, for an all-click impression with `n` candidates:
- **Scores don't matter on these impressions.** nDCG@k = 1 for every ranking, because every position has gain 1, so DCG = ideal DCG. MRR = `H_n / n`, where `H_n = 1 + 1/2 + ... + 1/n` (1, 0.75 and 0.611 for n = 1, 2, 3).
- **Model ordering is safe.** Every model gets the same constant from them, so with `E` impressions that have a click and a non-click and `A` all-click impressions, the order of models by mean MRR or nDCG doesn't change.
- **Absolute gaps shrink.** A per-impression gap Δ on the `E` impressions becomes `E·Δ / (E + A)`.
- **Relative gaps shrink further, because the mean moves toward the constant.** The nDCG lift of model M over baseline B goes from `Δ / m_B` to `E·Δ / (E·m_B + A)`. This matters for the "≥5% relative nDCG@10" research question, so reports state `n_all_click`.
- **Different denominators.** AUC averages over `E` impressions; MRR and nDCG average over `E + A`. Their per-impression means come from different sets, which is why each metric reports its own `n_eligible`.
- **Probably rare.** MIND impressions usually have many non-clicked candidates, so `A` should be small. The inventory reports `n_all_click_impressions` so this can be checked on the real data.

## D-011 Real-data inventory findings (2026-10-28)
`scripts/inspect_data.py --data data/raw` on MIND-small. The four TSV hashes were
recorded locally and checked OK. Aggregate counts only.

| | Train | Dev (holdout) |
|---|---|---|
| Impressions | 156,965 | 73,152 |
| Users | 50,000 | 50,000 |
| News items | 51,282 | 42,416 |
| Date range | 2019-11-09 to 11-14 | 2019-11-15 only |
| Candidates per impression (median / max) | 24 / 299 | 23 / 295 |
| Malformed rows | 0 | 0 |

Confirmed:
- **The default cut stands; no fallback is needed (D-005).** The cutoff is 2019-11-14 00:00. `fit` is 11-09 to 11-13 (126,695 impressions) and `tune` is 11-14 (30,270, a full day). The weekend days 11-09 and 11-10 have about half the volume of weekdays.
- **The eligibility rules are moot on this data (D-010).** No impression is no-click, all-click or single-candidate, so all four metrics average over the same impressions.
- **History is fixed per user.** No user's History differs across rows, and every candidate and history ID is in `news.tsv`.
- **Dev doesn't overlap train in time.**
- **Impression IDs restart per file.** All 73,152 dev IDs also occur in train, so any key spanning splits must include the split name (D-007).
- **Item cold start is severe for fit-only statistics.** Against `fit`, 80.7% of holdout candidate occurrences are unseen (70.96% of unique items). Against all of train, it's 21.3% (46.3%). Against `fit`, 29.2% of tune occurrences are unseen.
- **Few dev users appear in train impressions** (88.1% don't), but most users have a non-empty History. Only 1,407 dev users have none, against 892 in train. History items rarely appear as train candidates (6%).

## D-012 Ranker interface (REC-04 to REC-06)
- **Requests carry no labels.** Every ranker sees a `RankRequest`: impression ID, user, timestamp, History and candidate IDs. Click labels reach only the metrics. A test pins the request's fields, and other tests flip tune and holdout labels and check that every ranker's scores are unchanged.
- **Output.** `score(request)` returns `Ranking(scores, fallback)`. A higher score ranks higher, and remaining ties go to the seeded key (D-008). `fallback` is set when a personalized ranker had no usable History and used the popularity order instead.
- **Composite keys.** Rankers that order by several keys (for example, affinity then popularity) return dense ranks of the key tuple (`lexicographic_scores`). Two candidates tie only if every key is equal, and nothing depends on weights or float scaling.
- **Shared tie-breaker.** Both personalized rankers break their own ties by the popularity order and fall back to it for cold-start users. Any difference from `popularity` therefore comes from the personalized signal alone, and on zero-history users they equal `popularity` by construction.
- **No tuning.** Every parameter was fixed before any real-data run, and none was changed after seeing the tune results. A later tuning pass would select on `tune` only and get its own entry.

## D-013 Item metadata and the TF-IDF corpus (REC-05, REC-06)
- **Metadata from both files.** Category, subcategory, title and abstract come from the union of train and dev `news.tsv`, with train's row kept for a shared ID. This is item content known when an item is shown, not a label, and without it no dev-only item could be scored by content.
- **Fitting corpus.** The TF-IDF vocabulary and IDF are fitted only on items referenced (History or candidate) by the statistics source (D-014), never on the scored split. Words that first appear in dev items are ignored.
- **Missing items.** An ID missing from both `news.tsv` files gets 0 on every content and popularity key.

## D-014 Statistics source per split (REC-07)
| Scored split | Rankers fitted on |
|---|---|
| `tune` | `fit` |
| `holdout` | `fit` + `tune` (the spec's documented final refit; `holdout_statistics` in the config) |

- **Dev clicks are never features,** not even as-of within dev. The spec limits features to `fit` / `fit`+`tune`, and the owner asked that validation labels never be features. As a result, holdout popularity is up to a day stale, and items first shown in dev have no click count. That is 21.3% of holdout candidate occurrences against all of train, or 80.7% against `fit` alone (D-011).
- **History is used as given,** not augmented with in-window clicks, for every ranker. Augmentation is allowed by the spec if strictly time-ordered, and is left as a future ablation. Most dev users (88.1%) never appear in train impressions anyway.

## D-015 Ranker definitions (REC-04 to REC-06)
- **`random`.** Seeded `blake2b` scores, salted so they're independent of the metrics' tie keys.
- **`popularity`.** Keys are (item clicks, category clicks), both as-of and strictly before the impression (D-007), from the statistics source.
- **`category_affinity`.** Keys are (# known History items in the candidate's category + # in its (category, subcategory)), then the popularity keys. Dividing by History length is skipped because it doesn't change the order within an impression. History IDs missing from `news.tsv` are ignored, and if none remain, the ranker falls back.
- **`tfidf`.** The text is title + abstract. Vectorizer settings: English stop words, sublinear TF, `min_df=2`, L2-normalized rows. The user vector is the mean of the History item vectors, renormalized. Keys are (cosine, popularity keys). The ranker falls back if the user vector is zero: empty History, unknown items only, or no in-vocabulary terms.

## D-016 Reporting (REC-07)
- **Segments.** Each impression goes into a bucket by its History length, using edges `0, 1-5, 6-20, 21-50, 51+` from the config, finer than the spec's example. History is fixed per user (D-011), so the segments partition users as well as impressions.
- **Intervals.** User-grouped percentile bootstrap: 1,000 resamples, seed 13, 95%. A resample draws users with replacement and averages over all their impressions. Deltas against `popularity` use the same resamples, so they're paired. The intervals treat the logged candidate sets as fixed.
- **Costs.** `coverage@10` is the share of distinct candidate items placed in the top 10 of at least one impression. Latency is the wall time of `score()` per impression, as p50 and p95, in single-threaded Python on the owner's laptop. It excludes loading and fitting. Fit time is reported once per split for all models together, since they share the popularity index.
- **Tie counts.** `n_tied` counts impressions with any tied score, including ties that don't affect the metrics, so it's an upper bound on how much the seeded draw matters.

## D-017 Manifest and outputs (REC-07)
- **Manifest lock.** `eval/manifest.json` (v1: four TSV hashes, cutoff, split rule, row counts, label version) is locked with `--lock-manifest` and committed before the first holdout run. The runner refuses a holdout run if it's missing or doesn't match the data. Re-locking requires deleting the file by hand.
- **Order of runs.** Tune results were looked at once, as a bug check, before the lock. Nothing changed as a result.
- **Outputs.** All are written by `scripts/evaluate.py`:
  - `results/experiments.json`: every split, model, metric, interval and segment
  - `results/summary.json`: the `07` §9 export, holdout only, written only when holdout was evaluated
  - `reports/baseline_results.md`: tables and observations computed from the JSON
- **Privacy.** Outputs hold aggregate numbers only, and a test checks for user IDs and article text.
- **Deferred.** Segment figures, a sports slice and the interpretive memo (what to ship first) are left for a later ticket.

## D-018 Report additions (REC-07)
- **Figure.** `results/figures/ndcg_by_history.png` shows nDCG@10 by History bucket, one panel per split on a shared y-axis, with the mean and the user-bootstrap 95% interval. It's drawn with matplotlib's `Figure` API (no pyplot global state), saved with no software metadata, and listed in `summary.json` → `figures`. Colors are fixed per model (validated categorical slots 1-3), `random` is a dashed neutral reference, and every series is labeled directly as well as in the legend.
- **Sports slices.** Two slices: impressions with at least one `sports` candidate, and impressions whose user History has a `sports` item. They're labeled as slices of general-news data and overlap each other and the buckets. On real holdout, the candidate slice covers nearly every impression, so it adds little.
- **More than one comparison model.** `compare_to` (`popularity`, `category_affinity`) gives paired deltas against each, so later models are measured against the strongest simple baseline. `summary.json` keeps `popularity` as its `baseline`.
- **Fallback cells.** A model that fell back to popularity on every impression of a segment is shown as "fallback: = `popularity`" instead of a delta of zero.
- **Memo.** `reports/interpretation.md` is hand-written and included verbatim in the report. It holds **no numbers** (a test rejects decimals and percentages), so the numbers can't drift from the generated tables. It says which manifest it was written against.
- **Determinism check.** Re-running the unchanged models at commit `4e3b2f5` reproduced all 1,224 compared values (means, intervals, counts, deltas against popularity, tie, fallback and coverage counts) of the first holdout run (`58889cb`) exactly. Re-running holdout for reporting changes is therefore a reproducibility check, not a second look at the holdout.

## D-019 Parity with MIND's official scorer verified (REC-03)
On 2026-11-03 the owner downloaded `evaluate.py` from [msnews/MIND](https://github.com/msnews/MIND) (`master`) to the gitignored `data/archives/mind_evaluate.py`. Its SHA-256 is `c6195008ed19ba7e68145069cffbdbcc0582ff0e983e2a5966095274043cc2ca`. The four parity tests passed (`MIND_EVAL_PY=data/archives/mind_evaluate.py uv run pytest -k parity`). Our MRR, nDCG@5, nDCG@10 and AUC match the official functions on the tie-free parity cases. Tied cases are excluded by design (D-008), since the official script breaks ties by array position. The script isn't committed, and the parity tests still skip when `MIND_EVAL_PY` is unset.

## D-020 Phase 2 tune-only diagnostics: design and decision rules (fixed before any run)
**Why.** On tune, `tfidf` beat `category_affinity`; on holdout the order flipped (D-018). The holdout is locked, so the cause is investigated on train data only. `scripts/diagnose.py` loads `MINDsmall_train` and never reads dev.

**Stability (rolling origin).**
- Score each train day from 11-11 to 11-14 with rankers fitted on every train impression before that day's midnight.
- 11-14 under this rule is exactly the official tune protocol. The earlier days are `fit` impressions used as diagnostic targets only. No model or parameter is selected on them.

**Hypothesis H1 (tie-break mechanism).**
- `category_affinity` leaves more ties than `tfidf`, and both break ties by popularity. So `category_affinity` should gain more on days when popularity is informative.
- Supporting evidence would be:
  1. With random tie-breaks (`*_no_pop`), `tfidf` minus `category_affinity` is positive and steadier across days than with popularity tie-breaks.
  2. The gap with popularity tie-breaks shrinks on days where `popularity` minus `random` is larger.
  3. On tune, `category_affinity`'s deficit to `tfidf` is smaller in impressions where fewer candidates are unclicked in the statistics source.
- With four days, this is descriptive evidence, not a test.

**Ablations, on the tune protocol only.**
- Popularity counted over the last 24h or 48h.
- History augmented with the user's own earlier clicks in the statistics source, strictly time-ordered (`augment_history`, the extension D-014 deferred).
- Personalized rankers with a 24h popularity tie-break.

**Adoption rule.**
- A variant becomes a Phase 3 candidate only if both hold:
  - its paired nDCG@10 delta vs its base model on tune has a 95% CI entirely above 0
  - its CI for zero-History users isn't entirely below 0
- Candidates are added as new named models and evaluated in the single Phase 3 holdout run. The v1 baseline config and its holdout numbers stay unchanged.

**Bootstrap.** 500 user-grouped resamples, to keep run time manageable.

## D-021 Phase 2 findings (tune-only; holdout untouched)
Source: `results/tune_diagnostics.json` and `reports/tune_diagnostics.md`, generated at `4f77307`, which is also where D-020 was committed. The 11-14 row reproduces the committed tune numbers exactly, so the ranker refactor changed no behavior.

**The flip comes from day-to-day drift, not the tie-break. H1 is mostly not supported.**
- Paired nDCG@10 gap, `tfidf` minus `category_affinity`, by scored day:

  | | 11-11 | 11-12 | 11-13 | 11-14 |
  |---|---|---|---|---|
  | popularity tie-breaks | −0.006 | −0.013 | +0.005 | +0.014 |
  | random tie-breaks | −0.001 (CI includes 0) | −0.007 | +0.006 | +0.014 |

  With random tie-breaks the gap moves almost the same way, so prediction 1 fails.
- The tie-break helps `category_affinity` more on days when popularity is stronger: +0.005 and +0.007 on 11-11 and 11-12, against +0.001 on 11-13 and 11-14. That matches prediction 2's direction, but it accounts for a small part of the swing.
- On tune, `category_affinity`'s deficit isn't monotone in the unclicked share, so prediction 3 isn't supported.
- **Consequence:** which content ranker leads changes from day to day. Holdout (11-15) favored `category_affinity`, like 11-11 and 11-12. A single tune day isn't a reliable selector between them. Phase 3 should choose models and fit hybrid weights across several rolling-origin days, not 11-14 alone, and judge a hybrid on whether it holds up on every day.

**Ablations (tune protocol). Under the D-020 rule, five variants become Phase 3 candidates.**

| Variant | Δ nDCG@10 vs base [95% CI] |
|---|---|
| `popularity_24h` | +0.0049 [+0.0035, +0.0063] |
| `popularity_48h` | +0.0009 [+0.0003, +0.0016] (smaller than `popularity_24h`, but it passes the rule) |
| `category_affinity_aug` | +0.0045 [+0.0033, +0.0057]; zero-History users +0.030 [+0.013, +0.049] |
| `tfidf_aug` | +0.0046 [+0.0033, +0.0061] |
| `category_affinity_pop24h` | +0.0008 [+0.0003, +0.0014] |

- **Not a candidate:** `tfidf_pop24h`, −0.0002 with an interval that includes 0.
- **Augmentation helps cold start on tune.** Fallbacks drop from 772 to 160 tune impressions.
- **Expect a smaller effect on holdout.** 88.1% of dev users never appear in train impressions (D-011), so augmentation can only reach a minority of them.
- **What adoption means.** Candidates are added to the Phase 3 holdout run as named models. The v1 baseline config and its committed holdout numbers stay unchanged.

## D-022 Phase 3 protocol: hybrid folds and candidate handling (owner decision, 2026-11-05)
These are decisions for the next model step. Nothing here is implemented yet; this task's brief excludes new models.

**Hybrid weights use four chronological development days, not one.**
- Development days are the train days 11-11, 11-12, 11-13 and 11-14. For each day D, the base rankers are fitted only on train impressions before D's midnight, and their scores on D's impressions become that day's out-of-time features. D's own labels never reach the rankers that score D.
- Weights are fitted and selected on development data only. Exact folds:

  | Fold | Base rankers fitted on | Scored (features) | Labels used to fit weights |
  |---|---|---|---|
  | 1 | 11-09 to 11-10 | 11-11 | 11-11 |
  | 2 | 11-09 to 11-11 | 11-12 | 11-12 |
  | 3 | 11-09 to 11-12 | 11-13 | 11-13 |
  | 4 | 11-09 to 11-13 | 11-14 (tune) | 11-14 |

- Any weight-selection choice, such as regularization strength, is made by leave-one-day-out over these folds.
- Results are reported per day and on aggregate, with day-to-day stability, beside each base ranker.
- At evaluation time the frozen weights are applied to base rankers refitted on all train (fit + tune), following D-013.

**`popularity_48h` stays in the comparison.** It qualified under the D-020 rule, which was fixed before the run. It isn't removed because its observed effect was small. Whether a deployed hybrid uses it is decided from development results only.

**Exploratory vs confirmatory.** MIND-small dev has already been inspected (D-018, D-021). Any further comparison on it is labeled *exploratory holdout* in results and reports. Only a fresh evaluation set, with the comparison and rules fixed before it is opened, counts as *confirmatory* (proposed in `reports/reversal_memo.md`). The v1 manifest, evaluation conventions and leakage safeguards stay unchanged.

## D-023 Baseline diagnostics: design and rules (fixed before the real-data run)
**Why.** The brief asks for train/tune coverage, tie analysis, History-length and concentration segments, and a split of the tune/holdout reversal into a corpus-size part and an evaluation-population part. `scripts/baseline_diagnostics.py` reads `MINDsmall_train` only.

**Holdout use.** Holdout enters only as already-observed, descriptive reference points: the committed `results/experiments.json` (git `87067dd` results) and D-011's share of dev users never seen in train impressions (88.1%, user-level). Dev isn't loaded, and nothing is re-scored on it. Measures that would require reading dev, such as holdout History concentration, are left unmeasured.

**Design.** Each run scores the four baseline rankers on one train day, with a chosen statistics source.
- Days × statistics: 11-11 to 11-14, with all earlier train data, and with the trailing day only.
- Fixed targets: the tune day with trailing windows of 1, 2, 3, 4 days and all, plus seeded 25% and 50% subsamples of `fit`. The subsamples change size without changing recency.
- Population slices:
  - returning vs new users (the user has a train impression before the scored day)
  - top-category share of known History, with buckets [0, 1/3), [1/3, 1/2), [1/2, 2/3), [2/3, 1] for users with 6+ known items, plus separate "none" and "1-5 items" buckets
- Coverage and score-tie profiles per run.
- 500 user-grouped bootstrap samples.

**What I'd already seen.** Before fixing these rules I looked at two label-free counts on train:
- the share of each day's impressions from returning users (37%, 60%, 76%, 84% for 11-11 to 11-14)
- the spread of top-category shares on tune, which was used to sanity-check bucket sizes

Together with D-021's day-by-day gaps, the first count is what made returning vs new users a candidate explanation. That hypothesis was formed from the data, not before it.

**Verdict rules.**
- The swing is the holdout gap minus the official tune gap, where the gap is nDCG@10 for `tfidf` minus `category_affinity`. Each explanation gets a point estimate of the change in the tune gap it predicts:
  - **corpus size:** the largest move of the tune gap toward the holdout gap across the fixed-target runs
  - **History-length mix:** the tune gap reweighted to holdout's History-length bucket shares
  - **returning vs new:** the tune gap reweighted to holdout's new-user share
- The labels are fixed in advance:
  - "sufficient alone" means the explanation covers ≥ 50% of the swing
  - "partial" means 25–50%
  - "minor" means < 25%
- These are descriptive point estimates, not tests. The new-user reweighting treats a user-level share as an impression-level share, and assumes holdout's new users behave like tune's. Neither assumption can be checked without reading dev.

## D-024 Baseline diagnostics findings (train only; holdout used descriptively)
Source: `results/baseline_diagnostics.json` and `reports/baseline_diagnostics.md`, generated at `caae6bd` from a clean tree. A second run reproduced all 2,656 compared values exactly. D-023 was committed before either run. The interpretation is in `reports/reversal_memo.md`.

**Verdicts (D-023 rule).** The tune gap is +0.0137 and the holdout gap −0.0267, a swing of −0.0404. Each explanation's share of it:

| Explanation | Share of swing | Verdict |
|---|---|---|
| Corpus size (tune targets fixed; 25% subsample to all, and 1 to 5 trailing days) | 3% | minor |
| History-length mix | 0% | minor |
| Returning vs new users (my post-hoc hypothesis) | 4% | minor |

- **The returning-users hypothesis is not supported.** Within each train day, returning and new users show nearly the same gap.
- **The scored day drives the gap.** With a one-day statistics window, the gap by day is −0.0063, −0.0123, +0.0038 and +0.0125, nearly the same as with all earlier data. It shifts together within every History-length bucket with History, both returning and new users, and every concentration bucket except the smallest, most concentrated one.
- **Holdout's −0.0267 lies outside the train-day range,** but has the same sign as 11-11 and 11-12.

**Other findings.**
- **`category_affinity` ties.** 80–83% of clicked candidates share their primary score with an unclicked candidate, so the popularity tie-break orders them. For `tfidf` it is 15–17%.
- **Concentration (tune).** `tfidf` minus `category_affinity` falls as History concentration rises: +0.024, +0.014 and +0.008, then −0.012 (CI includes 0) for a top category above two thirds. The order of the lower three buckets holds on all four train days.
- **Coverage.** No candidate has an empty TF-IDF vector. 21–29% of candidates fall outside the TF-IDF fitting corpus. 54–60% were never clicked in the statistics source.
- **Not pre-registered.** Mean list length (35.6 to 40.4) is higher on the two days that favored `tfidf`. That's four points, a lead for follow-up E1.

**Consequences.**
- The user-bootstrap intervals ignore day-to-day variation, which is the dominant uncertainty here. Model comparisons must report consistency across days (D-022).
- The memo proposes E1 (within-category discrimination) and E2 (a leave-one-day-out hybrid with one pre-specified concentration interaction), plus a fresh confirmatory protocol. MIND-small dev results from now on are labeled exploratory.

## D-025 E2 development protocol (fixed before the real-data run, 2026-11-18)
This implements D-022 and the memo's E2. `scripts/hybrid_development.py` reads `MINDsmall_train` only. No holdout data is loaded or scored in this step.

**Folds** follow the D-022 table: development days 11-11 to 11-14, each scored by base rankers fitted on train before it.
- **Primary: leave one day out.** The weights that score day D are fitted on the other three days' rows.
  - Caveat: those days include later ones, and their features were built from statistics that contain D's clicks. D's labels never reach D's own features or weights directly.
- **Secondary check: forward chaining.** Weights fitted on earlier days only, scoring 11-12 to 11-14. It's reported, not used to decide anything.

**Model.**
- Logistic regression (L2, lbfgs) over candidate rows of non-fallback impressions, with each feature standardized on its training rows.
- Ranking uses the linear score.
- C is fixed at 1.0 in advance and not tuned. With millions of rows, L2 strength barely matters. C = 0.01 and 100 are reported as a sensitivity check only.
- The hybrid falls back to the popularity order, flagged, when none of its personalized features is available.

**Features.** All come from a label-free request.

| Feature | Definition |
|---|---|
| `pop_item`, `pop_category` | log1p as-of clicks on the item and on its category |
| `affinity` | (same-category + same-subcategory History count) / known History length |
| `tfidf` | cosine similarity |
| `tfidf_x_concentration` | `tfidf` × top-category share of the History, pre-specified from D-024 |

A main effect for concentration would be a per-user constant, which can't change a within-impression order, so concentration enters only through the interaction.

**Models compared.**
- `hybrid`: the primary model, using the features above.
- `hybrid_no_interaction`: an ablation without `tfidf_x_concentration`.
- `hybrid_full`: `hybrid` plus the optional groups that pass the inclusion rule. It isn't built if no group passes.
- Every base ranker and D-021 candidate as a named model, including `popularity_48h` (D-022).

**Inclusion rule for optional groups.** The groups are `pop_24h`, `pop_48h`, `affinity_aug` and `tfidf_aug`. `category_affinity_pop24h` adds no new signal beyond `pop_24h`. Each group is added alone to `hybrid` and kept iff its leave-one-day-out nDCG@10 is higher on at least three of the four days. This is a point-estimate rule on development data, applied once.

**Success rule (memo E2).** A model succeeds only if both hold:
- it beats both `category_affinity` and `tfidf` on at least three of four held-out days
- its pooled paired nDCG@10 interval against `category_affinity` is above 0

The pooled bootstrap resamples users across all four days. Relative improvement against `category_affinity` is reported next to the spec's ≥ 5% holdout target, but development results aren't holdout results.

**Frozen weights.** Weights fitted on all four days are saved in `results/hybrid_development.json` for a later holdout step. That step needs owner approval and follows D-022: exploratory on MIND-small dev, confirmatory only on fresh data.

## D-026 E2 development findings (train only; no holdout scored)
Source: `results/hybrid_development.json` and `reports/hybrid_development.md`, generated at `a73d98f` from a clean tree, where D-025 was committed. On each day, the live hybrid scores match the cached features the weights were fitted on. The base rankers' 11-14 numbers equal the official tune run.

**Success rule (D-025).** Pooled across the four held-out days:

| Model | Days beating both rivals | Pooled Δ vs `category_affinity` [95% CI] | Relative |
|---|---|---|---|
| `hybrid` (primary, pre-specified) | 4 of 4 | +0.0153 [+0.0140, +0.0167] | +4.1% |
| `hybrid_no_interaction` | 4 of 4 | +0.0157 [+0.0144, +0.0171] | +4.2% |
| `hybrid_full` | 4 of 4 | +0.0236 [+0.0223, +0.0250] | +6.3% |

All three meet the success rule. Forward chaining agrees: `hybrid` beats both rivals on 11-12, 11-13 and 11-14. The result doesn't depend on C: 0.01 to 100 change nDCG@10 by at most 0.0001.

**Failure modes and unexpected behavior.**
- **The pre-specified interaction adds nothing.** `tfidf_x_concentration` gets a stable negative weight (about −0.15 in every fold), which fits D-024's finding 6. But `hybrid_no_interaction` is as good or slightly better. The concentration effect is already captured by `affinity` and `tfidf` together.
- **Gains are uneven across days.** Against `category_affinity`, the hybrid gains +0.005 on 11-11 and 11-12 (days that favor `category_affinity`) but +0.025 to +0.027 on 11-13 and 11-14. It mostly tracks whichever content ranker is better that day. The holdout day favored `category_affinity`, so a gain near the small end is plausible there. That would be about 1–2% relative, below the spec's 5% target. This is an extrapolation, not a measurement.
- **Popularity gets little weight in `hybrid`.** `pop_item` drifts from −0.07 to +0.01 across folds, and `pop_category` is negative. Cumulative as-of popularity adds little once the content scores are in. Recent popularity does help: `pop_24h` raises the hybrid on all four days.
- **`hybrid_full` is optimistic and hard to interpret.**
  - All four optional groups passed the inclusion rule. The rule was evaluated on the same held-out days that report `hybrid_full`, so its development numbers carry selection optimism. The primary `hybrid` has no such step.
  - Its weights are collinear: popularity windows at −0.35, +0.54 and −0.19, and `affinity` drops to +0.04 because `affinity_aug` takes over. Individual weights can't be read as importances.
  - It is also the only hybrid that helps zero-History users: +0.031 [+0.022, +0.040] through augmentation, with fallbacks down from 2,849 to 1,136 pooled impressions.
- **The latency comparison in the run is unreliable.** `hybrid` shows a p95 of 0.7–1.1 ms against 0.2–0.3 ms for `hybrid_no_interaction`. An isolated re-timing on 6,000 tune requests gave the same feature cost for both, about 0.24 ms at p95. Treat the run's p95 columns for prebuilt rankers as rough. The p50s, about 0.1–0.3 ms, agree with the re-timing.

**Next decision for the owner.** Two things remain:
- A holdout step needs approval. Under D-022 it is *exploratory* on MIND-small dev. The comparison to freeze first: `hybrid` (primary), with `hybrid_full` and `hybrid_no_interaction` as secondary.
- *Confirmatory* testing needs fresh data, per the memo.

The frozen all-days weights for both hybrids are in the results JSON.

## D-027 E2 exploratory holdout: what is frozen before scoring (owner-approved, 2026-11-18)
The owner approved one *exploratory holdout* run of the E2 hybrids on MIND-small dev (D-022). Dev was already inspected (D-018, D-021), so this run is not confirmatory, and its numbers can't change any frozen decision. This entry, the code and `eval/hybrid_freeze.json` are committed before dev is scored. `scripts/hybrid_holdout.py` refuses to score unless the freeze file is committed and unchanged, and never overwrites a freeze.

**Frozen (train only, `--freeze`):**
- **Weights and standardization constants** for `hybrid`, `hybrid_no_interaction` and `hybrid_full`. Each is fitted on all four development days (11-11 to 11-14, D-025 folds) with C = 1.0.
  - `hybrid` and `hybrid_full` must reproduce the weights stored in `results/hybrid_development.json` (D-026), or the freeze fails.
  - `hybrid_no_interaction` wasn't stored there, so it's fitted the same way.
  - `hybrid_full` keeps the four optional groups selected in D-026.
- **The freeze also records:** the manifest's train file hashes (checked against disk), the TF-IDF parameters, the base model list (including `popularity_48h`, D-022), and the analysis plan below.

**Evaluation conventions (unchanged).**
- Base features come from all train (fit + tune), per D-013 and D-022. News metadata is merged as in D-013. Seed, tie-break, eligibility and metric definitions are unchanged.
- The v1 manifest is verified and not edited.
- `results/summary.json`, `results/experiments.json` and the v1 figures are not touched. The portfolio headline stays the v1 run unless the owner decides otherwise.
- **Continuity check:** the base rankers `random`, `popularity`, `category_affinity` and `tfidf` must reproduce the v1 holdout means exactly. If they don't, the run isn't comparable and is reported as such.

**Analysis plan.**
- **Primary:** `hybrid` vs `category_affinity` on nDCG@10, paired, user-grouped bootstrap, 1,000 samples, seed 13. The reading is one of:
  - *better, meets target*: interval above 0 and relative gain ≥ 5% (the spec's target)
  - *better, below target*: interval above 0, relative gain < 5%
  - *no clear difference*: interval includes 0
  - *worse*: interval below 0
- **Secondary (same readings):** `hybrid_no_interaction` and `hybrid_full` vs `category_affinity`, and each hybrid vs `tfidf`.
- **Descriptive only:** history-length segments, `sports` slices, costs, and the development-vs-holdout comparison.

**Expectations stated before scoring (from D-026 and D-024).**
1. `hybrid` beats `category_affinity`, but below the 5% target, around 1–2% relative. The holdout day favored `category_affinity`, and on development days like that the gain was about +0.005.
2. `hybrid_full`'s gain for zero-History users shrinks or disappears. 88% of holdout users are absent from train (D-011), so History augmentation rarely applies.
3. `hybrid_full` is optimistic by construction (D-026). A larger holdout gain for it wouldn't settle whether its extra groups help.

**Known limitation.** The standardization constants come from development features whose statistics covered two to five train days. At evaluation the statistics cover all six days, so cumulative popularity values are larger. Ranking within an impression isn't affected by a constant shift, but it is affected by scale. The cumulative popularity weights are near zero, so I expect only a small effect. The 24h and 48h windows have a fixed length and shift less.
