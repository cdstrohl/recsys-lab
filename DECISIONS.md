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
