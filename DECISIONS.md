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
