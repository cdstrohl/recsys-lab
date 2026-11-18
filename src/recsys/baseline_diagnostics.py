"""Baseline diagnostics: coverage, ties, segments, and why tune and holdout disagree (D-023).

Reads `MINDsmall_train` only. The holdout enters solely through the committed
`results/experiments.json` and one count from D-011, both already observed, so they
are used as descriptive reference points and nothing here feeds back into a model.

Every run scores the four baseline rankers on one train day's impressions (the
targets) with rankers fitted on a chosen statistics source:

- **Days x statistics.** Each listed day with all train before it ("all") and with
  trailing windows. The last train day with "all" is the official tune protocol.
- **Fixed targets, varying corpus (tune day).** Trailing windows of 1 to all days, and
  seeded random subsamples of `fit`. The targets stay the same, so any change in the
  `tfidf` minus `category_affinity` gap comes from the statistics source alone.

Each run is sliced by population: returning vs new users (a train impression before
the day), and History concentration (top-category share). Coverage and tie profiles
are recorded per run. Two descriptive verdicts compare how much of the tune-holdout
swing each explanation can account for, by the rules fixed in D-023.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from recsys.diagnostics import _subset, unseen_slices
from recsys.evaluate import EvalConfig, _git, evaluate_split, load_config, metric_label
from recsys.inventory import read_provenance, unseen_inventory
from recsys.loaders import TRAIN_DIR, load_mind_dir
from recsys.rankers import (
    CategoryAffinityRanker,
    PopularityRanker,
    Ranker,
    RankRequest,
    TfidfIndex,
    TfidfRanker,
    tfidf_corpus,
    top_category_share,
)
from recsys.schemas import Impression, NewsItem
from recsys.splits import assign_splits

REPRODUCE = (
    "uv run python scripts/baseline_diagnostics.py --config configs/baseline_diagnostics.yaml"
)
GAP = ("tfidf", "category_affinity")  # the pair whose order flipped between tune and holdout
FLOOR = ("popularity", "random")
MODELS = ("random", "popularity", "category_affinity", "tfidf")


@dataclass(frozen=True, slots=True)
class HoldoutReference:
    experiments: Path  # committed results; read, never written
    new_user_share: float | None  # users never in train impressions (D-011, user-level)


@dataclass(frozen=True, slots=True)
class BaselineDiagnosticsConfig:
    base: EvalConfig
    days: tuple[date, ...]
    day_windows: tuple[int | None, ...]  # trailing days of statistics; None = all before
    tune_windows: tuple[int | None, ...]
    tune_subsamples: tuple[float, ...]
    concentration_min_history: int
    concentration_edges: tuple[float, ...]
    holdout: HoldoutReference | None
    results: Path
    report: Path


def load_baseline_diagnostics_config(path: Path) -> BaselineDiagnosticsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw["baseline_diagnostics"]
    holdout = section.get("holdout_reference")
    return BaselineDiagnosticsConfig(
        base=load_config(path),
        days=tuple(d if isinstance(d, date) else date.fromisoformat(d) for d in section["days"]),
        day_windows=tuple(section["day_windows"]),
        tune_windows=tuple(section["tune_windows"]),
        tune_subsamples=tuple(float(f) for f in section.get("tune_subsamples", [])),
        concentration_min_history=int(section["concentration"]["min_known_history"]),
        concentration_edges=tuple(float(e) for e in section["concentration"]["edges"]),
        holdout=HoldoutReference(
            Path(holdout["experiments"]),
            holdout.get("new_user_share"),
        )
        if holdout
        else None,
        results=Path(section["outputs"]["results"]),
        report=Path(section["outputs"]["report"]),
    )


# --- Statistics sources and population slices -----------------------------------------------


def trailing(
    impressions: Sequence[Impression], end: datetime, days: int | None
) -> list[Impression]:
    """Impressions before `end`, and within the last `days` days of it if given."""
    start = end - timedelta(days) if days is not None else None
    return [i for i in impressions if i.timestamp < end and (start is None or i.timestamp >= start)]


def subsample(impressions: Sequence[Impression], fraction: float, seed: int) -> list[Impression]:
    """A seeded random share of impressions, chosen by a hash of the impression ID."""

    def draw(impression_id: str) -> float:
        digest = hashlib.blake2b(f"subsample\x1f{seed}\x1f{impression_id}".encode(), digest_size=8)
        return int.from_bytes(digest.digest(), "big") / 2**64

    return [i for i in impressions if draw(i.impression_id) < fraction]


def returning_slices(
    targets: Sequence[Impression], earlier: Sequence[Impression]
) -> dict[str, tuple[str, list[int]]]:
    """Impressions whose user has (returning) or hasn't (new) an impression in `earlier`."""
    seen = {i.user_id for i in earlier}
    rows: dict[bool, list[int]] = {True: [], False: []}
    for i, imp in enumerate(targets):
        rows[imp.user_id in seen].append(i)
    return {
        "returning_user": ("user has a train impression before this day", rows[True]),
        "new_user": ("user has no train impression before this day", rows[False]),
    }


def _pct(x: float) -> int:
    return round(x * 100)


def concentration_slices(
    targets: Sequence[Impression],
    news: Mapping[str, NewsItem],
    min_history: int,
    edges: Sequence[float],
) -> dict[str, tuple[str, list[int]]]:
    """Partition impressions by the top-category share of the user's known History.

    Users with fewer than `min_history` known items get their own buckets, since a
    share over one or two items says little about concentration.
    """
    bounds = [0.0, *edges, 1.0]
    slices: dict[str, tuple[str, list[int]]] = {
        "history_none": ("no known History item", []),
        f"history_1_{min_history - 1}": (f"1 to {min_history - 1} known History items", []),
    }
    names = []
    for low, high in pairwise(bounds):
        name = f"top_category_{_pct(low)}_{_pct(high)}"
        upper = "]" if high == 1.0 else ")"
        slices[name] = (
            f"{min_history}+ known items; top-category share in [{low:.0%}, {high:.0%}{upper}",
            [],
        )
        names.append(name)
    for i, imp in enumerate(targets):
        categories = [news[h].category for h in imp.history if h in news]
        if not categories:
            slices["history_none"][1].append(i)
        elif len(categories) < min_history:
            slices[f"history_1_{min_history - 1}"][1].append(i)
        else:
            share = top_category_share([news[h] for h in imp.history if h in news]) or 0.0
            bucket = next(b for b, high in enumerate(bounds[1:]) if share < high or high == 1.0)
            slices[names[bucket]][1].append(i)
    return slices


# --- Coverage and ties -----------------------------------------------------------------------


def _tie_measures(scores: Sequence[np.ndarray], clicked: Sequence[np.ndarray]) -> dict[str, Any]:
    """Mean share of tied candidate pairs per impression, and the share of clicked
    candidates sharing their score with an unclicked one (the tie-break places them)."""
    pair_shares = []
    n_clicked = n_clicked_tied = 0
    for s, c in zip(scores, clicked, strict=True):
        n = len(s)
        if n > 1:
            _, counts = np.unique(s, return_counts=True)
            pair_shares.append(float((counts * (counts - 1)).sum() / (n * (n - 1))))
        n_clicked += int(c.sum())
        n_clicked_tied += int(np.isin(s[c], s[~c]).sum())
    return {
        "n_impressions": len(scores),
        "share_pairs_tied": float(np.mean(pair_shares)) if pair_shares else None,
        "share_clicks_tied_with_a_non_click": n_clicked_tied / n_clicked if n_clicked else None,
    }


def coverage_and_ties(
    targets: Sequence[Impression],
    stats: Sequence[Impression],
    news: Mapping[str, NewsItem],
    tfidf_params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Label-free coverage of the targets by the statistics source, and score-tie profiles.

    Ties are on each ranker's own key: popularity's (item, category) clicks, and the
    personalized rankers' primary score alone, over impressions they don't fall back on.
    """
    stat_users = {i.user_id for i in stats}
    corpus = tfidf_corpus(stats)
    index = TfidfIndex(news, corpus, **tfidf_params)
    empty_rows = np.diff(index.matrix.indptr) == 0
    candidates = [c.news_id for imp in targets for c in imp.candidates]
    history = [h for imp in targets for h in imp.history]
    users = {i.user_id for i in targets}
    _, unclicked = unseen_slices(targets, stats)

    def share(part: int, whole: int) -> float | None:
        return part / whole if whole else None

    coverage = {
        "n_impressions": len(targets),
        "n_users": len(users),
        "n_statistics_impressions": len(stats),
        "candidates_per_impression": share(len(candidates), len(targets)),
        "share_impressions_user_in_statistics": share(
            sum(i.user_id in stat_users for i in targets), len(targets)
        ),
        "share_users_in_statistics": share(len(users & stat_users), len(users)),
        "share_impressions_no_known_history": share(
            sum(not any(h in news for h in i.history) for i in targets), len(targets)
        ),
        "share_history_items_unknown": share(sum(h not in news for h in history), len(history)),
        "share_candidates_unseen_as_candidate": unseen_inventory(stats, targets)[
            "share_occurrences_unseen_as_candidate"
        ],
        "share_candidates_unclicked_before": unclicked,
        "share_candidates_outside_tfidf_corpus": share(
            sum(n not in corpus for n in candidates), len(candidates)
        ),
        "share_candidates_zero_tfidf_vector": share(
            sum(n not in index.row or bool(empty_rows[index.row[n]]) for n in candidates),
            len(candidates),
        ),
        "tfidf_vocabulary_size": len(index.vectorizer.vocabulary_),
    }

    popularity = PopularityRanker(stats, news)
    rankers: dict[str, Ranker] = {
        "popularity": popularity,
        "category_affinity": CategoryAffinityRanker(popularity, news, tie_break=False),
        "tfidf": TfidfRanker(popularity, index, tie_break=False),
    }
    ties = {}
    for name, ranker in rankers.items():
        scores, clicked = [], []
        for imp in targets:
            ranking = ranker.score(RankRequest.from_impression(imp))
            if ranking.fallback:
                continue
            scores.append(ranking.scores)
            clicked.append(np.array([c.clicked for c in imp.candidates], dtype=bool))
        ties[name] = _tie_measures(scores, clicked)
    return coverage, ties


# --- Compact results -------------------------------------------------------------------------


def _paired(models: Mapping[str, Any], a: str, b: str, metric: str) -> dict[str, Any] | None:
    """a minus b with its paired CI, from whichever side recorded the delta."""
    delta = models.get(a, {}).get("metrics", {}).get(metric, {}).get("delta_vs", {}).get(b)
    if delta is not None:
        return dict(delta)
    delta = models.get(b, {}).get("metrics", {}).get(metric, {}).get("delta_vs", {}).get(a)
    if delta is None:
        return None
    return {"mean": -delta["mean"], "ci_low": -delta["ci_high"], "ci_high": -delta["ci_low"]}


def _block(n: int, models: Mapping[str, Any], metric: str) -> dict[str, Any]:
    return {
        "n_impressions": n,
        "means": {m: r["metrics"][metric]["mean"] for m, r in models.items()},
        "gap": _paired(models, *GAP, metric),
        "popularity_minus_random": _paired(models, *FLOOR, metric),
    }


def compact(result: Mapping[str, Any], metric: str) -> dict[str, Any]:
    """The headline metric, the gap and the floor for the split, its segments and slices."""
    models = result["models"]
    return {
        **_block(result["n_impressions"], models, metric),
        "n_users": result["n_users"],
        "n_statistics_impressions": result["n_statistics_impressions"],
        "share_impressions_tied": {
            m: r["n_tied"] / result["n_impressions"] for m, r in models.items()
        },
        "history_length": {
            b: _block(s["n_impressions"], s["models"], metric)
            for b, s in result["segments_by_history_length"].items()
        },
        "slices": {
            name: {
                "description": s["description"],
                **_block(s["n_impressions"], s["models"], metric),
            }
            for name, s in result["slices"].items()
        },
    }


def holdout_reference(reference: HoldoutReference, metric: str) -> dict[str, Any] | None:
    """Descriptive holdout figures from the committed results (already observed)."""
    if not reference.experiments.exists():
        return None
    experiments = json.loads(reference.experiments.read_text(encoding="utf-8"))
    holdout = experiments["splits"].get("holdout")
    if holdout is None:
        return None
    return {
        "source": reference.experiments.as_posix(),
        "git_sha": experiments.get("git_sha"),
        "statistics_source": holdout["statistics_source"],
        "share_candidates_unseen_as_candidate": holdout[
            "share_candidate_occurrences_unseen_in_statistics"
        ],
        "share_users_new_to_train": reference.new_user_share,
        **compact({**holdout, "slices": {}}, metric),
    }


# --- Run -------------------------------------------------------------------------------------


def _label(window: int | None, fraction: float | None = None) -> str:
    if fraction is not None:
        return f"all_sample{_pct(fraction)}"
    return "all" if window is None else f"{window}d"


def run_baseline_diagnostics(config: BaselineDiagnosticsConfig) -> dict[str, Any]:
    train = load_mind_dir(config.base.data_dir / TRAIN_DIR)
    assignment = assign_splits(train.impressions, ())  # no dev: holdout is never loaded
    news = train.news
    base = _subset(
        dataclasses.replace(config.base, slice_category=None),
        MODELS,
        ("category_affinity", "random"),
    )
    metric = f"ndcg@{max(base.ks)}"
    tfidf_params = next(
        {k: v for k, v in m.params.items() if k not in ("tie_break",)}
        for m in base.models
        if m.kind == "tfidf"
    )
    tune_day = assignment.cutoff.date()

    plan: list[tuple[date, int | None, float | None]] = [
        (d, w, None) for d in config.days for w in config.day_windows
    ]
    plan += [(tune_day, w, None) for w in config.tune_windows]
    plan += [(tune_day, None, f) for f in config.tune_subsamples]

    runs: dict[str, Any] = {}
    for day, window, fraction in plan:
        run_id = f"{day.isoformat()}/{_label(window, fraction)}"
        if run_id in runs:
            continue
        start = datetime.combine(day, time.min)
        earlier = trailing(train.impressions, start, None)
        stats = trailing(train.impressions, start, window)
        if fraction is not None:
            stats = subsample(stats, fraction, config.base.seed)
        targets = [i for i in train.impressions if start <= i.timestamp < start + timedelta(1)]
        if not stats or not targets:
            raise ValueError(f"{run_id}: needs train impressions before and on that day")
        slices = returning_slices(targets, earlier)
        slices.update(
            concentration_slices(
                targets, news, config.concentration_min_history, config.concentration_edges
            )
        )
        result = evaluate_split(
            run_id, targets, stats, run_id.split("/")[1], news, base, extra_slices=slices
        )
        coverage, ties = coverage_and_ties(targets, stats, news, tfidf_params)
        runs[run_id] = {
            "day": day.isoformat(),
            "statistics": _label(window, fraction),
            "window_days": window,
            "sample_fraction": fraction,
            "fixed_targets": day == tune_day,
            **compact(result, metric),
            "coverage": coverage,
            "ties": ties,
        }

    reference = holdout_reference(config.holdout, metric) if config.holdout else None
    new_share = config.holdout.new_user_share if config.holdout else None
    status = _git("status", "--porcelain", "--", ".", ":!results", ":!reports")
    provenance = read_provenance(config.base.data_dir)
    return {
        "project": "recsys-lab",
        "kind": "baseline_diagnostics",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(status) if status is not None else None,
        "reproduce": REPRODUCE,
        "data_provenance": [
            config.base.data_provenance if provenance == "unlabeled" else provenance
        ],
        "holdout_used": False,
        "config": {
            "seed": base.seed,
            "bootstrap_samples": base.bootstrap_samples,
            "metric": metric,
            "history_edges": list(base.history_edges),
            "tune_day": tune_day.isoformat(),
            "days": [d.isoformat() for d in config.days],
            "concentration_min_history": config.concentration_min_history,
            "concentration_edges": list(config.concentration_edges),
            "models": {m.name: {"ranker": m.kind, "params": m.params} for m in base.models},
        },
        "holdout_reference": reference,
        "runs": runs,
        "verdicts": verdicts(runs, tune_day.isoformat(), new_share, reference),
    }


# --- Verdicts (rules fixed in D-023) ---------------------------------------------------------


def _gap(block: Mapping[str, Any]) -> float | None:
    gap = block.get("gap")
    return gap["mean"] if gap else None


def verdicts(
    runs: Mapping[str, Any],
    tune_day: str,
    new_user_share: float | None,
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """How much of the tune-to-holdout gap swing each explanation accounts for.

    The swing is the holdout gap minus the official tune gap. Each explanation gets a
    point estimate of the change it predicts, as a fraction of the swing:
    - corpus size: the largest move of the tune gap across statistics sources with the
      tune targets fixed, signed toward holdout;
    - History-length mix: the tune gap reweighted to holdout's bucket composition;
    - returning vs new users: the tune gap reweighted to holdout's new-user share.
    An explanation is "sufficient alone" at >= 0.5 of the swing, "minor" below 0.25.
    """
    tune = runs.get(f"{tune_day}/all")
    tune_gap = _gap(tune) if tune else None
    holdout_gap = _gap(reference) if reference else None
    if tune is None or reference is None or tune_gap is None or holdout_gap is None:
        return {"available": False}
    swing = holdout_gap - tune_gap
    out: dict[str, Any] = {
        "available": True,
        "tune_gap": tune_gap,
        "holdout_gap": holdout_gap,
        "swing": swing,
    }

    def explained(predicted_change: float) -> dict[str, Any]:
        fraction = predicted_change / swing if swing else None
        label = (
            "n/a"
            if fraction is None
            else "sufficient alone"
            if fraction >= 0.5
            else "partial"
            if fraction >= 0.25
            else "minor"
        )
        return {"predicted_change": predicted_change, "fraction_of_swing": fraction, "label": label}

    fixed = [r for r in runs.values() if r["fixed_targets"]]
    moves = [g - tune_gap for r in fixed if (g := _gap(r)) is not None]
    toward = max(moves, key=lambda m: m * np.sign(swing)) if moves else 0.0
    out["corpus_size"] = {**explained(toward), "n_runs": len(fixed)}

    weights = {b: s["n_impressions"] for b, s in reference["history_length"].items()}
    total = sum(weights.values())
    reweighted = sum(
        weights[b] / total * (_gap(s) or 0.0)
        for b, s in tune["history_length"].items()
        if b in weights
    )
    out["history_length_mix"] = {**explained(reweighted - tune_gap), "reweighted_gap": reweighted}

    returning = _gap(tune["slices"].get("returning_user", {}))
    new = _gap(tune["slices"].get("new_user", {}))
    if new_user_share is not None and returning is not None and new is not None:
        reweighted = new_user_share * new + (1 - new_user_share) * returning
        out["returning_vs_new"] = {**explained(reweighted - tune_gap), "reweighted_gap": reweighted}
    return out


# --- Report ----------------------------------------------------------------------------------


def _num(value: float | None, fmt: str = ".4f") -> str:
    return "—" if value is None else format(value, fmt)


def _ci(delta: Mapping[str, Any] | None) -> str:
    if not delta or delta.get("ci_low") is None:
        return "—" if not delta else f"{delta['mean']:+.4f}"
    return f"{delta['mean']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


COVERAGE_ROWS = (
    ("n_statistics_impressions", "Statistics impressions", ",d"),
    ("n_impressions", "Scored impressions", ",d"),
    ("candidates_per_impression", "Candidates per impression", ".1f"),
    ("share_impressions_user_in_statistics", "Impressions whose user is in statistics", ".1%"),
    ("share_users_in_statistics", "Users in statistics", ".1%"),
    ("share_impressions_no_known_history", "Impressions with no known History", ".1%"),
    ("share_history_items_unknown", "History items missing from news metadata", ".1%"),
    ("share_candidates_unseen_as_candidate", "Candidates never a candidate in statistics", ".1%"),
    ("share_candidates_unclicked_before", "Candidates never clicked in statistics", ".1%"),
    ("share_candidates_outside_tfidf_corpus", "Candidates outside the TF-IDF corpus", ".1%"),
    ("share_candidates_zero_tfidf_vector", "Candidates with an empty TF-IDF vector", ".1%"),
    ("tfidf_vocabulary_size", "TF-IDF vocabulary", ",d"),
)


def render_baseline_diagnostics(results: Mapping[str, Any]) -> str:
    config = results["config"]
    label = metric_label(config["metric"])
    runs = results["runs"]
    ref = results["holdout_reference"]
    day_runs = {r["day"]: r for r in runs.values() if r["statistics"] == "all"}
    days = list(day_runs)
    tune_day = config["tune_day"]
    tune = day_runs[tune_day]
    gap_name = f"`{GAP[0]}` minus `{GAP[1]}`"
    lines = [
        "# Baseline diagnostics: coverage, ties, segments, and the tune/holdout reversal",
        "",
        "> Generated by `scripts/baseline_diagnostics.py`; do not edit by hand. Reads"
        " `MINDsmall_train` only. Holdout figures come from the committed"
        " `results/experiments.json` and DECISIONS.md D-011: they were observed before this"
        " analysis and are used descriptively, never to choose a model. Offline ranking"
        " metrics, **not** engagement or CTR. Rules fixed in DECISIONS.md D-023.",
        "",
        f"- Data provenance: `{', '.join(results['data_provenance'])}`",
        f"- Generated: {results['generated_at']} at git `{results['git_sha']}`"
        + (" (uncommitted changes)" if results["git_dirty"] else ""),
        f"- Reproduce: `{results['reproduce']}`",
        f"- Seed {config['seed']}; {config['bootstrap_samples']} user-grouped bootstrap samples;"
        f' 95% percentile intervals. "Gap" is the paired {label} difference {gap_name}.',
        f"- Each day is scored with rankers fitted on all train impressions before it"
        f" (`all`); {tune_day} with `all` is the official tune protocol.",
    ]
    if ref:
        lines.append(
            f"- Holdout reference: `{ref['source']}` (git `{ref['git_sha']}`), statistics"
            f" `{ref['statistics_source']}`."
        )

    columns = [*days, *(["holdout (ref.)"] if ref else [])]
    lines += ["", "## 1. Data coverage (label-free)", ""]
    rows = []
    for key, name, fmt in COVERAGE_ROWS:
        row = [name, *(_num(day_runs[d]["coverage"][key], fmt) for d in days)]
        if ref:
            value = {
                "n_statistics_impressions": ref["n_statistics_impressions"],
                "n_impressions": ref["n_impressions"],
                "share_candidates_unseen_as_candidate": ref["share_candidates_unseen_as_candidate"],
                "share_users_in_statistics": None
                if ref["share_users_new_to_train"] is None
                else 1 - ref["share_users_new_to_train"],
            }.get(key)
            row.append(_num(value, fmt))
        rows.append(row)
    lines += _table(["", *columns], rows)
    lines += [
        "",
        'Holdout values are the ones already recorded; "—" means not measured, because'
        " measuring it would mean reading dev.",
    ]

    lines += ["", "## 2. Score ties", ""]
    rows = []
    for model in ("popularity", "category_affinity", "tfidf"):
        rows.append(
            [
                f"`{model}`: candidate pairs tied",
                *(_num(day_runs[d]["ties"][model]["share_pairs_tied"], ".1%") for d in days),
                *(["—"] if ref else []),
            ]
        )
        rows.append(
            [
                f"`{model}`: clicks tied with a non-click",
                *(
                    _num(day_runs[d]["ties"][model]["share_clicks_tied_with_a_non_click"], ".1%")
                    for d in days
                ),
                *(["—"] if ref else []),
            ]
        )
    for model in ("popularity", "category_affinity", "tfidf"):
        rows.append(
            [
                f"`{model}`: impressions with any tie (with tie-breaks)",
                *(_num(day_runs[d]["share_impressions_tied"][model], ".1%") for d in days),
                *([_num(ref["share_impressions_tied"][model], ".1%")] if ref else []),
            ]
        )
    lines += _table(["", *columns], rows)
    lines += [
        "",
        "The first two rows per ranker use its own key only: popularity's (item, category)"
        " clicks, and the personalized rankers' primary score without the popularity"
        " tie-break, over impressions they don't fall back on. \"Clicks tied with a"
        ' non-click": the click shares its score with an unclicked candidate, so a'
        " tie-break decides their order.",
    ]

    lines += ["", "## 3. History length", "", "Share of impressions by History length:", ""]
    buckets = list(tune["history_length"])

    def share_row(run: Mapping[str, Any], b: str) -> str:
        return _num(run["history_length"][b]["n_impressions"] / run["n_impressions"], ".1%")

    lines += _table(
        ["History length", *columns],
        [
            [b, *(share_row(day_runs[d], b) for d in days), *([share_row(ref, b)] if ref else [])]
            for b in buckets
        ],
    )
    lines += ["", f"Gap by History length ({gap_name}):", ""]
    lines += _table(
        ["History length", *columns],
        [
            [
                b,
                *(_ci(day_runs[d]["history_length"][b]["gap"]) for d in days),
                *([_ci(ref["history_length"][b]["gap"])] if ref else []),
            ]
            for b in buckets
        ]
        + [
            [
                "**all**",
                *(_ci(day_runs[d]["gap"]) for d in days),
                *([_ci(ref["gap"])] if ref else []),
            ]
        ],
    )

    conc = [k for k in tune["slices"] if k.startswith(("history_", "top_category_"))]
    lines += [
        "",
        "## 4. History concentration",
        "",
        f"Top-category share of the user's known History; {config['concentration_min_history']}+"
        " known items for a share bucket. Share of impressions by day:",
        "",
    ]
    lines += _table(
        ["Bucket", *days],
        [
            [
                f"`{k}`",
                *(
                    _num(
                        day_runs[d]["slices"][k]["n_impressions"] / day_runs[d]["n_impressions"],
                        ".1%",
                    )
                    for d in days
                ),
            ]
            for k in conc
        ],
    )
    lines += ["", f"{label} on {tune_day} (official tune protocol) by concentration:", ""]
    lines += _table(
        ["Bucket", "Impressions", *(f"`{m}`" for m in MODELS), "Gap [95% CI]"],
        [
            [
                f"`{k}`",
                f"{tune['slices'][k]['n_impressions']:,}",
                *(_num(tune["slices"][k]["means"][m]) for m in MODELS),
                _ci(tune["slices"][k]["gap"]),
            ]
            for k in conc
        ],
    )
    lines += ["", f"Gap by concentration and day ({gap_name}):", ""]
    lines += _table(
        ["Bucket", *days],
        [[f"`{k}`", *(_ci(day_runs[d]["slices"][k]["gap"]) for d in days)] for k in conc],
    )

    lines += [
        "",
        "## 5. Corpus size vs evaluation population",
        "",
        f"### 5a. Same targets ({tune_day}), different statistics sources",
        "",
        "Only the statistics the rankers are fitted on change, so any movement in the gap"
        " is a corpus effect. Trailing windows also change recency; random subsamples of"
        " `all` change size alone.",
        "",
    ]
    fixed = [r for r in runs.values() if r["fixed_targets"]]
    fixed.sort(key=lambda r: r["n_statistics_impressions"])
    lines += _table(
        [
            "Statistics",
            "Impressions",
            "Never clicked",
            "Empty TF-IDF",
            "Vocabulary",
            "`popularity` minus `random`",
            "Gap [95% CI]",
        ],
        [
            [
                f"`{r['statistics']}`",
                f"{r['n_statistics_impressions']:,}",
                _num(r["coverage"]["share_candidates_unclicked_before"], ".1%"),
                _num(r["coverage"]["share_candidates_zero_tfidf_vector"], ".1%"),
                f"{r['coverage']['tfidf_vocabulary_size']:,}",
                _ci(r["popularity_minus_random"]),
                _ci(r["gap"]),
            ]
            for r in fixed
        ],
    )
    windows = sorted(
        {r["statistics"] for r in runs.values() if not r["fixed_targets"]},
        key=lambda s: (s == "all", s),
    )
    lines += [
        "",
        "### 5b. Different days, same rule for statistics",
        "",
        "Gap by scored day and statistics source (a trailing window keeps the corpus"
        " size roughly fixed while the day changes):",
        "",
    ]
    lines += _table(
        ["Day", *(f"`{w}`" for w in windows)],
        [
            [d, *(_ci(runs[f"{d}/{w}"]["gap"]) if f"{d}/{w}" in runs else "—" for w in windows)]
            for d in days
        ],
    )
    lines += [
        "",
        "### 5c. Returning vs new users",
        "",
        "Returning: the user has a train impression before the scored day. History is"
        " fixed per user in MIND, so this changes who is scored, not what the rankers see.",
        "",
    ]
    rows = []
    for d in days:
        r = day_runs[d]
        new = r["slices"]["new_user"]
        rows.append(
            [
                d,
                _num(new["n_impressions"] / r["n_impressions"], ".1%"),
                _ci(r["slices"]["returning_user"]["gap"]),
                _ci(new["gap"]),
                _ci(r["gap"]),
            ]
        )
    if ref:
        share = ref["share_users_new_to_train"]
        rows.append(
            [
                "holdout (ref.)",
                f"{_num(share, '.1%')} of users" if share is not None else "—",
                "—",
                "—",
                _ci(ref["gap"]),
            ]
        )
    lines += _table(["Day", "New-user impressions", "Gap, returning", "Gap, new", "Gap, all"], rows)

    v = results["verdicts"]
    lines += ["", "### 5d. How much of the swing each explanation can account for", ""]
    if not v.get("available"):
        lines.append("No holdout reference available, so the swing isn't defined.")
    else:
        lines += [
            f"Tune gap {v['tune_gap']:+.4f}; holdout gap {v['holdout_gap']:+.4f}; swing"
            f' {v["swing"]:+.4f}. Point estimates, rule from D-023: "sufficient alone" at'
            ' half the swing or more, "minor" under a quarter.',
            "",
        ]
        rows = []
        names = {
            "corpus_size": "Corpus size (largest move toward holdout, fixed targets)",
            "history_length_mix": "History-length mix (tune reweighted to holdout's buckets)",
            "returning_vs_new": "Returning vs new users (tune reweighted to holdout's new share)",
        }
        for key, name in names.items():
            if key in v:
                e = v[key]
                rows.append(
                    [
                        name,
                        f"{e['predicted_change']:+.4f}",
                        _num(e["fraction_of_swing"], ".0%"),
                        e["label"],
                    ]
                )
        lines += _table(["Explanation", "Predicted change", "Share of swing", "Verdict"], rows)
        lines += [
            "",
            "The new-user reweighting uses D-011's user-level share for holdout as an"
            " approximation of its impression-level share, and assumes holdout's new users"
            " behave like tune's. Both are untestable without reading dev.",
        ]
    return "\n".join(lines) + "\n"


def write_baseline_diagnostics(
    results: Mapping[str, Any], config: BaselineDiagnosticsConfig
) -> list[Path]:
    config.results.parent.mkdir(parents=True, exist_ok=True)
    config.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(render_baseline_diagnostics(results), encoding="utf-8")
    return [config.results, config.report]
