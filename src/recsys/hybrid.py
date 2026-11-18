"""E2: a leave-one-day-out logistic-regression hybrid of the baseline rankers (D-022, D-025).

Reads `MINDsmall_train` only; nothing here scores the holdout.

**Folds.** For each development day D, base rankers are fitted on train impressions
before D's midnight and score D's impressions: those scores are D's out-of-time
features, so D's labels never reach the rankers that produce them. Weights are
logistic-regression coefficients over candidate rows (clicked or not).
- *Leave one day out (primary):* the weights scored on day D are fitted on the
  other development days' rows only.
- *Forward chaining (secondary):* weights for day D are fitted on earlier days only.

**Features** (per candidate, from a label-free `RankRequest`):
- `pop_item`, `pop_category`: log1p of as-of clicks on the item and its category
- `affinity`: History share in the candidate's category + subcategory
- `tfidf`: cosine of TF-IDF vectors
- `tfidf_x_concentration`: `tfidf` x top-category share of the History (pre-specified
  in `reports/reversal_memo.md`, finding 6)
- optional groups, each kept only by the D-025 rule: 24h / 48h popularity, and
  `affinity` / `tfidf` on History augmented with the user's earlier clicks.

A per-user constant can't change a within-impression order, so concentration enters
only through the interaction. Users with no usable personalized feature fall back to
the popularity order and are flagged, as for the other personalized rankers.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from sklearn.linear_model import LogisticRegression

from recsys.evaluate import (
    EvalConfig,
    ModelRun,
    _git,
    _metric_block,
    _score_all,
    _segment,
    _with_ci,
    load_config,
    metric_label,
    summarize_runs,
)
from recsys.inventory import history_bucket, read_provenance
from recsys.loaders import TRAIN_DIR, load_mind_dir
from recsys.metrics import Eligibility, ImpressionMetrics, MetricSummary, evaluate, metric_names
from recsys.rankers import (
    CategoryAffinityRanker,
    HistoryAugmenter,
    PopularityRanker,
    Ranking,
    RankRequest,
    TfidfIndex,
    TfidfRanker,
    tfidf_corpus,
    top_category_share,
)
from recsys.schemas import Impression, NewsItem

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

REPRODUCE = "uv run python scripts/hybrid_development.py --config configs/hybrid.yaml"
BASE_FEATURES = ("pop_item", "pop_category", "affinity", "tfidf", "tfidf_x_concentration")
EXTRA_GROUPS: dict[str, tuple[str, ...]] = {
    "pop_24h": ("pop24_item", "pop24_category"),
    "pop_48h": ("pop48_item", "pop48_category"),
    "affinity_aug": ("affinity_aug",),
    "tfidf_aug": ("tfidf_aug",),
}
ALL_FEATURES = BASE_FEATURES + tuple(c for group in EXTRA_GROUPS.values() for c in group)
PERSONALIZED = frozenset(
    {"affinity", "tfidf", "tfidf_x_concentration", "affinity_aug", "tfidf_aug"}
)
PRIMARY = "hybrid"
NO_INTERACTION = "hybrid_no_interaction"
FULL = "hybrid_full"


# --- Features --------------------------------------------------------------------------------


class HybridFeatures:
    """Out-of-time feature rows for one statistics source (never the scored impressions)."""

    def __init__(
        self,
        stats: Sequence[Impression],
        news: Mapping[str, NewsItem],
        tfidf_params: Mapping[str, Any],
    ) -> None:
        self.popularity = PopularityRanker(stats, news)
        self._windows = {
            "pop24": PopularityRanker(stats, news, window_hours=24),
            "pop48": PopularityRanker(stats, news, window_hours=48),
        }
        index = TfidfIndex(news, tfidf_corpus(stats), **tfidf_params)
        augmenter = HistoryAugmenter(stats)
        self._affinity = CategoryAffinityRanker(self.popularity, news)
        self._affinity_aug = CategoryAffinityRanker(self.popularity, news, augmenter=augmenter)
        self._tfidf = TfidfRanker(self.popularity, index)
        self._tfidf_aug = TfidfRanker(self.popularity, index, augmenter=augmenter)

    def matrix(
        self, request: RankRequest, columns: Sequence[str] = ALL_FEATURES
    ) -> tuple[FloatArray, BoolArray]:
        """One row per candidate, and whether each column is available for this user.

        Unavailable personalized columns (no usable History) are zero.
        """
        need = set(columns)
        n = len(request.candidate_ids)
        values: dict[str, FloatArray | None] = {}
        if need & {"pop_item", "pop_category"}:
            item, category = self.popularity.keys(request)
            values["pop_item"], values["pop_category"] = np.log1p(item), np.log1p(category)
        for prefix, ranker in self._windows.items():
            if need & {f"{prefix}_item", f"{prefix}_category"}:
                item, category = ranker.keys(request)
                values[f"{prefix}_item"] = np.log1p(item)
                values[f"{prefix}_category"] = np.log1p(category)
        for name, affinity_ranker in (
            ("affinity", self._affinity),
            ("affinity_aug", self._affinity_aug),
        ):
            if name in need:
                counts = affinity_ranker.affinity(request)
                values[name] = (
                    None if counts is None else counts / len(affinity_ranker.profile(request))
                )
        if need & {"tfidf", "tfidf_x_concentration"}:
            similarity = self._tfidf.similarity(request)
            share = top_category_share(self._affinity.profile(request))
            values["tfidf"] = similarity
            values["tfidf_x_concentration"] = (
                None if similarity is None or share is None else similarity * share
            )
        if "tfidf_aug" in need:
            values["tfidf_aug"] = self._tfidf_aug.similarity(request)
        x = np.zeros((n, len(columns)))
        available = np.zeros(len(columns), dtype=bool)
        for j, column in enumerate(columns):
            value = values[column]
            if value is not None:
                x[:, j] = value
                available[j] = True
        return x, available


def is_fallback(columns: Sequence[str], available: BoolArray) -> bool:
    """True when no personalized column of this model is available for the user."""
    return not any(a for c, a in zip(columns, available, strict=True) if c in PERSONALIZED)


# --- Linear model and ranker -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearModel:
    """Logistic regression on standardized features; ranking uses the linear score."""

    columns: tuple[str, ...]
    mean: FloatArray
    scale: FloatArray
    coef: FloatArray
    intercept: float
    C: float
    n_rows: int

    def score(self, x: FloatArray) -> FloatArray:
        return np.asarray(((x - self.mean) / self.scale) @ self.coef, dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "C": self.C,
            "n_rows": self.n_rows,
            "intercept": self.intercept,
            "standardized_coef": dict(zip(self.columns, self.coef.tolist(), strict=True)),
            "mean": dict(zip(self.columns, self.mean.tolist(), strict=True)),
            "scale": dict(zip(self.columns, self.scale.tolist(), strict=True)),
        }


def fit_linear(x: FloatArray, y: npt.ArrayLike, columns: Sequence[str], C: float) -> LinearModel:
    labels = np.asarray(y, dtype=np.int64)
    if np.unique(labels).size < 2:
        raise ValueError("training rows need both clicked and unclicked candidates")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit((x - mean) / scale, labels)
    return LinearModel(
        tuple(columns),
        mean,
        scale,
        model.coef_[0].astype(np.float64),
        float(model.intercept_[0]),
        C,
        len(labels),
    )


class HybridRanker:
    """Scores a request with frozen weights over live out-of-time features."""

    def __init__(self, name: str, features: HybridFeatures, model: LinearModel) -> None:
        self.name = name
        self._features = features
        self._model = model

    def score(self, request: RankRequest) -> Ranking:
        x, available = self._features.matrix(request, self._model.columns)
        if is_fallback(self._model.columns, available):
            return Ranking(self._features.popularity.score(request).scores, fallback=True)
        return Ranking(self._model.score(x))


# --- Development days ------------------------------------------------------------------------


@dataclass(slots=True)
class DayData:
    """One development day: its impressions, out-of-time features (all columns) and labels."""

    day: date
    impressions: list[Impression]
    stats: list[Impression]
    features: HybridFeatures
    x: list[FloatArray]
    available: list[BoolArray]
    labels: list[npt.NDArray[np.int64]]
    popularity_scores: list[FloatArray]
    feature_seconds: float


def build_day(
    train: Sequence[Impression],
    day: date,
    news: Mapping[str, NewsItem],
    tfidf_params: Mapping[str, Any],
) -> DayData:
    start = datetime.combine(day, clock.min)
    stats = [i for i in train if i.timestamp < start]
    targets = [i for i in train if start <= i.timestamp < start + timedelta(1)]
    if not stats or not targets:
        raise ValueError(f"{day}: needs train impressions before and on that day")
    t0 = time.perf_counter()
    features = HybridFeatures(stats, news, tfidf_params)
    xs, available, labels, pop = [], [], [], []
    for imp in targets:
        request = RankRequest.from_impression(imp)
        x, a = features.matrix(request)
        xs.append(x)
        available.append(a)
        labels.append(np.array([int(c.clicked) for c in imp.candidates], dtype=np.int64))
        pop.append(features.popularity.score(request).scores)
    return DayData(
        day, targets, stats, features, xs, available, labels, pop, time.perf_counter() - t0
    )


def _columns_index(columns: Sequence[str]) -> list[int]:
    return [ALL_FEATURES.index(c) for c in columns]


def training_rows(
    days: Sequence[DayData], columns: Sequence[str]
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Candidate rows of every non-fallback impression on `days`, for these columns."""
    idx = _columns_index(columns)
    xs, ys = [], []
    for d in days:
        for x, a, y in zip(d.x, d.available, d.labels, strict=True):
            if not is_fallback(columns, a[idx]):
                xs.append(x[:, idx])
                ys.append(y)
    if not xs:
        raise ValueError("no non-fallback impressions to train on")
    return np.vstack(xs), np.concatenate(ys)


def cached_scores(day: DayData, model: LinearModel) -> list[FloatArray]:
    """The hybrid's scores on `day` from the cached features (same as `HybridRanker`)."""
    idx = _columns_index(model.columns)
    return [
        pop if is_fallback(model.columns, a[idx]) else model.score(x[:, idx])
        for x, a, pop in zip(day.x, day.available, day.popularity_scores, strict=True)
    ]


def cached_mean(
    day: DayData, model: LinearModel, metric: str, seed: int, ks: Sequence[int]
) -> float:
    mean = evaluate(day.impressions, cached_scores(day, model), seed=seed, ks=ks).means[metric]
    return float(mean) if mean is not None else float("nan")


def leave_one_day_out(
    days: Sequence[DayData], columns: Sequence[str], C: float
) -> dict[date, LinearModel]:
    """For each day, weights fitted on every other day's rows only."""
    return {
        held.day: fit_linear(
            *training_rows([d for d in days if d is not held], columns), columns, C
        )
        for held in days
    }


def forward_chaining(
    days: Sequence[DayData], columns: Sequence[str], C: float
) -> dict[date, LinearModel]:
    """For each day after the first, weights fitted on earlier days only."""
    return {
        days[i].day: fit_linear(*training_rows(days[:i], columns), columns, C)
        for i in range(1, len(days))
    }


# --- Rules (D-025) ---------------------------------------------------------------------------


def keep_extra(base: Sequence[float], with_extra: Sequence[float]) -> bool:
    """An optional group is kept iff it raises the held-out mean on all days but at most one."""
    better = sum(w > b for b, w in zip(base, with_extra, strict=True))
    return better >= len(base) - 1


def success(
    per_day: Mapping[str, Mapping[str, float]],
    pooled_delta: Mapping[str, Any] | None,
    model: str,
    rivals: Sequence[str],
) -> dict[str, Any]:
    """E2 succeeds iff `model` beats every rival on all days but at most one, and its
    pooled paired interval vs the first rival is above 0."""
    wins = [d for d, m in per_day.items() if all(m[model] > m[r] for r in rivals)]
    interval_ok = bool(
        pooled_delta and pooled_delta.get("ci_low") is not None and pooled_delta["ci_low"] > 0
    )
    days_ok = len(wins) >= len(per_day) - 1
    return {
        "days_beating_all_rivals": wins,
        "days_rule_met": days_ok,
        "pooled_interval_above_zero": interval_ok,
        "success": days_ok and interval_ok,
    }


# --- Pooling ---------------------------------------------------------------------------------


def _pooled_summary(rows: Sequence[ImpressionMetrics], ks: Sequence[int]) -> MetricSummary:
    names = metric_names(ks)
    collected: dict[str, list[float]] = {n: [] for n in names}
    for row in rows:
        for name, value in row.values.items():
            collected[name].append(value)
    return MetricSummary(
        means={n: float(np.mean(v)) if v else None for n, v in collected.items()},
        n_eligible={n: len(v) for n, v in collected.items()},
        n_impressions=len(rows),
        n_no_click=sum(r.eligibility is Eligibility.NO_CLICK for r in rows),
        n_all_click=sum(r.eligibility is Eligibility.ALL_CLICK for r in rows),
        n_tied=sum(r.tied for r in rows if r.eligibility is not Eligibility.NO_CLICK),
        per_impression=tuple(rows),
    )


def pool(
    day_runs: Sequence[tuple[Sequence[Impression], Sequence[ModelRun]]], config: EvalConfig
) -> dict[str, Any]:
    """Metrics over all held-out days together; the bootstrap resamples users across days."""
    impressions = [imp for imps, _ in day_runs for imp in imps]
    names = [r.name for r in day_runs[0][1]]
    runs = []
    for m, name in enumerate(names):
        parts = [rs[m] for _, rs in day_runs]
        runs.append(
            ModelRun(
                name,
                _pooled_summary([p for r in parts for p in r.summary.per_impression], config.ks),
                tuple(s for r in parts for s in r.scores),
                np.concatenate([r.fallback for r in parts]),
                np.concatenate([r.latencies_ms for r in parts]),
            )
        )
    users = sorted({imp.user_id for imp in impressions})
    pos = {u: i for i, u in enumerate(users)}
    user_index = np.array([pos[imp.user_id] for imp in impressions], dtype=np.int64)
    overall = _metric_block(runs, range(len(impressions)), user_index, config)
    buckets: dict[str, list[int]] = {
        history_bucket(e, config.history_edges): [] for e in config.history_edges
    }
    for i, imp in enumerate(impressions):
        buckets[history_bucket(len(imp.history), config.history_edges)].append(i)
    return {
        "n_impressions": len(impressions),
        "n_users": len(users),
        "models": {
            r.name: {"metrics": overall[r.name], "n_fallback": int(r.fallback.sum())} for r in runs
        },
        "segments_by_history_length": {
            b: _segment(runs, rows, impressions, user_index, config) for b, rows in buckets.items()
        },
    }


# --- Config and run --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HybridConfig:
    base: EvalConfig  # base rankers, seed, metrics, bootstrap; compare_to = the rivals
    days: tuple[date, ...]
    C: float
    sensitivity_C: tuple[float, ...]
    tfidf_model: str
    results: Path
    report: Path


def load_hybrid_config(path: Path) -> HybridConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw["hybrid"]
    return HybridConfig(
        base=load_config(path),
        days=tuple(d if isinstance(d, date) else date.fromisoformat(d) for d in section["days"]),
        C=float(section["C"]),
        sensitivity_C=tuple(float(c) for c in section.get("sensitivity_C", [])),
        tfidf_model=section.get("tfidf_model", "tfidf"),
        results=Path(section["outputs"]["results"]),
        report=Path(section["outputs"]["report"]),
    )


def run_hybrid_development(config: HybridConfig) -> dict[str, Any]:
    train = load_mind_dir(config.base.data_dir / TRAIN_DIR)  # dev is never loaded
    news = train.news
    base = dataclasses.replace(config.base, slice_category=None)
    metric = f"ndcg@{max(base.ks)}"
    spec = next(m for m in base.models if m.name == config.tfidf_model)
    tfidf_params = {k: v for k, v in spec.params.items() if k != "tie_break"}
    days = [build_day(train.impressions, d, news, tfidf_params) for d in sorted(config.days)]
    seed, ks = base.seed, base.ks

    def lodo_means(columns: Sequence[str], C: float) -> tuple[dict[date, LinearModel], list[float]]:
        models = leave_one_day_out(days, columns, C)
        return models, [cached_mean(d, models[d.day], metric, seed, ks) for d in days]

    primary_models, primary_means = lodo_means(BASE_FEATURES, config.C)
    no_int_cols = tuple(c for c in BASE_FEATURES if c != "tfidf_x_concentration")
    no_int_models, _ = lodo_means(no_int_cols, config.C)

    extras = {}
    kept: list[str] = []
    for group, columns in EXTRA_GROUPS.items():
        _, means = lodo_means(BASE_FEATURES + columns, config.C)
        keep = keep_extra(primary_means, means)
        extras[group] = {
            "columns": list(columns),
            "lodo_means": dict(zip((d.day.isoformat() for d in days), means, strict=True)),
            "days_better": sum(w > b for b, w in zip(primary_means, means, strict=True)),
            "kept": keep,
        }
        if keep:
            kept.append(group)
    full_cols = BASE_FEATURES + tuple(c for g in kept for c in EXTRA_GROUPS[g])
    full_models = lodo_means(full_cols, config.C)[0] if kept else None

    sensitivity = {
        str(C): dict(
            zip((d.day.isoformat() for d in days), lodo_means(BASE_FEATURES, C)[1], strict=True)
        )
        for C in config.sensitivity_C
    }
    forward = forward_chaining(days, BASE_FEATURES, config.C)
    forward_means = {
        d.day.isoformat(): cached_mean(d, forward[d.day], metric, seed, ks)
        for d in days
        if d.day in forward
    }

    per_day: dict[str, Any] = {}
    day_runs = []
    for d in days:
        hybrids = [
            HybridRanker(PRIMARY, d.features, primary_models[d.day]),
            HybridRanker(NO_INTERACTION, d.features, no_int_models[d.day]),
        ]
        if full_models is not None:
            hybrids.append(HybridRanker(FULL, d.features, full_models[d.day]))
        runs, fit_seconds = _score_all(d.impressions, d.stats, news, base, hybrids)
        result = summarize_runs(
            d.day.isoformat(),
            d.impressions,
            runs,
            d.stats,
            "train before the day",
            news,
            base,
            fit_seconds,
        )
        result["hybrid_feature_seconds"] = d.feature_seconds
        live = result["models"][PRIMARY]["metrics"][metric]["mean"]
        cached = primary_means[days.index(d)]
        result["live_matches_cached"] = bool(np.isclose(live, cached, rtol=0, atol=1e-12))
        per_day[d.day.isoformat()] = result
        day_runs.append((d.impressions, runs))

    pooled = pool(day_runs, base)
    rivals = list(base.compare_to)
    means_by_day = {
        day: {m: r["metrics"][metric]["mean"] for m, r in result["models"].items()}
        for day, result in per_day.items()
    }
    verdicts = {
        name: success(
            means_by_day,
            pooled["models"][name]["metrics"][metric]["delta_vs"].get(rivals[0]),
            name,
            rivals,
        )
        for name in (PRIMARY, NO_INTERACTION, *([FULL] if full_models else []))
    }
    final = {
        "primary": fit_linear(
            *training_rows(days, BASE_FEATURES), BASE_FEATURES, config.C
        ).to_json(),
    }
    if kept:
        final["full"] = fit_linear(*training_rows(days, full_cols), full_cols, config.C).to_json()

    status = _git("status", "--porcelain", "--", ".", ":!results", ":!reports")
    provenance = read_provenance(config.base.data_dir)
    return {
        "project": "recsys-lab",
        "kind": "hybrid_development",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(status) if status is not None else None,
        "reproduce": REPRODUCE,
        "data_provenance": [
            config.base.data_provenance if provenance == "unlabeled" else provenance
        ],
        "holdout_used": False,
        "config": {
            "seed": seed,
            "ks": list(ks),
            "metric": metric,
            "bootstrap_samples": base.bootstrap_samples,
            "history_edges": list(base.history_edges),
            "C": config.C,
            "rivals": rivals,
            "base_models": {m.name: {"ranker": m.kind, "params": m.params} for m in base.models},
            "features": {
                PRIMARY: list(BASE_FEATURES),
                NO_INTERACTION: list(no_int_cols),
                **({FULL: list(full_cols)} if kept else {}),
            },
        },
        "folds": [
            {
                "day": d.day.isoformat(),
                "statistics": f"train before {d.day.isoformat()}",
                "n_statistics_impressions": len(d.stats),
                "n_impressions": len(d.impressions),
                "lodo_weights_fitted_on": [o.day.isoformat() for o in days if o is not d],
                "forward_weights_fitted_on": [o.day.isoformat() for o in days if o.day < d.day],
                "lodo_primary_weights": primary_models[d.day].to_json(),
            }
            for d in days
        ],
        "extras": extras,
        "kept_extras": kept,
        "sensitivity_C": sensitivity,
        "forward_chaining_primary": forward_means,
        "per_day": per_day,
        "pooled": pooled,
        "verdicts": verdicts,
        "frozen_weights": final,
    }


# --- Report ----------------------------------------------------------------------------------


def _delta(entry: Mapping[str, Any], ref: str) -> str:
    d = entry.get("delta_vs", {}).get(ref)
    if d is None or d.get("ci_low") is None:
        return "—"
    return f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"


def _rows(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def render_hybrid(results: Mapping[str, Any]) -> str:
    config = results["config"]
    metric = config["metric"]
    label = metric_label(metric)
    rivals = config["rivals"]
    days = list(results["per_day"])
    models = list(next(iter(results["per_day"].values()))["models"])
    hybrids = [m for m in models if m.startswith("hybrid")]
    pooled = results["pooled"]
    lines = [
        "# E2 development results: leave-one-day-out hybrid",
        "",
        "> Generated by `scripts/hybrid_development.py`; do not edit by hand. Reads"
        " `MINDsmall_train` only: **no holdout data is loaded or scored.** Every number is"
        " development evidence (rolling-origin train days), not a holdout result. Offline"
        " ranking metrics, **not** engagement or CTR. Protocol: DECISIONS.md D-022, D-025.",
        "",
        f"- Data provenance: `{', '.join(results['data_provenance'])}`",
        f"- Generated: {results['generated_at']} at git `{results['git_sha']}`"
        + (" (uncommitted changes)" if results["git_dirty"] else ""),
        f"- Reproduce: `{results['reproduce']}`",
        f"- Seed {config['seed']}; {config['bootstrap_samples']} user-grouped bootstrap samples"
        " (users resampled across days when pooled); 95% percentile intervals.",
        f"- Logistic regression, L2, C = {config['C']} (fixed in advance), standardized features.",
        "",
        "## 1. Folds",
        "",
    ]
    lines += _rows(
        [
            "Held-out day",
            "Base rankers fitted on",
            "Scored impressions",
            "Weights fitted on (leave one day out)",
            "Weights fitted on (forward)",
        ],
        [
            [
                f["day"],
                f"{f['statistics']} ({f['n_statistics_impressions']:,} impressions)",
                f"{f['n_impressions']:,}",
                ", ".join(f["lodo_weights_fitted_on"]),
                ", ".join(f["forward_weights_fitted_on"]) or "—",
            ]
            for f in results["folds"]
        ],
    )
    feats = config["features"]
    lines += [
        "",
        "Features: "
        + "; ".join(f"`{m}` = {', '.join(f'`{c}`' for c in cols)}" for m, cols in feats.items())
        + ".",
    ]

    lines += ["", f"## 2. {label} by held-out day (leave one day out)", ""]
    rows = []
    for day in days:
        ms = results["per_day"][day]["models"]
        rows.append([day, *(f"{ms[m]['metrics'][metric]['mean']:.4f}" for m in models)])
    rows.append(["**pooled**", *(_with_ci(pooled["models"][m]["metrics"][metric]) for m in models)])
    lines += _rows(["Day", *(f"`{m}`" for m in models)], rows)
    lines += ["", "Paired Δ vs each rival [95% CI]:", ""]
    rows = []
    for name in hybrids:
        for ref in rivals:
            rows.append(
                [
                    f"`{name}` - `{ref}`",
                    *(
                        _delta(results["per_day"][d]["models"][name]["metrics"][metric], ref)
                        for d in days
                    ),
                    _delta(pooled["models"][name]["metrics"][metric], ref),
                ]
            )
    lines += _rows(["", *days, "pooled"], rows)

    lines += ["", "### Success rule (fixed before the run, D-025)", ""]
    for name, v in results["verdicts"].items():
        ref = pooled["models"][name]["metrics"][metric]["delta_vs"][rivals[0]]
        base_mean = pooled["models"][rivals[0]]["metrics"][metric]["mean"]
        lines.append(
            f"- `{name}`: beats {', '.join(f'`{r}`' for r in rivals)} on"
            f" {len(v['days_beating_all_rivals'])} of {len(days)} days"
            f" ({'met' if v['days_rule_met'] else 'not met'}); pooled interval vs `{rivals[0]}`"
            f" {'above 0' if v['pooled_interval_above_zero'] else 'not above 0'}"
            f" (relative {ref['mean'] / base_mean:+.1%})."
            f" **{'Success' if v['success'] else 'Not a success'}.**"
        )

    lines += [
        "",
        "## 3. Ablations and optional inputs",
        "",
        "Each optional group is added alone to `hybrid`; kept iff its leave-one-day-out"
        f" {label} is higher on all days but at most one (D-025).",
        "",
    ]
    primary_means = {
        d: results["per_day"][d]["models"]["hybrid"]["metrics"][metric]["mean"] for d in days
    }
    rows = [["`hybrid` (reference)", *(f"{primary_means[d]:.4f}" for d in days), "—", "—"]]
    for group, e in results["extras"].items():
        rows.append(
            [
                f"+ `{group}`",
                *(f"{e['lodo_means'][d]:.4f}" for d in days),
                f"{e['days_better']} of {len(days)}",
                "**kept**" if e["kept"] else "no",
            ]
        )
    lines += _rows(["Variant", *days, "Days better", "Kept"], rows)

    lines += ["", "## 4. Stability", "", "Standardized `hybrid` weights by fold:", ""]
    cols = config["features"]["hybrid"]
    lines += _rows(
        ["Held-out day", *(f"`{c}`" for c in cols), "Rows"],
        [
            [
                f["day"],
                *(f"{f['lodo_primary_weights']['standardized_coef'][c]:+.3f}" for c in cols),
                f"{f['lodo_primary_weights']['n_rows']:,}",
            ]
            for f in results["folds"]
        ],
    )
    lines += [
        "",
        f"`hybrid` {label} with other C (leave one day out; reported, not used to choose):",
        "",
    ]
    sens = results["sensitivity_C"]
    lines += _rows(
        ["C", *days],
        [[str(config["C"]), *(f"{primary_means[d]:.4f}" for d in days)]]
        + [[c, *(f"{v[d]:.4f}" for d in days)] for c, v in sens.items()],
    )
    fwd = results["forward_chaining_primary"]
    lines += ["", f"Forward chaining (weights from earlier days only), {label}:", ""]
    lines += _rows(
        ["Day", "`hybrid` (forward)", "`hybrid` (LODO)", *(f"`{r}`" for r in rivals)],
        [
            [
                d,
                f"{fwd[d]:.4f}",
                f"{primary_means[d]:.4f}",
                *(
                    f"{results['per_day'][d]['models'][r]['metrics'][metric]['mean']:.4f}"
                    for r in rivals
                ),
            ]
            for d in fwd
        ],
    )

    lines += ["", f"## 5. Pooled {label} by History length", ""]
    segs = pooled["segments_by_history_length"]
    show = ["popularity", *rivals, *hybrids]
    lines += _rows(
        ["History", "Impressions", *(f"`{m}`" for m in show), f"`hybrid` - `{rivals[0]}`"],
        [
            [
                b,
                f"{s['n_impressions']:,}",
                *(f"{s['models'][m]['metrics'][metric]['mean']:.4f}" for m in show),
                _delta(s["models"]["hybrid"]["metrics"][metric], rivals[0]),
            ]
            for b, s in segs.items()
        ],
    )

    lines += ["", "## 6. Costs (per day: p50 / p95 scoring ms, fallbacks, coverage@10)", ""]
    rows = []
    for m in models:
        cells = []
        for d in days:
            r = results["per_day"][d]["models"][m]
            cells.append(f"{r['latency_ms']['p50']:.2f} / {r['latency_ms']['p95']:.2f}")
        last = results["per_day"][days[-1]]["models"][m]
        rows.append(
            [
                f"`{m}`",
                *cells,
                f"{pooled['models'][m]['n_fallback']:,}",
                f"{last['coverage@10']:.1%}",
            ]
        )
    lines += _rows(["Model", *days, "Fallbacks (pooled)", f"coverage@10 ({days[-1]})"], rows)
    secs = ", ".join(f"{d}: {results['per_day'][d]['hybrid_feature_seconds']:.0f}s" for d in days)
    lines += [
        "",
        "Hybrid latency includes computing every feature live (three popularity indexes,"
        " two affinity and two TF-IDF passes for `hybrid_full`). Feature build for all"
        f" columns per day: {secs}.",
        "",
        "Live scoring matches the cached features used for fitting on every day: "
        + (
            "yes." if all(results["per_day"][d]["live_matches_cached"] for d in days) else "**no**."
        ),
    ]
    return "\n".join(lines) + "\n"


def write_hybrid(results: Mapping[str, Any], config: HybridConfig) -> list[Path]:
    config.results.parent.mkdir(parents=True, exist_ok=True)
    config.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(render_hybrid(results), encoding="utf-8")
    return [config.results, config.report]
