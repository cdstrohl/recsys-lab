"""Common evaluation runner: every model on the same impressions, candidates and seed.

Pipeline (see DECISIONS.md D-012 to D-017):
1. Load train and dev, assign fit / tune / holdout (D-005).
2. `tune` is scored with rankers fitted on `fit`. `holdout` is scored with rankers
   refitted on `fit` + `tune` (the documented final refit) and only after
   `eval/manifest.json` matches the data files.
3. Rankers see label-free `RankRequest`s. Labels are used only by the metrics.
4. Metrics are averaged per impression (D-008 to D-010), overall and by
   history-length bucket, with user-grouped bootstrap intervals and paired
   differences against the baseline model.
5. Outputs: `results/experiments.json` (everything), `results/summary.json` (the
   portfolio export, holdout only) and a Markdown report. All are generated here,
   never by hand, and carry aggregate numbers only (no user or impression IDs).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from recsys.inventory import history_bucket, read_provenance, unseen_inventory
from recsys.loaders import BEHAVIORS_FILE, DEV_DIR, NEWS_FILE, TRAIN_DIR, load_mind_dir
from recsys.metrics import MetricSummary, evaluate, metric_names, rank_order, tie_break_keys
from recsys.rankers import RankRequest, build_rankers
from recsys.schemas import Impression, NewsItem
from recsys.splits import Split, SplitAssignment, assign_splits

PROJECT = "recsys-lab"
MANIFEST_VERSION = "v1"
REPRODUCE = "uv run python scripts/evaluate.py --config configs/baseline.yaml"
DATA_FILES = tuple(f"{d}/{f}" for d in (TRAIN_DIR, DEV_DIR) for f in (NEWS_FILE, BEHAVIORS_FILE))
CI_LEVEL = 0.95

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

LIMITATIONS = (
    "Offline ranking metrics on logged MIND-small impressions only; they are not engagement,"
    " CTR or business-impact estimates.",
    "Labels come from impressions served by Microsoft News's own 2019 system, so they reflect"
    " that system's exposure; offline metrics can't predict online behavior.",
    "General news from November 2019; results don't show sports-specific behavior.",
    "MIND-small numbers aren't comparable to the public MIND leaderboard (scored on MIND-large"
    " test).",
    "Holdout statistics come from fit + tune (train) only; dev clicks are never features, so"
    " popularity is up to a day stale and items first shown in dev have no click history.",
    "Ties are broken by one seeded draw (D-008); MRR and nDCG of tie-heavy models can move"
    " with the seed. AUC counts ties as 0.5.",
    "Bootstrap intervals resample users and treat the logged candidate sets as fixed.",
)


class ManifestError(RuntimeError):
    """The evaluation manifest is missing or doesn't match the data."""


# --- Config ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    data_dir: Path
    models: tuple[ModelSpec, ...]
    data_provenance: str = "real_licensed"
    seed: int = 13
    splits: tuple[Split, ...] = (Split.TUNE, Split.HOLDOUT)
    holdout_include_tune: bool = True
    ks: tuple[int, ...] = (5, 10)
    history_edges: tuple[int, ...] = (0, 1, 6)
    baseline: str = "popularity"
    bootstrap_samples: int = 1000
    coverage_k: int = 10
    results_dir: Path = Path("results")
    report: Path = Path("reports/baseline_results.md")
    manifest: Path = Path("eval/manifest.json")

    def __post_init__(self) -> None:
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate model names in {names}")
        if self.baseline not in names:
            raise ValueError(f"baseline {self.baseline!r} must be one of the models {names}")
        if Split.FIT in self.splits:
            raise ValueError("fit is the statistics source and can't be an evaluation split")


def load_config(path: Path) -> EvalConfig:
    """Read a YAML config. Relative paths resolve against the current directory."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    outputs = raw.get("outputs", {})
    statistics = raw.get("holdout_statistics", "fit+tune")
    if statistics not in ("fit", "fit+tune"):
        raise ValueError(f"holdout_statistics must be 'fit' or 'fit+tune', got {statistics!r}")
    return EvalConfig(
        data_dir=Path(raw["data_dir"]),
        models=tuple(ModelSpec(m["name"], dict(m.get("params") or {})) for m in raw["models"]),
        data_provenance=raw.get("data_provenance", "real_licensed"),
        seed=int(raw.get("seed", 13)),
        splits=tuple(Split(s) for s in raw.get("splits", ["tune", "holdout"])),
        holdout_include_tune=statistics == "fit+tune",
        ks=tuple(raw.get("ks", [5, 10])),
        history_edges=tuple(raw.get("history_edges", [0, 1, 6])),
        baseline=raw.get("baseline", "popularity"),
        bootstrap_samples=int(raw.get("bootstrap_samples", 1000)),
        coverage_k=int(raw.get("coverage_k", 10)),
        results_dir=Path(outputs.get("results_dir", "results")),
        report=Path(outputs.get("report", "reports/baseline_results.md")),
        manifest=Path(outputs.get("manifest", "eval/manifest.json")),
    )


# --- Manifest --------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(data_dir: Path, assignment: SplitAssignment, provenance: str) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "data_provenance": provenance,
        "files": {name: sha256_file(data_dir / name) for name in DATA_FILES},
        "split_rule": {
            "fit": f"{TRAIN_DIR} impressions with timestamp < cutoff",
            "tune": f"{TRAIN_DIR} impressions with timestamp >= cutoff",
            "holdout": f"all {DEV_DIR} impressions (locked)",
            "cutoff": assignment.cutoff.isoformat(),
            "cutoff_rule": "midnight starting the calendar date of the latest train impression",
        },
        "row_counts": assignment.counts(),
        "label_version": "MIND-small original click labels, unmodified",
        "eligibility": "no-click excluded from all metrics; all-click excluded from AUC (D-010)",
        "tie_break": "seeded blake2b(seed, impression_id, news_id) (D-008)",
    }


def lock_manifest(
    path: Path, data_dir: Path, assignment: SplitAssignment, provenance: str
) -> dict[str, Any]:
    """Write the manifest once. Changing a locked manifest is a deliberate, manual act."""
    if path.exists():
        raise ManifestError(f"{path} already exists; delete it deliberately to re-lock")
    manifest = build_manifest(data_dir, assignment, provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(path: Path, data_dir: Path, assignment: SplitAssignment) -> str:
    """Check hashes, cutoff and row counts against the locked manifest; return its version."""
    if not path.exists():
        raise ManifestError(
            f"{path} not found; lock it (scripts/evaluate.py --lock-manifest) and commit it"
            " before any holdout run"
        )
    locked = json.loads(path.read_text(encoding="utf-8"))
    problems = [
        f"{name}: hash differs"
        for name, digest in locked["files"].items()
        if sha256_file(data_dir / name) != digest
    ]
    if locked["split_rule"]["cutoff"] != assignment.cutoff.isoformat():
        problems.append("cutoff differs")
    if locked["row_counts"] != assignment.counts():
        problems.append(f"row counts differ: {assignment.counts()} vs {locked['row_counts']}")
    if problems:
        raise ManifestError(f"{path} doesn't match the data: {'; '.join(problems)}")
    return str(locked["manifest_version"])


# --- Bootstrap -------------------------------------------------------------------------------


def user_bootstrap(
    user_index: IntArray,
    values: FloatArray,
    *,
    n_samples: int,
    seed: int,
    chunk: int = 100,
) -> FloatArray:
    """Bootstrap means that resample users, not impressions.

    `values` is (models, impressions); `user_index` maps each impression to a user in
    0..U-1. Each sample draws U users with replacement and averages over all of their
    impressions. Returns (n_samples, models). The same seed gives the same user draws,
    so differences between models are paired.
    """
    n_users = int(user_index.max()) + 1 if user_index.size else 0
    if n_users == 0 or n_samples < 1:
        return np.empty((0, values.shape[0]))
    sums = np.stack([np.bincount(user_index, weights=v, minlength=n_users) for v in values])
    counts = np.bincount(user_index, minlength=n_users).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_samples, chunk):
        size = min(chunk, n_samples - start)
        draws = rng.integers(0, n_users, size=(size, n_users))
        offsets = (draws + np.arange(size)[:, None] * n_users).ravel()
        weights = np.bincount(offsets, minlength=size * n_users).reshape(size, n_users)
        weights_f = weights.astype(np.float64)
        out.append((weights_f @ sums.T) / (weights_f @ counts)[:, None])
    return np.concatenate(out)


def _interval(samples: FloatArray) -> tuple[float, float] | tuple[None, None]:
    if samples.size == 0:
        return None, None
    alpha = (1 - CI_LEVEL) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return float(low), float(high)


# --- Scoring ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelRun:
    name: str
    summary: MetricSummary
    scores: tuple[FloatArray, ...]
    fallback: npt.NDArray[np.bool_]  # per impression
    latencies_ms: FloatArray


def _score_all(
    impressions: Sequence[Impression],
    stats: Sequence[Impression],
    news: Mapping[str, NewsItem],
    config: EvalConfig,
) -> tuple[list[ModelRun], float]:
    """Fit every model on `stats`, score label-free requests, then compute metrics."""
    requests = [RankRequest.from_impression(imp) for imp in impressions]
    start = time.perf_counter()
    specs = [(m.name, m.params) for m in config.models]
    rankers = build_rankers(specs, stats, news, seed=config.seed)
    fit_seconds = time.perf_counter() - start
    runs = []
    for ranker in rankers:
        scores, fallback, latencies = [], [], []
        for request in requests:
            t0 = time.perf_counter()
            ranking = ranker.score(request)
            latencies.append((time.perf_counter() - t0) * 1000)
            scores.append(ranking.scores)
            fallback.append(ranking.fallback)
        summary = evaluate(impressions, scores, seed=config.seed, ks=config.ks)
        runs.append(
            ModelRun(
                ranker.name,
                summary,
                tuple(scores),
                np.asarray(fallback, dtype=bool),
                np.asarray(latencies),
            )
        )
    return runs, fit_seconds


def _coverage(
    impressions: Sequence[Impression], scores: Sequence[FloatArray], seed: int, k: int
) -> float:
    """Share of distinct candidate items placed in the top k of at least one impression."""
    shown: set[str] = set()
    for imp, s in zip(impressions, scores, strict=True):
        ids = [c.news_id for c in imp.candidates]
        top = rank_order(s, tie_break_keys(imp.impression_id, ids, seed))[:k]
        shown.update(ids[i] for i in top)
    total = {c.news_id for imp in impressions for c in imp.candidates}
    return len(shown) / len(total) if total else 0.0


def _metric_block(
    runs: Sequence[ModelRun], rows: Sequence[int], user_index: IntArray, config: EvalConfig
) -> dict[str, dict[str, Any]]:
    """Mean, bootstrap CI and paired difference vs the baseline, per model and metric.

    `rows` selects impressions (e.g. one history bucket). Eligibility depends on labels
    only, so for a given metric every model is averaged over the same impressions.
    """
    base = [r.name for r in runs].index(config.baseline)
    block: dict[str, dict[str, Any]] = {r.name: {} for r in runs}
    for metric in metric_names(config.ks):
        eligible = [i for i in rows if metric in runs[0].summary.per_impression[i].values]
        values = np.array(
            [[r.summary.per_impression[i].values[metric] for i in eligible] for r in runs]
        ).reshape(len(runs), len(eligible))
        _, dense_users = np.unique(user_index[eligible], return_inverse=True)
        boot = user_bootstrap(
            dense_users.astype(np.int64),
            values,
            n_samples=config.bootstrap_samples,
            seed=config.seed,
        )
        for m, run in enumerate(runs):
            low, high = _interval(boot[:, m])
            entry: dict[str, Any] = {
                "mean": float(values[m].mean()) if eligible else None,
                "ci_low": low,
                "ci_high": high,
                "n_eligible": len(eligible),
            }
            if m != base and eligible:
                d_low, d_high = _interval(boot[:, m] - boot[:, base])
                entry["delta_vs_baseline"] = {
                    "mean": float(values[m].mean() - values[base].mean()),
                    "ci_low": d_low,
                    "ci_high": d_high,
                }
            block[run.name][metric] = entry
    return block


def evaluate_split(
    split: Split,
    impressions: Sequence[Impression],
    stats: Sequence[Impression],
    stats_label: str,
    news: Mapping[str, NewsItem],
    config: EvalConfig,
) -> dict[str, Any]:
    """Every model on one split: overall metrics, segments by history length, costs."""
    runs, fit_seconds = _score_all(impressions, stats, news, config)
    users = sorted({imp.user_id for imp in impressions})
    user_pos = {u: i for i, u in enumerate(users)}
    user_index = np.array([user_pos[imp.user_id] for imp in impressions], dtype=np.int64)
    first = runs[0].summary

    overall = _metric_block(runs, range(len(impressions)), user_index, config)
    models = {
        run.name: {
            "metrics": overall[run.name],
            "n_tied": run.summary.n_tied,
            "n_fallback": int(run.fallback.sum()),
            f"coverage@{config.coverage_k}": _coverage(
                impressions, run.scores, config.seed, config.coverage_k
            ),
            "latency_ms": {
                "p50": float(np.percentile(run.latencies_ms, 50)),
                "p95": float(np.percentile(run.latencies_ms, 95)),
            },
        }
        for run in runs
    }

    edges = config.history_edges
    buckets: dict[str, list[int]] = {history_bucket(e, edges): [] for e in edges}
    for i, imp in enumerate(impressions):
        buckets[history_bucket(len(imp.history), edges)].append(i)
    segments = {}
    for label, rows in buckets.items():
        block = _metric_block(runs, rows, user_index, config)
        segments[label] = {
            "n_impressions": len(rows),
            "n_users": len({impressions[i].user_id for i in rows}),
            "models": {
                run.name: {
                    "metrics": block[run.name],
                    "n_fallback": int(run.fallback[rows].sum()),
                }
                for run in runs
            },
        }

    return {
        "split": split.value,
        "statistics_source": stats_label,
        "n_statistics_impressions": len(stats),
        "n_impressions": first.n_impressions,
        "n_users": len(users),
        "n_no_click": first.n_no_click,
        "n_all_click": first.n_all_click,
        "share_candidate_occurrences_unseen_in_statistics": unseen_inventory(stats, impressions)[
            "share_occurrences_unseen_as_candidate"
        ],
        "fit_seconds_all_models": fit_seconds,
        "models": models,
        "segments_by_history_length": segments,
    }


# --- Run -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedData:
    assignment: SplitAssignment
    news: dict[str, NewsItem]
    provenance: str


def load_data(config: EvalConfig) -> LoadedData:
    train = load_mind_dir(config.data_dir / TRAIN_DIR)
    dev = load_mind_dir(config.data_dir / DEV_DIR)
    # Item metadata (category, title, abstract) is content, not labels (D-013).
    # Train's row wins if an ID appears in both files.
    news = {**dev.news, **train.news}
    provenance = read_provenance(config.data_dir)
    if provenance == "unlabeled":
        provenance = config.data_provenance
    return LoadedData(assign_splits(train.impressions, dev.impressions), news, provenance)


def lock_manifest_from_config(config: EvalConfig) -> dict[str, Any]:
    data = load_data(config)
    return lock_manifest(config.manifest, config.data_dir, data.assignment, data.provenance)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def run(config: EvalConfig) -> dict[str, Any]:
    """Evaluate every configured model on every configured split. Writes nothing."""
    data = load_data(config)
    manifest_version = None
    if Split.HOLDOUT in config.splits:
        manifest_version = verify_manifest(config.manifest, config.data_dir, data.assignment)

    splits = {}
    for split in config.splits:
        if split is Split.TUNE:
            stats, label = data.assignment.statistics_source(), "fit"
            targets = data.assignment.tune
        else:
            stats = data.assignment.statistics_source(include_tune=config.holdout_include_tune)
            label = "fit+tune" if config.holdout_include_tune else "fit"
            targets = data.assignment.holdout(allow_holdout=True)
        splits[split.value] = evaluate_split(split, targets, stats, label, data.news, config)

    # Generated outputs don't make the code dirty.
    status = _git("status", "--porcelain", "--", ".", ":!results", ":!reports")
    return {
        "project": PROJECT,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(status) if status is not None else None,
        "manifest_version": manifest_version,
        "reproduce": REPRODUCE,
        "data_provenance": [data.provenance],
        "config": {
            "seed": config.seed,
            "ks": list(config.ks),
            "history_edges": list(config.history_edges),
            "baseline": config.baseline,
            "bootstrap_samples": config.bootstrap_samples,
            "ci_level": CI_LEVEL,
            "coverage_k": config.coverage_k,
            "holdout_statistics": "fit+tune" if config.holdout_include_tune else "fit",
            "cutoff": data.assignment.cutoff.isoformat(),
            "models": [{"name": m.name, "params": m.params} for m in config.models],
        },
        "splits": splits,
        "limitations": list(LIMITATIONS),
    }


# --- Outputs ---------------------------------------------------------------------------------

METRIC_LABELS = {"auc": "AUC", "mrr": "MRR (MIND-style)"}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("ndcg", "NDCG"))


def build_summary(experiments: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `07` §9 portfolio export: holdout metrics only, or None without a holdout run."""
    holdout = experiments["splits"].get(Split.HOLDOUT.value)
    if holdout is None:
        return None
    baseline = experiments["config"]["baseline"]
    ks = experiments["config"]["ks"]
    order = [f"ndcg@{k}" for k in sorted(ks, reverse=True)] + ["mrr", "auc"]
    metrics = []
    for metric in order:
        base = holdout["models"][baseline]["metrics"][metric]
        for model, result in holdout["models"].items():
            entry = result["metrics"][metric]
            metrics.append(
                {
                    "name": metric_label(metric),
                    "model": model,
                    "value": entry["mean"],
                    "ci95": [entry["ci_low"], entry["ci_high"]],
                    "split": Split.HOLDOUT.value,
                    "n": entry["n_eligible"],
                    "baseline": baseline,
                    "baseline_value": base["mean"],
                }
            )
    return {
        "project": experiments["project"],
        "generated_at": experiments["generated_at"],
        "git_sha": experiments["git_sha"],
        "manifest_version": experiments["manifest_version"],
        "reproduce": experiments["reproduce"],
        "data_provenance": experiments["data_provenance"],
        "metrics": metrics,
        "figures": [],
        "limitations": experiments["limitations"],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _with_ci(entry: Mapping[str, Any]) -> str:
    if entry["ci_low"] is None:
        return _fmt(entry["mean"])
    return f"{_fmt(entry['mean'])} [{_fmt(entry['ci_low'])}, {_fmt(entry['ci_high'])}]"


def _delta(entry: Mapping[str, Any]) -> str:
    delta = entry.get("delta_vs_baseline")
    if delta is None:
        return "—"
    text = f"{delta['mean']:+.4f}"
    if delta["ci_low"] is not None:
        text += f" [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
    return text


def _excludes_zero(delta: Mapping[str, Any]) -> bool:
    low, high = delta["ci_low"], delta["ci_high"]
    return low is not None and (low > 0 or high < 0)


METHODS = (
    "- `random`: seeded random scores. Sanity floor.",
    "- `popularity`: as-of click count of the item in the statistics source, ties broken by"
    " as-of clicks on the item's category. Non-personalized.",
    "- `category_affinity`: (# History items in the candidate's category + # in its"
    " subcategory), ties broken by the popularity order. Users with no known History item"
    " fall back to `popularity`.",
    "- `tfidf`: cosine between the candidate's title+abstract TF-IDF vector and the mean"
    " History vector; vocabulary and IDF fitted on items referenced by the statistics source."
    " Ties broken by the popularity order; users with no usable History fall back to"
    " `popularity`.",
)


def render_report(experiments: Mapping[str, Any]) -> str:
    """Human-readable report, rendered only from `experiments` (no hand-entered numbers)."""
    config = experiments["config"]
    baseline, ks = config["baseline"], config["ks"]
    headline = f"ndcg@{max(ks)}"
    cov_key = f"coverage@{config['coverage_k']}"
    lines = [
        "# Baseline results",
        "",
        "> Generated by `scripts/evaluate.py`; do not edit by hand. Offline ranking metrics on"
        " logged impressions only, **not** engagement, CTR or business impact.",
        "",
        f"- Data provenance: `{', '.join(experiments['data_provenance'])}`",
        f"- Generated: {experiments['generated_at']} at git `{experiments['git_sha']}`"
        + (" (uncommitted changes)" if experiments["git_dirty"] else ""),
        f"- Manifest: `{experiments['manifest_version'] or 'not checked (no holdout run)'}`",
        f"- Reproduce: `{experiments['reproduce']}`",
        f"- Seed {config['seed']}; {config['bootstrap_samples']} user-grouped bootstrap samples;"
        f" {int(config['ci_level'] * 100)}% percentile intervals; fit/tune cutoff"
        f" {config['cutoff']}.",
        "",
        "## Methods",
        "",
        *METHODS,
        "",
        "MRR is MIND-style: the mean of 1/rank over all clicked candidates (D-009). Each"
        " metric is averaged per impression over the impressions eligible for it (D-010)."
        f" Deltas are paired differences against `{baseline}` on the same impressions, with"
        " intervals from the same user resamples.",
    ]
    for name, split in experiments["splits"].items():
        lines += [
            "",
            f"## Split: `{name}`",
            "",
            f"- {split['n_impressions']:,} impressions from {split['n_users']:,} users;"
            f" {split['n_no_click']} no-click (excluded), {split['n_all_click']} all-click"
            " (excluded from AUC).",
            f"- Models fitted on `{split['statistics_source']}`"
            f" ({split['n_statistics_impressions']:,} impressions);"
            f" {split['share_candidate_occurrences_unseen_in_statistics']:.1%} of candidate"
            " occurrences were never shown there.",
            "",
            f"| Model | {metric_label(headline)} [95% CI] | Δ vs {baseline} [95% CI] | "
            + " | ".join(metric_label(m) for m in metric_names(ks) if m != headline)
            + f" | Fallback | Tied | {cov_key.capitalize()} | p50 / p95 ms |",
            "|---|---|---|" + "---|" * (len(metric_names(ks)) - 1) + "---|---|---|---|",
        ]
        for model, result in split["models"].items():
            metrics = result["metrics"]
            others = [_fmt(metrics[m]["mean"]) for m in metric_names(ks) if m != headline]
            lat = result["latency_ms"]
            lines.append(
                f"| `{model}` | {_with_ci(metrics[headline])} | {_delta(metrics[headline])} | "
                + " | ".join(others)
                + f" | {result['n_fallback']:,} | {result['n_tied']:,} |"
                f" {result[cov_key]:.3f} | {lat['p50']:.3f} / {lat['p95']:.3f} |"
            )
        n_eligible = split["models"][baseline]["metrics"][headline]["n_eligible"]
        lines += [
            "",
            f"Eligible impressions for {metric_label(headline)}: {n_eligible:,}.",
            "",
            f"### {metric_label(headline)} by user history length",
            "",
            "| History length | Impressions | Users | "
            + " | ".join(f"`{m}`" for m in split["models"])
            + f" | Best Δ vs {baseline} [95% CI] |",
            "|---|---|---|" + "---|" * len(split["models"]) + "---|",
        ]
        for label, segment in split["segments_by_history_length"].items():
            seg_models = segment["models"]
            cells = [_with_ci(seg_models[m]["metrics"][headline]) for m in seg_models]
            deltas = {
                m: r["metrics"][headline]["delta_vs_baseline"]
                for m, r in seg_models.items()
                if "delta_vs_baseline" in r["metrics"][headline]
            }
            best = max(deltas, key=lambda m: deltas[m]["mean"]) if deltas else None
            best_text = f"`{best}` {_delta(seg_models[best]['metrics'][headline])}" if best else "—"
            lines.append(
                f"| {label} | {segment['n_impressions']:,} | {segment['n_users']:,} | "
                + " | ".join(cells)
                + f" | {best_text} |"
            )
        lines += ["", "### Observations (computed from the tables above)", ""]
        ranked = sorted(
            split["models"].items(),
            key=lambda item: item[1]["metrics"][headline]["mean"] or 0.0,
            reverse=True,
        )
        top, top_result = ranked[0]
        lines.append(
            f"- Highest {metric_label(headline)}: `{top}`"
            f" ({_fmt(top_result['metrics'][headline]['mean'])})."
        )
        for model, result in ranked:
            delta = result["metrics"][headline].get("delta_vs_baseline")
            if delta is None or delta["ci_low"] is None:
                continue
            verdict = "excludes 0" if _excludes_zero(delta) else "includes 0"
            lines.append(
                f"- `{model}` vs `{baseline}`: Δ {_delta(result['metrics'][headline])};"
                f" the interval {verdict}."
            )
        cold = split["segments_by_history_length"].get("0")
        if cold and cold["n_impressions"]:
            fell_back = [
                m for m, r in cold["models"].items() if r["n_fallback"] == cold["n_impressions"]
            ]
            if fell_back:
                lines.append(
                    f"- Zero-history users ({cold['n_impressions']:,} impressions):"
                    f" {', '.join(f'`{m}`' for m in fell_back)} fall back to `{baseline}` on every"
                    " impression, so they equal it there by construction."
                )
    lines += ["", "## Limitations", "", *(f"- {item}" for item in experiments["limitations"])]
    return "\n".join(lines) + "\n"


def write_outputs(experiments: Mapping[str, Any], config: EvalConfig) -> list[Path]:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    written = [config.results_dir / "experiments.json"]
    written[0].write_text(json.dumps(experiments, indent=2) + "\n", encoding="utf-8")
    summary = build_summary(experiments)
    if summary is not None:
        path = config.results_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(render_report(experiments), encoding="utf-8")
    written.append(config.report)
    return written
