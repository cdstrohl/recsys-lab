"""Tune-only diagnostics (Phase 2, D-020). Reads `MINDsmall_train` only, never dev.

1. **Stability (rolling origin).** Each listed train day D is scored with rankers fitted
   on every train impression before D's midnight. The last train day with this rule
   is exactly the official tune protocol (`fit` -> `tune`).
2. **Ablations.** The official tune protocol with model variants: recent-window
   popularity, History augmented with earlier own clicks, and random tie-breaks.

Both add slices by the share of an impression's candidates with no click in the
statistics source, which is where the popularity tie-break carries no information.
An ablation is flagged as a Phase 3 candidate only by the rule fixed in D-020.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import yaml

from recsys.evaluate import (
    EvalConfig,
    _git,
    _segment_tables,
    _with_ci,
    evaluate_split,
    load_config,
    metric_label,
)
from recsys.inventory import read_provenance
from recsys.loaders import TRAIN_DIR, load_mind_dir
from recsys.schemas import Impression
from recsys.splits import AsOfClickIndex, assign_splits

UNSEEN_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
REPRODUCE = "uv run python scripts/diagnose.py --config configs/diagnostics.yaml"


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    base: EvalConfig  # every model definition, seed, metrics, bootstrap
    stability_days: tuple[date, ...]
    stability_models: tuple[str, ...]
    stability_compare_to: tuple[str, ...]
    ablation_models: tuple[str, ...]
    ablation_compare_to: tuple[str, ...]
    results: Path
    report: Path


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def load_diagnostics_config(path: Path) -> DiagnosticsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw["diagnostics"]
    return DiagnosticsConfig(
        base=load_config(path),
        stability_days=tuple(_as_date(d) for d in section["stability"]["days"]),
        stability_models=tuple(section["stability"]["models"]),
        stability_compare_to=tuple(section["stability"]["compare_to"]),
        ablation_models=tuple(section["ablations"]["models"]),
        ablation_compare_to=tuple(section["ablations"]["compare_to"]),
        results=Path(section["outputs"]["results"]),
        report=Path(section["outputs"]["report"]),
    )


def _subset(config: EvalConfig, names: Sequence[str], compare_to: Sequence[str]) -> EvalConfig:
    by_name = {m.name: m for m in config.models}
    if missing := [n for n in names if n not in by_name]:
        raise ValueError(f"models {missing} aren't defined under `models`")
    return dataclasses.replace(
        config, models=tuple(by_name[n] for n in names), compare_to=tuple(compare_to)
    )


def unseen_slices(
    impressions: Sequence[Impression], stats: Sequence[Impression]
) -> tuple[dict[str, tuple[str, list[int]]], float]:
    """Bucket impressions by the share of candidates never clicked in `stats` before them.

    Also returns the share of all candidate occurrences that are unclicked.
    """
    index = AsOfClickIndex.from_impressions(stats)
    slices: dict[str, tuple[str, list[int]]] = {}
    for low, high in UNSEEN_BINS:
        upper = "]" if high == 1.0 else ")"
        slices[f"unclicked_{int(low * 100)}_{int(high * 100)}"] = (
            f"[{low:.0%}, {high:.0%}{upper} of candidates unclicked in statistics",
            [],
        )
    names = list(slices)
    unclicked = total = 0
    for i, imp in enumerate(impressions):
        ids = [c.news_id for c in imp.candidates]
        n_unclicked = sum(index.count_before(n, imp.timestamp) == 0 for n in ids)
        unclicked += n_unclicked
        total += len(ids)
        share = n_unclicked / len(ids)
        bucket = next(b for b, (_, high) in enumerate(UNSEEN_BINS) if share < high or high == 1.0)
        slices[names[bucket]][1].append(i)
    return slices, (unclicked / total if total else 0.0)


def run_diagnostics(config: DiagnosticsConfig) -> dict[str, Any]:
    train = load_mind_dir(config.base.data_dir / TRAIN_DIR)
    assignment = assign_splits(train.impressions, ())  # no dev: holdout is never loaded
    news = train.news
    base = dataclasses.replace(config.base, slice_category=None)

    stability_config = _subset(base, config.stability_models, config.stability_compare_to)
    stability = {}
    for day in config.stability_days:
        start = datetime.combine(day, time.min)
        stats = [imp for imp in train.impressions if imp.timestamp < start]
        targets = [
            imp for imp in train.impressions if start <= imp.timestamp < start + timedelta(1)
        ]
        if not stats or not targets:
            raise ValueError(f"{day}: needs train impressions before and on that day")
        slices, share = unseen_slices(targets, stats)
        result = evaluate_split(
            f"day {day}",
            targets,
            stats,
            f"train before {day}",
            news,
            stability_config,
            extra_slices=slices,
        )
        result["share_candidate_occurrences_unclicked_in_statistics"] = share
        stability[day.isoformat()] = result

    ablation_config = _subset(base, config.ablation_models, config.ablation_compare_to)
    fit = assignment.statistics_source()
    slices, share = unseen_slices(assignment.tune, fit)
    ablations = evaluate_split(
        "tune", assignment.tune, fit, "fit", news, ablation_config, extra_slices=slices
    )
    ablations["share_candidate_occurrences_unclicked_in_statistics"] = share
    status = _git("status", "--porcelain", "--", ".", ":!results", ":!reports")
    provenance = read_provenance(config.base.data_dir)
    return {
        "project": "recsys-lab",
        "kind": "tune_diagnostics",
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
            "history_edges": list(base.history_edges),
            "ks": list(base.ks),
            "models": {m.name: {"ranker": m.kind, "params": m.params} for m in config.base.models},
        },
        "stability": stability,
        "ablations": ablations,
    }


# --- Adoption rule (D-020) -------------------------------------------------------------------


def adoption(ablations: Mapping[str, Any], model: str, base: str, metric: str) -> tuple[bool, str]:
    """Phase 3 candidate iff Δ vs its base has CI > 0 overall and isn't below 0 for
    zero-History users (CI entirely negative)."""
    overall = ablations["models"][model]["metrics"][metric]["delta_vs"].get(base)
    cold = ablations["segments_by_history_length"].get("0", {}).get("models", {})
    cold_delta = cold.get(model, {}).get("metrics", {}).get(metric, {}).get("delta_vs", {})
    cold_delta = cold_delta.get(base)
    if overall is None or overall["ci_low"] is None:
        return False, "no interval"
    if overall["ci_low"] <= 0:
        return False, "overall interval not above 0"
    if cold_delta is not None and cold_delta["ci_high"] is not None and cold_delta["ci_high"] < 0:
        return False, "hurts zero-History users"
    return True, "overall interval above 0; zero-History users not hurt"


# --- Report ----------------------------------------------------------------------------------


def _pair(entry: Mapping[str, Any], ref: str, *, negate: bool = False) -> str:
    delta = entry.get("delta_vs", {}).get(ref)
    if delta is None or delta["ci_low"] is None:
        return "—"
    mean, low, high = delta["mean"], delta["ci_low"], delta["ci_high"]
    if negate:
        mean, low, high = -mean, -high, -low
    return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"


def render_diagnostics(results: Mapping[str, Any]) -> str:
    config = results["config"]
    metric = f"ndcg@{max(config['ks'])}"
    label = metric_label(metric)
    models = config["models"]
    lines = [
        "# Tune-only diagnostics (Phase 2)",
        "",
        "> Generated by `scripts/diagnose.py`; do not edit by hand. Reads `MINDsmall_train`"
        " only: the holdout is never loaded. Offline ranking metrics, **not** engagement or"
        " CTR. Decision rules are fixed in DECISIONS.md D-020.",
        "",
        f"- Data provenance: `{', '.join(results['data_provenance'])}`",
        f"- Generated: {results['generated_at']} at git `{results['git_sha']}`"
        + (" (uncommitted changes)" if results["git_dirty"] else ""),
        f"- Reproduce: `{results['reproduce']}`",
        f"- Seed {config['seed']}; {config['bootstrap_samples']} user-grouped bootstrap"
        " samples; 95% percentile intervals.",
        "",
        "## 1. Stability across days (rolling origin)",
        "",
        "Each day is scored with rankers fitted on all train impressions before it. The"
        " last day is the official tune protocol.",
        "",
        "| Day | Impressions | Fitted on | Unclicked candidates | `popularity` minus `random`"
        " | `tfidf` minus `category_affinity` | same, random tie-breaks |",
        "|---|---|---|---|---|---|---|",
    ]
    for day, split in results["stability"].items():
        m = split["models"]
        lines.append(
            f"| {day} | {split['n_impressions']:,} | {split['n_statistics_impressions']:,} |"
            f" {split['share_candidate_occurrences_unclicked_in_statistics']:.1%} |"
            f" {_pair(m['random']['metrics'][metric], 'popularity', negate=True)} |"
            f" {_pair(m['tfidf']['metrics'][metric], 'category_affinity')} |"
            f" {_pair(m['tfidf_no_pop']['metrics'][metric], 'category_affinity_no_pop')} |"
        )
    lines += [
        "",
        f'Paired Δ {label} [95% CI]. "Unclicked candidates": share of candidate occurrences'
        " with no click in the statistics source, where the popularity tie-break is"
        " uninformative.",
        "",
        f"{label} by day:",
        "",
        "| Day | "
        + " | ".join(f"`{n}`" for n in next(iter(results["stability"].values()))["models"])
        + " |",
        "|---|" + "---|" * len(next(iter(results["stability"].values()))["models"]),
    ]
    for day, split in results["stability"].items():
        lines.append(
            f"| {day} | "
            + " | ".join(_with_ci(r["metrics"][metric]) for r in split["models"].values())
            + " |"
        )
    tune = results["ablations"]
    unclicked = {k: v for k, v in tune["slices"].items() if k.startswith("unclicked_")}
    lines += [
        "",
        "## 2. Where the content rankers differ (tune protocol)",
        "",
        f"{label} by share of an impression's candidates unclicked in `fit`:",
        "",
        *_segment_tables(unclicked, "Unclicked share", metric, "category_affinity"),
        "",
        "## 3. Ablations (tune protocol: fitted on `fit`, scored on `tune`)",
        "",
        "Phase 3 candidate rule (D-020): paired Δ vs the variant's base model has a 95% CI"
        " above 0 overall, and its CI for zero-History users isn't entirely below 0.",
        "",
        f"| Model | Base | Params | {label} [95% CI] | Δ vs base [95% CI] |"
        " Δ vs base, zero History | Fallback | p95 ms | Phase 3 candidate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    cold = tune["segments_by_history_length"].get("0", {}).get("models", {})
    for name, result in tune["models"].items():
        base = models[name]["ranker"]
        params = models[name]["params"]
        shown = {
            k: v for k, v in params.items() if k not in ("stop_words", "sublinear_tf", "min_df")
        }
        entry = result["metrics"][metric]
        if name == base:
            lines.append(
                f"| `{name}` | — | — | {_with_ci(entry)} | — | — | {result['n_fallback']:,} |"
                f" {result['latency_ms']['p95']:.3f} | — |"
            )
            continue
        cold_entry = cold.get(name, {}).get("metrics", {}).get(metric, {})
        ok, why = adoption(tune, name, base, metric)
        lines.append(
            f"| `{name}` | `{base}` | `{json.dumps(shown)}` | {_with_ci(entry)} |"
            f" {_pair(entry, base)} | {_pair(cold_entry, base) if cold_entry else '—'} |"
            f" {result['n_fallback']:,} | {result['latency_ms']['p95']:.3f} |"
            f" {'**yes**' if ok else 'no'} ({why}) |"
        )
    lines += [
        "",
        f"Tune impressions: {tune['n_impressions']:,}; zero-History impressions:"
        f" {tune['segments_by_history_length'].get('0', {}).get('n_impressions', 0):,}.",
    ]
    return "\n".join(lines) + "\n"


def write_diagnostics(results: Mapping[str, Any], config: DiagnosticsConfig) -> list[Path]:
    config.results.parent.mkdir(parents=True, exist_ok=True)
    config.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(render_diagnostics(results), encoding="utf-8")
    return [config.results, config.report]
