"""Exploratory holdout run of the frozen E2 hybrids on MIND-small dev (D-022, D-027).

Two steps, run in this order and committed in between:
1. **Freeze** (`freeze_hybrids`, train only): refit every hybrid's weights on all four
   development days, check them against `results/hybrid_development.json`, and record
   the weights and the analysis plan in a freeze file. Dev is never loaded.
2. **Score** (`run_exploratory_holdout`): apply the frozen weights, unchanged, to base
   features built from all train (fit + tune, D-013), and score MIND-small dev.

MIND-small dev was already inspected (D-018, D-021), so every number from step 2 is an
*exploratory holdout* result. It can't change a frozen decision, and it isn't a
confirmatory test.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from recsys.evaluate import (
    _git,
    evaluate_split,
    load_data,
    metric_label,
    sha256_file,
    verify_manifest,
)
from recsys.hybrid import (
    FULL,
    NO_INTERACTION,
    PRIMARY,
    HybridConfig,
    HybridFeatures,
    HybridRanker,
    LinearModel,
    _delta,
    _rows,
    build_day,
    fit_linear,
    load_hybrid_config,
    training_rows,
)
from recsys.inventory import read_provenance
from recsys.loaders import TRAIN_DIR, load_mind_dir

LABEL = "exploratory holdout"
FREEZE_REPRODUCE = (
    "uv run python scripts/hybrid_holdout.py --config configs/hybrid_holdout.yaml --freeze"
)
REPRODUCE = "uv run python scripts/hybrid_holdout.py --config configs/hybrid_holdout.yaml"
HYBRIDS = (PRIMARY, NO_INTERACTION, FULL)


@dataclass(frozen=True, slots=True)
class HoldoutConfig:
    hybrid: HybridConfig  # the development config: base models, days, C, TF-IDF params
    bootstrap_samples: int
    slice_category: str | None
    primary: str
    versus: str
    target_relative: float
    secondary: tuple[str, ...]
    manifest: Path
    freeze: Path
    development: Path  # results/hybrid_development.json
    reference: Path  # results/experiments.json (v1, for continuity)
    results: Path
    report: Path


def load_holdout_config(path: Path) -> HoldoutConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    primary = raw["primary"]
    return HoldoutConfig(
        hybrid=load_hybrid_config(Path(raw["development_config"])),
        bootstrap_samples=int(raw["bootstrap_samples"]),
        slice_category=raw.get("slice_category"),
        primary=primary["model"],
        versus=primary["versus"],
        target_relative=float(primary["target_relative"]),
        secondary=tuple(raw["secondary"]),
        manifest=Path(raw["manifest"]),
        freeze=Path(raw["freeze"]),
        development=Path(raw["development_results"]),
        reference=Path(raw["reference"]),
        results=Path(raw["outputs"]["results"]),
        report=Path(raw["outputs"]["report"]),
    )


def linear_from_json(raw: Mapping[str, Any]) -> LinearModel:
    columns = tuple(raw["columns"])
    return LinearModel(
        columns,
        np.array([raw["mean"][c] for c in columns], dtype=np.float64),
        np.array([raw["scale"][c] for c in columns], dtype=np.float64),
        np.array([raw["standardized_coef"][c] for c in columns], dtype=np.float64),
        float(raw["intercept"]),
        float(raw["C"]),
        int(raw["n_rows"]),
    )


def _tfidf_params(config: HybridConfig) -> dict[str, Any]:
    spec = next(m for m in config.base.models if m.name == config.tfidf_model)
    return {k: v for k, v in spec.params.items() if k != "tie_break"}


def _train_hashes(manifest: Path, data_dir: Path) -> dict[str, str]:
    """The manifest's train file hashes, after checking the files on disk match them."""
    locked = json.loads(manifest.read_text(encoding="utf-8"))
    hashes = {n: h for n, h in locked["files"].items() if n.startswith(TRAIN_DIR)}
    wrong = [n for n, h in hashes.items() if sha256_file(data_dir / n) != h]
    if wrong:
        raise RuntimeError(f"train files differ from {manifest}: {', '.join(wrong)}")
    return hashes


# --- Step 1: freeze (train only) -------------------------------------------------------------


def freeze_hybrids(config: HoldoutConfig) -> dict[str, Any]:
    """All-days weights for every hybrid, plus the analysis plan. Never loads dev."""
    hybrid = config.hybrid
    development = json.loads(config.development.read_text(encoding="utf-8"))
    columns: dict[str, list[str]] = development["config"]["features"]
    train_hashes = _train_hashes(config.manifest, hybrid.base.data_dir)
    train = load_mind_dir(hybrid.base.data_dir / TRAIN_DIR)
    tfidf_params = _tfidf_params(hybrid)
    days = [build_day(train.impressions, d, train.news, tfidf_params) for d in sorted(hybrid.days)]
    weights = {
        name: fit_linear(*training_rows(days, cols), cols, hybrid.C).to_json()
        for name, cols in columns.items()
    }
    # The development run stored the primary and full weights; refitting must reproduce them.
    stored = {PRIMARY: "primary", FULL: "full"}
    for name, key in stored.items():
        if name not in weights:
            continue
        old = linear_from_json(development["frozen_weights"][key])
        new = linear_from_json(weights[name])
        if not np.allclose(old.coef, new.coef, rtol=0, atol=1e-9):
            raise RuntimeError(f"{name}: refitted weights differ from {config.development}")

    status = _git("status", "--porcelain", "--", ".", ":!results", ":!reports", ":!eval")
    provenance = read_provenance(hybrid.base.data_dir)
    return {
        "project": "recsys-lab",
        "kind": "hybrid_freeze",
        "label": LABEL,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(status) if status is not None else None,
        "reproduce": FREEZE_REPRODUCE,
        "data_provenance": [
            hybrid.base.data_provenance if provenance == "unlabeled" else provenance
        ],
        "holdout_used": False,
        "train_files": train_hashes,
        "development_results_sha256": sha256_file(config.development),
        "weights_fitted_on": [d.day.isoformat() for d in days],
        "statistics_at_evaluation": "fit+tune (all MINDsmall_train), D-013 / D-022",
        "tfidf_params": tfidf_params,
        "base_models": {m.name: {"ranker": m.kind, "params": m.params} for m in hybrid.base.models},
        "weights": weights,
        "plan": {
            "metric": f"ndcg@{max(hybrid.base.ks)}",
            "primary": {
                "model": config.primary,
                "versus": config.versus,
                "target_relative": config.target_relative,
            },
            "secondary": list(config.secondary),
            "rivals": list(hybrid.base.compare_to),
            "seed": hybrid.base.seed,
            "bootstrap_samples": config.bootstrap_samples,
            "slice_category": config.slice_category,
            "history_edges": list(hybrid.base.history_edges),
        },
    }


def write_freeze(freeze: Mapping[str, Any], config: HoldoutConfig) -> Path:
    config.freeze.parent.mkdir(parents=True, exist_ok=True)
    config.freeze.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return config.freeze


# --- Step 2: score the holdout ---------------------------------------------------------------


def reading(delta: Mapping[str, Any] | None, ref_mean: float, target: float) -> dict[str, Any]:
    """The pre-registered reading of a paired difference (D-027)."""
    if delta is None or delta.get("ci_low") is None:
        return {"relative": None, "reading": "not computed"}
    relative = delta["mean"] / ref_mean if ref_mean else None
    if delta["ci_high"] < 0:
        verdict = "worse"
    elif delta["ci_low"] <= 0:
        verdict = "no clear difference"
    elif relative is not None and relative >= target:
        verdict = "better, meets target"
    else:
        verdict = "better, below target"
    return {"relative": relative, "reading": verdict}


def run_exploratory_holdout(config: HoldoutConfig) -> dict[str, Any]:
    freeze = json.loads(config.freeze.read_text(encoding="utf-8"))
    plan = freeze["plan"]
    base = dataclasses.replace(
        config.hybrid.base,
        bootstrap_samples=plan["bootstrap_samples"],
        slice_category=plan["slice_category"],
        compare_to=tuple(plan["rivals"]),
    )
    data = load_data(base)
    manifest_version = verify_manifest(config.manifest, base.data_dir, data.assignment)
    stats = data.assignment.statistics_source(include_tune=True)
    targets = data.assignment.holdout(allow_holdout=True)
    features = HybridFeatures(stats, data.news, freeze["tfidf_params"])
    hybrids = [
        HybridRanker(name, features, linear_from_json(raw))
        for name, raw in freeze["weights"].items()
    ]
    split = evaluate_split(
        "holdout", targets, stats, "fit+tune", data.news, base, extra_rankers=hybrids
    )

    metric = plan["metric"]
    models = split["models"]
    target = plan["primary"]["target_relative"]
    versus = plan["primary"]["versus"]

    def read(model: str, ref: str) -> dict[str, Any]:
        entry = models[model]["metrics"][metric]
        return {
            "model": model,
            "versus": ref,
            "delta": entry["delta_vs"].get(ref),
            **reading(entry["delta_vs"].get(ref), models[ref]["metrics"][metric]["mean"], target),
        }

    primary = read(plan["primary"]["model"], versus)
    secondary = [read(m, versus) for m in plan["secondary"] if m in models]
    secondary += [
        read(m, ref)
        for m in (plan["primary"]["model"], *plan["secondary"])
        for ref in plan["rivals"]
        if ref != versus and m in models
    ]

    reference = json.loads(config.reference.read_text(encoding="utf-8"))
    v1 = reference["splits"]["holdout"]["models"]
    continuity = {
        name: all(
            np.isclose(
                models[name]["metrics"][m]["mean"],
                v1[name]["metrics"][m]["mean"],
                rtol=0,
                atol=1e-12,
            )
            for m in v1[name]["metrics"]
        )
        for name in v1
        if name in models
    }

    status = _git("status", "--porcelain", "--", ".", ":!results", ":!reports")
    return {
        "project": "recsys-lab",
        "kind": "hybrid_exploratory_holdout",
        "label": LABEL,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(status) if status is not None else None,
        "reproduce": REPRODUCE,
        "data_provenance": [data.provenance],
        "manifest_version": manifest_version,
        "holdout_used": True,
        "confirmatory": False,
        "freeze": {
            "path": str(config.freeze),
            "sha256": sha256_file(config.freeze),
            "frozen_at_git": freeze["git_sha"],
        },
        "plan": plan,
        "primary": primary,
        "secondary": secondary,
        "v1_continuity": continuity,
        "development": _development_summary(config.development, metric, versus),
        "holdout": split,
    }


def _development_summary(path: Path, metric: str, versus: str) -> dict[str, Any]:
    """Development deltas vs `versus`, per day and pooled, for side-by-side reading."""
    development = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for name in HYBRIDS:
        pooled = development["pooled"]["models"].get(name)
        if pooled is None:
            continue
        out[name] = {
            "pooled": pooled["metrics"][metric]["delta_vs"][versus],
            "per_day": {
                day: result["models"][name]["metrics"][metric]["delta_vs"][versus]["mean"]
                for day, result in development["per_day"].items()
            },
        }
    return out


# --- Report ----------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1%}"


def _reading_rows(readings: Sequence[Mapping[str, Any]], metric: str) -> list[list[str]]:
    return [
        [
            f"`{r['model']}` - `{r['versus']}`",
            _delta({"delta_vs": {r["versus"]: r["delta"]}}, r["versus"]),
            _pct(r["relative"]),
            f"**{r['reading']}**",
        ]
        for r in readings
    ]


def render_holdout(results: Mapping[str, Any]) -> str:
    plan = results["plan"]
    metric = plan["metric"]
    label = metric_label(metric)
    split = results["holdout"]
    models = list(split["models"])
    rivals = plan["rivals"]
    target = plan["primary"]["target_relative"]
    lines = [
        "# E2 exploratory holdout: frozen hybrids on MIND-small dev",
        "",
        "> Generated by `scripts/hybrid_holdout.py`; do not edit by hand. **Exploratory"
        " holdout, not confirmatory:** MIND-small dev was already inspected (D-018, D-021),"
        " so these numbers can't change a frozen decision and don't count as a fresh test"
        " (D-022). Weights and plan were frozen before scoring (D-027). Offline ranking"
        " metrics, **not** engagement or CTR.",
        "",
        f"- Data provenance: `{', '.join(results['data_provenance'])}`; manifest"
        f" `{results['manifest_version']}` verified, unchanged",
        f"- Generated: {results['generated_at']} at git `{results['git_sha']}`"
        + (" (uncommitted changes)" if results["git_dirty"] else ""),
        f"- Frozen weights and plan: `{results['freeze']['path']}` (frozen at git"
        f" `{results['freeze']['frozen_at_git']}`, sha256"
        f" `{results['freeze']['sha256'][:12]}`)",
        f"- Reproduce: `{results['reproduce']}`",
        f"- Base features from all train (fit + tune, {split['n_statistics_impressions']:,}"
        f" impressions); {split['n_impressions']:,} holdout impressions,"
        f" {split['n_users']:,} users",
        f"- Seed {plan['seed']}; {plan['bootstrap_samples']} user-grouped bootstrap samples;"
        " 95% percentile intervals.",
        "",
        "## 1. Primary comparison (pre-registered, D-027)",
        "",
        f"{label}; readings: *better, meets target* (interval above 0 and relative gain"
        f" ≥ {target:.0%}), *better, below target*, *no clear difference* (interval"
        " includes 0), *worse*.",
        "",
    ]
    header = ["Comparison", f"Δ {label} [95% CI]", "Relative", "Reading"]
    lines += _rows(header, _reading_rows([results["primary"]], metric))
    lines += ["", "## 2. Secondary comparisons", ""]
    lines += _rows(header, _reading_rows(results["secondary"], metric))

    metrics = list(split["models"][models[0]]["metrics"])
    lines += ["", "## 3. All models", ""]
    rows = []
    for m in models:
        entry = split["models"][m]["metrics"]
        rows.append(
            [
                f"`{m}`",
                *(
                    f"{entry[k]['mean']:.4f} [{entry[k]['ci_low']:.4f}, {entry[k]['ci_high']:.4f}]"
                    for k in metrics
                ),
                *(_delta(entry[metric], r) for r in rivals),
            ]
        )
    lines += _rows(
        ["Model", *(metric_label(k) for k in metrics), *(f"Δ {label} vs `{r}`" for r in rivals)],
        rows,
    )

    lines += [
        "",
        f"## 4. Development vs exploratory holdout (Δ {label} vs `{plan['primary']['versus']}`)",
        "",
        "Development numbers are leave-one-day-out on train days (D-026). Side by side for"
        " reading only; the holdout is one more day.",
        "",
    ]
    dev = results["development"]
    days = list(next(iter(dev.values()))["per_day"]) if dev else []
    rows = [
        [
            f"`{name}`",
            *(f"{d['per_day'][day]:+.4f}" for day in days),
            _delta({"delta_vs": {"x": d["pooled"]}}, "x"),
            _delta(split["models"][name]["metrics"][metric], plan["primary"]["versus"]),
        ]
        for name, d in dev.items()
        if name in split["models"]
    ]
    lines += _rows(["Model", *days, "development pooled", "exploratory holdout"], rows)

    lines += ["", f"## 5. By History length ({label})", ""]
    segments = split["segments_by_history_length"]
    versus = plan["primary"]["versus"]
    focus = [m for m in models if m in (versus, *rivals) or m.startswith("hybrid")]
    rows = []
    for bucket, seg in segments.items():
        cells = []
        for m in focus:
            entry = seg["models"][m]["metrics"][metric]
            cells.append("—" if entry["mean"] is None else f"{entry['mean']:.4f}")
        hybrid_entry = seg["models"][plan["primary"]["model"]]["metrics"][metric]
        rows.append([f"{bucket} ({seg['n_impressions']:,})", *cells, _delta(hybrid_entry, versus)])
    lines += _rows(
        [
            "History (impressions)",
            *(f"`{m}`" for m in focus),
            f"`{plan['primary']['model']}` - `{versus}`",
        ],
        rows,
    )

    if split["slices"]:
        lines += ["", f"## 6. Slices ({label}; slices of general-news data)", ""]
        rows = []
        for sl in split["slices"].values():
            hybrid_entry = sl["models"][plan["primary"]["model"]]["metrics"][metric]
            rows.append(
                [
                    f"{sl['description']} ({sl['n_impressions']:,})",
                    *(f"{sl['models'][m]['metrics'][metric]['mean']:.4f}" for m in focus),
                    _delta(hybrid_entry, versus),
                ]
            )
        lines += _rows(
            [
                "Slice (impressions)",
                *(f"`{m}`" for m in focus),
                f"`{plan['primary']['model']}` - `{versus}`",
            ],
            rows,
        )

    lines += ["", "## 7. Costs", ""]
    rows = [
        [
            f"`{m}`",
            f"{r['latency_ms']['p50']:.2f} / {r['latency_ms']['p95']:.2f}",
            f"{r['n_fallback']:,}",
            f"{r['n_tied']:,}",
            f"{r['coverage@10']:.1%}",
        ]
        for m, r in split["models"].items()
    ]
    lines += _rows(["Model", "p50 / p95 ms", "Fallbacks", "Tied", "coverage@10"], rows)
    continuity = results["v1_continuity"]
    same = all(continuity.values())
    lines += [
        "",
        "Hybrid latency includes computing every feature live; treat p95 for the hybrids as"
        " rough (D-026).",
        "",
        "Continuity with the v1 holdout run (`results/experiments.json`): base rankers"
        f" {', '.join(f'`{m}`' for m in continuity)} reproduce its means exactly: "
        + ("yes." if same else "**no**: " + ", ".join(m for m, ok in continuity.items() if not ok)),
    ]
    return "\n".join(lines) + "\n"


def write_holdout(results: Mapping[str, Any], config: HoldoutConfig) -> list[Path]:
    config.results.parent.mkdir(parents=True, exist_ok=True)
    config.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(render_holdout(results), encoding="utf-8")
    return [config.results, config.report]
