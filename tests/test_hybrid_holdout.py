"""E2 exploratory holdout on the synthetic fixtures: freeze, guards, readings, outputs."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import SYNTHETIC

from recsys.evaluate import EvalConfig, ModelSpec, lock_manifest_from_config, run
from recsys.hybrid import (
    BASE_FEATURES,
    HybridConfig,
    fit_linear,
    run_hybrid_development,
    write_hybrid,
)
from recsys.hybrid_holdout import (
    HoldoutConfig,
    freeze_hybrids,
    linear_from_json,
    reading,
    render_holdout,
    run_exploratory_holdout,
    write_freeze,
)
from recsys.loaders import TRAIN_DIR
from recsys.splits import Split, SplitAssignment

ROOT = Path(__file__).resolve().parents[1]
TFIDF = {"min_df": 1}
MODELS = (
    ModelSpec("random"),
    ModelSpec("popularity"),
    ModelSpec("popularity_48h", {"window_hours": 48}, "popularity"),
    ModelSpec("category_affinity"),
    ModelSpec("tfidf", TFIDF),
)


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> HoldoutConfig:
    """Freeze reads a train-only copy; scoring reads the full fixtures."""
    train_only = tmp_path_factory.mktemp("train_only")
    shutil.copytree(SYNTHETIC / TRAIN_DIR, train_only / TRAIN_DIR)
    shutil.copy(SYNTHETIC / "PROVENANCE.json", train_only / "PROVENANCE.json")
    out = tmp_path_factory.mktemp("out")
    base = EvalConfig(
        data_dir=train_only,
        models=MODELS,
        splits=(Split.TUNE,),
        history_edges=(0, 1, 6),
        compare_to=("category_affinity", "tfidf"),
        bootstrap_samples=50,
        manifest=out / "eval" / "manifest.json",
    )
    hybrid = HybridConfig(
        base=base,
        days=(date(2019, 11, 10), date(2019, 11, 11)),
        C=1.0,
        sensitivity_C=(),
        tfidf_model="tfidf",
        results=out / "hybrid_development.json",
        report=out / "hybrid_development.md",
    )
    write_hybrid(run_hybrid_development(hybrid), hybrid)
    reference = dataclasses.replace(
        base,
        data_dir=SYNTHETIC,
        splits=(Split.HOLDOUT,),
        results_dir=out / "v1",
        report=out / "v1" / "report.md",
        interpretation=None,
    )
    lock_manifest_from_config(reference)
    (out / "experiments.json").write_text(json.dumps(run(reference)), encoding="utf-8")
    return HoldoutConfig(
        hybrid=hybrid,
        bootstrap_samples=50,
        slice_category="sports",
        primary="hybrid",
        versus="category_affinity",
        target_relative=0.05,
        secondary=("hybrid_no_interaction", "hybrid_full"),
        manifest=base.manifest,
        freeze=out / "eval" / "hybrid_freeze.json",
        development=hybrid.results,
        reference=out / "experiments.json",
        results=out / "hybrid_exploratory_holdout.json",
        report=out / "hybrid_exploratory_holdout.md",
    )


@pytest.fixture(scope="module")
def freeze(config: HoldoutConfig) -> dict[str, Any]:
    original = SplitAssignment.holdout

    def forbidden(self: SplitAssignment, *, allow_holdout: bool = False) -> Any:
        raise AssertionError("the freeze touched the holdout")

    SplitAssignment.holdout = forbidden  # type: ignore[method-assign]
    try:
        frozen = freeze_hybrids(config)
    finally:
        SplitAssignment.holdout = original  # type: ignore[method-assign]
    write_freeze(frozen, config)
    return frozen


@pytest.fixture(scope="module")
def results(config: HoldoutConfig, freeze: dict[str, Any]) -> dict[str, Any]:
    scoring = dataclasses.replace(
        config,
        hybrid=dataclasses.replace(
            config.hybrid, base=dataclasses.replace(config.hybrid.base, data_dir=SYNTHETIC)
        ),
    )
    return run_exploratory_holdout(scoring)


def test_freeze_reads_train_only_and_records_every_hybrid(
    config: HoldoutConfig, freeze: dict[str, Any]
) -> None:
    assert freeze["holdout_used"] is False
    assert set(freeze["train_files"]) == {f"{TRAIN_DIR}/news.tsv", f"{TRAIN_DIR}/behaviors.tsv"}
    development = json.loads(config.development.read_text())
    assert set(freeze["weights"]) == set(development["config"]["features"])
    assert freeze["weights"]["hybrid"]["standardized_coef"] == pytest.approx(
        development["frozen_weights"]["primary"]["standardized_coef"], abs=1e-9
    )
    assert freeze["plan"]["primary"] == {
        "model": "hybrid",
        "versus": "category_affinity",
        "target_relative": 0.05,
    }


def test_freeze_refuses_weights_that_differ_from_development(
    config: HoldoutConfig, tmp_path: Path
) -> None:
    development = json.loads(config.development.read_text())
    development["frozen_weights"]["primary"]["standardized_coef"]["tfidf"] += 0.1
    tampered = tmp_path / "development.json"
    tampered.write_text(json.dumps(development))
    with pytest.raises(RuntimeError, match="refitted weights differ"):
        freeze_hybrids(dataclasses.replace(config, development=tampered))


def test_frozen_weights_round_trip_exactly() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(BASE_FEATURES)))
    model = fit_linear(x, (x[:, 0] > 0).astype(int), BASE_FEATURES, C=1.0)
    again = linear_from_json(json.loads(json.dumps(model.to_json())))
    np.testing.assert_array_equal(model.score(x), again.score(x))


def test_holdout_run_is_labeled_exploratory_and_matches_v1(
    results: dict[str, Any], config: HoldoutConfig
) -> None:
    assert results["label"] == "exploratory holdout"
    assert results["confirmatory"] is False and results["holdout_used"] is True
    assert results["holdout"]["statistics_source"] == "fit+tune"
    assert results["holdout"]["n_impressions"] == 3
    assert results["freeze"]["sha256"]
    assert results["v1_continuity"] and all(results["v1_continuity"].values())
    assert {"hybrid", "hybrid_no_interaction"} <= set(results["holdout"]["models"])
    assert results["primary"]["model"] == "hybrid"
    assert results["primary"]["versus"] == "category_affinity"


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"mean": 0.02, "ci_low": 0.01, "ci_high": 0.03}, "better, meets target"),
        ({"mean": 0.01, "ci_low": 0.005, "ci_high": 0.015}, "better, below target"),
        ({"mean": 0.01, "ci_low": 0.0, "ci_high": 0.02}, "no clear difference"),
        ({"mean": -0.01, "ci_low": -0.02, "ci_high": -0.001}, "worse"),
        (None, "not computed"),
    ],
)
def test_reading_rule(delta: dict[str, float] | None, expected: str) -> None:
    assert reading(delta, ref_mean=0.3, target=0.05)["reading"] == expected


def _script_config(config: HoldoutConfig, path: Path, freeze: Path) -> Path:
    raw = {
        "development_config": "configs/hybrid.yaml",
        "bootstrap_samples": 50,
        "primary": {"model": "hybrid", "versus": "category_affinity", "target_relative": 0.05},
        "secondary": [],
        "manifest": str(config.manifest),
        "freeze": str(freeze),
        "development_results": str(config.development),
        "reference": str(config.reference),
        "outputs": {"results": str(path / "r.json"), "report": str(path / "r.md")},
    }
    out = path / "config.yaml"
    out.write_text(json.dumps(raw))  # JSON is valid YAML
    return out


def _script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/hybrid_holdout.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_scoring_refuses_an_uncommitted_freeze(config: HoldoutConfig, tmp_path: Path) -> None:
    untracked = tmp_path / "freeze.json"
    untracked.write_text("{}")
    done = _script("--config", str(_script_config(config, tmp_path, untracked)))
    assert done.returncode == 1 and "must be committed" in done.stderr
    assert not (tmp_path / "r.json").exists()


def test_freeze_is_never_overwritten(
    config: HoldoutConfig, freeze: dict[str, Any], tmp_path: Path
) -> None:
    before = config.freeze.read_text()
    done = _script("--config", str(_script_config(config, tmp_path, config.freeze)), "--freeze")
    assert done.returncode == 1 and "never overwritten" in done.stderr
    assert config.freeze.read_text() == before


def test_report_is_labeled_and_holds_no_raw_records(results: dict[str, Any]) -> None:
    report = render_holdout(results)
    assert "Exploratory holdout, not confirmatory" in report
    assert "## 1. Primary comparison (pre-registered, D-027)" in report
    assert not re.search(r"\bU\d+\b", report)
    assert "[synthetic]" not in report
