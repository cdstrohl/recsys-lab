"""Runner tests on the synthetic fixtures: manifest guard, bootstrap, segments, outputs."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
from conftest import SYNTHETIC

from recsys.evaluate import (
    EvalConfig,
    ManifestError,
    ModelSpec,
    lock_manifest_from_config,
    run,
    user_bootstrap,
    write_outputs,
)
from recsys.splits import Split

MODELS = (
    ModelSpec("random"),
    ModelSpec("popularity"),
    ModelSpec("category_affinity"),
    ModelSpec("tfidf", {"min_df": 1}),
)


def make_config(tmp_path: Path, **overrides: object) -> EvalConfig:
    config = EvalConfig(
        data_dir=SYNTHETIC,
        models=MODELS,
        history_edges=(0, 1, 6),
        bootstrap_samples=50,
        results_dir=tmp_path / "results",
        report=tmp_path / "reports" / "baseline_results.md",
        manifest=tmp_path / "eval" / "manifest.json",
    )
    return dataclasses.replace(config, **overrides)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def locked(tmp_path_factory: pytest.TempPathFactory) -> EvalConfig:
    config = make_config(tmp_path_factory.mktemp("run"))
    lock_manifest_from_config(config)
    return config


@pytest.fixture(scope="module")
def experiments(locked: EvalConfig) -> dict[str, object]:
    return run(locked)


# --- Bootstrap -------------------------------------------------------------------------------


def test_bootstrap_resamples_users_not_impressions() -> None:
    # User 0 has two impressions scoring 1; user 1 has one scoring 0. Resampling two users
    # gives means 1 ({0, 0}), 2/3 ({0, 1}) or 0 ({1, 1}); an impression-level bootstrap
    # could also give 1/3.
    boot = user_bootstrap(np.array([0, 0, 1]), np.array([[1.0, 1.0, 0.0]]), n_samples=500, seed=1)
    assert boot.shape == (500, 1)
    assert set(np.round(boot[:, 0], 6)) <= {0.0, 0.666667, 1.0}


def test_bootstrap_is_seeded_and_paired() -> None:
    users = np.array([0, 1, 2, 3, 3])
    values = np.array([[0.1, 0.5, 0.9, 0.2, 0.4], [0.2, 0.6, 1.0, 0.3, 0.5]])
    a = user_bootstrap(users, values, n_samples=64, seed=7, chunk=10)
    np.testing.assert_array_equal(a, user_bootstrap(users, values, n_samples=64, seed=7))
    # Model 2 is model 1 plus 0.1 everywhere, so every paired difference is exactly 0.1.
    np.testing.assert_allclose(a[:, 1] - a[:, 0], 0.1)


# --- Manifest guard --------------------------------------------------------------------------


def test_holdout_run_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        run(make_config(tmp_path))


def test_tune_only_run_needs_no_manifest(tmp_path: Path) -> None:
    config = make_config(tmp_path, splits=(Split.TUNE,))
    experiments = run(config)
    assert list(experiments["splits"]) == ["tune"]
    written = write_outputs(experiments, config)
    assert not (config.results_dir / "summary.json").exists()
    assert config.report in written


def test_manifest_mismatch_blocks_holdout(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    lock_manifest_from_config(config)
    manifest = json.loads(config.manifest.read_text())
    manifest["files"]["MINDsmall_dev/behaviors.tsv"] = "0" * 64
    config.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ManifestError, match="hash differs"):
        run(config)


def test_lock_refuses_to_overwrite(locked: EvalConfig) -> None:
    with pytest.raises(ManifestError, match="already exists"):
        lock_manifest_from_config(locked)


# --- Results ---------------------------------------------------------------------------------


def test_holdout_metrics_match_hand_computation(experiments: dict[str, object]) -> None:
    holdout = experiments["splits"]["holdout"]  # type: ignore[index]
    assert holdout["statistics_source"] == "fit+tune"
    assert holdout["n_impressions"] == 3 and holdout["n_no_click"] == 1
    ndcg = {m: r["metrics"]["ndcg@10"] for m, r in holdout["models"].items()}
    # 101: popularity ranks the clicked N15 last of 3 (nDCG 1/log2(4) = 0.5);
    # category affinity ranks it first (1). 102: both put the clicked N13 first (1).
    assert ndcg["popularity"]["mean"] == pytest.approx(0.75)
    assert ndcg["category_affinity"]["mean"] == pytest.approx(1.0)
    assert ndcg["category_affinity"]["delta_vs_baseline"]["mean"] == pytest.approx(0.25)
    assert all(entry["n_eligible"] == 2 for entry in ndcg.values())
    assert "delta_vs_baseline" not in ndcg["popularity"]


def test_segments_partition_impressions_and_flag_cold_start(
    experiments: dict[str, object],
) -> None:
    for split in experiments["splits"].values():  # type: ignore[attr-defined]
        segments = split["segments_by_history_length"]
        assert list(segments) == ["0", "1-5", "6+"]
        assert sum(s["n_impressions"] for s in segments.values()) == split["n_impressions"]
        cold = segments["0"]
        for name in ("category_affinity", "tfidf"):
            assert cold["models"][name]["n_fallback"] == cold["n_impressions"]
            assert split["models"][name]["n_fallback"] == cold["n_impressions"]
        assert split["models"]["popularity"]["n_fallback"] == 0


def test_outputs_are_labeled_synthetic_and_hold_no_raw_records(
    locked: EvalConfig, experiments: dict[str, object]
) -> None:
    written = write_outputs(experiments, locked)
    summary = json.loads((locked.results_dir / "summary.json").read_text())
    assert summary["data_provenance"] == ["synthetic"]
    assert summary["manifest_version"] == "v1"
    assert {m["split"] for m in summary["metrics"]} == {"holdout"}
    assert {m["model"] for m in summary["metrics"]} == {m.name for m in MODELS}
    assert summary["limitations"]
    for path in written:
        text = path.read_text()
        assert not re.search(r"\bU\d+\b", text), f"user ID in {path}"
        assert "[synthetic]" not in text, f"article text in {path}"
    report = locked.report.read_text()
    assert "not** engagement" in report and "`synthetic`" in report
