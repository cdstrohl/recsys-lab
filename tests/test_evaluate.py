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
        compare_to=("popularity", "category_affinity"),
        bootstrap_samples=50,
        results_dir=tmp_path / "results",
        report=tmp_path / "reports" / "baseline_results.md",
        manifest=tmp_path / "eval" / "manifest.json",
        interpretation=tmp_path / "interpretation.md",
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
    assert ndcg["category_affinity"]["delta_vs"]["popularity"]["mean"] == pytest.approx(0.25)
    assert ndcg["popularity"]["delta_vs"]["category_affinity"]["mean"] == pytest.approx(-0.25)
    assert set(ndcg["tfidf"]["delta_vs"]) == {"popularity", "category_affinity"}
    assert all(entry["n_eligible"] == 2 for entry in ndcg.values())
    # A model is never compared with itself.
    assert "popularity" not in ndcg["popularity"]["delta_vs"]
    assert "category_affinity" not in ndcg["category_affinity"]["delta_vs"]


@pytest.mark.parametrize(
    ("split", "n_candidates", "n_history"),
    [
        # tune: 8-11 all show a sports item (N1 or N13); History has sports for U6 (9), U1 (10).
        ("tune", 4, 2),
        # holdout: 101-103 all show N15 or N13; History has sports for U1 (101), U5 (103).
        ("holdout", 3, 2),
    ],
)
def test_sports_slices_match_hand_counts(
    experiments: dict[str, object], split: str, n_candidates: int, n_history: int
) -> None:
    slices = experiments["splits"][split]["slices"]  # type: ignore[index]
    assert slices["candidates_include_sports"]["n_impressions"] == n_candidates
    assert slices["history_includes_sports"]["n_impressions"] == n_history
    assert "sports" in slices["history_includes_sports"]["description"]


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
        if path.suffix == ".png":
            continue
        text = path.read_text()
        assert not re.search(r"\bU\d+\b", text), f"user ID in {path}"
        assert "[synthetic]" not in text, f"article text in {path}"
    report = locked.report.read_text()
    assert "not** engagement" in report and "`synthetic`" in report


def test_figure_is_written_and_listed(locked: EvalConfig, experiments: dict[str, object]) -> None:
    write_outputs(experiments, locked)
    figure = locked.results_dir / "figures" / "ndcg_by_history.png"
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    summary = json.loads((locked.results_dir / "summary.json").read_text())
    assert summary["figures"] == [figure.as_posix()]
    assert "](../results/figures/ndcg_by_history.png)" in locked.report.read_text()


def test_report_marks_cold_start_fallback_instead_of_a_delta(
    locked: EvalConfig, experiments: dict[str, object]
) -> None:
    write_outputs(experiments, locked)
    # Holdout bucket "0" holds only impression 102 (U7, empty History).
    row = next(
        line
        for line in locked.report.read_text().splitlines()
        if line.startswith("| 0 | 1 | 1 |") and "fallback" in line
    )
    assert row.count("fallback: = `popularity`") == 2
    assert "(baseline)" in row


def test_interpretation_is_included_verbatim(
    tmp_path: Path, experiments: dict[str, object]
) -> None:
    config = make_config(tmp_path)
    config.interpretation.write_text("Category affinity is the one to beat.\n")  # type: ignore[union-attr]
    write_outputs(experiments, config)
    report = config.report.read_text()
    assert "## Interpretation" in report
    assert "Category affinity is the one to beat." in report


def test_repo_interpretation_holds_no_numbers() -> None:
    # Numbers live only in generated tables (D-018); the memo must not drift from them.
    path = Path(__file__).parents[1] / "reports" / "interpretation.md"
    if not path.exists():
        pytest.skip("no interpretation written yet")
    assert not re.search(r"\.\d|\d\s*%", path.read_text()), "decimal or percent in memo"
