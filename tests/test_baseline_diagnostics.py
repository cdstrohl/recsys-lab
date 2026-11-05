"""Baseline diagnostics on the synthetic fixtures: no dev access, hand counts, rules."""

from __future__ import annotations

import dataclasses
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import SYNTHETIC

from recsys.baseline_diagnostics import (
    BaselineDiagnosticsConfig,
    HoldoutReference,
    _tie_measures,
    concentration_slices,
    coverage_and_ties,
    returning_slices,
    run_baseline_diagnostics,
    subsample,
    trailing,
    verdicts,
    write_baseline_diagnostics,
)
from recsys.evaluate import EvalConfig, ModelSpec, lock_manifest_from_config, run, write_outputs
from recsys.loaders import TRAIN_DIR, MindDir
from recsys.splits import Split, SplitAssignment, assign_splits

MODELS = (
    ModelSpec("random"),
    ModelSpec("popularity"),
    ModelSpec("category_affinity"),
    ModelSpec("tfidf", {"min_df": 1}),
)
TUNE_START = datetime(2019, 11, 11)


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real (synthetic) holdout experiments.json, as the evaluation runner writes it."""
    root = tmp_path_factory.mktemp("reference")
    config = EvalConfig(
        data_dir=SYNTHETIC,
        models=MODELS,
        compare_to=("popularity", "category_affinity"),
        bootstrap_samples=50,
        results_dir=root / "results",
        report=root / "report.md",
        manifest=root / "manifest.json",
        interpretation=None,
    )
    lock_manifest_from_config(config)
    write_outputs(run(config), config)
    return config.results_dir / "experiments.json"


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory, reference: Path) -> BaselineDiagnosticsConfig:
    train_only = tmp_path_factory.mktemp("train_only")
    shutil.copytree(SYNTHETIC / TRAIN_DIR, train_only / TRAIN_DIR)
    shutil.copy(SYNTHETIC / "PROVENANCE.json", train_only / "PROVENANCE.json")
    out = tmp_path_factory.mktemp("out")
    return BaselineDiagnosticsConfig(
        base=EvalConfig(
            data_dir=train_only,
            models=MODELS,
            splits=(Split.TUNE,),
            history_edges=(0, 1, 6),
            compare_to=("category_affinity", "random"),
            bootstrap_samples=50,
        ),
        days=(date(2019, 11, 10), date(2019, 11, 11)),
        day_windows=(1, None),
        tune_windows=(1, None),
        tune_subsamples=(0.5,),
        concentration_min_history=2,
        concentration_edges=(0.5,),
        holdout=HoldoutReference(reference, new_user_share=0.75),
        results=out / "baseline_diagnostics.json",
        report=out / "baseline_diagnostics.md",
    )


@pytest.fixture(scope="module")
def results(config: BaselineDiagnosticsConfig) -> dict[str, Any]:
    original = SplitAssignment.holdout

    def forbidden(self: SplitAssignment, *, allow_holdout: bool = False) -> Any:
        raise AssertionError("diagnostics touched the holdout")

    SplitAssignment.holdout = forbidden  # type: ignore[method-assign]
    try:
        return run_baseline_diagnostics(config)
    finally:
        SplitAssignment.holdout = original  # type: ignore[method-assign]


def test_runs_without_dev_and_plans_every_run(results: dict[str, Any]) -> None:
    assert results["holdout_used"] is False
    assert results["data_provenance"] == ["synthetic"]
    assert set(results["runs"]) == {
        "2019-11-10/1d",
        "2019-11-10/all",
        "2019-11-11/1d",
        "2019-11-11/all",
        "2019-11-11/all_sample50",
    }
    assert [r["fixed_targets"] for r in results["runs"].values()].count(True) == 3
    assert results["holdout_reference"]["n_impressions"] == 3


def test_tune_day_with_all_statistics_is_the_official_tune_run(
    results: dict[str, Any], config: BaselineDiagnosticsConfig
) -> None:
    official = run(dataclasses.replace(config.base, data_dir=SYNTHETIC))["splits"]["tune"]
    ours = results["runs"]["2019-11-11/all"]
    assert ours["n_statistics_impressions"] == 7
    for model, entry in official["models"].items():
        assert ours["means"][model] == pytest.approx(entry["metrics"]["ndcg@10"]["mean"]), model


def test_statistics_sources(train: MindDir) -> None:
    window = trailing(train.impressions, TUNE_START, 1)
    assert [i.impression_id for i in window] == ["4", "5", "6", "7"]
    assert len(trailing(train.impressions, TUNE_START, None)) == 7
    fit = trailing(train.impressions, TUNE_START, None)
    half = subsample(fit, 0.5, seed=13)
    assert half == subsample(fit, 0.5, seed=13)
    assert subsample(fit, 0.0, seed=13) == [] and subsample(fit, 1.0, seed=13) == fit


def test_population_slices_match_hand_counts(train: MindDir) -> None:
    s = assign_splits(train.impressions, ())
    # Tune: 8 U3 [N4], 9 U6 [7 items, 3 sports], 10 U1 [N1 N2 sports], 11 U4 [N5].
    returning = returning_slices(s.tune, s.fit)
    assert returning["returning_user"][1] == [0, 2, 3] and returning["new_user"][1] == [1]
    conc = concentration_slices(s.tune, train.news, min_history=2, edges=(0.5,))
    assert {k: rows for k, (_, rows) in conc.items()} == {
        "history_none": [],
        "history_1_1": [0, 3],
        "top_category_0_50": [1],  # 3/7 sports
        "top_category_50_100": [2],  # 2/2 sports
    }


def test_coverage_matches_hand_counts(train: MindDir) -> None:
    s = assign_splits(train.impressions, ())
    coverage, ties = coverage_and_ties(s.tune, s.fit, train.news, {"min_df": 1})
    assert coverage["candidates_per_impression"] == pytest.approx(10 / 4)
    assert coverage["share_impressions_user_in_statistics"] == pytest.approx(3 / 4)
    assert coverage["share_users_in_statistics"] == pytest.approx(3 / 4)
    assert coverage["share_impressions_no_known_history"] == 0
    assert coverage["share_candidates_unclicked_before"] == pytest.approx(6 / 10)
    # The fit corpus covers N1-N13; tune shows N14 twice.
    assert coverage["share_candidates_outside_tfidf_corpus"] == pytest.approx(2 / 10)
    assert set(ties) == {"popularity", "category_affinity", "tfidf"}
    assert ties["popularity"]["n_impressions"] == 4


def test_tie_measures_hand_case() -> None:
    scores = [np.array([1.0, 1.0, 2.0]), np.array([3.0, 2.0])]
    clicked = [np.array([True, False, False]), np.array([True, False])]
    out = _tie_measures(scores, clicked)
    assert out["share_pairs_tied"] == pytest.approx((1 / 3 + 0) / 2)
    assert out["share_clicks_tied_with_a_non_click"] == pytest.approx(1 / 2)


def _run(gap: float, *, fixed: bool = True, ret: float = 0.0, new: float = 0.0) -> dict[str, Any]:
    def block(g: float, n: int = 10) -> dict[str, Any]:
        return {"n_impressions": n, "gap": {"mean": g}}

    return {
        **block(gap),
        "fixed_targets": fixed,
        "history_length": {"0": block(0.0, 1), "1+": block(gap, 9)},
        "slices": {"returning_user": block(ret), "new_user": block(new)},
    }


@pytest.mark.parametrize(
    ("other_gap", "ret", "new", "corpus", "population"),
    [
        (
            0.012,
            0.02,
            -0.04,
            "minor",
            "sufficient alone",
        ),  # corpus moves it away; new users drive it
        (-0.02, 0.01, 0.01, "sufficient alone", "minor"),  # a smaller corpus moves it halfway
        (-0.007, 0.012, -0.003, "partial", "partial"),
    ],
)
def test_verdict_rules(
    other_gap: float, ret: float, new: float, corpus: str, population: str
) -> None:
    runs = {"d/all": _run(0.01, ret=ret, new=new), "d/1d": _run(other_gap)}
    reference = {
        "gap": {"mean": -0.03},
        "history_length": {"0": {"n_impressions": 1}, "1+": {"n_impressions": 9}},
    }
    out = verdicts(runs, "d", 0.9, reference)
    assert out["swing"] == pytest.approx(-0.04)
    assert out["corpus_size"]["label"] == corpus
    assert out["returning_vs_new"]["label"] == population
    assert out["history_length_mix"]["label"] == "minor"  # same composition as tune


def test_verdicts_need_a_reference() -> None:
    assert verdicts({"d/all": _run(0.01)}, "d", 0.9, None) == {"available": False}


def test_report_is_written_without_raw_records(
    results: dict[str, Any], config: BaselineDiagnosticsConfig
) -> None:
    for path in write_baseline_diagnostics(results, config):
        text = path.read_text()
        assert not re.search(r"\bU\d+\b", text), f"user ID in {path}"
        assert "[synthetic]" not in text, f"article text in {path}"
    report = config.report.read_text()
    for heading in ("## 1. Data coverage", "## 2. Score ties", "## 3. History length",
                    "## 4. History concentration", "### 5d."):  # fmt: skip
        assert heading in report
    assert "holdout (ref.)" in report and "never to choose a model" in report
