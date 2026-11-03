"""Tune-only diagnostics on the synthetic fixtures: no dev access, consistency, slices, rule."""

from __future__ import annotations

import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import SYNTHETIC

from recsys.diagnostics import (
    DiagnosticsConfig,
    adoption,
    run_diagnostics,
    unseen_slices,
    write_diagnostics,
)
from recsys.evaluate import EvalConfig, ModelSpec, run
from recsys.loaders import TRAIN_DIR, MindDir
from recsys.splits import Split, SplitAssignment, assign_splits

TFIDF = {"min_df": 1}
MODELS = (
    ModelSpec("random"),
    ModelSpec("popularity"),
    ModelSpec("category_affinity"),
    ModelSpec("tfidf", TFIDF),
    ModelSpec("category_affinity_no_pop", {"tie_break": "none"}, "category_affinity"),
    ModelSpec("tfidf_no_pop", {**TFIDF, "tie_break": "none"}, "tfidf"),
    ModelSpec("popularity_24h", {"window_hours": 24}, "popularity"),
    ModelSpec("tfidf_aug", {**TFIDF, "augment_history": True}, "tfidf"),
)
STABILITY = ("random", "popularity", "category_affinity", "tfidf", "category_affinity_no_pop",
             "tfidf_no_pop")  # fmt: skip


@pytest.fixture(scope="module")
def train_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data dir holding MINDsmall_train only: the diagnostics must not need dev."""
    root = tmp_path_factory.mktemp("train_only")
    shutil.copytree(SYNTHETIC / TRAIN_DIR, root / TRAIN_DIR)
    shutil.copy(SYNTHETIC / "PROVENANCE.json", root / "PROVENANCE.json")
    return root


def make_config(data_dir: Path, out: Path) -> DiagnosticsConfig:
    base = EvalConfig(
        data_dir=data_dir,
        models=MODELS,
        splits=(Split.TUNE,),
        history_edges=(0, 1, 6),
        bootstrap_samples=50,
    )
    return DiagnosticsConfig(
        base=base,
        stability_days=(date(2019, 11, 10), date(2019, 11, 11)),
        stability_models=STABILITY,
        stability_compare_to=("popularity", "category_affinity", "category_affinity_no_pop"),
        ablation_models=("popularity", "category_affinity", "tfidf", "popularity_24h", "tfidf_aug"),
        ablation_compare_to=("popularity", "category_affinity", "tfidf"),
        results=out / "tune_diagnostics.json",
        report=out / "tune_diagnostics.md",
    )


@pytest.fixture(scope="module")
def results(
    train_only: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, Any], DiagnosticsConfig]:
    config = make_config(train_only, tmp_path_factory.mktemp("out"))
    original = SplitAssignment.holdout

    def forbidden(self: SplitAssignment, *, allow_holdout: bool = False) -> Any:
        raise AssertionError("diagnostics touched the holdout")

    SplitAssignment.holdout = forbidden  # type: ignore[method-assign]
    try:
        return run_diagnostics(config), config
    finally:
        SplitAssignment.holdout = original  # type: ignore[method-assign]


def test_runs_without_dev_and_records_it(results: tuple[dict[str, Any], DiagnosticsConfig]) -> None:
    out, _ = results
    assert out["holdout_used"] is False
    assert out["data_provenance"] == ["synthetic"]
    assert list(out["stability"]) == ["2019-11-10", "2019-11-11"]


def test_last_stability_day_matches_the_official_tune_run(
    results: tuple[dict[str, Any], DiagnosticsConfig],
) -> None:
    out, _ = results
    # Fitted on everything before 11-11 = fit, scored on 11-11 = tune.
    day = out["stability"]["2019-11-11"]
    assert day["n_statistics_impressions"] == 7 and day["n_impressions"] == 4
    official = run(
        EvalConfig(
            data_dir=SYNTHETIC,
            models=tuple(m for m in MODELS if m.name in STABILITY),
            splits=(Split.TUNE,),
            history_edges=(0, 1, 6),
            bootstrap_samples=50,
        )
    )["splits"]["tune"]
    for model in STABILITY:
        for metric, entry in official["models"][model]["metrics"].items():
            assert day["models"][model]["metrics"][metric]["mean"] == pytest.approx(
                entry["mean"]
            ), (model, metric)


def test_unclicked_slices_match_hand_counts(train: MindDir, dev: MindDir) -> None:
    s = assign_splits(train.impressions, dev.impressions)
    slices, share = unseen_slices(s.tune, s.fit)
    # Fit clicks: N3, N6, N7, N8, N10, N11, N12. Tune 8: N3 N14 N1 -> 2/3 unclicked;
    # 9: N6 N13 -> 1/2; 10: N13 N3 -> 1/2; 11: N13 N14 N3 -> 2/3. All in [50%, 75%).
    assert {k: len(rows) for k, (_, rows) in slices.items()} == {
        "unclicked_0_25": 0,
        "unclicked_25_50": 0,
        "unclicked_50_75": 4,
        "unclicked_75_100": 0,
    }
    assert share == pytest.approx(6 / 10)


def _ablations(overall: tuple[float, float], cold: tuple[float, float] | None) -> dict[str, Any]:
    def delta(ci: tuple[float, float]) -> dict[str, Any]:
        return {"delta_vs": {"base": {"mean": sum(ci) / 2, "ci_low": ci[0], "ci_high": ci[1]}}}

    cold_models = {"variant": {"metrics": {"ndcg@10": delta(cold)}}} if cold else {}
    return {
        "models": {"variant": {"metrics": {"ndcg@10": delta(overall)}}},
        "segments_by_history_length": {"0": {"models": cold_models}},
    }


@pytest.mark.parametrize(
    ("overall", "cold", "adopted"),
    [
        ((0.001, 0.01), (-0.02, 0.01), True),  # above 0; cold-start interval includes 0
        ((0.001, 0.01), None, True),  # no zero-History users
        ((-0.001, 0.01), (0.0, 0.01), False),  # overall interval includes 0
        ((0.001, 0.01), (-0.02, -0.001), False),  # hurts zero-History users
    ],
)
def test_adoption_rule(
    overall: tuple[float, float], cold: tuple[float, float] | None, adopted: bool
) -> None:
    assert adoption(_ablations(overall, cold), "variant", "base", "ndcg@10")[0] is adopted


def test_report_is_written_without_raw_records(
    results: tuple[dict[str, Any], DiagnosticsConfig],
) -> None:
    out, config = results
    for path in write_diagnostics(out, config):
        text = path.read_text()
        assert not re.search(r"\bU\d+\b", text), f"user ID in {path}"
        assert "[synthetic]" not in text, f"article text in {path}"
    report = config.report.read_text()
    assert "holdout is never loaded" in report
    assert "## 3. Ablations" in report and "`tfidf_aug`" in report
