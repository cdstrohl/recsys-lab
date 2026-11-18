"""E2 hybrid on the synthetic fixtures: folds, leakage guards, features, rules, outputs."""

from __future__ import annotations

import dataclasses
import re
import shutil
from datetime import date
from typing import Any

import numpy as np
import pytest
from conftest import SYNTHETIC

from recsys.evaluate import EvalConfig, ModelSpec
from recsys.hybrid import (
    BASE_FEATURES,
    HybridConfig,
    HybridFeatures,
    HybridRanker,
    build_day,
    fit_linear,
    keep_extra,
    leave_one_day_out,
    run_hybrid_development,
    success,
    training_rows,
    write_hybrid,
)
from recsys.loaders import TRAIN_DIR, MindDir
from recsys.rankers import RankRequest
from recsys.schemas import Candidate
from recsys.splits import Split, SplitAssignment, assign_splits

TFIDF = {"min_df": 1}
DAYS = (date(2019, 11, 10), date(2019, 11, 11))


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> HybridConfig:
    train_only = tmp_path_factory.mktemp("train_only")
    shutil.copytree(SYNTHETIC / TRAIN_DIR, train_only / TRAIN_DIR)
    shutil.copy(SYNTHETIC / "PROVENANCE.json", train_only / "PROVENANCE.json")
    out = tmp_path_factory.mktemp("out")
    return HybridConfig(
        base=EvalConfig(
            data_dir=train_only,
            models=(
                ModelSpec("random"),
                ModelSpec("popularity"),
                ModelSpec("popularity_48h", {"window_hours": 48}, "popularity"),
                ModelSpec("category_affinity"),
                ModelSpec("tfidf", TFIDF),
            ),
            splits=(Split.TUNE,),
            history_edges=(0, 1, 6),
            compare_to=("category_affinity", "tfidf"),
            bootstrap_samples=50,
        ),
        days=DAYS,
        C=1.0,
        sensitivity_C=(0.01,),
        tfidf_model="tfidf",
        results=out / "hybrid_development.json",
        report=out / "hybrid_development.md",
    )


@pytest.fixture(scope="module")
def results(config: HybridConfig) -> dict[str, Any]:
    original = SplitAssignment.holdout

    def forbidden(self: SplitAssignment, *, allow_holdout: bool = False) -> Any:
        raise AssertionError("E2 development touched the holdout")

    SplitAssignment.holdout = forbidden  # type: ignore[method-assign]
    try:
        return run_hybrid_development(config)
    finally:
        SplitAssignment.holdout = original  # type: ignore[method-assign]


@pytest.fixture(scope="module")
def days(train: MindDir) -> list[Any]:
    return [build_day(train.impressions, d, train.news, TFIDF) for d in DAYS]


def test_runs_without_dev_and_records_the_folds(results: dict[str, Any]) -> None:
    assert results["holdout_used"] is False
    assert results["data_provenance"] == ["synthetic"]
    folds = {f["day"]: f for f in results["folds"]}
    assert folds["2019-11-10"]["n_statistics_impressions"] == 3  # 11-09 only
    assert folds["2019-11-11"]["n_statistics_impressions"] == 7
    assert folds["2019-11-10"]["lodo_weights_fitted_on"] == ["2019-11-11"]
    assert folds["2019-11-11"]["lodo_weights_fitted_on"] == ["2019-11-10"]
    assert folds["2019-11-11"]["forward_weights_fitted_on"] == ["2019-11-10"]


def test_live_scoring_matches_the_cached_features(results: dict[str, Any]) -> None:
    assert all(day["live_matches_cached"] for day in results["per_day"].values())


def test_pooled_counts_add_up(results: dict[str, Any]) -> None:
    per_day = results["per_day"].values()
    assert results["pooled"]["n_impressions"] == sum(d["n_impressions"] for d in per_day) == 8
    hybrid = results["pooled"]["models"]["hybrid"]["metrics"]["ndcg@10"]
    weighted = sum(
        d["models"]["hybrid"]["metrics"]["ndcg@10"]["mean"] * d["n_impressions"] for d in per_day
    )
    assert hybrid["mean"] == pytest.approx(weighted / 8)


def test_leave_one_day_out_never_trains_on_the_held_out_day(days: list[Any]) -> None:
    models = leave_one_day_out(days, BASE_FEATURES, C=1.0)
    for held, other in ((days[0], days[1]), (days[1], days[0])):
        assert models[held.day].n_rows == len(training_rows([other], BASE_FEATURES)[1])


def test_training_rows_skip_fallback_impressions(days: list[Any]) -> None:
    # 11-10: impression 7 (U2) has an empty History; 4, 5, 6 have 12 + 2 + 1 candidates.
    _, y = training_rows([days[0]], BASE_FEATURES)
    assert len(y) == 15


def test_features_match_hand_values(train: MindDir) -> None:
    s = assign_splits(train.impressions, ())
    features = HybridFeatures(s.fit, train.news, TFIDF)
    request = RankRequest.from_impression(s.tune[2])  # 10: U1 [N1 N2], candidates N13 N3
    x, available = features.matrix(request, BASE_FEATURES)
    assert available.all()
    col = {c: x[:, j] for j, c in enumerate(BASE_FEATURES)}
    # N3 was clicked at 1 and 7 in fit, N13 never; N13 is sports (2 + 0 of 2 History items).
    np.testing.assert_allclose(col["pop_item"], [0.0, np.log1p(2)])
    np.testing.assert_allclose(col["affinity"], [1.0, 0.0])
    np.testing.assert_allclose(col["tfidf_x_concentration"], col["tfidf"])  # all-sports History


def _ranker(train: MindDir) -> tuple[HybridRanker, Any]:
    s = assign_splits(train.impressions, ())
    features = HybridFeatures(s.fit, train.news, TFIDF)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(BASE_FEATURES)))
    model = fit_linear(x, (x[:, 0] > 0).astype(int), BASE_FEATURES, C=1.0)
    return HybridRanker("hybrid", features, model), s


def test_hybrid_ignores_labels_and_candidate_order(train: MindDir) -> None:
    ranker, s = _ranker(train)
    imp = s.tune[3]  # 11: U4, candidates N13 N14 N3
    flipped = dataclasses.replace(
        imp, candidates=tuple(Candidate(c.news_id, not c.clicked) for c in imp.candidates)
    )
    base = ranker.score(RankRequest.from_impression(imp)).scores
    np.testing.assert_array_equal(base, ranker.score(RankRequest.from_impression(flipped)).scores)
    reordered = dataclasses.replace(imp, candidates=imp.candidates[::-1])
    np.testing.assert_array_equal(
        base[::-1], ranker.score(RankRequest.from_impression(reordered)).scores
    )


def test_cold_start_falls_back_to_popularity(train: MindDir) -> None:
    ranker, s = _ranker(train)
    cold = dataclasses.replace(s.tune[0], history=())
    request = RankRequest.from_impression(cold)
    ranking = ranker.score(request)
    assert ranking.fallback
    np.testing.assert_array_equal(ranking.scores, ranker._features.popularity.score(request).scores)


def test_fit_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both clicked and unclicked"):
        fit_linear(np.zeros((3, 2)), [0, 0, 0], ("a", "b"), C=1.0)


@pytest.mark.parametrize(
    ("base", "extra", "kept"),
    [
        ((0.30, 0.31, 0.32, 0.33), (0.31, 0.32, 0.31, 0.34), True),  # better on 3 of 4
        ((0.30, 0.31, 0.32, 0.33), (0.31, 0.30, 0.31, 0.34), False),  # better on 2 of 4
        ((0.30, 0.31, 0.32, 0.33), (0.30, 0.31, 0.32, 0.33), False),  # equal is not better
    ],
)
def test_keep_extra_rule(base: tuple[float, ...], extra: tuple[float, ...], kept: bool) -> None:
    assert keep_extra(base, extra) is kept


@pytest.mark.parametrize(
    ("hybrid", "ci_low", "ok"),
    [
        ((0.40, 0.41, 0.42, 0.43), 0.001, True),
        ((0.40, 0.41, 0.42, 0.10), 0.001, True),  # loses one day: still all but at most one
        ((0.40, 0.10, 0.42, 0.10), 0.001, False),  # loses two days
        ((0.40, 0.41, 0.42, 0.43), -0.001, False),  # pooled interval includes 0
    ],
)
def test_success_rule(hybrid: tuple[float, ...], ci_low: float, ok: bool) -> None:
    per_day = {
        str(i): {"hybrid": h, "category_affinity": 0.35, "tfidf": 0.36}
        for i, h in enumerate(hybrid)
    }
    delta = {"mean": 0.01, "ci_low": ci_low, "ci_high": 0.02}
    rivals = ("category_affinity", "tfidf")
    assert success(per_day, delta, "hybrid", rivals)["success"] is ok


def test_report_is_written_without_raw_records(
    results: dict[str, Any], config: HybridConfig
) -> None:
    for path in write_hybrid(results, config):
        text = path.read_text()
        assert not re.search(r"\bU\d+\b", text), f"user ID in {path}"
        assert "[synthetic]" not in text, f"article text in {path}"
    report = config.report.read_text()
    assert "no holdout data is loaded or scored" in report
    assert "### Success rule" in report and "`popularity_48h`" in report
