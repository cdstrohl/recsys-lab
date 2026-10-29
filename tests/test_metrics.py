"""Metric tests with hand-computed expected values.

Notation: ranks are 1-based after sorting by score descending.
DCG = sum over clicked ranks r <= k of 1 / log2(r + 1)  (binary gains: 2^1 - 1 = 1).
Useful constants: 1/log2(2) = 1, 1/log2(3) = 0.6309298, 1/log2(4) = 0.5,
1/log2(6) = 0.3868528, 1/log2(8) = 1/3.
"""

from __future__ import annotations

import importlib.util
import os
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from recsys.loaders import MindDir
from recsys.metrics import (
    Eligibility,
    auc,
    eligibility,
    evaluate,
    mrr,
    ndcg_at_k,
    rank_order,
    tie_break_keys,
)
from recsys.schemas import Candidate, Impression

SEED = 13


def keys_for(n: int, impression_id: str = "imp") -> np.ndarray:
    return tie_break_keys(impression_id, [f"N{i}" for i in range(n)], SEED)


def make_impression(
    impression_id: str, labels: list[int], history: tuple[str, ...] = ("H1",)
) -> Impression:
    return Impression(
        impression_id=impression_id,
        user_id=f"U-{impression_id}",
        timestamp=datetime(2019, 11, 9),
        history=history,
        candidates=tuple(Candidate(f"N{i}", bool(y)) for i, y in enumerate(labels)),
    )


# --- Single impressions, no ties -------------------------------------------------------------


def test_two_clicks_out_of_five_unsorted_input() -> None:
    # Scores are given out of order; sorted descending the labels are 0 1 0 0 1,
    # i.e. clicks at ranks 2 and 5.
    labels = [1, 0, 0, 1, 0]
    scores = [0.5, 0.6, 0.9, 0.8, 0.7]
    keys = keys_for(5)
    # MRR = (1/2 + 1/5) / 2 = 0.35
    assert mrr(labels, scores, keys) == pytest.approx(0.35)
    # DCG@5 = 1/log2(3) + 1/log2(6) = 0.6309298 + 0.3868528 = 1.0177826
    # IDCG@5 = 1 + 1/log2(3) = 1.6309298  ->  nDCG@5 = 0.6240505
    assert ndcg_at_k(labels, scores, keys, 5) == pytest.approx(0.6240505, abs=1e-7)
    # nDCG@10 is the same: only 5 candidates, so both DCGs run over min(10, 5) = 5.
    assert ndcg_at_k(labels, scores, keys, 10) == pytest.approx(0.6240505, abs=1e-7)
    # AUC: clicked scores {0.8, 0.5} vs non-clicked {0.9, 0.7, 0.6}.
    # 0.8 beats 0.7 and 0.6 (2 pairs); 0.5 beats none. 2 / (2 * 3) = 1/3
    assert auc(labels, scores) == pytest.approx(1 / 3)


def test_single_click_at_top_of_three() -> None:
    labels, scores, keys = [1, 0, 0], [3.0, 2.0, 1.0], keys_for(3)
    # Click at rank 1: MRR = 1, DCG = IDCG = 1, AUC = 2/2 = 1.
    assert mrr(labels, scores, keys) == 1.0
    assert ndcg_at_k(labels, scores, keys, 10) == 1.0
    assert auc(labels, scores) == 1.0


def test_fewer_than_k_candidates() -> None:
    labels, scores, keys = [0, 0, 1], [3.0, 2.0, 1.0], keys_for(3)
    # Click at rank 3 of 3. MRR = 1/3.
    assert mrr(labels, scores, keys) == pytest.approx(1 / 3)
    # DCG = 1/log2(4) = 0.5; IDCG uses min(k, 3) = 3 candidates: 1/log2(2) = 1.
    # nDCG@5 = nDCG@10 = 0.5 (not penalized for the missing positions 4..10).
    assert ndcg_at_k(labels, scores, keys, 5) == pytest.approx(0.5)
    assert ndcg_at_k(labels, scores, keys, 10) == pytest.approx(0.5)
    # AUC: the clicked item is below both non-clicked items: 0/2 = 0.
    assert auc(labels, scores) == 0.0


def test_twelve_candidates_click_below_cutoff_five() -> None:
    labels = [0] * 12
    labels[6] = 1
    scores = [float(12 - i) for i in range(12)]  # already sorted; click at rank 7
    keys = keys_for(12)
    # MRR = 1/7.
    assert mrr(labels, scores, keys) == pytest.approx(1 / 7)
    # nDCG@5: rank 7 > 5, DCG@5 = 0  ->  0.
    assert ndcg_at_k(labels, scores, keys, 5) == 0.0
    # nDCG@10: DCG = 1/log2(8) = 1/3, IDCG = 1  ->  1/3.
    assert ndcg_at_k(labels, scores, keys, 10) == pytest.approx(1 / 3)
    # AUC: the click beats the 5 candidates ranked 8..12 out of 11 non-clicked: 5/11.
    assert auc(labels, scores) == pytest.approx(5 / 11)


def test_multiple_clicks() -> None:
    labels, scores, keys = [1, 0, 1, 0], [4.0, 3.0, 2.0, 1.0], keys_for(4)
    # Clicks at ranks 1 and 3. MRR = (1 + 1/3) / 2 = 2/3.
    assert mrr(labels, scores, keys) == pytest.approx(2 / 3)
    # DCG@5 = 1 + 1/log2(4) = 1.5; IDCG@5 = 1 + 1/log2(3) = 1.6309298  ->  0.9197208
    assert ndcg_at_k(labels, scores, keys, 5) == pytest.approx(0.9197208, abs=1e-7)
    # AUC: rank 1 beats ranks 2 and 4 (2); rank 3 beats rank 4 (1). 3 / (2 * 2) = 0.75
    assert auc(labels, scores) == pytest.approx(0.75)


def test_mind_mrr_is_below_one_even_for_a_perfect_ranking() -> None:
    labels, scores, keys = [1, 1, 0], [3.0, 2.0, 1.0], keys_for(3)
    # Clicks at ranks 1 and 2: MIND-style MRR = (1 + 1/2) / 2 = 0.75, not 1.
    assert mrr(labels, scores, keys) == pytest.approx(0.75)
    assert ndcg_at_k(labels, scores, keys, 10) == 1.0
    assert auc(labels, scores) == 1.0


# --- No positives, all positives -------------------------------------------------------------


def test_no_click_raises_for_every_metric() -> None:
    labels, scores, keys = [0, 0, 0], [3.0, 2.0, 1.0], keys_for(3)
    assert eligibility(labels) is Eligibility.NO_CLICK
    with pytest.raises(ValueError):
        auc(labels, scores)
    with pytest.raises(ValueError):
        mrr(labels, scores, keys)
    with pytest.raises(ValueError):
        ndcg_at_k(labels, scores, keys, 10)


def test_all_click_is_score_independent_for_mrr_and_ndcg() -> None:
    labels = [1, 1, 1]
    assert eligibility(labels) is Eligibility.ALL_CLICK
    with pytest.raises(ValueError):
        auc(labels, [3.0, 2.0, 1.0])
    # Any ordering puts clicks at ranks 1, 2, 3: MRR = (1 + 1/2 + 1/3) / 3 = 11/18 = 0.6111111
    # and DCG = IDCG, so nDCG = 1. Reversing the scores changes nothing.
    for scores in ([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]):
        assert mrr(labels, scores, keys_for(3)) == pytest.approx(11 / 18)
        assert ndcg_at_k(labels, scores, keys_for(3), 10) == 1.0


# --- Ties ------------------------------------------------------------------------------------


def test_auc_counts_tied_pairs_as_half() -> None:
    # Clicked {0.5, 0.9} vs non-clicked {0.5, 0.1}.
    # 0.5 vs 0.5 tie (0.5), 0.5 > 0.1 (1), 0.9 > both (2). 3.5 / 4 = 0.875
    assert auc([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1]) == pytest.approx(0.875)
    # All scores equal: every pair is a tie, AUC = 0.5 regardless of labels.
    assert auc([1, 0, 0, 1], [7.0, 7.0, 7.0, 7.0]) == 0.5


def test_partial_tie_follows_the_seeded_key() -> None:
    labels, scores = [0, 1, 0], [2.0, 1.0, 1.0]
    keys = keys_for(3)
    # Candidate 0 is rank 1. Candidates 1 (clicked) and 2 tie; the lower key ranks first,
    # so the click is at rank 2 (MRR 1/2) or rank 3 (MRR 1/3).
    expected = 1 / 2 if keys[1] < keys[2] else 1 / 3
    assert mrr(labels, scores, keys) == pytest.approx(expected)
    assert rank_order(scores, keys)[0] == 0


def test_tie_break_is_independent_of_candidate_order() -> None:
    news_ids = ["N1", "N2", "N3", "N4"]
    labels = np.array([1, 0, 0, 1])
    scores = np.array([1.0, 1.0, 1.0, 0.5])
    perm = np.array([3, 1, 0, 2])
    keys = tie_break_keys("imp-7", news_ids, SEED)
    keys_perm = tie_break_keys("imp-7", [news_ids[i] for i in perm], SEED)
    assert mrr(labels, scores, keys) == mrr(labels[perm], scores[perm], keys_perm)
    assert ndcg_at_k(labels, scores, keys, 10) == ndcg_at_k(
        labels[perm], scores[perm], keys_perm, 10
    )


def test_tie_break_does_not_favor_the_first_candidate() -> None:
    # Two tied candidates, the first one clicked. A position-based tie-break would always
    # rank it first (MRR 1). The seeded hash ranks it first in roughly half of impressions.
    wins = sum(
        mrr([1, 0], [1.0, 1.0], tie_break_keys(f"imp-{i}", ["NA", "NB"], SEED)) == 1.0
        for i in range(400)
    )
    assert 140 < wins < 260
    # The seed matters: another seed changes the outcome for some impressions.
    other = [tie_break_keys(f"imp-{i}", ["NA", "NB"], SEED + 1) for i in range(400)]
    same = [tie_break_keys(f"imp-{i}", ["NA", "NB"], SEED) for i in range(400)]
    assert any((a[0] < a[1]) != (b[0] < b[1]) for a, b in zip(other, same, strict=True))


def test_tie_break_keys_are_deterministic() -> None:
    assert (keys_for(5) == keys_for(5)).all()


# --- Input validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([1, 2, 0], [1.0, 2.0, 3.0]),  # non-binary label
        ([], []),  # empty
        ([1, 0], [1.0]),  # length mismatch
        ([1, 0], [1.0, float("nan")]),  # NaN score
        ([1, 0], [1.0, float("inf")]),  # infinite score
    ],
)
def test_invalid_inputs_raise(labels: list[int], scores: list[float]) -> None:
    with pytest.raises(ValueError):
        auc(labels, scores)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([1, 0], [1.0, 0.0], keys_for(2), 0)


# --- Aggregation -----------------------------------------------------------------------------


def test_evaluate_averages_over_eligible_impressions() -> None:
    impressions = [
        make_impression("a", [1, 0, 0]),  # AUC 1, MRR 1, nDCG 1
        make_impression("b", [0, 0, 1], history=()),  # zero history: AUC 0, MRR 1/3, nDCG 0.5
        make_impression("c", [1, 1]),  # all-click: MRR (1 + 1/2)/2 = 0.75, nDCG 1, no AUC
        make_impression("d", [0, 0]),  # no-click: excluded from everything
    ]
    scores = [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]
    summary = evaluate(impressions, scores, seed=SEED)

    # AUC over a, b: (1 + 0) / 2 = 0.5
    # MRR over a, b, c: (1 + 1/3 + 3/4) / 3 = (12 + 4 + 9) / 36 = 25/36 = 0.6944444
    # nDCG@5 = nDCG@10 over a, b, c: (1 + 0.5 + 1) / 3 = 5/6 = 0.8333333
    assert summary.means["auc"] == pytest.approx(0.5)
    assert summary.means["mrr"] == pytest.approx(25 / 36)
    assert summary.means["ndcg@5"] == pytest.approx(5 / 6)
    assert summary.means["ndcg@10"] == pytest.approx(5 / 6)
    assert summary.n_eligible == {"auc": 2, "mrr": 3, "ndcg@5": 3, "ndcg@10": 3}
    assert (summary.n_impressions, summary.n_no_click, summary.n_all_click) == (4, 1, 1)
    assert summary.n_tied == 0
    assert summary.per_impression[3].values == {}


def test_evaluate_on_fixture_with_constant_scores(train: MindDir) -> None:
    # Every score is 0, so every multi-candidate impression is fully tied.
    scores = [[0.0] * len(imp.candidates) for imp in train.impressions]
    summary = evaluate(train.impressions, scores, seed=SEED)
    # 11 impressions: 3 is no-click; 5 and 6 are all-click; 8 remain for AUC.
    assert (summary.n_impressions, summary.n_no_click, summary.n_all_click) == (11, 1, 2)
    assert summary.n_eligible["auc"] == 8
    assert summary.n_eligible["mrr"] == 10
    # Fully tied scores give AUC exactly 0.5 per impression.
    assert summary.means["auc"] == 0.5
    # Tied among the 10 scored impressions: all but single-candidate impression 6.
    assert summary.n_tied == 9


def test_evaluate_rejects_misaligned_scores(train: MindDir) -> None:
    with pytest.raises(ValueError):
        evaluate(train.impressions, [[0.0]], seed=SEED)


def test_evaluate_reports_none_when_nothing_is_eligible() -> None:
    summary = evaluate([make_impression("x", [0, 0])], [[1.0, 0.0]], seed=SEED)
    assert summary.means["auc"] is None
    assert summary.n_eligible["mrr"] == 0


# --- Optional parity with MIND's official evaluation script ----------------------------------


def _load_official() -> ModuleType:
    path = os.environ.get("MIND_EVAL_PY")
    if not path:
        pytest.skip("set MIND_EVAL_PY to a local copy of MIND's evaluate.py to run parity")
    spec = importlib.util.spec_from_file_location("mind_official_eval", Path(path))
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # the official script imports scikit-learn
        pytest.skip(f"official script needs {exc.name}; run with `uv run --with scikit-learn`")
    return module


# Tie-free cases only: the official script breaks ties by argsort position, we don't.
PARITY_CASES = [
    ([1, 0, 0, 1, 0], [0.5, 0.6, 0.9, 0.8, 0.7]),
    ([0, 0, 1], [3.0, 2.0, 1.0]),
    ([1, 0, 1, 0], [4.0, 3.0, 2.0, 1.0]),
    ([0] * 6 + [1] + [0] * 5, [float(12 - i) for i in range(12)]),
]


@pytest.mark.parametrize(("labels", "scores"), PARITY_CASES)
def test_parity_with_official_mind_evaluation(labels: list[int], scores: list[float]) -> None:
    official = _load_official()
    y, s, keys = np.array(labels), np.array(scores), keys_for(len(labels))
    assert mrr(y, s, keys) == pytest.approx(official.mrr_score(y, s))
    for k in (5, 10):
        assert ndcg_at_k(y, s, keys, k) == pytest.approx(official.ndcg_score(y, s, k))
    assert auc(y, s) == pytest.approx(official.roc_auc_score(y, s))
