"""Ranker tests: label-free inputs, as-of popularity, cold-start fallback, order independence.

Hand-computed against the synthetic fixtures. Train clicks (fit + tune), by time:
N3 at 11/9 8:05, N6 and N8 at 11/9 9:15, N7, N10, N11, N12 on 11/10, N3 at 11/10
23:59:59 and 11/11 00:00, N13 twice at 11/11 7:20, N14 at 11/11 21:00.
Category clicks over all of train: news 4, sports 3, and 1 each for health, weather,
autos, entertainment and finance.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pytest

from recsys.loaders import MindDir
from recsys.rankers import (
    CategoryAffinityRanker,
    PopularityRanker,
    RandomRanker,
    Ranker,
    RankRequest,
    TfidfRanker,
    build_rankers,
    lexicographic_scores,
)
from recsys.schemas import Candidate, Impression, NewsItem
from recsys.splits import assign_splits

SEED = 13
ALL_MODELS: list[tuple[str, dict[str, object]]] = [
    ("random", {}),
    ("popularity", {}),
    ("category_affinity", {}),
    ("tfidf", {"min_df": 1}),
]


@pytest.fixture(scope="module")
def news(train: MindDir, dev: MindDir) -> dict[str, NewsItem]:
    return {**dev.news, **train.news}


@pytest.fixture(scope="module")
def holdout(train: MindDir, dev: MindDir) -> dict[str, RankRequest]:
    s = assign_splits(train.impressions, dev.impressions)
    return {
        imp.impression_id: RankRequest.from_impression(imp) for imp in s.holdout(allow_holdout=True)
    }


def make_request(
    candidates: Sequence[str],
    history: Sequence[str] = (),
    t: datetime = datetime(2019, 11, 12),
) -> RankRequest:
    return RankRequest("q", "U-q", t, tuple(history), tuple(candidates))


def flip(impressions: Sequence[Impression]) -> tuple[Impression, ...]:
    return tuple(
        dataclasses.replace(
            imp, candidates=tuple(Candidate(c.news_id, not c.clicked) for c in imp.candidates)
        )
        for imp in impressions
    )


def item(news_id: str, category: str, subcategory: str, title: str) -> NewsItem:
    return NewsItem(news_id, category, subcategory, title, "", "", "[]", "[]")


# --- Interface -------------------------------------------------------------------------------


def test_request_carries_no_labels(train: MindDir) -> None:
    fields = {f.name for f in dataclasses.fields(RankRequest)}
    assert fields == {"impression_id", "user_id", "timestamp", "history", "candidate_ids"}
    request = RankRequest.from_impression(train.impressions[0])
    assert request.candidate_ids == ("N3", "N4", "N5")


def test_lexicographic_scores_order_by_primary_then_secondary() -> None:
    # Primary [1, 0, 1, 0]; within primary 0 the secondary ties (5, 5); within 1, 0 < 2.
    scores = lexicographic_scores([1, 0, 1, 0], [0, 5, 2, 5])
    assert scores.tolist() == [1.0, 0.0, 2.0, 0.0]
    assert lexicographic_scores([0.3, 0.1, 0.2]).tolist() == [2.0, 0.0, 1.0]


def test_unknown_model_raises(train: MindDir, news: dict[str, NewsItem]) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_rankers([("neural", {})], train.impressions, news, seed=SEED)


# --- Popularity ------------------------------------------------------------------------------


def test_popularity_counts_only_clicks_strictly_before(
    train: MindDir, news: dict[str, NewsItem]
) -> None:
    pop = PopularityRanker(train.impressions, news)
    # At impression 8's time (11/11 00:00): N3 clicks from 1 and 7 count, 8's own doesn't.
    items, _ = pop.keys(make_request(["N3"], t=datetime(2019, 11, 11)))
    assert items.tolist() == [2.0]
    # At 11/11 7:20 the two simultaneous N13 clicks (9, 10) don't count; sports has N8 only.
    items, categories = pop.keys(make_request(["N13"], t=datetime(2019, 11, 11, 7, 20)))
    assert items.tolist() == [0.0]
    assert categories.tolist() == [1.0]


def test_popularity_on_holdout_breaks_item_ties_by_category(
    train: MindDir, news: dict[str, NewsItem], holdout: dict[str, RankRequest]
) -> None:
    pop = PopularityRanker(train.impressions, news)
    # 101: N15 (sports, 0 clicks), N3 (news, 3), N16 (news, 0). Keys (0,3), (3,4), (0,4).
    assert pop.score(holdout["101"]).scores.tolist() == [0.0, 2.0, 1.0]
    # 102: N17 (0,0), N13 (2,3), N18 (health: 0,1), N99 (missing from news: 0,0).
    assert pop.score(holdout["102"]).scores.tolist() == [0.0, 2.0, 1.0, 0.0]


# --- Category affinity -----------------------------------------------------------------------


def test_category_affinity_ranks_history_categories_first(
    train: MindDir, news: dict[str, NewsItem], holdout: dict[str, RankRequest]
) -> None:
    ranker = CategoryAffinityRanker(PopularityRanker(train.impressions, news), news)
    # U1's History is N1, N2 (both sports). N15 is sports: affinity 2; N3, N16: 0, then
    # popularity breaks the tie (N3 has 3 clicks).
    ranking = ranker.score(holdout["101"])
    assert not ranking.fallback
    assert ranking.scores.tolist() == [2.0, 1.0, 0.0]


def test_category_affinity_sparse_history_rewards_subcategory(
    train: MindDir, news: dict[str, NewsItem]
) -> None:
    ranker = CategoryAffinityRanker(PopularityRanker(train.impressions, news), news)
    # One History item, N1 (sports/football_nfl). N1 matches category + subcategory (2),
    # N15 (sports/golf) category only (1), N3 (news) nothing (0).
    ranking = ranker.score(make_request(["N3", "N1", "N15"], history=["N1"]))
    assert not ranking.fallback
    assert ranking.scores.tolist() == [0.0, 2.0, 1.0]


@pytest.mark.parametrize("history", [(), ("N99",)], ids=["empty", "unknown_items_only"])
def test_category_affinity_cold_start_falls_back_to_popularity(
    train: MindDir, news: dict[str, NewsItem], history: tuple[str, ...]
) -> None:
    pop = PopularityRanker(train.impressions, news)
    ranker = CategoryAffinityRanker(pop, news)
    request = make_request(["N17", "N13", "N18", "N99"], history=history)
    ranking = ranker.score(request)
    assert ranking.fallback
    np.testing.assert_array_equal(ranking.scores, pop.score(request).scores)


# --- TF-IDF ----------------------------------------------------------------------------------


@pytest.fixture
def toy_news() -> dict[str, NewsItem]:
    return {
        "A": item("A", "sports", "golf", "golf course storm"),
        "B": item("B", "news", "newsus", "council budget vote"),
        "C": item("C", "sports", "golf", "golf club title"),
        "Z": item("Z", "news", "newsus", "zebra quokka"),  # not in the fitted corpus
    }


def test_tfidf_ranks_by_similarity_to_history(toy_news: dict[str, NewsItem]) -> None:
    pop = PopularityRanker((), toy_news)
    ranker = TfidfRanker(pop, toy_news, ["A", "B", "C"], min_df=1)
    # History A shares "golf" with C and nothing with B.
    ranking = ranker.score(make_request(["B", "C"], history=["A"]))
    assert not ranking.fallback
    assert ranking.scores[1] > ranking.scores[0]


def test_tfidf_corpus_excludes_unlisted_items(toy_news: dict[str, NewsItem]) -> None:
    ranker = TfidfRanker(PopularityRanker((), toy_news), toy_news, ["A", "B", "C"], min_df=1)
    assert "zebra" not in ranker._vectorizer.vocabulary_
    assert ranker.vocabulary_size == 8


@pytest.mark.parametrize(
    "history", [(), ("Z",), ("missing",)], ids=["empty", "out_of_vocabulary", "unknown_item"]
)
def test_tfidf_without_usable_history_falls_back_to_popularity(
    toy_news: dict[str, NewsItem], history: tuple[str, ...]
) -> None:
    pop = PopularityRanker((), toy_news)
    ranker = TfidfRanker(pop, toy_news, ["A", "B", "C"], min_df=1)
    request = make_request(["A", "B", "C"], history=history)
    ranking = ranker.score(request)
    assert ranking.fallback
    np.testing.assert_array_equal(ranking.scores, pop.score(request).scores)


# --- Properties shared by every ranker -------------------------------------------------------


def all_rankers(stats: Sequence[Impression], news: dict[str, NewsItem]) -> list[Ranker]:
    return build_rankers(ALL_MODELS, stats, news, seed=SEED)


def test_cold_start_user_gets_popularity_order_from_personalized_rankers(
    train: MindDir, news: dict[str, NewsItem], holdout: dict[str, RankRequest]
) -> None:
    rankers = {r.name: r for r in all_rankers(train.impressions, news)}
    request = holdout["102"]  # U7: empty History
    expected = rankers["popularity"].score(request).scores
    for name in ("category_affinity", "tfidf"):
        ranking = rankers[name].score(request)
        assert ranking.fallback, name
        np.testing.assert_array_equal(ranking.scores, expected)


def test_scores_follow_candidates_not_their_positions(
    train: MindDir, news: dict[str, NewsItem], holdout: dict[str, RankRequest]
) -> None:
    for ranker in all_rankers(train.impressions, news):
        for request in holdout.values():
            reversed_request = dataclasses.replace(
                request, candidate_ids=request.candidate_ids[::-1]
            )
            forward = ranker.score(request).scores
            backward = ranker.score(reversed_request).scores
            np.testing.assert_array_equal(backward, forward[::-1], err_msg=ranker.name)


def test_flipping_tune_and_holdout_labels_leaves_scores_unchanged(
    train: MindDir, dev: MindDir, news: dict[str, NewsItem]
) -> None:
    base = assign_splits(train.impressions, dev.impressions)
    perturbed = assign_splits(base.fit + flip(base.tune), flip(base.holdout(allow_holdout=True)))
    targets = [
        RankRequest.from_impression(imp) for imp in base.tune + base.holdout(allow_holdout=True)
    ]
    for a, b in zip(
        all_rankers(base.statistics_source(), news),
        all_rankers(perturbed.statistics_source(), news),
        strict=True,
    ):
        for request in targets:
            np.testing.assert_array_equal(a.score(request).scores, b.score(request).scores)


def test_flipping_holdout_labels_leaves_refit_scores_unchanged(
    train: MindDir, dev: MindDir, news: dict[str, NewsItem]
) -> None:
    base = assign_splits(train.impressions, dev.impressions)
    perturbed = assign_splits(train.impressions, flip(dev.impressions))
    targets = [RankRequest.from_impression(imp) for imp in base.holdout(allow_holdout=True)]
    for a, b in zip(
        all_rankers(base.statistics_source(include_tune=True), news),
        all_rankers(perturbed.statistics_source(include_tune=True), news),
        strict=True,
    ):
        for request in targets:
            np.testing.assert_array_equal(a.score(request).scores, b.score(request).scores)


def test_random_ranker_is_seeded() -> None:
    request = make_request([f"N{i}" for i in range(20)])
    first = RandomRanker(SEED).score(request).scores
    np.testing.assert_array_equal(RandomRanker(SEED).score(request).scores, first)
    assert not np.array_equal(RandomRanker(SEED + 1).score(request).scores, first)
    assert np.unique(first).size == first.size
