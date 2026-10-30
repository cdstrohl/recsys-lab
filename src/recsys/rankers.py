"""Rankers behind one scoring interface.

Every ranker scores a `RankRequest`: one impression's user, timestamp, History and
candidate IDs, with **no click labels**. A higher score ranks higher. Score ties
are broken later by the metrics' seeded key (D-008), never by candidate order.

Rankers are fitted from a statistics source (`SplitAssignment.statistics_source()`)
and news metadata only. See DECISIONS.md D-012 to D-015.

- `random`: seeded random scores (sanity floor).
- `popularity`: as-of clicks on the item, ties broken by as-of clicks on its category.
- `category_affinity`: share of the user's History in the candidate's category and
  subcategory, ties broken by popularity order.
- `tfidf`: cosine between the candidate's title+abstract TF-IDF vector and the mean
  of the user's History vectors, ties broken by popularity order.

The two personalized rankers fall back to the popularity order, and flag it, when
the user has no usable History (user cold start).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys.metrics import tie_break_keys
from recsys.schemas import Impression, NewsItem
from recsys.splits import AsOfClickIndex, ClickEvent, click_events

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RankRequest:
    """What a ranker may see about one impression. Labels are deliberately absent."""

    impression_id: str
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    candidate_ids: tuple[str, ...]

    @classmethod
    def from_impression(cls, impression: Impression) -> RankRequest:
        return cls(
            impression_id=impression.impression_id,
            user_id=impression.user_id,
            timestamp=impression.timestamp,
            history=impression.history,
            candidate_ids=tuple(c.news_id for c in impression.candidates),
        )


@dataclass(frozen=True, slots=True)
class Ranking:
    scores: FloatArray
    fallback: bool = False  # personalized signal unavailable; popularity order used


class Ranker(Protocol):
    name: str

    def score(self, request: RankRequest) -> Ranking: ...


def lexicographic_scores(*keys: npt.ArrayLike) -> FloatArray:
    """Dense-rank scores ordering candidates by keys[0], then keys[1], ... (higher is better).

    Two candidates tie only if every key is equal. The result is ordinal: 0 is the
    worst group, and each distinct key tuple above it adds 1.
    """
    if not keys:
        raise ValueError("need at least one key")
    columns = np.stack([np.asarray(k, dtype=np.float64) for k in keys], axis=1)
    if columns.shape[0] == 0:
        return np.zeros(0)
    # lexsort sorts by the last key first, so pass the primary key last.
    order = np.lexsort(columns.T[::-1])
    ranked = columns[order]
    new_group = np.r_[True, np.any(ranked[1:] != ranked[:-1], axis=1)]
    scores = np.empty(columns.shape[0])
    scores[order] = np.cumsum(new_group) - 1
    return scores


class RandomRanker:
    """Seeded random scores, independent of candidate order and of the metrics' tie keys."""

    name = "random"

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def score(self, request: RankRequest) -> Ranking:
        salted_id = f"random\x1f{request.impression_id}"
        keys = tie_break_keys(salted_id, request.candidate_ids, self.seed)
        return Ranking(keys.astype(np.float64))


class PopularityRanker:
    """As-of click counts on the item, ties broken by as-of click counts on its category.

    Only clicks strictly before the request's timestamp, from the statistics source,
    count (D-007). Items never clicked there score 0 on the first key, so the
    category key orders the cold-start items.
    """

    name = "popularity"

    def __init__(self, stats: Sequence[Impression], news: Mapping[str, NewsItem]) -> None:
        self._news = news
        events = click_events(stats)
        self._items = AsOfClickIndex(events)
        self._categories = AsOfClickIndex(
            ClickEvent(e.timestamp, news[e.news_id].category) for e in events if e.news_id in news
        )

    def keys(self, request: RankRequest) -> tuple[FloatArray, FloatArray]:
        """(item clicks, category clicks) per candidate; unknown items get 0 for both."""
        t = request.timestamp
        items = self._items.counts_before(request.candidate_ids, t)
        categories = [
            self._categories.count_before(self._news[n].category, t) if n in self._news else 0
            for n in request.candidate_ids
        ]
        return np.asarray(items, dtype=np.float64), np.asarray(categories, dtype=np.float64)

    def score(self, request: RankRequest) -> Ranking:
        return Ranking(lexicographic_scores(*self.keys(request)))


class CategoryAffinityRanker:
    """History share of the candidate's category plus that of its subcategory.

    For a History of n known items, a candidate scores (#same category + #same
    subcategory) / n; the shared 1/n is dropped since it doesn't change the order.
    Ties are broken by popularity order. History items missing from the news
    metadata are ignored; if none remain, the ranker falls back to popularity.
    """

    name = "category_affinity"

    def __init__(self, popularity: PopularityRanker, news: Mapping[str, NewsItem]) -> None:
        self._popularity = popularity
        self._news = news

    def score(self, request: RankRequest) -> Ranking:
        profile = [self._news[h] for h in request.history if h in self._news]
        if not profile:
            return Ranking(self._popularity.score(request).scores, fallback=True)
        categories = Counter(item.category for item in profile)
        subcategories = Counter((item.category, item.subcategory) for item in profile)
        affinity = [
            categories[item.category] + subcategories[(item.category, item.subcategory)]
            if (item := self._news.get(n)) is not None
            else 0
            for n in request.candidate_ids
        ]
        return Ranking(lexicographic_scores(affinity, *self._popularity.keys(request)))


def news_text(item: NewsItem) -> str:
    return f"{item.title} {item.abstract}".strip()


class TfidfRanker:
    """Cosine similarity between candidate and user TF-IDF vectors over title + abstract.

    The vocabulary and IDF are fitted on the news items referenced (History or
    candidate) by the statistics source, never on the scored split. Item vectors are
    L2-normalized; the user vector is their mean over known History items. Ties
    (e.g. zero similarity) are broken by popularity order. A user with no known
    History item, or whose History has no in-vocabulary term, falls back to popularity.
    """

    name = "tfidf"

    def __init__(
        self,
        popularity: PopularityRanker,
        news: Mapping[str, NewsItem],
        corpus_ids: Iterable[str],
        **vectorizer_params: Any,
    ) -> None:
        self._popularity = popularity
        corpus = sorted({n for n in corpus_ids if n in news})
        if not corpus:
            raise ValueError("TF-IDF needs at least one known news item to fit on")
        self._vectorizer = TfidfVectorizer(**vectorizer_params)
        self._vectorizer.fit([news_text(news[n]) for n in corpus])
        ids = sorted(news)
        self._row = {n: i for i, n in enumerate(ids)}
        self._matrix = self._vectorizer.transform([news_text(news[n]) for n in ids]).tocsr()
        self._user_cache: dict[tuple[str, ...], FloatArray | None] = {}

    @property
    def vocabulary_size(self) -> int:
        return len(self._vectorizer.vocabulary_)

    def user_vector(self, history: tuple[str, ...]) -> FloatArray | None:
        """Unit-length mean of the History item vectors, or None if it would be zero."""
        if history not in self._user_cache:
            rows = [self._row[h] for h in history if h in self._row]
            vector: FloatArray | None = None
            if rows:
                mean = np.asarray(self._matrix[rows].mean(axis=0)).ravel()
                norm = float(np.linalg.norm(mean))
                vector = mean / norm if norm > 0 else None
            self._user_cache[history] = vector
        return self._user_cache[history]

    def score(self, request: RankRequest) -> Ranking:
        user = self.user_vector(request.history)
        if user is None:
            return Ranking(self._popularity.score(request).scores, fallback=True)
        similarity = np.zeros(len(request.candidate_ids))
        known = [i for i, n in enumerate(request.candidate_ids) if n in self._row]
        if known:
            rows = [self._row[request.candidate_ids[i]] for i in known]
            similarity[known] = self._matrix[rows] @ user
        return Ranking(lexicographic_scores(similarity, *self._popularity.keys(request)))


MODEL_NAMES = (
    RandomRanker.name,
    PopularityRanker.name,
    CategoryAffinityRanker.name,
    TfidfRanker.name,
)


def build_rankers(
    specs: Sequence[tuple[str, Mapping[str, Any]]],
    stats: Sequence[Impression],
    news: Mapping[str, NewsItem],
    *,
    seed: int,
) -> list[Ranker]:
    """Fit the named rankers from `stats` (a statistics source) and news metadata only."""
    popularity = PopularityRanker(stats, news)
    rankers: list[Ranker] = []
    for name, params in specs:
        if name == RandomRanker.name:
            rankers.append(RandomRanker(seed))
        elif name == PopularityRanker.name:
            rankers.append(popularity)
        elif name == CategoryAffinityRanker.name:
            rankers.append(CategoryAffinityRanker(popularity, news))
        elif name == TfidfRanker.name:
            corpus = {n for imp in stats for n in imp.history}
            corpus.update(c.news_id for imp in stats for c in imp.candidates)
            rankers.append(TfidfRanker(popularity, news, corpus, **params))
        else:
            raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    return rankers
