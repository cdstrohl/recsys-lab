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

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
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

    def __init__(self, seed: int, *, name: str = "random") -> None:
        self.name = name
        self.seed = seed

    def score(self, request: RankRequest) -> Ranking:
        salted_id = f"random\x1f{request.impression_id}"
        keys = tie_break_keys(salted_id, request.candidate_ids, self.seed)
        return Ranking(keys.astype(np.float64))


class PopularityRanker:
    """As-of click counts on the item, ties broken by as-of click counts on its category.

    Only clicks strictly before the request's timestamp, from the statistics source,
    count (D-007). With `window_hours`, only clicks in the last `window_hours` before
    the request count, for both keys (D-020). Items never clicked there score 0 on the
    first key, so the category key orders the cold-start items.
    """

    name = "popularity"

    def __init__(
        self,
        stats: Sequence[Impression],
        news: Mapping[str, NewsItem],
        *,
        window_hours: float | None = None,
        name: str = "popularity",
    ) -> None:
        self.name = name
        self._news = news
        self._window = timedelta(hours=window_hours) if window_hours is not None else None
        events = click_events(stats)
        self._items = AsOfClickIndex(events)
        self._categories = AsOfClickIndex(
            ClickEvent(e.timestamp, news[e.news_id].category) for e in events if e.news_id in news
        )

    def keys(self, request: RankRequest) -> tuple[FloatArray, FloatArray]:
        """(item clicks, category clicks) per candidate; unknown items get 0 for both."""
        t = request.timestamp
        since = t - self._window if self._window is not None else None
        items = self._items.counts_before(request.candidate_ids, t, since=since)
        categories = [
            self._categories.count_before(self._news[n].category, t, since=since)
            if n in self._news
            else 0
            for n in request.candidate_ids
        ]
        return np.asarray(items, dtype=np.float64), np.asarray(categories, dtype=np.float64)

    def score(self, request: RankRequest) -> Ranking:
        return Ranking(lexicographic_scores(*self.keys(request)))


class HistoryAugmenter:
    """The user's History plus their own clicks in the statistics source before the request.

    Clicks strictly before the request's timestamp are appended in time order, skipping
    items already present (D-020). History itself carries no timestamps and is kept
    as given. Never built from the scored split.
    """

    def __init__(self, stats: Iterable[Impression]) -> None:
        by_user: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        for imp in stats:
            by_user[imp.user_id].extend(
                (imp.timestamp, c.news_id) for c in imp.candidates if c.clicked
            )
        self._times: dict[str, list[datetime]] = {}
        self._items: dict[str, list[str]] = {}
        for user, events in by_user.items():
            events.sort()
            self._times[user] = [t for t, _ in events]
            self._items[user] = [n for _, n in events]

    def history(self, request: RankRequest) -> tuple[str, ...]:
        times = self._times.get(request.user_id)
        if not times:
            return request.history
        earlier = self._items[request.user_id][: bisect_left(times, request.timestamp)]
        seen = set(request.history)
        extra = []
        for news_id in earlier:
            if news_id not in seen:
                seen.add(news_id)
                extra.append(news_id)
        return request.history + tuple(extra)


class _Personalized:
    """Shared plumbing: History source, popularity tie-break and cold-start fallback.

    `tie_break=False` leaves ties among equal personalized scores to the metrics'
    seeded key (a diagnostic variant, D-020); the cold-start fallback still uses
    popularity, so variants differ only for users with usable History.
    """

    def __init__(
        self,
        popularity: PopularityRanker,
        *,
        tie_break: bool = True,
        augmenter: HistoryAugmenter | None = None,
    ) -> None:
        self._popularity = popularity
        self._tie_break = tie_break
        self._augmenter = augmenter

    def _history(self, request: RankRequest) -> tuple[str, ...]:
        return self._augmenter.history(request) if self._augmenter else request.history

    def _fallback(self, request: RankRequest) -> Ranking:
        return Ranking(self._popularity.score(request).scores, fallback=True)

    def _rank(self, request: RankRequest, primary: npt.ArrayLike) -> Ranking:
        if not self._tie_break:
            return Ranking(lexicographic_scores(primary))
        return Ranking(lexicographic_scores(primary, *self._popularity.keys(request)))


class CategoryAffinityRanker(_Personalized):
    """History share of the candidate's category plus that of its subcategory.

    For a History of n known items, a candidate scores (#same category + #same
    subcategory) / n; the shared 1/n is dropped since it doesn't change the order.
    Ties are broken by popularity order. History items missing from the news
    metadata are ignored; if none remain, the ranker falls back to popularity.
    """

    name = "category_affinity"

    def __init__(
        self,
        popularity: PopularityRanker,
        news: Mapping[str, NewsItem],
        *,
        name: str = "category_affinity",
        tie_break: bool = True,
        augmenter: HistoryAugmenter | None = None,
    ) -> None:
        super().__init__(popularity, tie_break=tie_break, augmenter=augmenter)
        self.name = name
        self._news = news

    def profile(self, request: RankRequest) -> list[NewsItem]:
        """The known items of the (possibly augmented) History."""
        return [self._news[h] for h in self._history(request) if h in self._news]

    def affinity(self, request: RankRequest) -> FloatArray | None:
        """#same category + #same subcategory per candidate; None without known History."""
        profile = self.profile(request)
        if not profile:
            return None
        categories = Counter(item.category for item in profile)
        subcategories = Counter((item.category, item.subcategory) for item in profile)
        return np.array(
            [
                categories[item.category] + subcategories[(item.category, item.subcategory)]
                if (item := self._news.get(n)) is not None
                else 0
                for n in request.candidate_ids
            ],
            dtype=np.float64,
        )

    def score(self, request: RankRequest) -> Ranking:
        affinity = self.affinity(request)
        if affinity is None:
            return self._fallback(request)
        return self._rank(request, affinity)


def top_category_share(profile: Sequence[NewsItem]) -> float | None:
    """Share of the profile in its most common category; None for an empty profile."""
    if not profile:
        return None
    return Counter(item.category for item in profile).most_common(1)[0][1] / len(profile)


def news_text(item: NewsItem) -> str:
    return f"{item.title} {item.abstract}".strip()


class TfidfIndex:
    """TF-IDF item vectors and cached user vectors, shared by every TF-IDF ranker variant.

    The vocabulary and IDF are fitted on `corpus_ids` (the news items referenced by
    the statistics source), never on the scored split. Item rows are L2-normalized.
    """

    def __init__(
        self, news: Mapping[str, NewsItem], corpus_ids: Iterable[str], **vectorizer_params: Any
    ) -> None:
        corpus = sorted({n for n in corpus_ids if n in news})
        if not corpus:
            raise ValueError("TF-IDF needs at least one known news item to fit on")
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        self.vectorizer.fit([news_text(news[n]) for n in corpus])
        ids = sorted(news)
        self.row = {n: i for i, n in enumerate(ids)}
        self.matrix = self.vectorizer.transform([news_text(news[n]) for n in ids]).tocsr()
        self._user_cache: dict[tuple[str, ...], FloatArray | None] = {}

    def user_vector(self, history: tuple[str, ...]) -> FloatArray | None:
        """Unit-length mean of the History item vectors, or None if it would be zero."""
        if history not in self._user_cache:
            rows = [self.row[h] for h in history if h in self.row]
            vector: FloatArray | None = None
            if rows:
                mean = np.asarray(self.matrix[rows].mean(axis=0)).ravel()
                norm = float(np.linalg.norm(mean))
                vector = mean / norm if norm > 0 else None
            self._user_cache[history] = vector
        return self._user_cache[history]


def tfidf_corpus(stats: Iterable[Impression]) -> set[str]:
    """News IDs the TF-IDF vocabulary is fitted on: History and candidates of `stats`."""
    corpus: set[str] = set()
    for imp in stats:
        corpus.update(imp.history)
        corpus.update(c.news_id for c in imp.candidates)
    return corpus


class TfidfRanker(_Personalized):
    """Cosine similarity between candidate and user TF-IDF vectors over title + abstract.

    The user vector is the mean of the known History item vectors. Ties (e.g. zero
    similarity) are broken by popularity order. A user with no known History item, or
    whose History has no in-vocabulary term, falls back to popularity.
    """

    name = "tfidf"

    def __init__(
        self,
        popularity: PopularityRanker,
        index: TfidfIndex,
        *,
        name: str = "tfidf",
        tie_break: bool = True,
        augmenter: HistoryAugmenter | None = None,
    ) -> None:
        super().__init__(popularity, tie_break=tie_break, augmenter=augmenter)
        self.name = name
        self._index = index

    @property
    def vocabulary_size(self) -> int:
        return len(self._index.vectorizer.vocabulary_)

    def similarity(self, request: RankRequest) -> FloatArray | None:
        """Cosine per candidate; None when the user vector would be zero."""
        user = self._index.user_vector(self._history(request))
        if user is None:
            return None
        row = self._index.row
        similarity = np.zeros(len(request.candidate_ids))
        known = [i for i, n in enumerate(request.candidate_ids) if n in row]
        if known:
            rows = [row[request.candidate_ids[i]] for i in known]
            similarity[known] = self._index.matrix[rows] @ user
        return similarity

    def score(self, request: RankRequest) -> Ranking:
        similarity = self.similarity(request)
        if similarity is None:
            return self._fallback(request)
        return self._rank(request, similarity)


MODEL_NAMES = (
    RandomRanker.name,
    PopularityRanker.name,
    CategoryAffinityRanker.name,
    TfidfRanker.name,
)
PERSONALIZED_PARAMS = ("tie_break", "augment_history", "popularity_window_hours")


def build_rankers(
    specs: Sequence[tuple[str, str, Mapping[str, Any]]],
    stats: Sequence[Impression],
    news: Mapping[str, NewsItem],
    *,
    seed: int,
) -> list[Ranker]:
    """Fit rankers from `stats` (a statistics source) and news metadata only.

    Each spec is (reported name, ranker kind, params). Personalized kinds accept
    `tie_break` ("popularity" or "none"), `augment_history` and
    `popularity_window_hours`; `tfidf` passes any other params to the vectorizer.
    """
    tie_breakers: dict[float | None, PopularityRanker] = {}
    tfidf_indexes: dict[str, TfidfIndex] = {}
    augmenter: HistoryAugmenter | None = None

    def tie_breaker(window: float | None) -> PopularityRanker:
        if window not in tie_breakers:
            tie_breakers[window] = PopularityRanker(stats, news, window_hours=window)
        return tie_breakers[window]

    rankers: list[Ranker] = []
    for name, kind, raw_params in specs:
        params = dict(raw_params)
        if kind == RandomRanker.name:
            ranker: Ranker = RandomRanker(seed, name=name)
        elif kind == PopularityRanker.name:
            window = params.pop("window_hours", None)
            ranker = PopularityRanker(stats, news, window_hours=window, name=name)
        elif kind in (CategoryAffinityRanker.name, TfidfRanker.name):
            tie_break = params.pop("tie_break", "popularity")
            if tie_break not in ("popularity", "none"):
                raise ValueError(f"{name}: tie_break must be 'popularity' or 'none'")
            augment = bool(params.pop("augment_history", False))
            if augment and augmenter is None:
                augmenter = HistoryAugmenter(stats)
            options: dict[str, Any] = {
                "name": name,
                "tie_break": tie_break == "popularity",
                "augmenter": augmenter if augment else None,
            }
            popularity = tie_breaker(params.pop("popularity_window_hours", None))
            if kind == CategoryAffinityRanker.name:
                ranker = CategoryAffinityRanker(popularity, news, **options)
            else:
                key = repr(sorted(params.items()))
                if key not in tfidf_indexes:
                    tfidf_indexes[key] = TfidfIndex(news, tfidf_corpus(stats), **params)
                params.clear()
                ranker = TfidfRanker(popularity, tfidf_indexes[key], **options)
        else:
            raise ValueError(f"unknown ranker {kind!r} for {name!r}; expected one of {MODEL_NAMES}")
        if params:
            raise ValueError(f"{name}: unknown params {sorted(params)}")
        rankers.append(ranker)
    return rankers
