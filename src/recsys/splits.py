"""The fit / tune / holdout split policy and as-of (point-in-time) click counts.

| Split   | Source                                          | Use                         |
|---------|-------------------------------------------------|-----------------------------|
| fit     | MINDsmall_train, before the last-day cutoff     | fit models, compute stats   |
| tune    | MINDsmall_train, at or after the cutoff         | model selection, debugging  |
| holdout | MINDsmall_dev                                   | locked; headline numbers    |

The cutoff is midnight at the start of the calendar date of the latest train
impression (naive timestamps, no time-zone conversion). Intervals are half-open,
so an impression at exactly 00:00:00 on the last day belongs to `tune`.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import StrEnum

from recsys.schemas import Impression


class Split(StrEnum):
    FIT = "fit"
    TUNE = "tune"
    HOLDOUT = "holdout"


class HoldoutLockedError(RuntimeError):
    """Raised when holdout impressions are requested without explicitly unlocking them."""


def last_day_cutoff(train: Sequence[Impression]) -> datetime:
    """Midnight at the start of the calendar date of the latest train impression."""
    if not train:
        raise ValueError("cannot compute a cutoff from no impressions")
    latest = max(imp.timestamp for imp in train)
    return datetime.combine(latest.date(), time.min)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    cutoff: datetime
    fit: tuple[Impression, ...]
    tune: tuple[Impression, ...]
    _holdout: tuple[Impression, ...] = field(repr=False)

    def holdout(self, *, allow_holdout: bool = False) -> tuple[Impression, ...]:
        """Holdout impressions. Unlock only for headline evaluation, never for tuning."""
        if not allow_holdout:
            raise HoldoutLockedError(
                "holdout is locked; pass allow_holdout=True only for headline evaluation"
            )
        return self._holdout

    def statistics_source(self, *, include_tune: bool = False) -> tuple[Impression, ...]:
        """Impressions that features and fitted parameters may use.

        `fit` by default; `fit` + `tune` only for the documented final refit.
        Holdout is never a statistics source.
        """
        return self.fit + self.tune if include_tune else self.fit

    def counts(self) -> dict[str, int]:
        return {
            Split.FIT.value: len(self.fit),
            Split.TUNE.value: len(self.tune),
            Split.HOLDOUT.value: len(self._holdout),
        }


def assign_splits(
    train: Sequence[Impression],
    dev: Sequence[Impression],
    *,
    cutoff: datetime | None = None,
) -> SplitAssignment:
    """Split train into fit/tune at `cutoff` (default: `last_day_cutoff`); dev is holdout.

    Pass `cutoff` explicitly only if the inventory shows the last day is a poor cut,
    and record the choice in DECISIONS.md. Uses timestamps only, never labels.
    """
    cutoff = cutoff if cutoff is not None else last_day_cutoff(train)
    fit = tuple(imp for imp in train if imp.timestamp < cutoff)
    tune = tuple(imp for imp in train if imp.timestamp >= cutoff)
    if not fit or not tune:
        raise ValueError(
            f"cutoff {cutoff.isoformat()} leaves fit={len(fit)}, tune={len(tune)}; both must be"
            " non-empty"
        )
    return SplitAssignment(cutoff, fit, tune, tuple(dev))


@dataclass(frozen=True, slots=True)
class ClickEvent:
    timestamp: datetime
    news_id: str


def click_events(impressions: Iterable[Impression]) -> list[ClickEvent]:
    """One event per clicked candidate, stamped with its impression's time.

    `History` clicks have no timestamp, so they are never events.
    """
    return [
        ClickEvent(imp.timestamp, c.news_id)
        for imp in impressions
        for c in imp.candidates
        if c.clicked
    ]


class AsOfClickIndex:
    """Point-in-time click counts per news item.

    `count_before(item, t)` counts clicks strictly before `t`, so an impression
    never sees its own labels or those of impressions with the same timestamp.
    Build it only from `SplitAssignment.statistics_source()`.
    """

    def __init__(self, events: Iterable[ClickEvent]) -> None:
        by_item: dict[str, list[datetime]] = defaultdict(list)
        for event in events:
            by_item[event.news_id].append(event.timestamp)
        self._times = {news_id: sorted(times) for news_id, times in by_item.items()}

    @classmethod
    def from_impressions(cls, impressions: Iterable[Impression]) -> AsOfClickIndex:
        return cls(click_events(impressions))

    def count_before(self, news_id: str, as_of: datetime, *, since: datetime | None = None) -> int:
        """Clicks on `news_id` with `since <= t < as_of` (`since=None`: no lower bound)."""
        times = self._times.get(news_id)
        if not times:
            return 0
        high = bisect_left(times, as_of)
        low = bisect_left(times, since) if since is not None else 0
        return max(0, high - low)

    def counts_before(
        self, news_ids: Iterable[str], as_of: datetime, *, since: datetime | None = None
    ) -> list[int]:
        return [self.count_before(n, as_of, since=since) for n in news_ids]

    def snapshot(self) -> dict[str, tuple[datetime, ...]]:
        """Full contents, for equality checks in leakage tests."""
        return {news_id: tuple(times) for news_id, times in sorted(self._times.items())}
