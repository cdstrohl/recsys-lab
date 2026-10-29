"""Leakage tests: statistics depend only on fit, and as-of counts only on prior clicks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

import pytest

from recsys.loaders import MindDir
from recsys.schemas import Candidate, Impression
from recsys.splits import AsOfClickIndex, assign_splits


def flip_labels(impressions: Sequence[Impression]) -> tuple[Impression, ...]:
    return tuple(
        replace(imp, candidates=tuple(Candidate(c.news_id, not c.clicked) for c in imp.candidates))
        for imp in impressions
    )


def all_queries(index: AsOfClickIndex, impressions: Sequence[Impression]) -> list[list[int]]:
    return [
        index.counts_before([c.news_id for c in imp.candidates], imp.timestamp)
        for imp in impressions
    ]


def by_id(mind: MindDir) -> dict[str, Impression]:
    return {imp.impression_id: imp for imp in mind.impressions}


def test_changing_tune_and_holdout_labels_leaves_fit_statistics_unchanged(
    train: MindDir, dev: MindDir
) -> None:
    base = assign_splits(train.impressions, dev.impressions)
    perturbed = assign_splits(
        base.fit + flip_labels(base.tune), flip_labels(base.holdout(allow_holdout=True))
    )
    assert perturbed.cutoff == base.cutoff
    assert perturbed.fit == base.fit

    base_index = AsOfClickIndex.from_impressions(base.statistics_source())
    perturbed_index = AsOfClickIndex.from_impressions(perturbed.statistics_source())
    assert perturbed_index.snapshot() == base_index.snapshot()

    # Same features for every tune and holdout impression.
    targets = base.tune + base.holdout(allow_holdout=True)
    assert all_queries(perturbed_index, targets) == all_queries(base_index, targets)


def test_fit_only_index_ignores_tune_clicks(train: MindDir, dev: MindDir) -> None:
    s = assign_splits(train.impressions, dev.impressions)
    eleven = by_id(train)["11"]
    # N13 is clicked in tune impressions 9 and 10 (7:20 AM), before impression 11 (9 PM),
    # but never clicked in fit (impression 7 has N13-0).
    assert AsOfClickIndex.from_impressions(s.fit).count_before("N13", eleven.timestamp) == 0
    assert (
        AsOfClickIndex.from_impressions(train.impressions).count_before("N13", eleven.timestamp)
        == 2
    )


def test_impression_never_sees_its_own_click(train: MindDir) -> None:
    index = AsOfClickIndex.from_impressions(train.impressions)
    imps = by_id(train)
    # N3 is clicked in 1 (11/9 8:05 AM), 7 (11/10 11:59:59 PM) and 8 (11/11 12:00:00 AM).
    assert index.count_before("N3", imps["8"].timestamp) == 2  # 1 and 7, not 8 itself
    assert index.count_before("N3", imps["7"].timestamp) == 1  # 1 only
    assert index.count_before("N3", imps["1"].timestamp) == 0


def test_identical_timestamps_do_not_see_each_other(train: MindDir) -> None:
    index = AsOfClickIndex.from_impressions(train.impressions)
    imps = by_id(train)
    # Impressions 9 and 10 both click N13 at 11/11 7:20:00 AM.
    assert imps["9"].timestamp == imps["10"].timestamp
    assert index.count_before("N13", imps["9"].timestamp) == 0
    assert index.count_before("N13", imps["11"].timestamp) == 2


def test_own_labels_never_change_own_features(train: MindDir) -> None:
    # For every impression, flipping its own labels leaves the counts it queries unchanged.
    index = AsOfClickIndex.from_impressions(train.impressions)
    for i, imp in enumerate(train.impressions):
        others = train.impressions[:i] + train.impressions[i + 1 :]
        flipped = AsOfClickIndex.from_impressions((*others, *flip_labels([imp])))
        assert all_queries(flipped, [imp]) == all_queries(index, [imp]), imp.impression_id


def test_history_items_are_not_click_events(train: MindDir) -> None:
    # N1 appears in many History lists but is never clicked as a candidate.
    index = AsOfClickIndex.from_impressions(train.impressions)
    assert index.count_before("N1", datetime(2100, 1, 1)) == 0


@pytest.mark.parametrize(
    ("since", "expected"),
    [
        (None, 2),  # impressions 1 and 7
        (datetime(2019, 11, 10), 1),  # impression 7 only
        (datetime(2019, 11, 10, 23, 59, 59), 1),  # lower bound is inclusive
        (datetime(2019, 11, 11), 0),  # window starts at as_of
        (datetime(2019, 11, 12), 0),  # window starts after as_of
    ],
)
def test_recent_window(train: MindDir, since: datetime | None, expected: int) -> None:
    index = AsOfClickIndex.from_impressions(train.impressions)
    assert index.count_before("N3", datetime(2019, 11, 11), since=since) == expected
