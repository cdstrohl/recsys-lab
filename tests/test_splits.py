from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pytest

from recsys.loaders import MindDir
from recsys.schemas import Impression
from recsys.splits import HoldoutLockedError, assign_splits, last_day_cutoff

LAST_DAY = datetime(2019, 11, 11, 0, 0, 0)


def ids(impressions: Sequence[Impression]) -> list[str]:
    return [imp.impression_id for imp in impressions]


def test_cutoff_is_midnight_of_last_train_date(train: MindDir) -> None:
    # Latest train impression is 11/11/2019 9:00:00 PM.
    assert last_day_cutoff(train.impressions) == LAST_DAY


def test_default_split_boundary(train: MindDir, dev: MindDir) -> None:
    s = assign_splits(train.impressions, dev.impressions)
    assert s.cutoff == LAST_DAY
    # Impression 7 (11/10 11:59:59 PM) is the last fit row; 8 (11/11 12:00:00 AM) is tune.
    assert ids(s.fit) == ["1", "2", "3", "4", "5", "6", "7"]
    assert ids(s.tune) == ["8", "9", "10", "11"]
    assert s.counts() == {"fit": 7, "tune": 4, "holdout": 3}


def test_fit_and_tune_partition_train(train: MindDir, dev: MindDir) -> None:
    s = assign_splits(train.impressions, dev.impressions)
    assert sorted(ids(s.fit) + ids(s.tune), key=int) == ids(train.impressions)
    assert max(i.timestamp for i in s.fit) < s.cutoff <= min(i.timestamp for i in s.tune)


def test_holdout_is_locked_by_default(train: MindDir, dev: MindDir) -> None:
    s = assign_splits(train.impressions, dev.impressions)
    with pytest.raises(HoldoutLockedError):
        s.holdout()
    assert ids(s.holdout(allow_holdout=True)) == ["101", "102", "103"]
    assert "_holdout" not in repr(s)


def test_statistics_source_never_includes_holdout(train: MindDir, dev: MindDir) -> None:
    s = assign_splits(train.impressions, dev.impressions)
    assert s.statistics_source() == s.fit
    assert s.statistics_source(include_tune=True) == s.fit + s.tune


def test_cutoff_override(train: MindDir, dev: MindDir) -> None:
    # Fallback cut: noon on 11/10. Impression 4 (10:00 AM) is fit; 5 (2:45 PM) is tune.
    s = assign_splits(train.impressions, dev.impressions, cutoff=datetime(2019, 11, 10, 12))
    assert ids(s.fit) == ["1", "2", "3", "4"]
    assert ids(s.tune) == ["5", "6", "7", "8", "9", "10", "11"]


@pytest.mark.parametrize(
    "cutoff", [datetime(2019, 11, 9, 0, 0), datetime(2019, 11, 12, 0, 0)], ids=["no_fit", "no_tune"]
)
def test_cutoff_leaving_a_split_empty_raises(
    train: MindDir, dev: MindDir, cutoff: datetime
) -> None:
    with pytest.raises(ValueError, match="both must be non-empty"):
        assign_splits(train.impressions, dev.impressions, cutoff=cutoff)


def test_empty_train_raises() -> None:
    with pytest.raises(ValueError):
        last_day_cutoff(())
