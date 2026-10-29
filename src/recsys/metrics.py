"""Per-impression ranking metrics and their average over eligible impressions.

Conventions (frozen by tests; see DECISIONS.md D-008 to D-010):
- Labels are binary clicks; a higher score ranks higher. Scores must be finite.
- MRR and nDCG need a strict ranking. Score ties are broken by a seeded hash of
  (seed, impression_id, news_id), lower key first. That is deterministic and
  independent of the order candidates appear in the impression.
- AUC is rank-based: a tied clicked/non-clicked pair counts 0.5. No tie-break.
- MRR is MIND-style: mean of 1/rank over *all* clicked candidates.
- nDCG@k uses gain 2^rel - 1 and discount log2(rank + 1); with fewer than k
  candidates both DCG and ideal DCG run over all of them, i.e. min(k, n).
- Eligibility: no-click impressions are excluded from every metric. All-click
  impressions are excluded from AUC only (undefined there).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from recsys.schemas import Impression

DEFAULT_KS: tuple[int, ...] = (5, 10)

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
KeyArray = npt.NDArray[np.uint64]


def _labels(labels: npt.ArrayLike) -> IntArray:
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("labels must be a non-empty 1-D sequence")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y


def _scores(scores: npt.ArrayLike, n: int) -> FloatArray:
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (n,):
        raise ValueError(f"expected {n} scores, got shape {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return s


def tie_break_keys(impression_id: str, news_ids: Sequence[str], seed: int) -> KeyArray:
    """A 64-bit key per candidate from (seed, impression_id, news_id)."""
    keys = [
        int.from_bytes(
            hashlib.blake2b(f"{seed}\x1f{impression_id}\x1f{n}".encode(), digest_size=8).digest(),
            "big",
        )
        for n in news_ids
    ]
    return np.array(keys, dtype=np.uint64)


def rank_order(scores: npt.ArrayLike, tie_keys: npt.ArrayLike) -> npt.NDArray[np.intp]:
    """Candidate indices, best first: score descending, then tie key ascending."""
    s = np.asarray(scores, dtype=np.float64)
    keys = np.asarray(tie_keys, dtype=np.uint64)
    if keys.shape != s.shape:
        raise ValueError("tie_keys must match scores")
    # lexsort sorts by the last key first.
    return np.lexsort((keys, -s))


def _ranked_labels(
    labels: npt.ArrayLike, scores: npt.ArrayLike, tie_keys: npt.ArrayLike
) -> IntArray:
    y = _labels(labels)
    if not y.any():
        raise ValueError("ranking metrics need at least one clicked candidate")
    return y[rank_order(_scores(scores, y.size), tie_keys)]


def auc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Share of (clicked, non-clicked) pairs ordered correctly; ties count 0.5."""
    y = _labels(labels)
    s = _scores(scores, y.size)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC needs at least one clicked and one non-clicked candidate")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def mrr(labels: npt.ArrayLike, scores: npt.ArrayLike, tie_keys: npt.ArrayLike) -> float:
    """MIND-style MRR: mean of 1/rank over all clicked candidates (1-based ranks)."""
    ranks = np.flatnonzero(_ranked_labels(labels, scores, tie_keys)) + 1
    return float(np.mean(1.0 / ranks))


def _dcg(relevance: IntArray, k: int) -> float:
    top = relevance[:k].astype(np.float64)
    discounts = np.log2(np.arange(2, top.size + 2))
    return float(np.sum((2.0**top - 1.0) / discounts))


def ndcg_at_k(
    labels: npt.ArrayLike, scores: npt.ArrayLike, tie_keys: npt.ArrayLike, k: int
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    ranked = _ranked_labels(labels, scores, tie_keys)
    ideal = np.sort(ranked)[::-1]
    return _dcg(ranked, k) / _dcg(ideal, k)


class Eligibility(StrEnum):
    ELIGIBLE = "eligible"  # at least one click and one non-click: every metric
    ALL_CLICK = "all_click"  # MRR and nDCG only
    NO_CLICK = "no_click"  # excluded from every metric


def eligibility(labels: npt.ArrayLike) -> Eligibility:
    y = _labels(labels)
    if not y.any():
        return Eligibility.NO_CLICK
    if y.all():
        return Eligibility.ALL_CLICK
    return Eligibility.ELIGIBLE


def metric_names(ks: Sequence[int] = DEFAULT_KS) -> tuple[str, ...]:
    return ("auc", "mrr", *(f"ndcg@{k}" for k in ks))


@dataclass(frozen=True, slots=True)
class ImpressionMetrics:
    impression_id: str
    eligibility: Eligibility
    values: dict[str, float]  # "auc" absent for all-click; empty for no-click
    tied: bool  # any two candidates share a score


@dataclass(frozen=True, slots=True)
class MetricSummary:
    means: dict[str, float | None]  # None when no impression is eligible for the metric
    n_eligible: dict[str, int]
    n_impressions: int
    n_no_click: int
    n_all_click: int
    n_tied: int  # impressions entering any metric whose scores contain a tie
    per_impression: tuple[ImpressionMetrics, ...]


def score_impression(
    impression: Impression,
    scores: npt.ArrayLike,
    *,
    seed: int,
    ks: Sequence[int] = DEFAULT_KS,
) -> ImpressionMetrics:
    labels = [int(c.clicked) for c in impression.candidates]
    s = _scores(scores, len(labels))
    status = eligibility(labels)
    tied = bool(np.unique(s).size < s.size)
    values: dict[str, float] = {}
    if status is not Eligibility.NO_CLICK:
        keys = tie_break_keys(
            impression.impression_id, [c.news_id for c in impression.candidates], seed
        )
        if status is Eligibility.ELIGIBLE:
            values["auc"] = auc(labels, s)
        values["mrr"] = mrr(labels, s, keys)
        for k in ks:
            values[f"ndcg@{k}"] = ndcg_at_k(labels, s, keys, k)
    return ImpressionMetrics(impression.impression_id, status, values, tied)


def evaluate(
    impressions: Sequence[Impression],
    scores: Sequence[npt.ArrayLike],
    *,
    seed: int,
    ks: Sequence[int] = DEFAULT_KS,
) -> MetricSummary:
    """Score every impression, then average each metric over impressions eligible for it."""
    if len(scores) != len(impressions):
        raise ValueError(f"{len(impressions)} impressions but {len(scores)} score lists")
    rows = tuple(
        score_impression(imp, s, seed=seed, ks=ks)
        for imp, s in zip(impressions, scores, strict=True)
    )
    names = metric_names(ks)
    collected: dict[str, list[float]] = {name: [] for name in names}
    for row in rows:
        for name, value in row.values.items():
            collected[name].append(value)
    return MetricSummary(
        means={name: float(np.mean(v)) if v else None for name, v in collected.items()},
        n_eligible={name: len(v) for name, v in collected.items()},
        n_impressions=len(rows),
        n_no_click=sum(r.eligibility is Eligibility.NO_CLICK for r in rows),
        n_all_click=sum(r.eligibility is Eligibility.ALL_CLICK for r in rows),
        n_tied=sum(r.tied for r in rows if r.eligibility is not Eligibility.NO_CLICK),
        per_impression=rows,
    )
