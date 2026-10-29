"""Data inventory for MIND directories: aggregate counts only, never raw records.

Answers the questions the spec says must be verified on real data before
modeling: row counts, date range, no-click impressions, history-length
distribution and the share of dev candidates unseen in train.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from recsys.loaders import DEV_DIR, TRAIN_DIR, MindDir, load_mind_dir
from recsys.schemas import Impression, NewsItem, ParseReport
from recsys.splits import assign_splits

# Lower edges of history-length buckets: "0", "1-5", "6+".
DEFAULT_HISTORY_EDGES: tuple[int, ...] = (0, 1, 6)
PROVENANCE_FILE = "PROVENANCE.json"


def history_bucket(length: int, edges: Sequence[int] = DEFAULT_HISTORY_EDGES) -> str:
    """Label for the bucket containing `length`; `edges` are ascending lower bounds from 0."""
    if not edges or edges[0] != 0 or list(edges) != sorted(set(edges)):
        raise ValueError(f"edges must be ascending, unique and start at 0, got {edges}")
    if length < 0:
        raise ValueError("history length cannot be negative")
    for i, low in enumerate(edges):
        high = edges[i + 1] - 1 if i + 1 < len(edges) else None
        if high is None:
            return f"{low}+"
        if length <= high:
            return str(low) if low == high else f"{low}-{high}"
    raise AssertionError("unreachable")


def _bucket_counts(lengths: Iterable[int], edges: Sequence[int]) -> dict[str, int]:
    counts = Counter(history_bucket(n, edges) for n in lengths)
    labels = [history_bucket(e, edges) for e in edges]
    return {label: counts.get(label, 0) for label in labels}


def _summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def news_inventory(news: Mapping[str, NewsItem]) -> dict[str, Any]:
    categories = Counter(item.category for item in news.values())
    return {
        "n_items": len(news),
        "n_empty_title": sum(not item.title.strip() for item in news.values()),
        "n_empty_abstract": sum(not item.abstract.strip() for item in news.values()),
        "categories": dict(categories.most_common()),
    }


def behaviors_inventory(
    impressions: Sequence[Impression],
    news_ids: Iterable[str],
    edges: Sequence[int] = DEFAULT_HISTORY_EDGES,
) -> dict[str, Any]:
    known = set(news_ids)
    histories: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for imp in impressions:
        histories[imp.user_id].add(imp.history)
    # A user whose History differs across rows breaks the "fixed pre-period list" assumption.
    user_history_len = {user: len(next(iter(h))) for user, h in histories.items() if len(h) == 1}

    candidate_ids = {c.news_id for imp in impressions for c in imp.candidates}
    history_ids = {n for imp in impressions for n in imp.history}
    timestamps = [imp.timestamp for imp in impressions]
    per_day = Counter(ts.date().isoformat() for ts in timestamps)

    return {
        "n_impressions": len(impressions),
        "n_users": len(histories),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
        "impressions_per_day": dict(sorted(per_day.items())),
        "n_candidates": sum(len(imp.candidates) for imp in impressions),
        "n_clicks": sum(imp.n_clicks for imp in impressions),
        "n_no_click_impressions": sum(imp.n_clicks == 0 for imp in impressions),
        "n_all_click_impressions": sum(imp.n_clicks == len(imp.candidates) for imp in impressions),
        "n_single_candidate_impressions": sum(len(imp.candidates) == 1 for imp in impressions),
        "candidates_per_impression": _summary([len(imp.candidates) for imp in impressions]),
        "clicks_per_impression": _summary([imp.n_clicks for imp in impressions]),
        "history_length_per_impression": _bucket_counts(
            (len(imp.history) for imp in impressions), edges
        ),
        "history_length_per_user": _bucket_counts(user_history_len.values(), edges),
        "n_users_inconsistent_history": len(histories) - len(user_history_len),
        "n_candidate_ids_missing_from_news": len(candidate_ids - known),
        "n_history_ids_missing_from_news": len(history_ids - known),
    }


def unseen_inventory(
    reference: Sequence[Impression], target: Sequence[Impression]
) -> dict[str, Any]:
    """How much of `target` (e.g. dev) is new relative to `reference` (e.g. train)."""
    ref_candidates = {c.news_id for imp in reference for c in imp.candidates}
    ref_items = ref_candidates | {n for imp in reference for n in imp.history}
    ref_users = {imp.user_id for imp in reference}
    ref_last = max((imp.timestamp for imp in reference), default=None)

    occurrences = [c.news_id for imp in target for c in imp.candidates]
    unique = set(occurrences)
    target_history = {n for imp in target for n in imp.history}
    target_users = {imp.user_id for imp in target}

    return {
        "candidate_occurrences": len(occurrences),
        "unique_candidates": len(unique),
        "share_occurrences_unseen_as_candidate": _share(
            sum(n not in ref_candidates for n in occurrences), len(occurrences)
        ),
        "share_unique_unseen_as_candidate": _share(len(unique - ref_candidates), len(unique)),
        "share_unique_unseen_anywhere": _share(len(unique - ref_items), len(unique)),
        "share_history_items_seen_as_candidate": _share(
            len(target_history & ref_candidates), len(target_history)
        ),
        "share_users_unseen": _share(len(target_users - ref_users), len(target_users)),
        "n_impressions_before_reference_end": (
            sum(imp.timestamp < ref_last for imp in target) if ref_last else 0
        ),
        # MIND may restart impression IDs per file; shared IDs mean keys must include the split.
        "n_impression_ids_shared": len(
            {imp.impression_id for imp in reference} & {imp.impression_id for imp in target}
        ),
    }


def _period_summary(impressions: Sequence[Impression]) -> dict[str, Any]:
    timestamps = [imp.timestamp for imp in impressions]
    return {
        "n_impressions": len(impressions),
        "n_users": len({imp.user_id for imp in impressions}),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
        "n_no_click_impressions": sum(imp.n_clicks == 0 for imp in impressions),
    }


def split_inventory(
    train: Sequence[Impression], dev: Sequence[Impression] | None = None
) -> dict[str, Any]:
    """Sizes of the default fit/tune cut and how new tune/dev candidates are relative to fit.

    A cut that can't be made is reported, not raised: it is the signal to pick
    another cutoff and record it in DECISIONS.md.
    """
    try:
        assignment = assign_splits(train, ())
    except ValueError as exc:
        return {"error": str(exc)}
    summary: dict[str, Any] = {
        "cutoff": assignment.cutoff.isoformat(),
        "fit": _period_summary(assignment.fit),
        "tune": _period_summary(assignment.tune),
        "tune_vs_fit": unseen_inventory(assignment.fit, assignment.tune),
    }
    if dev is not None:
        summary["holdout_vs_fit"] = unseen_inventory(assignment.fit, dev)
    return summary


def _parse_summary(report: ParseReport) -> dict[str, Any]:
    return {
        "rows_parsed": report.rows_parsed,
        "blank_lines": report.blank_lines,
        "issues": report.counts_by_reason(),
    }


def _dir_inventory(mind: MindDir, edges: Sequence[int]) -> dict[str, Any]:
    return {
        "parse": {
            "news.tsv": _parse_summary(mind.news_report),
            "behaviors.tsv": _parse_summary(mind.behaviors_report),
        },
        "news": news_inventory(mind.news),
        "behaviors": behaviors_inventory(mind.impressions, mind.news, edges),
    }


def read_provenance(data_dir: Path) -> str:
    path = data_dir / PROVENANCE_FILE
    if not path.exists():
        return "unlabeled"
    return str(json.loads(path.read_text(encoding="utf-8"))["provenance"])


def build_inventory(data_dir: Path, edges: Sequence[int] = DEFAULT_HISTORY_EDGES) -> dict[str, Any]:
    """Inventory `data_dir/MINDsmall_train` and, if present, `data_dir/MINDsmall_dev`.

    Parsing is lenient so every malformed row is counted rather than stopping the run.
    """
    train = load_mind_dir(data_dir / TRAIN_DIR, strict=False)
    inventory: dict[str, Any] = {
        "data_dir": str(data_dir),
        "data_provenance": read_provenance(data_dir),
        TRAIN_DIR: _dir_inventory(train, edges),
    }
    dev_dir = data_dir / DEV_DIR
    dev = load_mind_dir(dev_dir, strict=False) if dev_dir.is_dir() else None
    if dev is not None:
        inventory[DEV_DIR] = _dir_inventory(dev, edges)
        inventory["dev_vs_train"] = unseen_inventory(train.impressions, dev.impressions)
    inventory["splits"] = split_inventory(
        train.impressions, dev.impressions if dev is not None else None
    )
    return inventory


def format_inventory(value: Any, indent: int = 0) -> str:
    """Render a nested inventory dict as indented `key: value` lines."""
    pad = "  " * indent
    lines: list[str] = []
    for key, item in value.items():
        if isinstance(item, dict):
            lines.append(f"{pad}{key}:" + ("" if item else " {}"))
            if item:
                lines.append(format_inventory(item, indent + 1))
        elif isinstance(item, float):
            lines.append(f"{pad}{key}: {item:.4f}")
        else:
            lines.append(f"{pad}{key}: {item}")
    return "\n".join(lines)
