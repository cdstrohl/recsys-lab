"""Parsers for the original MIND `news.tsv` and `behaviors.tsv` files.

Both files are tab-separated with no header row. Parsing is strict by default:
the first malformed row raises `ParseError`. With `strict=False`, malformed rows
are skipped and recorded in the returned `ParseReport` (used by the inventory).
Blank lines are always skipped and counted. Error details name IDs and
timestamps only, never article text.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recsys.schemas import (
    BEHAVIORS_COLUMNS,
    NEWS_COLUMNS,
    Candidate,
    Impression,
    IssueReason,
    NewsItem,
    ParseError,
    ParseIssue,
    ParseReport,
)

TRAIN_DIR = "MINDsmall_train"
DEV_DIR = "MINDsmall_dev"
NEWS_FILE = "news.tsv"
BEHAVIORS_FILE = "behaviors.tsv"

# MIND format, e.g. "11/15/2019 8:55:22 AM". Parsed by hand so it doesn't depend on locale.
_TIMESTAMP_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2}):(\d{2}) (AM|PM)")


def parse_timestamp(text: str) -> datetime:
    """Parse a MIND timestamp into a naive datetime. Raises ValueError if invalid."""
    match = _TIMESTAMP_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"unrecognized timestamp {text!r}")
    month, day, year, hour, minute, second = (int(g) for g in match.groups()[:6])
    if not 1 <= hour <= 12:
        raise ValueError(f"hour out of range in {text!r}")
    hour = hour % 12 + (12 if match.group(7) == "PM" else 0)
    return datetime(year, month, day, hour, minute, second)


class _RowError(Exception):
    def __init__(self, reason: IssueReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _record(report: ParseReport, line_no: int, error: _RowError, *, strict: bool) -> None:
    issue = ParseIssue(line_no, error.reason, error.detail)
    if strict:
        raise ParseError(report.path, issue)
    report.issues.append(issue)


def _rows(
    path: Path, n_columns: int, report: ParseReport, *, strict: bool
) -> Iterator[tuple[int, list[str]]]:
    """Yield (line number, fields) for each non-blank row with the right column count.

    Lines are split on "\\n" only (a stray "\\r" inside a field is kept), and only the
    line terminator is stripped so trailing empty columns survive.
    """
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, start=1):
            try:
                line = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
            except UnicodeDecodeError as exc:
                error = _RowError(IssueReason.BAD_ENCODING, str(exc))
                _record(report, line_no, error, strict=strict)
                continue
            if not line.strip():
                report.blank_lines += 1
                continue
            fields = line.split("\t")
            if len(fields) != n_columns:
                error = _RowError(
                    IssueReason.COLUMN_COUNT,
                    f"expected {n_columns} columns, found {len(fields)}",
                )
                _record(report, line_no, error, strict=strict)
                continue
            yield line_no, fields


def _parse_news(fields: list[str]) -> NewsItem:
    item = NewsItem(*fields)
    if not item.news_id.strip():
        raise _RowError(IssueReason.EMPTY_ID, "empty news ID")
    return item


def _parse_impression(fields: list[str]) -> Impression:
    impression_id, user_id, time_text, history_text, impressions_text = fields
    if not impression_id.strip():
        raise _RowError(IssueReason.EMPTY_ID, "empty impression ID")
    if not user_id.strip():
        raise _RowError(IssueReason.EMPTY_ID, f"empty user ID in impression {impression_id}")
    try:
        timestamp = parse_timestamp(time_text)
    except ValueError as exc:
        raise _RowError(IssueReason.BAD_TIMESTAMP, str(exc)) from None

    tokens = impressions_text.split()
    if not tokens:
        raise _RowError(IssueReason.EMPTY_CANDIDATES, f"impression {impression_id} has none")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for token in tokens:
        news_id, sep, label = token.rpartition("-")
        if not sep or not news_id or label not in ("0", "1"):
            raise _RowError(IssueReason.BAD_CANDIDATE, f"bad candidate token {token!r}")
        if news_id in seen:
            raise _RowError(IssueReason.DUPLICATE_CANDIDATE, f"{news_id} repeated")
        seen.add(news_id)
        candidates.append(Candidate(news_id, label == "1"))

    return Impression(
        impression_id=impression_id,
        user_id=user_id,
        timestamp=timestamp,
        history=tuple(history_text.split()),
        candidates=tuple(candidates),
    )


def read_news(path: Path, *, strict: bool = True) -> tuple[dict[str, NewsItem], ParseReport]:
    """Read `news.tsv` into a dict keyed by news ID. Duplicate IDs keep the first row."""
    report = ParseReport(path)
    items: dict[str, NewsItem] = {}
    for line_no, fields in _rows(path, len(NEWS_COLUMNS), report, strict=strict):
        try:
            item = _parse_news(fields)
            if item.news_id in items:
                raise _RowError(IssueReason.DUPLICATE_ID, f"news ID {item.news_id} repeated")
        except _RowError as error:
            _record(report, line_no, error, strict=strict)
            continue
        items[item.news_id] = item
    report.rows_parsed = len(items)
    return items, report


def read_behaviors(
    path: Path, *, strict: bool = True
) -> tuple[tuple[Impression, ...], ParseReport]:
    """Read `behaviors.tsv` in file order. Duplicate impression IDs keep the first row."""
    report = ParseReport(path)
    impressions: list[Impression] = []
    seen: set[str] = set()
    for line_no, fields in _rows(path, len(BEHAVIORS_COLUMNS), report, strict=strict):
        try:
            impression = _parse_impression(fields)
            if impression.impression_id in seen:
                raise _RowError(
                    IssueReason.DUPLICATE_ID,
                    f"impression ID {impression.impression_id} repeated",
                )
        except _RowError as error:
            _record(report, line_no, error, strict=strict)
            continue
        seen.add(impression.impression_id)
        impressions.append(impression)
    report.rows_parsed = len(impressions)
    return tuple(impressions), report


@dataclass(frozen=True, slots=True)
class MindDir:
    """The parsed contents of one MIND directory, e.g. `MINDsmall_train/`."""

    name: str
    news: dict[str, NewsItem]
    impressions: tuple[Impression, ...]
    news_report: ParseReport
    behaviors_report: ParseReport


def load_mind_dir(directory: Path, *, strict: bool = True) -> MindDir:
    news, news_report = read_news(directory / NEWS_FILE, strict=strict)
    impressions, behaviors_report = read_behaviors(directory / BEHAVIORS_FILE, strict=strict)
    return MindDir(directory.name, news, impressions, news_report, behaviors_report)
