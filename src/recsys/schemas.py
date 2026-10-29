"""Typed records for the original MIND `news.tsv` and `behaviors.tsv` files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

NEWS_COLUMNS = (
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)
BEHAVIORS_COLUMNS = ("impression_id", "user_id", "time", "history", "impressions")


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One row of `news.tsv`. Missing text fields are empty strings."""

    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    # Raw JSON strings, left unparsed until a ticket needs them.
    title_entities: str
    abstract_entities: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """One `NewsID-label` token from the impressions column."""

    news_id: str
    clicked: bool


@dataclass(frozen=True, slots=True)
class Impression:
    """One row of `behaviors.tsv`: a user, a timestamp and a labeled candidate list.

    `history` is MIND's fixed pre-period click list; it carries no timestamps.
    `timestamp` is naive: MIND gives no time zone and none is assumed.
    """

    impression_id: str
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    candidates: tuple[Candidate, ...]

    @property
    def n_clicks(self) -> int:
        return sum(c.clicked for c in self.candidates)


class IssueReason(StrEnum):
    COLUMN_COUNT = "column_count"
    EMPTY_ID = "empty_id"
    DUPLICATE_ID = "duplicate_id"
    BAD_TIMESTAMP = "bad_timestamp"
    BAD_CANDIDATE = "bad_candidate"
    EMPTY_CANDIDATES = "empty_candidates"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    BAD_ENCODING = "bad_encoding"


@dataclass(frozen=True, slots=True)
class ParseIssue:
    line_no: int
    reason: IssueReason
    detail: str


class ParseError(ValueError):
    """Raised by strict parsing on the first malformed row."""

    def __init__(self, path: Path, issue: ParseIssue) -> None:
        super().__init__(f"{path}:{issue.line_no}: {issue.reason}: {issue.detail}")
        self.path = path
        self.issue = issue


@dataclass(slots=True)
class ParseReport:
    """What happened while reading one file. Blank lines are skipped, not errors."""

    path: Path
    rows_parsed: int = 0
    blank_lines: int = 0
    issues: list[ParseIssue] = field(default_factory=list)

    def counts_by_reason(self) -> dict[str, int]:
        return dict(sorted(Counter(i.reason.value for i in self.issues).items()))
