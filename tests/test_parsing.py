from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from conftest import MALFORMED

from recsys.loaders import MindDir, parse_timestamp, read_behaviors, read_news
from recsys.schemas import IssueReason, ParseError


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("11/9/2019 8:05:00 AM", datetime(2019, 11, 9, 8, 5, 0)),
        ("11/09/2019 08:05:00 AM", datetime(2019, 11, 9, 8, 5, 0)),
        ("11/11/2019 12:00:00 AM", datetime(2019, 11, 11, 0, 0, 0)),
        ("11/9/2019 12:30:00 PM", datetime(2019, 11, 9, 12, 30, 0)),
        ("11/10/2019 11:59:59 PM", datetime(2019, 11, 10, 23, 59, 59)),
    ],
)
def test_parse_timestamp(text: str, expected: datetime) -> None:
    assert parse_timestamp(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "2019-11-09 09:00:00",
        "11/9/2019 13:00:00 PM",
        "11/9/2019 0:10:00 AM",
        "2/30/2019 1:00:00 AM",
        "11/9/2019 8:05:00 am",
        "11/9/2019 8:05 AM",
    ],
)
def test_parse_timestamp_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_timestamp(text)


def test_news_fields(train: MindDir) -> None:
    news = train.news
    assert len(news) == 14
    assert train.news_report.issues == []
    assert news["N2"].abstract == ""
    # N7: empty abstract and empty entity columns at the end of the line survive.
    assert (news["N7"].abstract, news["N7"].title_entities, news["N7"].abstract_entities) == (
        "",
        "",
        "",
    )
    assert '"we\'re ready"' in news["N5"].title
    assert "Café" in news["N8"].title
    assert news["N1"].title_entities.startswith('[{"Label"')
    assert news["N1"].category == "sports"


def test_behaviors_fields(train: MindDir) -> None:
    imps = {imp.impression_id: imp for imp in train.impressions}
    assert [imp.impression_id for imp in train.impressions] == [str(i) for i in range(1, 12)]
    assert imps["2"].history == ()
    assert imps["9"].history == ("N1", "N2", "N4", "N5", "N6", "N7", "N8")
    assert imps["3"].n_clicks == 0
    assert imps["2"].n_clicks == 2
    four = imps["4"]
    assert len(four.candidates) == 12
    assert [c.news_id for c in four.candidates if c.clicked] == ["N7"]
    # Candidate order is preserved exactly as in the file.
    assert [c.news_id for c in imps["8"].candidates] == ["N3", "N14", "N1"]
    assert imps["8"].timestamp == datetime(2019, 11, 11, 0, 0, 0)


def test_malformed_behaviors_lenient() -> None:
    impressions, report = read_behaviors(MALFORMED / "behaviors_malformed.tsv", strict=False)
    assert [imp.impression_id for imp in impressions] == ["1", "11"]
    assert report.rows_parsed == 2
    assert report.blank_lines == 1
    assert report.counts_by_reason() == {
        "bad_candidate": 2,
        "bad_timestamp": 2,
        "column_count": 1,
        "duplicate_candidate": 1,
        "duplicate_id": 1,
        "empty_candidates": 1,
        "empty_id": 1,
    }
    assert [(i.line_no, i.reason) for i in report.issues] == [
        (2, IssueReason.COLUMN_COUNT),
        (3, IssueReason.BAD_TIMESTAMP),
        (4, IssueReason.BAD_CANDIDATE),
        (5, IssueReason.BAD_CANDIDATE),
        (6, IssueReason.EMPTY_CANDIDATES),
        (8, IssueReason.DUPLICATE_ID),
        (9, IssueReason.BAD_TIMESTAMP),
        (10, IssueReason.DUPLICATE_CANDIDATE),
        (12, IssueReason.EMPTY_ID),
    ]
    # The CRLF row parses cleanly: no "\r" stuck to the last label.
    crlf = impressions[1]
    assert [(c.news_id, c.clicked) for c in crlf.candidates] == [("N5", False), ("N6", True)]
    # The duplicate ID row (line 8) did not replace the first impression 1.
    assert impressions[0].user_id == "U1"


def test_malformed_behaviors_strict_raises_at_first_bad_row() -> None:
    with pytest.raises(ParseError) as excinfo:
        read_behaviors(MALFORMED / "behaviors_malformed.tsv")
    assert excinfo.value.issue.line_no == 2
    assert excinfo.value.issue.reason is IssueReason.COLUMN_COUNT


def test_malformed_news_lenient() -> None:
    news, report = read_news(MALFORMED / "news_malformed.tsv", strict=False)
    assert list(news) == ["N2", "N7"]
    assert news["N2"].category == "sports"  # first N2 row kept, duplicate skipped
    assert news["N7"].abstract_entities == ""
    assert report.blank_lines == 1
    assert report.counts_by_reason() == {"column_count": 2, "duplicate_id": 1, "empty_id": 1}
    # Issue details never echo article text.
    assert all("[synthetic]" not in issue.detail for issue in report.issues)


def test_malformed_news_strict_raises() -> None:
    with pytest.raises(ParseError, match=r"news_malformed\.tsv:2: column_count"):
        read_news(MALFORMED / "news_malformed.tsv")


def test_bad_encoding_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "behaviors.tsv"
    path.write_bytes(b"1\tU1\t11/9/2019 8:05:00 AM\t\tN1-1\n\xff\xfe\n")
    impressions, report = read_behaviors(path, strict=False)
    assert len(impressions) == 1
    assert report.counts_by_reason() == {"bad_encoding": 1}
