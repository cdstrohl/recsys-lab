from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SYNTHETIC

from recsys.inventory import build_inventory, history_bucket

REPO = Path(__file__).parent.parent


@pytest.mark.parametrize(
    ("length", "edges", "expected"),
    [
        (0, (0, 1, 6), "0"),
        (1, (0, 1, 6), "1-5"),
        (5, (0, 1, 6), "1-5"),
        (6, (0, 1, 6), "6+"),
        (500, (0, 1, 6), "6+"),
        (1, (0, 1, 2, 11), "1"),
        (10, (0, 1, 2, 11), "2-10"),
        (11, (0, 1, 2, 11), "11+"),
    ],
)
def test_history_bucket(length: int, edges: tuple[int, ...], expected: str) -> None:
    assert history_bucket(length, edges) == expected


@pytest.mark.parametrize("edges", [(), (1, 5), (0, 6, 1), (0, 1, 1)])
def test_history_bucket_rejects_bad_edges(edges: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        history_bucket(3, edges)


def test_inventory_on_synthetic_fixtures() -> None:
    inv = build_inventory(SYNTHETIC)
    assert inv["data_provenance"] == "synthetic"

    train = inv["MINDsmall_train"]
    assert train["parse"]["behaviors.tsv"] == {"rows_parsed": 11, "blank_lines": 0, "issues": {}}
    assert train["news"]["n_items"] == 14
    assert train["news"]["n_empty_abstract"] == 2  # N2, N7
    assert train["news"]["categories"]["sports"] == 4  # N1, N2, N8, N13

    b = train["behaviors"]
    assert b["n_impressions"] == 11
    assert b["n_users"] == 6
    assert b["first_timestamp"] == "2019-11-09T08:05:00"
    assert b["last_timestamp"] == "2019-11-11T21:00:00"
    assert b["impressions_per_day"] == {"2019-11-09": 3, "2019-11-10": 4, "2019-11-11": 4}
    # 3+4+3+12+2+1+2+3+2+2+3 candidates; clicks 1+2+0+1+2+1+1+1+1+1+1.
    assert b["n_candidates"] == 37
    assert b["n_clicks"] == 12
    assert b["n_no_click_impressions"] == 1  # impression 3
    assert b["n_all_click_impressions"] == 2  # impressions 5 and 6
    assert b["n_single_candidate_impressions"] == 1  # impression 6
    assert b["candidates_per_impression"]["max"] == 12
    # Per impression: empty history in 2 and 7; U6's 7-item history in 9.
    assert b["history_length_per_impression"] == {"0": 2, "1-5": 8, "6+": 1}
    # Per user: U2 empty; U1, U3, U4, U5 have 1-2 items; U6 has 7.
    assert b["history_length_per_user"] == {"0": 1, "1-5": 4, "6+": 1}
    assert b["n_users_inconsistent_history"] == 0
    assert b["n_candidate_ids_missing_from_news"] == 0

    dev = inv["MINDsmall_dev"]["behaviors"]
    assert dev["n_impressions"] == 3
    assert dev["n_no_click_impressions"] == 1
    assert dev["n_candidate_ids_missing_from_news"] == 1  # N99

    unseen = inv["dev_vs_train"]
    # Dev candidates: N15 N3 N16 | N17 N13 N18 N99 | N15 N16. Only N3 and N13 occur in train.
    assert unseen["candidate_occurrences"] == 9
    assert unseen["unique_candidates"] == 7
    assert unseen["share_occurrences_unseen_as_candidate"] == pytest.approx(7 / 9)
    assert unseen["share_unique_unseen_as_candidate"] == pytest.approx(5 / 7)
    assert unseen["share_unique_unseen_anywhere"] == pytest.approx(5 / 7)
    assert unseen["share_history_items_seen_as_candidate"] == 1.0  # N1, N2, N6
    assert unseen["share_users_unseen"] == pytest.approx(1 / 3)  # U7
    assert unseen["n_impressions_before_reference_end"] == 0


def test_inspect_data_script(tmp_path: Path) -> None:
    out = tmp_path / "inventory.json"
    result = subprocess.run(
        [sys.executable, "scripts/inspect_data.py", "--data", str(SYNTHETIC), "--json", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "data_provenance: synthetic" in result.stdout
    assert "[synthetic]" not in result.stdout  # aggregates only, no titles
    assert json.loads(out.read_text())["MINDsmall_train"]["behaviors"]["n_impressions"] == 11


def test_inspect_data_script_missing_data(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/inspect_data.py", "--data", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "data/README.md" in result.stderr
