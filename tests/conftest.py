"""Shared paths and parsed fixtures. All fixture data is synthetic (see PROVENANCE.json)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recsys.loaders import DEV_DIR, TRAIN_DIR, MindDir, load_mind_dir

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC = FIXTURES / "mind_synthetic"
MALFORMED = FIXTURES / "malformed"


@pytest.fixture(scope="session")
def train() -> MindDir:
    return load_mind_dir(SYNTHETIC / TRAIN_DIR)


@pytest.fixture(scope="session")
def dev() -> MindDir:
    return load_mind_dir(SYNTHETIC / DEV_DIR)
