"""Test fixtures for CFST extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
GOLD_DIR = TESTDATA_DIR / "jsondata"
PDF_DIR = TESTDATA_DIR / "pdfs"


@pytest.fixture
def gold_dir():
    return GOLD_DIR


@pytest.fixture
def pdf_dir():
    return PDF_DIR


@pytest.fixture
def load_gold():
    """Factory fixture to load a gold-standard JSON by paper ID prefix."""
    def _load(paper_id: str) -> dict:
        # Try exact match first
        candidates = list(GOLD_DIR.glob(f"{paper_id}*.json"))
        if not candidates:
            candidates = list(GOLD_DIR.glob(f"*{paper_id}*.json"))
        if not candidates:
            pytest.skip(f"No gold JSON found for {paper_id}")
        with open(candidates[0], encoding="utf-8") as f:
            return json.load(f)
    return _load
