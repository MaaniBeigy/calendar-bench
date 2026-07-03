"""Cache I/O edge cases in `MetLookup`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.scripts.scenarios.metrics.met_lookup import MetLookup


def test_load_cache_swallows_oserror(tmp_path: Path):
    """An `OSError` while reading the cache file leaves the cache empty."""
    cache_file = tmp_path / "met_cache.jsonl"
    cache_file.write_text("", encoding="utf-8")
    with patch.object(Path, "read_text", side_effect=OSError("vanished")):
        lookup = MetLookup(cache_path=cache_file)
    assert lookup._cache == {}  # type: ignore[attr-defined]


def test_load_cache_skips_empty_and_malformed_lines(tmp_path: Path):
    """Empty lines are skipped and malformed JSON is swallowed."""
    cache_file = tmp_path / "met_cache.jsonl"
    cache_file.write_text(
        "\n"
        "   \n"
        "not json\n"
        '{"query_text": "walking", "met": 3.5, "uri": "u", "cosine": 0.9}\n',
        encoding="utf-8",
    )
    lookup = MetLookup(cache_path=cache_file)
    assert "walking" in lookup._cache  # type: ignore[attr-defined]
