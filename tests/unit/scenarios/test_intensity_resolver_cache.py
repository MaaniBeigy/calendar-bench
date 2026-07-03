"""Cache I/O and edge-case helpers in `IntensityResolver`."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityRecord,
    IntensityResolver,
    MetQuartiles,
    build_event_intensity_map,
)

DATE = datetime.date(2026, 5, 1)
DEFAULT_QUARTILES = MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8)


def test_build_event_intensity_map_skips_category_with_no_events():
    """A category with no events is walked past, the next one is picked up."""
    cfg = {
        "categories": {
            "empty_cat": {"events": {}},
            "test": {
                "events": {
                    "walking": {"intensity": 2},
                    "no_intensity": {},
                }
            },
        }
    }
    out = build_event_intensity_map(cfg)
    assert out == {"walking": 2}


def test_record_for_task_skips_duration_when_min_or_max_is_zero():
    """`duration_min=0` short-circuits the duration disambiguator."""
    resolver = IntensityResolver(DEFAULT_QUARTILES)
    task = RecommendedTask(
        label="ghost",
        duration_min=0,
        duration_max=30,
        difficulty_level=2,
    )
    record = resolver.record(task)
    assert isinstance(record, IntensityRecord)
    assert record.met is None


def test_record_for_event_returns_cached_record_on_second_call():
    """Two successive resolutions of the same label return the same record."""
    resolver = IntensityResolver(DEFAULT_QUARTILES)
    ev = CalendarEvent(
        label="lunch",
        start_minutes=720,
        end_minutes=750,
        date=DATE,
        intensity=2,
    )
    first = resolver.record(ev)
    second = resolver.record(ev)
    assert first is second


def test_load_cache_swallows_oserror(tmp_path: Path):
    """An `OSError` while reading the cache file leaves the cache empty."""
    cache_file = tmp_path / "intensity_cache.jsonl"
    cache_file.write_text("", encoding="utf-8")
    with patch.object(Path, "read_text", side_effect=OSError("vanished")):
        resolver = IntensityResolver(DEFAULT_QUARTILES, cache_path=cache_file)
    assert resolver._cache == {}  # type: ignore[attr-defined]


def test_load_cache_skips_empty_and_malformed_lines(tmp_path: Path):
    """Empty lines are skipped and malformed JSON is swallowed."""
    cache_file = tmp_path / "intensity_cache.jsonl"
    cache_file.write_text(
        "\n"
        "   \n"
        "not json\n"
        '{"label": "walking", "intensity": 2, "level_bucket": 2, '
        '"met_bucket": 2, "met": 3.0, "matched_uri": "u", '
        '"query_text": "walking"}\n',
        encoding="utf-8",
    )
    resolver = IntensityResolver(DEFAULT_QUARTILES, cache_path=cache_file)
    assert "walking" in resolver._cache  # type: ignore[attr-defined]
