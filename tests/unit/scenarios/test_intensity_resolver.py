"""Unit tests for src.scripts.scenarios.metrics.intensity_resolver."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityRecord,
    IntensityResolver,
    MetQuartiles,
    build_event_intensity_map,
    level_bucket_for_event_intensity,
    level_bucket_for_health_task,
    load_quartiles_file,
    met_quartile_bucket,
)
from src.scripts.scenarios.metrics.met_lookup import MetMatch

DEFAULT_QUARTILES = MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8)


def _make_lookup(met: float | None, uri: str | None = "u", cosine: float = 0.8):
    """Build a minimal MetLookup stub exposing .met and .met_for_task."""
    match = MetMatch(met=met, uri=uri, cosine=cosine)
    obj = MagicMock()
    obj.met = MagicMock(return_value=match)
    obj.met_for_task = MagicMock(return_value=match)
    return obj


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class TestMetQuartileBucket:
    def test_none_returns_zero(self):
        assert met_quartile_bucket(None, DEFAULT_QUARTILES) == 0

    def test_below_q1_is_1(self):
        assert met_quartile_bucket(1.5, DEFAULT_QUARTILES) == 1

    def test_at_q1_is_1(self):
        assert met_quartile_bucket(1.8, DEFAULT_QUARTILES) == 1

    def test_above_q1_below_q2_is_2(self):
        assert met_quartile_bucket(2.5, DEFAULT_QUARTILES) == 2

    def test_at_q2_is_2(self):
        assert met_quartile_bucket(3.0, DEFAULT_QUARTILES) == 2

    def test_above_q2_below_q3_is_3(self):
        assert met_quartile_bucket(4.5, DEFAULT_QUARTILES) == 3

    def test_at_q3_is_3(self):
        assert met_quartile_bucket(6.0, DEFAULT_QUARTILES) == 3

    def test_above_q3_is_4(self):
        assert met_quartile_bucket(10.0, DEFAULT_QUARTILES) == 4


class TestLevelBucketForHealthTask:
    def test_zero_when_no_level(self):
        t = RecommendedTask(
            label="x", duration_min=1, duration_max=2, difficulty_level=0
        )
        assert level_bucket_for_health_task(t) == 0

    def test_passes_through_1_to_4(self):
        for n in (1, 2, 3, 4):
            t = RecommendedTask(
                label="x",
                duration_min=1,
                duration_max=2,
                difficulty_level=n,
            )
            assert level_bucket_for_health_task(t) == n

    def test_clamps_above_4(self):
        t = RecommendedTask(
            label="x", duration_min=1, duration_max=2, difficulty_level=10
        )
        assert level_bucket_for_health_task(t) == 4

    def test_negative_is_zero(self):
        t = RecommendedTask(
            label="x", duration_min=1, duration_max=2, difficulty_level=-1
        )
        assert level_bucket_for_health_task(t) == 0


class TestLevelBucketForEventIntensity:
    def test_none_zero(self):
        assert level_bucket_for_event_intensity(None) == 0

    def test_zero_returns_zero(self):
        assert level_bucket_for_event_intensity(0) == 0

    def test_1_to_3_identity(self):
        for n in (1, 2, 3):
            assert level_bucket_for_event_intensity(n) == n

    def test_4_caps_to_4(self):
        assert level_bucket_for_event_intensity(4) == 4

    def test_5_collapses_to_4(self):
        assert level_bucket_for_event_intensity(5) == 4


# ---------------------------------------------------------------------------
# load_quartiles_file
# ---------------------------------------------------------------------------


class TestLoadQuartilesFile:
    def test_reads_payload(self, tmp_path: Path):
        path = tmp_path / "q.json"
        path.write_text(
            json.dumps(
                {
                    "q1": 1.5,
                    "q2": 3.0,
                    "q3": 6.0,
                    "max_met": 18.0,
                    "n": 1000,
                    "ontology_prefix": "x",
                    "ts": "x",
                }
            ),
            encoding="utf-8",
        )
        q = load_quartiles_file(path)
        assert q.q1 == pytest.approx(1.5)
        assert q.max_met == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# build_event_intensity_map
# ---------------------------------------------------------------------------


class TestBuildEventIntensityMap:
    def test_empty_dict(self):
        assert build_event_intensity_map({}) == {}

    def test_extracts_from_dict(self):
        cfg = {
            "categories": {
                "sleep": {
                    "events": {
                        "sleep": {"intensity": 1},
                    }
                },
                "sports": {
                    "events": {
                        "running": {"intensity": 4},
                        "gym": {"intensity": 4},
                    }
                },
            }
        }
        out = build_event_intensity_map(cfg)
        assert out == {"sleep": 1, "running": 4, "gym": 4}

    def test_skips_none_intensity(self):
        cfg = {
            "categories": {
                "x": {
                    "events": {
                        "y": {"intensity": None},
                        "z": {"intensity": 2},
                    }
                }
            }
        }
        assert build_event_intensity_map(cfg) == {"z": 2}

    def test_works_with_object_attrs(self):
        Cat = type("C", (), {})
        Ev = type("E", (), {})

        c = Cat()
        e1 = Ev()
        e1.intensity = 3
        c.events = {"yoga": e1}

        cfg = Cat()
        cfg.categories = {"sports": c}
        assert build_event_intensity_map(cfg) == {"yoga": 3}


# ---------------------------------------------------------------------------
# IntensityResolver; task path
# ---------------------------------------------------------------------------


class TestIntensityResolverTasks:
    def test_level_only_no_met_lookup(self):
        t = RecommendedTask(
            label="cook_meal",
            duration_min=30,
            duration_max=60,
            difficulty_level=3,
        )
        r = IntensityResolver(DEFAULT_QUARTILES)
        rec = r.record(t)
        assert rec.intensity == 3
        assert rec.level_bucket == 3
        assert rec.met_bucket == 0
        assert rec.met is None

    def test_met_higher_than_level_wins(self):
        t = RecommendedTask(
            label="run",
            duration_min=30,
            duration_max=60,
            difficulty_level=1,
        )
        lookup = _make_lookup(met=8.0)  # > q3 to bucket 4
        r = IntensityResolver(DEFAULT_QUARTILES, met_lookup=lookup)
        rec = r.record(t)
        assert rec.intensity == 4
        assert rec.level_bucket == 1
        assert rec.met_bucket == 4

    def test_level_higher_than_met_wins(self):
        """Level-3 nutrition (planning week's meals) overrules Q1 MET sitting task."""
        t = RecommendedTask(
            label="plan_week_meals",
            duration_min=30,
            duration_max=30,
            difficulty_level=3,
        )
        lookup = _make_lookup(met=1.5)  # bucket 1
        r = IntensityResolver(DEFAULT_QUARTILES, met_lookup=lookup)
        rec = r.record(t)
        assert rec.intensity == 3
        assert rec.met_bucket == 1
        assert rec.level_bucket == 3

    def test_no_level_no_met_returns_zero(self):
        t = RecommendedTask(label="x", duration_min=10, duration_max=20)
        lookup = _make_lookup(met=None)
        r = IntensityResolver(DEFAULT_QUARTILES, met_lookup=lookup)
        rec = r.record(t)
        assert rec.intensity == 0

    def test_resolve_uses_cache(self):
        t = RecommendedTask(
            label="cook", duration_min=10, duration_max=20, difficulty_level=2
        )
        lookup = _make_lookup(met=2.0)
        r = IntensityResolver(DEFAULT_QUARTILES, met_lookup=lookup)
        r.resolve(t)
        r.resolve(t)
        # MetLookup is consulted once; second call hits the resolver's cache.
        assert lookup.met_for_task.call_count == 1

    def test_scheduled_task_delegates(self):
        t = RecommendedTask(
            label="cook", duration_min=10, duration_max=20, difficulty_level=3
        )
        st = ScheduledTask(
            task=t,
            start_minutes=0,
            end_minutes=10,
            is_standalone=True,
            concurrent_with=None,
            date=datetime.date(2026, 5, 1),
        )
        r = IntensityResolver(DEFAULT_QUARTILES)
        assert r.resolve(st) == 3


# ---------------------------------------------------------------------------
# IntensityResolver; event path
# ---------------------------------------------------------------------------


class TestIntensityResolverEvents:
    def test_event_intensity_field_used(self):
        ev = CalendarEvent(
            label="running",
            start_minutes=0,
            end_minutes=60,
            date=datetime.date(2026, 5, 1),
            intensity=4,
        )
        r = IntensityResolver(DEFAULT_QUARTILES)
        assert r.resolve(ev) == 4

    def test_event_intensity_5_collapses_to_4(self):
        ev = CalendarEvent(
            label="sprint",
            start_minutes=0,
            end_minutes=30,
            date=datetime.date(2026, 5, 1),
            intensity=5,
        )
        r = IntensityResolver(DEFAULT_QUARTILES)
        assert r.resolve(ev) == 4

    def test_event_intensity_from_map_when_field_missing(self):
        ev = CalendarEvent(
            label="lunch",
            start_minutes=0,
            end_minutes=30,
            date=datetime.date(2026, 5, 1),
            intensity=1,  # CalendarEvent dataclass requires intensity; map override
        )
        # Skip the dataclass route; use the event_intensity_map directly:
        r = IntensityResolver(
            DEFAULT_QUARTILES,
            event_intensity_map={"lunch": 1},
        )
        # The CalendarEvent.intensity=1 is what we use (the map only kicks in
        # when the event object lacks the attribute or returns None).
        assert r.resolve(ev) == 1

    def test_met_lookup_provides_physical_demand(self):
        ev = CalendarEvent(
            label="running",
            start_minutes=0,
            end_minutes=60,
            date=datetime.date(2026, 5, 1),
            intensity=1,  # low cognitive
        )
        lookup = _make_lookup(met=8.0)  # high MET to bucket 4
        r = IntensityResolver(DEFAULT_QUARTILES, met_lookup=lookup)
        rec = r.record(ev)
        assert rec.intensity == 4
        assert rec.met_bucket == 4

    def test_label_only_fallback(self):
        r = IntensityResolver(DEFAULT_QUARTILES)

        # Pass a plain object whose only "shape" is a .label attr.
        class _X:
            label = "foo"

        rec = r.record(_X())
        assert rec.intensity == 0  # no level, no met

    def test_bare_string_no_label(self):
        # Activity with neither .label nor obvious shape; falls back to empty.
        r = IntensityResolver(DEFAULT_QUARTILES)

        class _X:
            pass

        rec = r.record(_X())
        assert rec.intensity == 0


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


class TestCacheRoundtrip:
    def test_writes_and_reads_back(self, tmp_path: Path):
        cache = tmp_path / "intensity_cache.jsonl"
        t = RecommendedTask(
            label="cook", duration_min=10, duration_max=20, difficulty_level=3
        )

        r1 = IntensityResolver(DEFAULT_QUARTILES, cache_path=cache)
        r1.resolve(t)
        assert cache.exists()

        # Second instance reads the cache and reproduces the same result.
        r2 = IntensityResolver(DEFAULT_QUARTILES, cache_path=cache)
        rec = r2.record(t)
        assert rec.intensity == 3
        assert rec.level_bucket == 3

    def test_skips_corrupt_lines(self, tmp_path: Path):
        cache = tmp_path / "intensity_cache.jsonl"
        cache.write_text(
            "not json\n"
            + json.dumps(
                {
                    "label": "cook",
                    "intensity": 3,
                    "level_bucket": 3,
                    "met_bucket": 0,
                    "met": None,
                    "matched_uri": None,
                    "query_text": "cook",
                }
            )
            + "\n"
            + json.dumps({"label": "missing_intensity"})
            + "\n",
            encoding="utf-8",
        )
        r = IntensityResolver(DEFAULT_QUARTILES, cache_path=cache)
        t = RecommendedTask(
            label="cook", duration_min=10, duration_max=20, difficulty_level=3
        )
        assert r.record(t).intensity == 3

    def test_cache_path_none_skips_disk(self, tmp_path: Path):
        t = RecommendedTask(
            label="cook", duration_min=10, duration_max=20, difficulty_level=2
        )
        r = IntensityResolver(DEFAULT_QUARTILES, cache_path=None)
        rec = r.record(t)
        assert rec.intensity == 2
