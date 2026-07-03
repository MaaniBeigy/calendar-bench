"""Unit tests for src.scripts.scenarios.metrics.met_lookup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scripts.scenarios.metrics.met_lookup import (
    HUMAN_ACTIVITIES_PREFIX,
    MetLookup,
    MetMatch,
    build_query_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_driver_returning(record_dict):
    """Build a Neo4j-driver mock whose session.run returns *record_dict*.

    Pass `None` for *record_dict* to simulate "no result".
    """
    driver = MagicMock()
    session = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=session)
    cm.__exit__ = MagicMock(return_value=False)
    driver.session = MagicMock(return_value=cm)
    result = MagicMock()
    if record_dict is None:
        result.single = MagicMock(return_value=None)
    else:
        # Mimic a neo4j record: dict-like .get + .single()
        record = MagicMock()
        record.get = lambda key, default=None: record_dict.get(key, default)
        result.single = MagicMock(return_value=record)
    session.run = MagicMock(return_value=result)
    return driver, session


def _embedder_returning(vector):
    """Build a mock embedder whose embed_query returns *vector*."""
    e = MagicMock()
    e.embed_query = MagicMock(return_value=vector)
    return e


# ---------------------------------------------------------------------------
# build_query_text
# ---------------------------------------------------------------------------


class TestBuildQueryText:
    def test_full_recipe(self):
        text = build_query_text(
            display_name="Walk Over 10,000 Steps",
            description="Take a long walk during the day.",
            branch_local_name="PhysicalActivityTask",
            duration_minutes=60,
        )
        assert "Walk Over 10,000 Steps" in text
        assert "Take a long walk during the day." in text
        assert "PhysicalActivityTask" in text
        assert "duration 60 min" in text

    def test_only_display_name(self):
        text = build_query_text(display_name="Yoga")
        assert text == "Yoga"

    def test_skips_empty_fields(self):
        text = build_query_text(
            display_name="Yoga",
            description="",
            branch_local_name="   ",
            duration_minutes=None,
        )
        assert text == "Yoga"

    def test_zero_or_negative_duration_omitted(self):
        text = build_query_text(display_name="Yoga", duration_minutes=0)
        assert text == "Yoga"
        text = build_query_text(display_name="Yoga", duration_minutes=-5)
        assert text == "Yoga"

    def test_only_duration(self):
        # Only duration to produces a duration-prefix only string.
        text = build_query_text(display_name="", duration_minutes=30)
        assert text == "duration 30 min"


# ---------------------------------------------------------------------------
# MetLookup; cache-only mode (no driver / embedder)
# ---------------------------------------------------------------------------


class TestMetLookupCacheOnlyMode:
    def test_no_driver_returns_empty_match(self, tmp_path: Path):
        lk = MetLookup(driver=None, embedder=None, cache_path=tmp_path / "c.jsonl")
        m = lk.met("walk")
        assert m == MetMatch(met=None, uri=None, cosine=None)

    def test_empty_query_returns_empty_match(self, tmp_path: Path):
        lk = MetLookup(driver=None, embedder=None, cache_path=tmp_path / "c.jsonl")
        assert lk.met("") == MetMatch(met=None, uri=None, cosine=None)
        assert lk.met("   ") == MetMatch(met=None, uri=None, cosine=None)

    def test_cache_persists_empty_match(self, tmp_path: Path):
        cache = tmp_path / "c.jsonl"
        lk = MetLookup(driver=None, embedder=None, cache_path=cache)
        lk.met("walk")
        # Cache should have been written.
        assert cache.exists()
        lines = [
            json.loads(line)
            for line in cache.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert len(lines) == 1
        assert lines[0]["query_text"] == "walk"
        assert lines[0]["met"] is None


# ---------------------------------------------------------------------------
# MetLookup; live lookup
# ---------------------------------------------------------------------------


class TestMetLookupLiveQuery:
    def test_high_cosine_match_returned(self, tmp_path: Path):
        record = {
            "uri": "https://w3id.org/calendar-bench/human-activities/activity/walk",
            "met": 3.5,
            "score": 0.81,
        }
        driver, _ = _mock_driver_returning(record)
        embedder = _embedder_returning([0.1, 0.2, 0.3])
        lk = MetLookup(
            driver=driver,
            embedder=embedder,
            cache_path=tmp_path / "c.jsonl",
            min_cosine=0.55,
        )
        m = lk.met("walk")
        assert m.met == pytest.approx(3.5)
        assert m.uri.endswith("/walk")
        assert m.cosine == pytest.approx(0.81)

    def test_below_threshold_returns_none_met(self, tmp_path: Path):
        record = {"uri": "https://example.org/foo", "met": 4.0, "score": 0.30}
        driver, _ = _mock_driver_returning(record)
        embedder = _embedder_returning([0.0])
        lk = MetLookup(
            driver=driver,
            embedder=embedder,
            cache_path=tmp_path / "c.jsonl",
            min_cosine=0.55,
        )
        m = lk.met("foo")
        assert m.met is None
        assert m.uri is None
        assert m.cosine == pytest.approx(0.30)

    def test_no_record_returns_empty_match(self, tmp_path: Path):
        driver, _ = _mock_driver_returning(None)
        embedder = _embedder_returning([0.0])
        lk = MetLookup(
            driver=driver, embedder=embedder, cache_path=tmp_path / "c.jsonl"
        )
        m = lk.met("nonexistent")
        assert m == MetMatch(met=None, uri=None, cosine=None)

    def test_query_arguments_passed_through(self, tmp_path: Path):
        record = {"uri": "u", "met": 1.0, "score": 1.0}
        driver, session = _mock_driver_returning(record)
        embedder = _embedder_returning([0.1])
        lk = MetLookup(
            driver=driver,
            embedder=embedder,
            cache_path=tmp_path / "c.jsonl",
            top_k=8,
            prefix="https://example.org/",
        )
        lk.met("walk")
        kwargs = session.run.call_args.kwargs
        assert kwargs["index"] == "concept_embedding_index"
        assert kwargs["top_k"] == 8
        assert kwargs["vector"] == [0.1]
        assert kwargs["prefix"] == "https://example.org/"


# ---------------------------------------------------------------------------
# Cache load / save
# ---------------------------------------------------------------------------


class TestMetLookupCacheRoundtrip:
    def test_loads_existing_cache(self, tmp_path: Path):
        cache = tmp_path / "c.jsonl"
        cache.write_text(
            json.dumps(
                {
                    "query_text": "running",
                    "met": 7.5,
                    "uri": "https://example.org/run",
                    "cosine": 0.91,
                    "ts": "2026-05-13T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        lk = MetLookup(driver=None, embedder=None, cache_path=cache)
        m = lk.met("running")
        assert m.met == pytest.approx(7.5)
        assert m.uri == "https://example.org/run"
        assert m.cosine == pytest.approx(0.91)

    def test_skips_corrupt_lines(self, tmp_path: Path):
        cache = tmp_path / "c.jsonl"
        cache.write_text(
            "not json\n"
            + json.dumps({"query_text": "x", "met": 2.0, "uri": "u", "cosine": 0.7})
            + "\n"
            + '{"missing": "keys"}\n',
            encoding="utf-8",
        )
        lk = MetLookup(driver=None, embedder=None, cache_path=cache)
        # Only the well-formed entry should be cached.
        assert lk.met("x").met == pytest.approx(2.0)

    def test_appends_new_entry(self, tmp_path: Path):
        cache = tmp_path / "c.jsonl"
        record = {"uri": "u", "met": 3.0, "score": 0.8}
        driver, _ = _mock_driver_returning(record)
        embedder = _embedder_returning([0.1])
        lk = MetLookup(
            driver=driver, embedder=embedder, cache_path=cache, min_cosine=0.5
        )
        lk.met("walk")
        lk.met("walk")  # second call should hit cache
        # Driver session.run call count: once.
        assert driver.session.call_count == 1
        # Cache file has exactly one record.
        text = cache.read_text(encoding="utf-8")
        lines = [json.loads(line) for line in text.splitlines() if line]
        assert len(lines) == 1

    def test_cache_path_none_skips_disk(self, tmp_path: Path):
        record = {"uri": "u", "met": 1.0, "score": 0.7}
        driver, _ = _mock_driver_returning(record)
        embedder = _embedder_returning([0.1])
        lk = MetLookup(
            driver=driver, embedder=embedder, cache_path=None, min_cosine=0.5
        )
        m = lk.met("walk")
        assert m.met == pytest.approx(1.0)
        # Calling again should hit in-memory cache regardless.
        lk.met("walk")
        assert driver.session.call_count == 1

    def test_creates_parent_dirs(self, tmp_path: Path):
        cache = tmp_path / "a" / "b" / "c.jsonl"
        record = {"uri": "u", "met": 1.0, "score": 0.7}
        driver, _ = _mock_driver_returning(record)
        embedder = _embedder_returning([0.1])
        lk = MetLookup(
            driver=driver, embedder=embedder, cache_path=cache, min_cosine=0.5
        )
        lk.met("walk")
        assert cache.exists()


def test_default_prefix_is_human_activities():
    lk = MetLookup(driver=None, embedder=None)
    assert lk._prefix == HUMAN_ACTIVITIES_PREFIX
