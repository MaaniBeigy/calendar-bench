"""Unit tests for HealthTasks difficulty_level extraction and round-trip.

Covers difficulty_level extraction and persistence:

  * `ontology_bridge.fetch_difficulty_level` returns the integer Level
    for known fixtures and `0` when no Level ancestor exists.
  * `enrich_task_from_ontology` includes `difficulty_level` in its
    return dict.
  * `RecommendedTask.difficulty_level` defaults to 0 and round-trips
    through `parser._task_from_dict`, `cli._write_tasks`,
    `cli._read_tasks`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scripts.scenarios.cli import _read_tasks, _write_tasks
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.task_generation import ontology_bridge
from src.scripts.scenarios.task_generation.parser import _task_from_dict

# ---------------------------------------------------------------------------
# Helpers; mocked Neo4j session
# ---------------------------------------------------------------------------


def _session_returning(token, *, only_for_uri: str | None = None):
    """Build a mocked Neo4j session.

    The mock's session.run().single() returns a record carrying
    `{"token": token}`.  When *only_for_uri* is set, the mock returns
    None for any other URI passed in; this lets us test the URI-variant
    walk in :func:`fetch_difficulty_level`.
    """
    session = MagicMock()

    def _run(query, **kwargs):
        result = MagicMock()
        if only_for_uri is not None and kwargs.get("uri") != only_for_uri:
            result.single = MagicMock(return_value=None)
        else:
            if token is None:
                result.single = MagicMock(return_value=None)
            else:
                rec = MagicMock()
                rec.get = lambda key, default=None: {"token": token}.get(key, default)
                result.single = MagicMock(return_value=rec)
        return result

    session.run = MagicMock(side_effect=_run)
    return session


# ---------------------------------------------------------------------------
# fetch_difficulty_level
# ---------------------------------------------------------------------------


class TestFetchDifficultyLevel:
    def test_returns_3_for_level3(self):
        session = _session_returning("Level3")
        n = ontology_bridge.fetch_difficulty_level(
            "https://w3id.org/calendar-bench/health/task/cook", session
        )
        assert n == 3

    def test_returns_4_for_level4(self):
        session = _session_returning("Level4")
        n = ontology_bridge.fetch_difficulty_level("uri", session)
        assert n == 4

    def test_no_match_returns_zero(self):
        session = _session_returning(None)
        n = ontology_bridge.fetch_difficulty_level("uri", session)
        assert n == 0

    def test_invalid_token_returns_zero(self):
        session = _session_returning("Banana")
        # int("Banana") fails to caught -> 0
        n = ontology_bridge.fetch_difficulty_level("uri", session)
        assert n == 0

    def test_walks_uri_variants(self):
        # Underscore-form URI; the kebab-cased variant is what resolves.
        snake_uri = "https://w3id.org/calendar-bench/health/task/cook_meal"
        kebab_uri = "https://w3id.org/calendar-bench/health/task/cook-meal"
        session = _session_returning("Level2", only_for_uri=kebab_uri)
        n = ontology_bridge.fetch_difficulty_level(snake_uri, session)
        assert n == 2

    def test_empty_token_returns_zero(self):
        session = _session_returning("")
        n = ontology_bridge.fetch_difficulty_level("uri", session)
        assert n == 0


# ---------------------------------------------------------------------------
# enrich_task_from_ontology; includes difficulty_level
# ---------------------------------------------------------------------------


class TestEnrichIncludesDifficulty:
    def test_difficulty_level_key_always_set(self, monkeypatch):
        # Stub the three sub-functions to isolate enrich's behaviour.
        monkeypatch.setattr(
            "src.scripts.scenarios.task_generation.ontology_bridge.fetch_task_properties",
            lambda uri, session: {"display_name": "X"},
        )
        monkeypatch.setattr(
            "src.scripts.scenarios.task_generation.ontology_bridge.resolve_task_branch",
            lambda uri, session: None,
        )
        monkeypatch.setattr(
            "src.scripts.scenarios.task_generation.ontology_bridge.fetch_difficulty_level",
            lambda uri, session: 3,
        )
        out = ontology_bridge.enrich_task_from_ontology("uri", session=MagicMock())
        assert out["difficulty_level"] == 3
        assert out["display_name"] == "X"

    def test_zero_when_no_level_ancestor(self, monkeypatch):
        monkeypatch.setattr(
            "src.scripts.scenarios.task_generation.ontology_bridge.fetch_task_properties",
            lambda uri, session: {},
        )
        monkeypatch.setattr(
            "src.scripts.scenarios.task_generation.ontology_bridge.resolve_task_branch",
            lambda uri, session: None,
        )
        monkeypatch.setattr(
            "src.scripts.scenarios.task_generation.ontology_bridge.fetch_difficulty_level",
            lambda uri, session: 0,
        )
        out = ontology_bridge.enrich_task_from_ontology("uri", session=MagicMock())
        assert out["difficulty_level"] == 0


# ---------------------------------------------------------------------------
# RecommendedTask default + round-trip
# ---------------------------------------------------------------------------


class TestRecommendedTaskRoundtrip:
    def test_default_zero(self):
        t = RecommendedTask(label="x", duration_min=10, duration_max=20)
        assert t.difficulty_level == 0

    def test_parser_picks_up_field(self):
        raw = {"label": "yoga", "difficulty_level": 2}
        t = _task_from_dict(raw, task_overrides=None)
        assert t.difficulty_level == 2

    def test_parser_default_zero_when_missing(self):
        raw = {"label": "yoga"}
        t = _task_from_dict(raw, task_overrides=None)
        assert t.difficulty_level == 0

    def test_parser_handles_none_value(self):
        raw = {"label": "yoga", "difficulty_level": None}
        t = _task_from_dict(raw, task_overrides=None)
        assert t.difficulty_level == 0

    def test_overrides_win(self):
        raw = {"label": "yoga", "difficulty_level": 1}
        t = _task_from_dict(raw, task_overrides={"yoga": {"difficulty_level": 4}})
        assert t.difficulty_level == 4

    def test_write_read_roundtrip(self, tmp_path: Path):
        tasks = [
            RecommendedTask(
                label="yoga",
                duration_min=30,
                duration_max=60,
                intensity=2,
                difficulty_level=3,
            ),
            RecommendedTask(
                label="walk",
                duration_min=20,
                duration_max=40,
                intensity=2,
                difficulty_level=0,
            ),
        ]
        path = tmp_path / "tasks.json"
        _write_tasks(tasks, path)
        loaded = _read_tasks(path)
        assert loaded[0].difficulty_level == 3
        assert loaded[1].difficulty_level == 0

    def test_read_back_compat_missing_field(self, tmp_path: Path):
        path = tmp_path / "tasks.json"
        # JSON without difficulty_level; emulates pre-revision file.
        path.write_text(
            json.dumps(
                [
                    {
                        "label": "yoga",
                        "duration_min": 30,
                        "duration_max": 60,
                        "intensity": 2,
                    }
                ]
            ),
            encoding="utf-8",
        )
        loaded = _read_tasks(path)
        assert loaded[0].difficulty_level == 0


# ---------------------------------------------------------------------------
# ScheduledTask.parent_task_label default + roundtrip
# ---------------------------------------------------------------------------


class TestScheduledTaskParent:
    def test_default_none(self):
        import datetime

        st = ScheduledTask(
            task=RecommendedTask(label="x", duration_min=10, duration_max=20),
            start_minutes=0,
            end_minutes=10,
            is_standalone=True,
            concurrent_with=None,
            date=datetime.date(2026, 5, 1),
        )
        assert st.parent_task_label is None

    def test_can_set_parent(self):
        import datetime

        st = ScheduledTask(
            task=RecommendedTask(
                label="walk_morning", duration_min=20, duration_max=40
            ),
            start_minutes=0,
            end_minutes=20,
            is_standalone=True,
            concurrent_with=None,
            date=datetime.date(2026, 5, 1),
            parent_task_label="walk_10000_steps",
        )
        assert st.parent_task_label == "walk_10000_steps"
