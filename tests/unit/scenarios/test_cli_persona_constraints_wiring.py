"""Tests for the CLI helpers that wire L_pref v2 into augment.

Covers:

* :func:`_load_person_json_lookup`; reads `index.json` and returns a
  `{person_id: json_path}` mapping; tolerates missing / malformed files.
* :func:`_build_persona_constraints_for_trace`; materialises a
  :class:`PersonaConstraints` from a per-person snapshot, or returns
  `None` when wiring is incomplete (so the loss dispatcher falls back
  to the legacy preference scorer instead of crashing).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from unittest.mock import MagicMock

import fakeredis
import pytest

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.cli import (
    _build_persona_constraints_for_trace,
    _load_person_json_lookup,
    _wire_rl_constraints,
)
from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
from src.scripts.scenarios.metrics.preference_constraints import PersonaConstraints
from src.scripts.scenarios.metrics.preference_mapping import PreferenceMapper


def _wm() -> WindowMap:
    return WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        }
    )


def _event_config() -> EventConfig:
    walking = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
    )
    return EventConfig(
        categories={"sports": Category(name="sports", events={"walking": walking})}
    )


def _trace_with_id(person_id: str):
    tr = MagicMock()
    tr.person_id = person_id
    return tr


def _write_index(run_dir: Path, persons: list[dict]) -> None:
    (run_dir / "index.json").write_text(
        json.dumps({"persons": persons}), encoding="utf-8"
    )


def _write_person_json(path: Path, persona_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "person_id": persona_data.get("person_id", "p1"),
                "persona": persona_data,
                "days": [],
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# _load_person_json_lookup
# ---------------------------------------------------------------------------


class TestLoadPersonJsonLookup:
    def test_returns_mapping_from_index(self, tmp_path: Path):
        _write_index(
            tmp_path,
            [
                {"person_id": "p1", "json_path": "persons/p1.json"},
                {"person_id": "p2", "json_path": "persons/p2.json"},
            ],
        )
        out = _load_person_json_lookup(tmp_path)
        assert out == {
            "p1": "persons/p1.json",
            "p2": "persons/p2.json",
        }

    def test_missing_index_returns_empty(self, tmp_path: Path):
        assert _load_person_json_lookup(tmp_path) == {}

    def test_malformed_index_returns_empty(self, tmp_path: Path):
        (tmp_path / "index.json").write_text("not-json", encoding="utf-8")
        assert _load_person_json_lookup(tmp_path) == {}

    def test_index_missing_person_id_skipped(self, tmp_path: Path):
        _write_index(
            tmp_path,
            [
                {"json_path": "orphan.json"},
                {"person_id": "p1", "json_path": "persons/p1.json"},
            ],
        )
        out = _load_person_json_lookup(tmp_path)
        assert out == {"p1": "persons/p1.json"}


# ---------------------------------------------------------------------------
# _build_persona_constraints_for_trace
# ---------------------------------------------------------------------------


class TestBuildPersonaConstraintsForTrace:
    def test_returns_none_when_event_config_missing(self, tmp_path: Path):
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("p1"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/p1.json"},
            event_config=None,
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=None,
            mapper=None,
            experiment="e",
            scenario="s",
        )
        assert out is None

    def test_returns_none_when_window_map_missing(self, tmp_path: Path):
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("p1"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/p1.json"},
            event_config=_event_config(),
            window_map=None,
            horizon_days=28,
            horizon_start_date=None,
            mapper=None,
            experiment="e",
            scenario="s",
        )
        assert out is None

    def test_returns_none_when_trace_not_in_lookup(self, tmp_path: Path):
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("absent"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/p1.json"},
            event_config=_event_config(),
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=None,
            mapper=None,
            experiment="e",
            scenario="s",
        )
        assert out is None

    def test_returns_none_when_person_json_missing(self, tmp_path: Path):
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("p1"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/missing.json"},
            event_config=_event_config(),
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=None,
            mapper=None,
            experiment="e",
            scenario="s",
        )
        assert out is None

    def test_returns_none_when_persona_block_empty(self, tmp_path: Path):
        person_path = tmp_path / "persons/p1.json"
        person_path.parent.mkdir(parents=True, exist_ok=True)
        person_path.write_text(
            json.dumps({"person_id": "p1", "persona": {}, "days": []}),
            encoding="utf-8",
        )
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("p1"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/p1.json"},
            event_config=_event_config(),
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=None,
            mapper=None,
            experiment="e",
            scenario="s",
        )
        assert out is None

    def test_builds_persona_constraints_from_snapshot(self, tmp_path: Path):
        _write_person_json(
            tmp_path / "persons/p1.json",
            {
                "person_id": "p1",
                "persona_id": "parttime",
                "occupation_status": "parttime",
                "stages": [
                    {
                        "name": "walking",
                        "time": "morning",
                        "duration_minutes": 30,
                        "days": ["Mon"],
                    }
                ],
                "event_overrides": {},
            },
        )
        cache = PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))
        mapper = PreferenceMapper(cache=cache)
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("p1"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/p1.json"},
            event_config=_event_config(),
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=_dt.date(2026, 6, 1),
            mapper=mapper,
            experiment="exp_b",
            scenario="scen1",
        )
        assert isinstance(out, PersonaConstraints)
        # Stage list survived the round-trip.
        fired = out.stages_firing_on(_dt.date(2026, 6, 1))  # Monday
        assert len(fired) == 1
        assert fired[0].name == "walking"

    def test_builds_constraints_when_snapshot_lacks_occupation(self, tmp_path: Path):
        """A characteristics-only cohort snapshot must not be rejected."""
        _write_person_json(
            tmp_path / "persons/p1.json",
            {
                "person_id": "p1",
                "persona_id": "richpersona",
                "characteristics": {"gender": "female", "age": 41},
                "stages": [
                    {
                        "name": "walking",
                        "time": "morning",
                        "duration_minutes": 30,
                        "days": ["Mon"],
                    }
                ],
                "event_overrides": {},
            },
        )
        cache = PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))
        mapper = PreferenceMapper(cache=cache)
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("p1"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/p1.json"},
            event_config=_event_config(),
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=_dt.date(2026, 6, 1),
            mapper=mapper,
            experiment="exp_b",
            scenario="scen1",
        )
        assert isinstance(out, PersonaConstraints)

    def test_returns_none_on_schema_validation_failure(self, tmp_path: Path):
        """Schema validation failures are swallowed to `None` so the loss
        dispatcher falls back to legacy compute_l_pref."""
        _write_person_json(
            tmp_path / "persons/p1.json",
            {
                "person_id": "p1",
                "occupation_status": "student",
                # Invalid: `days` is not a list, so PersonaEventStage rejects it.
                "stages": [
                    {"name": "walking", "time": "morning", "days": "not a list"}
                ],
                "event_overrides": {},
            },
        )
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("p1"),
            run_dir=tmp_path,
            person_json_lookup={"p1": "persons/p1.json"},
            event_config=_event_config(),
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=None,
            mapper=None,
            experiment="e",
            scenario="s",
        )
        assert out is None

    def test_fallback_id_when_persona_id_absent(self, tmp_path: Path):
        """No `persona_id` / `person_id` in the snapshot to fall back to
        the trace's `person_id` for the Persona's id."""
        # Write a minimal snapshot; keys deliberately omit persona_id.
        person_path = tmp_path / "persons/p1.json"
        person_path.parent.mkdir(parents=True, exist_ok=True)
        person_path.write_text(
            json.dumps(
                {
                    "person_id": "from-trace",
                    "persona": {
                        "occupation_status": "parttime",
                        "stages": [],
                        "event_overrides": {},
                    },
                    "days": [],
                }
            ),
            encoding="utf-8",
        )
        out = _build_persona_constraints_for_trace(
            trace=_trace_with_id("from-trace"),
            run_dir=tmp_path,
            person_json_lookup={"from-trace": "persons/p1.json"},
            event_config=_event_config(),
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=None,
            mapper=None,
            experiment="e",
            scenario="s",
        )
        assert isinstance(out, PersonaConstraints)
        assert out.persona.id == "from-trace"


def _write_event_config_yaml(path: Path) -> None:
    path.write_text(
        "categories:\n"
        "  sports:\n"
        "    events:\n"
        "      walking:\n"
        "        category: sports\n"
        "        per_event_duration: {min: 15, max: 120, unit: minutes}\n"
        "        total_event_duration: {scale: day, min: 15, max: 240, unit: minutes}\n"
        "        total_event_episodes: {scale: day, min: 0, max: 1}\n",
        encoding="utf-8",
    )


def _run_with_windows():
    from types import SimpleNamespace

    return SimpleNamespace(
        time_windows={
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        },
        horizon_days=7,
        horizon_start_date=_dt.date(2026, 6, 1),
    )


class TestWireRlConstraints:
    def test_installs_callable_factories(self, tmp_path: Path):
        from types import SimpleNamespace

        event_yaml = tmp_path / "event_config.yaml"
        _write_event_config_yaml(event_yaml)
        _write_index(tmp_path, [])

        captured: dict = {}

        class _Aug:
            def set_constraint_factories(self, constraints_for, window_map_for):
                captured["constraints_for"] = constraints_for
                captured["window_map_for"] = window_map_for

        cfg = SimpleNamespace(
            calendar=SimpleNamespace(persona_events=str(event_yaml)),
            output=SimpleNamespace(dir=str(tmp_path)),
            id="s",
            evaluator_model=None,
            experiment="exp",
        )
        _wire_rl_constraints(_Aug(), run=_run_with_windows(), run_dir=tmp_path, cfg=cfg)
        assert callable(captured.get("constraints_for"))
        wmf = captured.get("window_map_for")
        assert callable(wmf)
        # The window-map factory hands back one shared map for every trace.
        assert wmf(object()) is wmf(object())

    def test_no_event_config_path_is_a_noop(self, tmp_path: Path):
        from types import SimpleNamespace

        installed = []

        class _Aug:
            def set_constraint_factories(self, *a):
                installed.append(a)

        cfg = SimpleNamespace(
            calendar=SimpleNamespace(persona_events=""),
            output=SimpleNamespace(dir=str(tmp_path)),
            id="s",
            evaluator_model=None,
        )
        _wire_rl_constraints(_Aug(), run=_run_with_windows(), run_dir=tmp_path, cfg=cfg)
        assert installed == []

    def test_installs_factories_when_neo4j_unavailable(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        import src.graphrag.neo4j_client as neo4j_client

        def _boom(*_a, **_k):
            raise RuntimeError("neo4j down")

        monkeypatch.setattr(neo4j_client, "make_driver", _boom)

        event_yaml = tmp_path / "event_config.yaml"
        _write_event_config_yaml(event_yaml)
        _write_index(tmp_path, [])

        captured: dict = {}

        class _Aug:
            def set_constraint_factories(self, constraints_for, window_map_for):
                captured["constraints_for"] = constraints_for

        cfg = SimpleNamespace(
            calendar=SimpleNamespace(persona_events=str(event_yaml)),
            output=SimpleNamespace(dir=str(tmp_path)),
            id="s",
            evaluator_model=None,
            experiment="exp",
        )
        _wire_rl_constraints(_Aug(), run=_run_with_windows(), run_dir=tmp_path, cfg=cfg)
        # The pref tier degrades to semantic-only but factories still install.
        assert callable(captured.get("constraints_for"))
