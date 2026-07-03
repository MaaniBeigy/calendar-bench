"""Unit tests for src.scripts.scenarios.calendar.loader.

Coverage targets:
  - load_persona_run with explicit config paths (no used_configs.json needed).
  - load_persona_run with used_configs.json (None paths to file read).
  - CalendarEvent field mapping: start/end, date, catalog enrichment.
  - apply_event_overrides called: persona override intensity/is_concurrent wins.
  - Missing event in catalog to safe defaults (is_concurrent=False, intensity=1).
  - Spillovers included as CalendarEvents on their day.
  - Events sorted by (date, start_minutes).
  - allen_pair_rules and ltl_rules populated from rules YAML.
  - time_windows and horizon_days from environment YAML.
  - Missing index.json to CalendarLoaderError.
  - Missing used_configs.json to CalendarLoaderError.
  - Malformed JSON to CalendarLoaderError.
  - Empty person list to empty traces.
  - Invalid EventOverride silently skipped.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
import yaml

from src.scripts.scenarios.calendar.loader import (
    CalendarLoaderError,
    LoadedRun,
    load_persona_run,
)

# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

_ENV_YAML = """\
seed: 12345
horizon:
  start_date: 2026-05-04
  weeks: 2
output:
  dir: ./output/test
time_windows:
  early_morning: [0, 400]
  morning: [400, 600]
  afternoon: [600, 960]
  evening: [960, 1260]
  night: [1260, 1440]
"""

_EVENT_YAML = """\
categories:
  sleep:
    events:
      sleep:
        per_event_duration: {min: 6, max: 9, unit: hours}
        total_event_duration: {scale: day, min: 6, max: 9, unit: hours}
        total_event_episodes: {scale: day, min: 1, max: 1}
        intensity: 1
        is_concurrent: false
        is_dividable: false
  work:
    events:
      office_work:
        per_event_duration: {min: 3, max: 8, unit: hours}
        total_event_duration: {scale: day, min: 3, max: 8, unit: hours}
        total_event_episodes: {scale: day, min: 1, max: 2}
        intensity: 3
        is_concurrent: false
        is_dividable: false
  eat:
    events:
      lunch:
        per_event_duration: {min: 30, max: 90, unit: minutes}
        total_event_duration: {scale: day, min: 30, max: 90, unit: minutes}
        total_event_episodes: {scale: day, min: 1, max: 1}
        intensity: 1
        is_concurrent: true
        is_dividable: false
        concurrent_with: [reading]
"""

_RULES_YAML = """\
rules:
  - id: no_overlap_sleep_work
    formula: "G ¬(sleep ∧ office_work)"

allen_pair_rules:
  - id: sleep_sep_running
    event_a: sleep
    event_b: office_work
    admissible_relations: [p, m, M, P]
"""


def _person_json(
    person_id: str = "p_0000",
    persona_id: str = "persona_a",
    event_overrides: dict | None = None,
    days: list | None = None,
) -> dict:
    return {
        "person_id": person_id,
        "persona_id": persona_id,
        "person_seed": 42,
        "persona": {
            "person_id": person_id,
            "persona_id": persona_id,
            "person_seed": 42,
            "instance_index": 0,
            "occupation_status": "fulltime",
            "stages": [],
            "jitter_applied": {"time_minutes": 15, "duration_minutes": 10},
            "event_overrides": event_overrides or {},
        },
        "days": days
        or [
            {
                "day_index": 0,
                "date": "2026-05-04",
                "weekday": "Mon",
                "events": {
                    "sleep": [{"start": 1380, "duration": 480}],
                    "office_work": [{"start": 540, "duration": 480}],
                },
                "spillovers": [],
            }
        ],
    }


def _index_json(persons: list[dict] | None = None) -> dict:
    return {
        "run_id": "test_run",
        "start_date": "2026-05-04",
        "weeks": 2,
        "persons": persons or [],
    }


def _write_run(
    tmp_path: Path,
    *,
    persons: list[dict] | None = None,
    event_yaml: str = _EVENT_YAML,
    rules_yaml: str = _RULES_YAML,
    env_yaml: str = _ENV_YAML,
    write_used_configs: bool = True,
) -> tuple[Path, Path, Path, Path]:
    """Write a synthetic run directory and return (run_dir, event_path, rules_path, env_path)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "persons").mkdir()

    event_path = tmp_path / "event_config.yaml"
    event_path.write_text(event_yaml, encoding="utf-8")

    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(rules_yaml, encoding="utf-8")

    env_path = tmp_path / "environment.yaml"
    env_path.write_text(env_yaml, encoding="utf-8")

    if write_used_configs:
        (run_dir / "used_configs.json").write_text(
            json.dumps(
                {
                    "environment": str(env_path),
                    "persona_config": str(tmp_path / "persona_config.yaml"),
                    "event_config": str(event_path),
                    "rules": str(rules_path),
                }
            ),
            encoding="utf-8",
        )

    person_entries = []
    for p in [_person_json()] if persons is None else persons:
        fname = f"{p['person_id']}.json"
        (run_dir / "persons" / fname).write_text(json.dumps(p), encoding="utf-8")
        person_entries.append(
            {
                "person_id": p["person_id"],
                "persona_id": p["persona_id"],
                "person_seed": p["person_seed"],
                "json_path": f"persons/{fname}",
                "ics_path": None,
                "days": len(p["days"]),
                "unsat": False,
            }
        )

    (run_dir / "index.json").write_text(
        json.dumps(_index_json(person_entries)), encoding="utf-8"
    )

    return run_dir, event_path, rules_path, env_path


# ---------------------------------------------------------------------------
# LoadedRun dataclass
# ---------------------------------------------------------------------------


class TestLoadedRun:
    def test_default_fields(self):
        lr = LoadedRun()
        assert lr.traces == []
        assert lr.allen_pair_rules == []
        assert lr.ltl_rules == []
        assert lr.time_windows == {}
        assert lr.horizon_days == 0
        assert lr.horizon_start_date is None

    def test_list_defaults_are_independent(self):
        lr1 = LoadedRun()
        lr2 = LoadedRun()
        lr1.traces.append(None)  # type: ignore[arg-type]
        assert len(lr2.traces) == 0

    def test_load_populates_horizon_start_date(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        assert result.horizon_start_date == datetime.date(2026, 5, 4)


# ---------------------------------------------------------------------------
# load_persona_run; happy paths
# ---------------------------------------------------------------------------


class TestLoadPersonaRunHappyPath:
    def test_explicit_paths_no_used_configs(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        assert isinstance(result, LoadedRun)
        assert len(result.traces) == 1

    def test_reads_used_configs_when_paths_are_none(self, tmp_path):
        run_dir, _, _, _ = _write_run(tmp_path)
        result = load_persona_run(run_dir)
        assert len(result.traces) == 1

    def test_horizon_days_is_weeks_times_seven(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        assert result.horizon_days == 2 * 7  # weeks=2

    def test_time_windows_loaded(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        assert "morning" in result.time_windows
        assert "night" in result.time_windows

    def test_ltl_rules_loaded(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        assert len(result.ltl_rules) == 1
        assert result.ltl_rules[0].id == "no_overlap_sleep_work"

    def test_allen_pair_rules_loaded(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        assert len(result.allen_pair_rules) == 1
        assert result.allen_pair_rules[0].id == "sleep_sep_running"

    def test_empty_persons_gives_empty_traces(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(
            tmp_path, persons=[], write_used_configs=False
        )
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        assert result.traces == []


# ---------------------------------------------------------------------------
# CalendarEvent field mapping
# ---------------------------------------------------------------------------


class TestCalendarEventFieldMapping:
    def _get_trace(self, tmp_path: Path, persons=None) -> object:
        run_dir, ep, rp, envp = _write_run(
            tmp_path, persons=persons, write_used_configs=False
        )
        return load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        ).traces[0]

    def test_start_minutes_from_start(self, tmp_path):
        trace = self._get_trace(tmp_path)
        sleep_ev = next(e for e in trace.events if e.label == "sleep")
        assert sleep_ev.start_minutes == 1380

    def test_end_minutes_is_start_plus_duration(self, tmp_path):
        trace = self._get_trace(tmp_path)
        sleep_ev = next(e for e in trace.events if e.label == "sleep")
        assert sleep_ev.end_minutes == 1380 + 480

    def test_date_parsed_from_iso_string(self, tmp_path):
        trace = self._get_trace(tmp_path)
        assert all(e.date == datetime.date(2026, 5, 4) for e in trace.events)

    def test_is_concurrent_from_catalog(self, tmp_path):
        trace = self._get_trace(tmp_path)
        sleep_ev = next(e for e in trace.events if e.label == "sleep")
        assert sleep_ev.is_concurrent is False

    def test_intensity_from_catalog(self, tmp_path):
        trace = self._get_trace(tmp_path)
        work_ev = next(e for e in trace.events if e.label == "office_work")
        assert work_ev.intensity == 3

    def test_display_label_defaults_to_none_and_effective_falls_back(self, tmp_path):
        trace = self._get_trace(tmp_path)
        work_ev = next(e for e in trace.events if e.label == "office_work")
        assert work_ev.display_label is None
        assert work_ev.effective_label == "office_work"
        assert work_ev.oracle_label == "office_work"

    def test_display_label_read_from_instance_label(self, tmp_path):
        persons = [
            _person_json(
                days=[
                    {
                        "day_index": 0,
                        "date": "2026-05-04",
                        "weekday": "Mon",
                        "events": {
                            "office_work": [
                                {"start": 540, "duration": 480, "label": "standup"}
                            ],
                        },
                        "spillovers": [],
                    }
                ]
            )
        ]
        trace = self._get_trace(tmp_path, persons=persons)
        work_ev = next(e for e in trace.events if e.label == "office_work")
        assert work_ev.display_label == "standup"
        assert work_ev.effective_label == "standup"
        assert work_ev.label == "office_work"
        assert work_ev.oracle_label == "standup (office_work)"

    def test_is_dividable_from_catalog(self, tmp_path):
        trace = self._get_trace(tmp_path)
        sleep_ev = next(e for e in trace.events if e.label == "sleep")
        assert sleep_ev.is_dividable is False

    def test_events_sorted_by_date_then_start(self, tmp_path):
        persons = [
            _person_json(
                days=[
                    {
                        "day_index": 0,
                        "date": "2026-05-04",
                        "weekday": "Mon",
                        "events": {
                            "office_work": [{"start": 540, "duration": 480}],
                            "sleep": [{"start": 1380, "duration": 480}],
                        },
                        "spillovers": [],
                    }
                ]
            )
        ]
        trace = self._get_trace(tmp_path, persons=persons)
        starts = [e.start_minutes for e in trace.events]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Catalog enrichment via persona overrides
# ---------------------------------------------------------------------------


class TestCatalogEnrichment:
    def _run(self, tmp_path: Path, persons: list[dict]) -> "LoadedRun":
        run_dir, ep, rp, envp = _write_run(
            tmp_path, persons=persons, write_used_configs=False
        )
        return load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )

    def test_persona_override_intensity_wins(self, tmp_path):
        """Persona event_override intensity: 5 must override catalog intensity: 3."""
        persons = [_person_json(event_overrides={"office_work": {"intensity": 5}})]
        result = self._run(tmp_path, persons)
        work_ev = next(e for e in result.traces[0].events if e.label == "office_work")
        assert work_ev.intensity == 5

    def test_persona_override_is_concurrent_wins(self, tmp_path):
        """Persona event_override is_concurrent: true must override catalog False."""
        persons = [
            _person_json(event_overrides={"office_work": {"is_concurrent": True}})
        ]
        result = self._run(tmp_path, persons)
        work_ev = next(e for e in result.traces[0].events if e.label == "office_work")
        assert work_ev.is_concurrent is True

    def test_unknown_event_in_catalog_uses_safe_defaults(self, tmp_path):
        """An event label not in the catalog must return safe defaults."""
        persons = [
            _person_json(
                days=[
                    {
                        "day_index": 0,
                        "date": "2026-05-04",
                        "weekday": "Mon",
                        "events": {"ghost_event": [{"start": 300, "duration": 60}]},
                        "spillovers": [],
                    }
                ]
            )
        ]
        result = self._run(tmp_path, persons)
        ghost = next(e for e in result.traces[0].events if e.label == "ghost_event")
        assert ghost.is_concurrent is False
        assert ghost.is_dividable is False
        assert ghost.concurrent_with == []
        assert ghost.intensity == 1

    def test_event_with_none_intensity_in_catalog_defaults_to_one(self, tmp_path):
        """EventDefinition.intensity = None (not set) must become 1 in CalendarEvent."""
        event_yaml_no_intensity = """\
categories:
  work:
    events:
      office_work:
        per_event_duration: {min: 3, max: 8, unit: hours}
        total_event_duration: {scale: day, min: 3, max: 8, unit: hours}
        total_event_episodes: {scale: day, min: 1, max: 2}
"""
        run_dir, _, rp, envp = _write_run(
            tmp_path,
            event_yaml=event_yaml_no_intensity,
            write_used_configs=False,
        )
        ep = tmp_path / "event_config.yaml"
        result = load_persona_run(
            run_dir,
            event_config_path=ep,
            rules_path=rp,
            environment_path=envp,
        )
        work_ev = next(e for e in result.traces[0].events if e.label == "office_work")
        assert work_ev.intensity == 1

    def test_invalid_event_override_silently_skipped(self, tmp_path):
        """A malformed event_override dict must not crash the loader."""
        persons = [
            _person_json(
                event_overrides={"office_work": "not_a_dict"}  # type: ignore[arg-type]
            )
        ]
        result = self._run(tmp_path, persons)
        assert len(result.traces) == 1  # still produces a trace

    def test_concurrent_with_from_catalog(self, tmp_path):
        """lunch.concurrent_with = [reading] must appear on the CalendarEvent."""
        persons = [
            _person_json(
                days=[
                    {
                        "day_index": 0,
                        "date": "2026-05-04",
                        "weekday": "Mon",
                        "events": {"lunch": [{"start": 720, "duration": 45}]},
                        "spillovers": [],
                    }
                ]
            )
        ]
        result = self._run(tmp_path, persons)
        lunch_ev = next(e for e in result.traces[0].events if e.label == "lunch")
        assert "reading" in lunch_ev.concurrent_with


# ---------------------------------------------------------------------------
# Spillover handling
# ---------------------------------------------------------------------------


class TestSpillovers:
    def test_spillover_included_as_calendar_event(self, tmp_path):
        persons = [
            _person_json(
                days=[
                    {
                        "day_index": 1,
                        "date": "2026-05-05",
                        "weekday": "Tue",
                        "events": {},
                        "spillovers": [
                            {
                                "type": "sleep",
                                "start": 0,
                                "duration": 60,
                                "orig_start": 1380,
                                "orig_duration": 540,
                                "event_idx": 0,
                            }
                        ],
                    }
                ]
            )
        ]
        run_dir, ep, rp, envp = _write_run(
            tmp_path, persons=persons, write_used_configs=False
        )
        result = load_persona_run(
            run_dir, event_config_path=ep, rules_path=rp, environment_path=envp
        )
        events = result.traces[0].events
        spill = next(
            (e for e in events if e.label == "sleep" and e.start_minutes == 0),
            None,
        )
        assert spill is not None
        assert spill.end_minutes == 60

    def test_spillover_enriched_from_catalog(self, tmp_path):
        """Spillover events must also get intensity/is_concurrent from the catalog."""
        persons = [
            _person_json(
                days=[
                    {
                        "day_index": 1,
                        "date": "2026-05-05",
                        "weekday": "Tue",
                        "events": {},
                        "spillovers": [
                            {
                                "type": "office_work",
                                "start": 0,
                                "duration": 30,
                                "orig_start": 1400,
                                "orig_duration": 60,
                                "event_idx": 0,
                            }
                        ],
                    }
                ]
            )
        ]
        run_dir, ep, rp, envp = _write_run(
            tmp_path, persons=persons, write_used_configs=False
        )
        result = load_persona_run(
            run_dir, event_config_path=ep, rules_path=rp, environment_path=envp
        )
        spill = next(e for e in result.traces[0].events if e.label == "office_work")
        assert spill.intensity == 3


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


class TestCalendarLoaderErrors:
    def test_missing_index_json_raises(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        (run_dir / "index.json").unlink()
        with pytest.raises(CalendarLoaderError, match="not found"):
            load_persona_run(
                run_dir, event_config_path=ep, rules_path=rp, environment_path=envp
            )

    def test_missing_used_configs_raises(self, tmp_path):
        run_dir, _, _, _ = _write_run(tmp_path, write_used_configs=False)
        with pytest.raises(CalendarLoaderError, match="not found"):
            load_persona_run(run_dir)

    def test_malformed_used_configs_raises(self, tmp_path):
        run_dir, _, _, _ = _write_run(tmp_path)
        (run_dir / "used_configs.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(CalendarLoaderError, match="JSON object"):
            load_persona_run(run_dir)

    def test_malformed_json_file_raises(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        (run_dir / "index.json").write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(CalendarLoaderError, match="could not read"):
            load_persona_run(
                run_dir, event_config_path=ep, rules_path=rp, environment_path=envp
            )

    def test_bad_persona_config_raises(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        with pytest.raises(CalendarLoaderError, match="failed to load"):
            load_persona_run(
                run_dir,
                event_config_path=Path("/nonexistent/event_config.yaml"),
                rules_path=rp,
                environment_path=envp,
            )

    def test_missing_person_json_raises(self, tmp_path):
        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)
        # Remove the person file after the index was written
        for f in (run_dir / "persons").iterdir():
            f.unlink()
        with pytest.raises(CalendarLoaderError, match="not found"):
            load_persona_run(
                run_dir, event_config_path=ep, rules_path=rp, environment_path=envp
            )

    def test_partial_explicit_paths_reads_used_configs_for_rest(self, tmp_path):
        """Providing only event_config_path still reads rules and env from used_configs."""
        run_dir, ep, rp, envp = _write_run(tmp_path)
        result = load_persona_run(run_dir, event_config_path=ep)
        assert len(result.traces) == 1

    def test_rules_path_provided_env_none_uses_used_configs(self, tmp_path):
        """Covers the False branch of 'if rules_path is None:' when rules is provided."""
        run_dir, ep, rp, envp = _write_run(tmp_path)
        # Provide rules_path but not event_config or environment: outer if fires,
        # event_config_path set from used_configs, rules_path skipped (not None),
        # environment_path set from used_configs.
        result = load_persona_run(run_dir, rules_path=rp)
        assert len(result.traces) == 1

    def test_all_but_event_config_provided_covers_env_false_branch(self, tmp_path):
        """Covers False branches of both 'if rules_path' and 'if environment_path'."""
        run_dir, ep, rp, envp = _write_run(tmp_path)
        # Only event_config_path is None: outer if fires, event_config set from
        # used_configs, rules_path and environment_path False branches taken.
        result = load_persona_run(run_dir, rules_path=rp, environment_path=envp)
        assert len(result.traces) == 1


# ---------------------------------------------------------------------------
# Exception branch coverage in _parse_person_json
# ---------------------------------------------------------------------------


class TestParsePersonJsonExceptBranch:
    def test_dict_override_failing_pydantic_silently_skipped(self, tmp_path):
        """An override dict that IS a dict but fails EventOverride validation
        must be silently skipped, not crash the loader (covers except:pass)."""
        # EventOverride has extra="forbid", so unknown keys raise ValidationError.
        persons = [
            _person_json(
                event_overrides={"office_work": {"completely_unknown_field": 99}}
            )
        ]
        run_dir, ep, rp, envp = _write_run(
            tmp_path, persons=persons, write_used_configs=False
        )
        result = load_persona_run(
            run_dir, event_config_path=ep, rules_path=rp, environment_path=envp
        )
        # The override is skipped; catalog default intensity (3) is used.
        work_ev = next(e for e in result.traces[0].events if e.label == "office_work")
        assert work_ev.intensity == 3


# ---------------------------------------------------------------------------
# json_path resolution: cwd-relative path (covers `direct.exists()` True branch)
# ---------------------------------------------------------------------------


class TestCwdRelativeJsonPath:
    def test_cwd_relative_json_path_in_index(self, tmp_path, monkeypatch):
        """Covers the `direct.exists()` True branch introduced to fix the doubled-path
        bug when the persona pipeline stores `json_path` relative to cwd
        (e.g. 'output/example_experiment/persons/p.json') rather than relative to run_dir
        (e.g. 'persons/p.json').
        """
        import os

        # Change cwd to tmp_path so a cwd-relative path resolves correctly.
        monkeypatch.chdir(tmp_path)

        run_dir, ep, rp, envp = _write_run(tmp_path, write_used_configs=False)

        # Re-write index.json with json_path relative to cwd (the real persona pipeline format).
        import json as _json

        index_path = run_dir / "index.json"
        index = _json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index["persons"]:
            # Convert "persons/p.json" to "run/persons/p.json" (relative to tmp_path cwd)
            entry["json_path"] = str(
                (run_dir / entry["json_path"]).relative_to(tmp_path)
            )
        index_path.write_text(_json.dumps(index), encoding="utf-8")

        result = load_persona_run(
            run_dir, event_config_path=ep, rules_path=rp, environment_path=envp
        )
        assert len(result.traces) == 1
