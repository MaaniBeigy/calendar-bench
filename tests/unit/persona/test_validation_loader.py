"""Unit tests for src.scripts.persona.validation.loader."""

from __future__ import annotations

import datetime as _dt
import json

from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule, Spillover
from src.scripts.persona.export.json_writer import write_person_json
from src.scripts.persona.validation.loader import (
    load_schedule,
    load_schedules,
    schedule_from_dict,
)
from tests.unit.persona.conftest import make_person


def _person(pid: str = "alice_0000") -> Person:
    return make_person(person_id=pid, persona_id="alice", stages=[])


def _sched(pid: str = "alice_0000") -> PersonSchedule:
    return PersonSchedule(
        person_id=pid,
        persona_id="alice",
        person_seed=42,
        days=[
            DaySchedule(
                day_index=0,
                date=_dt.date(2026, 5, 4),
                weekday="Mon",
                events={
                    "lunch": [EventInstance(event_name="lunch", start=750, duration=45)]
                },
                spillovers=[],
            ),
            DaySchedule(
                day_index=1,
                date=_dt.date(2026, 5, 5),
                weekday="Tue",
                events={},
                spillovers=[
                    Spillover(
                        event_name="sleep",
                        start=0,
                        duration=60,
                        orig_start=1380,
                        orig_duration=120,
                        event_idx=0,
                    )
                ],
            ),
        ],
    )


def test_round_trip_through_disk(tmp_path):
    original = _sched()
    target = write_person_json(_person(), original, tmp_path)
    restored = load_schedule(target)
    assert restored.person_id == original.person_id
    assert restored.persona_id == original.persona_id
    assert restored.person_seed == original.person_seed
    assert len(restored.days) == 2
    assert restored.days[0].weekday == "Mon"
    assert restored.days[0].date == _dt.date(2026, 5, 4)
    [lunch] = restored.days[0].events["lunch"]
    assert lunch.start == 750 and lunch.duration == 45
    [spill] = restored.days[1].spillovers
    assert spill.event_name == "sleep"
    assert spill.duration == 60


def test_load_schedules_returns_sorted_by_filename(tmp_path):
    write_person_json(_person("alice_0000"), _sched("alice_0000"), tmp_path)
    write_person_json(_person("alice_0002"), _sched("alice_0002"), tmp_path)
    write_person_json(_person("alice_0001"), _sched("alice_0001"), tmp_path)
    schedules = load_schedules(tmp_path)
    ids = [s.person_id for s in schedules]
    assert ids == ["alice_0000", "alice_0001", "alice_0002"]


def test_schedule_from_dict_default_event_idx():
    payload = {
        "person_id": "x_0000",
        "persona_id": "x",
        "person_seed": 1,
        "days": [
            {
                "day_index": 0,
                "date": "2026-05-04",
                "weekday": "Mon",
                "events": {},
                "spillovers": [
                    {
                        "type": "sleep",
                        "start": 0,
                        "duration": 30,
                        "orig_start": 1410,
                        "orig_duration": 60,
                        # event_idx omitted on purpose
                    }
                ],
            }
        ],
    }
    sched = schedule_from_dict(payload)
    assert sched.days[0].spillovers[0].event_idx == 0


def test_schedule_from_dict_handles_missing_keys():
    payload = {
        "person_id": "x_0000",
        "persona_id": "x",
        "person_seed": 1,
        "days": [
            {
                "day_index": 0,
                "date": "2026-05-04",
                "weekday": "Mon",
                # no events, no spillovers
            }
        ],
    }
    sched = schedule_from_dict(payload)
    assert sched.days[0].events == {}
    assert sched.days[0].spillovers == []


def test_load_schedule_from_string_path(tmp_path):
    write_person_json(_person(), _sched(), tmp_path)
    target = tmp_path / "alice_0000.json"
    sched = load_schedule(str(target))
    assert sched.person_id == "alice_0000"


def test_load_schedule_round_trips_arbitrary_dict(tmp_path):
    """schedule_from_dict accepts a dict directly without disk."""
    payload = {
        "person_id": "y_0000",
        "persona_id": "y",
        "person_seed": 9,
        "days": [],
    }
    target = tmp_path / "y_0000.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    sched = load_schedule(target)
    assert sched.days == []
