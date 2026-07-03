"""Unit tests for src.scripts.persona.export.json_writer."""

from __future__ import annotations

import datetime as _dt
import json

import pytest

from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule, Spillover
from src.scripts.persona.export.json_writer import (
    schedule_to_dict,
    write_person_json,
    write_population_json,
)
from tests.unit.persona.conftest import make_person, make_stage


def _person(person_id: str = "alice_0000") -> Person:
    return make_person(
        person_id=person_id,
        persona_id="alice",
        person_seed=42,
        occupation_status="student",
        stages=[make_stage("sleep", time="23:00")],
    )


def _schedule(person_id: str = "alice_0000") -> PersonSchedule:
    return PersonSchedule(
        person_id=person_id,
        persona_id="alice",
        person_seed=42,
        days=[
            DaySchedule(
                day_index=0,
                date=_dt.date(2026, 5, 4),
                weekday="Mon",
                events={
                    "sleep": [
                        EventInstance(event_name="sleep", start=1380, duration=420)
                    ],
                    "lunch": [
                        EventInstance(event_name="lunch", start=750, duration=45)
                    ],
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
                        duration=360,
                        orig_start=1380,
                        orig_duration=420,
                        event_idx=0,
                    )
                ],
            ),
        ],
    )


def test_schedule_to_dict_top_level_keys():
    payload = schedule_to_dict(_person(), _schedule())
    assert set(payload) == {
        "person_id",
        "persona_id",
        "person_seed",
        "persona",
        "days",
        "contexts",
    }
    assert payload["person_id"] == "alice_0000"
    assert payload["persona_id"] == "alice"
    assert payload["person_seed"] == 42


def test_schedule_to_dict_days_carry_iso_date_and_events():
    payload = schedule_to_dict(_person(), _schedule())
    days = payload["days"]
    assert isinstance(days, list) and len(days) == 2
    assert days[0]["date"] == "2026-05-04"
    assert days[0]["weekday"] == "Mon"
    assert days[0]["events"]["sleep"] == [{"start": 1380, "duration": 420}]
    assert days[0]["spillovers"] == []


def test_schedule_to_dict_serializes_event_label_when_present():
    schedule = PersonSchedule(
        person_id="alice_0000",
        persona_id="alice",
        person_seed=42,
        days=[
            DaySchedule(
                day_index=0,
                date=_dt.date(2026, 5, 4),
                weekday="Mon",
                events={
                    "office_work": [
                        EventInstance(
                            event_name="office_work",
                            start=540,
                            duration=60,
                            label="standup",
                        )
                    ],
                },
                spillovers=[],
            )
        ],
    )
    payload = schedule_to_dict(_person(), schedule)
    assert payload["days"][0]["events"]["office_work"] == [
        {"start": 540, "duration": 60, "label": "standup"}
    ]


def test_schedule_to_dict_emits_spillovers_with_legacy_keys():
    payload = schedule_to_dict(_person(), _schedule())
    spill = payload["days"][1]["spillovers"][0]
    assert spill == {
        "type": "sleep",
        "start": 0,
        "duration": 360,
        "orig_start": 1380,
        "orig_duration": 420,
        "event_idx": 0,
    }


def test_schedule_to_dict_event_keys_sorted():
    payload = schedule_to_dict(_person(), _schedule())
    keys = list(payload["days"][0]["events"].keys())
    assert keys == sorted(keys)


def test_schedule_to_dict_persona_block_includes_stages_and_occupation():
    payload = schedule_to_dict(_person(), _schedule())
    assert payload["persona"]["occupation_status"] == "student"
    assert isinstance(payload["persona"]["stages"], list)
    [stage] = payload["persona"]["stages"]
    assert stage["name"] == "sleep"
    assert stage["time"] == "23:00"


def test_write_person_json_creates_file_named_after_person_id(tmp_path):
    target = write_person_json(_person(), _schedule(), tmp_path)
    assert target.exists()
    assert target.name == "alice_0000.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["person_id"] == "alice_0000"


def test_write_person_json_creates_missing_directories(tmp_path):
    nested = tmp_path / "nested" / "dir"
    write_person_json(_person(), _schedule(), nested)
    assert (nested / "alice_0000.json").exists()


def test_write_population_json_writes_one_file_per_person(tmp_path):
    persons = [_person("alice_0000"), _person("alice_0001")]
    schedules = [_schedule("alice_0000"), _schedule("alice_0001")]
    paths = write_population_json(persons, schedules, tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
    names = sorted(p.name for p in paths)
    assert names == ["alice_0000.json", "alice_0001.json"]


def test_write_population_json_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="lengths must match"):
        write_population_json(
            [_person("alice_0000")],
            [_schedule("alice_0000"), _schedule("alice_0001")],
            tmp_path,
        )


def test_schedule_to_dict_handles_empty_days():
    sched = PersonSchedule(
        person_id="bob_0000",
        persona_id="bob",
        person_seed=1,
        days=[],
    )
    payload = schedule_to_dict(_person("bob_0000"), sched)
    assert payload["days"] == []
