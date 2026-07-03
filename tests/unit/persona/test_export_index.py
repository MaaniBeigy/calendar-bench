"""Unit tests for src.scripts.persona.export.index."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from src.scripts.persona.config.schema import (
    EnvironmentConfig,
    HorizonConfig,
    OutputConfig,
    ParallelismConfig,
    SolverConfig,
    WindowRange,
)
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.export.index import (
    PersonIndexEntry,
    RunIndex,
    build_index,
    index_to_dict,
    write_index,
)


def _env() -> EnvironmentConfig:
    return EnvironmentConfig(
        seed=1,
        horizon=HorizonConfig(start_date=_dt.date(2026, 5, 4), weeks=1),
        output=OutputConfig(dir="./out"),
        solver=SolverConfig(),
        time_windows={
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        },
        parallelism=ParallelismConfig(workers=1, executor="process"),
    )


def _schedule(person_id: str, *, with_event: bool = True) -> PersonSchedule:
    events: dict[str, list[EventInstance]] = (
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]}
        if with_event
        else {}
    )
    return PersonSchedule(
        person_id=person_id,
        persona_id="alice",
        person_seed=42,
        days=[
            DaySchedule(
                day_index=0,
                date=_dt.date(2026, 5, 4),
                weekday="Mon",
                events=events,
                spillovers=[],
            )
        ],
    )


def test_build_index_attaches_paths_and_unsat_flag(tmp_path):
    s1 = _schedule("alice_0000", with_event=True)
    s2 = _schedule("alice_0001", with_event=False)
    json_paths = [
        tmp_path / "persons" / "alice_0000.json",
        tmp_path / "persons" / "alice_0001.json",
    ]
    ics_paths = [
        tmp_path / "ics_per_person" / "alice_0000.ics",
        tmp_path / "ics_per_person" / "alice_0001.ics",
    ]
    index = build_index(
        [s1, s2],
        environment=_env(),
        json_paths=json_paths,
        ics_paths=ics_paths,
        run_id="example_001",
    )
    assert index.run_id == "example_001"
    assert index.start_date == "2026-05-04"
    assert index.weeks == 1
    assert len(index.persons) == 2
    assert index.persons[0].unsat is False
    assert index.persons[1].unsat is True
    assert index.persons[0].json_path.endswith("alice_0000.json")
    assert index.persons[0].ics_path is not None


def test_build_index_without_ics_paths_sets_ics_to_none():
    s1 = _schedule("alice_0000")
    index = build_index(
        [s1],
        environment=_env(),
        json_paths=[Path("persons/alice_0000.json")],
        ics_paths=None,
        run_id="run_2",
    )
    assert index.persons[0].ics_path is None


def test_build_index_rejects_length_mismatch():
    s1 = _schedule("alice_0000")
    with pytest.raises(ValueError, match="json_paths"):
        build_index(
            [s1],
            environment=_env(),
            json_paths=[Path("a.json"), Path("b.json")],
            run_id="x",
        )


def test_build_index_rejects_ics_length_mismatch():
    s1 = _schedule("alice_0000")
    with pytest.raises(ValueError, match="ics_paths"):
        build_index(
            [s1],
            environment=_env(),
            json_paths=[Path("a.json")],
            ics_paths=[Path("a.ics"), Path("b.ics")],
            run_id="x",
        )


def test_unsat_when_schedule_has_no_days():
    sched = PersonSchedule(
        person_id="x_0000",
        persona_id="x",
        person_seed=1,
        days=[],
    )
    index = build_index(
        [sched],
        environment=_env(),
        json_paths=[Path("x_0000.json")],
        run_id="r",
    )
    assert index.persons[0].unsat is True


def test_index_to_dict_round_trips_to_json():
    entry = PersonIndexEntry(
        person_id="x_0000",
        persona_id="x",
        person_seed=1,
        json_path="persons/x_0000.json",
        ics_path="ics/x_0000.ics",
        days=7,
        unsat=False,
    )
    rendered = index_to_dict(
        RunIndex(run_id="r", start_date="2026-05-04", weeks=1, persons=[entry])
    )
    encoded = json.dumps(rendered)
    assert "x_0000" in encoded


def test_write_index_creates_index_json(tmp_path):
    s1 = _schedule("alice_0000")
    json_paths = [tmp_path / "persons" / "alice_0000.json"]
    index = build_index(
        [s1],
        environment=_env(),
        json_paths=json_paths,
        run_id="r",
    )
    target = write_index(index, tmp_path)
    assert target == tmp_path / "index.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["run_id"] == "r"
    assert payload["persons"][0]["person_id"] == "alice_0000"
