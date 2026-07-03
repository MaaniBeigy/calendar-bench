"""Unit tests for src.scripts.persona.validation.check_persona."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.config.schema import (
    Persona,
    PersonaCommon,
    PersonaConfig,
    PersonaEventStage,
)
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.validation.check_persona import (
    PersonaViolation,
    check_persona_constraints,
)
from tests.unit.persona.conftest import make_person, make_stage


def _persona(
    pid: str = "alice",
    instances: int = 1,
    stages: list[PersonaEventStage] | None = None,
) -> Persona:
    return Persona(
        id=pid,
        instances=instances,
        occupation_status="student",
        stages=stages or [],
    )


def _empty_horizon(days: int) -> list[DaySchedule]:
    out: list[DaySchedule] = []
    base = _dt.date(2026, 5, 4)
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i in range(days):
        d = base + _dt.timedelta(days=i)
        out.append(
            DaySchedule(
                day_index=i,
                date=d,
                weekday=weekdays[d.weekday()],
                events={},
                spillovers=[],
            )
        )
    return out


def _sched(person_id: str, day_events: dict[int, dict[str, list]]) -> PersonSchedule:
    days = _empty_horizon(7)
    for i, ev_map in day_events.items():
        old = days[i]
        events = {
            name: [
                EventInstance(event_name=name, start=s, duration=d) for s, d in pairs
            ]
            for name, pairs in ev_map.items()
        }
        days[i] = DaySchedule(
            day_index=old.day_index,
            date=old.date,
            weekday=old.weekday,
            events=events,
            spillovers=[],
        )
    return PersonSchedule(
        person_id=person_id,
        persona_id="alice",
        person_seed=42,
        days=days,
    )


def _config(personas: list[Persona]) -> PersonaConfig:
    return PersonaConfig(common=PersonaCommon(), personas=personas)


def test_no_violations_when_everything_matches():
    cfg = _config([_persona(instances=1)])
    persons = [make_person(stages=[])]
    schedules = [_sched("alice_0000", {})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert out == []


def test_instance_count_mismatch_flagged():
    cfg = _config([_persona(instances=2)])
    persons = [make_person(stages=[])]
    schedules = [_sched("alice_0000", {})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert any(v.kind == "instance_count" and "expected 2" in v.detail for v in out)


def test_undeclared_persona_id_flagged():
    cfg = _config([_persona(instances=1, pid="alice")])
    persons = [
        make_person(stages=[]),
        make_person(person_id="ghost_0000", persona_id="ghost", stages=[]),
    ]
    schedules = [_sched(p.person_id, {}) for p in persons]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert any("ghost" in v.detail and "no declaration" in v.detail for v in out)


def test_occupation_mismatch_flagged():
    cfg = _config([_persona(instances=1)])
    persons = [make_person(occupation_status="fulltime", stages=[])]
    schedules = [_sched("alice_0000", {})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert any(v.kind == "occupation" for v in out)


def test_persons_with_lengths_mismatch_raises():
    cfg = _config([_persona()])
    with pytest.raises(ValueError, match="lengths must match"):
        check_persona_constraints(
            cfg,
            [make_person(stages=[])],
            [],
            start_date=_dt.date(2026, 5, 4),
        )


def test_weekly_cadence_under_target_flagged():
    """A stage on Wed + Fri expects two episodes per week. With only one
    realised, the cadence check flags it."""
    stages = [
        make_stage("padel", time="21:00", duration_minutes=60, days=["Wed", "Fri"]),
    ]
    cfg = _config([_persona(stages=stages)])
    persons = [make_person(stages=stages)]
    schedules = [_sched("alice_0000", {2: {"padel": [(1260, 60)]}})]  # Wed only
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert any(v.kind == "cadence" and "padel" in v.detail for v in out)


def test_weekly_cadence_silent_when_target_met():
    stages = [
        make_stage("padel", time="21:00", duration_minutes=60, days=["Wed"]),
    ]
    cfg = _config([_persona(stages=stages)])
    persons = [make_person(stages=stages)]
    schedules = [_sched("alice_0000", {2: {"padel": [(1260, 60)]}})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert all(v.kind != "cadence" for v in out)


def test_multi_day_stage_cadence_flagged():
    """A stage on Mon + Tue expects two episodes; only one realised."""
    stages = [
        make_stage("reading", time="20:00", duration_minutes=60, days=["Mon", "Tue"]),
    ]
    cfg = _config([_persona(stages=stages)])
    persons = [make_person(stages=stages)]
    schedules = [_sched("alice_0000", {0: {"reading": [(1200, 60)]}})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert any(v.kind == "cadence" and "reading" in v.detail for v in out)


def test_weekend_only_stage_cadence_flagged_when_missing():
    stages = [
        make_stage("family_time", time="afternoon", duration_minutes=180, days=["Sat"])
    ]
    cfg = _config([_persona(stages=stages)])
    persons = [make_person(stages=stages)]
    schedules = [_sched("alice_0000", {})]  # Saturday missing
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert any(v.kind == "cadence" and "family_time" in v.detail for v in out)


def test_dated_stage_missing_on_its_date_flagged():
    stages = [
        make_stage(
            "dentist",
            time="14:00",
            duration_minutes=30,
            days=[],
            date=_dt.date(2026, 5, 6),
        )
    ]
    cfg = _config([_persona(stages=stages)])
    persons = [make_person(stages=stages)]
    schedules = [_sched("alice_0000", {})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert any(v.kind == "planned_date" for v in out)


def test_dated_stage_present_on_date_silent():
    stages = [
        make_stage(
            "dentist",
            time="14:00",
            duration_minutes=30,
            days=[],
            date=_dt.date(2026, 5, 6),
        )
    ]
    cfg = _config([_persona(stages=stages)])
    persons = [make_person(stages=stages)]
    schedules = [_sched("alice_0000", {2: {"dentist": [(840, 30)]}})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert all(v.kind != "planned_date" for v in out)


def test_dated_stage_outside_horizon_skipped():
    stages = [
        make_stage(
            "dentist",
            time="14:00",
            duration_minutes=30,
            days=[],
            date=_dt.date(2026, 12, 1),
        )
    ]
    cfg = _config([_persona(stages=stages)])
    persons = [make_person(stages=stages)]
    schedules = [_sched("alice_0000", {})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    assert all(v.kind != "planned_date" for v in out)


def test_undeclared_persona_does_not_skip_occupation_check():
    """A `Person` with a persona_id that has no declaration should not crash
    the occupation check."""
    cfg = _config([_persona(instances=1)])
    persons = [
        make_person(
            person_id="ghost_0000",
            persona_id="ghost",
            occupation_status="student",
            stages=[],
        )
    ]
    schedules = [_sched("ghost_0000", {})]
    out = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    # Reports the missing declaration but does not crash.
    assert any(v.kind == "instance_count" for v in out)
    # Skips the occupation comparison since there is no declared expectation.
    assert all(v.kind != "occupation" for v in out)


def test_violation_is_a_dataclass_instance():
    v = PersonaViolation(
        persona_id="x", person_id=None, kind="instance_count", detail="..."
    )
    assert v.persona_id == "x"
