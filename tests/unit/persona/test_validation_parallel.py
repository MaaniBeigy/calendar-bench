"""Unit tests for src.scripts.persona.validation.parallel."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    Persona,
    PersonaCommon,
    PersonaConfig,
    PersonaEventStage,
    TemporalRule,
    TotalDuration,
)
from src.scripts.persona.domain.event import Catalog, EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.validation.check_continuity import check_continuity
from src.scripts.persona.validation.check_event import check_event_constraints
from src.scripts.persona.validation.check_ltl import check_ltl_rules
from src.scripts.persona.validation.check_persona import check_persona_constraints
from src.scripts.persona.validation.parallel import run_validation_parallel
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


def _person(
    pid: str = "alice_0000",
    persona_id: str = "alice",
    stages: list[PersonaEventStage] | None = None,
) -> Person:
    return make_person(
        person_id=pid,
        persona_id=persona_id,
        person_seed=42,
        occupation_status="student",
        stages=stages or [],
    )


def _sched(pid: str, day_events: list[dict]) -> PersonSchedule:
    days: list[DaySchedule] = []
    base = _dt.date(2026, 5, 4)
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, ev_map in enumerate(day_events):
        d = base + _dt.timedelta(days=i)
        evs = {
            name: [
                EventInstance(event_name=name, start=s, duration=du) for s, du in pairs
            ]
            for name, pairs in ev_map.items()
        }
        days.append(
            DaySchedule(
                day_index=i,
                date=d,
                weekday=weekdays[d.weekday()],
                events=evs,
                spillovers=[],
            )
        )
    return PersonSchedule(person_id=pid, persona_id="alice", person_seed=42, days=days)


def _empty_catalog() -> Catalog:
    return Catalog(categories={}, events_by_name={})


def _lunch_catalog() -> Catalog:
    e = EventDefinition(
        name="lunch",
        category="eat",
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=30, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )
    cfg = EventConfig(categories={"eat": Category(name="eat", events={"lunch": e})})
    return Catalog.from_event_config(cfg)


def _persona_config(personas: list[Persona]) -> PersonaConfig:
    return PersonaConfig(common=PersonaCommon(), personas=personas)


# -------------------------------------------------------------------------------------
# ----------------------- equivalence with the sequential checks ----------------------
# -------------------------------------------------------------------------------------


def test_parallel_matches_sequential_on_clean_population():
    cfg = _persona_config([_persona(instances=2)])
    persons = [_person("alice_0000"), _person("alice_0001")]
    schedules = [_sched(p.person_id, [{}]) for p in persons]
    catalog = _empty_catalog()
    rules: list[TemporalRule] = []

    seq = check_persona_constraints(
        cfg, persons, schedules, start_date=_dt.date(2026, 5, 4)
    )
    par_report = run_validation_parallel(
        cfg,
        persons,
        schedules,
        catalog,
        rules,
        start_date=_dt.date(2026, 5, 4),
        workers=1,
    )
    assert par_report.persona == seq
    assert par_report.event == []
    assert par_report.ltl == []
    assert par_report.continuity == []


def test_parallel_aggregates_event_violations_in_input_order():
    catalog = _lunch_catalog()
    cfg = _persona_config([_persona(instances=2)])
    persons = [_person("alice_0000"), _person("alice_0001")]
    schedules = [
        _sched("alice_0000", [{"lunch": [(750, 10)]}]),  # under-duration
        _sched("alice_0001", [{"lunch": [(750, 200)]}]),  # over-duration
    ]
    seq = check_event_constraints(schedules, persons, catalog)
    par = run_validation_parallel(
        cfg,
        persons,
        schedules,
        catalog,
        [],
        start_date=_dt.date(2026, 5, 4),
        workers=2,
    )
    assert par.event == seq


def test_parallel_aggregates_ltl_violations():
    catalog = _empty_catalog()
    cfg = _persona_config([_persona(instances=1)])
    persons = [_person()]
    schedules = [_sched("alice_0000", [{"sleep": [(0, 480)], "work": [(400, 120)]}])]
    rule = TemporalRule(id="r1", formula="G ¬(sleep ∧ work)")
    seq = check_ltl_rules(
        schedules, [rule], {persons[0].person_id: persons[0].occupation_status}
    )
    par = run_validation_parallel(
        cfg,
        persons,
        schedules,
        catalog,
        [rule],
        start_date=_dt.date(2026, 5, 4),
        workers=2,
    )
    assert par.ltl == seq


def test_parallel_aggregates_continuity_violations():
    cfg = _persona_config([_persona(instances=1)])
    persons = [_person()]
    schedules = [
        _sched(
            "alice_0000", [{"sleep": [(1380, 120)]}, {}]
        ),  # missing spillover on day 1
    ]
    seq = check_continuity(schedules)
    par = run_validation_parallel(
        cfg,
        persons,
        schedules,
        _empty_catalog(),
        [],
        start_date=_dt.date(2026, 5, 4),
        workers=2,
    )
    assert par.continuity == seq


# -------------------------------------------------------------------------------------
# ------------------------- determinism across worker counts --------------------------
# -------------------------------------------------------------------------------------


def test_parallel_workers_1_equals_workers_4():
    catalog = _lunch_catalog()
    cfg = _persona_config([_persona(instances=4)])
    persons = [_person(f"alice_{i:04d}") for i in range(4)]
    schedules = [_sched(p.person_id, [{"lunch": [(750, 5)]}]) for p in persons]
    a = run_validation_parallel(
        cfg,
        persons,
        schedules,
        catalog,
        [],
        start_date=_dt.date(2026, 5, 4),
        workers=1,
    )
    b = run_validation_parallel(
        cfg,
        persons,
        schedules,
        catalog,
        [],
        start_date=_dt.date(2026, 5, 4),
        workers=4,
    )
    assert a == b


def test_parallel_workers_none_resolves_to_cpu_count():
    cfg = _persona_config([_persona(instances=1)])
    persons = [_person()]
    schedules = [_sched("alice_0000", [{}])]
    out = run_validation_parallel(
        cfg,
        persons,
        schedules,
        _empty_catalog(),
        [],
        start_date=_dt.date(2026, 5, 4),
        workers=None,
    )
    assert out.total == 0


def test_parallel_workers_zero_resolves_to_cpu_count():
    cfg = _persona_config([_persona(instances=1)])
    persons = [_person()]
    schedules = [_sched("alice_0000", [{}])]
    out = run_validation_parallel(
        cfg,
        persons,
        schedules,
        _empty_catalog(),
        [],
        start_date=_dt.date(2026, 5, 4),
        workers=0,
    )
    assert out.total == 0


# -------------------------------------------------------------------------------------
# ----------------------------- edge cases / error paths ------------------------------
# -------------------------------------------------------------------------------------


def test_parallel_empty_population_skips_executor():
    cfg = _persona_config([_persona(instances=1)])
    out = run_validation_parallel(
        cfg, [], [], _empty_catalog(), [], start_date=_dt.date(2026, 5, 4)
    )
    # Persona-level violations still fire (instance_count = 0 vs declared 1).
    assert any(v.kind == "instance_count" for v in out.persona)
    assert out.event == []
    assert out.ltl == []
    assert out.continuity == []


def test_parallel_length_mismatch_raises():
    cfg = _persona_config([_persona(instances=1)])
    with pytest.raises(ValueError, match="lengths must match"):
        run_validation_parallel(
            cfg,
            [_person()],
            [],
            _empty_catalog(),
            [],
            start_date=_dt.date(2026, 5, 4),
        )


def test_parallel_persona_cadence_per_person_violations_merged():
    """Both persons stage padel on Wed but neither realizes it, so the
    cadence check fires once per person."""
    stages = [make_stage("padel", time="21:00", duration_minutes=60, days=["Wed"])]
    cfg = _persona_config([_persona(stages=stages, instances=2)])
    persons = [
        _person("alice_0000", stages=stages),
        _person("alice_0001", stages=stages),
    ]
    # Build 7-day horizons. Wed = day index 2. Both persons miss the padel slot.
    schedules = [_sched(p.person_id, [{} for _ in range(7)]) for p in persons]
    out = run_validation_parallel(
        cfg,
        persons,
        schedules,
        _empty_catalog(),
        [],
        start_date=_dt.date(2026, 5, 4),
        workers=2,
    )
    cadence_persons = [v.person_id for v in out.persona if v.kind == "cadence"]
    assert sorted(cadence_persons) == ["alice_0000", "alice_0001"]
