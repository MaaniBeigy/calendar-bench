"""Yearly-anchor integration with src.scripts.persona.planner.horizon."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.config.schema import (
    EnvironmentConfig,
    HorizonConfig,
    OutputConfig,
    ParallelismConfig,
    PersonaEventStage,
    SolverConfig,
    WindowRange,
)
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.event_config.loader import load_catalog
from src.scripts.persona.planner.horizon import plan_horizon
from tests.unit.persona.conftest import make_person, make_stage


def _env(*, enable_yearly_pass: bool = True) -> EnvironmentConfig:
    return EnvironmentConfig(
        seed=1,
        horizon=HorizonConfig(
            start_date=_dt.date(2026, 5, 4),
            weeks=2,
            enable_yearly_pass=enable_yearly_pass,
        ),
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


def _student_with_dated_stage(stage: PersonaEventStage | None = None) -> Person:
    """Student persona with a routine plus an optional date-anchored stage."""
    extras = [stage] if stage else []
    return make_person(
        occupation_status="student",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time="12:30", duration_minutes=45),
            make_stage("dinner", time="19:00", duration_minutes=60),
            *extras,
        ],
    )


def test_dated_stage_appears_at_its_date(event_yaml):
    stage = make_stage(
        "dentist",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student_with_dated_stage(stage), catalog, _env())
    target_day = next(d for d in schedule.days if d.date == _dt.date(2026, 5, 12))
    [dentist] = target_day.events["dentist"]
    assert dentist.start == 14 * 60
    assert dentist.duration == 30


def test_dated_stage_does_not_appear_on_other_days(event_yaml):
    stage = make_stage(
        "dentist",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student_with_dated_stage(stage), catalog, _env())
    for day in schedule.days:
        if day.date != _dt.date(2026, 5, 12):
            assert "dentist" not in day.events


def test_anchor_blocks_overlapping_solver_events(event_yaml):
    """Other events placed on the anchor day must not overlap the anchor range."""
    stage = make_stage(
        "dentist",
        time="13:00",
        duration_minutes=60,  # blocks 780-840
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student_with_dated_stage(stage), catalog, _env())
    target_day = next(d for d in schedule.days if d.date == _dt.date(2026, 5, 12))
    blocked_start, blocked_end = 13 * 60, 14 * 60
    for name, events in target_day.events.items():
        if name == "dentist":
            continue
        for ev in events:
            assert ev.start >= blocked_end or ev.start + ev.duration <= blocked_start


def test_disabling_yearly_pass_drops_anchors_for_uncatalogued_events(event_yaml):
    """With the yearly pass off, a date-anchored stage whose name does
    not appear in the catalog cannot reach the schedule - only the
    yearly pass injects it as a fixed slot."""
    stage = make_stage(
        "conference",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(
        _student_with_dated_stage(stage),
        catalog,
        _env(enable_yearly_pass=False),
    )
    target_day = next(d for d in schedule.days if d.date == _dt.date(2026, 5, 12))
    assert "conference" not in target_day.events


def test_disabling_yearly_pass_unpins_catalog_anchor_time(event_yaml):
    """With the yearly pass off, a catalog-defined dated stage is still
    solved (the allocator counts it via the date trigger), but the
    start time is whatever the solver picks, not the anchor's HH:MM.
    """
    stage = make_stage(
        "dentist",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    catalog = load_catalog(event_yaml)
    pinned = plan_horizon(
        _student_with_dated_stage(stage), catalog, _env(enable_yearly_pass=True)
    )
    unpinned = plan_horizon(
        _student_with_dated_stage(stage), catalog, _env(enable_yearly_pass=False)
    )
    pinned_day = next(d for d in pinned.days if d.date == _dt.date(2026, 5, 12))
    unpinned_day = next(d for d in unpinned.days if d.date == _dt.date(2026, 5, 12))
    assert pinned_day.events["dentist"][0].start == 14 * 60
    # Anchor is gone; the solver may legally pick any allowed slot.
    assert "dentist" in unpinned_day.events
    assert len(unpinned_day.events["dentist"]) == 1


def test_yearly_pass_event_outside_catalog_appears_only_with_anchor(event_yaml):
    """The free-form event name shows up exactly once when the anchor is on."""
    stage = make_stage(
        "conference",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student_with_dated_stage(stage), catalog, _env())
    target_day = next(d for d in schedule.days if d.date == _dt.date(2026, 5, 12))
    [conference] = target_day.events["conference"]
    assert conference.start == 14 * 60
    assert conference.duration == 30
