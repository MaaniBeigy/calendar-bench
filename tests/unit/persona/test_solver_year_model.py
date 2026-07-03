"""Unit tests for src.scripts.persona.solver.year_model."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.config.schema import PersonaEventStage, WindowRange
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.solver.year_model import (
    YearlyAnchor,
    anchors_for_day,
    occupied_from_anchors,
    plan_yearly_anchors,
)
from tests.unit.persona.conftest import make_person, make_stage


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


def _person(extra_stages: list[PersonaEventStage] | None = None) -> Person:
    """A student persona with a daily routine plus any extra stages."""
    return make_person(
        occupation_status="student",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time="12:30", duration_minutes=45),
            make_stage("dinner", time="19:00", duration_minutes=60),
            *(extra_stages or []),
        ],
    )


def test_no_dated_stages_returns_empty_list():
    """A persona with only weekly-cadence stages produces no anchors."""
    anchors = plan_yearly_anchors(
        _person(),
        start_date=_dt.date(2026, 5, 4),
        horizon_days=14,
        window_map=_wm(),
    )
    assert anchors == []


def test_exact_time_dated_stage_is_resolved_in_minutes():
    stage = make_stage(
        "dentist",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 12),
    )
    anchors = plan_yearly_anchors(
        _person([stage]),
        start_date=_dt.date(2026, 5, 4),
        horizon_days=14,
        window_map=_wm(),
    )
    assert anchors == [
        YearlyAnchor(name="dentist", day_index=8, start=14 * 60, duration=30)
    ]


def test_window_token_resolves_to_window_start():
    stage = make_stage(
        "dentist",
        time="afternoon",
        duration_minutes=45,
        days=[],
        date=_dt.date(2026, 5, 6),
    )
    anchors = plan_yearly_anchors(
        _person([stage]),
        start_date=_dt.date(2026, 5, 4),
        horizon_days=7,
        window_map=_wm(),
    )
    assert anchors[0].start == 600  # afternoon window start
    assert anchors[0].duration == 45


def test_dated_stages_outside_horizon_are_dropped():
    in_horizon = make_stage(
        "exam",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 6),
    )
    too_early = make_stage(
        "dentist",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 4, 30),
    )
    too_late = make_stage(
        "checkup",
        time="14:00",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 30),
    )
    anchors = plan_yearly_anchors(
        _person([too_early, too_late, in_horizon]),
        start_date=_dt.date(2026, 5, 4),
        horizon_days=14,
        window_map=_wm(),
    )
    assert [a.name for a in anchors] == ["exam"]


def test_anchors_are_sorted_by_day_then_start_then_name():
    stages = [
        make_stage(
            "b_event",
            time="14:00",
            duration_minutes=30,
            days=[],
            date=_dt.date(2026, 5, 7),
        ),
        make_stage(
            "a_event",
            time="14:00",
            duration_minutes=30,
            days=[],
            date=_dt.date(2026, 5, 5),
        ),
        make_stage(
            "c_event",
            time="10:00",
            duration_minutes=30,
            days=[],
            date=_dt.date(2026, 5, 5),
        ),
    ]
    anchors = plan_yearly_anchors(
        _person(stages),
        start_date=_dt.date(2026, 5, 4),
        horizon_days=7,
        window_map=_wm(),
    )
    assert [a.name for a in anchors] == ["c_event", "a_event", "b_event"]


def test_negative_horizon_days_is_rejected():
    with pytest.raises(ValueError, match="horizon_days must be"):
        plan_yearly_anchors(
            _person(),
            start_date=_dt.date(2026, 5, 4),
            horizon_days=-1,
            window_map=_wm(),
        )


def test_anchor_extending_past_midnight_is_rejected():
    stage = make_stage(
        "late",
        time="23:30",
        duration_minutes=120,  # would end at 25:30 next day
        days=[],
        date=_dt.date(2026, 5, 5),
    )
    with pytest.raises(ValueError, match="extend past midnight"):
        plan_yearly_anchors(
            _person([stage]),
            start_date=_dt.date(2026, 5, 4),
            horizon_days=7,
            window_map=_wm(),
        )


def test_dated_stage_without_time_or_duration_is_skipped():
    """A `date`-anchored stage that does not pin both `time` and
    `duration_minutes` cannot be a yearly anchor - it falls through to
    the regular day solver instead."""
    no_time = make_stage(
        "appointment",
        duration_minutes=30,
        days=[],
        date=_dt.date(2026, 5, 6),
    )
    no_duration = make_stage(
        "checkup",
        time="14:00",
        days=[],
        date=_dt.date(2026, 5, 6),
    )
    anchors = plan_yearly_anchors(
        _person([no_time, no_duration]),
        start_date=_dt.date(2026, 5, 4),
        horizon_days=7,
        window_map=_wm(),
    )
    assert anchors == []


def test_anchors_for_day_filters_by_day_index():
    a1 = YearlyAnchor(name="x", day_index=0, start=0, duration=30)
    a2 = YearlyAnchor(name="y", day_index=2, start=600, duration=60)
    a3 = YearlyAnchor(name="z", day_index=2, start=900, duration=15)
    assert anchors_for_day([a1, a2, a3], 2) == [a2, a3]
    assert anchors_for_day([a1, a2, a3], 5) == []


def test_occupied_from_anchors_drops_zero_duration():
    a1 = YearlyAnchor(name="x", day_index=0, start=100, duration=30)
    a2 = YearlyAnchor(name="y", day_index=0, start=200, duration=0)
    assert occupied_from_anchors([a1, a2]) == [(100, 130)]


def test_yearly_anchor_end_property():
    a = YearlyAnchor(name="x", day_index=0, start=100, duration=30)
    assert a.end == 130


def test_persona_with_no_stages_returns_empty_anchor_list():
    """Edge case: a Person with `stages=[]` cannot produce any yearly anchor."""
    person = Person(
        person_id="empty_0000",
        persona_id="empty",
        person_seed=1,
        instance_index=0,
        occupation_status="student",
        stages=[],
        jitter_applied=__import__(
            "src.scripts.persona.config.schema", fromlist=["JitterConfig"]
        ).JitterConfig(),
    )
    anchors = plan_yearly_anchors(
        person,
        start_date=_dt.date(2026, 5, 4),
        horizon_days=7,
        window_map=_wm(),
    )
    assert anchors == []
