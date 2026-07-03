"""Unit tests for src.scripts.persona.solver.week_model."""

from __future__ import annotations

import pytest

from src.scripts.persona.config.schema import WindowRange
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.event_config.loader import load_catalog
from src.scripts.persona.solver.week_model import WeeklyTemplate, build_weekly_template
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


def _student(*, extra_stages=None) -> Person:
    """Student with the standard daily routine plus optional extra stages."""
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


def test_template_has_seven_weekdays(event_yaml):
    catalog = load_catalog(event_yaml)
    template = build_weekly_template(_student(), catalog, window_map=_wm())
    assert isinstance(template, WeeklyTemplate)
    assert set(template.weekday_to_counts) == {
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    }


def test_template_pins_persona_activity_only_to_stated_weekdays(event_yaml):
    catalog = load_catalog(event_yaml)
    person = _student(
        extra_stages=[
            make_stage("padel", time="21:00", duration_minutes=60, days=["Wed"]),
            make_stage("padel", time="14:00", duration_minutes=60, days=["Fri"]),
        ]
    )
    template = build_weekly_template(person, catalog, window_map=_wm())
    assert template.weekday_to_counts["Wed"]["padel"] == 1
    assert template.weekday_to_counts["Fri"]["padel"] == 1
    assert template.weekday_to_counts["Mon"]["padel"] == 0
    assert template.weekday_to_counts["Sat"]["padel"] == 0


def test_template_for_weekday_returns_counts(event_yaml):
    catalog = load_catalog(event_yaml)
    template = build_weekly_template(_student(), catalog, window_map=_wm())
    counts = template.for_weekday("Mon")
    assert counts["sleep"] == 1
    assert counts["lunch"] == 1


def test_template_for_weekday_rejects_unknown_token(event_yaml):
    catalog = load_catalog(event_yaml)
    template = build_weekly_template(_student(), catalog, window_map=_wm())
    with pytest.raises(KeyError, match="weekday"):
        template.for_weekday("Funday")


def test_template_for_day_index_wraps_modulo_seven(event_yaml):
    catalog = load_catalog(event_yaml)
    template = build_weekly_template(_student(), catalog, window_map=_wm())
    # day 0 = Mon, day 7 = Mon, day 8 = Tue.
    assert template.for_day_index(0) == template.for_day_index(7)
    assert template.for_day_index(1) == template.for_day_index(8)


def test_template_excludes_dated_stages_from_recurring_template(event_yaml):
    """A persona with only date-anchored (one-off) entries for an event
    contributes zero to the recurring weekly template - the horizon
    pipeline injects those per-day instead."""
    catalog = load_catalog(event_yaml)
    template = build_weekly_template(_student(), catalog, window_map=_wm())
    for counts in template.weekday_to_counts.values():
        assert counts["dentist"] == 0
