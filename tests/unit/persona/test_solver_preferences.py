"""Unit tests for src.scripts.persona.solver.preferences."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.solver.preferences import (
    DayPreferences,
    _hhmm_to_minutes,
    _maybe_window,
    build_day_preferences,
)
from tests.unit.persona.conftest import make_person, make_stage


def test_returns_dataclass_with_three_separate_dicts():
    prefs = build_day_preferences(
        make_person(), weekday="Mon", date=_dt.date(2026, 5, 4)
    )
    assert isinstance(prefs, DayPreferences)
    assert prefs.starts is not prefs.durations
    assert prefs.windows is not prefs.starts
    assert prefs.windows is not prefs.durations


def test_default_routine_stages_pin_meals_and_sleep():
    prefs = build_day_preferences(
        make_person(), weekday="Mon", date=_dt.date(2026, 5, 4)
    )
    assert prefs.starts["first_eat"] == 7 * 60 + 30
    assert prefs.starts["lunch"] == 12 * 60 + 30
    assert prefs.starts["dinner"] == 19 * 60
    assert prefs.starts["sleep"] == 23 * 60
    assert prefs.durations["first_eat"] == 20
    assert prefs.durations["lunch"] == 45
    assert prefs.durations["dinner"] == 60


def test_window_token_keeps_duration_and_records_window():
    person = make_person(
        stages=[make_stage("lunch", time="afternoon", duration_minutes=45)]
    )
    prefs = build_day_preferences(person, weekday="Mon", date=_dt.date(2026, 5, 4))
    assert "lunch" not in prefs.starts
    assert prefs.windows["lunch"] == "afternoon"
    assert prefs.durations["lunch"] == 45


def test_stage_with_only_time_pins_start_no_duration():
    """A stage that pins start but leaves duration to the catalog (e.g.
    `office_work` where the solver picks the block length) registers a
    `starts` entry but no `durations` entry."""
    person = make_person(
        stages=[
            make_stage(
                "office_work", time="09:00", days=["Mon", "Tue", "Wed", "Thu", "Fri"]
            )
        ]
    )
    prefs = build_day_preferences(person, weekday="Mon", date=_dt.date(2026, 5, 4))
    assert prefs.starts["office_work"] == 9 * 60
    assert "office_work" not in prefs.durations


def test_stage_pins_only_for_matching_weekday():
    person = make_person(
        stages=[make_stage("running", time="06:30", duration_minutes=45, days=["Mon"])]
    )
    on_mon = build_day_preferences(person, weekday="Mon", date=_dt.date(2026, 5, 4))
    on_tue = build_day_preferences(person, weekday="Tue", date=_dt.date(2026, 5, 5))
    assert on_mon.starts["running"] == 6 * 60 + 30
    assert on_mon.durations["running"] == 45
    assert "running" not in on_tue.starts
    assert "running" not in on_tue.durations


def test_stage_with_window_token_records_window():
    person = make_person(
        stages=[
            make_stage("reading", time="evening", duration_minutes=60, days=["Mon"])
        ]
    )
    prefs = build_day_preferences(person, weekday="Mon", date=_dt.date(2026, 5, 4))
    assert "reading" not in prefs.starts
    assert prefs.windows["reading"] == "evening"
    assert prefs.durations["reading"] == 60


def test_stage_anchored_by_date_fires_only_on_matching_date():
    person = make_person(
        stages=[
            make_stage(
                "dentist",
                time="14:00",
                duration_minutes=30,
                days=[],
                date=_dt.date(2026, 5, 12),
            )
        ]
    )
    on_date = build_day_preferences(person, weekday="Tue", date=_dt.date(2026, 5, 12))
    other = build_day_preferences(person, weekday="Mon", date=_dt.date(2026, 5, 11))
    assert on_date.starts["dentist"] == 14 * 60
    assert on_date.durations["dentist"] == 30
    assert "dentist" not in other.starts


def test_stage_with_no_time_set_records_only_duration():
    """A stage with `duration_minutes` but no `time` contributes a
    `durations` entry only - the day model picks the start."""
    person = make_person(
        stages=[make_stage("running", duration_minutes=45, days=["Mon"])]
    )
    prefs = build_day_preferences(person, weekday="Mon", date=_dt.date(2026, 5, 4))
    assert "running" not in prefs.starts
    assert "running" not in prefs.windows
    assert prefs.durations["running"] == 45


# -------------------------------------------------------------------------------------
# Helpers - direct coverage
# -------------------------------------------------------------------------------------


def test_hhmm_to_minutes_rejects_malformed_inputs():
    assert _hhmm_to_minutes("not-a-time") is None
    assert _hhmm_to_minutes("12:") is None
    assert _hhmm_to_minutes("12") is None
    assert _hhmm_to_minutes(123) is None  # type: ignore[arg-type]


def test_maybe_window_recognises_named_windows():
    assert _maybe_window("morning") == "morning"
    assert _maybe_window("afternoon") == "afternoon"
    assert _maybe_window("evening") == "evening"
    assert _maybe_window("night") == "night"
    assert _maybe_window("early_morning") == "early_morning"


def test_maybe_window_returns_none_for_non_windows():
    assert _maybe_window(None) is None
    assert _maybe_window("") is None
    assert _maybe_window("12:00") is None
    assert _maybe_window("not-a-window") is None


def test_stage_with_unrecognised_time_string_records_neither_start_nor_window():
    """If a stage's `time` parses as neither HH:MM nor a named window
    (only reachable when the schema validator is bypassed), the stage
    contributes no entry to either `starts` or `windows`. The duration,
    if present, is still recorded."""
    from src.scripts.persona.config.schema import PersonaEventStage

    # Construct via model_construct to bypass the field validator that
    # would normally reject "not-a-window".
    odd_stage = PersonaEventStage.model_construct(
        name="thing",
        time="not-a-window",
        duration_minutes=30,
        days=["Mon"],
        date=None,
    )
    person = make_person(stages=[odd_stage])
    prefs = build_day_preferences(person, weekday="Mon", date=_dt.date(2026, 5, 4))
    assert "thing" not in prefs.starts
    assert "thing" not in prefs.windows
    assert prefs.durations["thing"] == 30
