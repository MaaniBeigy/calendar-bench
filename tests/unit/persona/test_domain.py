"""Unit tests for src.scripts.persona.domain.* dataclasses."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.config.defaults import (
    DEFAULT_TIME_WINDOWS,
    WEEKDAY_INDEX,
    WEEKDAY_ORDER,
)
from src.scripts.persona.domain.event import Catalog, EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule, Spillover


def test_event_instance_is_immutable():
    ev = EventInstance(event_name="sleep", start=0, duration=480)
    with pytest.raises((AttributeError, TypeError)):
        ev.start = 100  # type: ignore[misc]


def test_day_schedule_defaults():
    day = DaySchedule(day_index=0, date=_dt.date(2026, 5, 4), weekday="Mon")
    assert day.events == {}
    assert day.spillovers == []


def test_person_schedule_defaults():
    sched = PersonSchedule(
        person_id="alice_0000", persona_id="alice", person_seed=12345
    )
    assert sched.days == []


def test_spillover_default_event_idx():
    s = Spillover(
        event_name="sleep",
        start=0,
        duration=60,
        orig_start=1380,
        orig_duration=120,
    )
    assert s.event_idx == 0


def test_catalog_from_event_config(event_yaml):
    from src.scripts.persona.config.loader import load_event

    cfg = load_event(event_yaml)
    catalog = Catalog.from_event_config(cfg)
    assert "sleep" in catalog
    assert catalog.get("sleep").category == "sleep"


def test_default_time_windows_match_named_set():
    assert set(DEFAULT_TIME_WINDOWS) == {
        "early_morning",
        "morning",
        "afternoon",
        "evening",
        "night",
    }


def test_weekday_order_and_index_consistent():
    assert len(WEEKDAY_ORDER) == 7
    for i, name in enumerate(WEEKDAY_ORDER):
        assert WEEKDAY_INDEX[name] == i
