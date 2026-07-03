"""Resolve `TimeframeSpec` against a persona horizon."""

from __future__ import annotations

import datetime

import pytest

from src.scripts.scenarios.config.schema import TimeframeSpec
from src.scripts.scenarios.config.timeframe import (
    ResolvedWindow,
    TimeframeError,
    resolve_timeframe,
)

HORIZON_START = datetime.date(2026, 6, 1)
HORIZON_WEEKS = 4


class TestResolvedWindow:
    def test_days(self):
        w = ResolvedWindow(HORIZON_START, 2)
        assert w.days == 14

    def test_end_inclusive(self):
        w = ResolvedWindow(HORIZON_START, 2)
        assert w.end_date_inclusive == datetime.date(2026, 6, 14)


class TestDefaultPath:
    def test_none_spec_returns_full_horizon(self):
        w = resolve_timeframe(None, HORIZON_START, HORIZON_WEEKS)
        assert w == ResolvedWindow(HORIZON_START, HORIZON_WEEKS)

    def test_zero_horizon_rejected(self):
        with pytest.raises(TimeframeError, match="horizon_weeks must be >= 1"):
            resolve_timeframe(None, HORIZON_START, 0)


class TestWeekForm:
    def test_first_week(self):
        spec = TimeframeSpec(scale="week", start=1, end=1)
        w = resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)
        assert w == ResolvedWindow(HORIZON_START, 1)

    def test_middle_two_weeks(self):
        spec = TimeframeSpec(scale="week", start=2, end=3)
        w = resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)
        assert w == ResolvedWindow(datetime.date(2026, 6, 8), 2)

    def test_last_week(self):
        spec = TimeframeSpec(scale="week", start=4, end=4)
        w = resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)
        assert w == ResolvedWindow(datetime.date(2026, 6, 22), 1)

    def test_end_past_horizon(self):
        spec = TimeframeSpec(scale="week", start=1, end=5)
        with pytest.raises(TimeframeError, match="past horizon"):
            resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)


class TestDatesForm:
    def test_aligned_start(self):
        spec = TimeframeSpec(
            scale="dates",
            start=datetime.date(2026, 6, 8),
            end=datetime.date(2026, 6, 21),
        )
        w = resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)
        assert w == ResolvedWindow(datetime.date(2026, 6, 8), 2)

    def test_full_horizon(self):
        spec = TimeframeSpec(
            scale="dates",
            start=HORIZON_START,
            end=datetime.date(2026, 6, 28),
        )
        w = resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)
        assert w == ResolvedWindow(HORIZON_START, 4)

    def test_start_before_horizon_rejected(self):
        spec = TimeframeSpec(
            scale="dates",
            start=datetime.date(2026, 5, 25),
            end=datetime.date(2026, 6, 7),
        )
        with pytest.raises(TimeframeError, match="before horizon start"):
            resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)

    def test_end_past_horizon_rejected(self):
        spec = TimeframeSpec(
            scale="dates",
            start=datetime.date(2026, 6, 22),
            end=datetime.date(2026, 7, 5),
        )
        with pytest.raises(TimeframeError, match="past horizon end"):
            resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)

    def test_non_aligned_start_rejected(self):
        spec = TimeframeSpec(
            scale="dates",
            start=datetime.date(2026, 6, 3),
            end=datetime.date(2026, 6, 16),
        )
        with pytest.raises(TimeframeError, match="not week-aligned"):
            resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS)


class TestScenarioIdEcho:
    def test_scenario_id_in_message(self):
        spec = TimeframeSpec(scale="week", start=1, end=99)
        with pytest.raises(TimeframeError, match="scenario='my_id'"):
            resolve_timeframe(spec, HORIZON_START, HORIZON_WEEKS, scenario_id="my_id")
