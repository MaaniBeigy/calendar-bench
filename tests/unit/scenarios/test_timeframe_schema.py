"""Schema-level validation for `TimeframeSpec`."""

from __future__ import annotations

import datetime

import pytest

from src.scripts.scenarios.config.schema import ScenarioDefinition, TimeframeSpec


class TestWeekForm:
    def test_valid(self):
        spec = TimeframeSpec(scale="week", start=1, end=2)
        assert spec.scale == "week"
        assert spec.start == 1
        assert spec.end == 2

    def test_single_week(self):
        spec = TimeframeSpec(scale="week", start=3, end=3)
        assert spec.end == spec.start

    def test_end_before_start(self):
        with pytest.raises(ValueError, match="must be >= timeframe.start"):
            TimeframeSpec(scale="week", start=3, end=2)

    def test_start_zero_rejected(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            TimeframeSpec(scale="week", start=0, end=1)

    def test_start_must_be_int(self):
        with pytest.raises(ValueError, match="must be an integer"):
            TimeframeSpec(scale="week", start=datetime.date(2026, 6, 1), end=2)

    def test_end_must_be_int(self):
        with pytest.raises(ValueError, match="must be an integer"):
            TimeframeSpec(scale="week", start=1, end=datetime.date(2026, 6, 14))


class TestDatesForm:
    def test_valid_two_weeks(self):
        spec = TimeframeSpec(
            scale="dates",
            start=datetime.date(2026, 6, 1),
            end=datetime.date(2026, 6, 14),
        )
        assert spec.scale == "dates"

    def test_single_week(self):
        spec = TimeframeSpec(
            scale="dates",
            start=datetime.date(2026, 6, 1),
            end=datetime.date(2026, 6, 7),
        )
        assert spec.start == datetime.date(2026, 6, 1)

    def test_non_multiple_of_seven_rejected(self):
        with pytest.raises(ValueError, match="multiple of 7 days"):
            TimeframeSpec(
                scale="dates",
                start=datetime.date(2026, 6, 1),
                end=datetime.date(2026, 6, 10),
            )

    def test_end_before_start(self):
        with pytest.raises(ValueError, match="must be >= timeframe.start"):
            TimeframeSpec(
                scale="dates",
                start=datetime.date(2026, 6, 14),
                end=datetime.date(2026, 6, 7),
            )

    def test_start_must_be_date(self):
        with pytest.raises(ValueError, match="must be an ISO date"):
            TimeframeSpec(scale="dates", start=1, end=datetime.date(2026, 6, 7))

    def test_end_must_be_date(self):
        with pytest.raises(ValueError, match="must be an ISO date"):
            TimeframeSpec(scale="dates", start=datetime.date(2026, 6, 1), end=7)


class TestScenarioDefinitionField:
    def test_optional_default_none(self):
        sd = ScenarioDefinition(id="s1")
        assert sd.timeframe is None

    def test_accepts_timeframe(self):
        sd = ScenarioDefinition(
            id="s1",
            timeframe=TimeframeSpec(scale="week", start=1, end=2),
        )
        assert sd.timeframe is not None
        assert sd.timeframe.scale == "week"
