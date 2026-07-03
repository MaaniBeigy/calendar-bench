"""Unit tests for PTIME per-criterion utility functions."""

from __future__ import annotations

import datetime

import pytest

from src.scripts.persona.config.schema import WindowRange
from src.scripts.scenarios.augmentation.ptime import (
    score_candidate,
    utility_duration,
    utility_overlap,
    utility_time,
)
from src.scripts.scenarios.metrics.allen import _build_rule_index
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)


# ---------------------------------------------------------------------------
# utility_time
# ---------------------------------------------------------------------------


def test_utility_time_no_windows_is_neutral():
    assert utility_time(420, 480, {}) == 1.0


def test_utility_time_full_inside_window():
    tw = {"morning": WindowRange(start=360, end=720)}
    assert utility_time(420, 480, tw) == pytest.approx(1.0)


def test_utility_time_partial_overlap():
    tw = {"morning": WindowRange(start=360, end=480)}
    # candidate [420, 540), 60-min span, 60 inside morning window.
    # overlap covers 480-420 = 60 of 120 minutes; utility = 60/120 = 0.5.
    assert utility_time(420, 540, tw) == pytest.approx(0.5)


def test_utility_time_outside_all_windows():
    tw = {"morning": WindowRange(start=360, end=420)}
    assert utility_time(600, 660, tw) == 0.0


def test_utility_time_zero_span_is_safe():
    # Avoid division by zero on a degenerate candidate.
    tw = {"morning": WindowRange(start=360, end=720)}
    assert utility_time(420, 420, tw) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# utility_duration
# ---------------------------------------------------------------------------


def test_utility_duration_minimum_preference():
    t = make_task(duration_min=30, duration_max=60)
    assert utility_duration(420, 450, t, "minimum") == 1.0  # 30 min = min
    assert utility_duration(420, 480, t, "minimum") == 0.0  # 60 min = max
    assert utility_duration(420, 465, t, "minimum") == pytest.approx(0.5)


def test_utility_duration_maximum_preference():
    t = make_task(duration_min=30, duration_max=60)
    assert utility_duration(420, 450, t, "maximum") == 0.0
    assert utility_duration(420, 480, t, "maximum") == 1.0
    assert utility_duration(420, 465, t, "maximum") == pytest.approx(0.5)


def test_utility_duration_midpoint_preference_peaks_at_center():
    t = make_task(duration_min=30, duration_max=60)
    assert utility_duration(420, 465, t, "midpoint") == pytest.approx(1.0)
    assert utility_duration(420, 450, t, "midpoint") == 0.0
    assert utility_duration(420, 480, t, "midpoint") == 0.0


def test_utility_duration_out_of_band_is_zero():
    t = make_task(duration_min=30, duration_max=60)
    assert utility_duration(420, 440, t, "minimum") == 0.0  # too short
    assert utility_duration(420, 540, t, "minimum") == 0.0  # too long


def test_utility_duration_collapsed_band():
    t = make_task(duration_min=30, duration_max=30)
    assert utility_duration(420, 450, t, "minimum") == 1.0


# ---------------------------------------------------------------------------
# utility_overlap
# ---------------------------------------------------------------------------


def test_utility_overlap_empty_day_is_neutral():
    rule_index = _build_rule_index([])
    assert utility_overlap(make_task(), 420, 480, DATE, {}, rule_index) == 1.0


def test_utility_overlap_no_concurrent_events_is_neutral():
    # Event before the candidate; no temporal overlap means it doesn't count.
    ev = make_event(label="lunch", start_minutes=180, end_minutes=240, date=DATE)
    rule_index = _build_rule_index([])
    assert utility_overlap(make_task(), 420, 480, DATE, {DATE: [ev]}, rule_index) == 1.0


def test_utility_overlap_concurrent_event_default_separates():
    # Default rule = R_SEP; an overlapping placement is inadmissible.
    ev = make_event(label="lunch", start_minutes=420, end_minutes=480, date=DATE)
    rule_index = _build_rule_index([])
    task = make_task(label="running", is_concurrent=False)
    assert utility_overlap(task, 420, 480, DATE, {DATE: [ev]}, rule_index) == 0.0


def test_utility_overlap_concurrent_event_with_concurrent_task_is_admissible():
    ev = make_event(
        label="lunch",
        start_minutes=420,
        end_minutes=480,
        date=DATE,
        is_concurrent=True,
    )
    rule_index = _build_rule_index([])
    task = make_task(label="podcast", is_concurrent=True)
    assert utility_overlap(task, 420, 480, DATE, {DATE: [ev]}, rule_index) == 1.0


# ---------------------------------------------------------------------------
# score_candidate end-to-end
# ---------------------------------------------------------------------------


def test_score_candidate_returns_value_and_breakdown():
    task = make_task(duration_min=30, duration_max=60)
    rule_index = _build_rule_index([])
    score, util = score_candidate(
        task=task,
        start=420,
        end=450,
        day=DATE,
        events_by_date={},
        time_windows={},
        rule_index=rule_index,
        importance={
            "time": 1 / 4,
            "duration": 1 / 4,
            "overlap": 1 / 4,
            "stability": 1 / 4,
        },
        interaction={},
        duration_preference="minimum",
    )
    assert set(util) == {"time", "duration", "overlap", "stability"}
    # All criteria neutral or full with default mins; score should be 1.0.
    assert score == pytest.approx(1.0)
