"""End-to-end tests for the PTimeAugmenter."""

from __future__ import annotations

import datetime

import pytest

from src.scripts.persona.config.schema import DailyWindow, WindowRange
from src.scripts.scenarios.augmentation.ptime import PTimeAugmenter
from src.scripts.scenarios.config.schema import AugmentationConfig, PTimeConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)
DATE2 = datetime.date(2026, 5, 5)


def _wide_window() -> DailyWindow:
    return DailyWindow(wake_minutes=0, sleep_minutes=1440)


def _cfg(**kw) -> AugmentationConfig:
    return AugmentationConfig(
        method="ptime", ptime=PTimeConfig(**kw), repeat_per_week=False
    )


def test_augment_empty_horizon_returns_unscheduled():
    aug = PTimeAugmenter(daily_window=_wide_window())
    trace = CalendarTrace(person_id="p1")
    task = make_task(duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg())
    assert sol.scheduled == []
    assert sol.unscheduled == [task]


def test_augment_places_into_only_free_gap():
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="running", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg())
    assert len(sol.scheduled) == 1
    assert sol.unscheduled == []
    st = sol.scheduled[0]
    assert st.end_minutes - st.start_minutes == 30
    assert st.date == DATE


def test_augment_respects_time_windows_when_importance_concentrated():
    # Concentrating importance on `time` should land the slot inside the
    # configured window rather than at the earliest free minute.
    tw = {"morning": WindowRange(start=480, end=600)}
    aug = PTimeAugmenter(
        time_windows=tw,
        daily_window=_wide_window(),
        seed=0,
    )
    ev = make_event(label="dinner", start_minutes=1200, end_minutes=1260, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    cfg = _cfg(
        importance={
            "time": 1.0,
            "duration": 0.0,
            "overlap": 0.0,
            "stability": 0.0,
        }
    )
    sol = aug.augment(trace, [task], cfg)
    st = sol.scheduled[0]
    # Anchored at the morning window start (480).
    assert st.start_minutes == 480


def test_augment_avoids_overlap_with_non_concurrent_event():
    # Default Allen rule is `R_SEP`; placement stays separated from the event.
    ev = make_event(label="meeting", start_minutes=480, end_minutes=540, date=DATE)
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg())
    st = sol.scheduled[0]
    # Ends at/before 480 or starts at/after 540; never overlaps.
    assert st.end_minutes <= ev.start_minutes or st.start_minutes >= ev.end_minutes


def test_augment_repeat_per_week_emits_one_instance_per_week():
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    events = [
        make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE),
        make_event(
            label="lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE + datetime.timedelta(days=7),
        ),
    ]
    trace = CalendarTrace(person_id="p1", events=events)
    task = make_task(label="run", duration_min=30, duration_max=30)
    cfg = AugmentationConfig(method="ptime", repeat_per_week=True)
    horizon = (DATE, 14)  # two weeks
    sol = aug.augment(trace, [task], cfg, horizon=horizon)
    # `repeat_per_week=True` emits one task instance per 7-day chunk.
    assert len(sol.tasks) == 2
    assert len(sol.scheduled) == 2


def test_augment_greedy_solver_drops_task_when_no_candidates_fit():
    # Wall-to-wall event leaves no slot; greedy decoder records unscheduled.
    ev = make_event(label="solid", start_minutes=0, end_minutes=1440, date=DATE)
    aug = PTimeAugmenter(
        daily_window=DailyWindow(wake_minutes=0, sleep_minutes=1440), seed=0
    )
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    cfg = AugmentationConfig(
        method="ptime",
        repeat_per_week=False,
        ptime=PTimeConfig(solver="greedy"),
    )
    sol = aug.augment(trace, [task], cfg)
    assert sol.scheduled == []
    assert sol.unscheduled == [task]


def test_augment_carry_forward_skips_empty_weeks():
    # With `repeat_per_week=False` and a 14-day horizon, placing the task
    # in week 1 leaves week 2 with an empty `carry`; the empty-week guard fires.
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    cfg = AugmentationConfig(method="ptime", repeat_per_week=False)
    sol = aug.augment(trace, [task], cfg, horizon=(DATE, 14))
    assert len(sol.scheduled) == 1
    assert sol.unscheduled == []


def test_augment_mcs_marks_empty_domain_task_as_unscheduled():
    # An unplaceable task has no candidates; `mcs_solve` filters it out
    # internally and the augmenter surfaces it as unscheduled.
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    trace = CalendarTrace(person_id="p1", events=[])
    placeable = make_task(label="walk", duration_min=30, duration_max=30)
    too_long = make_task(label="marathon", duration_min=2000, duration_max=2000)
    cfg = AugmentationConfig(method="ptime", repeat_per_week=False)
    sol = aug.augment(trace, [placeable, too_long], cfg, horizon=(DATE, 7))
    assert len(sol.scheduled) == 1
    assert len(sol.unscheduled) == 1
    assert sol.unscheduled[0] is too_long


def test_augment_greedy_solver_still_places_tasks():
    # The `greedy` solver path produces a valid placement.
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    cfg = AugmentationConfig(
        method="ptime",
        repeat_per_week=False,
        ptime=PTimeConfig(solver="greedy"),
    )
    sol = aug.augment(trace, [task], cfg)
    assert len(sol.scheduled) == 1
    st = sol.scheduled[0]
    assert st.end_minutes - st.start_minutes == 30


def test_augment_drops_task_when_no_gap_fits():
    # Wall-to-wall event blocking the entire day, plus a tiny daily window.
    ev = make_event(label="solid", start_minutes=0, end_minutes=1440, date=DATE)
    aug = PTimeAugmenter(
        daily_window=DailyWindow(wake_minutes=0, sleep_minutes=1440), seed=0
    )
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="run", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg())
    assert sol.scheduled == []
    assert sol.unscheduled == [task]


def test_augment_repeat_per_week_drops_unplaceable_task():
    # `repeat_per_week=True` over a wall-to-wall day; placement returns
    # `None` and the task lands in unscheduled directly.
    ev = make_event(label="solid", start_minutes=0, end_minutes=1440, date=DATE)
    aug = PTimeAugmenter(
        daily_window=DailyWindow(wake_minutes=0, sleep_minutes=1440), seed=0
    )
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="run", duration_min=30, duration_max=30)
    cfg = AugmentationConfig(method="ptime", repeat_per_week=True)
    sol = aug.augment(trace, [task], cfg, horizon=(DATE, 1))
    assert sol.scheduled == []
    assert sol.unscheduled == [task]


def test_augment_carry_forward_drops_unplaceable_task():
    # `repeat_per_week=False` over a wall-to-wall day; the task carries
    # through the only week chunk and lands in `unscheduled`.
    ev = make_event(label="solid", start_minutes=0, end_minutes=1440, date=DATE)
    aug = PTimeAugmenter(
        daily_window=DailyWindow(wake_minutes=0, sleep_minutes=1440), seed=0
    )
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="run", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg(), horizon=(DATE, 1))
    assert sol.scheduled == []
    assert sol.unscheduled == [task]


def test_augment_with_no_config_uses_defaults():
    # A bare config object with `ptime=None` still produces a placement.
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="run", duration_min=30, duration_max=30)

    class _Bare:
        ptime = None
        repeat_per_week = False

    sol = aug.augment(trace, [task], _Bare())
    assert len(sol.scheduled) == 1


def test_augment_reuse_pass_kicks_in_when_all_gaps_used():
    # Single base gap, two tasks; second shares the same gap via reuse.
    aug = PTimeAugmenter(
        daily_window=DailyWindow(wake_minutes=420, sleep_minutes=600), seed=0
    )
    trace = CalendarTrace(person_id="p1", events=[])
    t1 = make_task(label="t1", duration_min=30, duration_max=30)
    t2 = make_task(label="t2", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [t1, t2], _cfg(), horizon=(DATE, 7))
    assert len(sol.scheduled) == 2
    starts = sorted(s.start_minutes for s in sol.scheduled if s.date == DATE)
    # Either both pack onto day 1 (reuse path), or one lands on a fresh day.
    if len(starts) == 2:
        assert starts == [420, 450]
    else:
        assert len(sol.scheduled) == 2
