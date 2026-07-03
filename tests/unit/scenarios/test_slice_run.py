"""Slice a `LoadedRun` to a `ResolvedWindow`."""

from __future__ import annotations

import datetime

from src.scripts.persona.config.schema import DailyWindow, WindowRange
from src.scripts.scenarios.calendar.loader import LoadedRun, slice_run_for_window
from src.scripts.scenarios.config.timeframe import ResolvedWindow
from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode


def _event(d: datetime.date) -> CalendarEvent:
    return CalendarEvent(label="x", start_minutes=540, end_minutes=600, date=d)


def _ctx(d: datetime.date) -> ContextEpisode:
    return ContextEpisode(
        name="happy",
        category="mood_emotion",
        date=d,
        start_minutes=100,
        end_minutes=140,
    )


def _trace_with_dates(person_id: str, dates: list[datetime.date]) -> CalendarTrace:
    return CalendarTrace(
        person_id=person_id,
        events=[_event(d) for d in dates],
        contexts=[_ctx(d) for d in dates],
        persona_id="persona_a",
    )


HORIZON_START = datetime.date(2026, 6, 1)
ALL_DATES = [HORIZON_START + datetime.timedelta(days=i) for i in range(28)]


def _make_run() -> LoadedRun:
    return LoadedRun(
        traces=[_trace_with_dates("p1", ALL_DATES), _trace_with_dates("p2", ALL_DATES)],
        time_windows={"morning": WindowRange(start=400, end=600)},
        daily_window=DailyWindow(wake_minutes=360, sleep_minutes=1320),
        horizon_days=28,
        horizon_start_date=HORIZON_START,
    )


class TestSliceRunForWindow:
    def test_identity_when_window_matches_horizon(self):
        run = _make_run()
        sliced = slice_run_for_window(run, ResolvedWindow(HORIZON_START, 4))
        assert sliced.horizon_days == 28
        assert sliced.horizon_start_date == HORIZON_START
        assert all(len(t.events) == 28 for t in sliced.traces)
        assert all(len(t.contexts) == 28 for t in sliced.traces)

    def test_inner_two_weeks(self):
        run = _make_run()
        sliced = slice_run_for_window(run, ResolvedWindow(datetime.date(2026, 6, 8), 2))
        assert sliced.horizon_days == 14
        assert sliced.horizon_start_date == datetime.date(2026, 6, 8)
        for t in sliced.traces:
            assert len(t.events) == 14
            assert min(e.date for e in t.events) == datetime.date(2026, 6, 8)
            assert max(e.date for e in t.events) == datetime.date(2026, 6, 21)
            assert len(t.contexts) == 14

    def test_first_week_only(self):
        run = _make_run()
        sliced = slice_run_for_window(run, ResolvedWindow(HORIZON_START, 1))
        for t in sliced.traces:
            assert all(
                HORIZON_START <= e.date <= datetime.date(2026, 6, 7) for e in t.events
            )
            assert len(t.events) == 7

    def test_preserves_rules_and_windows(self):
        run = _make_run()
        sliced = slice_run_for_window(run, ResolvedWindow(HORIZON_START, 2))
        assert sliced.time_windows == run.time_windows
        assert sliced.daily_window == run.daily_window
        assert sliced.allen_pair_rules == run.allen_pair_rules

    def test_preserves_person_id_and_persona_id(self):
        run = _make_run()
        sliced = slice_run_for_window(run, ResolvedWindow(HORIZON_START, 1))
        assert [t.person_id for t in sliced.traces] == ["p1", "p2"]
        assert all(t.persona_id == "persona_a" for t in sliced.traces)
