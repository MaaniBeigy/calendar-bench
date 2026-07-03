"""Unit tests for the dumb FCFS greedy augmenter.

The augmenter is a baseline: pure earliest-fit, no semantic / MET /
intensity awareness, no merges, no backtracking.  The RNG seed is the
only stochastic input.
"""

from __future__ import annotations

import datetime

import pytest

from src.scripts.persona.config.schema import DailyWindow, WindowRange
from src.scripts.scenarios.augmentation.base import Augmenter
from src.scripts.scenarios.augmentation.greedy import (
    GreedyAugmenter,
    _earliest_fit_slot,
    _free_gaps,
    _get_horizon_dates,
    _get_weekly_chunks,
    _merge_intervals,
    _tasks_for_week,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, GreedyConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)
DATE2 = datetime.date(2026, 5, 5)


def _trace(*events) -> CalendarTrace:
    return CalendarTrace(person_id="p001", events=list(events))


def _config(repeat_per_week: bool = False) -> AugmentationConfig:
    return AugmentationConfig(
        method="greedy",
        repeat_per_week=repeat_per_week,
        greedy=GreedyConfig(strategy="preference_first"),
    )


def _greedy(*, seed: int = 0, daily_window=None) -> GreedyAugmenter:
    if daily_window is None:
        daily_window = DailyWindow(wake_minutes=0, sleep_minutes=1440)
    return GreedyAugmenter(daily_window=daily_window, seed=seed)


# ---------------------------------------------------------------------------
# base.py abstract
# ---------------------------------------------------------------------------


class TestAugmenterBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            Augmenter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# _get_horizon_dates / _get_weekly_chunks
# ---------------------------------------------------------------------------


class TestHorizonHelpers:
    def test_empty_calendar(self):
        assert _get_horizon_dates(CalendarTrace(person_id="x")) == []

    def test_single_date(self):
        assert _get_horizon_dates(_trace(make_event(date=DATE))) == [DATE]

    def test_date_range(self):
        result = _get_horizon_dates(
            _trace(make_event(date=DATE), make_event(date=DATE2))
        )
        assert result == [DATE, DATE2]

    def test_weekly_chunks_empty(self):
        assert _get_weekly_chunks([]) == []

    def test_tasks_for_week_empty_list_returns_empty(self):
        assert _tasks_for_week([], 0) == []

    def test_tasks_for_week_clamps_past_last(self):
        a, b = make_task("a"), make_task("b")
        # index 5 past the 2-week list clamps to the last week.
        assert [t.label for t in _tasks_for_week([[a], [b]], 5)] == ["b"]

    def test_weekly_chunks_seven(self):
        days = [DATE + datetime.timedelta(days=i) for i in range(7)]
        chunks = _get_weekly_chunks(days)
        assert len(chunks) == 1
        assert chunks[0] == days

    def test_weekly_chunks_eight_days_two_chunks(self):
        days = [DATE + datetime.timedelta(days=i) for i in range(8)]
        chunks = _get_weekly_chunks(days)
        assert len(chunks) == 2
        assert len(chunks[0]) == 7
        assert len(chunks[1]) == 1

    def test_horizon_hint_overrides_event_dates(self):
        result = _get_horizon_dates(
            CalendarTrace(person_id="x"),
            horizon=(DATE, 5),
        )
        assert result == [DATE + datetime.timedelta(days=i) for i in range(5)]

    def test_horizon_hint_used_even_with_events(self):
        result = _get_horizon_dates(
            _trace(make_event(date=DATE)),
            horizon=(DATE + datetime.timedelta(days=7), 7),
        )
        assert DATE not in result
        assert result[0] == DATE + datetime.timedelta(days=7)
        assert len(result) == 7


# ---------------------------------------------------------------------------
# _merge_intervals
# ---------------------------------------------------------------------------


class TestMergeIntervals:
    def test_empty(self):
        assert _merge_intervals([]) == []

    def test_single(self):
        assert _merge_intervals([(0, 30)]) == [(0, 30)]

    def test_overlapping(self):
        assert _merge_intervals([(0, 30), (20, 60)]) == [(0, 60)]

    def test_touching(self):
        assert _merge_intervals([(0, 30), (30, 60)]) == [(0, 60)]

    def test_non_overlapping(self):
        assert _merge_intervals([(0, 30), (60, 90)]) == [(0, 30), (60, 90)]


# ---------------------------------------------------------------------------
# _free_gaps
# ---------------------------------------------------------------------------


class TestFreeGaps:
    def test_empty_day_one_gap(self):
        gaps = _free_gaps(DATE, {}, {}, min_duration=30)
        assert gaps == [(0, 1440)]

    def test_single_event_two_gaps(self):
        events = {DATE: [make_event(start_minutes=500, end_minutes=600, date=DATE)]}
        gaps = _free_gaps(DATE, events, {}, min_duration=30)
        assert (0, 500) in gaps
        assert (600, 1440) in gaps

    def test_placement_blocks_gap(self):
        gaps = _free_gaps(
            DATE,
            {},
            {DATE: [(500, 600)]},
            min_duration=30,
        )
        assert (0, 500) in gaps
        assert (600, 1440) in gaps

    def test_window_clip(self):
        gaps = _free_gaps(
            DATE, {}, {}, min_duration=10, window_start=360, window_end=1320
        )
        assert gaps == [(360, 1320)]

    def test_too_small_gap_skipped(self):
        # Event at 500..530; gap (530..560) is 30; min_duration=31 to no gap.
        events = {DATE: [make_event(start_minutes=500, end_minutes=530, date=DATE)]}
        gaps = _free_gaps(DATE, events, {DATE: [(560, 700)]}, min_duration=31)
        # gap (530..560) is too small (30 < 31).
        assert (530, 560) not in gaps


# ---------------------------------------------------------------------------
# _earliest_fit_slot
# ---------------------------------------------------------------------------


class TestEarliestFitSlot:
    def test_finds_earliest_gap(self):
        import random

        task = make_task(duration_min=30, duration_max=30)
        slot = _earliest_fit_slot(
            task,
            [DATE],
            events_by_date={},
            placed_by_date={},
            used_base_gaps=set(),
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is not None
        s, e, day, base = slot
        assert s == 0
        assert e == 30
        assert day == DATE
        assert base == (0, 1440)

    def test_no_room_returns_none(self):
        import random

        # Fill the entire window.
        ev = make_event(start_minutes=0, end_minutes=1440, date=DATE)
        task = make_task(duration_min=30, duration_max=30)
        slot = _earliest_fit_slot(
            task,
            [DATE],
            events_by_date={DATE: [ev]},
            placed_by_date={},
            used_base_gaps=set(),
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is None

    def test_walks_to_later_day_when_first_full(self):
        import random

        ev = make_event(start_minutes=0, end_minutes=1440, date=DATE)
        task = make_task(duration_min=30, duration_max=30)
        slot = _earliest_fit_slot(
            task,
            [DATE, DATE2],
            events_by_date={DATE: [ev]},
            placed_by_date={},
            used_base_gaps=set(),
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is not None
        assert slot[2] == DATE2

    def test_prefers_untouched_gap_over_partially_used(self):
        """The "distinct gap" rule should beat raw earliest-fit when
        an untouched gap exists later in the day."""
        import random

        # Two gaps on DATE: [0, 60) and [120, 180), separated by an event.
        ev = make_event(start_minutes=60, end_minutes=120, date=DATE)
        task = make_task(duration_min=20, duration_max=20)
        used = {(DATE, 0, 60)}  # first gap already used by an earlier task
        slot = _earliest_fit_slot(
            task,
            [DATE],
            events_by_date={DATE: [ev]},
            placed_by_date={DATE: [(0, 20)]},
            used_base_gaps=used,
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is not None
        s, e, day, base = slot
        # Greedy should pick the untouched [120, 180) gap, not pack
        # into the already-used first gap.
        assert s == 120
        assert e == 140
        assert base == (120, 1440)

    def test_falls_back_to_reuse_when_no_untouched_gap_fits(self):
        """When every base gap is touched, fall back to reusing one."""
        import random

        ev = make_event(start_minutes=60, end_minutes=1440, date=DATE)
        task = make_task(duration_min=20, duration_max=20)
        used = {(DATE, 0, 60)}
        slot = _earliest_fit_slot(
            task,
            [DATE],
            events_by_date={DATE: [ev]},
            placed_by_date={DATE: [(0, 20)]},
            used_base_gaps=used,
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is not None
        s, e, day, base = slot
        # Only gap is [0, 60), already used.  Fallback packs after
        # the placed [0, 20) task to start at 20.
        assert s == 20
        assert e == 40
        assert base == (0, 60)

    def test_skips_untouched_gap_too_small(self):
        """An untouched base gap that is shorter than `duration_min`
        must be skipped; covers the size-filter branch in the first
        pass."""
        import random

        # Two gaps: [0, 10) is too small for a 20-min task; [60, 200)
        # fits.  Pre-fix the algorithm would not advance past the
        # tiny first gap.
        ev1 = make_event(start_minutes=10, end_minutes=60, date=DATE)
        ev2 = make_event(start_minutes=200, end_minutes=1440, date=DATE)
        task = make_task(duration_min=20, duration_max=20)
        slot = _earliest_fit_slot(
            task,
            [DATE],
            events_by_date={DATE: [ev1, ev2]},
            placed_by_date={},
            used_base_gaps=set(),
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is not None
        s, _e, _day, base = slot
        assert s == 60
        assert base == (60, 200)

    def test_fallback_skips_full_gap_and_too_small_remainder(self):
        """Fallback pass: a fully-occupied gap returns `None` from
        `_gap_starting_minute` (covered branch).  A gap whose
        remainder is shorter than the task is skipped via the size
        check on the remainder (covered branch)."""
        import random

        # Single base gap [0, 60); already filled with [0, 50) so only
        # 10 min remain; not enough for a 20-min task.  Algorithm
        # should return None (no fit anywhere).
        ev = make_event(start_minutes=60, end_minutes=1440, date=DATE)
        task = make_task(duration_min=20, duration_max=20)
        used = {(DATE, 0, 60)}
        slot = _earliest_fit_slot(
            task,
            [DATE],
            events_by_date={DATE: [ev]},
            placed_by_date={DATE: [(0, 50)]},
            used_base_gaps=used,
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is None

    def test_gap_starting_minute_returns_inner_cursor_when_gap_between_placed(self):
        """`_gap_starting_minute` should return the inner cursor when
        there is room between two already-placed tasks in the same
        base gap.  Exercises the "if s > cursor: return cursor" path."""
        from src.scripts.scenarios.augmentation.greedy import _gap_starting_minute

        # Base gap [0, 60), two placements at [0, 10) and [30, 50);
        # gap between 10 and 30 to cursor=10 is returned.
        out = _gap_starting_minute(
            (0, 60),
            DATE,
            {DATE: [(0, 10), (30, 50)]},
        )
        assert out == 10

    def test_gap_starting_minute_returns_none_when_fully_filled(self):
        """When placed tasks fill the entire base gap, the helper
        returns `None`."""
        from src.scripts.scenarios.augmentation.greedy import _gap_starting_minute

        out = _gap_starting_minute(
            (0, 60),
            DATE,
            {DATE: [(0, 60)]},
        )
        assert out is None

    def test_gap_starting_minute_returns_gap_start_when_no_placed(self):
        """When no placed tasks intersect the base gap, the helper
        returns the gap's start (the early-return path)."""
        from src.scripts.scenarios.augmentation.greedy import _gap_starting_minute

        # Placed task on this day but outside the queried base gap.
        out = _gap_starting_minute(
            (0, 60),
            DATE,
            {DATE: [(100, 200)]},
        )
        assert out == 0

    def test_unscheduled_in_repeat_per_week_mode(self):
        """When a week has no free space anywhere, the per-week task
        instance lands in the `unscheduled` list; covers the
        repeat_per_week branch of `augment`."""
        # Fill the entire window on week 1 (and only week 1).
        week1_blocker = make_event(
            label="blk",
            start_minutes=0,
            end_minutes=1440,
            date=DATE,
        )
        cfg = _config(repeat_per_week=True)
        task = make_task(duration_min=30, duration_max=30)
        # 1-week horizon (single day).  The repeat-per-week loop
        # iterates one chunk; the task cannot fit.
        sol = _greedy().augment(_trace(week1_blocker), [task], cfg)
        assert sol.scheduled == []
        assert sol.unscheduled == [task]

    def test_base_gaps_skips_event_entirely_before_window(self):
        """An event whose clipped end falls at or before `prev` does
        not update the cursor and does not emit a spurious gap; covers
        the `clip_e > prev` False branch in `_base_gaps`."""
        from src.scripts.scenarios.augmentation.greedy import _base_gaps

        # Window [100, 200); event entirely outside on the left side
        # (clip_s=100 == prev, clip_e=50 < prev=100 after clipping).
        ev_pre = make_event(start_minutes=0, end_minutes=50, date=DATE)
        gaps = _base_gaps(
            DATE,
            {DATE: [ev_pre]},
            window_start=100,
            window_end=200,
        )
        # The full window is free; the pre-window event contributes
        # nothing.
        assert gaps == [(100, 200)]

    def test_free_gaps_skips_event_entirely_before_window(self):
        """Same edge case for `_free_gaps`; defensive branches on
        clipped events outside the window."""
        ev_pre = make_event(start_minutes=0, end_minutes=50, date=DATE)
        gaps = _free_gaps(
            DATE,
            {DATE: [ev_pre]},
            placed_by_date={},
            min_duration=10,
            window_start=100,
            window_end=200,
        )
        assert gaps == [(100, 200)]

    def test_free_gaps_skips_too_small_tail(self):
        """When the trailing free interval is shorter than
        `min_duration`, `_free_gaps` must not emit it; covers
        the final-tail size guard."""
        # Event ends at 195; window_end at 200; tail [195, 200) is 5
        # min, smaller than min_duration=10.
        ev = make_event(start_minutes=100, end_minutes=195, date=DATE)
        gaps = _free_gaps(
            DATE,
            {DATE: [ev]},
            placed_by_date={},
            min_duration=10,
            window_start=0,
            window_end=200,
        )
        # Only the [0, 100) head remains.
        assert gaps == [(0, 100)]

    def test_fallback_continues_past_fully_filled_gap(self):
        """When the first (used) base gap is fully filled, the
        fallback walks on to the next gap; covers the
        `start_minute is None` continue branch."""
        import random

        # Two base gaps: [0, 10) and [60, 200).  Pre-test setup
        # marks both as used and fills the first one entirely with a
        # placed (0, 10) task, leaving a remainder of 0 minutes.  The
        # second still has room.
        ev = make_event(start_minutes=10, end_minutes=60, date=DATE)
        ev2 = make_event(start_minutes=200, end_minutes=1440, date=DATE)
        task = make_task(duration_min=20, duration_max=20)
        used = {(DATE, 0, 10), (DATE, 60, 200)}
        slot = _earliest_fit_slot(
            task,
            [DATE],
            events_by_date={DATE: [ev, ev2]},
            placed_by_date={DATE: [(0, 10), (60, 80)]},
            used_base_gaps=used,
            window_start=0,
            window_end=1440,
            rng=random.Random(0),
        )
        assert slot is not None
        s, e, _day, base = slot
        # Skipped the fully-filled [0, 10) gap; reused the second one
        # starting at minute 80 (after the placed (60, 80) task).
        assert s == 80
        assert e == 100
        assert base == (60, 200)


# ---------------------------------------------------------------------------
# GreedyAugmenter.augment
# ---------------------------------------------------------------------------


class TestGreedyAugment:
    def test_empty_horizon_returns_unscheduled(self):
        task = make_task()
        sol = _greedy().augment(_trace(), [task], _config())
        assert sol.scheduled == []
        assert sol.unscheduled == [task]

    def test_single_task_placed(self):
        task = make_task(duration_min=30, duration_max=30)
        sol = _greedy().augment(_trace(make_event(date=DATE)), [task], _config())
        assert len(sol.scheduled) == 1
        st = sol.scheduled[0]
        assert st.is_standalone is True
        assert st.concurrent_with is None
        assert st.end_minutes - st.start_minutes == 30

    def test_no_semantic_compatibility_consumed(self):
        """GreedyAugmenter signature must NOT accept a `semantic` kwarg."""
        with pytest.raises(TypeError):
            GreedyAugmenter(semantic="anything")  # type: ignore[call-arg]

    def test_no_merge_field_used(self):
        """allow_merge in config is ignored; greedy never co-schedules."""
        task = make_task(label="reading", duration_min=15, duration_max=15)
        # An event labelled "lunch" with concurrent_with=[reading]; would
        # be a merge candidate under the old impl, but pure FCFS must
        # leave it standalone.
        ev = make_event(
            label="lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
            concurrent_with=["reading"],
        )
        cfg = AugmentationConfig(
            method="greedy",
            allow_merge=True,
            merge_threshold=0.0,
        )
        sol = _greedy().augment(_trace(ev), [task], cfg)
        assert len(sol.scheduled) == 1
        assert sol.scheduled[0].concurrent_with is None

    def test_reproducible_with_same_seed(self):
        ev = make_event(date=DATE)
        task = make_task()
        sol_a = _greedy(seed=42).augment(_trace(ev), [task], _config())
        sol_b = _greedy(seed=42).augment(_trace(ev), [task], _config())
        assert (sol_a.scheduled[0].start_minutes, sol_a.scheduled[0].date) == (
            sol_b.scheduled[0].start_minutes,
            sol_b.scheduled[0].date,
        )

    def test_repeat_per_week_replicates_tasks(self):
        task = make_task(duration_min=30, duration_max=30)
        # 2-week horizon
        events = [
            make_event(date=DATE),
            make_event(date=DATE + datetime.timedelta(days=7)),
        ]
        cfg = _config(repeat_per_week=True)
        sol = _greedy().augment(_trace(*events), [task], cfg)
        assert len(sol.tasks) == 2  # one instance per week

    def test_one_shot_repeat_per_week_false(self):
        task = make_task(duration_min=30, duration_max=30)
        events = [make_event(date=DATE)]
        cfg = _config(repeat_per_week=False)
        sol = _greedy().augment(_trace(*events), [task], cfg)
        assert len(sol.tasks) == 1

    def test_weekly_tasks_places_a_distinct_batch_per_week(self):
        a = make_task("a", duration_min=30, duration_max=30)
        b = make_task("b", duration_min=30, duration_max=30)
        events = [
            make_event(date=DATE),
            make_event(date=DATE + datetime.timedelta(days=7)),
        ]
        sol = _greedy().augment(
            _trace(*events),
            [a, b],
            _config(repeat_per_week=True),
            weekly_tasks=[[a], [b]],
        )
        placed = {st.task.label: st.date for st in sol.scheduled}
        assert placed["a"] < placed["b"]  # a in week 1, b in week 2
        assert len(sol.tasks) == 2

    def test_single_week_list_broadcasts_to_every_week(self):
        a = make_task("a", duration_min=30, duration_max=30)
        events = [
            make_event(date=DATE),
            make_event(date=DATE + datetime.timedelta(days=7)),
        ]
        sol = _greedy().augment(
            _trace(*events),
            [a],
            _config(repeat_per_week=True),
            weekly_tasks=[[a]],
        )
        assert len(sol.tasks) == 2  # broadcast: one instance each week

    def test_unplaceable_task_in_unscheduled(self):
        ev = make_event(start_minutes=0, end_minutes=1440, date=DATE)
        task = make_task(duration_min=30, duration_max=30)
        sol = _greedy().augment(_trace(ev), [task], _config())
        assert sol.scheduled == []
        assert sol.unscheduled == [task]

    def test_carryover_in_one_shot_mode(self):
        """Tasks unplaced in week 1 carry forward to week 2 in one-shot mode."""
        # Fill every day in week 1 so the task cannot land before DATE+7.
        week1_blockers = [
            make_event(
                label=f"blk_{i}",
                start_minutes=0,
                end_minutes=1440,
                date=DATE + datetime.timedelta(days=i),
            )
            for i in range(7)
        ]
        week2_event = make_event(date=DATE + datetime.timedelta(days=7))
        cfg = _config(repeat_per_week=False)
        task = make_task(duration_min=30, duration_max=30)
        sol = _greedy().augment(_trace(*week1_blockers, week2_event), [task], cfg)
        # Task carries forward and lands on the first free day in week 2.
        assert len(sol.scheduled) == 1
        assert sol.scheduled[0].date == DATE + datetime.timedelta(days=7)

    def test_multiple_tasks_spread_across_distinct_gaps(self):
        """Five short tasks plus a calendar with multiple base gaps must
        end up in *different* gaps, not stacked back-to-back in the
        earliest one; that's the user-facing intent of "fill one
        slot, move on to the next"."""
        # Day with 4 distinct base gaps separated by 3 events:
        #   [0, 60), [120, 180), [240, 300), [360, 1440)
        events = [
            make_event(label="ev1", start_minutes=60, end_minutes=120, date=DATE),
            make_event(label="ev2", start_minutes=180, end_minutes=240, date=DATE),
            make_event(label="ev3", start_minutes=300, end_minutes=360, date=DATE),
        ]
        # Five 5-minute tasks; enough to use all four gaps and reuse
        # one for the fifth.
        tasks = [
            make_task(label=f"t{i}", duration_min=5, duration_max=5) for i in range(5)
        ]
        cfg = _config(repeat_per_week=False)
        sol = _greedy().augment(_trace(*events), tasks, cfg)
        assert len(sol.scheduled) == 5
        starts_in_placement_order = [st.start_minutes for st in sol.scheduled]
        # First four placements visit the four distinct gap starts in
        # date+time order: each task moves on to a new free slot.
        assert starts_in_placement_order[:4] == [0, 120, 240, 360]
        # Fifth task overflows; it falls back to reusing the first
        # gap (now occupied by task 0 in [0, 5)), starting at minute 5.
        assert starts_in_placement_order[4] == 5

    def test_repeat_per_week_uses_distinct_gaps_per_week(self):
        """With `repeat_per_week=True` the per-week task list lands in
        distinct gaps each week; the 'used_base_gaps' set is keyed by
        date so week-2 placements start fresh."""
        # Build a 2-week horizon with two distinct gaps per day.
        events = []
        for d in range(14):
            day = DATE + datetime.timedelta(days=d)
            events.append(
                make_event(
                    label=f"split_{d}",
                    start_minutes=600,
                    end_minutes=660,
                    date=day,
                )
            )
        tasks = [
            make_task(label="t1", duration_min=5, duration_max=5),
            make_task(label="t2", duration_min=5, duration_max=5),
        ]
        cfg = _config(repeat_per_week=True)
        sol = _greedy().augment(_trace(*events), tasks, cfg)
        # repeat_per_week to 2 weeks × 2 tasks = 4 instances.
        assert len(sol.scheduled) == 4
        # Within each week, the two tasks land in two *different* base
        # gaps, not stacked in the morning gap.
        by_date: dict = {}
        for st in sol.scheduled:
            by_date.setdefault(st.date, []).append(st.start_minutes)
        for day, starts in by_date.items():
            assert len(starts) == 2
            assert (
                starts[0] != starts[1]
            ), f"On {day}, both tasks stacked at {starts}; expected distinct gaps"
