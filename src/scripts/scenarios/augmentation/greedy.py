"""Pure FCFS greedy calendar augmenter.

The greedy augmenter is the dumb baseline: it does not consume
semantic compatibility, MET, intensity, or rule context. Each desired
task is placed in an unused free gap, in input order.

  1. Compute the day's *base gaps*; the contiguous free intervals
     between existing calendar events, clipped to the daily window.
  2. Walk the horizon in date order; pick the earliest base gap that
     (a) has not yet received a task this run and (b) is wide enough
     for the task's `duration_min`.  Place the task at the gap's
     starting minute.
  3. When **no** untouched gap fits the task anywhere in the horizon,
     fall back to placing it in the earliest gap that still has room
     (a reused gap) so the augmenter still makes progress on dense
     calendars instead of dropping the task.  Ties between
     equally-earliest gaps are broken by the seeded RNG.

Smart behavior (semantic merges, MET-aware spreading, intensity-aware
adjacency, …) is the job of the LLM-agent and RL augmenters.  Keeping
greedy dumb makes it a useful loss-baseline that every other augmenter
must beat to justify its complexity.

**Half-open intervals; adjacency is not overlap.**  Gaps are
`[start, end)` ranges and a task placed at the gap's starting minute
ends up *adjacent* to the preceding event (`task.start_minutes ==
prev_event.end_minutes`).  This is mathematically disjoint -
`[07:41, 07:51)` and `[07:51, 08:11)` share no minute; but the
weekly-chart renderer draws adjacent blocks side-by-side inside the
same hour band, which can make them *look* concurrent.  The two are
not concurrent; if the augmenter must leave a buffer between
placements, that is a separate behavior change (a
`separation_minutes` knob on :class:`GreedyConfig`) and not the
default.  `L_cal`'s `buffer_minutes` already punishes adjacency
at evaluate time without altering greedy's placements.
"""

from __future__ import annotations

import datetime
import logging
import random
from collections import defaultdict
from typing import Any

from src.scripts.persona.config.schema import DailyWindow, WindowRange
from src.scripts.scenarios.augmentation.base import Augmenter
from src.scripts.scenarios.config.schema import AugmentationConfig
from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _get_horizon_dates(
    calendar: CalendarTrace,
    horizon: tuple[datetime.date, int] | None = None,
) -> list[datetime.date]:
    """Return every date inside the horizon.

    When `horizon=(start_date, days)` is given, expand exactly that range;
    otherwise fall back to the first-to-last event date in `calendar`.
    """
    if horizon is not None:
        start, days = horizon
        return [start + datetime.timedelta(days=i) for i in range(days)]
    if not calendar.events:
        return []
    dates = sorted({e.date for e in calendar.events})
    result: list[datetime.date] = []
    d = dates[0]
    while d <= dates[-1]:
        result.append(d)
        d += datetime.timedelta(days=1)
    return result


def _get_weekly_chunks(
    dates: list[datetime.date],
) -> list[list[datetime.date]]:
    """Split a list of consecutive dates into 7-day chunks (weeks)."""
    chunks: list[list[datetime.date]] = []
    for i in range(0, len(dates), 7):
        chunks.append(dates[i : i + 7])
    return chunks


def _tasks_for_week(
    weekly_tasks: list[list[RecommendedTask]], index: int
) -> list[RecommendedTask]:
    """Return a copy of week `index`'s task batch, broadcasting a shorter list.

    A single-element list (a uniform config) is broadcast to every week;
    a per-week list is indexed directly, clamping past the last entry.
    """
    if not weekly_tasks:
        return []
    chosen = weekly_tasks[index] if index < len(weekly_tasks) else weekly_tasks[-1]
    return list(chosen)


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or touching integer intervals; input must be sorted."""
    if not intervals:
        return []
    result = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= result[-1][1]:
            result[-1][1] = max(result[-1][1], e)
        else:
            result.append([s, e])
    return [(a, b) for a, b in result]


def _base_gaps(
    day: datetime.date,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    window_start: int,
    window_end: int,
) -> list[tuple[int, int]]:
    """Return the *base* gaps for *day*; the maximal free intervals
    between existing calendar events, clipped to the daily window.

    Unlike :func:`_free_gaps`, this ignores any tasks already placed
    by the augmenter, so the resulting list is stable across the run
    and can be used to identify which gaps are still untouched.
    """
    occupied: list[tuple[int, int]] = [
        (e.start_minutes, e.end_minutes) for e in events_by_date.get(day, [])
    ]
    merged = _merge_intervals(sorted(occupied))
    gaps: list[tuple[int, int]] = []
    prev = window_start
    for s, e in merged:
        clip_s = max(s, window_start)
        clip_e = min(e, window_end)
        if clip_s > prev:
            gaps.append((prev, clip_s))
        if clip_e > prev:
            prev = clip_e
    if window_end > prev:
        gaps.append((prev, window_end))
    return gaps


def _free_gaps(
    day: datetime.date,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    placed_by_date: dict[datetime.date, list[tuple[int, int]]],
    min_duration: int,
    window_start: int = 0,
    window_end: int = 1440,
) -> list[tuple[int, int]]:
    """Return free `(start, end)` intervals on *day* of length ≥ *min_duration*.

    Clipped to `[window_start, window_end)`.  Accounts for both
    base events and tasks already placed by the augmenter, so the
    returned gaps reflect the calendar's *current* free time; the
    appropriate view for the fallback path that reuses an already-
    touched base gap.
    """
    occupied: list[tuple[int, int]] = [
        (e.start_minutes, e.end_minutes) for e in events_by_date.get(day, [])
    ] + list(placed_by_date.get(day, []))

    merged = _merge_intervals(sorted(occupied))
    gaps: list[tuple[int, int]] = []
    prev = window_start
    for s, e in merged:
        clip_s = max(s, window_start)
        clip_e = min(e, window_end)
        if clip_s > prev and clip_s - prev >= min_duration:
            gaps.append((prev, clip_s))
        if clip_e > prev:
            prev = clip_e
    if window_end - prev >= min_duration:
        gaps.append((prev, window_end))
    return gaps


def _gap_starting_minute(
    base_gap: tuple[int, int],
    day: datetime.date,
    placed_by_date: dict[datetime.date, list[tuple[int, int]]],
) -> int | None:
    """Return the earliest minute inside *base_gap* that has room for a
    task, given already-placed tasks on *day*.

    Returns `None` when the gap has been completely filled by
    earlier placements.  When the gap is fresh (no overlap with
    placed tasks), this is simply the gap's start.
    """
    gs, ge = base_gap
    placed = [(s, e) for s, e in placed_by_date.get(day, []) if e > gs and s < ge]
    if not placed:
        return gs
    placed.sort()
    cursor = gs
    for s, e in placed:
        if s > cursor:
            return cursor
        cursor = max(cursor, e)
    return cursor if cursor < ge else None


def _earliest_fit_slot(
    task: RecommendedTask,
    horizon: list[datetime.date],
    events_by_date: dict,
    placed_by_date: dict,
    used_base_gaps: set[tuple[datetime.date, int, int]],
    *,
    window_start: int,
    window_end: int,
    rng: random.Random,
) -> tuple[int, int, datetime.date, tuple[int, int]] | None:
    """Find a placement for *task*; return `(start, end, day, base_gap)`.

    First pass; *distinct* gaps: walk the horizon in date order
    looking for the **earliest** untouched base gap (one not in
    `used_base_gaps`) of length ≥ `task.duration_min`.  Only the
    earliest is collected per call; gaps are disjoint within a day so
    there is no true tie, but the seeded `rng.choice` is preserved
    to keep the API surface stable for future overrides (e.g. random
    epoch tie-breaks).

    Fallback pass; *reuse*: if no untouched gap fits anywhere in the
    horizon, walk again and pick the earliest gap that still has room
    *after* prior placements.  This matches the "fills it, goes to a
    new free slot" intent; distinct gaps come first, and when the
    calendar runs out of fresh gaps the augmenter keeps packing tasks
    into already-used gaps rather than dropping them.

    Returns `None` when no fit exists anywhere in the horizon.
    """
    # First pass; earliest untouched gap.
    untouched: list[tuple[int, int, datetime.date, tuple[int, int]]] = []
    for day in horizon:
        for base in _base_gaps(day, events_by_date, window_start, window_end):
            key = (day, base[0], base[1])
            if key in used_base_gaps:
                continue
            if base[1] - base[0] < task.duration_min:
                continue
            start = base[0]
            end = min(base[1], start + max(task.duration_min, task.duration_max))
            untouched.append((start, end, day, base))
            # Only collect the earliest untouched gap on this day -
            # gaps are sorted by start so the first qualifying one is
            # the day's earliest.
            break
        if untouched:
            # And only the earliest day with any candidate is kept.
            break

    if untouched:
        return rng.choice(untouched)

    # Fallback; earliest reusable gap.
    reused: list[tuple[int, int, datetime.date, tuple[int, int]]] = []
    for day in horizon:
        for base in _base_gaps(day, events_by_date, window_start, window_end):
            start_minute = _gap_starting_minute(base, day, placed_by_date)
            if start_minute is None:
                continue
            if base[1] - start_minute < task.duration_min:
                continue
            end = min(base[1], start_minute + max(task.duration_min, task.duration_max))
            reused.append((start_minute, end, day, base))
            break
        if reused:
            break

    if not reused:
        return None
    return rng.choice(reused)


# ---------------------------------------------------------------------------
# GreedyAugmenter
# ---------------------------------------------------------------------------


class GreedyAugmenter(Augmenter):
    """Pure FCFS greedy augmenter.

    Args:
        time_windows: kept for backwards-compatible API but **unused**
            by the placement algorithm (FCFS does not consider epoch
            preferences).  Surfaces in the signature so existing
            callers (cli wiring) keep compiling.
        daily_window: hard waking-hours bound applied to every day.
            Defaults to `[06:00, 22:00)` via `DailyWindow`.
        seed: RNG seed for tie-breaking; typically forwarded from
            `environment.seed`.  Defaults to 0 when omitted.
    """

    def __init__(
        self,
        time_windows: dict[str, Any] | None = None,
        daily_window: DailyWindow | None = None,
        seed: int | None = None,
    ) -> None:
        # Unused fields kept on self for backwards-compatible introspection.
        self._time_windows: dict[str, WindowRange] = time_windows or {}
        self._daily_window: DailyWindow = daily_window or DailyWindow()
        self._seed = int(seed) if seed is not None else 0
        self._rng = random.Random(self._seed)

    def augment(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
        config: AugmentationConfig,
        horizon: tuple[datetime.date, int] | None = None,
        *,
        weekly_tasks: list[list[RecommendedTask]] | None = None,
    ) -> SchedulingSolution:
        """Place recommended tasks week by week and return a solution.

        `weekly_tasks` gives each week its own batch; when omitted, `tasks`
        is broadcast to every week (the uniform case).

        Pure FCFS: each task gets the earliest free slot ≥ its
        `duration_min` inside the daily window.  No merges, no
        backtracking, no preference scoring.

        The `observation` block on `config` parses for cross-method
        consistency (LLM vs greedy comparisons keep the same scenario
        config shape) but is otherwise ignored. A single INFO log is
        emitted per run when the block is non-default so the scenarios
        author can confirm greedy intentionally skipped the signal.
        """
        observation = getattr(config, "observation", None)
        if observation is not None and (observation.contexts or observation.host_flags):
            logger.info(
                "greedy: observation block ignored "
                "(contexts=%s, host_flags=%s, context_detail=%s)",
                list(observation.contexts),
                list(observation.host_flags),
                observation.context_detail,
            )
        horizon_dates = _get_horizon_dates(calendar, horizon=horizon)
        if not horizon_dates:
            aug_cal = AugmentedCalendar(
                person_id=calendar.person_id,
                base_events=list(calendar.events),
            )
            return SchedulingSolution(
                person_id=calendar.person_id,
                augmented_calendar=aug_cal,
                tasks=list(tasks),
                scheduled=[],
                unscheduled=list(tasks),
            )

        events_by_date: dict[datetime.date, list[CalendarEvent]] = defaultdict(list)
        for e in calendar.events:
            events_by_date[e.date].append(e)
        for d in events_by_date:
            events_by_date[d].sort(key=lambda e: e.start_minutes)

        placed_by_date: dict[datetime.date, list[tuple[int, int]]] = defaultdict(list)
        # Tracks which base gaps have already received a task this run.
        # The set is keyed by `(date, gap_start, gap_end)` so each
        # day's gaps are tracked independently; placement on one
        # day's morning gap does not "use up" another day's morning gap.
        used_base_gaps: set[tuple[datetime.date, int, int]] = set()

        scheduled: list[ScheduledTask] = []
        unscheduled: list[RecommendedTask] = []
        all_task_instances: list[RecommendedTask] = []

        weekly_chunks = _get_weekly_chunks(horizon_dates)
        weeks = weekly_tasks if weekly_tasks is not None else [tasks]
        wake = self._daily_window.wake_minutes
        sleep = self._daily_window.sleep_minutes

        if config.repeat_per_week:
            for week_index, week_dates in enumerate(weekly_chunks):
                week_tasks = _tasks_for_week(weeks, week_index)
                all_task_instances.extend(week_tasks)
                for task in week_tasks:
                    slot = _earliest_fit_slot(
                        task,
                        week_dates,
                        events_by_date,
                        placed_by_date,
                        used_base_gaps,
                        window_start=wake,
                        window_end=sleep,
                        rng=self._rng,
                    )
                    if slot is None:
                        unscheduled.append(task)
                        continue
                    s, e, day, base = slot
                    st = ScheduledTask(
                        task=task,
                        start_minutes=s,
                        end_minutes=e,
                        is_standalone=True,
                        concurrent_with=None,
                        date=day,
                    )
                    scheduled.append(st)
                    placed_by_date[day].append((s, e))
                    used_base_gaps.add((day, base[0], base[1]))
        else:
            all_task_instances = list(tasks)
            remaining: list[RecommendedTask] = list(tasks)
            for week_dates in weekly_chunks:
                week_unplaced: list[RecommendedTask] = []
                for task in remaining:
                    slot = _earliest_fit_slot(
                        task,
                        week_dates,
                        events_by_date,
                        placed_by_date,
                        used_base_gaps,
                        window_start=wake,
                        window_end=sleep,
                        rng=self._rng,
                    )
                    if slot is None:
                        week_unplaced.append(task)
                        continue
                    s, e, day, base = slot
                    scheduled.append(
                        ScheduledTask(
                            task=task,
                            start_minutes=s,
                            end_minutes=e,
                            is_standalone=True,
                            concurrent_with=None,
                            date=day,
                        )
                    )
                    placed_by_date[day].append((s, e))
                    used_base_gaps.add((day, base[0], base[1]))
                remaining = week_unplaced
            unscheduled = remaining

        aug_cal = AugmentedCalendar(
            person_id=calendar.person_id,
            base_events=list(calendar.events),
            scheduled_tasks=scheduled,
        )
        return SchedulingSolution(
            person_id=calendar.person_id,
            augmented_calendar=aug_cal,
            tasks=all_task_instances,
            scheduled=scheduled,
            unscheduled=unscheduled,
        )
