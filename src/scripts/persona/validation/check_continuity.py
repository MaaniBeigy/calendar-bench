"""Cross-day correctness checks: spillover bookkeeping and weekday consistency.

Three rules apply across the whole horizon:

- Day 0 must carry an empty `spillovers` list. Day 0 has no prior day to
  inherit from.
- Each day's `spillovers` must equal the previous day's set of events that
  ended past minute 1440 (with `start = 0`, `duration = overflow`,
  `event_idx` matching the source).
- Each day's events must not overlap any range listed in its `spillovers`.

Weekday and date are also checked: dates must increment by one, and weekday
must match the date's actual weekday name.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from src.scripts.persona.config.defaults import WEEKDAY_ORDER
from src.scripts.persona.domain.schedule import PersonSchedule

DAY_MINUTES = 1440


@dataclass(frozen=True, slots=True)
class ContinuityViolation:
    """One cross-day inconsistency."""

    person_id: str
    day_index: int
    kind: str  # day0_spillover | spillover_mismatch | overlap | date | weekday
    detail: str


def _expected_spillovers(
    prev_day_events: dict[str, list],
) -> list[tuple[str, int, int, int, int]]:
    """Return (event_name, spill_dur, orig_start, orig_dur, event_idx) tuples."""
    out: list[tuple[str, int, int, int, int]] = []
    for name, events in prev_day_events.items():
        for idx, ev in enumerate(events):
            if ev.start + ev.duration > DAY_MINUTES:
                overflow = (ev.start + ev.duration) - DAY_MINUTES
                out.append((name, overflow, ev.start, ev.duration, idx))
    return out


def _check_one(
    schedule: PersonSchedule,
) -> list[ContinuityViolation]:
    out: list[ContinuityViolation] = []
    if not schedule.days:
        return out

    if schedule.days[0].spillovers:
        out.append(
            ContinuityViolation(
                person_id=schedule.person_id,
                day_index=0,
                kind="day0_spillover",
                detail=f"day 0 has {len(schedule.days[0].spillovers)} spillovers; expected 0",
            )
        )

    for i in range(1, len(schedule.days)):
        prev = schedule.days[i - 1]
        cur = schedule.days[i]
        expected = _expected_spillovers(prev.events)
        actual = [
            (s.event_name, s.duration, s.orig_start, s.orig_duration, s.event_idx)
            for s in cur.spillovers
        ]
        if sorted(expected) != sorted(actual):
            out.append(
                ContinuityViolation(
                    person_id=schedule.person_id,
                    day_index=cur.day_index,
                    kind="spillover_mismatch",
                    detail=(
                        f"expected spillovers {sorted(expected)}, "
                        f"got {sorted(actual)}"
                    ),
                )
            )
        for spill in cur.spillovers:
            occ_start = spill.start
            occ_end = spill.start + spill.duration
            for events in cur.events.values():
                for ev in events:
                    if not (ev.start + ev.duration <= occ_start or ev.start >= occ_end):
                        out.append(
                            ContinuityViolation(
                                person_id=schedule.person_id,
                                day_index=cur.day_index,
                                kind="overlap",
                                detail=(
                                    f"event {ev.event_name} (start={ev.start}, "
                                    f"dur={ev.duration}) overlaps spillover "
                                    f"[{occ_start}, {occ_end})"
                                ),
                            )
                        )

    base = schedule.days[0].date
    for i, day in enumerate(schedule.days):
        expected_date = base + _dt.timedelta(days=i)
        if day.date != expected_date:
            out.append(
                ContinuityViolation(
                    person_id=schedule.person_id,
                    day_index=day.day_index,
                    kind="date",
                    detail=f"expected date {expected_date.isoformat()}, got {day.date.isoformat()}",
                )
            )
        expected_weekday = WEEKDAY_ORDER[day.date.weekday()]
        if day.weekday != expected_weekday:
            out.append(
                ContinuityViolation(
                    person_id=schedule.person_id,
                    day_index=day.day_index,
                    kind="weekday",
                    detail=f"expected weekday {expected_weekday!r}, got {day.weekday!r}",
                )
            )
    return out


def check_continuity(
    schedules: list[PersonSchedule],
) -> list[ContinuityViolation]:
    """Run continuity checks across every schedule."""
    out: list[ContinuityViolation] = []
    for schedule in schedules:
        out.extend(_check_one(schedule))
    return out


__all__ = ["ContinuityViolation", "check_continuity"]
