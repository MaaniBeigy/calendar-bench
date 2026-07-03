"""Write an augmented per-person calendar as an ICS file."""

from __future__ import annotations

import datetime
from pathlib import Path

from icalendar import Calendar, Event

from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution


def _minutes_to_datetime(date: datetime.date, minutes: int) -> datetime.datetime:
    """Convert `(date, minutes-from-midnight)` to a naive `datetime`."""
    extra_days, mins_in_day = divmod(minutes, 1440)
    return datetime.datetime.combine(
        date + datetime.timedelta(days=extra_days),
        datetime.time(hour=mins_in_day // 60, minute=mins_in_day % 60),
    )


def _base_event_component(base_ev) -> Event:
    """Build the ICS Event component for a base `CalendarEvent`."""
    ev = Event()
    ev.add("summary", base_ev.effective_label.replace("_", " ").title())
    ev.add("dtstart", _minutes_to_datetime(base_ev.date, base_ev.start_minutes))
    ev.add("dtend", _minutes_to_datetime(base_ev.date, base_ev.end_minutes))
    ev.add("categories", ["BASE_CALENDAR"])
    return ev


def _scheduled_task_component(st) -> Event:
    """Build the ICS Event component for an augmented `ScheduledTask`."""
    ev = Event()
    summary = st.task.effective_display_name
    if st.concurrent_with:
        summary += f" (during {st.concurrent_with})"
    ev.add("summary", summary)
    ev.add("dtstart", _minutes_to_datetime(st.date, st.start_minutes))
    ev.add("dtend", _minutes_to_datetime(st.date, st.end_minutes))
    desc = st.task.effective_description or (
        f"intensity: {st.task.intensity} | is_dividable: {st.task.is_dividable}"
    )
    ev.add("description", desc)
    ev.add("categories", ["AUGMENTED"])
    return ev


def solution_to_ical_bytes(
    solution: SchedulingSolution,
    calendar: CalendarTrace | None = None,
) -> bytes:
    """Render a full augmented calendar as an ICS byte string.

    Args:
        solution: augmentation result containing the scheduled tasks.
        calendar: the person's original `CalendarTrace`; when provided its
            events are interleaved with the augmented tasks so the resulting
            ICS file lists every entry in chronological order; the base
            events keep `CATEGORIES:BASE_CALENDAR` and the augmentations
            keep `CATEGORIES:AUGMENTED` so a calendar app can still
            color them differently.
    """
    cal = Calendar()
    cal.add("prodid", "-//CalendarBench Scenarios//EN")
    cal.add("version", "2.0")

    # Build a single (date, start_minutes); sort key. Base events get a
    # secondary key of 0 so they appear before any task that starts at the
    # same minute (deterministic but otherwise arbitrary).
    items: list[tuple] = []
    if calendar is not None:
        for base_ev in calendar.events:
            items.append(((base_ev.date, base_ev.start_minutes, 0), "base", base_ev))
    for st in solution.scheduled:
        items.append(((st.date, st.start_minutes, 1), "task", st))
    items.sort(key=lambda x: x[0])

    for _, kind, payload in items:
        if kind == "base":
            cal.add_component(_base_event_component(payload))
        else:
            cal.add_component(_scheduled_task_component(payload))

    return cal.to_ical()


def write_augmented_ics(
    solution: SchedulingSolution,
    out_path: Path,
    *,
    calendar: CalendarTrace | None = None,
) -> Path:
    """Write the full augmented calendar for *solution* to *out_path* (ICS).

    Args:
        solution: augmentation result.
        out_path: destination `.ics` file path (parent dirs created).
        calendar: original `CalendarTrace`; passed through to
            :func:`solution_to_ical_bytes` to include base events.

    Returns:
        The written file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(solution_to_ical_bytes(solution, calendar=calendar))
    return out_path
