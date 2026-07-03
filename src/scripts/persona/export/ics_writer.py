"""Render a `PersonSchedule` as a per-person ICS calendar file.

The mapping is intentionally simple: every (event_name, start, duration)
becomes one VEVENT, with a `dtstart` derived from the day's date. Sleep is
also emitted as a `go_to_bed` and `wake_up` pair so the resulting calendar
matches the legacy ICS layout for downstream consumers.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from icalendar import Calendar, Event

from src.scripts.persona.config.schema import EventConfig
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

_BOOKEND_MINUTES = 1


def _to_datetime(date: _dt.date, minutes: int) -> _dt.datetime:
    """Combine a date with a minutes-since-midnight offset.

    Spillovers/sleep can have `start + duration` past 1440. The caller is
    expected to keep `minutes` in [0, 1440); we clamp the extra hours by
    rolling the date forward.
    """
    extra_days, mins_in_day = divmod(minutes, 24 * 60)
    return _dt.datetime.combine(
        date + _dt.timedelta(days=extra_days),
        _dt.time(hour=mins_in_day // 60, minute=mins_in_day % 60),
    )


def _label_for(
    event_name: str,
    catalog: Catalog | None,
    event_index: int,
) -> str | None:
    """Return the rotating task-label for an event if the catalog set any."""
    if catalog is None or event_name not in catalog:
        return None
    event = catalog.get(event_name)
    if event.calendar_variations is None or not event.calendar_variations.task_labels:
        return None
    labels = event.calendar_variations.task_labels
    return labels[event_index % len(labels)]


def _add_sleep_block(
    cal: Calendar,
    person_id: str,
    date: _dt.date,
    start: int,
    duration: int,
) -> None:
    dtstart = _to_datetime(date, start)
    dtend = dtstart + _dt.timedelta(minutes=duration)

    sleep_event = Event()
    sleep_event.add("summary", f"sleep (Person {person_id})")
    sleep_event.add("dtstart", dtstart)
    sleep_event.add("dtend", dtend)
    sleep_event.add(
        "description", f"Generated event for sleep, duration {duration} min"
    )
    cal.add_component(sleep_event)

    go_to_bed = Event()
    go_to_bed.add("summary", f"go_to_bed (Person {person_id})")
    go_to_bed.add("dtstart", dtstart)
    go_to_bed.add("dtend", dtstart + _dt.timedelta(minutes=_BOOKEND_MINUTES))
    go_to_bed.add("description", "Generated event for go_to_bed (was sleep start)")
    cal.add_component(go_to_bed)

    wake_up = Event()
    wake_up.add("summary", f"wake_up (Person {person_id})")
    wake_up.add("dtstart", dtend)
    wake_up.add("dtend", dtend + _dt.timedelta(minutes=_BOOKEND_MINUTES))
    wake_up.add("description", "Generated event for wake_up (was sleep end)")
    cal.add_component(wake_up)


def _add_generic_event(
    cal: Calendar,
    person_id: str,
    date: _dt.date,
    event_name: str,
    start: int,
    duration: int,
    label: str | None,
) -> None:
    dtstart = _to_datetime(date, start)
    dtend = dtstart + _dt.timedelta(minutes=duration)
    event = Event()
    if label is not None:
        summary = f"{label} (Person {person_id})"
        description = (
            f"Generated event for {label} ({event_name}), duration {duration} min"
        )
    else:
        summary = f"{event_name} (Person {person_id})"
        description = f"Generated event for {event_name}, duration {duration} min"
    event.add("summary", summary)
    event.add("dtstart", dtstart)
    event.add("dtend", dtend)
    event.add("description", description)
    cal.add_component(event)


def _emit_day(
    cal: Calendar,
    person_id: str,
    day: DaySchedule,
    catalog: Catalog | None,
    counters: dict[str, int],
) -> None:
    for event_name in sorted(day.events):
        for ev in day.events[event_name]:
            if event_name == "sleep":
                _add_sleep_block(cal, person_id, day.date, ev.start, ev.duration)
                continue
            if ev.label is not None:
                label = ev.label
            else:
                label = _label_for(event_name, catalog, counters.get(event_name, 0))
                counters[event_name] = counters.get(event_name, 0) + 1
            _add_generic_event(
                cal,
                person_id=person_id,
                date=day.date,
                event_name=event_name,
                start=ev.start,
                duration=ev.duration,
                label=label,
            )


def _add_context_episode(
    cal: Calendar,
    person_id: str,
    episode,
) -> None:
    dtstart = _to_datetime(episode.date, episode.start_minutes)
    dtend = _to_datetime(episode.date, episode.end_minutes)
    event = Event()
    event.add("summary", f"[ctx:{episode.category}] {episode.name}")
    event.add("dtstart", dtstart)
    event.add("dtend", dtend)
    event.add("categories", episode.category)
    desc_parts = [f"category={episode.category}", f"name={episode.name}"]
    if episode.dimension:
        desc_parts.append(f"dimension={episode.dimension}")
    if episode.polarity:
        desc_parts.append(f"polarity={episode.polarity}")
    if episode.ontology_uri:
        desc_parts.append(f"iri={episode.ontology_uri}")
    event.add("description", "; ".join(desc_parts))
    cal.add_component(event)


def _emit_contexts(cal: Calendar, schedule: PersonSchedule) -> None:
    for episode in sorted(schedule.contexts, key=lambda e: (e.date, e.start_minutes)):
        _add_context_episode(cal, schedule.person_id, episode)


def schedule_to_ics_bytes(
    schedule: PersonSchedule,
    catalog: Catalog | None = None,
) -> bytes:
    """Render a `PersonSchedule` to an ICS byte string."""
    cal = Calendar()
    cal.add("prodid", "-//calendar-bench//persona//EN")
    cal.add("version", "2.0")
    counters: dict[str, int] = {}
    for day in schedule.days:
        _emit_day(cal, schedule.person_id, day, catalog, counters)
    _emit_contexts(cal, schedule)
    return cal.to_ical()


def write_person_ics(
    schedule: PersonSchedule,
    out_dir: Path | str,
    catalog: Catalog | None = None,
) -> Path:
    """Write `<out_dir>/<person_id>.ics` and return the path."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    target = out_dir_path / f"{schedule.person_id}.ics"
    target.write_bytes(schedule_to_ics_bytes(schedule, catalog=catalog))
    return target


def write_population_ics(
    schedules: list[PersonSchedule],
    out_dir: Path | str,
    catalog: Catalog | None = None,
) -> list[Path]:
    """Write one ICS per person and return the paths."""
    return [write_person_ics(s, out_dir, catalog=catalog) for s in schedules]


def catalog_from_event_config(event_config: EventConfig) -> Catalog:
    """Convenience re-exposure for callers that only have an EventConfig."""
    return Catalog.from_event_config(event_config)


__all__ = [
    "catalog_from_event_config",
    "schedule_to_ics_bytes",
    "write_person_ics",
    "write_population_ics",
]
