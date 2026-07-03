"""Validate every emitted event against its persona's effective catalog.

Three flavours of violation are reported:

- `per_event_duration`: a single event's duration is outside `[min, max]`.
- `per_day`: total duration or episode count on a day is outside `[min, max]`.
- `weekday`: the catalog restricts the event to certain weekdays and the
  emitted day is not one of them.

Each persona may override the global catalog through `Person.event_overrides`.
The validator merges the base catalog with each person's overrides so a
student running 30-60 min and a fulltime worker running 60-90 min both
pass when their schedules sit inside their own bounds, even though the
two cohorts share the global event name.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.scripts.persona.config.schema import EventDefinition
from src.scripts.persona.domain.event import Catalog, apply_event_overrides
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule


def _to_minutes(value: int, unit: str) -> int:
    return value * 60 if unit == "hours" else value


@dataclass(frozen=True, slots=True)
class EventViolation:
    """One violation found by `check_event_constraints`."""

    person_id: str
    day_index: int
    event_name: str
    kind: str  # per_event_duration | per_day | weekday
    detail: str
    event_idx: int | None = None


def _check_per_event(
    schedule: PersonSchedule,
    event_def: EventDefinition,
) -> list[EventViolation]:
    out: list[EventViolation] = []
    name = event_def.name
    min_dur = _to_minutes(
        event_def.per_event_duration.min, event_def.per_event_duration.unit
    )
    max_dur = _to_minutes(
        event_def.per_event_duration.max, event_def.per_event_duration.unit
    )
    for day in schedule.days:
        events = day.events.get(name, [])
        for idx, ev in enumerate(events):
            if ev.duration < min_dur:
                out.append(
                    EventViolation(
                        person_id=schedule.person_id,
                        day_index=day.day_index,
                        event_name=name,
                        kind="per_event_duration",
                        detail=f"duration {ev.duration} < min {min_dur}",
                        event_idx=idx,
                    )
                )
            elif ev.duration > max_dur:
                out.append(
                    EventViolation(
                        person_id=schedule.person_id,
                        day_index=day.day_index,
                        event_name=name,
                        kind="per_event_duration",
                        detail=f"duration {ev.duration} > max {max_dur}",
                        event_idx=idx,
                    )
                )
    return out


def _check_per_day(
    schedule: PersonSchedule,
    event_def: EventDefinition,
) -> list[EventViolation]:
    if event_def.total_event_duration.scale != "day":
        return []
    if event_def.total_event_episodes.scale != "day":
        return []
    name = event_def.name
    min_total = _to_minutes(
        event_def.total_event_duration.min, event_def.total_event_duration.unit
    )
    max_total = _to_minutes(
        event_def.total_event_duration.max, event_def.total_event_duration.unit
    )
    min_ep = event_def.total_event_episodes.min
    max_ep = event_def.total_event_episodes.max
    out: list[EventViolation] = []
    for day in schedule.days:
        events = day.events.get(name, [])
        if not events:
            continue
        total = sum(ev.duration for ev in events)
        episodes = len(events)
        if total < min_total:
            out.append(
                EventViolation(
                    person_id=schedule.person_id,
                    day_index=day.day_index,
                    event_name=name,
                    kind="per_day",
                    detail=f"total duration {total} < min {min_total}",
                )
            )
        if total > max_total:
            out.append(
                EventViolation(
                    person_id=schedule.person_id,
                    day_index=day.day_index,
                    event_name=name,
                    kind="per_day",
                    detail=f"total duration {total} > max {max_total}",
                )
            )
        if episodes < min_ep:
            out.append(
                EventViolation(
                    person_id=schedule.person_id,
                    day_index=day.day_index,
                    event_name=name,
                    kind="per_day",
                    detail=f"episodes {episodes} < min {min_ep}",
                )
            )
        if episodes > max_ep:
            out.append(
                EventViolation(
                    person_id=schedule.person_id,
                    day_index=day.day_index,
                    event_name=name,
                    kind="per_day",
                    detail=f"episodes {episodes} > max {max_ep}",
                )
            )
    return out


def _check_weekday(
    schedule: PersonSchedule,
    event_def: EventDefinition,
) -> list[EventViolation]:
    if event_def.weekdays is None:
        return []
    allowed = set(event_def.weekdays)
    name = event_def.name
    out: list[EventViolation] = []
    for day in schedule.days:
        if name not in day.events or not day.events[name]:
            continue
        if day.weekday in allowed:
            continue
        out.append(
            EventViolation(
                person_id=schedule.person_id,
                day_index=day.day_index,
                event_name=name,
                kind="weekday",
                detail=f"emitted on {day.weekday} but allowed only on {sorted(allowed)}",
            )
        )
    return out


def check_event_constraints(
    schedules: list[PersonSchedule],
    persons: list[Person],
    base_catalog: Catalog,
) -> list[EventViolation]:
    """Run all three checks against each person's effective catalog.

    `schedules` and `persons` must be paired by index. The function applies
    `apply_event_overrides(base_catalog, person.event_overrides)` once per
    person and validates that person's schedule against the result. A persona
    that does not override any event sees the base catalog unchanged.
    """
    if len(schedules) != len(persons):
        raise ValueError(
            f"schedules ({len(schedules)}) and persons ({len(persons)}) "
            "lengths must match"
        )
    violations: list[EventViolation] = []
    for schedule, person in zip(schedules, persons):
        effective = apply_event_overrides(base_catalog, person.event_overrides)
        for event_def in effective.events_by_name.values():
            violations.extend(_check_per_event(schedule, event_def))
            violations.extend(_check_per_day(schedule, event_def))
            violations.extend(_check_weekday(schedule, event_def))
    return violations


__all__ = ["EventViolation", "check_event_constraints"]
