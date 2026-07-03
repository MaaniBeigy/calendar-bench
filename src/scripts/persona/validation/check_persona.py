"""Validate that the realized population matches its persona templates.

We check three properties:

- per-persona instance counts match `Persona.instances`;
- every `Person.occupation_status` matches the legacy `occupation_status:`
  persona field when one is declared (per-axis characteristic conformance
  is covered by `check_characteristics`);
- every persona-stated stage appears in the realized schedule at the
  cadence the persona declared. Stages with `days` contribute one
  expected episode per matching weekday in the horizon; stages with
  `date` contribute one expected episode on that date.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from dataclasses import dataclass

from src.scripts.persona.config.defaults import WEEKDAY_INDEX
from src.scripts.persona.config.schema import Persona, PersonaConfig
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule


@dataclass(frozen=True, slots=True)
class PersonaViolation:
    """One mismatch between persona expectation and realized output."""

    persona_id: str
    person_id: str | None
    kind: str  # instance_count | occupation | cadence | planned_date
    detail: str


def _check_instance_counts(
    persona_config: PersonaConfig, persons: list[Person]
) -> list[PersonaViolation]:
    counts = Counter(p.persona_id for p in persons)
    out: list[PersonaViolation] = []
    declared: dict[str, Persona] = {p.id: p for p in persona_config.personas}
    for persona in persona_config.personas:
        actual = counts.get(persona.id, 0)
        if actual != persona.instances:
            out.append(
                PersonaViolation(
                    persona_id=persona.id,
                    person_id=None,
                    kind="instance_count",
                    detail=f"expected {persona.instances} persons, got {actual}",
                )
            )
    for persona_id in counts:
        if persona_id not in declared:
            out.append(
                PersonaViolation(
                    persona_id=persona_id,
                    person_id=None,
                    kind="instance_count",
                    detail=f"persona {persona_id!r} has persons but no declaration",
                )
            )
    return out


def _check_occupation(
    persona_config: PersonaConfig, persons: list[Person]
) -> list[PersonaViolation]:
    declared = {p.id: p.occupation_status for p in persona_config.personas}
    out: list[PersonaViolation] = []
    for person in persons:
        expected = declared.get(person.persona_id)
        if expected is None:
            continue  # already reported by _check_instance_counts
        if person.occupation_status != expected:
            out.append(
                PersonaViolation(
                    persona_id=person.persona_id,
                    person_id=person.person_id,
                    kind="occupation",
                    detail=(
                        f"expected occupation {expected!r}, "
                        f"got {person.occupation_status!r}"
                    ),
                )
            )
    return out


def _weekdays_in_horizon(start_date: _dt.date, horizon_days: int, target: str) -> int:
    target_idx = WEEKDAY_INDEX[target]
    count = 0
    for offset in range(horizon_days):
        if (start_date + _dt.timedelta(days=offset)).weekday() == target_idx:
            count += 1
    return count


def _check_stage_cadence(
    person: Person,
    schedule: PersonSchedule,
    start_date: _dt.date,
) -> list[PersonaViolation]:
    """Aggregate expected episodes from `person.stages` and compare to realised."""
    out: list[PersonaViolation] = []
    horizon_days = len(schedule.days)
    counts: dict[str, int] = {}
    for day in schedule.days:
        for name, events in day.events.items():
            counts[name] = counts.get(name, 0) + len(events)

    expected_per_event: dict[str, int] = {}
    schedule_dates: set[_dt.date] = {d.date for d in schedule.days}

    for stage in person.stages:
        # Weekly cadence: one expected episode per matching weekday
        # inside the horizon.
        for weekday in stage.days:
            expected_per_event[stage.name] = expected_per_event.get(
                stage.name, 0
            ) + _weekdays_in_horizon(start_date, horizon_days, weekday)
        # One-off date trigger: one expected episode on that date, only
        # if the date falls inside the run horizon (otherwise the
        # schedule has no day to host it).
        if stage.date is not None and stage.date in schedule_dates:
            expected_per_event[stage.name] = expected_per_event.get(stage.name, 0) + 1

    for event_name, expected in expected_per_event.items():
        actual = counts.get(event_name, 0)
        if actual < expected:
            out.append(
                PersonaViolation(
                    persona_id=person.persona_id,
                    person_id=person.person_id,
                    kind="cadence",
                    detail=(
                        f"event {event_name!r}: expected at least {expected} "
                        f"episodes, found {actual}"
                    ),
                )
            )
    return out


def _check_dated_stages(
    person: Person, schedule: PersonSchedule
) -> list[PersonaViolation]:
    """One-off `date`-anchored stages must produce an event on that date."""
    out: list[PersonaViolation] = []
    by_date: dict[_dt.date, dict[str, int]] = {}
    for day in schedule.days:
        per_name: dict[str, int] = {}
        for name, events in day.events.items():
            per_name[name] = len(events)
        by_date[day.date] = per_name
    for stage in person.stages:
        if stage.date is None:
            continue
        if stage.date not in by_date:
            continue  # date outside horizon - nothing to check
        if by_date[stage.date].get(stage.name, 0) == 0:
            out.append(
                PersonaViolation(
                    persona_id=person.persona_id,
                    person_id=person.person_id,
                    kind="planned_date",
                    detail=(
                        f"event {stage.name!r} expected on "
                        f"{stage.date.isoformat()} but missing"
                    ),
                )
            )
    return out


def check_persona_constraints(
    persona_config: PersonaConfig,
    persons: list[Person],
    schedules: list[PersonSchedule],
    *,
    start_date: _dt.date,
) -> list[PersonaViolation]:
    """Run all three persona checks and return the flat violation list."""
    if len(persons) != len(schedules):
        raise ValueError(
            f"persons ({len(persons)}) and schedules ({len(schedules)}) "
            "lengths must match"
        )

    violations: list[PersonaViolation] = []
    violations.extend(_check_instance_counts(persona_config, persons))
    violations.extend(_check_occupation(persona_config, persons))

    for person, schedule in zip(persons, schedules):
        violations.extend(_check_stage_cadence(person, schedule, start_date))
        violations.extend(_check_dated_stages(person, schedule))
    return violations


__all__ = ["PersonaViolation", "check_persona_constraints"]
