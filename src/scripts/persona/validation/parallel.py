"""Parallel runner for the validation suite.

Per-person checks (`check_event`, `check_ltl`, `check_continuity`, plus the
per-person slice of `check_persona`) are pure and operate on one schedule
at a time, so they parallelize cleanly across a thread pool. Persona-level
checks (instance counts, occupation declarations) stay sequential since
they are cheap and need the full population in scope.

Output is byte-identical to the sequential runner: violations are merged in
input-person order to keep the report deterministic across worker counts.
"""

from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.scripts.persona.config.schema import PersonaConfig, TemporalRule
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule
from src.scripts.persona.validation.check_characteristics import (
    check_characteristic_distributions,
)
from src.scripts.persona.validation.check_contexts import check_context_distributions
from src.scripts.persona.validation.check_continuity import (
    ContinuityViolation,
    check_continuity,
)
from src.scripts.persona.validation.check_event import (
    EventViolation,
    check_event_constraints,
)
from src.scripts.persona.validation.check_ltl import LTLViolation, check_ltl_rules
from src.scripts.persona.validation.check_persona import (
    PersonaViolation,
    _check_dated_stages,
    _check_instance_counts,
    _check_occupation,
    _check_stage_cadence,
)
from src.scripts.persona.validation.report import ValidationReport


@dataclass(frozen=True, slots=True)
class _PerPersonResult:
    """Output of one parallel task: per-person violations, in stable input order."""

    persona: list[PersonaViolation]
    event: list[EventViolation]
    ltl: list[LTLViolation]
    continuity: list[ContinuityViolation]


def _resolve_workers(workers: int | None) -> int:
    if workers is None or workers <= 0:
        return os.cpu_count() or 1
    return workers


def _per_person_task(
    person: Person,
    schedule: PersonSchedule,
    catalog: Catalog,
    rules: list[TemporalRule],
    start_date: _dt.date,
) -> _PerPersonResult:
    """Run every per-person check for a single (person, schedule) pair."""
    persona_v = _check_stage_cadence(person, schedule, start_date)
    persona_v.extend(_check_dated_stages(person, schedule))
    event_v = check_event_constraints([schedule], [person], catalog)
    ltl_v = check_ltl_rules([schedule], rules, {person.person_id: person})
    continuity_v = check_continuity([schedule])
    return _PerPersonResult(
        persona=persona_v,
        event=event_v,
        ltl=ltl_v,
        continuity=continuity_v,
    )


def run_validation_parallel(
    persona_config: PersonaConfig,
    persons: Sequence[Person],
    schedules: Sequence[PersonSchedule],
    catalog: Catalog,
    rules: list[TemporalRule],
    *,
    start_date: _dt.date,
    workers: int | None = None,
) -> ValidationReport:
    """Run the validation suite in parallel and merge results in input order.

    `workers=None` or `workers <= 0` resolves to `os.cpu_count()`. With
    `workers == 1` the run is serial (no executor overhead). Persona-level
    checks (instance counts, occupation) are always serial since they need
    the full population.
    """
    if len(persons) != len(schedules):
        raise ValueError(
            f"persons ({len(persons)}) and schedules ({len(schedules)}) "
            "lengths must match"
        )

    population_persona = list(_check_instance_counts(persona_config, list(persons)))
    population_persona.extend(_check_occupation(persona_config, list(persons)))
    characteristic_v = check_characteristic_distributions(persona_config, persons)
    schedules_by_id = {s.person_id: s for s in schedules}
    context_v = check_context_distributions(persons, schedules_by_id)

    workers = _resolve_workers(workers)
    if not persons:
        return ValidationReport(
            persona=population_persona, characteristics=characteristic_v
        )

    args = [(p, s, catalog, rules, start_date) for p, s in zip(persons, schedules)]

    if workers == 1:
        results = [_per_person_task(*a) for a in args]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda a: _per_person_task(*a), args))

    persona: list[PersonaViolation] = list(population_persona)
    event: list[EventViolation] = []
    ltl: list[LTLViolation] = []
    continuity: list[ContinuityViolation] = []
    for r in results:
        persona.extend(r.persona)
        event.extend(r.event)
        ltl.extend(r.ltl)
        continuity.extend(r.continuity)

    return ValidationReport(
        persona=persona,
        event=event,
        ltl=ltl,
        continuity=continuity,
        characteristics=characteristic_v,
        contexts=context_v,
    )


__all__ = ["run_validation_parallel"]
