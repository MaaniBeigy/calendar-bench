"""Per-day event-count allocation driven by persona statements."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable

from src.scripts.persona.config.defaults import WEEKDAY_ORDER
from src.scripts.persona.config.schema import EventDefinition
from src.scripts.persona.constraints.extract import get_event_constraints
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.sampling.gating import matches_requires

# -------------------------------------------------------------------------------------
# ------------------------------- per-event count rules -------------------------------
# -------------------------------------------------------------------------------------


def _persona_count(
    person: Person,
    event_def: EventDefinition,
    weekday: str,
    date: _dt.date,
) -> int:
    """Whether the persona staged `event_def` on this calendar day.

    The catalog gates (`requires`, `weekdays`) come first - they apply
    to every person regardless of staging. After that, the persona
    must have a stage entry whose `name` matches the event AND whose
    calendar trigger fires today (`date` matches, or `weekday` is in
    `days`). Returns 1 when at least one matching stage fires, else 0.

    The schema is event-name-agnostic: there is no special-case
    handling for `sleep`, `lunch`, `office_work`, `running`, or any
    other name. A persona that wants any of those events declares an
    appropriate stage entry; a persona that does not stays at zero.
    """
    if not matches_requires(person, event_def.requires):
        return 0

    # Catalog-level weekday restriction: an event listing `weekdays:`
    # only fires on those weekdays for everyone. Mirrors the validator's
    # `_check_weekday` rule so the allocator never produces a schedule
    # the validator would reject.
    if event_def.weekdays is not None and weekday not in event_def.weekdays:
        return 0

    name = event_def.name
    for stage in person.stages:
        if stage.name != name:
            continue
        if stage.date is not None and stage.date == date:
            return 1
        if weekday in stage.days:
            return 1
    return 0


# -------------------------------------------------------------------------------------
# ------------------------------------- public API ------------------------------------
# -------------------------------------------------------------------------------------


def allocate_day_counts(
    person: Person,
    catalog: Catalog,
    *,
    day_idx: int,
    total_days: int,
    date: _dt.date,
    window_map: WindowMap,
) -> dict[str, int]:
    """Per-event episode counts on one day for one person."""
    day_of_week = day_idx % 7
    weekday = WEEKDAY_ORDER[day_of_week]
    counts: dict[str, int] = {}
    for event_name, event_def in catalog.events_by_name.items():
        persona_count = _persona_count(person, event_def, weekday, date)
        if persona_count == 0:
            counts[event_name] = 0
            continue
        constraints = get_event_constraints(
            event_def,
            day_idx=day_idx,
            total_days=total_days,
            window_map=window_map,
            day_of_week=day_of_week,
        )
        if constraints.event_disabled:
            counts[event_name] = 0
            continue
        # The catalog's `total_event_episodes` plus any seasonality / trend
        # patterns drive the per-day count - matches the legacy
        # `compute_event_counts.py` behavior where catalog `base_count`
        # is authoritative. Persona staging only acts as a "this person
        # opted-in" floor: events the persona staged on this day fire at
        # least once even if the catalog `base_count` is 0 (e.g. an
        # opportunistic event with `{min:0, max:1}` would otherwise round
        # to 0 in the absence of seasonality).
        counts[event_name] = max(persona_count, constraints.base_count)
    return counts


def allocate_horizon(
    person: Person,
    catalog: Catalog,
    *,
    horizon_days: int,
    start_date: _dt.date,
    window_map: WindowMap,
) -> list[dict[str, int]]:
    """Per-day episode counts across the run horizon."""
    if horizon_days < 0:
        raise ValueError(f"horizon_days must be >= 0, got {horizon_days}")
    counts_per_day: list[dict[str, int]] = []
    for day_idx in range(horizon_days):
        date = start_date + _dt.timedelta(days=day_idx)
        counts_per_day.append(
            allocate_day_counts(
                person,
                catalog,
                day_idx=day_idx,
                total_days=horizon_days,
                date=date,
                window_map=window_map,
            )
        )
    return counts_per_day


def days_with_event(
    counts_per_day: Iterable[dict[str, int]], event_name: str
) -> list[int]:
    """Return the day indices on which `event_name` has a non-zero count."""
    return [i for i, day in enumerate(counts_per_day) if day.get(event_name, 0) > 0]


__all__ = [
    "allocate_day_counts",
    "allocate_horizon",
    "days_with_event",
]
