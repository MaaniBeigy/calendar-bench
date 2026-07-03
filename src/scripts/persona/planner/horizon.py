"""Single-person horizon orchestrator: allocate, solve each day, propagate spillover."""

from __future__ import annotations

import datetime as _dt

import z3

from src.scripts.persona.config.defaults import WEEKDAY_ORDER
from src.scripts.persona.config.schema import EnvironmentConfig, TemporalRule
from src.scripts.persona.domain.event import (
    Catalog,
    EventInstance,
    apply_event_overrides,
)
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule, Spillover
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.planner.allocator import allocate_horizon
from src.scripts.persona.solver.day_model import build_day_model
from src.scripts.persona.solver.preferences import build_day_preferences
from src.scripts.persona.solver.spillover import (
    extract_spillovers,
    occupied_from_spillovers,
)
from src.scripts.persona.solver.year_model import (
    anchors_for_day,
    occupied_from_anchors,
    plan_yearly_anchors,
)

CROSS_MIDNIGHT_DEFAULT: frozenset[str] = frozenset({"sleep"})


def plan_horizon(
    person: Person,
    catalog: Catalog,
    environment: EnvironmentConfig,
    *,
    window_map: WindowMap | None = None,
    ltl_rules: list[TemporalRule] | None = None,
) -> PersonSchedule:
    """Run the full per-person pipeline across the configured horizon."""
    if window_map is None:
        window_map = WindowMap.from_config(environment.time_windows)

    horizon_days = environment.horizon.weeks * 7
    start_date = environment.horizon.start_date

    # Per-person effective catalog: every persona-stated override on an
    # event name in the global catalog replaces the matching field. Other
    # events are passed through unchanged.
    effective_catalog = apply_event_overrides(catalog, person.event_overrides)

    counts_per_day = allocate_horizon(
        person,
        effective_catalog,
        horizon_days=horizon_days,
        start_date=start_date,
        window_map=window_map,
    )

    yearly_anchors = (
        plan_yearly_anchors(
            person,
            start_date=start_date,
            horizon_days=horizon_days,
            window_map=window_map,
        )
        if environment.horizon.enable_yearly_pass
        else []
    )

    days: list[DaySchedule] = []
    prev_spillovers: list[Spillover] = []

    maximize = (
        "sleep" if environment.solver.optimize_objective == "maximize_sleep" else None
    )

    for day_idx in range(horizon_days):
        weekday = WEEKDAY_ORDER[day_idx % 7]
        date = start_date + _dt.timedelta(days=day_idx)
        counts = counts_per_day[day_idx]
        # Anchors fall on a fixed date and may collide with persona-stated
        # event names (e.g. dentist) that the catalog also defines. Suppress
        # the catalog version on the anchor day so the solver does not double
        # book the slot.
        anchors_today = anchors_for_day(yearly_anchors, day_idx)
        if anchors_today:
            counts = dict(counts)
            for anchor in anchors_today:
                if anchor.name in counts:
                    counts[anchor.name] = 0
        events_for_day = {name: effective_catalog.get(name) for name in counts}
        occupied = occupied_from_spillovers(prev_spillovers) + occupied_from_anchors(
            anchors_today
        )
        prefs = build_day_preferences(person, weekday=weekday, date=date)

        day_events, new_spillovers = _solve_one_day(
            day_idx=day_idx,
            horizon_days=horizon_days,
            events=events_for_day,
            counts=counts,
            window_map=window_map,
            occupied=occupied,
            step_minutes=environment.solver.step_minutes,
            maximize=maximize,
            preferred_starts=prefs.starts,
            preferred_durations=prefs.durations,
            preferred_windows=prefs.windows,
            person_seed=person.person_seed,
            ltl_rules=ltl_rules,
            person=person,
        )
        for anchor in anchors_today:
            day_events.setdefault(anchor.name, []).append(
                EventInstance(
                    event_name=anchor.name,
                    start=anchor.start,
                    duration=anchor.duration,
                )
            )

        days.append(
            DaySchedule(
                day_index=day_idx,
                date=date,
                weekday=weekday,
                events=day_events,
                spillovers=list(prev_spillovers),
            )
        )
        prev_spillovers = new_spillovers

    return PersonSchedule(
        person_id=person.person_id,
        persona_id=person.persona_id,
        person_seed=person.person_seed,
        days=days,
    )


def _solve_one_day(
    *,
    day_idx: int,
    horizon_days: int,
    events: dict,
    counts: dict[str, int],
    window_map: WindowMap,
    occupied: list[tuple[int, int]],
    step_minutes: int,
    maximize: str | None,
    preferred_starts: dict[str, int] | None = None,
    preferred_durations: dict[str, int] | None = None,
    preferred_windows: dict[str, str] | None = None,
    person_seed: int = 0,
    ltl_rules: list[TemporalRule] | None = None,
    person: Person | None = None,
) -> tuple[dict[str, list[EventInstance]], list[Spillover]]:
    if not events:
        return {}, []

    def _build(
        windows_for_attempt: dict[str, str] | None,
        seed_for_attempt: int,
        rules_for_attempt: list[TemporalRule] | None,
    ):
        return build_day_model(
            day_idx=day_idx,
            total_days=horizon_days,
            events=events,
            counts=counts,
            window_map=window_map,
            occupied_ranges=occupied,
            step_minutes=step_minutes,
            cross_midnight_events=CROSS_MIDNIGHT_DEFAULT,
            maximize_event=maximize,
            preferred_starts=preferred_starts,
            preferred_durations=preferred_durations,
            preferred_windows=windows_for_attempt,
            person_seed=seed_for_attempt,
            ltl_rules=rules_for_attempt,
            person=person,
        )

    # Retry-relax ladder: each row tightens or loosens what we ask Z3
    # to satisfy. Cadence wins, so we stop at the first sat attempt.
    #   1. windows + random pin + LTL
    #   2. windows + no random pin + LTL
    #   3. no windows + no random pin + LTL
    #   4. no windows + no random pin + no LTL (last-ditch)
    attempts = [
        (preferred_windows, person_seed, ltl_rules),
        (preferred_windows, 0, ltl_rules),
        (None, 0, ltl_rules),
        (None, 0, None),
    ]
    solver = None
    event_vars = None
    for windows_for_attempt, seed_for_attempt, rules_for_attempt in attempts:
        solver, event_vars = _build(
            windows_for_attempt, seed_for_attempt, rules_for_attempt
        )
        if solver.check() == z3.sat:
            break
    else:
        return {}, []

    model = solver.model()
    day_events: dict[str, list[EventInstance]] = {}
    for name, vars_for_type in event_vars.items():
        if not vars_for_type:
            continue
        day_events[name] = [
            EventInstance(
                event_name=name,
                start=model[s].as_long(),
                duration=model[d].as_long(),
            )
            for s, d in vars_for_type
        ]
    new_spillovers = extract_spillovers(model, event_vars)
    return day_events, new_spillovers
