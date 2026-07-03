"""z3 day-scale scheduler. One Optimize instance per call, no module-level state."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Mapping, Sequence

import z3

from src.scripts.persona.config.schema import EventDefinition, TemporalRule
from src.scripts.persona.constraints.extract import (
    EventDayConstraints,
    get_event_constraints,
)
from src.scripts.persona.constraints.windows import resolve_window_starts
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap


def _episode_seed(
    person_seed: int, day_idx: int, event_name: str, episode_idx: int
) -> int:
    """Mix a person-stable, episode-specific seed for the random fallback.

    BLAKE2b hashing keeps the seed deterministic across machines and run
    counts: same person + same day + same event = same picked minute.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(int(person_seed).to_bytes(8, "big", signed=False))
    h.update(b"|")
    h.update(int(day_idx).to_bytes(4, "big", signed=False))
    h.update(b"|")
    h.update(event_name.encode("utf-8"))
    h.update(b"|")
    h.update(int(episode_idx).to_bytes(4, "big", signed=False))
    return int.from_bytes(h.digest(), "big", signed=False)


DAY_MINUTES = 1440

EventVars = dict[str, list[tuple[z3.ArithRef, z3.ArithRef]]]


# -------------------------------------------------------------------------------------
# ----------------------------------- public entrypoint --------------------------------
# -------------------------------------------------------------------------------------


def build_day_model(
    *,
    day_idx: int,
    total_days: int,
    events: Mapping[str, EventDefinition],
    counts: Mapping[str, int],
    window_map: WindowMap,
    occupied_ranges: Iterable[tuple[int, int]] = (),
    step_minutes: int = 10,
    free_minutes_minimum: int = 0,
    cross_midnight_events: frozenset[str] = frozenset({"sleep"}),
    maximize_event: str | None = "sleep",
    preferred_starts: Mapping[str, int] | None = None,
    preferred_durations: Mapping[str, int] | None = None,
    preferred_windows: Mapping[str, str] | None = None,
    person_seed: int = 0,
    ltl_rules: Sequence[TemporalRule] | None = None,
    person: Person | None = None,
) -> tuple[z3.Optimize | z3.Solver, EventVars]:
    """Build a z3 solver instance for one persona-day.

    `counts` gives the number of episodes to schedule per event type. Events in
    `cross_midnight_events` are allowed to extend past minute 1440; everything else
    must end before midnight. `maximize_event` adds a soft objective that maximizes the
    total duration of that event type (useful for sleep) and may be set to None to disable.
    `preferred_starts` and `preferred_durations` pin the first episode of an
    event type to a persona-stated minute or minute-count, so two persons
    sampled from the same template who jittered to different times solve to
    different schedules. A preference outside the catalog's allowed window or
    duration range is silently ignored, so the solver still finds a schedule.
    `preferred_windows` restricts an event's `allowed_starts` to a single
    persona-stated window token (e.g. `morning`); without this, an event
    whose catalog has no `temporal_patterns` would otherwise be free to
    land anywhere on the day.
    For events with no persona-stated start, the solver picks one
    deterministically from `person_seed` so per-person variation still
    surfaces; pass `person_seed=0` to keep the legacy deterministic behavior.
    """
    day_of_week = day_idx % 7
    ctx = z3.Context()
    # Use the cheaper `Solver` when no optimization objective is set.
    # `Optimize` carries pseudo-boolean / weighted-MaxSAT machinery that
    # is unnecessary for pure satisfiability and adds significant per-day
    # overhead once events stack up (e.g. ~30 events/day on a multi-episode
    # spec like smoking 0-22/day). The two interfaces are
    # `add` / `check` / `model` compatible; `maximize` is the only call
    # that requires `Optimize`.
    solver: z3.Optimize | z3.Solver
    if maximize_event is None:
        solver = z3.Solver(ctx=ctx)
    else:
        solver = z3.Optimize(ctx=ctx)

    starts = dict(preferred_starts) if preferred_starts else {}
    durations_map = dict(preferred_durations) if preferred_durations else {}
    windows_map = dict(preferred_windows) if preferred_windows else {}
    occupied_list: list[tuple[int, int]] = list(occupied_ranges)
    # Sequential accumulator: every event placed in this loop adds its
    # interval here so the next event's pin / random pick can avoid it.
    occupied_so_far: list[tuple[int, int]] = list(occupied_list)

    event_vars: EventVars = {}
    flat: list[tuple[str, z3.ArithRef, z3.ArithRef]] = []

    # Placement order: events with a persona-pinned HH:MM start come first
    # in catalog (chronological-friendly) order so their fixed start
    # populates `occupied_so_far` early. The remaining events - those with
    # only a window pin or no pin at all - are placed in descending order
    # of "footprint" (= max of pinned duration and `per_event_max`). This
    # matters when several non-pinned events compete for the same window:
    # if a small one is placed first, its random pick or solver choice can
    # land in a slot that leaves the bigger one with no feasible start.
    # Placing the biggest competitor first lets it claim the slot it needs
    # while smaller events still have room to maneuver around it.
    for event_name in events:
        event_vars[event_name] = []

    ordered_names = _order_event_placement(
        events,
        starts=starts,
        durations_map=durations_map,
    )

    for event_name in ordered_names:
        event_def = events[event_name]
        constraints = get_event_constraints(
            event_def,
            day_idx=day_idx,
            total_days=total_days,
            window_map=window_map,
            day_of_week=day_of_week,
        )
        num_events = counts.get(event_name, 0)
        if num_events == 0:
            event_vars[event_name] = []
            continue

        crosses_midnight = event_name in cross_midnight_events
        per_event, placed_interval = _add_event_episodes(
            solver=solver,
            ctx=ctx,
            event_name=event_name,
            num_events=num_events,
            day_idx=day_idx,
            constraints=constraints,
            window_map=window_map,
            step_minutes=step_minutes,
            preferred_start=starts.get(event_name),
            preferred_duration=durations_map.get(event_name),
            preferred_window=windows_map.get(event_name),
            person_seed=person_seed,
            occupied_ranges=occupied_so_far,
            crosses_midnight=crosses_midnight,
        )
        event_vars[event_name] = per_event
        for s, d in per_event:
            flat.append((event_name, s, d))
        if placed_interval is not None:
            occupied_so_far.append(placed_interval)

    _add_day_bounds(solver, flat, cross_midnight_events)
    overlap_allowed_pairs = _ltl_overlap_allowed_pairs(ltl_rules, person)
    overlap_allowed_pairs |= _concurrent_with_pairs(events)
    _add_pairwise_non_overlap(solver, flat, overlap_allowed_pairs=overlap_allowed_pairs)
    _add_spillover_non_overlap(solver, flat, occupied_list)
    _add_free_time_floor(solver, flat, occupied_list, free_minutes_minimum, ctx=ctx)
    if ltl_rules:
        _add_ltl_constraints(solver, event_vars, ltl_rules, person=person)
    _add_objective(solver, event_vars, maximize_event)

    return solver, event_vars


# -------------------------------------------------------------------------------------
# ------------------------------- per-event sub-builders ------------------------------
# -------------------------------------------------------------------------------------


def _per_event_max_minutes(spec: EventDefinition) -> int:
    """Return `per_event_duration.max` in minutes regardless of the YAML unit."""
    if spec.per_event_duration.unit == "hours":
        return spec.per_event_duration.max * 60
    return spec.per_event_duration.max


def _order_event_placement(
    events: Mapping[str, EventDefinition],
    *,
    starts: Mapping[str, int],
    durations_map: Mapping[str, int],
) -> list[str]:
    """Return event names in placement order: pinned-start first, then largest first.

    Events whose start is persona-pinned to an HH:MM keep catalog
    insertion order (group 0). Window-only and unconstrained events go in
    group 1, sorted by descending "footprint" (max of pinned duration
    and `per_event_duration.max`) so the longest competitor for a shared
    window claims its slot before smaller ones can block it. Ties keep
    catalog insertion order.
    """
    decorated: list[tuple[int, int, int, str]] = []
    for idx, name in enumerate(events):
        spec = events[name]
        per_max = _per_event_max_minutes(spec)
        pinned_dur = durations_map.get(name) or 0
        footprint = max(per_max, pinned_dur)
        group = 0 if name in starts else 1
        decorated.append((group, -footprint, idx, name))
    decorated.sort()
    return [t[3] for t in decorated]


def _start_fits_allowed_windows(start: int, ranges: list[tuple[int, int]]) -> bool:
    """An empty `ranges` list means 'any minute is allowed'."""
    if not ranges:
        return True
    return any(low <= start < high for low, high in ranges)


def _quantize_duration(
    duration: int, *, low: int, high: int, step_minutes: int
) -> int | None:
    """Snap a persona-stated duration to a step-aligned minute count.

    Returns None when the duration cannot fit into `[low, high]` after
    snapping, so the caller can fall back to the unconstrained range.
    """
    snapped = (duration // step_minutes) * step_minutes
    if snapped < step_minutes:
        snapped = step_minutes
    if snapped < low or snapped > high:
        return None
    return snapped


def _start_clear_of_occupied(
    start: int,
    *,
    duration_min: int,
    occupied_ranges: list[tuple[int, int]],
) -> bool:
    """Whether a start at `start` can host at least `duration_min` minutes
    without overlapping any occupied range. Used to decide whether a
    persona-pinned or random-picked start is feasible before the solver
    sees it; if not, the caller falls back to the disjunction so the
    solver can choose a feasible minute itself.
    """
    if not occupied_ranges:
        return True
    for occ_start, occ_end in occupied_ranges:
        if start >= occ_end:
            continue
        if start + duration_min <= occ_start:
            continue
        return False
    return True


def _start_fits_in_day(
    start: int, duration_min: int, *, crosses_midnight: bool
) -> bool:
    """Reject pins that would force the event past midnight on a day where
    cross-midnight is not allowed."""
    if crosses_midnight:
        return True
    return start + duration_min <= DAY_MINUTES


def _add_event_episodes(
    *,
    solver: z3.Optimize | z3.Solver,
    ctx: z3.Context,
    event_name: str,
    num_events: int,
    day_idx: int,
    constraints: EventDayConstraints,
    window_map: WindowMap,
    step_minutes: int,
    preferred_start: int | None = None,
    preferred_duration: int | None = None,
    preferred_window: str | None = None,
    person_seed: int = 0,
    occupied_ranges: list[tuple[int, int]] | None = None,
    crosses_midnight: bool = False,
) -> tuple[list[tuple[z3.ArithRef, z3.ArithRef]], tuple[int, int] | None]:
    """Return the z3 vars for `num_events` episodes plus the [start, end)
    interval the first episode was pinned to (or None if it was left as
    the disjunction over allowed_starts).
    """
    occupied_list: list[tuple[int, int]] = list(occupied_ranges or ())

    # Apply the persona-stated window: prefer the persona's window when
    # the catalog allows it, fall back to the catalog's allowed_windows
    # otherwise. The catalog stays authoritative when both disagree, so a
    # mistyped or out-of-scope persona token does not silently break the
    # event.
    effective_windows: tuple[str, ...] = constraints.allowed_windows
    if (
        preferred_window is not None
        and preferred_window in window_map
        and (
            not constraints.allowed_windows
            or preferred_window in constraints.allowed_windows
        )
    ):
        effective_windows = (preferred_window,)

    if effective_windows:
        allowed_starts = sorted(
            resolve_window_starts(effective_windows, window_map, step_minutes)
        )
    else:
        allowed_starts = list(range(0, DAY_MINUTES, step_minutes))

    pinned_duration: int | None = None
    if preferred_duration is not None:
        pinned_duration = _quantize_duration(
            preferred_duration,
            low=constraints.per_event_min,
            high=constraints.per_event_max,
            step_minutes=step_minutes,
        )

    # If pinning episode 0 to `pinned_duration` would force the
    # `total_event_duration` constraint to be unsatisfiable (typical on
    # heavy multi-episode days where seasonality + trend push num_events
    # high while the persona's per-episode duration stays long, e.g.
    # weekend walking 8x/day at 60min vs the catalog cap of 240min/day),
    # drop the pin so episode 0's duration becomes a free variable in
    # [per_event_min, per_event_max] and z3 can pick a feasible
    # combination. Persona intent (per-episode duration) is best-effort,
    # not a hard floor when it collides with catalog totals.
    if pinned_duration is not None and num_events > 0:
        min_total_with_pin = (
            pinned_duration + (num_events - 1) * constraints.per_event_min
        )
        max_total_with_pin = (
            pinned_duration + (num_events - 1) * constraints.per_event_max
        )
        if (
            min_total_with_pin > constraints.total_max
            or max_total_with_pin < constraints.total_min
        ):
            pinned_duration = None

    # Two safety clearances, with different roles:
    #
    # `safety_others` is the footprint we have to leave around OTHER
    # events' occupied ranges. When `pinned_duration` is set, the solver
    # commits the first episode to that exact duration, so reserving
    # more is overcommitting room (and on tight days squeezes random
    # picks into corners that block trailing episodes). When there is
    # no pin, the solver picks any duration in [per_event_min,
    # per_event_max], so we have to assume worst-case `per_event_max`.
    #
    # `safety_self` is the spacing reserved between successive episodes
    # of THIS event during the random-pick fallback. Subsequent episodes
    # have their durations decided by z3, so we only need to leave room
    # for the smallest valid duration: the solver can always shrink
    # `actual_duration` to `per_event_min` to satisfy pairwise
    # non-overlap. Using `per_event_max` here would overcommit when
    # several multi-episode events share a tight window (e.g.
    # evening-pinned walking + stress).
    safety_others = (
        pinned_duration if pinned_duration is not None else constraints.per_event_max
    )
    safety_self = (
        pinned_duration if pinned_duration is not None else constraints.per_event_min
    )

    def _start_safe_for(s: int, duration_min: int) -> bool:
        if not _start_clear_of_occupied(
            s,
            duration_min=duration_min,
            occupied_ranges=occupied_list,
        ):
            return False
        return _start_fits_in_day(s, duration_min, crosses_midnight=crosses_midnight)

    pinned_start: int | None = None
    if preferred_start is not None:
        effective_window_ranges = [
            window_map.get(w) for w in effective_windows if w in window_map
        ]
        if _start_fits_allowed_windows(
            preferred_start, effective_window_ranges
        ) and _start_safe_for(preferred_start, safety_others):
            pinned_start = preferred_start

    safe_random_starts = [
        s for s in allowed_starts if _start_safe_for(s, safety_others)
    ]

    placed_first_start: int | None = None
    # Same-event prior-episode intervals. A multi-episode random fallback
    # has to avoid its own earlier pins as well as other events' pins,
    # otherwise two episodes could land on the same minute and the
    # pairwise non-overlap constraint would make the whole day unsat.
    prior_episode_intervals: list[tuple[int, int]] = []

    per_event: list[tuple[z3.ArithRef, z3.ArithRef]] = []
    for i in range(num_events):
        start = z3.Int(f"{event_name}_start_{i}_day{day_idx}", ctx=ctx)
        duration = z3.Int(f"{event_name}_duration_{i}_day{day_idx}", ctx=ctx)
        per_event.append((start, duration))

        chosen: int | None = None
        if i == 0 and pinned_start is not None:
            chosen = pinned_start
        elif i == 0 and person_seed and safe_random_starts:
            # Pin episode 0 to a per-person random minute so two persons
            # sampled from the same template diverge. Episodes 1..K are
            # left as a disjunction over `allowed_starts` so z3 can pack
            # them around the pin (and around other events sharing this
            # window) without the random pick over-reserving space - the
            # legacy "random-pick every episode" approach would lock in
            # tight pins that interact badly when several multi-episode
            # events share a window (e.g. evening-pinned walking + stress).
            rng = random.Random(_episode_seed(person_seed, day_idx, event_name, i))
            chosen = rng.choice(safe_random_starts)

        if chosen is not None:
            # `chosen` is only set on episode 0 (HH:MM pin or random pin);
            # subsequent episodes always fall through to the disjunction
            # below so z3 can pack them around this anchor.
            solver.add(start == chosen)
            placed_first_start = chosen
            prior_episode_intervals.append((chosen, chosen + safety_self))
        else:
            solver.add(z3.Or([start == t for t in allowed_starts]))

        if i == 0 and pinned_duration is not None:
            solver.add(duration == pinned_duration)
        else:
            solver.add(duration >= constraints.per_event_min)
            solver.add(duration <= constraints.per_event_max)
            solver.add(duration > 0)
            solver.add(duration % step_minutes == 0)

    durations = [d for _, d in per_event]
    solver.add(z3.Sum(durations) >= constraints.total_min)
    solver.add(z3.Sum(durations) <= constraints.total_max)

    # Window-fraction floors come from catalog seasonality patterns
    # (e.g. "boost morning placements"). When the persona has restricted
    # this event to a different window via `preferred_window`, every
    # episode start is already pinned inside that window; a fraction
    # floor for any other window would be unsatisfiable. The persona is
    # authoritative on placement, so we keep only the floors whose window
    # is actually reachable. When `effective_windows` is empty (no
    # restriction at all), every floor stays in scope.
    effective_windows_set = set(effective_windows)
    for window_name, min_frac in constraints.window_fraction_constraints:
        if window_name not in window_map:  # pragma: no cover  (extractor pre-filters)
            continue
        if effective_windows_set and window_name not in effective_windows_set:
            continue
        win_start, win_end = window_map.get(window_name)
        in_window = [z3.And(s >= win_start, s < win_end) for s, _ in per_event]
        solver.add(
            z3.Sum([z3.If(c, 1, 0) for c in in_window]) >= int(min_frac * num_events)
        )

    placed_interval: tuple[int, int] | None = None
    if placed_first_start is not None:
        d_estimate = (
            pinned_duration
            if pinned_duration is not None
            else constraints.per_event_min
        )
        placed_interval = (placed_first_start, placed_first_start + d_estimate)

    return per_event, placed_interval


# -------------------------------------------------------------------------------------
# ---------------------------------- day-wide rules -----------------------------------
# -------------------------------------------------------------------------------------


def _add_day_bounds(
    solver: z3.Optimize | z3.Solver,
    flat: list[tuple[str, z3.ArithRef, z3.ArithRef]],
    cross_midnight_events: frozenset[str],
) -> None:
    for name, s, d in flat:
        solver.add(s >= 0)
        solver.add(s < DAY_MINUTES)
        solver.add(d > 0)
        if name not in cross_midnight_events:
            solver.add(s + d <= DAY_MINUTES)


def _add_pairwise_non_overlap(
    solver: z3.Optimize | z3.Solver,
    flat: list[tuple[str, z3.ArithRef, z3.ArithRef]],
    overlap_allowed_pairs: set[frozenset[str]] | None = None,
) -> None:
    """Enforce pairwise non-overlap; pairs in `overlap_allowed_pairs` skip the check."""
    skip = overlap_allowed_pairs or set()
    for i in range(len(flat)):
        name1, s1, d1 = flat[i]
        for j in range(i + 1, len(flat)):
            name2, s2, d2 = flat[j]
            if name1 != name2 and frozenset({name1, name2}) in skip:
                continue
            solver.add(z3.Or(s1 + d1 <= s2, s2 + d2 <= s1))


def _add_spillover_non_overlap(
    solver: z3.Optimize | z3.Solver,
    flat: list[tuple[str, z3.ArithRef, z3.ArithRef]],
    occupied_ranges: Iterable[tuple[int, int]],
) -> None:
    for occ_start, occ_end in occupied_ranges:
        for _, s, d in flat:
            solver.add(z3.Or(s + d <= occ_start, s >= occ_end))


def _add_free_time_floor(
    solver: z3.Optimize | z3.Solver,
    flat: list[tuple[str, z3.ArithRef, z3.ArithRef]],
    occupied_ranges: Iterable[tuple[int, int]],
    free_minutes_minimum: int,
    *,
    ctx: z3.Context,
) -> None:
    if free_minutes_minimum <= 0:
        return
    occupied_minutes = sum(max(0, e - s) for s, e in occupied_ranges)
    max_scheduled = DAY_MINUTES - free_minutes_minimum
    if occupied_minutes > max_scheduled:
        solver.add(z3.BoolVal(False, ctx=ctx))
        return
    if not flat:
        return
    durations = [d for _, _, d in flat]
    solver.add(z3.Sum(durations) + occupied_minutes <= max_scheduled)


def _add_objective(
    solver: z3.Optimize | z3.Solver,
    event_vars: EventVars,
    maximize_event: str | None,
) -> None:
    if maximize_event is None:
        return
    target = event_vars.get(maximize_event)
    if not target:
        return
    solver.maximize(z3.Sum([d for _, d in target]))


# -------------------------------------------------------------------------------------
# ------------------------------------- LTL bridge ------------------------------------
# -------------------------------------------------------------------------------------


_LTL_NO_OVERLAP_RE = re.compile(r"^\s*G\s+¬\(\s*([^\s∧]+)\s*∧\s*([^\s)]+)\s*\)\s*$")
_LTL_MIN_OVERLAP_RE = re.compile(
    r"^\s*G\s+\(\s*([^\s→]+)\s*→\s*F\s*\(\s*([^\s∧]+)\s*∧\s*([^\s)]+)\s*\)\s*\)\s*$"
)
_LTL_EVENTUAL_CO_OCCUR_RE = re.compile(
    r"^\s*F\s*\(\s*([^\s∧]+)\s*∧\s*([^\s)]+)\s*\)\s*$"
)
_LTL_IMPLIES_FUTURE_RE = re.compile(
    r"^\s*G\s+\(\s*([^\s→]+)\s*→\s*F\s+([^\s)]+)\s*\)\s*$"
)
_CONTEXT_PREFIX = "context:"


def _parse_event_ltl(formula: str) -> tuple[str, str, str] | None:
    """Match `formula` to one of the supported LTL shapes; returns (kind, a, b)."""
    text = formula.strip()
    m = _LTL_NO_OVERLAP_RE.match(text)
    if m:
        return ("no_overlap", m.group(1), m.group(2))
    m = _LTL_MIN_OVERLAP_RE.match(text)
    if m and m.group(1) == m.group(2):
        return ("min_overlap", m.group(1), m.group(3))
    m = _LTL_EVENTUAL_CO_OCCUR_RE.match(text)
    if m:
        return ("eventual_co_occur", m.group(1), m.group(2))
    m = _LTL_IMPLIES_FUTURE_RE.match(text)
    if m:
        return ("implies_future", m.group(1), m.group(2))
    return None


def _ltl_rule_applies_to(person: Person | None, applies_to: dict) -> bool:
    """Return True when `person` matches the rule's `applies_to` filter."""
    if not applies_to:
        return True
    if person is None:
        return False
    for key, allowed in applies_to.items():
        if key == "personas":
            if person.persona_id not in allowed:
                return False
            continue
        value = person.characteristics.get(key)
        if value is None:
            return False
        lowered_allowed = {str(v).lower() for v in allowed}
        if value not in allowed and str(value).lower() not in lowered_allowed:
            return False
    return True


def _ltl_overlap_allowed_pairs(
    rules: Sequence[TemporalRule] | None,
    person: Person | None,
) -> set[frozenset[str]]:
    """Collect atom-pair names that an overlap-requiring LTL rule wants to share time."""
    pairs: set[frozenset[str]] = set()
    if not rules:
        return pairs
    for rule in rules:
        if not _ltl_rule_applies_to(person, rule.applies_to):
            continue
        parsed = _parse_event_ltl(rule.formula)
        if parsed is None:
            continue
        kind, a_name, b_name = parsed
        if kind not in ("min_overlap", "eventual_co_occur"):
            continue
        pairs.add(frozenset({a_name, b_name}))
    return pairs


def _concurrent_with_pairs(
    events: Mapping[str, EventDefinition],
) -> set[frozenset[str]]:
    """Event pairs exempt from non-overlap via the `concurrent_with` whitelist.

    Permission to overlap is config data, not an LTL rule: LTL can only
    forbid (`G ¬(A∧B)`) or require (`F(A∧B)`), never merely allow. A pair
    is exempt when either side names the other, mirroring the evaluator's
    `concurrent_with` handling so base generation and scoring agree.
    `is_concurrent` is intentionally not consulted here.
    """
    pairs: set[frozenset[str]] = set()
    for name, defn in events.items():
        for ref in defn.concurrent_with:
            if ref != name:
                pairs.add(frozenset({name, ref}))
    return pairs


def _add_ltl_constraints(
    solver: z3.Optimize | z3.Solver,
    atom_vars: dict,
    rules: Sequence[TemporalRule],
    *,
    person: Person | None = None,
) -> None:
    """Translate LTL rules into Z3 constraints over `atom_vars`.

    `atom_vars` maps an atom name (an event label or `context:<member>`)
    to a list of `(start, duration)` tuples; either side of a tuple may
    be a Z3 expression or a Python int, so callers can mix already-solved
    constants with fresh decision variables in the same model.
    """
    for rule in rules:
        if not _ltl_rule_applies_to(person, rule.applies_to):
            continue
        parsed = _parse_event_ltl(rule.formula)
        if parsed is None:
            continue
        kind, a_name, b_name = parsed
        a_eps = atom_vars.get(a_name) or []
        b_eps = atom_vars.get(b_name) or []
        if not a_eps or not b_eps:
            continue
        if kind == "no_overlap":
            for s1, d1 in a_eps:
                for s2, d2 in b_eps:
                    solver.add(z3.Or(s1 + d1 <= s2, s2 + d2 <= s1))
        elif kind == "implies_future":
            for s1, d1 in a_eps:
                solver.add(z3.Or([s2 >= s1 + d1 for s2, _ in b_eps]))
        elif kind == "min_overlap":
            for s1, d1 in a_eps:
                overlaps = [z3.And(s1 < s2 + d2, s1 + d1 > s2) for s2, d2 in b_eps]
                solver.add(z3.Or(overlaps))
        elif kind == "eventual_co_occur":  # pragma: no branch - parser produces 4 kinds
            # Per-day enforcement: when both fire today, at least one
            # pair must overlap. Days where one side does not fire are
            # silently skipped (the outer `not a_eps or not b_eps` guard).
            pair_overlaps = [
                z3.And(s1 < s2 + d2, s1 + d1 > s2)
                for s1, d1 in a_eps
                for s2, d2 in b_eps
            ]
            solver.add(z3.Or(pair_overlaps))
