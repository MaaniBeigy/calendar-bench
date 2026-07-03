"""Greedy second-pass placer for context episodes.

Runs after the event Z3 solver. For each enabled `(category, member)`
pair the placer samples an episode count, draws a duration per
episode, and finds a non-overlapping slot inside any `mode: fix`
windows declared on the member. Already-placed events on the same day
are treated as fixed obstacles; mutual exclusion forbids overlap
inside any category whose `mutually_exclusive` flag is `True`, with
the same rule applied across same-`dimension` members.

The placer is deterministic: same `(person_seed, day_index)` plus the
same persona config produce the same episodes.
"""

from __future__ import annotations

import datetime as _dt
import random
from collections.abc import Sequence
from dataclasses import dataclass

import z3

from src.scripts.persona.config.schema import (
    ContextMember,
    TemporalPattern,
    TemporalRule,
)
from src.scripts.persona.context.resolver import ResolvedCategory
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.domain.time_windows import WindowMap

WEEKDAY_TOKENS: dict[int, str] = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


@dataclass(frozen=True, slots=True)
class _Interval:
    """An ordered minute interval used inside the placer only."""

    start: int
    end: int


def _per_event_minutes(member: ContextMember) -> tuple[int, int]:
    """Resolve the per-event duration band to minutes."""
    band = member.per_event_duration
    factor = 60 if band.unit == "hours" else 1
    return int(band.min * factor), int(band.max * factor)


def _episodes_for_day(
    member: ContextMember,
    rng: random.Random,
) -> int:
    """Sample an episode count for one day."""
    band = member.total_event_episodes
    if band.scale != "day":
        # Weekly / monthly scales are flattened to a per-day average so
        # the greedy placer stays per-day. The validator catches scale
        # mismatches later.
        return rng.randint(band.min, band.max) // 7 if band.max else 0
    return rng.randint(band.min, band.max)


def _fix_windows(
    member: ContextMember,
    weekday: str,
    window_map: WindowMap,
) -> tuple[bool, list[_Interval]]:
    """Resolve `mode: fix` patterns to a `(enabled_today, windows)` pair.

    A pattern naming weekday tokens enables the member only on those
    weekdays. Window tokens (`morning`, `night`, ...) restrict placement
    to those minute ranges. A pattern with both restricts on both axes.
    Returns `(False, [])` when any `fix` pattern excludes today's
    weekday.
    """
    weekday_set = set(WEEKDAY_TOKENS.values())
    windows: list[_Interval] = []
    for pattern in member.temporal_patterns:
        if pattern.mode != "fix":
            continue
        tokens = _within_tokens(pattern)
        weekday_tokens = [t for t in tokens if t in weekday_set]
        window_tokens = [t for t in tokens if t not in weekday_set]
        if weekday_tokens and weekday not in weekday_tokens:
            return False, []
        for name in window_tokens:
            if name in window_map:
                start, end = window_map.get(name)
                windows.append(_Interval(start, end))
    return True, windows


def _within_tokens(pattern: TemporalPattern) -> list[str]:
    within = pattern.details.get("within")
    if isinstance(within, str):
        return [within]
    if isinstance(within, list):
        return [str(v) for v in within]
    return []


def _free_slot(
    duration: int,
    obstacles: list[_Interval],
    bounds: list[_Interval] | None,
    rng: random.Random,
) -> _Interval | None:
    """Find one non-overlapping slot of length `duration` honouring bounds."""
    candidate_bounds = bounds or [_Interval(0, 24 * 60)]
    for bound in candidate_bounds:
        free_starts = _enumerate_free_starts(bound, duration, obstacles)
        if not free_starts:
            continue
        start = rng.choice(free_starts)
        return _Interval(start, start + duration)
    return None


def _enumerate_free_starts(
    bound: _Interval,
    duration: int,
    obstacles: list[_Interval],
) -> list[int]:
    """List all start minutes inside `bound` whose slot is free."""
    if duration <= 0 or bound.end - bound.start < duration:
        return []
    cursor = bound.start
    starts: list[int] = []
    sorted_obs = sorted(
        (o for o in obstacles if o.end > bound.start and o.start < bound.end),
        key=lambda o: o.start,
    )
    for ob in sorted_obs:
        free_end = min(ob.start, bound.end)
        if free_end - cursor >= duration:
            starts.extend(range(cursor, free_end - duration + 1))
        cursor = max(cursor, ob.end)
        if cursor >= bound.end:
            break
    if bound.end - cursor >= duration:
        starts.extend(range(cursor, bound.end - duration + 1))
    return starts


def _seed_for_day(person_seed: int, day_index: int, salt: str) -> int:
    """Mix the person seed with the day index and a category/member salt."""
    base = (person_seed * 1_000_003 + day_index * 31) & 0xFFFFFFFF
    for ch in salt:
        base = (base * 131 + ord(ch)) & 0xFFFFFFFF
    return base


def plan_contexts_for_day(
    person: Person,
    day: DaySchedule,
    resolved: dict[str, ResolvedCategory],
    window_map: WindowMap,
    *,
    ltl_rules: Sequence[TemporalRule] | None = None,
) -> list[ContextEpisode]:
    """Place context episodes for one person on one day."""
    if ltl_rules:
        z3_episodes = _solve_contexts_with_ltl(
            person, day, resolved, window_map, ltl_rules
        )
        if z3_episodes is not None:
            return z3_episodes
    weekday = day.weekday
    placed: list[ContextEpisode] = []

    # Track per-category placed intervals (for mutual exclusion within
    # category) and per-dimension placed intervals (for trait pairs).
    by_category: dict[str, list[_Interval]] = {}
    by_dimension: dict[str, list[_Interval]] = {}

    for cat_name, category in resolved.items():
        category_placed: list[_Interval] = []
        for member_name, member in category.members.items():
            rng = random.Random(
                _seed_for_day(
                    person.person_seed,
                    day.day_index,
                    f"{cat_name}:{member_name}",
                )
            )
            episode_count = _episodes_for_day(member, rng)
            if episode_count <= 0:
                continue
            lo, hi = _per_event_minutes(member)
            if hi <= 0:
                continue
            enabled_today, windows = _fix_windows(member, weekday, window_map)
            if not enabled_today:
                continue
            for _ in range(episode_count):
                duration = rng.randint(lo, hi)
                # Contexts are person STATES (mood, energy, opportunity)
                # and may overlap any event freely; non-overlap with
                # specific events is enforced via LTL `G !(E AND C)`
                # rules in temporal_relation_rules.yaml.
                obstacles: list[_Interval] = []
                if category.mutually_exclusive:
                    obstacles += category_placed
                if member.dimension:
                    obstacles += by_dimension.get(member.dimension, [])
                slot = _free_slot(duration, obstacles, windows or None, rng)
                if slot is None:
                    continue
                placed.append(
                    ContextEpisode(
                        name=member_name,
                        category=cat_name,
                        date=day.date,
                        start_minutes=slot.start,
                        end_minutes=slot.end,
                        ontology_uri=member.ontology_uri,
                        dimension=member.dimension,
                        polarity=member.polarity,
                        instrument=member.instrument,
                        theory_mappings=(
                            dict(member.theory_mappings)
                            if member.theory_mappings is not None
                            else None
                        ),
                    )
                )
                category_placed.append(slot)
                if member.dimension:
                    by_dimension.setdefault(member.dimension, []).append(slot)
        by_category[cat_name] = category_placed
    return placed


def _solve_contexts_with_ltl(
    person: Person,
    day: DaySchedule,
    resolved: dict[str, ResolvedCategory],
    window_map: WindowMap,
    ltl_rules: Sequence[TemporalRule],
) -> list[ContextEpisode] | None:
    """Try z3-based placement honouring LTL rules; return None on unsat."""
    from src.scripts.persona.solver.context_day_model import (
        build_context_day_model,
        extract_context_episodes,
    )

    event_placements: dict[str, list[tuple[int, int]]] = {
        name: [(ev.start, ev.duration) for ev in evs]
        for name, evs in day.events.items()
    }
    category_by_member = {
        member_name: cat_name
        for cat_name, cat in resolved.items()
        for member_name in cat.members
    }
    solver, context_vars, member_lookup = build_context_day_model(
        day_idx=day.day_index,
        weekday=day.weekday,
        day_date=day.date,
        resolved=resolved,
        event_placements=event_placements,
        window_map=window_map,
        person=person,
        ltl_rules=ltl_rules,
        person_seed=person.person_seed,
    )
    if solver.check() != z3.sat:
        return None
    return extract_context_episodes(
        solver.model(),
        context_vars,
        member_lookup,
        category_by_member=category_by_member,
        day_date=day.date,
    )


def plan_contexts(
    person: Person,
    schedule: PersonSchedule,
    resolved: dict[str, ResolvedCategory],
    window_map: WindowMap,
    *,
    ltl_rules: Sequence[TemporalRule] | None = None,
) -> PersonSchedule:
    """Return a copy of `schedule` with placed `ContextEpisode`s attached."""
    episodes: list[ContextEpisode] = []
    for day in schedule.days:
        episodes.extend(
            plan_contexts_for_day(
                person, day, resolved, window_map, ltl_rules=ltl_rules
            )
        )
    return PersonSchedule(
        person_id=schedule.person_id,
        persona_id=schedule.persona_id,
        person_seed=schedule.person_seed,
        days=schedule.days,
        contexts=episodes,
    )


__all__ = [
    "plan_contexts",
    "plan_contexts_for_day",
]
