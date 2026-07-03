"""z3 second-pass placer for context episodes; mirrors `build_day_model`.

Runs after the event solver. Event placements arrive as fixed
`(start, duration)` integer tuples and feed the same LTL constraint
helper that the event phase uses, so rules like
`G ¬(swimming ∧ context:home)` or
`G (gym → F (gym ∧ context:self_efficacy))` become hard z3 constraints
spanning both layers.

When z3 returns `unsat` the orchestrator can drop LTL rules and fall
back to the legacy greedy placer (see `context.planner.plan_contexts`).
"""

from __future__ import annotations

import datetime as _dt
import random
from collections.abc import Sequence

import z3

from src.scripts.persona.config.schema import ContextMember, TemporalRule
from src.scripts.persona.context.planner import (
    _episodes_for_day,
    _fix_windows,
    _per_event_minutes,
    _seed_for_day,
)
from src.scripts.persona.context.resolver import ResolvedCategory
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.solver.day_model import (
    _CONTEXT_PREFIX,
    DAY_MINUTES,
    _add_ltl_constraints,
)

ContextVars = dict[str, list[tuple[z3.ArithRef, z3.ArithRef]]]


def build_context_day_model(
    *,
    day_idx: int,
    weekday: str,
    day_date: _dt.date,
    resolved: dict[str, ResolvedCategory],
    event_placements: dict[str, list[tuple[int, int]]],
    window_map: WindowMap,
    person: Person | None = None,
    ltl_rules: Sequence[TemporalRule] | None = None,
    person_seed: int = 0,
) -> tuple[z3.Solver, ContextVars, dict[str, ContextMember]]:
    """Build a z3 model placing context episodes around fixed events."""
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)

    atom_vars: dict = {}
    # Wrap event placements as z3.IntVal so mixed event/context LTL
    # constraints produce z3 BoolRef expressions even when both atoms
    # in a rule are already-solved events.
    for ev_name, instances in event_placements.items():
        atom_vars[ev_name] = [
            (z3.IntVal(s, ctx=ctx), z3.IntVal(d, ctx=ctx)) for s, d in instances
        ]

    context_vars: ContextVars = {}
    member_lookup: dict[str, ContextMember] = {}
    by_dimension: dict[str, list[tuple[z3.ArithRef, z3.ArithRef]]] = {}

    for cat_name, category in resolved.items():
        category_episode_vars: list[tuple[z3.ArithRef, z3.ArithRef]] = []
        for member_name, member in category.members.items():
            rng = random.Random(
                _seed_for_day(person_seed, day_idx, f"{cat_name}:{member_name}")
            )
            episode_count = _episodes_for_day(member, rng)
            if episode_count <= 0:
                continue
            lo, hi = _per_event_minutes(member)
            if hi <= 0:
                continue
            enabled, windows = _fix_windows(member, weekday, window_map)
            if not enabled:
                continue
            member_vars: list[tuple[z3.ArithRef, z3.ArithRef]] = []
            for ep_idx in range(episode_count):
                s = z3.Int(f"ctx_{cat_name}_{member_name}_{ep_idx}_start", ctx=ctx)
                d = z3.Int(f"ctx_{cat_name}_{member_name}_{ep_idx}_duration", ctx=ctx)
                solver.add(d >= lo, d <= hi)
                solver.add(s >= 0)
                solver.add(s + d <= DAY_MINUTES)
                if windows:
                    window_clauses = [
                        z3.And(s >= w.start, s + d <= w.end) for w in windows
                    ]
                    solver.add(
                        window_clauses[0]
                        if len(window_clauses) == 1
                        else z3.Or(*window_clauses)
                    )
                member_vars.append((s, d))
            total_band = member.total_event_duration
            total_factor = 60 if total_band.unit == "hours" else 1
            total_min = int(total_band.min) * total_factor
            total_max = int(total_band.max) * total_factor
            durations = [v[1] for v in member_vars]
            solver.add(z3.Sum(durations) >= total_min)
            solver.add(z3.Sum(durations) <= total_max)
            for i in range(len(member_vars)):
                s1, d1 = member_vars[i]
                for j in range(i + 1, len(member_vars)):
                    s2, d2 = member_vars[j]
                    solver.add(z3.Or(s1 + d1 <= s2, s2 + d2 <= s1))
            context_vars[member_name] = member_vars
            member_lookup[member_name] = member
            atom_vars[f"{_CONTEXT_PREFIX}{member_name}"] = member_vars
            category_episode_vars.extend(member_vars)
            if member.dimension:
                by_dimension.setdefault(member.dimension, []).extend(member_vars)

        if category.mutually_exclusive and len(category_episode_vars) > 1:
            for i in range(len(category_episode_vars)):
                s1, d1 = category_episode_vars[i]
                for j in range(i + 1, len(category_episode_vars)):
                    s2, d2 = category_episode_vars[j]
                    solver.add(z3.Or(s1 + d1 <= s2, s2 + d2 <= s1))

    for grouped in by_dimension.values():
        if len(grouped) <= 1:
            continue
        for i in range(len(grouped)):
            s1, d1 = grouped[i]
            for j in range(i + 1, len(grouped)):
                s2, d2 = grouped[j]
                solver.add(z3.Or(s1 + d1 <= s2, s2 + d2 <= s1))

    if ltl_rules:
        _add_ltl_constraints(solver, atom_vars, ltl_rules, person=person)

    return solver, context_vars, member_lookup


def extract_context_episodes(
    model: z3.ModelRef,
    context_vars: ContextVars,
    member_lookup: dict[str, ContextMember],
    *,
    category_by_member: dict[str, str],
    day_date: _dt.date,
) -> list[ContextEpisode]:
    """Read solved start/duration values back into `ContextEpisode` rows."""
    out: list[ContextEpisode] = []
    for member_name, episodes in context_vars.items():
        member = member_lookup[member_name]
        cat_name = category_by_member[member_name]
        for s, d in episodes:
            start = model[s].as_long()
            duration = model[d].as_long()
            out.append(
                ContextEpisode(
                    name=member_name,
                    category=cat_name,
                    date=day_date,
                    start_minutes=start,
                    end_minutes=start + duration,
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
    return out


__all__ = [
    "ContextVars",
    "build_context_day_model",
    "extract_context_episodes",
]
