"""Retry helper for build/check/accept loops over a z3 model."""

from __future__ import annotations

from collections.abc import Callable

import z3

from src.scripts.persona.solver.spillover import EventVars

BuildFn = Callable[[int], tuple[z3.Solver | z3.Optimize, EventVars]]
AcceptFn = Callable[[z3.ModelRef, EventVars], bool]


def _always_accept(_model: z3.ModelRef, _vars: EventVars) -> bool:
    return True


def solve_with_retry(
    build_for_attempt: BuildFn,
    accept: AcceptFn = _always_accept,
    *,
    max_attempts: int = 25,
) -> tuple[z3.ModelRef, EventVars] | None:
    """Run build/check/accept up to `max_attempts` times.

    The builder receives the attempt index (0-based) so callers can vary the z3 random
    seed between attempts. The first `(model, event_vars)` pair accepted by `accept`
    is returned; `None` if every attempt is unsat or rejected.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    for attempt in range(max_attempts):
        solver, event_vars = build_for_attempt(attempt)
        if solver.check() != z3.sat:
            continue
        model = solver.model()
        if accept(model, event_vars):
            return model, event_vars
    return None
