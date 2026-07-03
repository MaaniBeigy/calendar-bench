"""Detect events that cross midnight and convert them into next-day obstacles."""

from __future__ import annotations

from collections.abc import Iterable

import z3

from src.scripts.persona.domain.schedule import Spillover

DAY_MINUTES = 1440

# Type alias matching the shape used by `solver.day_model`.
EventVars = dict[str, list[tuple[z3.ArithRef, z3.ArithRef]]]


def extract_spillovers(
    model: z3.ModelRef,
    event_vars: EventVars,
    *,
    day_minutes: int = DAY_MINUTES,
) -> list[Spillover]:
    """Return one `Spillover` for every event whose `start + duration > day_minutes`."""
    spillovers: list[Spillover] = []
    for event_name, vars_for_type in event_vars.items():
        for idx, (start_var, duration_var) in enumerate(vars_for_type):
            s_val = model[start_var].as_long()
            d_val = model[duration_var].as_long()
            if s_val + d_val > day_minutes:
                spillovers.append(
                    Spillover(
                        event_name=event_name,
                        start=0,
                        duration=(s_val + d_val) - day_minutes,
                        orig_start=s_val,
                        orig_duration=d_val,
                        event_idx=idx,
                    )
                )
    return spillovers


def occupied_from_spillovers(
    spillovers: Iterable[Spillover],
) -> list[tuple[int, int]]:
    """Translate spillovers into `[start, end)` ranges occupied on the next day."""
    return [(s.start, s.start + s.duration) for s in spillovers if s.duration > 0]
