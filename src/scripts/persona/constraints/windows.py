"""Resolve window names against a WindowMap into allowed start minutes."""

from __future__ import annotations

from collections.abc import Iterable

from src.scripts.persona.domain.time_windows import WindowMap


def resolve_window_starts(
    window_names: Iterable[str],
    window_map: WindowMap,
    step_minutes: int = 10,
) -> set[int]:
    """Build the set of allowed start minutes inside any of the named windows.

    Unknown names are silently skipped so an event can list day-name tokens
    (Saturday, weekend) alongside true window names without breaking start
    enumeration.
    """
    if step_minutes < 1:
        raise ValueError(f"step_minutes must be >= 1, got {step_minutes}")
    starts: set[int] = set()
    for name in window_names:
        if name in window_map:
            start, end = window_map.get(name)
            starts.update(range(start, end, step_minutes))
    return starts
