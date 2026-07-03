"""Reusable seeded sampling helpers (no shared global RNG)."""

from __future__ import annotations

import random


def jitter_minutes(rng: random.Random, *, bound: int) -> int:
    """Uniform integer offset in [-bound, +bound]. bound=0 always returns 0."""
    if bound < 0:
        raise ValueError(f"bound must be non-negative, got {bound}")
    if bound == 0:
        return 0
    return rng.randint(-bound, bound)


def jittered_time_minutes(
    rng: random.Random, *, base_minutes: int, bound: int, day_minutes: int = 1440
) -> int:
    """Apply a bounded uniform jitter to a clock time, clamped to a single day."""
    offset = jitter_minutes(rng, bound=bound)
    raw = base_minutes + offset
    if raw < 0:
        return 0
    if raw >= day_minutes:
        return day_minutes - 1
    return raw


def jittered_duration_minutes(
    rng: random.Random, *, base_minutes: int, bound: int, min_minutes: int = 1
) -> int:
    """Apply a bounded uniform jitter to a duration, never going below min_minutes."""
    offset = jitter_minutes(rng, bound=bound)
    return max(min_minutes, base_minutes + offset)
