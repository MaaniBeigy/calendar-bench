"""Unit tests for src.scripts.persona.sampling.distributions."""

from __future__ import annotations

import random

import pytest

from src.scripts.persona.sampling.distributions import (
    jitter_minutes,
    jittered_duration_minutes,
    jittered_time_minutes,
)


def test_jitter_minutes_zero_bound_is_no_op():
    rng = random.Random(0)
    assert jitter_minutes(rng, bound=0) == 0


def test_jitter_minutes_within_bound():
    rng = random.Random(1234)
    for _ in range(200):
        assert -10 <= jitter_minutes(rng, bound=10) <= 10


def test_jitter_minutes_rejects_negative_bound():
    with pytest.raises(ValueError):
        jitter_minutes(random.Random(0), bound=-1)


def test_jittered_time_clamped_to_day():
    rng = random.Random(0)
    # base 30 min, bound 60 min: raw can be -30, but result must be ≥ 0
    for _ in range(100):
        v = jittered_time_minutes(rng, base_minutes=30, bound=60)
        assert 0 <= v < 1440


def test_jittered_time_clamped_to_day_upper():
    rng = random.Random(0)
    for _ in range(100):
        v = jittered_time_minutes(rng, base_minutes=1430, bound=60)
        assert 0 <= v < 1440


def test_jittered_duration_respects_min():
    rng = random.Random(0)
    for _ in range(100):
        v = jittered_duration_minutes(rng, base_minutes=5, bound=30, min_minutes=1)
        assert v >= 1


def test_jitter_is_seed_reproducible():
    a = random.Random(42)
    b = random.Random(42)
    seq_a = [jitter_minutes(a, bound=10) for _ in range(20)]
    seq_b = [jitter_minutes(b, bound=10) for _ in range(20)]
    assert seq_a == seq_b
