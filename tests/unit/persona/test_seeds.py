"""Unit tests for src.scripts.persona.sampling.seeds."""

from __future__ import annotations

import pytest

from src.scripts.persona.sampling.seeds import mix_axis_seed, mix_seed


def test_mix_seed_is_pure():
    a = mix_seed(42, "alice", 0)
    b = mix_seed(42, "alice", 0)
    assert a == b


def test_mix_seed_changes_with_root():
    assert mix_seed(1, "alice", 0) != mix_seed(2, "alice", 0)


def test_mix_seed_changes_with_persona_id():
    assert mix_seed(42, "alice", 0) != mix_seed(42, "bob", 0)


def test_mix_seed_changes_with_index():
    assert mix_seed(42, "alice", 0) != mix_seed(42, "alice", 1)


def test_mix_seed_returns_64_bit_unsigned_int():
    s = mix_seed(42, "alice", 0)
    assert 0 <= s < 2**64


def test_mix_seed_order_independent_across_personas():
    """Seed for (root, "alice", 0) is the same whether bob was sampled first or not."""
    seeds_a_first = (mix_seed(7, "alice", 0), mix_seed(7, "bob", 0))
    seeds_b_first = (mix_seed(7, "bob", 0), mix_seed(7, "alice", 0))
    assert seeds_a_first[0] == seeds_b_first[1]
    assert seeds_a_first[1] == seeds_b_first[0]


def test_mix_seed_rejects_negative_root():
    with pytest.raises(ValueError):
        mix_seed(-1, "alice", 0)


def test_mix_seed_rejects_negative_index():
    with pytest.raises(ValueError):
        mix_seed(0, "alice", -1)


def test_mix_seed_rejects_empty_persona_id():
    with pytest.raises(ValueError):
        mix_seed(0, "", 0)


def test_mix_axis_seed_is_pure():
    assert mix_axis_seed(1, "alice", "age") == mix_axis_seed(1, "alice", "age")


def test_mix_axis_seed_changes_with_root():
    assert mix_axis_seed(1, "alice", "age") != mix_axis_seed(2, "alice", "age")


def test_mix_axis_seed_changes_with_persona_id():
    assert mix_axis_seed(1, "alice", "age") != mix_axis_seed(1, "bob", "age")


def test_mix_axis_seed_changes_with_axis_name():
    assert mix_axis_seed(1, "alice", "age") != mix_axis_seed(1, "alice", "income")


def test_mix_axis_seed_is_disjoint_from_person_index_seed():
    """Per-axis draws must not collide with per-instance draws on the same persona."""
    assert mix_axis_seed(1, "alice", "0") != mix_seed(1, "alice", 0)


def test_mix_axis_seed_rejects_negative_root():
    with pytest.raises(ValueError):
        mix_axis_seed(-1, "alice", "age")


def test_mix_axis_seed_rejects_empty_persona_id():
    with pytest.raises(ValueError):
        mix_axis_seed(0, "", "age")


def test_mix_axis_seed_rejects_empty_axis():
    with pytest.raises(ValueError):
        mix_axis_seed(0, "alice", "")
