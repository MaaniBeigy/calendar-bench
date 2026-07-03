"""Unit tests for `draw_axis_values`."""

from __future__ import annotations

from collections import Counter

import pytest

from src.scripts.persona.config.schema import (
    BooleanDist,
    CategoricalDist,
    ClipRange,
    ScipyDist,
)
from src.scripts.persona.sampling.characteristics import draw_axis_values


def test_categorical_realised_counts_match_within_one():
    dist = CategoricalDist(type="categorical", values={"a": 0.5, "b": 0.3, "c": 0.2})
    out = draw_axis_values("axis", dist, "p", 100, root_seed=42)
    counts = Counter(out)
    assert counts["a"] == 50
    assert counts["b"] == 30
    assert counts["c"] == 20
    assert len(out) == 100


def test_categorical_handles_rounding_diff_on_largest_bucket():
    dist = CategoricalDist(type="categorical", values={"a": 0.34, "b": 0.33, "c": 0.33})
    out = draw_axis_values("axis", dist, "p", 10, root_seed=1)
    counts = Counter(out)
    assert sum(counts.values()) == 10
    assert counts["a"] >= 3


def test_categorical_deterministic_under_same_seed():
    dist = CategoricalDist(type="categorical", values={"a": 0.5, "b": 0.5})
    a = draw_axis_values("x", dist, "p1", 20, root_seed=7)
    b = draw_axis_values("x", dist, "p1", 20, root_seed=7)
    assert a == b


def test_categorical_diverges_under_different_persona_id():
    dist = CategoricalDist(type="categorical", values={"a": 0.5, "b": 0.5})
    a = draw_axis_values("x", dist, "p1", 20, root_seed=7)
    b = draw_axis_values("x", dist, "p2", 20, root_seed=7)
    assert a != b


def test_categorical_diverges_under_different_axis_name():
    dist = CategoricalDist(type="categorical", values={"a": 0.5, "b": 0.5})
    a = draw_axis_values("axis_a", dist, "p1", 20, root_seed=7)
    b = draw_axis_values("axis_b", dist, "p1", 20, root_seed=7)
    assert a != b


def test_boolean_realised_counts_match_p_true():
    out = draw_axis_values(
        "has_kids", BooleanDist(type="boolean", p_true=0.30), "p", 100, root_seed=3
    )
    counts = Counter(out)
    assert counts[True] == 30
    assert counts[False] == 70


def test_boolean_all_true_when_p_true_is_one():
    out = draw_axis_values(
        "x", BooleanDist(type="boolean", p_true=1.0), "p", 5, root_seed=1
    )
    assert all(v is True for v in out)


def test_boolean_all_false_when_p_true_is_zero():
    out = draw_axis_values(
        "x", BooleanDist(type="boolean", p_true=0.0), "p", 5, root_seed=1
    )
    assert all(v is False for v in out)


def test_scipy_norm_returns_ints_when_dtype_int():
    dist = ScipyDist(type="norm", params={"loc": 40, "scale": 8}, dtype="int")
    out = draw_axis_values("age", dist, "p", 50, root_seed=11)
    assert len(out) == 50
    assert all(isinstance(v, int) for v in out)


def test_scipy_norm_returns_floats_when_dtype_float():
    dist = ScipyDist(type="norm", params={"loc": 0.0, "scale": 1.0}, dtype="float")
    out = draw_axis_values("z", dist, "p", 20, root_seed=11)
    assert all(isinstance(v, float) for v in out)


def test_scipy_norm_with_clip_stays_inside_bounds():
    dist = ScipyDist(
        type="norm",
        params={"loc": 40, "scale": 30},
        clip=ClipRange(min=20, max=60),
        dtype="int",
    )
    out = draw_axis_values("age", dist, "p", 200, root_seed=3)
    assert all(20 <= v <= 60 for v in out)


def test_scipy_poisson_returns_integers():
    dist = ScipyDist(type="poisson", params={"mu": 7})
    out = draw_axis_values("count", dist, "p", 50, root_seed=2)
    assert all(isinstance(v, (int,)) for v in out)


def test_scipy_uniform_is_deterministic():
    dist = ScipyDist(type="uniform", params={"loc": 0, "scale": 10}, dtype="int")
    a = draw_axis_values("u", dist, "p", 30, root_seed=99)
    b = draw_axis_values("u", dist, "p", 30, root_seed=99)
    assert a == b


def test_scipy_clip_exhaustion_falls_back_to_bound():
    dist = ScipyDist(
        type="norm",
        params={"loc": 1000, "scale": 0.001},
        clip=ClipRange(min=0, max=1),
    )
    with pytest.warns(UserWarning, match="clip exhausted"):
        out = draw_axis_values("rare", dist, "p", 3, root_seed=42)
    assert all(0 <= v <= 1 for v in out)


def test_scipy_dist_without_dtype_preserves_native_type():
    dist = ScipyDist(type="poisson", params={"mu": 3})
    out = draw_axis_values("c", dist, "p", 10, root_seed=1)
    assert len(out) == 10


def test_sample_size_zero_raises():
    dist = CategoricalDist(type="categorical", values={"a": 1.0})
    with pytest.raises(ValueError, match="sample_size must be positive"):
        draw_axis_values("x", dist, "p", 0, root_seed=1)


def test_unsupported_dist_type_raises():
    class _Other:
        pass

    with pytest.raises(TypeError, match="unsupported distribution"):
        draw_axis_values("x", _Other(), "p", 5, root_seed=1)  # type: ignore[arg-type]
