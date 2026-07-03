"""Unit tests for the PTIME 2-order Choquet integral."""

from __future__ import annotations

import math

import pytest

from src.scripts.scenarios.augmentation.ptime import (
    choquet_2order,
    default_importance,
    merge_importance,
    merge_interaction,
)
from src.scripts.scenarios.config.schema import _PTIME_CRITERIA


def test_default_importance_uniform_sum_one():
    n = len(_PTIME_CRITERIA)
    imp = default_importance()
    assert set(imp) == set(_PTIME_CRITERIA)
    assert all(v == pytest.approx(1.0 / n) for v in imp.values())
    assert math.isclose(sum(imp.values()), 1.0)


def test_merge_importance_preserves_overrides():
    n = len(_PTIME_CRITERIA)
    imp = merge_importance({"time": 0.9, "overlap": 0.1})
    assert imp["time"] == 0.9
    assert imp["overlap"] == 0.1
    for c in _PTIME_CRITERIA:
        if c not in {"time", "overlap"}:
            assert imp[c] == pytest.approx(1.0 / n)


def test_merge_interaction_defaults_zero_then_overrides():
    n = len(_PTIME_CRITERIA)
    inter = merge_interaction({"duration|overlap": 0.4})
    assert inter["duration|overlap"] == 0.4
    nonzero = {k: v for k, v in inter.items() if v != 0.0}
    assert nonzero == {"duration|overlap": 0.4}
    assert len(inter) == n * (n - 1) // 2  # unordered pairs


def test_choquet_all_ones_uniform_returns_one():
    util = {c: 1.0 for c in _PTIME_CRITERIA}
    f = choquet_2order(util, default_importance(), merge_interaction({}))
    assert f == pytest.approx(1.0)


def test_choquet_all_zeros_returns_zero():
    util = {c: 0.0 for c in _PTIME_CRITERIA}
    f = choquet_2order(util, default_importance(), merge_interaction({}))
    assert f == 0.0


def test_choquet_pair_interaction_uses_min():
    util = {c: 0.0 for c in _PTIME_CRITERIA}
    util["time"] = 0.6
    util["duration"] = 0.4
    imp = {c: 0.0 for c in _PTIME_CRITERIA}
    inter = {"duration|time": 0.5}
    # Linear part = 0; interaction term = 0.5 * min(0.6, 0.4) = 0.2
    assert choquet_2order(util, imp, inter) == pytest.approx(0.2)


def test_choquet_substitutive_interaction_subtracts():
    util = {c: 0.0 for c in _PTIME_CRITERIA}
    util["time"] = 1.0
    util["duration"] = 1.0
    imp = {"time": 0.5, "duration": 0.5}
    for c in _PTIME_CRITERIA:
        imp.setdefault(c, 0.0)
    inter = {"duration|time": -0.5}
    # Linear = 0.5 + 0.5 = 1.0; interaction = -0.5 * min(1, 1) = -0.5.
    assert choquet_2order(util, imp, inter) == pytest.approx(0.5)


def test_choquet_missing_criterion_defaults_to_zero():
    n = len(_PTIME_CRITERIA)
    util = {"time": 1.0}  # other criteria absent
    imp = default_importance()
    f = choquet_2order(util, imp, merge_interaction({}))
    # Only "time" contributes; (1/n) * 1 = 1/n.
    assert f == pytest.approx(1.0 / n)
