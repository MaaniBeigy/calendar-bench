"""Tests for the PTIME MCS branch-and-bound solver."""

from __future__ import annotations

import datetime

import pytest

from src.scripts.scenarios.augmentation.ptime import (
    Candidate,
    Solution,
    Variable,
    _build_variables,
    _candidates_conflict,
    _choquet_upper_bound,
    _mono_criterion_search,
    _picks_to_scheduled,
    mcs_solve,
    merge_importance,
    merge_interaction,
)
from src.scripts.scenarios.config.schema import _PTIME_CRITERIA
from tests.unit.scenarios.conftest import make_task

DATE = datetime.date(2026, 5, 4)
DATE2 = datetime.date(2026, 5, 5)


def _cand(start: int, end: int, *, util: dict[str, float], day=DATE) -> Candidate:
    full = {c: util.get(c, 0.0) for c in _PTIME_CRITERIA}
    return Candidate(
        start=start,
        end=end,
        day=day,
        base_gap=(start, end),
        score=0.0,
        utilities=full,
    )


# ---------------------------------------------------------------------------
# Solution.utility_means + choquet
# ---------------------------------------------------------------------------


def test_solution_utility_means_zero_for_unscheduled_picks():
    sol = Solution(picks=(_cand(420, 450, util={"time": 1.0, "duration": 1.0}), None))
    means = sol.utility_means()
    # Mean over 2 picks: first contributes 1.0, second 0.0; mean is 0.5.
    assert means["time"] == pytest.approx(0.5)
    assert means["duration"] == pytest.approx(0.5)
    assert means["overlap"] == 0.0


def test_solution_choquet_uses_importance_and_interaction():
    sol = Solution(picks=(_cand(420, 450, util={"time": 1.0, "duration": 1.0}),))
    imp = {c: 0.0 for c in _PTIME_CRITERIA}
    imp["time"] = 0.5
    imp["duration"] = 0.5
    score = sol.choquet(imp, merge_interaction({}))
    assert score == pytest.approx(1.0)


def test_solution_empty_picks_returns_zero_means():
    sol = Solution(picks=())
    assert sol.utility_means() == {c: 0.0 for c in _PTIME_CRITERIA}


# ---------------------------------------------------------------------------
# _candidates_conflict
# ---------------------------------------------------------------------------


def test_candidates_conflict_different_days_never_conflict():
    a = _cand(420, 480, util={}, day=DATE)
    b = _cand(420, 480, util={}, day=DATE2)
    assert _candidates_conflict(a, b) is False


def test_candidates_conflict_overlapping_same_day():
    a = _cand(420, 480, util={}, day=DATE)
    b = _cand(450, 510, util={}, day=DATE)
    assert _candidates_conflict(a, b) is True


def test_candidates_conflict_adjacent_same_day_does_not_conflict():
    # Half-open intervals; adjacency is allowed.
    a = _cand(420, 480, util={}, day=DATE)
    b = _cand(480, 540, util={}, day=DATE)
    assert _candidates_conflict(a, b) is False


# ---------------------------------------------------------------------------
# _build_variables (most-constrained-first ordering)
# ---------------------------------------------------------------------------


def test_build_variables_orders_by_domain_size():
    t1 = make_task(label="t1")
    t2 = make_task(label="t2")
    t3 = make_task(label="t3")
    pairs = [
        (t1, [_cand(420, 450, util={}), _cand(450, 480, util={})]),
        (t2, [_cand(420, 450, util={})]),
        (
            t3,
            [
                _cand(420, 450, util={}),
                _cand(450, 480, util={}),
                _cand(480, 510, util={}),
            ],
        ),
    ]
    vars_ = _build_variables(pairs)
    sizes = [len(v.candidates) for v in vars_]
    assert sizes == sorted(sizes)
    assert vars_[0].task is t2  # smallest domain first


# ---------------------------------------------------------------------------
# _mono_criterion_search
# ---------------------------------------------------------------------------


def test_mono_criterion_search_picks_max_per_var_when_unconstrained():
    t = make_task(label="t1")
    var = Variable(
        index=0,
        task=t,
        candidates=(
            _cand(420, 450, util={"time": 0.2}),
            _cand(480, 510, util={"time": 0.9}),
            _cand(540, 570, util={"time": 0.5}),
        ),
    )
    sol = _mono_criterion_search(
        [var], "time", locked_thresholds={}, incumbent=None, time_budget_seconds=1.0
    )
    assert sol is not None
    assert sol.picks[0].start == 480


def test_mono_criterion_search_respects_time_conflicts_across_vars():
    t1, t2 = make_task(label="t1"), make_task(label="t2")
    # Both vars have best candidate at 420..480; conflict forces var 2 to pick
    # the next-best (later) slot.
    v1 = Variable(
        index=0,
        task=t1,
        candidates=(_cand(420, 480, util={"time": 1.0}),),
    )
    v2 = Variable(
        index=1,
        task=t2,
        candidates=(
            _cand(420, 480, util={"time": 1.0}),
            _cand(540, 600, util={"time": 0.4}),
        ),
    )
    sol = _mono_criterion_search(
        [v1, v2], "time", locked_thresholds={}, incumbent=None, time_budget_seconds=1.0
    )
    assert sol is not None
    assert sol.picks[0].start == 420
    assert sol.picks[1].start == 540


def test_mono_criterion_search_drops_var_when_no_candidate():
    # Empty domain forces the skip branch; pick is `None`.
    t = make_task(label="t1")
    var = Variable(index=0, task=t, candidates=())
    sol = _mono_criterion_search(
        [var], "time", locked_thresholds={}, incumbent=None, time_budget_seconds=1.0
    )
    assert sol is not None
    assert sol.picks == (None,)


def test_mono_criterion_search_locked_threshold_prevents_regression():
    # Lock `time` at 0.9 so only the high-`time` candidate satisfies the lock.
    t = make_task(label="t1")
    var = Variable(
        index=0,
        task=t,
        candidates=(
            _cand(420, 450, util={"time": 0.1, "duration": 1.0}),
            _cand(480, 510, util={"time": 1.0, "duration": 0.2}),
        ),
    )
    sol = _mono_criterion_search(
        [var],
        "duration",
        locked_thresholds={"time": 0.9},
        incumbent=None,
        time_budget_seconds=1.0,
    )
    assert sol is not None
    # The duration-rich candidate violates the time lock; search falls
    # back to the time-rich candidate which still clears 0.9.
    assert sol.picks[0].start == 480


# ---------------------------------------------------------------------------
# _choquet_upper_bound
# ---------------------------------------------------------------------------


def test_choquet_upper_bound_per_variable_max():
    t = make_task(label="t1")
    var = Variable(
        index=0,
        task=t,
        candidates=(
            _cand(420, 450, util={"time": 0.3}),
            _cand(480, 510, util={"time": 0.9}),
        ),
    )
    ub = _choquet_upper_bound([var], merge_importance({}), merge_interaction({}))
    assert ub["time"] == pytest.approx(0.9)
    # No candidate touches `duration` so the bound is 0.
    assert ub["duration"] == 0.0


# ---------------------------------------------------------------------------
# mcs_solve end-to-end
# ---------------------------------------------------------------------------


def test_mcs_solve_returns_none_when_no_candidates():
    t = make_task(label="t1")
    assert mcs_solve([(t, [])], merge_importance({}), merge_interaction({})) is None


def test_mcs_solve_returns_variables_and_solution():
    t1, t2 = make_task(label="t1"), make_task(label="t2")
    pairs = [
        (
            t1,
            [
                _cand(420, 450, util={"time": 1.0, "duration": 0.2}),
                _cand(480, 510, util={"time": 0.6, "duration": 0.8}),
            ],
        ),
        (
            t2,
            [
                _cand(420, 450, util={"time": 0.4, "duration": 1.0}, day=DATE2),
                _cand(540, 570, util={"time": 0.9, "duration": 0.5}, day=DATE2),
            ],
        ),
    ]
    importance = {c: 0.0 for c in _PTIME_CRITERIA}
    importance["time"] = 1.0
    result = mcs_solve(
        pairs, importance, merge_interaction({}), time_budget_seconds=1.0
    )
    assert result is not None
    variables, sol = result
    assert len(variables) == 2
    assert all(p is not None for p in sol.picks)


def test_mcs_solve_skips_pairs_with_empty_candidate_lists():
    # `mcs_solve` drops empty-domain tasks before building variables.
    t1, t2 = make_task(label="t1"), make_task(label="t2")
    pairs = [
        (t1, [_cand(420, 450, util={"time": 1.0})]),
        (t2, []),
    ]
    result = mcs_solve(pairs, merge_importance({}), merge_interaction({}))
    assert result is not None
    variables, _sol = result
    assert len(variables) == 1
    assert variables[0].task is t1


# ---------------------------------------------------------------------------
# _picks_to_scheduled
# ---------------------------------------------------------------------------


def test_picks_to_scheduled_splits_into_two_lists():
    t1, t2, t3 = make_task(label="t1"), make_task(label="t2"), make_task(label="t3")
    variables = [
        Variable(index=0, task=t1, candidates=(_cand(420, 450, util={}),)),
        Variable(index=1, task=t2, candidates=(_cand(450, 480, util={}),)),
        Variable(index=2, task=t3, candidates=(_cand(480, 510, util={}),)),
    ]
    picks = (
        _cand(420, 450, util={}),
        None,
        _cand(480, 510, util={}),
    )
    sol = Solution(picks=picks)
    scheduled, unscheduled = _picks_to_scheduled(variables, sol, person_id="p1")
    assert [s.task.label for s in scheduled] == ["t1", "t3"]
    assert [u.label for u in unscheduled] == ["t2"]
