"""Unit tests for src.scripts.scenarios.metrics.allen.

Coverage targets:
  - All 13 Allen relations (one canonical example each).
  - Converse symmetry: if I_i rel I_j then I_j inv(rel) I_i.
  - Boundary / edge cases: adjacent, minimal-duration intervals.
  - is_overlapping: all 13 relations classified correctly.
  - Module-level constants: R_SEP, R_MERGE, R_ALL cardinality and disjointness.
"""

from __future__ import annotations

import pytest

from src.scripts.scenarios.metrics.allen import (
    R_ALL,
    R_MERGE,
    R_SEP,
    AllenRelation,
    compute_allen_relation,
    is_overlapping,
)

# Short alias used throughout this module.
rel = compute_allen_relation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_enum_has_thirteen_members(self):
        assert len(AllenRelation) == 13

    def test_r_sep_has_four_members(self):
        assert len(R_SEP) == 4

    def test_r_merge_has_nine_members(self):
        assert len(R_MERGE) == 9

    def test_r_sep_and_r_merge_are_disjoint(self):
        assert R_SEP & R_MERGE == frozenset()

    def test_r_all_is_union_of_sep_and_merge(self):
        assert R_ALL == R_SEP | R_MERGE

    def test_r_all_covers_every_relation(self):
        assert R_ALL == frozenset(AllenRelation)

    def test_r_sep_members(self):
        assert R_SEP == {
            AllenRelation.p,
            AllenRelation.m,
            AllenRelation.M,
            AllenRelation.P,
        }

    def test_r_merge_contains_equals(self):
        assert AllenRelation.e in R_MERGE


# ---------------------------------------------------------------------------
# All 13 Allen relations; one canonical example each
# ---------------------------------------------------------------------------


class TestAllThirteenRelations:
    """Canonical examples covering every Allen relation.

    Notation: [s, e) with times in minutes.
    """

    def test_p_precedes(self):
        # [10, 20) strictly before [30, 40); gap of 10
        assert rel(10, 20, 30, 40) == AllenRelation.p

    def test_m_meets(self):
        # [10, 20) end touches [20, 30) start
        assert rel(10, 20, 20, 30) == AllenRelation.m

    def test_o_overlaps(self):
        # [10, 25) overlaps [20, 40): 10 < 20 < 25 < 40
        assert rel(10, 25, 20, 40) == AllenRelation.o

    def test_s_starts(self):
        # [10, 20) shares start with [10, 40), ends first
        assert rel(10, 20, 10, 40) == AllenRelation.s

    def test_d_during(self):
        # [15, 25) is entirely inside [10, 30)
        assert rel(15, 25, 10, 30) == AllenRelation.d

    def test_f_finishes(self):
        # [20, 30) shares end with [10, 30), starts later
        assert rel(20, 30, 10, 30) == AllenRelation.f

    def test_e_equals(self):
        # [10, 30) is identical to [10, 30)
        assert rel(10, 30, 10, 30) == AllenRelation.e

    def test_P_preceded_by(self):
        # [30, 40) is strictly after [10, 20)
        assert rel(30, 40, 10, 20) == AllenRelation.P

    def test_M_met_by(self):
        # [20, 30) start touches [10, 20) end
        assert rel(20, 30, 10, 20) == AllenRelation.M

    def test_O_overlapped_by(self):
        # [20, 40) is overlapped from left by [10, 25): 10 < 20 < 25 < 40
        assert rel(20, 40, 10, 25) == AllenRelation.O

    def test_S_started_by(self):
        # [10, 40) shares start with [10, 20), ends later
        assert rel(10, 40, 10, 20) == AllenRelation.S

    def test_D_contains(self):
        # [10, 40) entirely contains [15, 25)
        assert rel(10, 40, 15, 25) == AllenRelation.D

    def test_F_finished_by(self):
        # [10, 30) shares end with [20, 30), starts earlier
        assert rel(10, 30, 20, 30) == AllenRelation.F


# ---------------------------------------------------------------------------
# Converse (inverse) symmetry
# ---------------------------------------------------------------------------

_CONVERSE: dict[AllenRelation, AllenRelation] = {
    AllenRelation.p: AllenRelation.P,
    AllenRelation.m: AllenRelation.M,
    AllenRelation.o: AllenRelation.O,
    AllenRelation.s: AllenRelation.S,
    AllenRelation.d: AllenRelation.D,
    AllenRelation.f: AllenRelation.F,
    AllenRelation.e: AllenRelation.e,
    AllenRelation.P: AllenRelation.p,
    AllenRelation.M: AllenRelation.m,
    AllenRelation.O: AllenRelation.o,
    AllenRelation.S: AllenRelation.s,
    AllenRelation.D: AllenRelation.d,
    AllenRelation.F: AllenRelation.f,
}

_SYMMETRY_CASES = [
    (10, 20, 30, 40),  # p / P
    (10, 20, 20, 30),  # m / M
    (10, 25, 20, 40),  # o / O
    (10, 20, 10, 40),  # s / S
    (15, 25, 10, 30),  # d / D
    (20, 30, 10, 30),  # f / F
    (10, 30, 10, 30),  # e (self-converse)
]


class TestConverseSymmetry:
    @pytest.mark.parametrize("si,ei,sj,ej", _SYMMETRY_CASES)
    def test_forward_backward_are_converses(self, si, ei, sj, ej):
        forward = rel(si, ei, sj, ej)
        backward = rel(sj, ej, si, ei)
        assert backward == _CONVERSE[forward], (
            f"rel({si},{ei},{sj},{ej})={forward.value!r} but "
            f"rel({sj},{ej},{si},{ei})={backward.value!r} "
            f"(expected {_CONVERSE[forward].value!r})"
        )

    def test_converse_map_is_complete(self):
        assert set(_CONVERSE.keys()) == set(AllenRelation)

    def test_converse_map_is_involutive(self):
        for r, c in _CONVERSE.items():
            assert _CONVERSE[c] == r, f"converse of converse of {r} is not {r}"


# ---------------------------------------------------------------------------
# Boundary and edge cases
# ---------------------------------------------------------------------------


class TestBoundaryCases:
    def test_gap_of_one_is_p_not_m(self):
        # e_i = 19 < s_j = 20: precedes, not meets
        assert rel(10, 19, 20, 30) == AllenRelation.p

    def test_touching_at_one_point_is_m(self):
        # e_i == s_j = 20: meets
        assert rel(10, 20, 20, 30) == AllenRelation.m

    def test_minimum_duration_intervals_p(self):
        # Smallest non-instantaneous intervals: [0,1) before [2,3)
        assert rel(0, 1, 2, 3) == AllenRelation.p

    def test_minimum_duration_intervals_m(self):
        # [0,1) meets [1,2)
        assert rel(0, 1, 1, 2) == AllenRelation.m

    def test_minimum_duration_intervals_d(self):
        # [1,2) is during [0,3)
        assert rel(1, 2, 0, 3) == AllenRelation.d

    def test_same_start_same_end_is_e(self):
        assert rel(720, 780, 720, 780) == AllenRelation.e

    def test_large_minute_values_p(self):
        # horizon across 4 weeks: 4 * 7 * 1440 = 40320 minutes
        assert rel(0, 1440, 1441, 2880) == AllenRelation.p

    def test_large_minute_values_d(self):
        # one day's sleep inside a multi-week horizon
        assert rel(10080, 10560, 0, 40320) == AllenRelation.d

    def test_o_requires_strict_inequalities(self):
        # overlaps requires s_i < s_j < e_i < e_j (all strict)
        r = rel(10, 25, 20, 40)
        assert r == AllenRelation.o
        # If e_i == s_j it would be m, not o
        assert rel(10, 20, 20, 40) == AllenRelation.m

    def test_f_requires_same_end_and_later_start(self):
        # [20, 30) finishes [10, 30): s_i > s_j, e_i == e_j
        assert rel(20, 30, 10, 30) == AllenRelation.f
        # One minute earlier start to d (during)
        assert rel(20, 29, 10, 30) == AllenRelation.d

    def test_midnight_boundary(self):
        # sleep crosses midnight: [1380, 1440) for day d, [0, 60) for day d+1
        # Cross-day events would be modelled as two separate intervals.
        # Within the same day: [1380, 1440) precedes nothing; but two
        # same-day events: [1380, 1440) and [60, 120) are P (preceded_by).
        assert rel(1380, 1440, 60, 120) == AllenRelation.P


# ---------------------------------------------------------------------------
# is_overlapping
# ---------------------------------------------------------------------------


class TestIsOverlapping:
    """is_overlapping returns True iff the relation is in R_MERGE."""

    # --- Relations that are NOT overlapping (R_SEP) ---
    def test_p_not_overlapping(self):
        assert not is_overlapping(10, 20, 30, 40)

    def test_m_not_overlapping(self):
        assert not is_overlapping(10, 20, 20, 30)

    def test_M_not_overlapping(self):
        assert not is_overlapping(20, 30, 10, 20)

    def test_P_not_overlapping(self):
        assert not is_overlapping(30, 40, 10, 20)

    # --- Relations that ARE overlapping (R_MERGE) ---
    def test_o_overlapping(self):
        assert is_overlapping(10, 25, 20, 40)

    def test_O_overlapping(self):
        assert is_overlapping(20, 40, 10, 25)

    def test_s_overlapping(self):
        assert is_overlapping(10, 20, 10, 40)

    def test_S_overlapping(self):
        assert is_overlapping(10, 40, 10, 20)

    def test_d_overlapping(self):
        assert is_overlapping(15, 25, 10, 30)

    def test_D_overlapping(self):
        assert is_overlapping(10, 40, 15, 25)

    def test_f_overlapping(self):
        assert is_overlapping(20, 30, 10, 30)

    def test_F_overlapping(self):
        assert is_overlapping(10, 30, 20, 30)

    def test_e_overlapping(self):
        assert is_overlapping(10, 30, 10, 30)

    def test_all_r_merge_members_are_overlapping(self):
        """Exhaustively verify every R_MERGE member returns is_overlapping=True."""
        cases = {
            AllenRelation.o: (10, 25, 20, 40),
            AllenRelation.O: (20, 40, 10, 25),
            AllenRelation.s: (10, 20, 10, 40),
            AllenRelation.S: (10, 40, 10, 20),
            AllenRelation.d: (15, 25, 10, 30),
            AllenRelation.D: (10, 40, 15, 25),
            AllenRelation.f: (20, 30, 10, 30),
            AllenRelation.F: (10, 30, 20, 30),
            AllenRelation.e: (10, 30, 10, 30),
        }
        for expected_rel, (si, ei, sj, ej) in cases.items():
            assert rel(si, ei, sj, ej) == expected_rel
            assert is_overlapping(
                si, ei, sj, ej
            ), f"Expected is_overlapping for {expected_rel.value!r}"

    def test_all_r_sep_members_are_not_overlapping(self):
        """Exhaustively verify every R_SEP member returns is_overlapping=False."""
        cases = {
            AllenRelation.p: (10, 20, 30, 40),
            AllenRelation.m: (10, 20, 20, 30),
            AllenRelation.M: (20, 30, 10, 20),
            AllenRelation.P: (30, 40, 10, 20),
        }
        for expected_rel, (si, ei, sj, ej) in cases.items():
            assert rel(si, ei, sj, ej) == expected_rel
            assert not is_overlapping(
                si, ei, sj, ej
            ), f"Expected NOT is_overlapping for {expected_rel.value!r}"


def test_selector_matcher_resolve_handles_missing_class_closure_entry() -> None:
    """SelectorMatcher.resolve returns empty closure fields when the URI is unknown."""
    from src.scripts.scenarios.domain.task import RecommendedTask
    from src.scripts.scenarios.metrics.allen import SelectorMatcher

    matcher = SelectorMatcher(
        class_closure={"http://example.org/other": frozenset({"X"})},
    )
    task = RecommendedTask(
        label="x",
        duration_min=10,
        duration_max=20,
        intensity=2,
        is_concurrent=False,
        is_dividable=False,
        display_name="X",
        ontology_uri="http://example.org/missing-from-closure",
    )
    out = matcher.resolve(task)
    assert out.health_task_uri is None
    assert out.health_task_classes == frozenset()
