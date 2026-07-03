"""Tests for the prompt-ablation design library."""

from __future__ import annotations

import pytest

from src.scripts.scenarios.augmentation.prompts.ablation_designs import (
    DESIGN_NAMES,
    Variant,
    build_design,
    design_matrix,
)
from src.scripts.scenarios.augmentation.prompts.augment_oneshot import ABLATABLE_BLOCKS

_N_BLOCKS = len(ABLATABLE_BLOCKS)
_BASELINE_ID = "0" * _N_BLOCKS


# ---------------------------------------------------------------------------
# single
# ---------------------------------------------------------------------------


class TestSingle:
    def test_returns_one_variant(self):
        variants = build_design("single")
        assert len(variants) == 1

    def test_baseline_is_all_zeros(self):
        v = build_design("single")[0]
        assert v.variant_id == _BASELINE_ID
        assert v.ablated == frozenset()


# ---------------------------------------------------------------------------
# leave_one_out
# ---------------------------------------------------------------------------


class TestLeaveOneOut:
    def test_count_equals_blocks_plus_baseline(self):
        variants = build_design("leave_one_out")
        assert len(variants) == _N_BLOCKS + 1

    def test_first_variant_is_baseline(self):
        variants = build_design("leave_one_out")
        assert variants[0].variant_id == _BASELINE_ID

    def test_each_block_ablated_exactly_once(self):
        variants = build_design("leave_one_out")
        ablated_lists = [v.ablated for v in variants[1:]]
        for v in variants[1:]:
            assert len(v.ablated) == 1
        seen = frozenset().union(*ablated_lists)
        assert seen == frozenset(ABLATABLE_BLOCKS)

    def test_variant_ids_unique(self):
        variants = build_design("leave_one_out")
        ids = [v.variant_id for v in variants]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Plackett-Burman 12
# ---------------------------------------------------------------------------


class TestPlackettBurman12:
    def test_count_is_twelve(self):
        variants = build_design("plackett_burman_12", fold=False)
        assert len(variants) == 12

    def test_all_variant_ids_unique(self):
        variants = build_design("plackett_burman_12", fold=False)
        ids = [v.variant_id for v in variants]
        assert len(ids) == len(set(ids))

    def test_first_eleven_columns_balanced_six_six(self):
        """First 11 columns are standard PB-12 and must be 6/6 balanced."""
        matrix = design_matrix(build_design("plackett_burman_12", fold=False))
        for col in range(_N_BLOCKS - 2):
            column = [row[col] for row in matrix]
            assert column.count(+1) == 6
            assert column.count(-1) == 6

    def test_kept_columns_always_one_in_pb12_base(self):
        """Cols 11 (`split_dividable_tasks`) and 12 (`context_block`) are all +1 in PB-12 base."""
        matrix = design_matrix(build_design("plackett_burman_12", fold=False))
        for col_idx in (_N_BLOCKS - 2, _N_BLOCKS - 1):
            column = [row[col_idx] for row in matrix]
            assert column == [+1] * 12

    def test_includes_baseline_row(self):
        """PB-12 anchors on at least one extreme over the 11 testable factors."""
        variants = build_design("plackett_burman_12", fold=False)
        test_factors = frozenset(ABLATABLE_BLOCKS[:-2])
        assert any(
            (v.ablated & test_factors) in (frozenset(), test_factors) for v in variants
        ), "PB-12 must anchor on an extreme over the 11 testable factors"


# ---------------------------------------------------------------------------
# Plackett-Burman 24 (folded)
# ---------------------------------------------------------------------------


class TestPlackettBurman24:
    def test_count_is_at_least_twelve(self):
        variants = build_design("plackett_burman_24")
        # Folding doubles the row count; dedup collapses rows that
        # were their own sign mirror.
        assert len(variants) > 12

    def test_explicit_alias_folds(self):
        """`plackett_burman_24` matches `plackett_burman_12` with fold=True."""
        a = build_design("plackett_burman_24")
        b = build_design("plackett_burman_12", fold=True)
        assert {v.variant_id for v in a} == {v.variant_id for v in b}

    def test_columns_are_orthogonal_under_folding(self):
        """PB-24 columns are pairwise orthogonal; cols 11 and 12 share the always-kept fold pattern and are co-aliased."""
        matrix = design_matrix(build_design("plackett_burman_24"))
        aliased = {(_N_BLOCKS - 2, _N_BLOCKS - 1)}
        for i in range(_N_BLOCKS):
            col_i = [row[i] for row in matrix]
            for j in range(i + 1, _N_BLOCKS):
                if (i, j) in aliased:
                    continue
                col_j = [row[j] for row in matrix]
                ip = sum(a * b for a, b in zip(col_i, col_j))
                assert ip == 0, (
                    f"columns {i} and {j} are not orthogonal under "
                    f"folded PB-24 (inner product = {ip})"
                )

    def test_kept_columns_are_co_aliased_under_folding(self):
        """Cols 11 and 12 follow the same always-kept-then-flipped pattern, so PB-24 cannot separate them."""
        matrix = design_matrix(build_design("plackett_burman_24"))
        col_a = [row[_N_BLOCKS - 2] for row in matrix]
        col_b = [row[_N_BLOCKS - 1] for row in matrix]
        assert col_a == col_b

    def test_includes_all_zeros_and_all_ones_over_testable_factors(self):
        """PB-24 anchors on both extremes for the 11 testable factors."""
        ids = {v.variant_id for v in build_design("plackett_burman_24")}
        n_test = _N_BLOCKS - 2
        assert any(
            vid[:n_test] == "0" * n_test for vid in ids
        ), "PB-24 must include an all-kept extreme over the 11 testable factors"
        assert any(
            vid[:n_test] == "1" * n_test for vid in ids
        ), "PB-24 must include an all-ablated extreme over the 11 testable factors"

    def test_kept_columns_balanced_under_folding(self):
        """Cols 11 and 12 are +1 in PB-12 base and -1 in the fold; PB-24 is 12/12 for each."""
        matrix = design_matrix(build_design("plackett_burman_24"))
        for col_idx in (_N_BLOCKS - 2, _N_BLOCKS - 1):
            col = [row[col_idx] for row in matrix]
            assert col.count(+1) == 12
            assert col.count(-1) == 12


# ---------------------------------------------------------------------------
# Plackett-Burman 16
# ---------------------------------------------------------------------------


class TestPlackettBurman16:
    def test_count_is_sixteen(self):
        variants = build_design("plackett_burman_16")
        assert len(variants) == 16

    def test_variant_ids_unique(self):
        variants = build_design("plackett_burman_16")
        ids = [v.variant_id for v in variants]
        assert len(ids) == len(set(ids))

    def test_includes_all_kept_baseline(self):
        ids = {v.variant_id for v in build_design("plackett_burman_16")}
        assert _BASELINE_ID in ids

    def test_every_factor_column_is_balanced_eight_eight(self):
        """Each of the 13 factor columns has 8 kept and 8 ablated runs."""
        matrix = design_matrix(build_design("plackett_burman_16"))
        for col_idx in range(_N_BLOCKS):
            col = [row[col_idx] for row in matrix]
            assert col.count(+1) == 8, f"column {col_idx} is not balanced (8 +1 / 8 -1)"
            assert col.count(-1) == 8

    def test_all_factor_columns_pairwise_orthogonal(self):
        """Every pair of factor columns has zero inner product; no co-aliasing."""
        matrix = design_matrix(build_design("plackett_burman_16"))
        for i in range(_N_BLOCKS):
            col_i = [row[i] for row in matrix]
            for j in range(i + 1, _N_BLOCKS):
                col_j = [row[j] for row in matrix]
                ip = sum(a * b for a, b in zip(col_i, col_j))
                assert ip == 0, (
                    f"columns {i} and {j} are not orthogonal under "
                    f"PB-16 (inner product = {ip})"
                )

    def test_context_block_separable_from_split_dividable_tasks(self):
        """Cols 11 (split_dividable_tasks) and 12 (context_block) carry distinct patterns under PB-16; no co-aliasing."""
        matrix = design_matrix(build_design("plackett_burman_16"))
        col_a = [row[_N_BLOCKS - 2] for row in matrix]
        col_b = [row[_N_BLOCKS - 1] for row in matrix]
        assert col_a != col_b
        assert col_a != [-x for x in col_b]


# ---------------------------------------------------------------------------
# full_factorial reserved
# ---------------------------------------------------------------------------


class TestFullFactorial:
    def test_full_factorial_raises(self):
        with pytest.raises(NotImplementedError, match="reserved"):
            build_design("full_factorial")

    def test_full_factorial_is_a_valid_name(self):
        assert "full_factorial" in DESIGN_NAMES


# ---------------------------------------------------------------------------
# custom
# ---------------------------------------------------------------------------


class TestCustom:
    def test_passes_through_user_variants(self):
        variants = build_design(
            "custom",
            custom=[
                ("baseline", []),
                ("no_rot1", ["rule_of_thumb_1"]),
                ("no_rules", ["rule_of_thumb_1", "rule_of_thumb_2"]),
            ],
        )
        ids = [v.variant_id for v in variants]
        assert ids == ["baseline", "no_rot1", "no_rules"]
        assert variants[1].ablated == frozenset({"rule_of_thumb_1"})
        assert variants[2].ablated == frozenset({"rule_of_thumb_1", "rule_of_thumb_2"})

    def test_requires_non_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_design("custom", custom=[])

    def test_requires_custom_argument(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_design("custom")

    def test_rejects_unknown_block_name(self):
        with pytest.raises(ValueError, match="unknown block"):
            build_design("custom", custom=[("bad", ["does_not_exist"])])

    def test_rejects_blank_id(self):
        with pytest.raises(ValueError, match="non-empty string"):
            build_design("custom", custom=[("", ["rule_of_thumb_1"])])


# ---------------------------------------------------------------------------
# build_design dispatch
# ---------------------------------------------------------------------------


class TestBuildDesignDispatch:
    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="unknown design"):
            build_design("not_a_real_design")

    def test_every_design_name_resolvable_or_explicitly_reserved(self):
        """Every advertised name must build or raise `NotImplementedError`."""
        for name in DESIGN_NAMES:
            if name == "custom":
                build_design(name, custom=[("v", [])])
                continue
            try:
                variants = build_design(name)
            except NotImplementedError:
                continue
            assert all(isinstance(v, Variant) for v in variants)


# ---------------------------------------------------------------------------
# design_matrix shape
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    def test_one_row_per_variant(self):
        variants = build_design("leave_one_out")
        m = design_matrix(variants)
        assert len(m) == len(variants)

    def test_one_column_per_block(self):
        variants = build_design("leave_one_out")
        m = design_matrix(variants)
        for row in m:
            assert len(row) == _N_BLOCKS

    def test_baseline_row_is_all_plus_one(self):
        m = design_matrix(build_design("leave_one_out"))
        assert m[0] == [+1] * _N_BLOCKS

    def test_full_ablation_row_is_all_minus_one(self):
        variants = build_design(
            "custom",
            custom=[("all_off", list(ABLATABLE_BLOCKS))],
        )
        m = design_matrix(variants)
        assert m[0] == [-1] * _N_BLOCKS
