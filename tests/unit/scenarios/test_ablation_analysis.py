"""Tests for `src.scripts.scenarios.export.ablation_analysis`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.scripts.scenarios.augmentation.prompts.ablation_designs import build_design
from src.scripts.scenarios.augmentation.prompts.augment_oneshot import ABLATABLE_BLOCKS
from src.scripts.scenarios.export.ablation_analysis import (
    AblationFit,
    BlockEffect,
    VariantRecord,
    _as_float,
    _design_matrix,
    _read_one_variant,
    _read_variants_index,
    aggregate_ablation,
    fit_all_responses,
    read_ablation_records,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_variant(
    method_dir: Path,
    variant_id: str,
    *,
    average_total: float | None,
    average_gains: dict[str, float | None] | None = None,
    augment_tokens: float | None = None,
    augment_wall: float | None = None,
    augment_usd: float | None = None,
) -> None:
    """Write one variant's evaluation sidecars to disk."""
    eval_dir = method_dir / "ablation" / variant_id / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    total_payload = {
        "average_total_gain": average_total,
        "average_gains": average_gains or {},
    }
    (eval_dir / "total_scheduling_gain.json").write_text(
        json.dumps(total_payload), encoding="utf-8"
    )
    tel_payload = {
        "by_stage": {
            "augmentation": {
                "tokens_total": augment_tokens,
                "wall_time_seconds_total": augment_wall,
                "estimated_usd_total": augment_usd,
            }
        }
    }
    (eval_dir / "telemetry.json").write_text(json.dumps(tel_payload), encoding="utf-8")


def _write_variants_index(method_dir: Path, variants) -> None:
    (method_dir / "ablation").mkdir(parents=True, exist_ok=True)
    (method_dir / "ablation" / "variants.json").write_text(
        json.dumps(
            {
                "variants": [
                    {"variant_id": v.variant_id, "ablated": sorted(v.ablated)}
                    for v in variants
                ]
            }
        ),
        encoding="utf-8",
    )


def _build_pb24_tree(tmp_path: Path) -> Path:
    """Build a PB-24 tree with `y = 0.5 + 0.1*x_0 + 0.05*x_5 + noise`."""
    method_dir = tmp_path / "scenarios" / "scen" / "llm_agent"
    method_dir.mkdir(parents=True, exist_ok=True)
    variants = build_design("plackett_burman_24")
    _write_variants_index(method_dir, variants)
    rng = np.random.default_rng(42)
    for v in variants:
        x = [-1 if b in v.ablated else +1 for b in ABLATABLE_BLOCKS]
        y = 0.5 + 0.1 * x[0] + 0.05 * x[5] + rng.normal(0, 0.001)
        _write_variant(
            method_dir,
            v.variant_id,
            average_total=y,
            average_gains={
                "recommended_task_coverage": y,
                "user_preference_deviation": y,
            },
            augment_tokens=1000.0 + 100.0 * x[0],
            augment_wall=10.0,
            augment_usd=0.01,
        )
    return method_dir


# ---------------------------------------------------------------------------
# Disk readers
# ---------------------------------------------------------------------------


class TestDiskReader:
    def test_read_variants_index_round_trip(self, tmp_path):
        variants = build_design("leave_one_out")
        method_dir = tmp_path / "scenarios" / "s" / "m"
        _write_variants_index(method_dir, variants)
        pairs = _read_variants_index(method_dir)
        assert [vid for vid, _ in pairs] == [v.variant_id for v in variants]

    def test_missing_index_returns_empty(self, tmp_path):
        assert _read_variants_index(tmp_path) == []

    def test_unparseable_index_returns_empty(self, tmp_path):
        (tmp_path / "ablation").mkdir()
        (tmp_path / "ablation" / "variants.json").write_text("not json")
        assert _read_variants_index(tmp_path) == []

    def test_read_one_variant_missing_sidecar_returns_none(self, tmp_path):
        method_dir = tmp_path / "scenarios" / "s" / "m"
        method_dir.mkdir(parents=True)
        assert _read_one_variant(method_dir, "0" * 11, frozenset()) is None

    def test_read_one_variant_unparseable_total_returns_none(self, tmp_path):
        method_dir = tmp_path / "scenarios" / "s" / "m"
        eval_dir = method_dir / "ablation" / ("0" * 11) / "evaluation"
        eval_dir.mkdir(parents=True)
        (eval_dir / "total_scheduling_gain.json").write_text("garbled")
        assert _read_one_variant(method_dir, "0" * 11, frozenset()) is None

    def test_read_one_variant_pulls_responses_and_cost(self, tmp_path):
        method_dir = tmp_path / "scenarios" / "s" / "m"
        method_dir.mkdir(parents=True)
        _write_variant(
            method_dir,
            "0" * 11,
            average_total=0.5,
            average_gains={"recommended_task_coverage": 0.8},
            augment_tokens=123.0,
            augment_wall=4.0,
            augment_usd=0.001,
        )
        rec = _read_one_variant(method_dir, "0" * 11, frozenset())
        assert rec is not None
        assert rec.responses["total"] == 0.5
        assert rec.responses["G_cov"] == 0.8
        assert rec.augment_tokens == 123.0
        assert rec.augment_wall_seconds == 4.0
        assert rec.augment_usd == 0.001

    def test_read_ablation_records_filters_missing_variants(self, tmp_path):
        method_dir = tmp_path / "scenarios" / "s" / "m"
        method_dir.mkdir(parents=True)
        variants = build_design("leave_one_out")[:3]
        _write_variants_index(method_dir, variants)
        # Write only the first variant's sidecars; the other two
        # are silently skipped.
        _write_variant(method_dir, variants[0].variant_id, average_total=0.5)
        records = read_ablation_records(method_dir)
        assert [r.variant_id for r in records] == [variants[0].variant_id]

    def test_telemetry_absent_leaves_cost_fields_none(self, tmp_path):
        method_dir = tmp_path / "scenarios" / "s" / "m"
        method_dir.mkdir(parents=True)
        _write_variant(method_dir, "0" * 11, average_total=0.5)
        rec = _read_one_variant(method_dir, "0" * 11, frozenset())
        assert rec is not None
        # Remove the telemetry file to simulate the no-telemetry case.
        (
            method_dir / "ablation" / ("0" * 11) / "evaluation" / "telemetry.json"
        ).unlink()
        rec = _read_one_variant(method_dir, "0" * 11, frozenset())
        assert rec is not None
        assert rec.augment_tokens is None
        assert rec.augment_wall_seconds is None
        assert rec.augment_usd is None


# ---------------------------------------------------------------------------
# _as_float
# ---------------------------------------------------------------------------


class TestAsFloat:
    def test_none_passthrough(self):
        assert _as_float(None) is None

    def test_int_coerced(self):
        assert _as_float(3) == 3.0

    def test_float_passthrough(self):
        assert _as_float(0.5) == 0.5

    def test_nan_normalised_to_none(self):
        assert _as_float(float("nan")) is None

    def test_garbage_returns_none(self):
        assert _as_float("not a number") is None


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    def test_shape(self, tmp_path):
        method_dir = _build_pb24_tree(tmp_path)
        records = read_ablation_records(method_dir)
        x = _design_matrix(records)
        assert x.shape == (len(records), len(ABLATABLE_BLOCKS))

    def test_baseline_row_all_plus_one(self):
        rec = VariantRecord(
            variant_id="0" * 11,
            ablated=frozenset(),
            responses={"total": 0.5},
            augment_tokens=None,
            augment_wall_seconds=None,
            augment_usd=None,
        )
        x = _design_matrix([rec])
        assert (x[0] == 1.0).all()

    def test_fully_ablated_row_all_minus_one(self):
        rec = VariantRecord(
            variant_id="1" * 11,
            ablated=frozenset(ABLATABLE_BLOCKS),
            responses={"total": 0.5},
            augment_tokens=None,
            augment_wall_seconds=None,
            augment_usd=None,
        )
        x = _design_matrix([rec])
        assert (x[0] == -1.0).all()


# ---------------------------------------------------------------------------
# OLS fit
# ---------------------------------------------------------------------------


class TestOLSFit:
    def test_fit_recovers_planted_effects(self, tmp_path):
        method_dir = _build_pb24_tree(tmp_path)
        records = read_ablation_records(method_dir)
        fits = fit_all_responses(records)
        assert "total" in fits
        total_fit = fits["total"]
        by_block = {e.block: e for e in total_fit.effects}
        # Planted: block 0 at 0.1, block 5 at 0.05, rest near 0.
        assert by_block[ABLATABLE_BLOCKS[0]].beta == pytest.approx(0.1, abs=0.01)
        assert by_block[ABLATABLE_BLOCKS[5]].beta == pytest.approx(0.05, abs=0.01)
        for j, block in enumerate(ABLATABLE_BLOCKS):
            if j in (0, 5):
                continue
            assert abs(by_block[block].beta) < 0.01

    def test_fit_returns_standard_errors(self, tmp_path):
        method_dir = _build_pb24_tree(tmp_path)
        records = read_ablation_records(method_dir)
        fits = fit_all_responses(records)
        for effect in fits["total"].effects:
            assert effect.se >= 0.0

    def test_all_constant_columns_returns_none(self):
        """No column varies, so the fit bails out."""
        recs = [
            VariantRecord(
                variant_id="0" * 11,
                ablated=frozenset(),
                responses={"total": 0.5},
                augment_tokens=None,
                augment_wall_seconds=None,
                augment_usd=None,
            )
            for _ in range(20)
        ]
        out = fit_all_responses(recs)
        assert "total" not in out

    def test_partially_constant_column_zeroed_in_output(self, tmp_path):
        """Columns with no variance surface as `beta = 0, se = 0`."""
        # PB-24 design, but skip variants that ablate the first block,
        # so block 0's column is identically `+1`.
        method_dir = tmp_path / "scenarios" / "scen" / "llm_agent"
        method_dir.mkdir(parents=True, exist_ok=True)
        variants = [
            v
            for v in build_design("plackett_burman_24")
            if ABLATABLE_BLOCKS[0] not in v.ablated
        ]
        _write_variants_index(method_dir, variants)
        for v in variants:
            _write_variant(
                method_dir,
                v.variant_id,
                average_total=0.5,
                augment_wall=1.0,
                augment_tokens=10.0,
                augment_usd=0.0,
            )
        records = read_ablation_records(method_dir)
        fits = fit_all_responses(records)
        if "total" in fits:
            by_block = {e.block: e for e in fits["total"].effects}
            assert by_block[ABLATABLE_BLOCKS[0]].beta == 0.0
            assert by_block[ABLATABLE_BLOCKS[0]].se == 0.0


class TestDiskReaderEdgeCases:
    def test_variants_index_with_non_string_id_skipped(self, tmp_path):
        """Non-string variant ids are skipped, not raised."""
        method_dir = tmp_path / "scenarios" / "s" / "m"
        (method_dir / "ablation").mkdir(parents=True)
        (method_dir / "ablation" / "variants.json").write_text(
            json.dumps(
                {
                    "variants": [
                        {"variant_id": 1234, "ablated": []},
                        {"variant_id": "0" * 11, "ablated": []},
                    ]
                }
            )
        )
        pairs = _read_variants_index(method_dir)
        assert [vid for vid, _ in pairs] == ["0" * 11]

    def test_fit_for_response_with_no_signal_returns_none(self):
        """All-None response values get skipped instead of fit."""
        recs = [
            VariantRecord(
                variant_id="0" * 11,
                ablated=frozenset(),
                responses={"total": None},
                augment_tokens=None,
                augment_wall_seconds=None,
                augment_usd=None,
            )
        ]
        out = fit_all_responses(recs)
        assert "total" not in out

    def test_too_few_runs_returns_none(self, tmp_path):
        """Fewer rows than coefficients leaves the fit underdetermined."""
        method_dir = tmp_path / "scenarios" / "s" / "m"
        method_dir.mkdir(parents=True)
        variants = build_design("leave_one_out")[:2]
        _write_variants_index(method_dir, variants)
        for v in variants:
            _write_variant(method_dir, v.variant_id, average_total=0.5)
        records = read_ablation_records(method_dir)
        out = fit_all_responses(records)
        assert "total" not in out

    def test_empty_records_returns_empty_dict(self):
        assert fit_all_responses([]) == {}

    def test_cost_response_recovered(self, tmp_path):
        method_dir = _build_pb24_tree(tmp_path)
        records = read_ablation_records(method_dir)
        fits = fit_all_responses(records)
        # Cost response is `augment_tokens = 1000 + 100 * x_0`.
        token_fit = fits["augment_tokens"]
        block0_eff = next(
            e for e in token_fit.effects if e.block == ABLATABLE_BLOCKS[0]
        )
        assert block0_eff.beta == pytest.approx(100.0, abs=1e-6)


# ---------------------------------------------------------------------------
# aggregate_ablation
# ---------------------------------------------------------------------------


class TestAggregateAblation:
    def test_round_trip(self, tmp_path):
        _build_pb24_tree(tmp_path)
        result = aggregate_ablation(
            experiment_dir=tmp_path,
            scenario_id="scen",
            method="llm_agent",
            design_label="plackett_burman_24",
        )
        assert result is not None
        assert result.scenario_id == "scen"
        assert result.method == "llm_agent"
        assert result.design_label == "plackett_burman_24"
        assert len(result.records) > 11
        assert "total" in result.fits

    def test_missing_method_dir_returns_none(self, tmp_path):
        assert aggregate_ablation(tmp_path, "missing", "llm_agent", "single") is None


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_block_effect_frozen(self):
        e = BlockEffect(block="x", beta=0.1, se=0.02)
        with pytest.raises(Exception):
            e.beta = 0.0  # type: ignore[misc]

    def test_ablation_fit_carries_expected_fields(self):
        fit = AblationFit(
            response="total",
            intercept=0.5,
            effects=[BlockEffect(block="a", beta=0.1, se=0.01)],
            n_runs=24,
            residual_std=0.001,
        )
        assert fit.n_runs == 24
        assert fit.effects[0].block == "a"


# ---------------------------------------------------------------------------
# aliased-column drop scan inside _fit_one_response
# ---------------------------------------------------------------------------


class TestAliasedColumnDropping:
    """Aliased columns must drop out so the OLS does not crash on rank deficiency."""

    def test_aliased_and_sign_flipped_columns_drop_to_zero(self) -> None:
        """Two columns identical or sign-flipped of column 0 surface as zero betas."""
        from src.scripts.scenarios.export.ablation_analysis import _fit_one_response

        n_blocks = len(ABLATABLE_BLOCKS)
        n_rows = n_blocks + 4
        rng = np.random.default_rng(0)
        base = rng.choice([-1.0, 1.0], size=(n_rows, n_blocks))
        base[:, 1] = base[:, 0]
        base[:, 2] = -base[:, 0]
        y_raw = list(rng.normal(size=n_rows))

        fit = _fit_one_response("total", y_raw, base)
        assert fit is not None
        beta_by_block = {eff.block: eff.beta for eff in fit.effects}
        assert beta_by_block[ABLATABLE_BLOCKS[1]] == 0.0
        assert beta_by_block[ABLATABLE_BLOCKS[2]] == 0.0

    def test_three_way_alias_walks_already_dropped_columns(self) -> None:
        """Three columns aliased to column 0 force the inner loop to skip dropped entries."""
        from src.scripts.scenarios.export.ablation_analysis import _fit_one_response

        n_blocks = len(ABLATABLE_BLOCKS)
        n_rows = n_blocks + 4
        rng = np.random.default_rng(1)
        base = rng.choice([-1.0, 1.0], size=(n_rows, n_blocks))
        base[:, 1] = base[:, 0]
        base[:, 2] = base[:, 0]
        base[:, 3] = -base[:, 0]
        y_raw = list(rng.normal(size=n_rows))

        fit = _fit_one_response("total", y_raw, base)
        assert fit is not None
        beta_by_block = {eff.block: eff.beta for eff in fit.effects}
        for idx in (1, 2, 3):
            assert beta_by_block[ABLATABLE_BLOCKS[idx]] == 0.0

    def test_constant_column_skipped_by_alias_scan(self) -> None:
        """A constant factor column is detected up front and skipped during alias scan."""
        from src.scripts.scenarios.export.ablation_analysis import _fit_one_response

        n_blocks = len(ABLATABLE_BLOCKS)
        n_rows = n_blocks + 4
        rng = np.random.default_rng(42)
        base = rng.choice([-1.0, 1.0], size=(n_rows, n_blocks))
        # Pin column 5 to a constant so the up-front variance check sets
        # keep_cols[6] to False and the alias scan must skip that column.
        base[:, 5] = 1.0
        y_raw = list(rng.normal(size=n_rows))

        fit = _fit_one_response("total", y_raw, base)
        assert fit is not None
        beta_by_block = {eff.block: eff.beta for eff in fit.effects}
        assert beta_by_block[ABLATABLE_BLOCKS[5]] == 0.0
