"""Tests for the distribution-aware weekly-gain statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.export.report_writer import (
    CONVERGENCE_TAU,
    ONSET_TAU,
    _interquartile_mean,
    _onset_week,
    _percentile,
    _person_series,
    aggregate_weekly_gain_stats,
    cohort_learning_split,
    ols_slope,
    per_person_trajectory,
    weekly_gain_distribution,
)


def _rows(gains: list[float | None], start_week: int = 1) -> list[dict]:
    return [
        {
            "week_index": start_week + i,
            "week_start": f"2026-06-{1 + 7 * i:02d}",
            "weighted_gain": g,
        }
        for i, g in enumerate(gains)
    ]


# ---------------------------------------------------------------------------
# Percentile and IQM
# ---------------------------------------------------------------------------


def test_percentile_single_value():
    assert _percentile([0.5], 0.5) == 0.5


def test_percentile_linear_interpolation():
    assert _percentile([0, 1, 2, 3], 0.5) == pytest.approx(1.5)
    assert _percentile([0, 1, 2, 3], 0.25) == pytest.approx(0.75)
    assert _percentile([0, 1, 2, 3], 1.0) == pytest.approx(3.0)


def test_iqm_small_sample_uses_all():
    assert _interquartile_mean([1.0, 3.0]) == pytest.approx(2.0)


def test_iqm_trims_quartiles():
    assert _interquartile_mean([0, 0, 1, 2, 3, 4, 9, 9]) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# OLS slope
# ---------------------------------------------------------------------------


def test_ols_slope_rising_positive():
    assert ols_slope([1, 2, 3], [0.1, 0.2, 0.3]) > 0


def test_ols_slope_flat_zero():
    assert ols_slope([1, 2, 3], [0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_ols_slope_declining_negative():
    assert ols_slope([1, 2, 3], [0.3, 0.2, 0.1]) < 0


def test_ols_slope_single_point_zero():
    assert ols_slope([1], [0.5]) == 0.0


def test_ols_slope_zero_variance_x_zero():
    assert ols_slope([2, 2], [0.1, 0.9]) == 0.0


# ---------------------------------------------------------------------------
# Per-person series and onset
# ---------------------------------------------------------------------------


def test_person_series_sorts_and_drops_none():
    rows = [
        {"week_index": 3, "weighted_gain": 0.3},
        {"week_index": 1, "weighted_gain": 0.1},
        {"week_index": 2, "weighted_gain": None},
    ]
    weeks, gains = _person_series(rows)
    assert weeks == [1, 3]
    assert gains == [0.1, 0.3]


def test_onset_week_none_when_too_short():
    assert _onset_week([1], [0.5], ONSET_TAU) is None


def test_onset_week_detects_sustained_rise():
    assert _onset_week([1, 2, 3], [0.5, 0.6, 0.7], 0.05) == 2


def test_onset_week_none_when_not_sustained():
    assert _onset_week([1, 2, 3], [0.5, 0.7, 0.5], 0.05) is None


def test_onset_week_late_onset():
    assert _onset_week([1, 2, 3], [0.5, 0.52, 0.7], 0.05) == 3


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def test_weekly_distribution_skips_none_and_aggregates():
    by_person = {"a": _rows([0.1, 0.2, 0.3]), "b": _rows([0.3, 0.4, 0.5])}
    by_person["a"][1]["weighted_gain"] = None
    dist = weekly_gain_distribution(by_person)
    assert [d["week_index"] for d in dist] == [1, 2, 3]
    wk2 = next(d for d in dist if d["week_index"] == 2)
    assert wk2["scored_persons"] == 1
    assert "median_weighted_gain" in wk2
    assert "iqm_weighted_gain" in wk2
    assert "p25_weighted_gain" in wk2


def test_trajectory_fields_and_convergence():
    by_person = {
        "up": _rows([0.4, 0.45, 0.6, 0.7]),
        "flat": _rows([0.5, 0.5, 0.5, 0.5]),
        "empty": [{"week_index": 1, "weighted_gain": None}],
    }
    traj = per_person_trajectory(by_person)
    assert "empty" not in traj
    assert traj["up"]["converged"] is True
    assert traj["up"]["slope"] > 0
    assert traj["up"]["onset_week"] is not None
    assert traj["flat"]["converged"] is False


def test_cohort_split_counts():
    by_person = {
        "up": _rows([0.3, 0.35, 0.5, 0.6]),
        "flat": _rows([0.5, 0.5, 0.5, 0.5]),
        "down": _rows([0.7, 0.65, 0.5, 0.4]),
    }
    split = cohort_learning_split(per_person_trajectory(by_person))
    assert split == {"converged": 1, "flat": 1, "declined": 1, "n": 3}


def test_aggregate_stats_reads_persons_dir(tmp_path: Path):
    persons = tmp_path / "persons"
    persons.mkdir()
    for pid, gains in [("p0", [0.3, 0.4, 0.6, 0.7]), ("p1", [0.5, 0.5, 0.5, 0.5])]:
        (persons / f"{pid}_weekly_gain.json").write_text(
            json.dumps({"person_id": pid, "weeks": _rows(gains)})
        )
    stats = aggregate_weekly_gain_stats(persons)
    assert len(stats["weeks"]) == 4
    assert set(stats["trajectory"]) == {"p0", "p1"}
    assert stats["split"]["n"] == 2
    assert stats["split"]["converged"] == 1


def test_constants_default():
    assert ONSET_TAU == 0.05
    assert CONVERGENCE_TAU == 0.05
