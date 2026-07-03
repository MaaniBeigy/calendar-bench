"""Season-walk arithmetic in `_scale_indices`."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.constraints.extract import _scale_indices


def test_season_walk_skips_year_bump_when_season_index_not_spring():
    """Spring to summer: one season elapsed, same year, no year bump."""
    horizon_start = _dt.date(2026, 3, 1)
    season_idx, total = _scale_indices(
        "season",
        day_idx=92,
        total_days=100,
        horizon_start_date=horizon_start,
    )
    assert season_idx == 1
    assert total == 2


def test_season_walk_bumps_year_when_season_index_returns_to_spring():
    """Winter to next spring rolls the year forward."""
    horizon_start = _dt.date(2025, 12, 1)
    season_idx, total = _scale_indices(
        "season",
        day_idx=120,
        total_days=150,
        horizon_start_date=horizon_start,
    )
    assert season_idx == 1
    assert total == 2


def test_season_walk_handles_multi_season_horizon():
    """Winter to spring (year bump) then spring to summer (no bump)."""
    horizon_start = _dt.date(2025, 12, 1)
    season_idx, total = _scale_indices(
        "season",
        day_idx=200,
        total_days=210,
        horizon_start_date=horizon_start,
    )
    assert season_idx == 2
    assert total == 3


def test_season_falls_back_to_bucket_without_start_date():
    """Missing horizon date triggers the 90-day bucket approximation."""
    season_idx, total = _scale_indices(
        "season",
        day_idx=180,
        total_days=365,
        horizon_start_date=None,
    )
    assert season_idx == 2
    assert total == 5
