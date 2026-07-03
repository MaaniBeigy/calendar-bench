"""Unit tests for src.scripts.persona.constraints.windows."""

from __future__ import annotations

import pytest

from src.scripts.persona.config.schema import WindowRange
from src.scripts.persona.constraints.windows import resolve_window_starts
from src.scripts.persona.domain.time_windows import WindowMap


def _wm() -> WindowMap:
    return WindowMap.from_config(
        {
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "night": WindowRange(start=1260, end=1440),
        }
    )


def test_resolve_single_window_default_step():
    starts = resolve_window_starts(["morning"], _wm())
    assert min(starts) == 400
    assert max(starts) == 590
    assert 600 not in starts  # range is half-open
    # Step is 10 by default.
    assert all(s % 10 == 0 for s in starts)
    assert len(starts) == 20  # (600 - 400) / 10


def test_resolve_multiple_windows():
    starts = resolve_window_starts(["morning", "night"], _wm())
    assert 400 in starts
    assert 1260 in starts
    assert 600 not in starts
    assert 1440 not in starts


def test_resolve_skips_unknown_window_names():
    """Day-name and weekend tokens listed alongside windows are silently skipped."""
    starts = resolve_window_starts(["Saturday", "morning", "weekend"], _wm())
    assert 400 in starts
    # Saturday and weekend produce no minutes.
    assert min(starts) == 400


def test_resolve_empty_iterable_returns_empty_set():
    assert resolve_window_starts([], _wm()) == set()


def test_resolve_custom_step_minutes():
    starts = resolve_window_starts(["morning"], _wm(), step_minutes=30)
    assert starts == {400, 430, 460, 490, 520, 550, 580}


def test_resolve_rejects_zero_step():
    with pytest.raises(ValueError, match="step_minutes"):
        resolve_window_starts(["morning"], _wm(), step_minutes=0)


def test_resolve_rejects_negative_step():
    with pytest.raises(ValueError, match="step_minutes"):
        resolve_window_starts(["morning"], _wm(), step_minutes=-5)
