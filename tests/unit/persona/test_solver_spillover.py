"""Unit tests for src.scripts.persona.solver.spillover."""

from __future__ import annotations

import z3

from src.scripts.persona.domain.schedule import Spillover
from src.scripts.persona.solver.spillover import (
    extract_spillovers,
    occupied_from_spillovers,
)


def _solve(constraints: list) -> tuple[z3.ModelRef, dict]:
    """Tiny helper: build start/duration vars, fix them, return model + event_vars."""
    ctx = z3.Context()
    s = z3.Solver(ctx=ctx)
    event_vars: dict[str, list[tuple[z3.ArithRef, z3.ArithRef]]] = {}
    for name, pairs in constraints:
        bound = []
        for idx, (start_val, dur_val) in enumerate(pairs):
            start = z3.Int(f"{name}_s_{idx}", ctx=ctx)
            dur = z3.Int(f"{name}_d_{idx}", ctx=ctx)
            s.add(start == start_val)
            s.add(dur == dur_val)
            bound.append((start, dur))
        event_vars[name] = bound
    assert s.check() == z3.sat
    return s.model(), event_vars


def test_no_spillover_when_event_ends_at_day_boundary():
    model, ev = _solve([("sleep", [(1380, 60)])])  # ends at 1440 exactly
    assert extract_spillovers(model, ev) == []


def test_no_spillover_when_event_ends_before_midnight():
    model, ev = _solve([("dinner", [(1100, 60)])])
    assert extract_spillovers(model, ev) == []


def test_spillover_carries_overflow_to_next_day():
    model, ev = _solve([("sleep", [(1380, 120)])])  # 23:00 + 120m = 01:00 next day
    out = extract_spillovers(model, ev)
    assert out == [
        Spillover(
            event_name="sleep",
            start=0,
            duration=60,
            orig_start=1380,
            orig_duration=120,
            event_idx=0,
        )
    ]


def test_multiple_events_only_overflowing_ones_returned():
    model, ev = _solve(
        [
            ("sleep", [(1380, 120)]),
            ("dinner", [(1100, 60)]),
        ]
    )
    out = extract_spillovers(model, ev)
    assert len(out) == 1
    assert out[0].event_name == "sleep"


def test_spillover_event_index_preserves_position():
    model, ev = _solve(
        [
            ("study", [(1380, 30), (1380, 120)]),  # 2nd one spills
        ]
    )
    out = extract_spillovers(model, ev)
    assert len(out) == 1
    assert out[0].event_idx == 1


def test_extract_spillovers_handles_empty_event_vars():
    ctx = z3.Context()
    s = z3.Solver(ctx=ctx)
    s.check()
    assert extract_spillovers(s.model(), {}) == []


def test_custom_day_minutes_changes_threshold():
    model, ev = _solve([("nap", [(700, 100)])])
    # With a 720-minute "day", 700 + 100 = 800 spills 80.
    out = extract_spillovers(model, ev, day_minutes=720)
    assert out == [
        Spillover(
            event_name="nap",
            start=0,
            duration=80,
            orig_start=700,
            orig_duration=100,
            event_idx=0,
        )
    ]


def test_occupied_from_spillovers_pairs_start_and_end():
    s1 = Spillover(
        event_name="sleep",
        start=0,
        duration=60,
        orig_start=1380,
        orig_duration=120,
        event_idx=0,
    )
    s2 = Spillover(
        event_name="study",
        start=0,
        duration=15,
        orig_start=1430,
        orig_duration=25,
        event_idx=0,
    )
    assert occupied_from_spillovers([s1, s2]) == [(0, 60), (0, 15)]


def test_occupied_from_spillovers_drops_zero_duration():
    s = Spillover(
        event_name="x",
        start=0,
        duration=0,
        orig_start=1440,
        orig_duration=0,
        event_idx=0,
    )
    assert occupied_from_spillovers([s]) == []


def test_occupied_from_spillovers_handles_empty_list():
    assert occupied_from_spillovers([]) == []
