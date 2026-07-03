"""Unit tests for src.scripts.persona.solver.retry."""

from __future__ import annotations

import pytest
import z3

from src.scripts.persona.solver.retry import solve_with_retry


def _sat_builder(value: int = 1):
    """Build factory: returns a fresh sat solver with one int var pinned to `value`."""

    def build(_attempt: int):
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        x = z3.Int("x", ctx=ctx)
        s.add(x == value)
        return s, {"event": [(x, x)]}

    return build


def _unsat_builder():
    def build(_attempt: int):
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        x = z3.Int("x", ctx=ctx)
        s.add(x == 1)
        s.add(x == 2)
        return s, {"event": [(x, x)]}

    return build


def test_returns_first_sat_model_on_attempt_zero():
    out = solve_with_retry(_sat_builder(value=7))
    assert out is not None
    model, event_vars = out
    start_var, _ = event_vars["event"][0]
    assert model[start_var].as_long() == 7


def test_returns_none_when_every_attempt_unsat():
    assert solve_with_retry(_unsat_builder(), max_attempts=3) is None


def test_returns_none_when_accept_rejects_all():
    out = solve_with_retry(_sat_builder(), accept=lambda *_: False, max_attempts=3)
    assert out is None


def test_advances_past_unsat_attempts():
    """First two attempts unsat, third sat: helper returns the third's model."""
    calls: list[int] = []

    def build(attempt: int):
        calls.append(attempt)
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        x = z3.Int("x", ctx=ctx)
        if attempt < 2:
            s.add(x == 1)
            s.add(x == 2)  # unsat
        else:
            s.add(x == 99)
        return s, {"event": [(x, x)]}

    out = solve_with_retry(build, max_attempts=5)
    assert out is not None
    assert calls == [0, 1, 2]
    model, vars_ = out
    assert model[vars_["event"][0][0]].as_long() == 99


def test_accept_receives_model_and_event_vars():
    captured = {}

    def accept(model, event_vars):
        captured["model"] = model
        captured["event_vars"] = event_vars
        return True

    out = solve_with_retry(_sat_builder(), accept=accept)
    assert out is not None
    assert "model" in captured
    assert captured["event_vars"] == out[1]


def test_accept_rejection_continues_to_next_attempt():
    attempts: list[int] = []

    def build(attempt: int):
        attempts.append(attempt)
        ctx = z3.Context()
        s = z3.Solver(ctx=ctx)
        x = z3.Int(f"x{attempt}", ctx=ctx)
        s.add(x == attempt)
        return s, {"event": [(x, x)]}

    def accept(model, event_vars):
        # Accept only when start equals 2.
        return model[event_vars["event"][0][0]].as_long() == 2

    out = solve_with_retry(build, accept=accept, max_attempts=5)
    assert out is not None
    assert attempts == [0, 1, 2]


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError, match="max_attempts"):
        solve_with_retry(_sat_builder(), max_attempts=0)


def test_default_accept_passes_through():
    """The default `accept` accepts everything sat."""
    out = solve_with_retry(_sat_builder(value=42))
    assert out is not None
    _, event_vars = out
    assert event_vars["event"]
