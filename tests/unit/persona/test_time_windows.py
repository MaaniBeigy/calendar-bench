"""Unit tests for src.scripts.persona.domain.time_windows."""

from __future__ import annotations

import pytest

from src.scripts.persona.config.schema import WindowRange
from src.scripts.persona.domain.time_windows import (
    NAMED_WINDOWS,
    WindowMap,
    parse_time_token,
)


def test_parse_exact_time():
    a = parse_time_token("07:30")
    assert a.kind == "exact"
    assert a.minutes == 7 * 60 + 30
    assert a.token is None


def test_parse_named_window():
    a = parse_time_token("morning")
    assert a.kind == "window"
    assert a.token == "morning"
    assert a.minutes is None


def test_parse_strips_whitespace():
    a = parse_time_token("  morning  ")
    assert a.kind == "window"
    assert a.raw == "morning"


def test_parse_unknown_raises():
    """`after_dinner`/`after_lunch` are not valid tokens any more - the
    grammar is HH:MM or named window only."""
    with pytest.raises(ValueError):
        parse_time_token("whenever")
    with pytest.raises(ValueError):
        parse_time_token("after_dinner")
    with pytest.raises(ValueError):
        parse_time_token("after_lunch")


def test_named_window_set_matches_constant():
    assert NAMED_WINDOWS == frozenset(
        {"early_morning", "morning", "afternoon", "evening", "night"}
    )


def test_window_map_from_config():
    ranges = {
        "morning": WindowRange(start=400, end=600),
        "evening": WindowRange(start=960, end=1260),
    }
    wm = WindowMap.from_config(ranges)
    assert wm.get("morning") == (400, 600)
    assert "evening" in wm
    assert "night" not in wm


def test_window_map_unknown_key_raises():
    wm = WindowMap.from_config({"morning": WindowRange(start=400, end=600)})
    with pytest.raises(KeyError):
        wm.get("evening")
