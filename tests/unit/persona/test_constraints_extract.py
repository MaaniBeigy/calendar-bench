"""Unit tests for src.scripts.persona.constraints.extract."""

from __future__ import annotations

from typing import Any

import pytest

from src.scripts.persona.config.schema import (
    DurationRange,
    EpisodeRange,
    EventDefinition,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.constraints.extract import (
    EventDayConstraints,
    get_event_constraints,
)
from src.scripts.persona.domain.time_windows import WindowMap


def _wm() -> WindowMap:
    return WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        }
    )


def _event(
    *,
    name: str = "thing",
    category: str = "general",
    per_event: tuple[int, int, str] = (10, 30, "minutes"),
    total: tuple[int, int, str, str] = (10, 60, "day", "minutes"),
    episodes: tuple[int, int, str] = (1, 2, "day"),
    patterns: list[TemporalPattern] | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category=category,
        per_event_duration=DurationRange(
            min=per_event[0], max=per_event[1], unit=per_event[2]
        ),
        total_event_duration=TotalDuration(
            min=total[0], max=total[1], scale=total[2], unit=total[3]
        ),
        total_event_episodes=EpisodeRange(
            scale=episodes[2], min=episodes[0], max=episodes[1]
        ),
        temporal_patterns=patterns or [],
    )


def _pattern(mode: str, **details: Any) -> TemporalPattern:
    return TemporalPattern(mode=mode, details=details)


# -------------------------------------------------------------------------------------
# ----------------------------------- baseline shape ----------------------------------
# -------------------------------------------------------------------------------------


def test_no_patterns_returns_midpoint_base_count():
    """min=2, max=6 gives midpoint 4."""
    ev = _event(episodes=(2, 6, "day"))
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert isinstance(out, EventDayConstraints)
    assert out.min_count == 2
    assert out.max_count == 6
    assert out.base_count == 4
    assert out.allowed_windows == ()
    assert out.window_fraction_constraints == ()


def test_hours_unit_converts_to_minutes():
    ev = _event(per_event=(6, 9, "hours"), total=(6, 9, "day", "hours"))
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 360
    assert out.per_event_max == 540
    assert out.total_min == 360
    assert out.total_max == 540


# -------------------------------------------------------------------------------------
# --------------------------------- event_disabled flag -------------------------------
# -------------------------------------------------------------------------------------


def test_event_disabled_flag_false_for_natural_zero_midpoint():
    """min=0, max=1 has midpoint 0 but is NOT disabled by seasonality."""
    ev = _event(episodes=(0, 1, "day"))
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 0
    assert out.event_disabled is False


def test_event_disabled_flag_true_when_weekday_seasonality_excludes_today():
    ev = _event(
        episodes=(1, 2, "day"),
        patterns=[_pattern("seasonality", scale="weekday", amount=100, within=["Mon"])],
    )
    saturday = get_event_constraints(
        ev, day_idx=5, total_days=7, window_map=_wm(), day_of_week=5
    )
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert saturday.event_disabled is True
    assert monday.event_disabled is False


# -------------------------------------------------------------------------------------
# ------------------------------------- mode: fix -------------------------------------
# -------------------------------------------------------------------------------------


def test_fix_with_string_within_in_window_map():
    ev = _event(patterns=[_pattern("fix", within="night")])
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.allowed_windows == ("night",)


def test_fix_with_list_within_only_keeps_known_windows():
    ev = _event(patterns=[_pattern("fix", within=["morning", "Saturday"])])
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.allowed_windows == ("morning",)


def test_fix_with_string_within_unknown_returns_no_window():
    ev = _event(patterns=[_pattern("fix", within="zenith")])
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.allowed_windows == ()


# -------------------------------------------------------------------------------------
# ----------------------- seasonality, scale: weekday, amount: 100 --------------------
# -------------------------------------------------------------------------------------


def test_weekday_full_amount_disables_event_off_weekdays():
    ev = _event(
        episodes=(1, 2, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="weekday",
                amount=100,
                within=["Mon", "Tue", "Wed", "Thu", "Fri"],
            )
        ],
    )
    saturday = get_event_constraints(
        ev, day_idx=5, total_days=7, window_map=_wm(), day_of_week=5
    )
    assert saturday.min_count == 0
    assert saturday.max_count == 0
    assert saturday.base_count == 0


def test_weekday_full_amount_keeps_event_on_weekday():
    ev = _event(
        episodes=(1, 2, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="weekday",
                amount=100,
                within=["Mon", "Tue", "Wed", "Thu", "Fri"],
            )
        ],
    )
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.min_count == 1
    assert monday.max_count == 2
    assert monday.base_count == 1  # midpoint of [1, 2]


def test_weekday_full_amount_accepts_long_form_names():
    """Both 'Mon' and 'Monday' tokens map to weekday index 0."""
    ev = _event(
        episodes=(1, 2, "day"),
        patterns=[
            _pattern(
                "seasonality", scale="weekday", amount=100, within=["Monday", "Tuesday"]
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=2, total_days=7, window_map=_wm(), day_of_week=2
    )
    assert out.base_count == 0  # Wednesday is not in within


# -------------------------------------------------------------------------------------
# ------------------ seasonality, scale: weekday, amount < 100 (boost) ----------------
# -------------------------------------------------------------------------------------


def test_weekday_partial_amount_increases_inside_within():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="weekday",
                amount=50,
                direction="increasing",
                within=["Mon"],
            )
        ],
    )
    # Midpoint 5, on Monday gets * 1.5 = 7.5 to rounds to 8.
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.base_count == 8


def test_weekday_partial_amount_decreases_inside_within():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="weekday",
                amount=40,
                direction="decreasing",
                within=["Mon"],
            )
        ],
    )
    # Midpoint 5, on Monday gets * 0.6 = 3.
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.base_count == 3


def test_weekday_partial_amount_outside_within_unchanged():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="weekday",
                amount=50,
                direction="increasing",
                within=["Mon"],
            )
        ],
    )
    tuesday = get_event_constraints(
        ev, day_idx=1, total_days=7, window_map=_wm(), day_of_week=1
    )
    assert tuesday.base_count == 5


# -------------------------------------------------------------------------------------
# --------------------------- seasonality, weekend boost arm --------------------------
# -------------------------------------------------------------------------------------


def test_weekend_seasonality_increases_on_saturday():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                amount=20,
                direction="increasing",
                within=["Saturday", "Sunday"],
            )
        ],
    )
    saturday = get_event_constraints(
        ev, day_idx=5, total_days=7, window_map=_wm(), day_of_week=5
    )
    # Midpoint 5 * 1.2 = 6.
    assert saturday.base_count == 6


def test_weekend_seasonality_skipped_on_weekday():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                amount=20,
                direction="increasing",
                within=["Saturday", "Sunday"],
            )
        ],
    )
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.base_count == 5


def test_weekend_string_token_seasonality_increases_on_sunday():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern("seasonality", amount=10, direction="increasing", within="weekend")
        ],
    )
    sunday = get_event_constraints(
        ev, day_idx=6, total_days=7, window_map=_wm(), day_of_week=6
    )
    assert sunday.base_count == 6  # 5 * 1.10 = 5.5 to round to 6


def test_weekend_string_token_skipped_on_weekday():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern("seasonality", amount=10, direction="increasing", within="weekend")
        ],
    )
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.base_count == 5


def test_weekend_seasonality_decreases_when_direction_decreasing():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern("seasonality", amount=40, direction="decreasing", within="weekend")
        ],
    )
    saturday = get_event_constraints(
        ev, day_idx=5, total_days=7, window_map=_wm(), day_of_week=5
    )
    # 5 * 0.6 = 3.
    assert saturday.base_count == 3


def test_saturday_only_seasonality_inactive_on_sunday():
    """Legacy: 'Saturday' string on Sunday leaves base_count unchanged."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality", amount=20, direction="increasing", within="Saturday"
            )
        ],
    )
    sunday = get_event_constraints(
        ev, day_idx=6, total_days=7, window_map=_wm(), day_of_week=6
    )
    assert sunday.base_count == 5


def test_sunday_only_seasonality_active_on_sunday():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern("seasonality", amount=20, direction="increasing", within="Sunday")
        ],
    )
    sunday = get_event_constraints(
        ev, day_idx=6, total_days=7, window_map=_wm(), day_of_week=6
    )
    assert sunday.base_count == 6  # 5 * 1.2


# -------------------------------------------------------------------------------------
# ------------------------ seasonality, window-fraction baseline ----------------------
# -------------------------------------------------------------------------------------


def test_window_fraction_with_amount_100_pins_boost_window_only():
    ev = _event(
        patterns=[_pattern("seasonality", amount=100, within=["morning"])],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    fractions = dict(out.window_fraction_constraints)
    assert fractions["morning"] == 1.0
    assert fractions["evening"] == 0.0


def test_window_fraction_with_partial_boost_distributes_baseline():
    ev = _event(
        patterns=[_pattern("seasonality", amount=20, within=["morning"])],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    fractions = dict(out.window_fraction_constraints)
    # 5 named windows, n_baseline=4, n_boosted=1.
    expected_baseline = (1.0 - 0.20) / 5
    expected_boost = expected_baseline + 0.20 / 1
    assert fractions["evening"] == pytest.approx(expected_baseline)
    assert fractions["morning"] == pytest.approx(expected_boost)


def test_window_fraction_uniform_when_no_boost_window_recognised():
    """Within only contains tokens absent from window_map."""
    ev = _event(
        patterns=[_pattern("seasonality", amount=20, within=["unknown_zone"])],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    fractions = dict(out.window_fraction_constraints)
    assert all(v == pytest.approx(0.2) for v in fractions.values())
    assert set(fractions.keys()) == {
        "early_morning",
        "morning",
        "afternoon",
        "evening",
        "night",
    }


def test_window_fraction_normalised_when_total_above_one():
    """Two boost windows at amount=80 each would sum > 1.0; result is rescaled."""
    ev = _event(
        patterns=[
            _pattern("seasonality", amount=80, within=["morning"]),
            _pattern("seasonality", amount=80, within=["evening"]),
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    total = sum(f for _, f in out.window_fraction_constraints)
    assert total == pytest.approx(1.0)


# -------------------------------------------------------------------------------------
# ------------------------------------- mode: trend -----------------------------------
# -------------------------------------------------------------------------------------


def test_trend_increasing_before_start_unchanged():
    ev = _event(
        episodes=(0, 20, "day"),
        patterns=[
            _pattern("trend", direction="increasing", amount=10, start=5, end=10)
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 10  # midpoint, no shift yet


def test_trend_increasing_at_end_full_shift():
    ev = _event(
        episodes=(0, 20, "day"),
        patterns=[_pattern("trend", direction="increasing", amount=10, start=1, end=5)],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=14, window_map=_wm(), day_of_week=0
    )
    # Day idx 10 is past end day 5 (0-indexed 4). progress=1.0 to base 10 + 10 = 20.
    assert out.base_count == 20


def test_trend_increasing_midway():
    ev = _event(
        episodes=(0, 20, "day"),
        patterns=[_pattern("trend", direction="increasing", amount=10, start=1, end=5)],
    )
    # trend_start = 0, trend_end = 4. Midpoint of [0..4] is 2 to progress 0.5.
    out = get_event_constraints(
        ev, day_idx=2, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 15


def test_trend_decreasing_subtracts():
    ev = _event(
        episodes=(0, 20, "day"),
        patterns=[_pattern("trend", direction="decreasing", amount=8, start=1, end=5)],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=14, window_map=_wm(), day_of_week=0
    )
    # Past end to full shift. 10 - 8 = 2.
    assert out.base_count == 2


def test_trend_with_unknown_direction_no_shift():
    """A pattern with neither increasing nor decreasing leaves base unchanged."""
    ev = _event(
        episodes=(0, 20, "day"),
        patterns=[_pattern("trend", direction="sideways", amount=5, start=1, end=5)],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 10


def test_trend_with_end_le_start_is_noop():
    ev = _event(
        episodes=(0, 20, "day"),
        patterns=[
            _pattern("trend", direction="increasing", amount=5, start=10, end=10)
        ],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 10


# -------------------------------------------------------------------------------------
# --------------------------- composition / clamping behaviour ------------------------
# -------------------------------------------------------------------------------------


def test_base_count_clamped_to_max():
    ev = _event(
        episodes=(0, 5, "day"),
        patterns=[
            _pattern("trend", direction="increasing", amount=100, start=1, end=2)
        ],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 5  # clamped to max


def test_base_count_clamped_to_min():
    ev = _event(
        episodes=(2, 5, "day"),
        patterns=[
            _pattern("trend", direction="decreasing", amount=100, start=1, end=2)
        ],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 2  # clamped to min


def test_fix_with_empty_details_returns_no_window():
    """A fix pattern lacking 'within' adds nothing."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[TemporalPattern(mode="fix", details={})],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 5
    assert out.allowed_windows == ()


def test_weekday_seasonality_with_no_within_disables_event_when_amount_100():
    """Empty within at amount=100 means the day is never 'in' the allowed set."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[_pattern("seasonality", scale="weekday", amount=100)],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 0
    assert out.min_count == 0
    assert out.max_count == 0


def test_window_seasonality_with_no_within_returns_uniform_fractions():
    """No within and amount<100 results in uniform allocation across all windows."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[_pattern("seasonality", amount=20)],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    fractions = dict(out.window_fraction_constraints)
    assert all(v == pytest.approx(0.2) for v in fractions.values())


# -------------------------------------------------------------------------------------
# ----------------------- string-form within (single weekday name) --------------------
# -------------------------------------------------------------------------------------


def test_weekday_seasonality_with_string_within_active_on_match():
    """A single weekday string in 'within' takes the str arm of _within_indices."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern("seasonality", scale="weekday", amount=100, within="Monday")
        ],
    )
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.base_count == 5  # midpoint, day in within

    tuesday = get_event_constraints(
        ev, day_idx=1, total_days=7, window_map=_wm(), day_of_week=1
    )
    assert tuesday.base_count == 0  # day not in within, amount=100 to disabled


def test_weekday_seasonality_with_unrecognized_string_disables_at_amount_100():
    """A within string that isn't a weekday name returns no allowed indices."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern("seasonality", scale="weekday", amount=100, within="Funday")
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 0


# -------------------------------------------------------------------------------------
# ------------- window-scope seasonality with non-day-token string within -------------
# -------------------------------------------------------------------------------------


def test_window_seasonality_with_string_window_name_falls_through_to_fractions():
    """A within string that isn't 'weekend'/Saturday/Sunday hits the fall-through arm."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[_pattern("seasonality", amount=40, within="morning")],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    fractions = dict(out.window_fraction_constraints)
    # boosted=["morning"], baseline=4 other windows.
    expected_baseline = (1.0 - 0.40) / 5
    expected_boost = expected_baseline + 0.40 / 1
    assert fractions["morning"] == pytest.approx(expected_boost)
    assert fractions["evening"] == pytest.approx(expected_baseline)


# -------------------------------------------------------------------------------------
# --------------------------- empty window map (n_allowed == 0) -----------------------
# -------------------------------------------------------------------------------------


def test_window_fraction_with_empty_window_map_returns_no_fractions():
    """An empty WindowMap leaves _window_fraction_constraints with nothing to allocate."""
    from src.scripts.persona.domain.time_windows import WindowMap as _WM

    empty_wm = _WM(ranges={})
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[_pattern("seasonality", amount=20, within=["unknown_zone"])],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=empty_wm, day_of_week=0
    )
    assert out.window_fraction_constraints == ()


# -------------------------------------------------------------------------------------
# ----------------------- target: duration; trend ------------------------------------
# -------------------------------------------------------------------------------------


def test_duration_trend_increasing_shifts_bounds_at_end():
    """At day 27 (end of 28-day trend, amount=20) delta==20 min."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=20,
                start=1,
                end=28,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=27, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 80
    assert out.per_event_max == 110
    assert out.total_min == 80
    assert out.total_max == 110


def test_duration_trend_increasing_zero_at_start():
    """At day 0 the trend has not kicked in yet: delta==0."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=20,
                start=1,
                end=28,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 60
    assert out.per_event_max == 90


def test_duration_trend_midpoint_adds_half_amount():
    """At the midpoint (day 13 of start=1, end=28) delta≈10 min."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=20,
                start=1,
                end=28,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=13, total_days=28, window_map=_wm(), day_of_week=0
    )
    # progress = (13 - 0) / (27 - 0) = 13/27 ≈ 0.481 to delta ≈ 9.6 to round to 10
    assert out.per_event_min == 70
    assert out.per_event_max == 100


def test_duration_trend_decreasing_subtracts_from_bounds():
    """Decreasing duration trend reduces per_event bounds."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="decreasing",
                amount=30,
                start=1,
                end=10,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=20, total_days=28, window_map=_wm(), day_of_week=0
    )
    # past end to full delta = -30; per_event_min = max(0, 60-30) = 30
    assert out.per_event_min == 30
    assert out.per_event_max == 60


def test_duration_trend_clamps_min_at_zero():
    """A very large decreasing trend does not push per_event_min below 0."""
    ev = _event(
        per_event=(20, 40, "minutes"),
        total=(20, 40, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="decreasing",
                amount=100,
                start=1,
                end=5,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 0
    assert out.per_event_max == 0  # max(0, 40-100)=0 then max(per_event_min,0)=0


def test_duration_trend_hours_unit_then_delta():
    """Duration trend on an hours-based event: delta is in minutes."""
    ev = _event(
        per_event=(1, 2, "hours"),
        total=(1, 2, "day", "hours"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=30,
                start=1,
                end=5,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=28, window_map=_wm(), day_of_week=0
    )
    # base: min=60, max=120 (hourstominutes). delta=+30 min at end.
    assert out.per_event_min == 90
    assert out.per_event_max == 150


def test_duration_trend_end_le_start_is_noop():
    """A trend with end <= start leaves duration unchanged."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=20,
                start=5,
                end=5,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 60
    assert out.per_event_max == 90


def test_duration_trend_does_not_affect_base_count():
    """target=duration must leave base_count (episode count) unchanged."""
    ev = _event(
        episodes=(0, 4, "day"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=30,
                start=1,
                end=28,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=27, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 2  # midpoint of [0, 4], unaffected by duration trend


# -------------------------------------------------------------------------------------
# --------------------- target: duration; seasonality (weekend) ----------------------
# -------------------------------------------------------------------------------------


def test_duration_seasonality_weekend_increases_on_saturday():
    """+10 min on weekends (Saturday) via seasonality target=duration."""
    ev = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                within=["Saturday", "Sunday"],
                direction="increasing",
                amount=10,
            )
        ],
    )
    saturday = get_event_constraints(
        ev, day_idx=5, total_days=7, window_map=_wm(), day_of_week=5
    )
    assert saturday.per_event_min == 40
    assert saturday.per_event_max == 70


def test_duration_seasonality_weekend_no_change_on_weekday():
    """Weekday: duration unchanged for a weekend-only duration seasonality."""
    ev = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                within=["Saturday", "Sunday"],
                direction="increasing",
                amount=10,
            )
        ],
    )
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.per_event_min == 30
    assert monday.per_event_max == 60


def test_duration_seasonality_weekend_string_token():
    """'weekend' string token triggers on Sunday."""
    ev = _event(
        per_event=(45, 90, "minutes"),
        total=(45, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                within="weekend",
                direction="increasing",
                amount=15,
            )
        ],
    )
    sunday = get_event_constraints(
        ev, day_idx=6, total_days=7, window_map=_wm(), day_of_week=6
    )
    assert sunday.per_event_min == 60
    assert sunday.per_event_max == 105


def test_duration_seasonality_decreasing_on_weekend():
    """Decreasing direction subtracts minutes on weekends."""
    ev = _event(
        per_event=(60, 120, "minutes"),
        total=(60, 120, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                within="weekend",
                direction="decreasing",
                amount=20,
            )
        ],
    )
    saturday = get_event_constraints(
        ev, day_idx=5, total_days=7, window_map=_wm(), day_of_week=5
    )
    assert saturday.per_event_min == 40
    assert saturday.per_event_max == 100


def test_duration_seasonality_saturday_string_inactive_on_sunday():
    """String form 'Saturday' (not in a list) does not fire on Sunday.

    A `within` list like `["Saturday"]` is treated as "any weekend token"
    by the legacy `_classify_window_within` logic and fires on both Sat and
    Sun.  Only the bare string `"Saturday"` is Saturday-specific.
    """
    ev = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                within="Saturday",  # bare string to Saturday-only
                direction="increasing",
                amount=10,
            )
        ],
    )
    sunday = get_event_constraints(
        ev, day_idx=6, total_days=7, window_map=_wm(), day_of_week=6
    )
    assert sunday.per_event_min == 30  # unchanged on Sunday
    assert sunday.per_event_max == 60


# -------------------------------------------------------------------------------------
# ---------------------- target: duration; seasonality (weekday) ---------------------
# -------------------------------------------------------------------------------------


def test_duration_seasonality_weekday_scale_fires_on_match():
    """scale=weekday, within=Mon: +15 min on Mondays."""
    ev = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                scale="weekday",
                within=["Mon"],
                direction="increasing",
                amount=15,
            )
        ],
    )
    monday = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert monday.per_event_min == 45
    assert monday.per_event_max == 75


def test_duration_seasonality_weekday_scale_no_change_off_match():
    """scale=weekday, within=Mon: Tuesday is unchanged."""
    ev = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                scale="weekday",
                within=["Mon"],
                direction="increasing",
                amount=15,
            )
        ],
    )
    tuesday = get_event_constraints(
        ev, day_idx=1, total_days=7, window_map=_wm(), day_of_week=1
    )
    assert tuesday.per_event_min == 30
    assert tuesday.per_event_max == 60


def test_duration_seasonality_weekday_does_not_affect_episode_count():
    """target=duration weekday seasonality must not change base_count."""
    ev = _event(
        episodes=(0, 6, "day"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                scale="weekday",
                within=["Mon"],
                direction="increasing",
                amount=20,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 3  # midpoint of [0, 6], unaffected


# -------------------------------------------------------------------------------------
# --------------------- target: duration; combined trend + seasonality ---------------
# -------------------------------------------------------------------------------------


def test_duration_trend_and_seasonality_accumulate():
    """trend (+20 at end) + weekend seasonality (+10) adds 30 min on last Saturday."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=20,
                start=1,
                end=28,
            ),
            _pattern(
                "seasonality",
                target="duration",
                within="weekend",
                direction="increasing",
                amount=10,
            ),
        ],
    )
    # day_idx=26 is day 27 (last Saturday of 4-week run).
    # trend progress ≈ 26/27 ≈ 0.963 to delta ≈ 19.3 to round 19.
    # weekend seasonality: +10. Total delta = 29.
    out = get_event_constraints(
        ev, day_idx=26, total_days=28, window_map=_wm(), day_of_week=5  # Saturday
    )
    assert out.per_event_min == 60 + 29
    assert out.per_event_max == 90 + 29


def test_duration_trend_before_start_day_is_noop():
    """day_idx < trend start: delta == 0, bounds unchanged."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=20,
                start=10,
                end=28,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=5, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 60
    assert out.per_event_max == 90


def test_duration_trend_unknown_direction_is_noop():
    """An unrecognised direction leaves duration unchanged (neither incr nor decr)."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="sideways",
                amount=20,
                start=1,
                end=10,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=10, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 60
    assert out.per_event_max == 90


def test_duration_window_scope_seasonality_target_duration_no_effect_on_weekday():
    """Window-scope seasonality (within=morning) with target=duration: no delta.

    Window names are time-of-day buckets, not day-type conditions; the
    implementation deliberately returns 0 for them under target=duration.
    """
    ev = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                within="morning",
                direction="increasing",
                amount=15,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.per_event_min == 30
    assert out.per_event_max == 60


# -------------------------------------------------------------------------------------
# --------------------- new scales + explicit unit field ------------------------------
# -------------------------------------------------------------------------------------

import datetime as _dt  # noqa: E402


def test_week_indexed_trend_ramps_episode_count_per_week():
    """`scale: week, start: 1, end: 4, amount: 3` to +3 episodes by end of week 4."""
    ev = _event(
        episodes=(0, 10, "week"),
        patterns=[
            _pattern(
                "trend",
                scale="week",
                direction="increasing",
                amount=3,
                start=1,
                end=4,
            )
        ],
    )
    # Day 0 (week 0) to progress 0.0 to base_count = midpoint(0,10)=5
    early = get_event_constraints(
        ev, day_idx=0, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert early.base_count == 5
    # Day 21 (week 3) to progress 1.0 to 5 + 3 = 8
    late = get_event_constraints(
        ev, day_idx=21, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert late.base_count == 8
    # Day 7 (week 1) to progress 1/3 to 5 + 1 = 6
    mid = get_event_constraints(
        ev, day_idx=7, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert mid.base_count == 6


def test_month_indexed_trend_with_horizon_start_date_uses_calendar_arithmetic():
    """`scale: month` walks the calendar; January start to month idx 0."""
    ev = _event(
        episodes=(0, 10, "month"),
        patterns=[
            _pattern(
                "trend",
                scale="month",
                direction="increasing",
                amount=4,
                start=1,
                end=4,  # 4-month ramp
            )
        ],
    )
    horizon = _dt.date(2026, 1, 15)  # mid-January
    # Day 0 to month 0 (Jan) to progress 0 to base_count = 5
    jan = get_event_constraints(
        ev,
        day_idx=0,
        total_days=120,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert jan.base_count == 5
    # Day 90 to April to month 3 to progress 1.0 to 5 + 4 = 9
    apr = get_event_constraints(
        ev,
        day_idx=90,
        total_days=120,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert apr.base_count == 9


def test_month_seasonality_within_may_june_boosts_only_in_those_months():
    """`scale: month, within: [May, June], amount: 30` to +30% during May/June."""
    ev = _event(
        episodes=(0, 10, "day"),  # base = midpoint = 5
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                within=["May", "June"],
                amount=30,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    # April (day 100) to no boost
    april = get_event_constraints(
        ev,
        day_idx=100,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert april.base_count == 5
    # May 15 (day 134) to 5 × 1.3 = 6.5 to banker's round to 6
    may = get_event_constraints(
        ev,
        day_idx=134,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert may.base_count == 6


def test_season_seasonality_within_spring_boosts_only_in_spring():
    """`scale: season, within: [spring], amount: 30` to +30% in March-May."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="season",
                within=["spring"],
                amount=30,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    # January (winter) to no boost
    jan = get_event_constraints(
        ev,
        day_idx=10,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert jan.base_count == 5
    # April (spring) to 5 * 1.3 = 6.5 to banker's round to 6
    apr = get_event_constraints(
        ev,
        day_idx=100,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert apr.base_count == 6


def test_explicit_unit_hours_on_duration_trend_multiplies_amount_by_60():
    """`unit: hours, amount: 1, scale: week, start: 1, end: 4` to +60 min by week 4."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                scale="week",
                unit="hours",
                direction="increasing",
                amount=1,
                start=1,
                end=4,
            )
        ],
    )
    # Day 0 (week 0) to progress 0 to +0 min
    early = get_event_constraints(
        ev, day_idx=0, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert early.per_event_min == 60
    assert early.per_event_max == 90
    # Day 21 (week 3) to progress 1.0 to +60 min
    late = get_event_constraints(
        ev, day_idx=21, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert late.per_event_min == 120
    assert late.per_event_max == 150


def test_explicit_unit_minutes_matches_legacy_target_duration():
    """`unit: minutes` semantics must equal `target: duration` (legacy)."""
    ev_unit = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                unit="minutes",
                direction="increasing",
                amount=20,
                start=1,
                end=28,
            )
        ],
    )
    ev_target = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                direction="increasing",
                amount=20,
                start=1,
                end=28,
            )
        ],
    )
    out_unit = get_event_constraints(
        ev_unit, day_idx=14, total_days=28, window_map=_wm(), day_of_week=0
    )
    out_target = get_event_constraints(
        ev_target, day_idx=14, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert out_unit.per_event_min == out_target.per_event_min
    assert out_unit.per_event_max == out_target.per_event_max


def test_explicit_unit_count_episodes_is_absolute():
    """`unit: count` applies an absolute episode delta."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "trend",
                target="episodes",
                unit="count",
                direction="increasing",
                amount=4,
                start=1,
                end=10,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=9, total_days=10, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 9  # 5 + 4 at full progress


def test_incompatible_target_unit_is_rejected():
    """A duration unit on target: episodes fails schema validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _pattern(
            "trend",
            target="episodes",
            unit="minutes",
            direction="increasing",
            amount=20,
            start=1,
            end=10,
        )


def test_count_episode_seasonality_adds_absolute_on_matching_weekday():
    """`target: episodes, unit: count` adds an absolute count on matching days."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                target="episodes",
                unit="count",
                scale="weekday",
                within=["Tue"],
                direction="increasing",
                amount=2,
            )
        ],
    )
    on = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=1
    )
    off = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert on.base_count == 7  # midpoint 5 + 2
    assert off.base_count == 5


def test_count_episode_seasonality_decreasing_subtracts():
    """`unit: count, direction: decreasing` subtracts an absolute count."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                target="episodes",
                unit="count",
                scale="weekday",
                within=["Tue"],
                direction="decreasing",
                amount=2,
            )
        ],
    )
    on = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=1
    )
    assert on.base_count == 3  # midpoint 5 - 2


def test_count_seasonality_weekend_tokens_without_scale():
    """`unit: count` with weekend tokens (no scale) adds on Sat/Sun only."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                target="episodes",
                unit="count",
                within=["Sat", "Sun"],
                direction="increasing",
                amount=3,
            )
        ],
    )
    sat = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=5
    )
    mon = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert sat.base_count == 8
    assert mon.base_count == 5


def test_percent_episode_seasonality_scales_on_matching_weekday():
    """`target: episodes, unit: percent` scales the count by amount%."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                target="episodes",
                unit="percent",
                scale="weekday",
                within=["Tue"],
                direction="increasing",
                amount=40,
            )
        ],
    )
    on = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=1
    )
    assert on.base_count == 7  # round(5 * 1.4)


def test_percent_duration_seasonality_scales_bounds():
    """`target: duration, unit: percent` scales the duration bounds by amount%."""
    ev = _event(
        per_event=(100, 200, "minutes"),
        total=(100, 200, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                unit="percent",
                scale="weekday",
                within=["Tue"],
                direction="increasing",
                amount=50,
            )
        ],
    )
    on = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=1
    )
    off = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert (on.per_event_min, on.per_event_max) == (150, 300)
    assert (off.per_event_min, off.per_event_max) == (100, 200)


def test_percent_duration_trend_scales_bounds_at_full_progress():
    """`trend, target: duration, unit: percent` ramps a multiplicative factor."""
    ev = _event(
        per_event=(100, 200, "minutes"),
        total=(100, 200, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                unit="percent",
                scale="week",
                direction="increasing",
                amount=50,
                start=1,
                end=2,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=7, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert (out.per_event_min, out.per_event_max) == (150, 300)


def test_percent_episode_trend_scales_at_full_progress():
    """`trend, target: episodes, unit: percent` ramps the count by amount%."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "trend",
                target="episodes",
                unit="percent",
                scale="week",
                direction="increasing",
                amount=40,
                start=1,
                end=2,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=7, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 7  # round(5 * 1.4)


def test_percent_episode_trend_decreasing():
    """`trend, target: episodes, unit: percent, decreasing` shrinks the count."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "trend",
                target="episodes",
                unit="percent",
                scale="week",
                direction="decreasing",
                amount=40,
                start=1,
                end=2,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=7, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 3  # round(5 * 0.6)


def test_percent_duration_trend_decreasing():
    """`trend, target: duration, unit: percent, decreasing` shrinks the bounds."""
    ev = _event(
        per_event=(100, 200, "minutes"),
        total=(100, 200, "day", "minutes"),
        patterns=[
            _pattern(
                "trend",
                target="duration",
                unit="percent",
                scale="week",
                direction="decreasing",
                amount=50,
                start=1,
                end=2,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=7, total_days=14, window_map=_wm(), day_of_week=0
    )
    assert (out.per_event_min, out.per_event_max) == (50, 100)


def test_percent_duration_seasonality_month_scale():
    """`seasonality, target: duration, unit: percent, scale: month`."""
    ev = _event(
        per_event=(100, 200, "minutes"),
        total=(100, 200, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                unit="percent",
                scale="month",
                within=["January"],
                direction="increasing",
                amount=50,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert (out.per_event_min, out.per_event_max) == (150, 300)


def test_percent_duration_seasonality_season_scale():
    """`seasonality, target: duration, unit: percent, scale: season`."""
    ev = _event(
        per_event=(100, 200, "minutes"),
        total=(100, 200, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                unit="percent",
                scale="season",
                within=["winter"],
                direction="increasing",
                amount=50,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert (out.per_event_min, out.per_event_max) == (150, 300)


def test_percent_duration_seasonality_weekend_no_scale():
    """`seasonality, target: duration, unit: percent` with weekend tokens, no scale."""
    ev = _event(
        per_event=(100, 200, "minutes"),
        total=(100, 200, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                target="duration",
                unit="percent",
                within=["Sat", "Sun"],
                direction="increasing",
                amount=50,
            )
        ],
    )
    sat = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=5
    )
    mon = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert (sat.per_event_min, sat.per_event_max) == (150, 300)
    assert (mon.per_event_min, mon.per_event_max) == (100, 200)


def test_percent_trend_flat_before_start_saturated_after_end():
    """Percent trend ramp: progress 0 before `start`, 1 after `end`."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "trend",
                target="episodes",
                unit="percent",
                scale="week",
                direction="increasing",
                amount=40,
                start=2,
                end=3,
            )
        ],
    )
    before = get_event_constraints(
        ev, day_idx=0, total_days=28, window_map=_wm(), day_of_week=0
    )
    after = get_event_constraints(
        ev, day_idx=27, total_days=28, window_map=_wm(), day_of_week=0
    )
    assert before.base_count == 5  # week 0 < start: progress 0
    assert after.base_count == 7  # week 3 > end: round(5 * 1.4)


def test_count_month_seasonality_adds_absolute():
    """`scale: month, unit: count` adds an absolute count in matching months."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                target="episodes",
                unit="count",
                scale="month",
                within=["January"],
                direction="increasing",
                amount=2,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=7, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 7  # day 0 defaults to January; midpoint 5 + 2


def test_unknown_unit_is_rejected():
    """An unrecognised unit fails validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _pattern("seasonality", target="episodes", unit="furlongs", amount=1)


def test_percent_without_target_is_rejected():
    """`unit: percent` without an explicit target fails validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _pattern(
            "seasonality",
            unit="percent",
            scale="weekday",
            within=["Tue"],
            amount=10,
        )


def test_month_seasonality_amount_100_disables_outside_months():
    """`scale: month, within: [June], amount: 100` to episodes 0 outside June."""
    ev = _event(
        episodes=(2, 5, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                within=["June"],
                amount=100,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    # March to outside to disabled
    march = get_event_constraints(
        ev,
        day_idx=60,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert march.event_disabled is True
    assert march.base_count == 0
    # June to inside to base preserved
    june = get_event_constraints(
        ev,
        day_idx=160,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert june.event_disabled is False
    assert june.base_count >= 2


def test_season_seasonality_amount_100_disables_outside_seasons():
    """`scale: season, within: [winter], amount: 100` to episodes 0 outside winter."""
    ev = _event(
        episodes=(2, 5, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="season",
                within=["winter"],
                amount=100,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    # Jan to winter to ok
    jan = get_event_constraints(
        ev,
        day_idx=5,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert jan.event_disabled is False
    # April to spring to disabled
    apr = get_event_constraints(
        ev,
        day_idx=100,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert apr.event_disabled is True


def test_month_seasonality_no_horizon_date_falls_back_to_jan_anchor():
    """Without `horizon_start_date` the extractor treats day 0 as Jan 1."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                within=["January"],
                amount=50,
                direction="increasing",
            )
        ],
    )
    # Day 5 with no horizon start to calendar month 1 (Jan) to boosted
    out = get_event_constraints(
        ev, day_idx=5, total_days=30, window_map=_wm(), day_of_week=0
    )
    # base 5 * 1.5 = 7.5 to round to 8 (clamped to 10)
    assert out.base_count == 8


def test_seasonality_short_month_names_supported():
    """`Jun` / `Jul` short names map to the same calendar months as full names."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                within=["Jun"],
                amount=20,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    june = get_event_constraints(
        ev,
        day_idx=160,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    # 5 * 1.2 = 6.0 to 6
    assert june.base_count == 6


def test_season_within_unknown_name_is_silently_skipped():
    """An unknown season name in `within` short-circuits to no-op."""
    ev = _event(
        episodes=(2, 4, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="season",
                within=["nonsense"],
                amount=50,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    out = get_event_constraints(
        ev,
        day_idx=0,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert out.base_count == 3  # midpoint preserved


def test_duration_seasonality_month_scope_with_horizon_date():
    """`scale: month, unit: minutes` shifts duration only inside the named months."""
    ev = _event(
        per_event=(30, 60, "minutes"),
        total=(30, 60, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                unit="minutes",
                within=["June"],
                amount=15,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    # June 10 to boosted
    june = get_event_constraints(
        ev,
        day_idx=160,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert june.per_event_min == 45
    assert june.per_event_max == 75
    # April to not boosted
    apr = get_event_constraints(
        ev,
        day_idx=100,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert apr.per_event_min == 30
    assert apr.per_event_max == 60


def test_duration_seasonality_season_scope_with_horizon_date():
    """`scale: season, unit: hours` boosts duration only in the named season."""
    ev = _event(
        per_event=(60, 90, "minutes"),
        total=(60, 90, "day", "minutes"),
        patterns=[
            _pattern(
                "seasonality",
                scale="season",
                unit="hours",
                within=["summer"],
                amount=1,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    # July (summer) to +60 min
    jul = get_event_constraints(
        ev,
        day_idx=181,
        total_days=365,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert jul.per_event_min == 120
    assert jul.per_event_max == 150
    # January (winter) to no shift
    jan = get_event_constraints(
        ev,
        day_idx=5,
        total_days=365,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert jan.per_event_min == 60
    assert jan.per_event_max == 90


def test_total_days_one_handles_single_day_horizon():
    """A 1-day horizon must not divide by zero in any scale indexer."""
    ev = _event(
        episodes=(1, 1, "day"),
        patterns=[
            _pattern(
                "trend",
                scale="month",
                direction="increasing",
                amount=1,
                start=1,
                end=1,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=0, total_days=1, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 1


def test_unknown_scale_falls_back_to_day_indexed():
    """Unknown scales (e.g. `foo`) fall back to day-indexed ramp."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "trend",
                scale="foo",
                direction="increasing",
                amount=2,
                start=1,
                end=10,
            )
        ],
    )
    out = get_event_constraints(
        ev, day_idx=9, total_days=10, window_map=_wm(), day_of_week=0
    )
    assert out.base_count == 7  # base 5 + full progress 2


def test_horizon_start_date_with_winter_wrap_around_for_season_index():
    """A horizon that starts in Dec and lasts into Feb computes season indices correctly."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "trend",
                scale="season",
                direction="increasing",
                amount=2,
                start=1,
                end=2,
            )
        ],
    )
    horizon = _dt.date(2026, 12, 1)
    # Dec 1 to winter (season 4) to season-of-horizon 0
    dec = get_event_constraints(
        ev,
        day_idx=0,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert dec.base_count == 5
    # March (90 days later) to spring to season-of-horizon 1 to progress 1.0 to 5+2=7
    march = get_event_constraints(
        ev,
        day_idx=95,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert march.base_count == 7


def test_month_seasonality_decreasing_direction():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                within=["August"],
                amount=40,
                direction="decreasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    aug = get_event_constraints(
        ev,
        day_idx=220,
        total_days=365,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert aug.base_count == 3  # 5 * 0.6


def test_season_seasonality_decreasing_direction():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="season",
                within=["autumn"],
                amount=40,
                direction="decreasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    oct_ = get_event_constraints(
        ev,
        day_idx=280,
        total_days=365,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert oct_.base_count == 3


def test_seasonality_month_single_string_within():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                within="May",
                amount=40,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    may = get_event_constraints(
        ev,
        day_idx=134,
        total_days=180,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert may.base_count == 7  # 5 * 1.4


def test_seasonality_season_single_string_within():
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="season",
                within="summer",
                amount=40,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    jul = get_event_constraints(
        ev,
        day_idx=185,
        total_days=365,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert jul.base_count == 7  # 5 * 1.4


def test_season_trend_without_horizon_date_uses_90_day_buckets():
    """Without `horizon_start_date` the `season` scale falls back to 90-day buckets."""
    ev = _event(
        episodes=(0, 10, "day"),
        patterns=[
            _pattern(
                "trend",
                scale="season",
                direction="increasing",
                amount=4,
                start=1,
                end=2,
            )
        ],
    )
    early = get_event_constraints(
        ev, day_idx=0, total_days=180, window_map=_wm(), day_of_week=0
    )
    assert early.base_count == 5
    later = get_event_constraints(
        ev, day_idx=90, total_days=180, window_map=_wm(), day_of_week=0
    )
    assert later.base_count == 9


def test_seasonality_month_within_nonstring_silently_skipped():
    ev = _event(
        episodes=(2, 4, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="month",
                within=42,  # invalid type to empty allowed to no-op
                amount=50,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    out = get_event_constraints(
        ev,
        day_idx=10,
        total_days=30,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert out.base_count == 3


def test_seasonality_season_within_nonstring_silently_skipped():
    ev = _event(
        episodes=(2, 4, "day"),
        patterns=[
            _pattern(
                "seasonality",
                scale="season",
                within=42,
                amount=50,
                direction="increasing",
            )
        ],
    )
    horizon = _dt.date(2026, 1, 1)
    out = get_event_constraints(
        ev,
        day_idx=10,
        total_days=30,
        window_map=_wm(),
        day_of_week=0,
        horizon_start_date=horizon,
    )
    assert out.base_count == 3


def test_calendar_month_to_season_returns_autumn_index():
    """Sept-Nov maps to the autumn season index (3)."""
    from src.scripts.persona.constraints.extract import _calendar_month_to_season

    assert _calendar_month_to_season(9) == 3
    assert _calendar_month_to_season(10) == 3
    assert _calendar_month_to_season(11) == 3
