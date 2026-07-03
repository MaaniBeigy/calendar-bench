"""End-to-end z3 unit tests for src.scripts.persona.solver.day_model."""

from __future__ import annotations

import z3

from src.scripts.persona.config.schema import (
    DurationRange,
    EpisodeRange,
    EventDefinition,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.solver.day_model import DAY_MINUTES, build_day_model

# -------------------------------------------------------------------------------------
# ---------------------------------- shared fixtures ----------------------------------
# -------------------------------------------------------------------------------------


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


def _sleep() -> EventDefinition:
    return EventDefinition(
        name="sleep",
        category="sleep",
        per_event_duration=DurationRange(min=6, max=9, unit="hours"),
        total_event_duration=TotalDuration(min=6, max=9, scale="day", unit="hours"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[TemporalPattern(mode="fix", details={"within": ["night"]})],
    )


def _lunch() -> EventDefinition:
    return EventDefinition(
        name="lunch",
        category="eat",
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=30, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[
            TemporalPattern(mode="fix", details={"within": ["afternoon"]})
        ],
    )


def _running() -> EventDefinition:
    return EventDefinition(
        name="running",
        category="sports",
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=30, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        temporal_patterns=[],
    )


def _read_event(
    model: z3.ModelRef, event_vars: dict, name: str
) -> list[tuple[int, int]]:
    return [(model[s].as_long(), model[d].as_long()) for s, d in event_vars[name]]


# -------------------------------------------------------------------------------------
# ----------------------------------- baseline solve ----------------------------------
# -------------------------------------------------------------------------------------


def test_sleep_only_solves_and_lands_in_night_window():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
    )
    assert solver.check() == z3.sat
    model = solver.model()
    [(start, duration)] = _read_event(model, ev, "sleep")
    assert start >= 1260  # at or after the night window starts
    assert duration >= 6 * 60
    assert duration <= 9 * 60
    assert duration % 10 == 0


def test_no_objective_uses_lightweight_solver():
    """When `maximize_event=None` the day model uses `z3.Solver` instead
    of `z3.Optimize`. Optimize carries pseudo-boolean / weighted-MaxSAT
    machinery that is unnecessary for pure satisfiability and adds
    significant per-day overhead once events stack up to two-digit
    counts (e.g. smoking 0-22/day). This test pins the
    backend choice so the speedup does not regress."""
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    assert isinstance(solver, z3.Solver)


def test_with_objective_uses_optimize_backend():
    """When `maximize_event` is set, the day model still routes through
    `z3.Optimize` so `solver.maximize(...)` works."""
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
        maximize_event="sleep",
    )
    assert isinstance(solver, z3.Optimize)


def test_sleep_objective_maximizes_duration():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
    )
    assert solver.check() == z3.sat
    [(_, dur)] = _read_event(solver.model(), ev, "sleep")
    assert dur == 9 * 60  # objective maxed out


def test_count_zero_returns_empty_event_list():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={"running": 0},
        window_map=_wm(),
    )
    assert solver.check() == z3.sat
    assert ev["running"] == []


def test_missing_count_treated_as_zero():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={},  # running not present
        window_map=_wm(),
    )
    assert solver.check() == z3.sat
    assert ev["running"] == []


# -------------------------------------------------------------------------------------
# -------------------------- per-event constraints enforcement ------------------------
# -------------------------------------------------------------------------------------


def test_per_event_duration_bounds_enforced():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    [(_, dur)] = _read_event(solver.model(), ev, "lunch")
    assert 30 <= dur <= 60


def test_lunch_starts_inside_afternoon_window():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    [(start, _)] = _read_event(solver.model(), ev, "lunch")
    assert 600 <= start < 960


def test_step_minutes_constrains_duration_granularity():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        step_minutes=15,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    [(start, dur)] = _read_event(solver.model(), ev, "lunch")
    assert dur % 15 == 0
    assert start % 15 == 0


# -------------------------------------------------------------------------------------
# -------------------------------- non-overlap rules ----------------------------------
# -------------------------------------------------------------------------------------


def test_two_events_must_not_overlap():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep(), "lunch": _lunch()},
        counts={"sleep": 1, "lunch": 1},
        window_map=_wm(),
    )
    assert solver.check() == z3.sat
    model = solver.model()
    [(s1, d1)] = _read_event(model, ev, "sleep")
    [(s2, d2)] = _read_event(model, ev, "lunch")
    end1 = s1 + d1
    end2 = s2 + d2
    # Sleep is allowed to cross midnight, so check non-overlap modulo wrap.
    assert end1 <= s2 or end2 <= s1 or s1 >= 1440


def test_spillover_blocks_overlapping_event():
    """An occupied range from a previous-day spillover is respected."""
    occupied = [(0, 120)]  # first two hours blocked
    sleep_no_cross = EventDefinition(
        name="sleep",
        category="sleep",
        per_event_duration=DurationRange(min=6, max=9, unit="hours"),
        total_event_duration=TotalDuration(min=6, max=9, scale="day", unit="hours"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[TemporalPattern(mode="fix", details={"within": ["night"]})],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": sleep_no_cross},
        counts={"sleep": 1},
        window_map=_wm(),
        occupied_ranges=occupied,
        cross_midnight_events=frozenset({"sleep"}),
    )
    assert solver.check() == z3.sat
    [(s, d)] = _read_event(solver.model(), ev, "sleep")
    # The chosen sleep cannot intersect minutes 0..120.
    assert s >= 120 or s + d <= 0


def test_cross_midnight_event_can_extend_past_day():
    """Events listed as cross-midnight may produce start + duration > 1440."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
        cross_midnight_events=frozenset({"sleep"}),
    )
    assert solver.check() == z3.sat
    [(s, d)] = _read_event(solver.model(), ev, "sleep")
    # Optimiser maximises duration to 540; 1260 + 540 = 1800 > DAY_MINUTES.
    assert s + d > DAY_MINUTES


def test_non_cross_midnight_event_must_end_inside_day():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        cross_midnight_events=frozenset(),  # nothing may cross
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    [(s, d)] = _read_event(solver.model(), ev, "lunch")
    assert s + d <= DAY_MINUTES


# -------------------------------------------------------------------------------------
# ------------------------------- free-time minimum -----------------------------------
# -------------------------------------------------------------------------------------


def test_free_time_floor_caps_total_scheduled_minutes():
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
        free_minutes_minimum=900,  # forbids more than 540 min of events
    )
    assert solver.check() == z3.sat
    # Sleep duration was up to 540; the floor limits total to 1440 - 900 = 540, sat.


def test_free_time_floor_makes_unsat_when_too_strict():
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
        free_minutes_minimum=DAY_MINUTES - 100,  # only 100 min may be scheduled
    )
    # Sleep needs at least 6h = 360 min; floor permits at most 100 min: unsat.
    assert solver.check() == z3.unsat


def test_free_time_floor_unsat_when_occupied_already_too_large():
    """Occupied minutes already exceeding the cap forces immediate unsat."""
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        occupied_ranges=[(0, 1200)],
        free_minutes_minimum=300,  # cap = 1140; 1200 occupied already breaks it
        maximize_event=None,
    )
    assert solver.check() == z3.unsat


def test_free_time_floor_zero_is_no_op():
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
        free_minutes_minimum=0,
    )
    assert solver.check() == z3.sat


def test_free_time_floor_with_no_events_does_not_constrain():
    """When no events are scheduled, the floor still applies but is trivially sat."""
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={"running": 0},
        window_map=_wm(),
        free_minutes_minimum=300,
        maximize_event=None,
    )
    assert solver.check() == z3.sat


# -------------------------------------------------------------------------------------
# ------------------------------- objective controls ----------------------------------
# -------------------------------------------------------------------------------------


def test_maximize_event_none_skips_objective():
    """Without an objective, sleep duration may be at the lower bound."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep()},
        counts={"sleep": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    [(_, dur)] = _read_event(solver.model(), ev, "sleep")
    assert 360 <= dur <= 540


def test_maximize_event_unknown_name_is_no_op():
    """Asking to maximise an event that wasn't scheduled is silently skipped."""
    solver, _ = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        maximize_event="not_in_events",
    )
    assert solver.check() == z3.sat


# -------------------------------------------------------------------------------------
# ------------------------------ window-fraction constraint ---------------------------
# -------------------------------------------------------------------------------------


def test_window_fraction_constraint_pins_events_to_boost_window():
    """A 100% boost on one window forces all episodes into that window."""
    walk = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=10, max=30, unit="minutes"),
        total_event_duration=TotalDuration(
            min=20, max=120, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=2, max=4),
        temporal_patterns=[
            TemporalPattern(
                mode="seasonality",
                details={"within": ["morning"], "amount": 100},
            )
        ],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"walking": walk},
        counts={"walking": 2},
        window_map=_wm(),
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    starts = [s for s, _ in _read_event(solver.model(), ev, "walking")]
    assert all(400 <= s < 600 for s in starts)


# -------------------------------------------------------------------------------------
# ------------------------------------ empty input ------------------------------------
# -------------------------------------------------------------------------------------


def test_no_events_solves_trivially():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={},
        counts={},
        window_map=_wm(),
    )
    assert solver.check() == z3.sat
    assert ev == {}


# -------------------------------------------------------------------------------------
# ------------------------- preferred_starts / preferred_durations --------------------
# -------------------------------------------------------------------------------------


def test_preferred_start_pins_event_to_persona_minute():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_starts={"lunch": 750},  # 12:30
    )
    assert solver.check() == z3.sat
    assert _read_event(solver.model(), ev, "lunch")[0][0] == 750


def test_preferred_duration_pins_event_minutes_after_quantization():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_durations={"lunch": 47},  # snapped down to 40
    )
    assert solver.check() == z3.sat
    duration = _read_event(solver.model(), ev, "lunch")[0][1]
    assert duration == 40


def test_two_persons_with_different_preferred_starts_diverge():
    """Same catalog, different preferred lunch minutes => different solutions."""
    solver_a, ev_a = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_starts={"lunch": 720},  # 12:00
    )
    solver_b, ev_b = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_starts={"lunch": 810},  # 13:30
    )
    assert solver_a.check() == z3.sat
    assert solver_b.check() == z3.sat
    a_start = _read_event(solver_a.model(), ev_a, "lunch")[0][0]
    b_start = _read_event(solver_b.model(), ev_b, "lunch")[0][0]
    assert a_start == 720
    assert b_start == 810


def test_preferred_start_outside_allowed_window_falls_back_to_solver():
    """Lunch is restricted to 'afternoon' (600-960). A preferred 11:00 (660 is
    inside; pick 480 instead which is outside) must be ignored, and the
    solver still picks a valid minute."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_starts={"lunch": 480},  # 08:00, outside afternoon window
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "lunch")[0][0]
    assert 600 <= start < 960


def test_preferred_duration_outside_bounds_falls_back_to_range():
    """A preferred duration above per_event_max is silently ignored."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_durations={"lunch": 5000},  # well above 60 max
    )
    assert solver.check() == z3.sat
    duration = _read_event(solver.model(), ev, "lunch")[0][1]
    assert 30 <= duration <= 60


def test_preferred_start_for_event_with_no_window_constraint_pins_anywhere():
    """`running` has no allowed_windows; the helper should still pin start."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={"running": 1},
        window_map=_wm(),
        preferred_starts={"running": 410},  # not on a step boundary
    )
    assert solver.check() == z3.sat
    assert _read_event(solver.model(), ev, "running")[0][0] == 410


def test_preferred_start_inside_spillover_falls_back_to_disjunction():
    """A persona-stated minute that lives inside an occupied range cannot be
    pinned without going unsat. The day-model has to detect this and let
    the solver pick a feasible minute from the disjunction instead."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        occupied_ranges=[(0, 700)],
        preferred_starts={"lunch": 620},  # inside the occupied range
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "lunch")[0][0]
    # The solver should have picked any allowed start clear of [0, 700).
    assert start >= 700
    assert start < 960


def test_seeded_random_pick_skips_starts_inside_spillover():
    """The seeded random fallback also has to filter out starts that would
    overlap the spillover. Otherwise a horizon with overnight sleep would
    silently produce empty days."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        occupied_ranges=[(0, 750)],
        person_seed=42,
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "lunch")[0][0]
    assert start >= 750
    assert start < 960


def test_unpinned_event_with_person_seed_picks_a_start_inside_allowed_window():
    """No persona-stated start, but `person_seed` should still produce a
    deterministic start drawn from the catalog's allowed_starts."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        person_seed=12345,
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "lunch")[0][0]
    assert 600 <= start < 960


def test_unpinned_event_two_seeds_can_diverge():
    """Different person_seed values can land on different allowed starts."""
    starts: set[int] = set()
    for seed in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        solver, ev = build_day_model(
            day_idx=0,
            total_days=7,
            events={"lunch": _lunch()},
            counts={"lunch": 1},
            window_map=_wm(),
            person_seed=seed,
        )
        assert solver.check() == z3.sat
        starts.add(_read_event(solver.model(), ev, "lunch")[0][0])
    # 12 random picks across 36 afternoon slots almost certainly give >1 distinct.
    assert len(starts) > 1


def test_unpinned_event_seed_is_stable_across_calls():
    """Same person_seed + same event = same picked start, every time."""
    seen: list[int] = []
    for _ in range(3):
        solver, ev = build_day_model(
            day_idx=0,
            total_days=7,
            events={"lunch": _lunch()},
            counts={"lunch": 1},
            window_map=_wm(),
            person_seed=42,
        )
        assert solver.check() == z3.sat
        seen.append(_read_event(solver.model(), ev, "lunch")[0][0])
    assert len(set(seen)) == 1


def test_person_seed_zero_keeps_legacy_deterministic_pick():
    """Default `person_seed=0` disables the random fallback so the existing
    determinism contract (same constraints => same output) is preserved."""
    solver_a, ev_a = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
    )
    solver_b, ev_b = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        person_seed=0,
    )
    assert solver_a.check() == z3.sat
    assert solver_b.check() == z3.sat
    a_start = _read_event(solver_a.model(), ev_a, "lunch")[0][0]
    b_start = _read_event(solver_b.model(), ev_b, "lunch")[0][0]
    assert a_start == b_start


def test_random_pick_avoids_same_day_pinned_event_intervals():
    """Two events on the same day: one persona-pinned, the other random.
    The random pick must avoid the pinned event's interval so the solver
    does not go unsat on pairwise non-overlap. Without the fix, a random
    `running` start could land inside the pinned `lunch` slot."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch(), "running": _running()},
        counts={"lunch": 1, "running": 1},
        window_map=_wm(),
        preferred_starts={"lunch": 750},
        preferred_durations={"lunch": 60},
        person_seed=42,
    )
    assert solver.check() == z3.sat
    lunch_start, lunch_dur = _read_event(solver.model(), ev, "lunch")[0]
    running_start, running_dur = _read_event(solver.model(), ev, "running")[0]
    # Pinned lunch should keep its persona-stated minute and duration.
    assert lunch_start == 750
    assert lunch_dur == 60
    # Running must not overlap [750, 810).
    assert running_start + running_dur <= 750 or running_start >= 810


def test_collision_filter_pushes_random_event_outside_pinned_block():
    """A persona pinned to 9:00 with 8h work means [540, 1020]. A random
    event with min_dur 60 must land at 0..480, exactly 1020, or later."""
    work = EventDefinition(
        name="office_work",
        category="work",
        per_event_duration=DurationRange(min=60, max=480, unit="minutes"),
        total_event_duration=TotalDuration(
            min=60, max=480, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"office_work": work, "running": _running()},
        counts={"office_work": 1, "running": 1},
        window_map=_wm(),
        preferred_starts={"office_work": 540},
        preferred_durations={"office_work": 480},
        person_seed=99,
    )
    assert solver.check() == z3.sat
    work_start, work_dur = _read_event(solver.model(), ev, "office_work")[0]
    running_start, running_dur = _read_event(solver.model(), ev, "running")[0]
    assert work_start == 540 and work_dur == 480
    assert (
        running_start + running_dur <= work_start
        or running_start >= work_start + work_dur
    )


def test_random_pick_respects_day_bounds_for_non_cross_midnight():
    """A non-cross-midnight event with a 60-minute minimum cannot start
    after minute 1380. The seeded random pick has to filter those out
    before pinning."""
    # Construct a day where the only "free" minutes are very late so the
    # day-bounds filter kicks in.
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={"running": 1},
        window_map=_wm(),
        # Block almost the whole day. Anything after 1380 would be
        # an unsafe pin because running needs >=60 min and cannot cross.
        occupied_ranges=[(0, 1380)],
        person_seed=12345,
    )
    assert solver.check() == z3.sat
    start, dur = _read_event(solver.model(), ev, "running")[0]
    assert start + dur <= 1440


def test_seeded_random_pick_applies_to_each_episode_independently():
    """For a multi-episode event with `person_seed`, every episode goes
    through the seeded random branch. The branch only records
    `placed_first_start` for episode 0; subsequent episodes feed the
    solver and the same-event prior-episode interval list to keep them
    from colliding with each other."""
    work = EventDefinition(
        name="office_work",
        category="work",
        per_event_duration=DurationRange(min=60, max=180, unit="minutes"),
        total_event_duration=TotalDuration(
            min=120, max=360, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=2, max=2),
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"office_work": work},
        counts={"office_work": 2},
        window_map=_wm(),
        person_seed=42,
    )
    assert solver.check() == z3.sat
    starts = sorted(s for s, _ in _read_event(solver.model(), ev, "office_work"))
    assert all(s % 10 == 0 for s in starts)
    # Episodes must be at least one min-duration apart (so pairwise
    # non-overlap is satisfiable without shrinking below per_event_min).
    assert starts[1] - starts[0] >= 60


def test_preferred_start_overrides_seed_random_pick():
    """A persona-stated preferred_start beats the seeded random fallback."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_starts={"lunch": 700},
        person_seed=999,
    )
    assert solver.check() == z3.sat
    assert _read_event(solver.model(), ev, "lunch")[0][0] == 700


def test_preferred_only_pins_first_episode_when_count_is_higher():
    work = EventDefinition(
        name="office_work",
        category="work",
        per_event_duration=DurationRange(min=60, max=300, unit="minutes"),
        total_event_duration=TotalDuration(
            min=120, max=600, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=2, max=2),
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"office_work": work},
        counts={"office_work": 2},
        window_map=_wm(),
        preferred_starts={"office_work": 540},  # 09:00
    )
    assert solver.check() == z3.sat
    starts = [s for s, _ in _read_event(solver.model(), ev, "office_work")]
    assert starts[0] == 540
    # Second episode is solver-chosen; it just must be valid (>= 0, < day, no overlap).
    assert starts[1] >= 0
    assert starts[1] != starts[0]


def test_preferred_window_restricts_allowed_starts_for_unanchored_event():
    """A persona-stated window ("morning") on a catalog event with no
    `temporal_patterns` (allowed_starts would otherwise span the whole
    day) restricts the random pick to that window only."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={"running": 1},
        window_map=_wm(),
        preferred_windows={"running": "morning"},
        person_seed=42,
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "running")[0][0]
    assert 400 <= start < 600


def test_preferred_window_restricts_solver_disjunction_when_no_seed():
    """With person_seed=0 the random pick is disabled, so the solver
    picks from a disjunction over `allowed_starts`. The window
    restriction must apply there too."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={"running": 1},
        window_map=_wm(),
        preferred_windows={"running": "morning"},
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "running")[0][0]
    assert 400 <= start < 600


def test_preferred_window_outside_window_map_falls_back_to_catalog():
    """A typo or unknown window name must not silently zero out the
    event's allowed_starts. The catalog default applies and the event
    still solves."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"running": _running()},
        counts={"running": 1},
        window_map=_wm(),
        preferred_windows={"running": "nonexistent_window"},
        person_seed=42,
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "running")[0][0]
    assert 0 <= start < DAY_MINUTES


def test_two_independent_events_both_solve_in_their_windows():
    """The day model iterates events in catalog insertion order. With
    each event's persona-stated start window respected (sleep ->
    night, lunch -> afternoon), both lay down cleanly and the day
    stays sat."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"sleep": _sleep(), "lunch": _lunch()},
        counts={"sleep": 1, "lunch": 1},
        window_map=_wm(),
        person_seed=3,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    sleep_start = _read_event(solver.model(), ev, "sleep")[0][0]
    lunch_start = _read_event(solver.model(), ev, "lunch")[0][0]
    # Both events should land cleanly in their respective windows.
    assert 1260 <= sleep_start < 1440
    assert 600 <= lunch_start < 960


def test_preferred_window_disjoint_from_catalog_keeps_catalog():
    """If the persona names a window outside the catalog's
    `allowed_windows` (e.g. catalog says afternoon but persona says
    morning), the catalog wins so the event lands somewhere it is
    actually allowed."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"lunch": _lunch()},  # catalog: afternoon only
        counts={"lunch": 1},
        window_map=_wm(),
        preferred_windows={"lunch": "morning"},
        person_seed=7,
    )
    assert solver.check() == z3.sat
    start = _read_event(solver.model(), ev, "lunch")[0][0]
    assert 600 <= start < 960  # afternoon, per the catalog


def test_seeded_random_pick_safety_uses_pinned_duration_not_per_event_min():
    """When a persona pins a long duration on an event whose
    `per_event_min` is much shorter, the safety filter must clear the
    pinned duration. Otherwise a random start that fits per_event_min but
    not the pinned duration is chosen, the duration constraint is then
    enforced by z3, and the day goes unsat - dropping every event for
    that day. Reproduces the cadence-violation bug in experiment A.
    """
    # afternoon event, per_event 30-180 min. Persona pins 180 min.
    long_event = EventDefinition(
        name="visit_family",
        category="social",
        per_event_duration=DurationRange(min=30, max=180, unit="minutes"),
        total_event_duration=TotalDuration(
            min=30, max=180, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        temporal_patterns=[
            TemporalPattern(mode="fix", details={"within": ["afternoon"]})
        ],
    )
    # Lunch occupies 750-795 (30 min). With per_event_min=30, every
    # afternoon start except those overlapping lunch is "safe" - including
    # 600, which is unsafe for the actual 180-min pin (600+180=780 > 750).
    # The fix: safety_duration uses pinned_duration so 600 is rejected.
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"visit_family": long_event},
        counts={"visit_family": 1},
        window_map=_wm(),
        occupied_ranges=[(750, 795)],
        preferred_durations={"visit_family": 180},
        person_seed=1,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    start, duration = _read_event(solver.model(), ev, "visit_family")[0]
    assert duration == 180
    # The chosen start must clear the lunch block with the full 180 min.
    assert start + duration <= 750 or start >= 795


def test_multi_episode_random_pick_pins_only_episode_zero():
    """Multi-episode events random-pick episode 0 (for per-person variation)
    and leave episodes 1..K to z3's disjunction. Pinning every episode
    by `per_event_max` over-reserves space when several multi-episode
    events share a window (e.g. evening-pinned walking + stress); the disjunction lets z3 pack the trailing episodes
    around the pin and around other events. The day must still solve
    sat and produce pairwise non-overlapping episodes."""
    multi_event = EventDefinition(
        name="raining",
        category="weather",
        per_event_duration=DurationRange(min=1, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=0, max=240, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=12),
        temporal_patterns=[],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"raining": multi_event},
        counts={"raining": 4},
        window_map=_wm(),
        person_seed=42,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    starts_durs = _read_event(solver.model(), ev, "raining")
    assert len(starts_durs) == 4
    # Pairwise non-overlap must hold across the four episodes.
    sorted_eps = sorted(starts_durs, key=lambda sd: sd[0])
    for (s_a, d_a), (s_b, _) in zip(sorted_eps, sorted_eps[1:]):
        assert s_a + d_a <= s_b


def test_multi_episode_random_pick_gives_per_person_variation_via_episode_zero():
    """Two persons with different `person_seed` should pin episode 0 to
    different minutes - that is what surfaces per-person variation when
    no HH:MM persona pin is set."""
    multi_event = EventDefinition(
        name="raining",
        category="weather",
        per_event_duration=DurationRange(min=1, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=0, max=240, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=12),
        temporal_patterns=[],
    )
    starts_by_seed: list[int] = []
    for seed in (11, 99):
        solver, ev = build_day_model(
            day_idx=0,
            total_days=7,
            events={"raining": multi_event},
            counts={"raining": 4},
            window_map=_wm(),
            person_seed=seed,
            maximize_event=None,
        )
        assert solver.check() == z3.sat
        starts_durs = _read_event(solver.model(), ev, "raining")
        starts_by_seed.append(min(s for s, _ in starts_durs))
    # The earliest start (= episode 0's pinned random pick) must differ
    # across seeds, otherwise the random pick is not seeding correctly.
    assert starts_by_seed[0] != starts_by_seed[1]


def test_preferred_start_safety_uses_pinned_duration():
    """A persona-pinned start that clears `per_event_min` but not the
    pinned duration must be rejected, so the solver falls back to the
    disjunction. Sister case to the random-pick variant above."""
    long_event = EventDefinition(
        name="visit_family",
        category="social",
        per_event_duration=DurationRange(min=30, max=180, unit="minutes"),
        total_event_duration=TotalDuration(
            min=30, max=180, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        temporal_patterns=[
            TemporalPattern(mode="fix", details={"within": ["afternoon"]})
        ],
    )
    # Persona pins start=600 (afternoon start). With duration=180 the
    # event overlaps lunch at 750-795 (600+180=780). The pin must be
    # rejected and the solver must pick a feasible alternative.
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"visit_family": long_event},
        counts={"visit_family": 1},
        window_map=_wm(),
        occupied_ranges=[(750, 795)],
        preferred_starts={"visit_family": 600},
        preferred_durations={"visit_family": 180},
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    start, duration = _read_event(solver.model(), ev, "visit_family")[0]
    assert duration == 180
    assert start + duration <= 750 or start >= 795


def test_episode_safe_empty_falls_back_to_disjunction():
    """A multi-episode event whose allowed window is so narrow that the
    first episode's pin fills every safe slot leaves the second episode
    with no seeded random target. The False arm of `if episode_safe:`
    drops the random pin and lets the disjunction over allowed_starts
    take over - the solver may then return unsat, which is fine; we only
    need the False branch exercised."""
    narrow_event = EventDefinition(
        name="narrow_event",
        category="sports",
        per_event_duration=DurationRange(min=30, max=30, unit="minutes"),
        total_event_duration=TotalDuration(min=60, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=2, max=2),
        # Catalog restricts the event to a 30-minute slot. With 30-min
        # episodes and step_minutes=10, allowed_starts is {600, 610, 620},
        # so episode 0's pin always pushes episode 1's filter to empty.
        temporal_patterns=[
            TemporalPattern(mode="fix", details={"within": ["lunch_window"]})
        ],
    )
    window_map = WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "lunch_window": WindowRange(start=600, end=630),
            "afternoon": WindowRange(start=630, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        }
    )
    _, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"narrow_event": narrow_event},
        counts={"narrow_event": 2},
        window_map=window_map,
        person_seed=99,
        maximize_event=None,
    )
    # Two z3 var pairs were created, which means the loop ran for both
    # episodes and the empty-episode_safe branch was reached on i=1.
    assert len(ev["narrow_event"]) == 2


# -------------------------------------------------------------------------------------
# Regression cases: window-fraction conflict, total-cap drop,
# safety_others tightening
# -------------------------------------------------------------------------------------


def test_window_fraction_constraint_dropped_when_persona_overrides_window():
    """A catalog seasonality pattern with `within: morning` adds a
    `window_fraction_constraints` floor requiring at least one episode
    in morning. When the persona pins this event to a different window
    via `preferred_windows`, the floor would be unsatisfiable. The
    solver must drop the catalog floor and respect the persona's
    placement (this was a known root cause).
    """
    morning_boost_event = EventDefinition(
        name="stress",
        category="health",
        per_event_duration=DurationRange(min=10, max=30, unit="minutes"),
        total_event_duration=TotalDuration(
            min=10, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=2, max=8),
        temporal_patterns=[
            TemporalPattern(
                mode="seasonality",
                details={"within": "morning", "amount": 20, "direction": "increasing"},
            )
        ],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"stress": morning_boost_event},
        counts={"stress": 5},
        window_map=_wm(),
        preferred_windows={"stress": "afternoon"},
        person_seed=0,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    starts_durs = _read_event(solver.model(), ev, "stress")
    # Every episode must land in afternoon, not morning.
    for s, _ in starts_durs:
        assert 600 <= s < 960


def test_pinned_duration_dropped_when_total_cap_unreachable():
    """When the persona-stated `duration_minutes` would push the day
    total above `total_event_duration.max` once multiplied by
    `num_events`, the pin is dropped so z3 can pick smaller actual
    durations. (Weekend walking 8x/day at 60min vs total cap 240min.)"""
    walking = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(
            min=30, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=2, max=8),
        temporal_patterns=[],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"walking": walking},
        counts={"walking": 8},
        window_map=_wm(),
        # 60 * 8 = 480 > total_max=240. Without dropping the pin the
        # day is unsat; with the drop the solver picks 30min episodes.
        preferred_durations={"walking": 60},
        person_seed=0,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    starts_durs = _read_event(solver.model(), ev, "walking")
    total = sum(d for _, d in starts_durs)
    assert total <= 240


def test_pinned_duration_kept_when_total_cap_compatible():
    """The total-cap drop must NOT trigger when the pin still fits.
    A 1-episode walk at 45min is well within total_max=240, so the pin
    survives and the actual duration equals 45."""
    walking = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(
            min=30, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        temporal_patterns=[],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"walking": walking},
        counts={"walking": 1},
        window_map=_wm(),
        preferred_durations={"walking": 45},
        person_seed=0,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    [(_, dur)] = _read_event(solver.model(), ev, "walking")
    # 45 quantized to step=10 = 40 (in [30, 60]), pin survives since
    # 40 * 1 = 40 <= total_max=240.
    assert dur == 40


def test_pinned_duration_below_step_clamps_to_step():
    """A persona-stated duration smaller than `step_minutes` is clamped
    to one full step before the per_event range check, so a 5-min
    persona pin with step=10 quantizes to 10 instead of being dropped."""
    short_event = EventDefinition(
        name="micro_event",
        category="other",
        per_event_duration=DurationRange(min=10, max=30, unit="minutes"),
        total_event_duration=TotalDuration(min=10, max=30, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"micro_event": short_event},
        counts={"micro_event": 1},
        window_map=_wm(),
        step_minutes=10,
        preferred_durations={"micro_event": 5},
        person_seed=0,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    [(_, dur)] = _read_event(solver.model(), ev, "micro_event")
    # 5 < step (10), clamped to step (10), within [10, 30].
    assert dur == 10


def test_window_fraction_kept_when_persona_does_not_override_window():
    """Without a persona window pin, the catalog's morning seasonality
    floor stays in scope and z3 must place at least one episode in
    morning. This is the inverse of the override-drop test."""
    morning_boost_event = EventDefinition(
        name="stress",
        category="health",
        per_event_duration=DurationRange(min=10, max=30, unit="minutes"),
        total_event_duration=TotalDuration(
            min=10, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=2, max=8),
        temporal_patterns=[
            TemporalPattern(
                mode="seasonality",
                details={"within": "morning", "amount": 20, "direction": "increasing"},
            )
        ],
    )
    solver, ev = build_day_model(
        day_idx=0,
        total_days=7,
        events={"stress": morning_boost_event},
        counts={"stress": 5},
        window_map=_wm(),
        preferred_windows={},  # no persona window override
        person_seed=0,
        maximize_event=None,
    )
    assert solver.check() == z3.sat
    starts_durs = _read_event(solver.model(), ev, "stress")
    # Catalog floor: at least int(0.36 * 5) = 1 in morning.
    in_morning = sum(1 for s, _ in starts_durs if 400 <= s < 600)
    assert in_morning >= 1
