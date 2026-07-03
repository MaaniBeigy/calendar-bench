"""Unit tests for src.scripts.persona.planner.horizon."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.config.schema import (
    DurationRange,
    EnvironmentConfig,
    EpisodeRange,
    EventDefinition,
    HorizonConfig,
    OutputConfig,
    ParallelismConfig,
    SolverConfig,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.event_config.loader import load_catalog
from src.scripts.persona.planner.horizon import plan_horizon
from tests.unit.persona.conftest import make_person, make_stage


def _env(weeks: int = 1) -> EnvironmentConfig:
    return EnvironmentConfig(
        seed=1,
        horizon=HorizonConfig(start_date=_dt.date(2026, 5, 4), weeks=weeks),
        output=OutputConfig(dir="./out"),
        solver=SolverConfig(),
        time_windows={
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        },
        parallelism=ParallelismConfig(workers=1, executor="process"),
    )


def _student() -> Person:
    """A student persona with a complete daily routine."""
    return make_person(occupation_status="student")


def test_plan_horizon_returns_one_day_per_horizon_day(event_yaml):
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    assert len(schedule.days) == 7
    for idx, day in enumerate(schedule.days):
        assert day.day_index == idx


def test_plan_horizon_dates_increment_by_one(event_yaml):
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    base = _dt.date(2026, 5, 4)
    for idx, day in enumerate(schedule.days):
        assert day.date == base + _dt.timedelta(days=idx)


def test_plan_horizon_assigns_correct_weekdays(event_yaml):
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    expected = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert [d.weekday for d in schedule.days] == expected


def test_plan_horizon_records_persona_identity_on_schedule(event_yaml):
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    assert schedule.person_id == "alice_0000"
    assert schedule.persona_id == "alice"
    assert schedule.person_seed == 42


def test_plan_horizon_solves_routine_events_each_day(event_yaml):
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    for day in schedule.days:
        assert "sleep" in day.events and len(day.events["sleep"]) == 1
        assert "lunch" in day.events and len(day.events["lunch"]) == 1


def test_plan_horizon_propagates_spillover_to_next_day(event_yaml):
    """Day 0's sleep crossing midnight shows up in Day 1's `spillovers` list."""
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))

    [sleep0] = schedule.days[0].events["sleep"]
    if sleep0.start + sleep0.duration > 1440:
        day1 = schedule.days[1]
        assert any(s.event_name == "sleep" for s in day1.spillovers)
        spill = next(s for s in day1.spillovers if s.event_name == "sleep")
        assert spill.duration == (sleep0.start + sleep0.duration) - 1440


def test_plan_horizon_day1_events_avoid_spillover_range(event_yaml):
    """Day 1's events must not collide with the night spillover range."""
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))

    day1 = schedule.days[1]
    if not day1.spillovers:
        return  # nothing carried over
    spill = day1.spillovers[0]
    occ_start, occ_end = spill.start, spill.start + spill.duration

    for events in day1.events.values():
        for ev in events:
            ev_end = ev.start + ev.duration
            assert ev.start >= occ_end or ev_end <= occ_start


def test_plan_horizon_is_deterministic(event_yaml):
    catalog = load_catalog(event_yaml)
    a = plan_horizon(_student(), catalog, _env(weeks=1))
    b = plan_horizon(_student(), catalog, _env(weeks=1))
    assert a == b


def test_plan_horizon_day_zero_has_no_prior_spillovers(event_yaml):
    """Day 0 must always carry an empty spillover list."""
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    assert schedule.days[0].spillovers == []


def test_plan_horizon_with_empty_catalog_returns_empty_days():
    """An empty catalog still produces one DaySchedule per day, all empty."""
    empty_catalog = Catalog(categories={}, events_by_name={})
    schedule = plan_horizon(_student(), empty_catalog, _env(weeks=1))
    assert len(schedule.days) == 7
    for day in schedule.days:
        assert day.events == {}
        assert day.spillovers == []


def test_plan_horizon_retries_without_preferred_windows_when_first_attempt_unsat():
    """If the persona-stated window leaves no feasible slot for an
    opportunistic event, the day's first solver attempt is unsat. The
    planner must retry once with `preferred_windows={}` so the event
    falls back to the catalog-default allowed_starts (full day) and
    cadence is preserved at the cost of window-intent.
    """
    walking = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=60, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=60, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        temporal_patterns=[],
    )
    # The blocker is also persona-stated, pinned at 06:40 (=minute 400)
    # for 200 minutes. With the persona pin, blocker deterministically
    # occupies the entire morning window [400, 600).
    blocker = EventDefinition(
        name="morning_block",
        category="other",
        per_event_duration=DurationRange(min=200, max=200, unit="minutes"),
        total_event_duration=TotalDuration(
            min=200, max=200, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[],
    )
    catalog = Catalog(
        categories={},
        events_by_name={"morning_block": blocker, "walking": walking},
    )
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    person = make_person(
        person_id="p0",
        person_seed=99,
        occupation_status="student",
        stages=[
            make_stage(
                "morning_block", time="06:40", duration_minutes=200, days=weekdays
            ),
            make_stage("walking", time="morning", duration_minutes=60, days=weekdays),
        ],
    )
    schedule = plan_horizon(person, catalog, _env(weeks=1))
    days_with_walking = [d for d in schedule.days if d.events.get("walking")]
    assert len(days_with_walking) == 7
    for day in schedule.days:
        walks = day.events["walking"]
        assert len(walks) == 1
        block = day.events["morning_block"][0]
        assert block.start == 400 and block.duration == 200
        walk = walks[0]
        assert walk.start + walk.duration <= 400 or walk.start >= 600


def test_plan_horizon_returns_empty_day_when_solver_is_unsat():
    """An event with contradictory duration / total bounds forces unsat."""
    bad_sleep = EventDefinition(
        name="sleep",
        category="sleep",
        per_event_duration=DurationRange(min=60, max=60, unit="minutes"),
        total_event_duration=TotalDuration(
            min=120, max=120, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )
    catalog = Catalog(categories={}, events_by_name={"sleep": bad_sleep})
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    assert len(schedule.days) == 7
    for day in schedule.days:
        assert day.events == {}


def test_plan_horizon_falls_back_to_no_random_when_random_pin_blocks_day():
    """If the random ep-0 pin happens to land on a minute that leaves
    no feasible joint assignment for the trailing episodes (a real
    case), the planner must retry once
    with `person_seed=0` so z3 places every episode itself. Tested by
    constructing a single-episode-window event whose window is so
    narrow that any non-degenerate random pick fights with the
    persona-pinned meal in the same window."""
    # Tight evening window: only two step-aligned starts; both feasible
    # with z3 alone, but a random pick at the wrong minute can collide
    # with the dinner pin so cadence requires the no-random retry.
    walk = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=20, max=20, unit="minutes"),
        total_event_duration=TotalDuration(min=20, max=20, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[
            TemporalPattern(mode="fix", details={"within": ["evening"]})
        ],
    )
    dinner = EventDefinition(
        name="dinner",
        category="eat",
        per_event_duration=DurationRange(min=30, max=30, unit="minutes"),
        total_event_duration=TotalDuration(min=30, max=30, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[
            TemporalPattern(mode="fix", details={"within": ["evening"]})
        ],
    )
    catalog = Catalog(
        categories={},
        events_by_name={"dinner": dinner, "walking": walk},
    )
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    person = make_person(
        person_seed=12345,
        stages=[
            make_stage("dinner", time="19:00", duration_minutes=30, days=weekdays),
            # Walking with no HH:MM pin - just a window. Random pick
            # will choose a minute that may collide with dinner.
            make_stage("walking", time="evening", duration_minutes=20, days=weekdays),
        ],
    )
    schedule = plan_horizon(person, catalog, _env(weeks=1))
    # Cadence: every day has both events.
    for day in schedule.days:
        assert len(day.events.get("dinner", [])) == 1
        assert len(day.events.get("walking", [])) == 1


def test_plan_horizon_returns_empty_day_when_window_relaxed_retry_also_unsat():
    """Both solver attempts fail: the persona-stated window pushes the
    first attempt to unsat, and the retry without `preferred_windows` is
    still unsat because the event itself is fundamentally infeasible
    (1 episode of 60min cannot sum to a required total of 120min)."""
    bad_walking = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=60, max=60, unit="minutes"),
        total_event_duration=TotalDuration(
            min=120, max=120, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )
    catalog = Catalog(categories={}, events_by_name={"walking": bad_walking})
    person = make_person(
        stages=[
            make_stage("walking", time="morning", duration_minutes=60),
        ]
    )
    schedule = plan_horizon(person, catalog, _env(weeks=1))
    assert len(schedule.days) == 7
    for day in schedule.days:
        assert day.events == {}


def test_plan_horizon_skips_zero_count_event_vars(event_yaml):
    """Catalog events the persona does not stage produce no entries."""
    catalog = load_catalog(event_yaml)
    schedule = plan_horizon(_student(), catalog, _env(weeks=1))
    for day in schedule.days:
        # The student persona built by make_person() only stages routine
        # events (sleep + meals). All other catalog events stay empty.
        assert "padel" not in day.events
        assert "running" not in day.events
        assert "gym" not in day.events
        assert "sleep" in day.events
        assert "lunch" in day.events


def test_plan_horizon_solver_objective_none_is_accepted(event_yaml):
    """`optimize_objective=none` skips the maximize step but still solves."""
    catalog = load_catalog(event_yaml)
    env = EnvironmentConfig(
        seed=1,
        horizon=HorizonConfig(start_date=_dt.date(2026, 5, 4), weeks=1),
        output=OutputConfig(dir="./out"),
        solver=SolverConfig(optimize_objective="none"),
        time_windows={
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        },
        parallelism=ParallelismConfig(workers=1, executor="process"),
    )
    schedule = plan_horizon(_student(), catalog, env)
    assert len(schedule.days) == 7
    for day in schedule.days:
        assert "sleep" in day.events


def _student_with_lunch_at(time_str: str, duration: int = 45) -> Person:
    """Student persona with a custom lunch time."""
    return make_person(
        occupation_status="student",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time=time_str, duration_minutes=duration),
            make_stage("dinner", time="19:00", duration_minutes=60),
        ],
    )


def test_plan_horizon_persons_with_distinct_lunch_times_solve_to_distinct_schedules(
    event_yaml,
):
    """Two persons differing only in their lunch time produce different
    schedules - the day-model honours each persona's pinned start."""
    catalog = load_catalog(event_yaml)
    env = _env(weeks=1)
    a = plan_horizon(_student_with_lunch_at("12:00"), catalog, env)
    b = plan_horizon(_student_with_lunch_at("13:30"), catalog, env)
    a_lunches = [d.events["lunch"][0].start for d in a.days]
    b_lunches = [d.events["lunch"][0].start for d in b.days]
    assert a_lunches != b_lunches
    assert all(start == 720 for start in a_lunches)
    assert all(start == 810 for start in b_lunches)


def test_plan_horizon_persons_with_distinct_lunch_durations_solve_to_distinct_schedules(
    event_yaml,
):
    catalog = load_catalog(event_yaml)
    env = _env(weeks=1)
    short = plan_horizon(_student_with_lunch_at("12:30", duration=30), catalog, env)
    long_ = plan_horizon(_student_with_lunch_at("12:30", duration=60), catalog, env)
    short_durations = [d.events["lunch"][0].duration for d in short.days]
    long_durations = [d.events["lunch"][0].duration for d in long_.days]
    assert short_durations != long_durations
    assert all(d == 30 for d in short_durations)
    assert all(d == 60 for d in long_durations)


def _student_with_seed(seed: int) -> Person:
    """Same persona body, different `person_seed` - drives the random
    fallback for events with no HH:MM pin (here a study window-token
    stage)."""
    return make_person(
        person_id=f"alice_{seed:04d}",
        person_seed=seed,
        occupation_status="student",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time="12:30", duration_minutes=45),
            make_stage("dinner", time="19:00", duration_minutes=60),
            make_stage(
                "study", time="morning", days=["Mon", "Tue", "Wed", "Thu", "Fri"]
            ),
        ],
    )


def test_plan_horizon_two_persons_same_persona_diverge_via_person_seed(event_yaml):
    """Same persona body, different person_seed: the day-model's seeded
    random fallback gives the two persons different schedules for
    events with no HH:MM pin."""
    catalog = load_catalog(event_yaml)
    env = _env(weeks=1)
    a = plan_horizon(_student_with_seed(seed=11), catalog, env)
    b = plan_horizon(_student_with_seed(seed=99), catalog, env)
    a_signature = [
        (name, ev.start, ev.duration)
        for day in a.days
        for name, evs in sorted(day.events.items())
        for ev in evs
    ]
    b_signature = [
        (name, ev.start, ev.duration)
        for day in b.days
        for name, evs in sorted(day.events.items())
        for ev in evs
    ]
    assert a_signature != b_signature
