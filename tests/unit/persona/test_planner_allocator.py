"""Unit tests for src.scripts.persona.planner.allocator."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.config.schema import (
    DurationRange,
    EpisodeRange,
    EventDefinition,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.event_config.loader import load_catalog
from src.scripts.persona.planner.allocator import (
    allocate_day_counts,
    allocate_horizon,
    days_with_event,
)
from tests.unit.persona.conftest import make_person, make_stage

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


def _core_event(name: str, category: str) -> EventDefinition:
    return EventDefinition(
        name=name,
        category=category,
        per_event_duration=DurationRange(min=10, max=30, unit="minutes"),
        total_event_duration=TotalDuration(min=10, max=30, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )


def _opportunistic(name: str, category: str = "leisure") -> EventDefinition:
    return EventDefinition(
        name=name,
        category=category,
        per_event_duration=DurationRange(min=10, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=10, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
    )


def _office_work() -> EventDefinition:
    return EventDefinition(
        name="office_work",
        category="work",
        requires={"occupation_status": _make_role(in_=["fulltime", "parttime"])},
        weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"],
        per_event_duration=DurationRange(min=3, max=8, unit="hours"),
        total_event_duration=TotalDuration(min=3, max=8, scale="day", unit="hours"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=[
            TemporalPattern(
                mode="seasonality",
                details={
                    "scale": "weekday",
                    "amount": 100,
                    "within": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                },
            )
        ],
    )


def _make_role(*, in_=None, eq=None):
    """Build a RolePredicate with the alias-aware constructor."""
    from src.scripts.persona.config.schema import RolePredicate

    kwargs = {}
    if in_ is not None:
        kwargs["in"] = in_
    if eq is not None:
        kwargs["eq"] = eq
    return RolePredicate.model_validate(kwargs)


def _catalog(events: list[EventDefinition]) -> Catalog:
    """Build a small Catalog by hand from a flat event list."""
    return Catalog(
        categories={},
        events_by_name={ev.name: ev for ev in events},
    )


_START_DATE = _dt.date(2026, 5, 4)  # Monday
_DEFAULT_DATE = _START_DATE


# -------------------------------------------------------------------------------------
# ---------------------------------- routine events -----------------------------------
# -------------------------------------------------------------------------------------


def test_routine_event_counted_every_day():
    """An event whose name matches a stage entry on `days` containing
    today's weekday is counted once."""
    cat = _catalog([_core_event("sleep", "sleep"), _core_event("lunch", "eat")])
    person = make_person(
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("lunch", time="12:30", duration_minutes=45),
        ]
    )
    counts = allocate_day_counts(
        person,
        cat,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert counts == {"sleep": 1, "lunch": 1}


def test_event_not_counted_when_requires_fails():
    """An event with a `requires` gate respects it - even when the
    persona has staged it."""
    sleep_for_students = EventDefinition(
        name="study_sleep",
        category="sleep",
        requires={"occupation_status": _make_role(eq="student")},
        per_event_duration=DurationRange(min=6, max=9, unit="hours"),
        total_event_duration=TotalDuration(min=6, max=9, scale="day", unit="hours"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )
    cat = _catalog([sleep_for_students])
    fulltime = make_person(
        occupation_status="fulltime",
        stages=[make_stage("study_sleep", time="22:00")],
    )
    counts = allocate_day_counts(
        fulltime,
        cat,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert counts == {"study_sleep": 0}


# -------------------------------------------------------------------------------------
# ---------------------------- catalog `weekdays` gate --------------------------------
# -------------------------------------------------------------------------------------


def test_office_work_counted_on_weekday_for_fulltime_persona():
    cat = _catalog([_office_work()])
    person = make_person(
        occupation_status="fulltime",
        stages=[
            make_stage(
                "office_work", time="09:00", days=["Mon", "Tue", "Wed", "Thu", "Fri"]
            )
        ],
    )
    monday = allocate_day_counts(
        person,
        cat,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert monday["office_work"] == 1


def test_office_work_zero_on_weekend_via_catalog_weekdays_gate():
    cat = _catalog([_office_work()])
    person = make_person(
        occupation_status="fulltime",
        stages=[
            make_stage(
                "office_work", time="09:00", days=["Mon", "Tue", "Wed", "Thu", "Fri"]
            )
        ],
    )
    saturday = allocate_day_counts(
        person,
        cat,
        day_idx=5,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert saturday["office_work"] == 0


def test_office_work_zero_when_persona_did_not_stage_it():
    cat = _catalog([_office_work()])
    person = make_person(occupation_status="fulltime", stages=[])  # no stage entry
    monday = allocate_day_counts(
        person,
        cat,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert monday["office_work"] == 0


def test_office_work_zero_when_requires_fails_for_student():
    cat = _catalog([_office_work()])
    student = make_person(
        occupation_status="student",
        stages=[make_stage("office_work", time="09:00", days=["Mon"])],
    )
    monday = allocate_day_counts(
        student,
        cat,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert monday["office_work"] == 0


# -------------------------------------------------------------------------------------
# ---------------------------- weekly-cadence stages ----------------------------------
# -------------------------------------------------------------------------------------


def test_stage_counted_only_on_listed_weekdays():
    padel = _opportunistic("padel", "sports")
    cat = _catalog([padel])
    person = make_person(
        stages=[make_stage("padel", time="21:00", duration_minutes=60, days=["Wed"])]
    )
    monday = allocate_day_counts(
        person,
        cat,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    wednesday = allocate_day_counts(
        person,
        cat,
        day_idx=2,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert monday["padel"] == 0
    assert wednesday["padel"] == 1


def test_weekend_only_stage_counted_only_on_its_day():
    family = _opportunistic("family_time", "social")
    cat = _catalog([family])
    person = make_person(
        stages=[
            make_stage(
                "family_time", time="afternoon", duration_minutes=120, days=["Sat"]
            )
        ]
    )
    monday = allocate_day_counts(
        person, cat, day_idx=0, total_days=7, date=_DEFAULT_DATE, window_map=_wm()
    )
    saturday = allocate_day_counts(
        person, cat, day_idx=5, total_days=7, date=_DEFAULT_DATE, window_map=_wm()
    )
    sunday = allocate_day_counts(
        person, cat, day_idx=6, total_days=7, date=_DEFAULT_DATE, window_map=_wm()
    )
    assert monday["family_time"] == 0
    assert saturday["family_time"] == 1
    assert sunday["family_time"] == 0


# -------------------------------------------------------------------------------------
# ---------------------------- date-anchored stages -----------------------------------
# -------------------------------------------------------------------------------------


def test_dated_stage_counted_only_on_target_date():
    dentist = _opportunistic("dentist", "health")
    cat = _catalog([dentist])
    target = _START_DATE + _dt.timedelta(days=8)
    person = make_person(
        stages=[
            make_stage(
                "dentist",
                time="14:00",
                duration_minutes=30,
                days=[],
                date=target,
            )
        ]
    )
    on_target = allocate_day_counts(
        person,
        cat,
        day_idx=8,
        total_days=14,
        date=target,
        window_map=_wm(),
    )
    off_target = allocate_day_counts(
        person,
        cat,
        day_idx=0,
        total_days=14,
        date=_START_DATE,
        window_map=_wm(),
    )
    assert on_target["dentist"] == 1
    assert off_target["dentist"] == 0


# -------------------------------------------------------------------------------------
# -------------------------- catalog events not stated by persona ---------------------
# -------------------------------------------------------------------------------------


def test_unstated_opportunistic_event_is_zero():
    swim = _opportunistic("swimming", "sports")
    cat = _catalog([swim])
    person = make_person(stages=[])  # no stages staged at all
    counts = allocate_day_counts(
        person, cat, day_idx=0, total_days=7, date=_DEFAULT_DATE, window_map=_wm()
    )
    assert counts["swimming"] == 0


# -------------------------------------------------------------------------------------
# ---------------------------------- allocate_horizon ---------------------------------
# -------------------------------------------------------------------------------------


def test_allocate_horizon_returns_one_dict_per_day():
    cat = _catalog([_core_event("sleep", "sleep")])
    person = make_person(stages=[make_stage("sleep", time="23:00")])
    out = allocate_horizon(
        person,
        cat,
        horizon_days=14,
        start_date=_START_DATE,
        window_map=_wm(),
    )
    assert len(out) == 14
    assert all(d["sleep"] == 1 for d in out)


def test_allocate_horizon_zero_days_returns_empty_list():
    cat = _catalog([_core_event("sleep", "sleep")])
    person = make_person(stages=[make_stage("sleep", time="23:00")])
    assert (
        allocate_horizon(
            person,
            cat,
            horizon_days=0,
            start_date=_START_DATE,
            window_map=_wm(),
        )
        == []
    )


def test_allocate_horizon_rejects_negative_days():
    cat = _catalog([_core_event("sleep", "sleep")])
    person = make_person(stages=[make_stage("sleep", time="23:00")])
    with pytest.raises(ValueError, match="horizon_days"):
        allocate_horizon(
            person,
            cat,
            horizon_days=-1,
            start_date=_START_DATE,
            window_map=_wm(),
        )


def test_allocate_horizon_threads_dated_stage_correctly():
    """A dated stage lands on its day index based on start_date offset."""
    dentist = _opportunistic("dentist", "health")
    cat = _catalog([dentist])
    target = _START_DATE + _dt.timedelta(days=3)
    person = make_person(
        stages=[
            make_stage(
                "dentist",
                time="14:00",
                duration_minutes=30,
                days=[],
                date=target,
            )
        ]
    )
    out = allocate_horizon(
        person,
        cat,
        horizon_days=7,
        start_date=_START_DATE,
        window_map=_wm(),
    )
    assert out[3]["dentist"] == 1
    assert out[0]["dentist"] == 0
    assert out[6]["dentist"] == 0


# -------------------------------------------------------------------------------------
# --------------------------------- days_with_event -----------------------------------
# -------------------------------------------------------------------------------------


def test_days_with_event_collects_non_zero_days():
    counts = [
        {"reading": 0, "sleep": 1},
        {"reading": 1, "sleep": 1},
        {"reading": 0, "sleep": 1},
        {"reading": 1, "sleep": 1},
    ]
    assert days_with_event(counts, "reading") == [1, 3]
    assert days_with_event(counts, "sleep") == [0, 1, 2, 3]


def test_days_with_event_handles_missing_event_key():
    counts = [{"sleep": 1}, {"sleep": 1}]
    assert days_with_event(counts, "running") == []


# -------------------------------------------------------------------------------------
# ----------------------- integration with example event_config -----------------------
# -------------------------------------------------------------------------------------


def test_allocator_against_example_catalog(event_yaml):
    """Sanity check using the bundled `event_config.yaml` catalog."""
    catalog = load_catalog(event_yaml)
    person = make_person(
        occupation_status="fulltime",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time="12:30", duration_minutes=45),
            make_stage("dinner", time="19:00", duration_minutes=60),
            make_stage(
                "office_work", time="09:00", days=["Mon", "Tue", "Wed", "Thu", "Fri"]
            ),
        ],
    )
    monday = allocate_day_counts(
        person,
        catalog,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    # Routine + work events present.
    assert monday["sleep"] == 1
    assert monday["first_eat"] == 1
    assert monday["lunch"] == 1
    assert monday["dinner"] == 1
    assert monday["office_work"] == 1
    # study fails the requires gate (occupation != student).
    assert monday["study"] == 0
    # Unstated opportunistic events are 0.
    assert monday["padel"] == 0
    assert monday["running"] == 0


def test_catalog_weekday_restriction_zeros_count_on_disallowed_weekday(event_yaml):
    """An event whose catalog has `weekdays: [Mon-Fri]` must NOT be
    counted on Sat or Sun, even when the persona would otherwise stage
    it. Without this gate the allocator would feed weekend office_work
    / study to the solver and the validator would flag every weekend
    day as a violation."""
    catalog = load_catalog(event_yaml)
    person = make_person(
        occupation_status="fulltime",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time="12:30", duration_minutes=45),
            make_stage("dinner", time="19:00", duration_minutes=60),
            make_stage(
                "office_work",
                time="09:00",
                days=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            ),
        ],
    )
    saturday = allocate_day_counts(
        person,
        catalog,
        day_idx=5,
        total_days=7,
        date=_DEFAULT_DATE + _dt.timedelta(days=5),
        window_map=_wm(),
    )
    sunday = allocate_day_counts(
        person,
        catalog,
        day_idx=6,
        total_days=7,
        date=_DEFAULT_DATE + _dt.timedelta(days=6),
        window_map=_wm(),
    )
    # Office_work is restricted to Mon-Fri in the catalog, even though
    # the persona staged it for every day.
    assert saturday["office_work"] == 0
    assert sunday["office_work"] == 0
    # Routine events still apply on weekends.
    assert saturday["sleep"] == 1
    assert sunday["lunch"] == 1


def test_catalog_weekday_restriction_blocks_persona_stated_activity():
    """An event with `weekdays:` only fires on those weekdays. A persona
    staging that event on Sun for a weekday-only event still counts
    zero on Sun, because the catalog gate sits before the persona-stage
    check by design (catalog rules win)."""
    weekday_only_padel = EventDefinition(
        name="padel",
        category="sports",
        weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"],
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=30, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
    )
    catalog = Catalog(
        categories={},
        events_by_name={"padel": weekday_only_padel},
    )
    person = make_person(
        stages=[
            make_stage("padel", time="14:00", duration_minutes=60, days=["Wed", "Sun"]),
        ]
    )
    wednesday = allocate_day_counts(
        person,
        catalog,
        day_idx=2,
        total_days=7,
        date=_DEFAULT_DATE + _dt.timedelta(days=2),
        window_map=_wm(),
    )
    sunday = allocate_day_counts(
        person,
        catalog,
        day_idx=6,
        total_days=7,
        date=_DEFAULT_DATE + _dt.timedelta(days=6),
        window_map=_wm(),
    )
    assert wednesday["padel"] == 1
    assert sunday["padel"] == 0


# -------------------------------------------------------------------------------------
# ----------------------- multi-episode events from catalog base_count ----------------
# -------------------------------------------------------------------------------------


def test_persona_staged_event_uses_catalog_base_count_for_multi_episode():
    """A persona-staged event with `total_event_episodes: {0, 22}` (e.g.
    smoking) must fire `base_count` times per day, not just once. This
    matches the legacy `compute_event_counts.py` semantics where the
    catalog drives the per-day count."""
    smoking = EventDefinition(
        name="smoking",
        category="health",
        per_event_duration=DurationRange(min=10, max=10, unit="minutes"),
        total_event_duration=TotalDuration(min=0, max=300, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=22),
        temporal_patterns=[],
    )
    catalog = _catalog([smoking])
    person = make_person(
        stages=[make_stage("smoking", time="evening", duration_minutes=10)]
    )
    counts = allocate_day_counts(
        person,
        catalog,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    # base_count = (0 + (22 - 0) // 2) = 11.
    assert counts["smoking"] == 11


def test_unstaged_event_allocates_to_zero():
    """An event the persona did not stage allocates to zero - the
    schema is fully event-name-agnostic, so even an "environmental"
    event like raining defaults to 0 unless staged or filtered in via
    a `requires` predicate that matches the persona."""
    raining = EventDefinition(
        name="raining",
        category="weather",
        per_event_duration=DurationRange(min=1, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=0, max=240, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=12),
        temporal_patterns=[],
    )
    catalog = _catalog([raining])
    person = make_person(stages=[])
    counts = allocate_day_counts(
        person,
        catalog,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    assert counts["raining"] == 0


def test_unstaged_event_fires_when_persona_stages_it():
    """Same `raining` event the previous test left to zero - here a
    persona stages it daily, so the catalog's multi-episode
    `base_count` (6) drives the per-day count."""
    raining = EventDefinition(
        name="raining",
        category="weather",
        per_event_duration=DurationRange(min=1, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=0, max=240, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=12),
        temporal_patterns=[],
    )
    catalog = _catalog([raining])
    person = make_person(
        stages=[make_stage("raining", time="morning", duration_minutes=10)]
    )
    counts = allocate_day_counts(
        person,
        catalog,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    # Persona staged on Mon to persona_count=1; catalog base_count=6;
    # max(1, 6) = 6.
    assert counts["raining"] == 6


def test_persona_staged_event_with_zero_base_count_still_fires_once():
    """Backward-compatible floor: when the catalog `base_count` rounds
    to 0 (e.g. `{min: 0, max: 1}` with no seasonality boost) but the
    persona explicitly staged the event, the count is still 1."""
    running = EventDefinition(
        name="running",
        category="sports",
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=0, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        temporal_patterns=[],
    )
    catalog = _catalog([running])
    person = make_person(
        stages=[make_stage("running", time="06:30", duration_minutes=45, days=["Mon"])]
    )
    counts = allocate_day_counts(
        person,
        catalog,
        day_idx=0,
        total_days=7,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    # base_count = 0; persona staged on Mon to floor to 1.
    assert counts["running"] == 1


def test_persona_staged_event_is_zeroed_when_constraints_disable_it():
    """Even when the persona stages an event on this weekday, a catalog
    `seasonality` pattern with `amount=100` and a `within` list that
    excludes today must zero the count - the event is disabled by the
    constraints extractor regardless of staging."""
    workday_only_walk = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=15, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=15, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        temporal_patterns=[
            TemporalPattern(
                mode="seasonality",
                details={
                    "scale": "weekday",
                    "amount": 100,
                    "within": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                },
            )
        ],
    )
    cat = _catalog([workday_only_walk])
    # Persona stages walking every day - including the weekend - but the
    # catalog disables the event on weekends, so the count must be 0.
    person = make_person(
        stages=[
            make_stage("walking", time="morning", duration_minutes=30),
        ]
    )
    sunday = allocate_day_counts(
        person,
        cat,
        day_idx=6,  # Sunday
        total_days=7,
        date=_DEFAULT_DATE + _dt.timedelta(days=6),
        window_map=_wm(),
    )
    assert sunday["walking"] == 0


def test_multi_episode_count_is_modulated_by_seasonality_and_trend():
    """Trend ramps actually move the per-day count - the legacy
    `quitting smoking` pattern relies on this."""
    smoking_quitting = EventDefinition(
        name="smoking",
        category="health",
        per_event_duration=DurationRange(min=10, max=10, unit="minutes"),
        total_event_duration=TotalDuration(min=0, max=300, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=22),
        temporal_patterns=[
            TemporalPattern(
                mode="trend",
                details={
                    # Day-indexed ramp: from day 1 to day 7 of the horizon.
                    # (Pre-Step-0 the same shape was authored with
                    # `scale: season`; the extractor used to treat that as
                    # day-indexed.  Post-Step-0, `scale: season` means a
                    # season-of-horizon index, so we re-author this as
                    # `scale: day` to preserve the original semantics.)
                    "scale": "day",
                    "direction": "decreasing",
                    "amount": 10,
                    "start": 1,
                    "end": 7,
                },
            )
        ],
    )
    catalog = _catalog([smoking_quitting])
    person = make_person(
        stages=[make_stage("smoking", time="evening", duration_minutes=10)]
    )
    day_0 = allocate_day_counts(
        person,
        catalog,
        day_idx=0,
        total_days=14,
        date=_DEFAULT_DATE,
        window_map=_wm(),
    )
    day_7 = allocate_day_counts(
        person,
        catalog,
        day_idx=7,
        total_days=14,
        date=_DEFAULT_DATE + _dt.timedelta(days=7),
        window_map=_wm(),
    )
    # Day 0 (start of the ramp): full base_count = 11.
    assert day_0["smoking"] == 11
    # Day 7 (after the ramp): base_count - 10 = 1, clamped at min = 0.
    # The persona-staged floor keeps it at 1.
    assert day_7["smoking"] == 1
