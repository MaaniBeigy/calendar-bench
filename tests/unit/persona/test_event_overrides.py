"""Unit tests for per-persona event overrides.

Covers `EventOverride` schema validation, the `apply_event_overrides`
catalog merger, and the horizon-level wiring that makes a persona's
overrides drive its schedule.
"""

from __future__ import annotations

import datetime as _dt

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventDefinition,
    EventOverride,
    JitterConfig,
    Persona,
    PersonaCommon,
    PersonaConfig,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.event import Catalog, apply_event_overrides
from src.scripts.persona.sampling.persona_sampler import sample_population


def _running_event() -> EventDefinition:
    return EventDefinition(
        name="running",
        category="sports",
        per_event_duration=DurationRange(min=30, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=30, max=120, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
    )


def _catalog_with_running() -> Catalog:
    running = _running_event()
    return Catalog(
        categories={"sports": Category(name="sports", events={"running": running})},
        events_by_name={"running": running},
    )


# -------------------------------------------------------------------------------------
# ----------------------------- EventOverride schema ----------------------------------
# -------------------------------------------------------------------------------------


def test_event_override_default_is_all_none():
    """Empty overrides leave every catalog field intact."""
    override = EventOverride()
    assert override.per_event_duration is None
    assert override.total_event_episodes is None
    assert override.temporal_patterns is None


def test_event_override_accepts_partial_fields():
    """Any subset of EventDefinition fields can be set on an override."""
    override = EventOverride(
        per_event_duration=DurationRange(min=20, max=40, unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=2),
    )
    assert override.per_event_duration.min == 20
    assert override.total_event_episodes.max == 2


def test_event_override_validates_duration_order():
    """Validators on nested fields still fire (max < min must be rejected)."""
    with pytest.raises(ValidationError):
        EventOverride(per_event_duration=DurationRange(min=60, max=20, unit="minutes"))


# -------------------------------------------------------------------------------------
# ----------------------------- apply_event_overrides ---------------------------------
# -------------------------------------------------------------------------------------


def test_apply_overrides_returns_same_catalog_when_overrides_empty():
    catalog = _catalog_with_running()
    out = apply_event_overrides(catalog, {})
    assert out is catalog  # short-circuit returns the same object


def test_apply_overrides_replaces_only_set_fields():
    catalog = _catalog_with_running()
    override = EventOverride(
        per_event_duration=DurationRange(min=20, max=40, unit="minutes")
    )
    merged = apply_event_overrides(catalog, {"running": override})
    running = merged.get("running")
    # Replaced field
    assert running.per_event_duration.min == 20
    assert running.per_event_duration.max == 40
    # Untouched fields
    assert running.total_event_episodes.max == 1
    assert running.category == "sports"


def test_apply_overrides_can_swap_temporal_patterns():
    catalog = _catalog_with_running()
    new_patterns = [
        TemporalPattern(
            mode="seasonality",
            details={"within": ["Sat", "Sun"], "amount": 50, "direction": "increasing"},
        )
    ]
    merged = apply_event_overrides(
        catalog,
        {"running": EventOverride(temporal_patterns=new_patterns)},
    )
    running = merged.get("running")
    assert len(running.temporal_patterns) == 1
    assert running.temporal_patterns[0].mode == "seasonality"


def test_apply_overrides_ignores_unknown_event_names():
    """Overriding a name that the catalog does not define is a no-op."""
    catalog = _catalog_with_running()
    merged = apply_event_overrides(
        catalog,
        {"unknown_event": EventOverride(intensity=2)},
    )
    assert "unknown_event" not in merged
    assert merged.get("running").category == "sports"


def test_apply_overrides_preserves_categories_and_indexing():
    """The merged catalog keeps the same categories and the events_by_name index."""
    catalog = _catalog_with_running()
    merged = apply_event_overrides(
        catalog,
        {"running": EventOverride(intensity=4)},
    )
    assert "sports" in merged.categories
    assert "running" in merged.events_by_name
    assert merged.get("running").intensity == 4


def test_apply_overrides_does_not_mutate_input_catalog():
    catalog = _catalog_with_running()
    apply_event_overrides(
        catalog,
        {"running": EventOverride(intensity=5)},
    )
    # Original catalog stays at the EventDefinition default (None).
    assert catalog.get("running").intensity is None


# -------------------------------------------------------------------------------------
# ------------------------- Persona / Person plumbing ---------------------------------
# -------------------------------------------------------------------------------------


from tests.unit.persona.conftest import make_stage  # noqa: E402


def test_persona_default_event_overrides_is_empty():
    persona = Persona(
        id="alice",
        instances=1,
        occupation_status="student",
    )
    assert persona.event_overrides == {}


def test_persona_event_overrides_can_carry_partial_definitions():
    persona = Persona(
        id="alice",
        instances=1,
        occupation_status="student",
        event_overrides={
            "running": EventOverride(
                per_event_duration=DurationRange(min=20, max=40, unit="minutes")
            )
        },
    )
    assert persona.event_overrides["running"].per_event_duration.min == 20


def test_sampler_copies_event_overrides_onto_each_person():
    """Every Person sampled from a persona gets its template's overrides."""
    persona = Persona(
        id="alice",
        instances=2,
        occupation_status="student",
        event_overrides={
            "running": EventOverride(
                per_event_duration=DurationRange(min=15, max=45, unit="minutes")
            )
        },
    )
    cfg = PersonaConfig(common=PersonaCommon(), personas=[persona])
    persons = sample_population(cfg, root_seed=1)
    assert len(persons) == 2
    for person in persons:
        assert "running" in person.event_overrides
        assert person.event_overrides["running"].per_event_duration.min == 15


# -------------------------------------------------------------------------------------
# ----------------------- horizon: per-persona effective catalog ----------------------
# -------------------------------------------------------------------------------------


def _env_for_horizon():
    from src.scripts.persona.config.schema import (
        EnvironmentConfig,
        HorizonConfig,
        OutputConfig,
        ParallelismConfig,
        SolverConfig,
    )

    return EnvironmentConfig(
        seed=1,
        horizon=HorizonConfig(start_date=_dt.date(2026, 5, 4), weeks=1),
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


def test_plan_horizon_applies_persona_event_overrides(event_yaml):
    """A persona that overrides `running` to require duration in [15, 30]
    minutes must produce running events inside that range."""
    from src.scripts.persona.event_config.loader import load_catalog
    from src.scripts.persona.planner.horizon import plan_horizon
    from tests.unit.persona.conftest import make_person

    catalog = load_catalog(event_yaml)
    person = make_person(
        person_id="alice_0000",
        persona_id="alice",
        person_seed=42,
        occupation_status="student",
        stages=[
            make_stage("sleep", time="23:00"),
            make_stage("first_eat", time="07:30", duration_minutes=20),
            make_stage("lunch", time="12:30", duration_minutes=45),
            make_stage("dinner", time="19:00", duration_minutes=60),
            make_stage("running", time="morning", days=["Tue"]),
        ],
        event_overrides={
            "running": EventOverride(
                # Both bounds must be tightened together. The base catalog
                # carries `total_event_duration: 30-120 min`, so an override
                # that only narrows `per_event_duration` to [15, 30] would
                # leave the day-level total unsatisfiable.
                per_event_duration=DurationRange(min=15, max=30, unit="minutes"),
                total_event_duration=TotalDuration(
                    min=15, max=30, scale="day", unit="minutes"
                ),
            )
        },
    )
    schedule = plan_horizon(person, catalog, _env_for_horizon())
    tuesday = next(d for d in schedule.days if d.weekday == "Tue")
    [run] = tuesday.events["running"]
    assert 15 <= run.duration <= 30


def test_plan_horizon_overrides_do_not_leak_across_persons(event_yaml):
    """Two persons with different event_overrides on the same catalog event
    must each see only their own override."""
    from src.scripts.persona.event_config.loader import load_catalog
    from src.scripts.persona.planner.horizon import plan_horizon
    from tests.unit.persona.conftest import make_person

    catalog = load_catalog(event_yaml)
    common_stages = [
        make_stage("sleep", time="23:00"),
        make_stage("first_eat", time="07:30", duration_minutes=20),
        make_stage("lunch", time="12:30", duration_minutes=45),
        make_stage("dinner", time="19:00", duration_minutes=60),
        make_stage("running", time="morning", duration_minutes=25, days=["Tue"]),
    ]
    short_runner = make_person(
        person_id="alice_short",
        persona_id="alice",
        person_seed=7,
        occupation_status="student",
        stages=common_stages,
        event_overrides={
            "running": EventOverride(
                per_event_duration=DurationRange(min=15, max=20, unit="minutes"),
                total_event_duration=TotalDuration(
                    min=15, max=20, scale="day", unit="minutes"
                ),
            )
        },
    )
    long_runner = make_person(
        person_id="alice_long",
        persona_id="alice",
        person_seed=7,
        occupation_status="student",
        stages=common_stages,
        event_overrides={
            "running": EventOverride(
                per_event_duration=DurationRange(min=90, max=120, unit="minutes"),
                total_event_duration=TotalDuration(
                    min=90, max=120, scale="day", unit="minutes"
                ),
            )
        },
    )
    short_sched = plan_horizon(short_runner, catalog, _env_for_horizon())
    long_sched = plan_horizon(long_runner, catalog, _env_for_horizon())
    short_run = next(d for d in short_sched.days if d.weekday == "Tue").events[
        "running"
    ][0]
    long_run = next(d for d in long_sched.days if d.weekday == "Tue").events["running"][
        0
    ]
    assert 15 <= short_run.duration <= 20
    assert 90 <= long_run.duration <= 120
