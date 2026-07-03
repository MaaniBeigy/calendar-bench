"""Tests for `metrics/preference_constraints.py`."""

from __future__ import annotations

import datetime as _dt
from unittest.mock import MagicMock

import fakeredis
import pytest

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    EventOverride,
    JitterConfig,
    Persona,
    PersonaCommon,
    PersonaEventStage,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
from src.scripts.scenarios.metrics.preference_constraints import (
    PersonaConstraints,
    ResolvedStage,
    _parse_hhmm,
    _resolve_event_def,
    _stage_fires_on,
)
from src.scripts.scenarios.metrics.preference_mapping import (
    MappedEvent,
    PreferenceMapper,
)

HEALTH_PREFIX = "https://w3id.org/calendar-bench/health/task/"
ACTIVITY_PREFIX = "https://w3id.org/calendar-bench/human-activities/activity/"


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


def _make_event(
    name: str,
    *,
    category: str = "sports",
    health_task_iri: str | None = None,
    human_activity_iri: str | None = None,
    intensity: int | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category=category,
        intensity=intensity,
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        health_task_iri=health_task_iri,
        human_activity_iri=human_activity_iri,
    )


def _make_event_config(events: list[EventDefinition]) -> EventConfig:
    return EventConfig(
        categories={
            "sports": Category(name="sports", events={ev.name: ev for ev in events})
        }
    )


def _make_persona(
    stages: list[PersonaEventStage] | None = None,
    overrides: dict[str, EventOverride] | None = None,
) -> Persona:
    return Persona(
        id="p1",
        occupation_status="parttime",
        stages=stages or [],
        event_overrides=overrides or {},
        jitter=JitterConfig(),
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_hhmm_valid(self):
        assert _parse_hhmm("07:30") == 7 * 60 + 30
        assert _parse_hhmm("00:00") == 0
        assert _parse_hhmm("23:59") == 23 * 60 + 59

    @pytest.mark.parametrize(
        "bad", ["morning", "", "abc", "24:00", "12:60", "no-colon", "ab:cd", None]
    )
    def test_parse_hhmm_invalid_returns_none(self, bad):
        if bad is None:
            assert _parse_hhmm(bad) is None  # type: ignore[arg-type]
        else:
            assert _parse_hhmm(bad) is None

    def test_stage_fires_on_explicit_date(self):
        target = _dt.date(2026, 6, 15)
        assert _stage_fires_on([], target, target) is True
        assert _stage_fires_on([], _dt.date(2026, 6, 14), target) is False

    def test_stage_fires_on_weekday(self):
        monday = _dt.date(2026, 6, 15)  # Monday
        assert _stage_fires_on(["Mon"], None, monday) is True
        assert _stage_fires_on(["Tue"], None, monday) is False
        assert _stage_fires_on([], None, monday) is False


# ---------------------------------------------------------------------------
# _resolve_event_def
# ---------------------------------------------------------------------------


class TestResolveEventDef:
    def test_no_override_returns_catalog(self):
        catalog = _make_event(
            "walking", health_task_iri=HEALTH_PREFIX + "schedule-a-30-minute-walk"
        )
        out = _resolve_event_def(catalog, None)
        assert out == catalog

    def test_override_replaces_per_event_duration(self):
        catalog = _make_event("walking")
        override = EventOverride(
            per_event_duration=DurationRange(min=30, max=60, unit="minutes")
        )
        out = _resolve_event_def(catalog, override)
        assert out.per_event_duration.min == 30
        assert out.per_event_duration.max == 60

    def test_override_replaces_iris(self):
        catalog = _make_event("walking", health_task_iri=HEALTH_PREFIX + "old-task")
        override = EventOverride(
            health_task_iri=HEALTH_PREFIX + "new-task",
            human_activity_iri=ACTIVITY_PREFIX + "new-activity",
        )
        out = _resolve_event_def(catalog, override)
        assert out.health_task_iri.endswith("new-task")
        assert out.human_activity_iri.endswith("new-activity")

    def test_override_replaces_temporal_patterns(self):
        catalog = _make_event("walking")
        override = EventOverride(
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality", details={"within": "morning", "amount": 20}
                )
            ]
        )
        out = _resolve_event_def(catalog, override)
        assert len(out.temporal_patterns) == 1
        assert out.temporal_patterns[0].mode == "seasonality"

    def test_override_unset_field_inherits_catalog_value(self):
        """An override whose `intensity` is None must NOT overwrite a
        catalog intensity."""
        catalog = _make_event("walking", intensity=3)
        override = EventOverride(
            per_event_duration=DurationRange(min=30, max=60, unit="minutes")
        )
        out = _resolve_event_def(catalog, override)
        assert out.intensity == 3

    def test_override_can_set_intensity_concurrent_and_concurrent_with(self):
        catalog = _make_event("walking")
        override = EventOverride(
            intensity=4,
            is_concurrent=True,
            is_dividable=True,
            concurrent_with=["yoga"],
        )
        out = _resolve_event_def(catalog, override)
        assert out.intensity == 4
        assert out.is_concurrent is True
        assert out.is_dividable is True
        assert out.concurrent_with == ["yoga"]

    def test_override_sets_every_remaining_field(self):
        """Cover the `requires` / `weekdays` / `ontology_iri` /
        `total_event_duration` / `total_event_episodes` /
        `calendar_variations` branches of `_resolve_event_def`."""
        from src.scripts.persona.config.schema import CalendarVariations, RolePredicate

        catalog = _make_event("walking")
        override = EventOverride(
            requires={"occupation_status": RolePredicate(in_=["parttime"])},
            weekdays=["Mon", "Wed", "Fri"],
            ontology_iri="https://example.org/legacy-iri",
            total_event_duration=TotalDuration(
                min=30, max=120, scale="day", unit="minutes"
            ),
            total_event_episodes=EpisodeRange(scale="day", min=1, max=2),
            calendar_variations=CalendarVariations(task_labels=["walk_task"]),
        )
        out = _resolve_event_def(catalog, override)
        assert out.weekdays == ["Mon", "Wed", "Fri"]
        assert out.ontology_iri == "https://example.org/legacy-iri"
        assert out.total_event_duration.min == 30
        assert out.total_event_episodes.max == 2
        assert out.calendar_variations.task_labels == ["walk_task"]
        assert "occupation_status" in out.requires


# ---------------------------------------------------------------------------
# PersonaConstraints
# ---------------------------------------------------------------------------


class TestPersonaConstraintsBasics:
    def test_constructor_resolves_all_catalog_events(self):
        ev_cfg = _make_event_config([_make_event("walking"), _make_event("cycling")])
        persona = _make_persona()
        pc = PersonaConstraints(
            persona=persona,
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
        )
        assert sorted(pc.event_names) == ["cycling", "walking"]

    def test_resolved_event_def_for_returns_none_for_unknown_label(self):
        ev_cfg = _make_event_config([_make_event("walking")])
        pc = PersonaConstraints(
            persona=_make_persona(),
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
        )
        assert pc.resolved_event_def_for("unknown") is None

    def test_resolved_event_def_overlays_override(self):
        ev_cfg = _make_event_config(
            [_make_event("walking", health_task_iri=HEALTH_PREFIX + "old")]
        )
        persona = _make_persona(
            overrides={
                "walking": EventOverride(
                    health_task_iri=HEALTH_PREFIX + "schedule-a-30-minute-walk",
                    per_event_duration=DurationRange(min=30, max=75, unit="minutes"),
                )
            }
        )
        pc = PersonaConstraints(
            persona=persona,
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
        )
        ev = pc.resolved_event_def_for("walking")
        assert ev is not None
        assert ev.health_task_iri.endswith("schedule-a-30-minute-walk")
        assert ev.per_event_duration.min == 30

    def test_all_resolved_events_returns_copy(self):
        ev_cfg = _make_event_config([_make_event("walking")])
        pc = PersonaConstraints(
            persona=_make_persona(),
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
        )
        snap1 = pc.all_resolved_events()
        snap2 = pc.all_resolved_events()
        assert snap1 is not snap2  # copy on read
        assert snap1 == snap2

    def test_horizon_properties_match_constructor_inputs(self):
        ev_cfg = _make_event_config([_make_event("walking")])
        persona = _make_persona()
        pc = PersonaConstraints(
            persona=persona,
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
            horizon_start_date=_dt.date(2026, 1, 1),
        )
        assert pc.horizon_days == 28
        assert pc.horizon_start_date == _dt.date(2026, 1, 1)
        assert pc.persona is persona


# ---------------------------------------------------------------------------
# day_constraints_for
# ---------------------------------------------------------------------------


class TestDayConstraints:
    def test_day_constraints_for_known_label(self):
        ev_cfg = _make_event_config([_make_event("walking")])
        pc = PersonaConstraints(
            persona=_make_persona(),
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
        )
        out = pc.day_constraints_for("walking", day_idx=0, day_of_week=0)
        assert out is not None
        assert out.per_event_min == 15
        assert out.per_event_max == 120

    def test_day_constraints_for_unknown_label_returns_none(self):
        ev_cfg = _make_event_config([_make_event("walking")])
        pc = PersonaConstraints(
            persona=_make_persona(),
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
        )
        assert pc.day_constraints_for("unknown", day_idx=0, day_of_week=0) is None

    def test_day_constraints_honours_horizon_start_date_for_month_pattern(self):
        ev = _make_event("walking")
        # Add a month-seasonality pattern via override.
        ev_cfg = _make_event_config([ev])
        persona = _make_persona(
            overrides={
                "walking": EventOverride(
                    temporal_patterns=[
                        TemporalPattern(
                            mode="seasonality",
                            details={
                                "scale": "month",
                                "within": ["June"],
                                "amount": 50,
                                "direction": "increasing",
                            },
                        )
                    ]
                )
            }
        )
        pc = PersonaConstraints(
            persona=persona,
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=180,
            horizon_start_date=_dt.date(2026, 1, 1),
        )
        june_day = pc.day_constraints_for("walking", day_idx=160, day_of_week=0)
        feb_day = pc.day_constraints_for("walking", day_idx=40, day_of_week=0)
        assert june_day is not None and feb_day is not None
        # June (boosted) base_count should be >= Feb (baseline) base_count.
        assert june_day.base_count >= feb_day.base_count


# ---------------------------------------------------------------------------
# stages_firing_on
# ---------------------------------------------------------------------------


class TestStagesFiringOn:
    def test_stage_with_hhmm_time_returns_explicit_window(self):
        stage = PersonaEventStage(
            name="walking", time="07:30", duration_minutes=45, days=["Mon"]
        )
        pc = PersonaConstraints(
            persona=_make_persona(stages=[stage]),
            event_config=_make_event_config([_make_event("walking")]),
            window_map=_wm(),
            horizon_days=28,
        )
        # Monday
        fired = pc.stages_firing_on(_dt.date(2026, 6, 15))
        assert len(fired) == 1
        assert fired[0].window_start == 7 * 60 + 30
        assert fired[0].window_end == 7 * 60 + 30 + 45
        assert fired[0].duration_minutes == 45

    def test_stage_with_window_token_returns_daypart_window(self):
        stage = PersonaEventStage(
            name="walking", time="morning", duration_minutes=45, days=["Mon"]
        )
        pc = PersonaConstraints(
            persona=_make_persona(stages=[stage]),
            event_config=_make_event_config([_make_event("walking")]),
            window_map=_wm(),
            horizon_days=28,
        )
        fired = pc.stages_firing_on(_dt.date(2026, 6, 15))
        assert fired[0].window_start == 400
        assert fired[0].window_end == 600

    def test_stage_without_time_returns_full_day_window(self):
        stage = PersonaEventStage(name="reading", days=["Mon"])
        pc = PersonaConstraints(
            persona=_make_persona(stages=[stage]),
            event_config=_make_event_config([_make_event("reading")]),
            window_map=_wm(),
            horizon_days=28,
        )
        fired = pc.stages_firing_on(_dt.date(2026, 6, 15))
        assert fired[0].window_start == 0
        assert fired[0].window_end == 1440

    def test_stage_fires_only_on_listed_weekdays(self):
        stage = PersonaEventStage(
            name="walking", time="morning", duration_minutes=45, days=["Mon", "Wed"]
        )
        pc = PersonaConstraints(
            persona=_make_persona(stages=[stage]),
            event_config=_make_event_config([_make_event("walking")]),
            window_map=_wm(),
            horizon_days=28,
        )
        monday = pc.stages_firing_on(_dt.date(2026, 6, 15))  # Mon
        tuesday = pc.stages_firing_on(_dt.date(2026, 6, 16))  # Tue
        assert len(monday) == 1
        assert tuesday == []

    def test_stage_fires_on_exact_date(self):
        target = _dt.date(2026, 6, 15)
        stage = PersonaEventStage(
            name="dentist", time="14:00", duration_minutes=30, date=target
        )
        pc = PersonaConstraints(
            persona=_make_persona(stages=[stage]),
            event_config=_make_event_config([_make_event("walking")]),  # any catalog
            window_map=_wm(),
            horizon_days=28,
        )
        assert len(pc.stages_firing_on(target)) == 1
        assert pc.stages_firing_on(target + _dt.timedelta(days=1)) == []

    def test_stage_with_unknown_window_token_falls_back_to_full_day(self):
        """A stage `time` that is neither HH:MM nor a recognised window
        falls through to the full-day default."""
        # Build a fresh persona stage but bypass the schema validation
        # so we can test the resolver's fallback path; we construct via
        # model_construct to skip the `_validate_time_token` check.
        stage = PersonaEventStage.model_construct(
            name="weird", time="elevenses", duration_minutes=30, days=["Mon"]
        )
        pc = PersonaConstraints(
            persona=_make_persona(stages=[stage]),
            event_config=_make_event_config([_make_event("weird")]),
            window_map=_wm(),
            horizon_days=28,
        )
        fired = pc.stages_firing_on(_dt.date(2026, 6, 15))
        assert fired[0].window_start == 0
        assert fired[0].window_end == 1440

    def test_stage_copies_iris(self):
        stage = PersonaEventStage(
            name="walking",
            time="morning",
            days=["Mon"],
            human_activity_iri=ACTIVITY_PREFIX + "walking",
            health_task_iri=HEALTH_PREFIX + "schedule-a-walk",
        )
        pc = PersonaConstraints(
            persona=_make_persona(stages=[stage]),
            event_config=_make_event_config([_make_event("walking")]),
            window_map=_wm(),
            horizon_days=28,
        )
        fired = pc.stages_firing_on(_dt.date(2026, 6, 15))
        assert fired[0].human_activity_iri.endswith("/walking")
        assert fired[0].health_task_iri.endswith("/schedule-a-walk")

    def test_no_stages_returns_empty(self):
        pc = PersonaConstraints(
            persona=_make_persona(stages=[]),
            event_config=_make_event_config([_make_event("walking")]),
            window_map=_wm(),
            horizon_days=28,
        )
        assert pc.stages_firing_on(_dt.date(2026, 6, 15)) == []


# ---------------------------------------------------------------------------
# matched_events_for
# ---------------------------------------------------------------------------


class TestMatchedEventsFor:
    def test_no_mapper_returns_empty(self):
        pc = PersonaConstraints(
            persona=_make_persona(),
            event_config=_make_event_config([_make_event("walking")]),
            window_map=_wm(),
            horizon_days=28,
        )
        task = RecommendedTask(
            label="schedule-a-30-minute-walk",
            duration_min=30,
            duration_max=30,
        )
        assert pc.matched_events_for(task) == []

    def test_delegates_to_mapper_with_experiment_and_scenario(self):
        cache = PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))
        mapper = PreferenceMapper(cache=cache)
        ev_cfg = _make_event_config(
            [
                _make_event(
                    "walking",
                    health_task_iri=HEALTH_PREFIX + "schedule-a-30-minute-walk",
                )
            ]
        )
        pc = PersonaConstraints(
            persona=_make_persona(),
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
            mapper=mapper,
            experiment="exp_b",
            scenario="nutrition_l1",
        )
        task = RecommendedTask(
            label="schedule-a-30-minute-walk",
            duration_min=30,
            duration_max=30,
            ontology_uri=HEALTH_PREFIX + "schedule-a-30-minute-walk",
        )
        out = pc.matched_events_for(task)
        assert len(out) == 1
        assert out[0].event_name == "walking"
        assert out[0].weight == 1.0
        # Sanity: the cache was hit (HASH key matches our experiment/scenario).
        all_ = cache.all_fields("exp_b", "nutrition_l1")
        assert ("schedule-a-30-minute-walk", "walking") in all_

    def test_returns_no_match_when_mapper_returns_zero_weight(self):
        cache = PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))
        mapper = MagicMock(spec=PreferenceMapper)
        mapper.matched_events.return_value = []
        ev_cfg = _make_event_config([_make_event("walking")])
        pc = PersonaConstraints(
            persona=_make_persona(),
            event_config=ev_cfg,
            window_map=_wm(),
            horizon_days=28,
            mapper=mapper,
            experiment="e",
            scenario="s",
        )
        task = RecommendedTask(label="any", duration_min=10, duration_max=10)
        assert pc.matched_events_for(task) == []
        assert mapper.matched_events.called
        # Ensure experiment/scenario were forwarded.
        kwargs = mapper.matched_events.call_args.kwargs
        assert kwargs["experiment"] == "e"
        assert kwargs["scenario"] == "s"


class TestResolvedStage:
    def test_frozen_dataclass_equality(self):
        a = ResolvedStage(name="walking", window_start=0, window_end=120)
        b = ResolvedStage(name="walking", window_start=0, window_end=120)
        assert a == b

    def test_optional_fields_default_to_none(self):
        s = ResolvedStage(name="walking", window_start=0, window_end=120)
        assert s.duration_minutes is None
        assert s.human_activity_iri is None
        assert s.health_task_iri is None


# ---------------------------------------------------------------------------
# Mapped event re-export sanity
# ---------------------------------------------------------------------------


def test_mapped_event_re_export_still_works():
    """Smoke test; `MappedEvent` is the public type the scorers consume."""
    me = MappedEvent(
        event_name="walking", weight=1.0, source_tier="literal", evidence={}
    )
    assert me.event_name == "walking"
