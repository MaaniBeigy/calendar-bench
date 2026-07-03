"""Tests for `compute_l_pref_v2` (the L_pref aggregator)."""

from __future__ import annotations

import datetime as _dt

import fakeredis
import pytest

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    EventOverride,
    Persona,
    PersonaEventStage,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.domain.calendar import AugmentedCalendar
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.loss import (
    LossComponents,
    SchedulingLoss,
    compute_l_pref_v2,
)
from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
from src.scripts.scenarios.metrics.preference_constraints import PersonaConstraints
from src.scripts.scenarios.metrics.preference_mapping import PreferenceMapper
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

HEALTH_PREFIX = "https://w3id.org/calendar-bench/health/task/"


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
    health_task_iri: str | None = None,
    per_event_min: int = 15,
    per_event_max: int = 120,
    total_min: int = 15,
    total_max: int = 240,
    total_scale: str = "day",
    episodes_min: int = 0,
    episodes_max: int = 1,
    temporal_patterns: list[TemporalPattern] | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category="sports",
        per_event_duration=DurationRange(
            min=per_event_min, max=per_event_max, unit="minutes"
        ),
        total_event_duration=TotalDuration(
            min=total_min, max=total_max, scale=total_scale, unit="minutes"
        ),
        total_event_episodes=EpisodeRange(
            scale="day", min=episodes_min, max=episodes_max
        ),
        health_task_iri=health_task_iri,
        temporal_patterns=temporal_patterns or [],
    )


def _persona(stages: list[PersonaEventStage] | None = None) -> Persona:
    return Persona(
        id="p1",
        occupation_status="parttime",
        stages=stages or [],
    )


def _pc(
    events: list[EventDefinition],
    *,
    persona: Persona | None = None,
    mapper: PreferenceMapper | None = None,
) -> PersonaConstraints:
    return PersonaConstraints(
        persona=persona or _persona(),
        event_config=EventConfig(
            categories={
                "sports": Category(name="sports", events={ev.name: ev for ev in events})
            }
        ),
        window_map=_wm(),
        horizon_days=28,
        horizon_start_date=_dt.date(2026, 6, 1),
        mapper=mapper,
    )


def _desired(label: str, *, ontology_uri: str | None = None) -> RecommendedTask:
    return RecommendedTask(
        label=label,
        duration_min=10,
        duration_max=60,
        ontology_uri=ontology_uri,
    )


def _scheduled(
    task: RecommendedTask, start: int, end: int, date: _dt.date | None = None
) -> ScheduledTask:
    return ScheduledTask(
        task=task,
        start_minutes=start,
        end_minutes=end,
        is_standalone=True,
        concurrent_with=None,
        date=date or _dt.date(2026, 6, 1),
    )


def _solution(
    tasks: list[RecommendedTask], scheduled: list[ScheduledTask]
) -> SchedulingSolution:
    scheduled_labels = {st.task.label for st in scheduled}
    return SchedulingSolution(
        person_id="p1",
        augmented_calendar=AugmentedCalendar(person_id="p1"),
        tasks=tasks,
        scheduled=scheduled,
        unscheduled=[t for t in tasks if t.label not in scheduled_labels],
    )


def _mapper_with_literal_hit() -> PreferenceMapper:
    cache = PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))
    return PreferenceMapper(cache=cache)


# ---------------------------------------------------------------------------
# Aggregator basics
# ---------------------------------------------------------------------------


class TestAggregatorBasics:
    def test_returns_none_when_no_tasks(self):
        pc = _pc([_make_event("walking")])
        solution = _solution([], [])
        out, rows, _ = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=_wm()
        )
        assert out is None
        assert rows == []

    def test_returns_none_when_no_legs_have_signal(self):
        """Task with no matching event and no firing stages to no legs."""
        pc = _pc([_make_event("walking")])
        task = _desired("unknown-task", ontology_uri=None)
        sched = _scheduled(task, 480, 510)
        solution = _solution([task], [sched])
        out, rows, _ = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=_wm()
        )
        assert out is None
        assert rows == []

    def test_perfect_alignment_returns_zero(self):
        """Literal IRI match + duration inside band + no other legs to 0."""
        ev = _make_event(
            "walking",
            health_task_iri=HEALTH_PREFIX + "schedule-a-walk",
            per_event_min=15,
            per_event_max=60,
        )
        pc = _pc([ev], mapper=_mapper_with_literal_hit())
        task = _desired(
            "schedule-a-walk", ontology_uri=HEALTH_PREFIX + "schedule-a-walk"
        )
        sched = _scheduled(task, 480, 510)  # 30 min, inside [15, 60]
        solution = _solution([task], [sched])
        out, _, _ = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=_wm()
        )
        assert out == 0.0

    def test_duration_violation_surfaces(self):
        ev = _make_event(
            "walking",
            health_task_iri=HEALTH_PREFIX + "schedule-a-walk",
            per_event_min=30,
            per_event_max=60,
            total_min=30,
            total_max=60,
        )
        pc = _pc([ev], mapper=_mapper_with_literal_hit())
        task = _desired(
            "schedule-a-walk", ontology_uri=HEALTH_PREFIX + "schedule-a-walk"
        )
        sched = _scheduled(task, 480, 495)  # 15 min, below per_event 30
        solution = _solution([task], [sched])
        out, _, _ = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=_wm()
        )
        # Legs that fire with signal:
        #   per_occurrence_duration: 15 < 30 to (30-15)/30 = 0.5
        #   per_scale_duration (day total): 15 < 30 to (30-15)/30 = 0.5
        #   per_scale_episodes: 1 episode, band [0, 1] to 0.0 (inside band)
        # No temporal patterns, no firing stages.
        # Mean = (0.5 + 0.5 + 0.0) / 3 = 1.0/3
        assert out == pytest.approx(1.0 / 3)

    def test_temporal_pattern_rows_surfaced(self):
        ev = _make_event(
            "walking",
            health_task_iri=HEALTH_PREFIX + "walk",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        pc = _pc([ev], mapper=_mapper_with_literal_hit())
        task = _desired("walk", ontology_uri=HEALTH_PREFIX + "walk")
        sched = _scheduled(task, 1000, 1030)  # afternoon to outside morning
        solution = _solution([task], [sched])
        loss, rows, _ = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=_wm()
        )
        assert loss is not None
        assert any(r.mode == "fix" for r in rows)

    def test_persona_stage_clash_contributes_to_score(self):
        ev = _make_event(
            "reading",
            health_task_iri=HEALTH_PREFIX + "read-a-book",
        )
        # The persona reads in the morning; we'll schedule a "swim" task
        # on top, which clashes.
        stage = PersonaEventStage(
            name="reading", time="morning", duration_minutes=120, days=["Mon"]
        )
        persona = _persona(stages=[stage])
        pc = _pc([ev], persona=persona, mapper=_mapper_with_literal_hit())
        task = _desired("swim_laps", ontology_uri=HEALTH_PREFIX + "swim-laps")
        sched = _scheduled(task, 420, 480, _dt.date(2026, 6, 1))  # Monday morning
        solution = _solution([task], [sched])
        semantic = SemanticCompatibility(matrix={("swim_laps", "reading"): 0.1})
        out, _, _ = compute_l_pref_v2(
            solution,
            persona_constraints=pc,
            window_map=_wm(),
            semantic=semantic,
        )
        assert out is not None
        assert out > 0  # clash contributes


# ---------------------------------------------------------------------------
# Dividable task; multiple placements per RecommendedTask
# ---------------------------------------------------------------------------


class TestDividableTask:
    def test_per_scale_duration_aggregates_multiple_placements_under_same_label(self):
        """Two 20-min placements should sum to 40 min/day; if the band
        is [50, 100], both occurrences together still under-fill."""
        ev = _make_event(
            "walking",
            health_task_iri=HEALTH_PREFIX + "walk",
            per_event_min=10,
            per_event_max=60,
            total_min=50,
            total_max=100,
        )
        pc = _pc([ev], mapper=_mapper_with_literal_hit())
        task = _desired("walk", ontology_uri=HEALTH_PREFIX + "walk")
        scheds = [
            _scheduled(task, 480, 500),  # 20 min
            _scheduled(task, 600, 620),  # 20 min
        ]
        solution = _solution([task], scheds)
        out, _, _ = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=_wm()
        )
        # Legs that fire:
        #   per_occurrence (×2): both 20 min in [10, 60] to 0, 0
        #   per_scale_duration: day total 40 < 50 to 0.2
        #   per_scale_episodes: 2 placements, default day band [0, 1]
        #     to (2-1)/1 = 1.0 (clipped)
        # Mean = (0 + 0 + 0.2 + 1.0) / 4 = 0.3
        assert out == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# SchedulingLoss dispatcher integration
# ---------------------------------------------------------------------------


class TestSchedulingLossDispatch:
    def test_pref_is_none_when_constraints_omitted(self):
        from src.scripts.scenarios.config.schema import LossWeights
        from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
        from src.scripts.scenarios.metrics.intensity_resolver import (
            IntensityResolver,
            MetQuartiles,
        )

        weights = LossWeights(
            lambda_cov=0.2,
            lambda_cal=0.2,
            lambda_pref=0.13,
            lambda_disp=0.1,
            lambda_merge=0.2,
            lambda_spread=0.1,
            lambda_divide=0.07,
        )
        loss = SchedulingLoss(
            weights=weights,
            semantic=SemanticCompatibility(matrix={}),
            ruleset=RuleSet(rules=[]),
            matcher=SelectorMatcher(),
            resolver=IntensityResolver(
                quartiles=MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8)
            ),
        )
        from src.scripts.scenarios.domain.calendar import CalendarTrace

        solution = SchedulingSolution(
            person_id="p1",
            augmented_calendar=AugmentedCalendar(person_id="p1"),
            tasks=[],
            scheduled=[],
            unscheduled=[],
        )
        cal_trace = CalendarTrace(person_id="p1", events=[])
        _total, comp = loss.compute(solution, cal_trace, horizon_epochs=28)
        # Without persona_constraints + window_map the L_pref leg is N/A.
        assert comp.pref is None

    def test_dispatch_to_v2_when_constraints_supplied(self):
        from src.scripts.scenarios.config.schema import LossWeights
        from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
        from src.scripts.scenarios.metrics.intensity_resolver import (
            IntensityResolver,
            MetQuartiles,
        )

        weights = LossWeights(
            lambda_cov=0.2,
            lambda_cal=0.2,
            lambda_pref=0.13,
            lambda_disp=0.1,
            lambda_merge=0.2,
            lambda_spread=0.1,
            lambda_divide=0.07,
        )
        loss = SchedulingLoss(
            weights=weights,
            semantic=SemanticCompatibility(matrix={}),
            ruleset=RuleSet(rules=[]),  # noqa: F841; empty rule list is fine
            matcher=SelectorMatcher(),
            resolver=IntensityResolver(
                quartiles=MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8)
            ),
        )
        from src.scripts.scenarios.domain.calendar import CalendarTrace

        ev = _make_event(
            "walking",
            health_task_iri=HEALTH_PREFIX + "walk",
            per_event_min=15,
            per_event_max=60,
        )
        pc = _pc([ev], mapper=_mapper_with_literal_hit())
        task = _desired("walk", ontology_uri=HEALTH_PREFIX + "walk")
        sched = _scheduled(task, 480, 510)
        solution = _solution([task], [sched])
        cal_trace = CalendarTrace(person_id="p1", events=[])
        total, comp = loss.compute(
            solution,
            cal_trace,
            horizon_epochs=28,
            persona_constraints=pc,
            window_map=_wm(),
        )
        assert comp.pref == 0.0  # perfectly aligned


# ---------------------------------------------------------------------------
# LossComponents; pref now accepts None
# ---------------------------------------------------------------------------


class TestLossComponentsPrefNone:
    def test_components_accept_none_pref(self):
        from src.scripts.scenarios.config.schema import LossWeights

        comp = LossComponents(
            cov=0.0,
            cal=0.0,
            pref=None,  # newly allowed
            disp=0.0,
            merge=0.0,
            spread=0.0,
            divide=None,
        )
        weights = LossWeights(
            lambda_cov=0.2,
            lambda_cal=0.2,
            lambda_pref=0.13,
            lambda_disp=0.1,
            lambda_merge=0.2,
            lambda_spread=0.1,
            lambda_divide=0.07,
        )
        # pref dropped from sum to all-zero remaining components to weighted = 0.
        assert comp.weighted(weights) == 0.0
