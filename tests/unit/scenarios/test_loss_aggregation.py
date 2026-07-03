"""Per-leg aggregation helpers in `loss.py`."""

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
    Persona,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.domain.calendar import AugmentedCalendar
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.loss import (
    LegStats,
    _leg_stats_from_buckets,
    compute_l_pref_v2,
)
from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
from src.scripts.scenarios.metrics.preference_constraints import PersonaConstraints
from src.scripts.scenarios.metrics.preference_mapping import PreferenceMapper

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
    temporal_patterns: list[TemporalPattern] | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category="sports",
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=2),
        health_task_iri=health_task_iri,
        temporal_patterns=temporal_patterns or [],
    )


def _pc(
    events: list[EventDefinition],
    *,
    mapper: PreferenceMapper | None = None,
) -> PersonaConstraints:
    return PersonaConstraints(
        persona=Persona(id="p1", occupation_status="parttime", stages=[]),
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


def test_leg_stats_from_buckets_computes_mape_when_supplied():
    """Non-empty `mape_values` produces a populated mean."""
    stats = _leg_stats_from_buckets("leg", [0.1, 0.2], mape_values=[10.0, 30.0])
    assert isinstance(stats, LegStats)
    assert stats.mean_loss == pytest.approx(0.15)
    assert stats.mape == pytest.approx(20.0)
    assert stats.applicable_count == 2


def test_leg_stats_from_buckets_leaves_mape_none_for_empty_list():
    """Empty `mape_values` keeps mape `None` even when `values` is populated."""
    stats = _leg_stats_from_buckets("leg", [0.5], mape_values=[])
    assert stats.mape is None


def test_compute_l_pref_v2_skips_per_task_legs_for_unscheduled_task():
    """A recommended task with no scheduled placement is silently skipped."""
    ev = _make_event(
        "walking",
        health_task_iri=HEALTH_PREFIX + "schedule-a-walk",
    )
    pc = _pc([ev], mapper=_mapper_with_literal_hit())
    placed = _desired("schedule-a-walk", ontology_uri=HEALTH_PREFIX + "schedule-a-walk")
    unscheduled_task = _desired(
        "another-walk", ontology_uri=HEALTH_PREFIX + "schedule-a-walk"
    )
    placed_sched = _scheduled(placed, 480, 510)
    solution = _solution([placed, unscheduled_task], [placed_sched])
    out, rows, leg_stats = compute_l_pref_v2(
        solution, persona_constraints=pc, window_map=_wm()
    )
    assert out is not None
    assert isinstance(leg_stats, list)


def test_compute_l_pref_v2_trend_pattern_populates_pattern_mape_by_mode():
    """A `trend` pattern produces rows whose mape feeds the per-mode aggregate."""
    ev = _make_event(
        "walking",
        health_task_iri=HEALTH_PREFIX + "schedule-a-walk",
        temporal_patterns=[
            TemporalPattern(
                mode="trend",
                details={
                    "target": "episodes",
                    "scale": "week",
                    "direction": "increasing",
                    "amount": 1,
                    "start": 1,
                    "end": 4,
                },
            )
        ],
    )
    pc = _pc([ev], mapper=_mapper_with_literal_hit())
    task = _desired("schedule-a-walk", ontology_uri=HEALTH_PREFIX + "schedule-a-walk")
    s1 = _scheduled(task, 480, 510)
    s2 = _scheduled(task, 540, 600)
    solution = _solution([task], [s1, s2])
    out, rows, leg_stats = compute_l_pref_v2(
        solution, persona_constraints=pc, window_map=_wm()
    )
    assert out is not None
    trend_leg = next((s for s in leg_stats if s.name == "temporal_pattern_trend"), None)
    assert trend_leg is not None
    assert trend_leg.mape is not None
