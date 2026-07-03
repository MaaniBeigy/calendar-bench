"""Regression tests for the evaluate oracle split and weekly-chunk alignment.

L_pref uses the embedding oracle and L_merge uses the judge; weekly chunks are
consecutive calendar weeks anchored at the horizon start.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.scenarios.config.schema import LossWeights
from src.scripts.scenarios.domain.calendar import AugmentedCalendar, CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityResolver,
    MetQuartiles,
)
from src.scripts.scenarios.metrics.loss import SchedulingLoss
from src.scripts.scenarios.metrics.preference_constraints import PersonaConstraints
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility
from tests.unit.scenarios.conftest import make_scheduled, make_task

DATE = _dt.date(2026, 6, 1)  # a Monday; ISO week 23


class _JudgeNoStrategy:
    """Mimics LLMJudgeOracle: exposes `score` but NOT `score_with_strategy`."""

    def __init__(self) -> None:
        self.score_calls = 0

    def score(self, label_a: str, label_b: str) -> float:
        self.score_calls += 1
        return 0.5


def _resolver() -> IntensityResolver:
    return IntensityResolver(
        quartiles=MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8)
    )


def _weights() -> LossWeights:
    return LossWeights(
        lambda_cov=0.2,
        lambda_cal=0.2,
        lambda_pref=0.13,
        lambda_disp=0.1,
        lambda_merge=0.2,
        lambda_spread=0.1,
        lambda_divide=0.07,
    )


# ---------------------------------------------------------------------------
# Bug 1: merge_semantic split
# ---------------------------------------------------------------------------


def test_merge_semantic_defaults_to_semantic():
    """Without `merge_semantic`, the merge leg shares the pref oracle."""
    embedding = SemanticCompatibility(matrix={})
    loss = SchedulingLoss(
        weights=_weights(),
        semantic=embedding,
        ruleset=RuleSet(rules=[]),
        matcher=SelectorMatcher(),
        resolver=_resolver(),
    )
    assert loss._merge_semantic is embedding


def test_merge_semantic_overrides_only_merge():
    """A supplied `merge_semantic` is stored separately from `semantic`."""
    embedding = SemanticCompatibility(matrix={})
    judge = _JudgeNoStrategy()
    loss = SchedulingLoss(
        weights=_weights(),
        semantic=embedding,
        ruleset=RuleSet(rules=[]),
        matcher=SelectorMatcher(),
        resolver=_resolver(),
        merge_semantic=judge,
    )
    assert loss._semantic is embedding
    assert loss._merge_semantic is judge


def test_pref_leg_uses_embedding_not_judge():
    """L_pref uses the embedding `semantic`, not the judge in `merge_semantic`."""
    # A persona stage that overlaps the placement so the persona-stage
    # semantic sub-leg fires and calls `score_with_strategy`.
    from src.scripts.persona.config.schema import (
        Category,
        DurationRange,
        EpisodeRange,
        EventConfig,
        EventDefinition,
        Persona,
        PersonaEventStage,
        TotalDuration,
        WindowRange,
    )
    from src.scripts.persona.domain.time_windows import WindowMap

    window_map = WindowMap.from_config(
        {
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
        }
    )
    stage = PersonaEventStage(
        name="reading", time="morning", duration_minutes=120, days=["Mon"]
    )
    event = EventDefinition(
        name="reading",
        category="sports",
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
    )
    persona_constraints = PersonaConstraints(
        persona=Persona(id="p1", occupation_status="parttime", stages=[stage]),
        event_config=EventConfig(
            categories={"sports": Category(name="sports", events={"reading": event})}
        ),
        window_map=window_map,
        horizon_days=28,
        horizon_start_date=DATE,
        mapper=None,
    )

    task = make_task(label="swim_laps", duration_min=30, duration_max=60)
    sched = make_scheduled(task=task, start_minutes=420, end_minutes=480, date=DATE)
    solution = SchedulingSolution(
        person_id="p1",
        augmented_calendar=AugmentedCalendar(person_id="p1"),
        tasks=[task],
        scheduled=[sched],
        unscheduled=[],
    )

    # The pref oracle scores the clash; the judge has no `score_with_strategy`.
    embedding = SemanticCompatibility(matrix={("swim_laps", "reading"): 0.1})
    judge = _JudgeNoStrategy()
    loss = SchedulingLoss(
        weights=_weights(),
        semantic=embedding,
        ruleset=RuleSet(rules=[]),
        matcher=SelectorMatcher(),
        resolver=_resolver(),
        merge_semantic=judge,
    )

    # A judge with no `score_with_strategy` must not reach the pref leg.
    _total, comp = loss.compute(
        solution,
        CalendarTrace(person_id="p1", events=[]),
        horizon_epochs=28,
        persona_constraints=persona_constraints,
        window_map=window_map,
    )
    assert comp.pref is not None and comp.pref > 0  # the morning clash registered


# ---------------------------------------------------------------------------
# Bug 2: calendar-aligned weekly chunks
# ---------------------------------------------------------------------------


def _solution_on_dates(dates: list[_dt.date]) -> SchedulingSolution:
    sched = [make_scheduled(date=d) for d in dates]
    return SchedulingSolution(
        person_id="p1",
        augmented_calendar=AugmentedCalendar(person_id="p1"),
        tasks=[],
        scheduled=sched,
        unscheduled=[],
    )


def test_weekly_chunks_are_calendar_aligned_not_sparse():
    """Sparse placements over 8 calendar weeks group into 8 calendar-week chunks."""
    from src.scripts.scenarios.cli import _resolve_weekly_chunks_for_solution

    # Two placements per week (day 0 and day 6) across 8 weeks.
    dates = [DATE + _dt.timedelta(days=7 * w) for w in range(8)] + [
        DATE + _dt.timedelta(days=7 * w + 6) for w in range(8)
    ]
    solution = _solution_on_dates(dates)
    chunks = _resolve_weekly_chunks_for_solution(solution, horizon_start_date=DATE)
    assert len(chunks) == 8
    assert chunks[0][0] == DATE
    for chunk in chunks:
        assert len(chunk) == 7
        for i in range(1, len(chunk)):
            assert (chunk[i] - chunk[i - 1]).days == 1
    # Consecutive weeks are 7 calendar days apart.
    assert (chunks[1][0] - chunks[0][0]).days == 7


def test_weekly_chunks_fall_back_to_first_placement():
    """With no horizon start, weeks anchor at the earliest placement date."""
    from src.scripts.scenarios.cli import _resolve_weekly_chunks_for_solution

    dates = [DATE + _dt.timedelta(days=d) for d in (0, 8, 13)]
    chunks = _resolve_weekly_chunks_for_solution(_solution_on_dates(dates))
    assert chunks[0][0] == DATE
    assert len(chunks) == 2  # 14-day span, 2 weeks


def test_weekly_chunks_anchor_at_horizon_even_when_first_placement_later():
    """An empty leading week is preserved so trajectories align to the horizon."""
    from src.scripts.scenarios.cli import _resolve_weekly_chunks_for_solution

    # First placement in week 2; horizon starts a week earlier.
    dates = [DATE + _dt.timedelta(days=d) for d in (7, 10)]
    chunks = _resolve_weekly_chunks_for_solution(
        _solution_on_dates(dates), horizon_start_date=DATE
    )
    assert chunks[0][0] == DATE  # week 1 kept even though it has no placements
    assert len(chunks) == 2


def test_weekly_chunks_empty_solution():
    """Nothing scheduled returns no weeks."""
    from src.scripts.scenarios.cli import _resolve_weekly_chunks_for_solution

    assert _resolve_weekly_chunks_for_solution(_solution_on_dates([])) == []
