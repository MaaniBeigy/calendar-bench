"""_eligible_pairs and compute_l_cal widen for (task x context) when context rules exist."""

from __future__ import annotations

import datetime

import pytest

from src.scripts.persona.config.schema import AllenPairRule, SelectorPredicate
from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
from src.scripts.scenarios.metrics.loss import (
    _eligible_pairs,
    _rules_reference_context,
    _temporal_gap_task_context,
    compute_l_cal,
)

DATE = datetime.date(2026, 5, 4)


def _scheduled() -> ScheduledTask:
    return ScheduledTask(
        task=RecommendedTask(label="tea-time", duration_min=10, duration_max=20),
        start_minutes=540,
        end_minutes=560,
        is_standalone=True,
        concurrent_with=None,
        date=DATE,
    )


def _ctx(start: int = 480, end: int = 600) -> ContextEpisode:
    return ContextEpisode(
        name="tired",
        category="energy_state",
        date=DATE,
        start_minutes=start,
        end_minutes=end,
    )


def _solution_with(scheduled: list[ScheduledTask]) -> SchedulingSolution:
    aug = AugmentedCalendar(person_id="p1", scheduled_tasks=scheduled)
    return SchedulingSolution(
        person_id="p1",
        augmented_calendar=aug,
        tasks=[s.task for s in scheduled],
        scheduled=scheduled,
        unscheduled=[],
    )


def test_rules_reference_context_detects_kind_context_endpoint() -> None:
    rules = [
        AllenPairRule(
            id="r1",
            event_a=SelectorPredicate(name="tea-time"),
            event_b=SelectorPredicate(kind="context", name="tired"),
            admissible_relations=["p", "P"],
        )
    ]
    assert _rules_reference_context(rules) is True


def test_rules_reference_context_returns_false_when_no_context_rules() -> None:
    rules = [
        AllenPairRule(
            id="r1",
            event_a=SelectorPredicate(name="tea-time"),
            event_b=SelectorPredicate(name="lunch"),
            admissible_relations=["p", "P"],
        )
    ]
    assert _rules_reference_context(rules) is False


def test_eligible_pairs_excludes_contexts_when_flag_off() -> None:
    cal = CalendarTrace(
        person_id="p1",
        events=[
            CalendarEvent(label="lunch", start_minutes=720, end_minutes=765, date=DATE)
        ],
        contexts=[_ctx()],
    )
    pairs = _eligible_pairs(_solution_with([_scheduled()]), cal, include_contexts=False)
    assert all(not isinstance(b, ContextEpisode) for _, b in pairs)


def test_eligible_pairs_includes_task_context_pair_when_flag_on() -> None:
    cal = CalendarTrace(
        person_id="p1",
        events=[],
        contexts=[_ctx()],
    )
    pairs = _eligible_pairs(_solution_with([_scheduled()]), cal, include_contexts=True)
    assert any(isinstance(b, ContextEpisode) for _, b in pairs)


def test_temporal_gap_task_context_zero_when_overlapping() -> None:
    task = _scheduled()  # 540..560
    ep = _ctx(start=500, end=560)  # overlaps
    assert _temporal_gap_task_context(task, ep) == 0.0


def test_temporal_gap_task_context_positive_when_disjoint() -> None:
    task = _scheduled()  # 540..560
    ep = _ctx(start=600, end=700)
    assert _temporal_gap_task_context(task, ep) == 40.0


def test_compute_l_cal_widens_when_rule_uses_kind_context() -> None:
    """A context-aware Allen rule routes the (task, context) pair through compute_l_cal."""
    rules = [
        AllenPairRule(
            id="tired_separated_from_tea",
            event_a=SelectorPredicate(kind="task", name="tea-time"),
            event_b=SelectorPredicate(kind="context", name="tired"),
            admissible_relations=["p", "P"],
            buffer=30,
        )
    ]
    cal = CalendarTrace(
        person_id="p1",
        events=[],
        contexts=[_ctx(start=540, end=560)],
    )
    solution = _solution_with([_scheduled()])
    loss = compute_l_cal(solution, cal, RuleSet(rules), SelectorMatcher())
    assert loss > 0.0


def test_compute_l_cal_unchanged_when_no_context_rule() -> None:
    """Without context selectors, contexts in the trace do not enter L_cal."""
    rules = [
        AllenPairRule(
            id="r1",
            event_a=SelectorPredicate(name="tea-time"),
            event_b=SelectorPredicate(name="lunch"),
            admissible_relations=[
                "p",
                "P",
                "d",
                "D",
                "o",
                "O",
                "s",
                "S",
                "f",
                "F",
                "e",
                "m",
                "M",
            ],
        )
    ]
    cal = CalendarTrace(
        person_id="p1",
        events=[
            CalendarEvent(label="lunch", start_minutes=720, end_minutes=765, date=DATE)
        ],
        contexts=[_ctx()],
    )
    loss = compute_l_cal(
        _solution_with([_scheduled()]), cal, RuleSet(rules), SelectorMatcher()
    )
    assert loss == pytest.approx(0.0)
