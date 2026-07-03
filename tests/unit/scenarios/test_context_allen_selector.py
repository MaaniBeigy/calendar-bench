"""SelectorMatcher routes by SelectorPredicate.kind across event/task/context candidates."""

from __future__ import annotations

import datetime

from src.scripts.persona.config.schema import SelectorPredicate
from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import SelectorMatcher

DATE = datetime.date(2026, 5, 4)


def _context(name: str = "tired", category: str = "energy_state") -> ContextEpisode:
    return ContextEpisode(
        name=name,
        category=category,
        date=DATE,
        start_minutes=480,
        end_minutes=540,
    )


def _event(label: str = "lunch") -> CalendarEvent:
    return CalendarEvent(
        label=label,
        start_minutes=720,
        end_minutes=765,
        date=DATE,
    )


def _task(label: str = "tea-time") -> RecommendedTask:
    return RecommendedTask(label=label, duration_min=10, duration_max=20)


def _scheduled(task: RecommendedTask) -> ScheduledTask:
    return ScheduledTask(
        task=task,
        start_minutes=540,
        end_minutes=560,
        is_standalone=True,
        concurrent_with=None,
        date=DATE,
    )


def test_kind_context_matches_context_episode() -> None:
    sel = SelectorPredicate(kind="context", name="tired")
    matcher = SelectorMatcher()
    assert matcher.matches(sel, _context("tired")) is True
    assert matcher.matches(sel, _context("happy")) is False


def test_kind_context_rejects_calendar_event() -> None:
    sel = SelectorPredicate(kind="context", name="lunch")
    matcher = SelectorMatcher()
    assert matcher.matches(sel, _event("lunch")) is False


def test_kind_context_rejects_scheduled_task() -> None:
    sel = SelectorPredicate(kind="context", name="tea-time")
    matcher = SelectorMatcher()
    assert matcher.matches(sel, _scheduled(_task())) is False


def test_kind_task_matches_desired_and_scheduled_task() -> None:
    sel = SelectorPredicate(kind="task", name="tea-time")
    matcher = SelectorMatcher()
    assert matcher.matches(sel, _task("tea-time")) is True
    assert matcher.matches(sel, _scheduled(_task("tea-time"))) is True


def test_kind_task_rejects_calendar_event_and_context() -> None:
    sel = SelectorPredicate(kind="task", name="lunch")
    matcher = SelectorMatcher()
    assert matcher.matches(sel, _event("lunch")) is False
    assert matcher.matches(sel, _context("lunch")) is False


def test_default_kind_event_rejects_context_episode() -> None:
    sel = SelectorPredicate(name="lunch")  # default kind="event"
    matcher = SelectorMatcher()
    assert matcher.matches(sel, _context("lunch")) is False


def test_default_kind_event_still_matches_calendar_event_and_tasks_for_backcompat() -> (
    None
):
    """Default `kind=event` keeps the historical loose typing across events and tasks."""
    sel = SelectorPredicate(name="lunch")
    matcher = SelectorMatcher()
    assert matcher.matches(sel, _event("lunch")) is True
    assert matcher.matches(sel, _task("lunch")) is True


def test_resolve_context_episode_returns_label_only() -> None:
    matcher = SelectorMatcher()
    resolved = matcher.resolve(_context("anxious"))
    assert resolved.label == "anxious"
    assert resolved.intensity is None
    assert resolved.domain is None
    assert resolved.met is None
