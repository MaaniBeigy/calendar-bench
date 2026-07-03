"""Shared fixtures for scenarios-pipeline unit tests."""

from __future__ import annotations

import datetime

import pytest

from src.scripts.scenarios.config import schema as _schema
from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask

# The 15 Context category slugs declared in the Neo4j Context ontology.
# Pre-populating the cache here lets unit tests load `ObservationConfig`
# without opening a Neo4j connection from every test process.
_CONTEXT_CATEGORY_SLUGS: frozenset[str] = frozenset(
    {
        "mood_emotion",
        "energy_state",
        "physiological",
        "stress",
        "location",
        "social_context",
        "weather_environment",
        "behaviour_state",
        "capability_opportunity",
        "goal_intention",
        "trait_state",
        "ttm_process",
        "bct",
        "sdt_regulation",
        "delivery_style",
    }
)
_schema._load_context_categories._cache = _CONTEXT_CATEGORY_SLUGS  # type: ignore[attr-defined]

DATE = datetime.date(2026, 5, 4)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_task(
    label: str = "running",
    *,
    duration_min: int = 30,
    duration_max: int = 60,
    intensity: int = 3,
    is_dividable: bool = False,
    is_concurrent: bool = False,
    ontology_uri: str | None = None,
    display_name: str = "",
    description: str = "",
    **_legacy: object,
) -> RecommendedTask:
    """Build a RecommendedTask with sensible defaults; legacy epoch kwargs are dropped."""
    return RecommendedTask(
        label=label,
        duration_min=duration_min,
        duration_max=duration_max,
        intensity=intensity,
        is_dividable=is_dividable,
        is_concurrent=is_concurrent,
        ontology_uri=ontology_uri,
        display_name=display_name,
        description=description,
    )


def make_scheduled(
    task: RecommendedTask | None = None,
    *,
    start_minutes: int = 480,
    end_minutes: int = 540,
    is_standalone: bool = True,
    concurrent_with: str | None = None,
    date: datetime.date = DATE,
) -> ScheduledTask:
    """Build a ScheduledTask with sensible defaults for tests."""
    return ScheduledTask(
        task=task or make_task(),
        start_minutes=start_minutes,
        end_minutes=end_minutes,
        is_standalone=is_standalone,
        concurrent_with=concurrent_with,
        date=date,
    )


def make_event(
    label: str = "lunch",
    *,
    start_minutes: int = 720,
    end_minutes: int = 780,
    date: datetime.date = DATE,
    is_concurrent: bool = False,
    is_dividable: bool = False,
    concurrent_with: list[str] | None = None,
    intensity: int = 1,
    display_label: str | None = None,
) -> CalendarEvent:
    """Build a CalendarEvent with sensible defaults for tests."""
    return CalendarEvent(
        label=label,
        start_minutes=start_minutes,
        end_minutes=end_minutes,
        date=date,
        is_concurrent=is_concurrent,
        is_dividable=is_dividable,
        concurrent_with=concurrent_with or [],
        intensity=intensity,
        display_label=display_label,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_task() -> RecommendedTask:
    return make_task()


@pytest.fixture
def sample_event() -> CalendarEvent:
    return make_event()


@pytest.fixture
def empty_trace() -> CalendarTrace:
    return CalendarTrace(person_id="p001")


@pytest.fixture
def empty_augmented() -> AugmentedCalendar:
    return AugmentedCalendar(person_id="p001")
