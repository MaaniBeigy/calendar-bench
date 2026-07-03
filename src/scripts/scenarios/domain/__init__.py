"""Domain types for the scenarios pipeline."""

from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask

__all__ = [
    "AugmentedCalendar",
    "CalendarEvent",
    "CalendarTrace",
    "RecommendedTask",
    "ScheduledTask",
    "SchedulingSolution",
]
