"""Default values for persona-pipeline configuration."""

from __future__ import annotations

from typing import Final

DEFAULT_SEED: Final[int] = 20260503

# Standard time windows in minutes since midnight.
DEFAULT_TIME_WINDOWS: Final[dict[str, list[int]]] = {
    "early_morning": [0, 400],
    "morning": [400, 600],
    "afternoon": [600, 960],
    "evening": [960, 1260],
    "night": [1260, 1440],
}

DEFAULT_STEP_MINUTES: Final[int] = 10
DEFAULT_MAX_ATTEMPTS: Final[int] = 25

DEFAULT_JITTER_TIME_MINUTES: Final[int] = 15
DEFAULT_JITTER_DURATION_MINUTES: Final[int] = 10

WEEKDAY_ORDER: Final[tuple[str, ...]] = (
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
)
WEEKDAY_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(WEEKDAY_ORDER)}
