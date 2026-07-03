"""Greedy augmenter parses ObservationConfig but ignores it (one INFO log per run)."""

from __future__ import annotations

import datetime
import logging

import pytest

from src.scripts.persona.config.schema import DailyWindow
from src.scripts.scenarios.augmentation.greedy import GreedyAugmenter
from src.scripts.scenarios.config.schema import AugmentationConfig, ObservationConfig
from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace
from src.scripts.scenarios.domain.task import RecommendedTask


def _calendar() -> CalendarTrace:
    return CalendarTrace(
        person_id="p1",
        events=[
            CalendarEvent(
                label="lunch",
                start_minutes=720,
                end_minutes=765,
                date=datetime.date(2026, 5, 4),
            ),
        ],
    )


def _tasks() -> list[RecommendedTask]:
    return [
        RecommendedTask(label="walk", duration_min=10, duration_max=20, intensity=2),
    ]


def _config_with_observation(obs: ObservationConfig) -> AugmentationConfig:
    return AugmentationConfig(method="greedy", observation=obs)


def test_greedy_with_empty_observation_does_not_log() -> None:
    augmenter = GreedyAugmenter(daily_window=DailyWindow())
    config = AugmentationConfig(method="greedy")
    logger = logging.getLogger("src.scripts.scenarios.augmentation.greedy")
    seen: list[str] = []

    class _Hook(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            seen.append(record.getMessage())

    hook = _Hook(level=logging.INFO)
    logger.addHandler(hook)
    try:
        augmenter.augment(_calendar(), _tasks(), config)
    finally:
        logger.removeHandler(hook)
    assert not any("observation block ignored" in m for m in seen)


def test_greedy_with_populated_observation_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    augmenter = GreedyAugmenter(daily_window=DailyWindow())
    config = _config_with_observation(
        ObservationConfig(
            contexts=["mood_emotion"],
            host_flags=["is_concurrent"],
        )
    )
    with caplog.at_level(
        logging.INFO, logger="src.scripts.scenarios.augmentation.greedy"
    ):
        augmenter.augment(_calendar(), _tasks(), config)
    messages = [r.getMessage() for r in caplog.records]
    matches = [m for m in messages if "observation block ignored" in m]
    assert len(matches) == 1
    assert "mood_emotion" in matches[0]
    assert "is_concurrent" in matches[0]


def test_greedy_output_byte_identical_with_or_without_observation_block() -> None:
    """Observation must not change Greedy's placements (it's a deterministic FCFS)."""
    augmenter = GreedyAugmenter(daily_window=DailyWindow(), seed=42)
    config_a = AugmentationConfig(method="greedy")
    config_b = _config_with_observation(
        ObservationConfig(
            contexts=["mood_emotion"],
            host_flags=["is_concurrent"],
            task_flags=["duration_min"],
        )
    )
    sol_a = augmenter.augment(_calendar(), _tasks(), config_a)
    sol_b = GreedyAugmenter(daily_window=DailyWindow(), seed=42).augment(
        _calendar(), _tasks(), config_b
    )
    starts_a = [
        (s.task.label, s.date.isoformat(), s.start_minutes, s.end_minutes)
        for s in sol_a.scheduled
    ]
    starts_b = [
        (s.task.label, s.date.isoformat(), s.start_minutes, s.end_minutes)
        for s in sol_b.scheduled
    ]
    assert starts_a == starts_b
