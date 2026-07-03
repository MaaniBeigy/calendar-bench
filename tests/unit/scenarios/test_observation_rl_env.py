"""SchedulingEnv extends the observation vector when observation.contexts is set."""

from __future__ import annotations

import datetime

import numpy as np

from src.scripts.scenarios.augmentation.rl.env import (
    _N_BUCKETS,
    _N_DAYS,
    _N_TASK_FEATURES,
    SchedulingEnv,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, ObservationConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)


def _trace(*events, contexts=()):
    return CalendarTrace(person_id="p1", events=list(events), contexts=list(contexts))


def _config():
    return AugmentationConfig()


def test_obs_dim_unchanged_when_observation_contexts_empty() -> None:
    tasks = [make_task("yoga", duration_min=30, duration_max=60)]
    env = SchedulingEnv(
        calendar=_trace(
            make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)
        ),
        tasks=tasks,
        config=_config(),
        observation=ObservationConfig(),
    )
    expected = _N_DAYS * _N_BUCKETS + 1 * _N_TASK_FEATURES + 1
    assert env.observation_space.shape == (expected,)


def test_obs_dim_grows_by_seven_per_enabled_category() -> None:
    tasks = [make_task("yoga", duration_min=30, duration_max=60)]
    env = SchedulingEnv(
        calendar=_trace(
            make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)
        ),
        tasks=tasks,
        config=_config(),
        observation=ObservationConfig(contexts=["mood_emotion", "energy_state"]),
    )
    base = _N_DAYS * _N_BUCKETS + 1 * _N_TASK_FEATURES + 1
    assert env.observation_space.shape == (base + _N_DAYS * 2,)


def test_context_features_zero_when_no_episodes_match_visible_categories() -> None:
    tasks = [make_task("yoga", duration_min=30, duration_max=60)]
    env = SchedulingEnv(
        calendar=_trace(
            make_event("sleep", start_minutes=0, end_minutes=60, date=DATE),
            contexts=[
                ContextEpisode(
                    name="energetic",
                    category="energy_state",
                    date=DATE,
                    start_minutes=480,
                    end_minutes=540,
                )
            ],
        ),
        tasks=tasks,
        config=_config(),
        observation=ObservationConfig(contexts=["mood_emotion"]),
    )
    obs, _ = env.reset()
    tail = obs[-(_N_DAYS * 1) :]
    assert float(tail.sum()) == 0.0


def test_context_features_one_when_episode_present_in_enabled_category() -> None:
    tasks = [make_task("yoga", duration_min=30, duration_max=60)]
    env = SchedulingEnv(
        calendar=_trace(
            make_event("sleep", start_minutes=0, end_minutes=60, date=DATE),
            contexts=[
                ContextEpisode(
                    name="energetic",
                    category="energy_state",
                    date=DATE,
                    start_minutes=480,
                    end_minutes=540,
                )
            ],
        ),
        tasks=tasks,
        config=_config(),
        observation=ObservationConfig(contexts=["energy_state"]),
    )
    obs, _ = env.reset()
    # First day, first (and only) category slot should be 1.0; remaining days 0.
    tail = obs[-(_N_DAYS * 1) :]
    assert float(tail[0]) == 1.0
    assert float(tail[1:].sum()) == 0.0
