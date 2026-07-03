"""Tests for the cross-week epsilon curriculum and the auto-pick task-skip guard."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from src.scripts.scenarios.augmentation.rl.env import (
    AutoTaskActionWrapper,
    SchedulingEnv,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, RLConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)


# ---------------------------------------------------------------------------
# Auto-pick failure mode (max_task_attempts)
# ---------------------------------------------------------------------------


def _unplaceable_env(max_task_attempts: int) -> SchedulingEnv:
    """Build an env whose single task can never be placed (extends past midnight)."""
    # duration_min larger than a day so every (day, bucket) placement fails
    # the `end_min > 1440` guard; the only outcome is repeated invalid steps.
    task = make_task("impossible", duration_min=1500, duration_max=1500)
    env = SchedulingEnv(
        calendar=CalendarTrace(person_id="p1", events=[]),
        tasks=[task],
        config=AugmentationConfig(method="rl"),
        max_steps=100,
        max_task_attempts=max_task_attempts,
    )
    env.reset(options={"week_index": 0})
    return env


def _step(env, task=0, day=0, bucket=0, merge=0):
    return env.step(np.array([task, day, bucket, merge], dtype=np.int64))


def test_max_task_attempts_zero_keeps_legacy_spin():
    """With the guard off, an unplaceable task never terminates early."""
    env = _unplaceable_env(max_task_attempts=0)
    terminated = False
    for _ in range(5):
        _obs, _r, terminated, _trunc, _info = _step(env)
        assert not terminated
    assert env._skipped_task_indices == set()


def test_max_task_attempts_gives_up_after_cap():
    """After `max_task_attempts` failures the task is given up and the episode ends."""
    env = _unplaceable_env(max_task_attempts=3)
    terminated = False
    for i in range(3):
        _obs, _r, terminated, _trunc, info = _step(env)
        if i < 2:
            assert not terminated
    # third failed attempt hits the cap, so the task is skipped and all resolve.
    assert 0 in env._skipped_task_indices
    assert terminated is True


def test_skipped_task_not_in_scheduled_solution():
    """A given-up task stays unscheduled in the produced solution."""
    env = _unplaceable_env(max_task_attempts=2)
    for _ in range(2):
        _step(env)
    sol = env.get_solution()
    assert sol.scheduled == []
    assert len(sol.unscheduled) == 1


def test_valid_placement_does_not_count_as_attempt():
    """A successful placement never increments the failure counter."""
    task = make_task("walk", duration_min=30, duration_max=30)
    # A morning event anchors the horizon date; the task places at 06:00,
    # clear of the 12:00 event.
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    env = SchedulingEnv(
        calendar=CalendarTrace(person_id="p1", events=[ev]),
        tasks=[task],
        config=AugmentationConfig(method="rl"),
        max_steps=50,
        max_task_attempts=2,
    )
    env.reset(options={"week_index": 0})
    # day 0 is DATE (it carries the event); bucket 24 = 06:00.
    day_idx = env._week_dates.index(DATE)
    _obs, _r, terminated, _trunc, _info = _step(env, day=day_idx, bucket=24)
    assert terminated is True  # placed, so resolved
    assert env._task_attempts.get(0, 0) == 0
    assert env._skipped_task_indices == set()


def test_auto_pick_wrapper_skips_given_up_task():
    """`AutoTaskActionWrapper` advances past a given-up task to the next one."""
    t0 = make_task("impossible", duration_min=1500, duration_max=1500)
    t1 = make_task("walk", duration_min=30, duration_max=30)
    base = SchedulingEnv(
        calendar=CalendarTrace(person_id="p1", events=[]),
        tasks=[t0, t1],
        config=AugmentationConfig(method="rl"),
        max_steps=50,
        max_task_attempts=1,
    )
    wrapper = AutoTaskActionWrapper(base)
    base.reset(options={"week_index": 0})
    # First 3D action targets task 0 (the only unresolved task), fails, and
    # task 0 is given up after one attempt.
    full0 = wrapper.action(np.array([0, 0, 0], dtype=np.int64))
    assert int(full0[0]) == 0
    base.step(full0)
    assert 0 in base._skipped_task_indices
    # Next prepended task_idx must skip the given-up task 0 and pick task 1.
    full1 = wrapper.action(np.array([0, 36, 0], dtype=np.int64))
    assert int(full1[0]) == 1


# ---------------------------------------------------------------------------
# Cross-week epsilon curriculum wiring
# ---------------------------------------------------------------------------


def test_cross_week_epsilon_default_off():
    """The opt-in epsilon curriculum defaults off so proven configs are unchanged."""
    assert RLConfig().cross_week_epsilon is False
    assert RLConfig().max_task_attempts == 0


def test_cross_week_epsilon_pins_per_week_rate(monkeypatch):
    """When enabled, the augmenter pins the agent epsilon once per week."""
    from src.scripts.scenarios.augmentation.rl import augmenter as aug_mod
    from src.scripts.scenarios.augmentation.rl.augmenter import RLAugmenter

    pinned: list[float] = []

    class _SpyAgent:
        model = None

        def __init__(self, *a, **k):
            pass

        def act(self, obs, *, deterministic=False):
            return 0

        def remember(self, *a, **k):
            pass

        def learn(self, *a, **k):
            pass

        def save(self, *a, **k):
            pass

        def load(self, *a, **k):
            pass

        def set_epsilon(self, value):
            pinned.append(float(value))

    monkeypatch.setattr(aug_mod, "RLAgent", _SpyAgent)

    aug = RLAugmenter(seed=0)
    cfg = AugmentationConfig(
        method="rl",
        repeat_per_week=True,
        rl=RLConfig(
            policy="random",
            train_steps_per_week=1,
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_weeks=4,
            cross_week_epsilon=True,
        ),
    )
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    aug.augment(trace, [task], cfg, horizon=(DATE, 21))  # 3 weeks
    # One pin per week on the linear 1.0 to 0.0 over 4 weeks curve.
    assert pinned == pytest.approx([1.0, 0.75, 0.5])


def test_cross_week_epsilon_off_does_not_pin(monkeypatch):
    """With the flag off, the augmenter never calls set_epsilon."""
    from src.scripts.scenarios.augmentation.rl import augmenter as aug_mod
    from src.scripts.scenarios.augmentation.rl.augmenter import RLAugmenter

    calls: list[float] = []

    class _SpyAgent:
        model = None

        def __init__(self, *a, **k):
            pass

        def act(self, obs, *, deterministic=False):
            return 0

        def remember(self, *a, **k):
            pass

        def learn(self, *a, **k):
            pass

        def save(self, *a, **k):
            pass

        def load(self, *a, **k):
            pass

        def set_epsilon(self, value):
            calls.append(float(value))

    monkeypatch.setattr(aug_mod, "RLAgent", _SpyAgent)

    aug = RLAugmenter(seed=0)
    cfg = AugmentationConfig(
        method="rl",
        repeat_per_week=True,
        rl=RLConfig(policy="random", train_steps_per_week=1),
    )
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    aug.augment(trace, [task], cfg, horizon=(DATE, 14))
    assert calls == []
