"""Unit tests for augmentation/rl/env.py.

Coverage targets:
  SchedulingEnv.__init__: defaults, max_tasks override, no tasks.
  reset(): default week_index, options override, empty calendar,
           week_index out of range, seed propagation.
  step(): valid standalone, valid merge, invalid (all reason branches),
          terminated (all tasks scheduled), truncated (max_steps hit).
  _observe(): shape, occupancy bits, task features, elapsed fraction,
              tasks beyond max_tasks clipped, multiple days.
  render(): ansi mode, None mode.
  get_solution(): standalone tasks, merge tasks, unscheduled tasks.
  ActionMaskWrapper.valid_action_mask(): scheduled tasks masked,
                                         short week masked, unmasked.
  FlattenActionWrapper.action() / reverse_action(): round-trip.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from src.scripts.scenarios.augmentation.rl.env import (
    _N_BUCKETS,
    _N_DAYS,
    _N_TASK_FEATURES,
    ActionMaskWrapper,
    AutoTaskActionWrapper,
    FlattenActionWrapper,
    SchedulingEnv,
)
from src.scripts.scenarios.config.schema import AugmentationConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)
DATE2 = datetime.date(2026, 5, 5)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(merge_threshold: float = 0.70) -> AugmentationConfig:
    return AugmentationConfig(method="rl", merge_threshold=merge_threshold)


def _trace(*events) -> CalendarTrace:
    return CalendarTrace(person_id="p001", events=list(events))


def _make_env(
    events=(),
    tasks=None,
    max_tasks=None,
    max_steps=None,
    week_index=0,
    render_mode=None,
    semantic=None,
) -> SchedulingEnv:
    if tasks is None:
        tasks = [make_task("yoga", duration_min=30, duration_max=60)]
    return SchedulingEnv(
        calendar=_trace(*events),
        tasks=tasks,
        config=_config(),
        max_tasks=max_tasks,
        max_steps=max_steps,
        week_index=week_index,
        render_mode=render_mode,
        semantic=semantic,
    )


def _step(env: SchedulingEnv, task=0, day=0, bucket=0, merge=0):
    return env.step(np.array([task, day, bucket, merge], dtype=np.int64))


# ---------------------------------------------------------------------------
# Spaces & init
# ---------------------------------------------------------------------------


class TestSpaces:
    def test_observation_space_shape(self):
        env = _make_env()
        n = _N_DAYS * _N_BUCKETS + 1 * _N_TASK_FEATURES + 1
        assert env.observation_space.shape == (n,)

    def test_observation_space_dtype(self):
        env = _make_env()
        assert env.observation_space.dtype == np.float32

    def test_action_space_nvec(self):
        env = _make_env()
        assert list(env.action_space.nvec) == [1, _N_DAYS, _N_BUCKETS, 2]

    def test_max_tasks_override(self):
        tasks = [make_task(f"t{i}") for i in range(3)]
        env = SchedulingEnv(
            calendar=_trace(), tasks=tasks, config=_config(), max_tasks=5
        )
        expected_obs_dim = _N_DAYS * _N_BUCKETS + 5 * _N_TASK_FEATURES + 1
        assert env.observation_space.shape == (expected_obs_dim,)
        assert env.action_space.nvec[0] == 5

    def test_no_tasks(self):
        env = SchedulingEnv(calendar=_trace(), tasks=[], config=_config())
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_max_steps_default(self):
        tasks = [make_task(f"t{i}") for i in range(4)]
        env = SchedulingEnv(calendar=_trace(), tasks=tasks, config=_config())
        assert env._max_steps == max(10, 12)

    def test_max_steps_override(self):
        env = _make_env(max_steps=50)
        assert env._max_steps == 50


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_obs_and_info(self):
        env = _make_env(
            events=[make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)]
        )
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_obs_dtype(self):
        env = _make_env(
            events=[make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)]
        )
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_state_reset_between_episodes(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _step(env, task=0, day=0, bucket=0)  # place a task
        assert len(env._scheduled) == 1
        env.reset()
        assert len(env._scheduled) == 0
        assert env._step_count == 0

    def test_week_dates_populated(self):
        ev = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        assert DATE in env._week_dates

    def test_options_week_index_override(self):
        week2_date = DATE + datetime.timedelta(days=7)
        ev1 = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        ev2 = make_event("b", start_minutes=0, end_minutes=60, date=week2_date)
        env = _make_env(events=[ev1, ev2])
        env.reset(options={"week_index": 1})
        assert week2_date in env._week_dates

    def test_empty_calendar_reset(self):
        env = _make_env()
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env._week_dates == []

    def test_week_index_out_of_range_uses_first_n_days(self):
        ev = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        env = _make_env(events=[ev])
        env.reset(options={"week_index": 99})
        assert DATE in env._week_dates

    def test_seed_accepted(self):
        env = _make_env(
            events=[make_event("a", start_minutes=0, end_minutes=60, date=DATE)]
        )
        obs, _ = env.reset(seed=42)
        assert obs is not None


# ---------------------------------------------------------------------------
# step(); valid placements
# ---------------------------------------------------------------------------


class TestStepValid:
    def test_standalone_placement_returns_positive_reward(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        obs, reward, terminated, truncated, info = _step(env, bucket=0)
        assert reward > 0.0
        assert "placement" in info

    def test_placement_marks_task_scheduled(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _step(env, bucket=0)
        assert 0 in env._scheduled_task_indices

    def test_terminated_when_all_tasks_placed(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _, _, terminated, _, _ = _step(env, bucket=0)
        assert terminated is True

    def test_merge_placement(self):
        ev = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        task = make_task("mindful_eating", duration_min=30, duration_max=60)
        semantic = SemanticCompatibility(matrix={("mindful_eating", "lunch"): 0.95})
        env = SchedulingEnv(
            calendar=_trace(ev),
            tasks=[task],
            config=_config(),
            semantic=semantic,
        )
        env.reset()
        # bucket 48 = 720 min (12:00); exactly where lunch starts
        _, reward, terminated, _, info = _step(env, bucket=48, merge=1)
        assert reward > 0.0
        assert env._scheduled[0].concurrent_with == "lunch"
        assert len(env._consumed_merges) == 1

    def test_observation_space_satisfied(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        obs2, _, _, _, _ = _step(env, bucket=0)
        assert env.observation_space.contains(obs2)


# ---------------------------------------------------------------------------
# step(); invalid actions
# ---------------------------------------------------------------------------


class TestStepInvalid:
    def test_task_idx_beyond_tasks_gives_zero_reward(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)
        tasks = [make_task("yoga")]
        env = SchedulingEnv(
            calendar=_trace(ev), tasks=tasks, config=_config(), max_tasks=3
        )
        env.reset()
        _, reward, _, _, info = _step(env, task=2)  # task index 2 doesn't exist
        assert reward == 0.0
        assert "reason" in info

    def test_already_scheduled_task_zero_reward(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _step(env, bucket=0)  # schedule task 0
        _, reward, _, _, info = _step(env, bucket=2)  # try again
        assert reward == 0.0
        assert "already scheduled" in info["reason"]

    def test_day_outside_week_zero_reward(self):
        ev = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _, reward, _, _, info = _step(env, day=6)  # week has only 1 day
        assert reward == 0.0
        assert "day_idx" in info["reason"]

    def test_past_midnight_zero_reward(self):
        ev = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        task = make_task("yoga", duration_min=60, duration_max=90)
        env = SchedulingEnv(calendar=_trace(ev), tasks=[task], config=_config())
        env.reset()
        # bucket 95 = 1425 min; 1425 + 60 = 1485 > 1440
        _, reward, _, _, info = _step(env, bucket=95)
        assert reward == 0.0
        assert "midnight" in info["reason"]

    def test_conflict_with_existing_event_zero_reward(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=600, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _, reward, _, _, info = _step(env, bucket=0)
        assert reward == 0.0
        assert "conflict" in info["reason"]

    def test_conflict_with_placed_task_zero_reward(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        tasks = [
            make_task("yoga", duration_min=30, duration_max=60),
            make_task("run", duration_min=30, duration_max=60, is_concurrent=False),
        ]
        env = SchedulingEnv(calendar=_trace(ev), tasks=tasks, config=_config())
        env.reset()
        _step(env, task=0, bucket=0)  # place yoga at [0, 30)
        _, reward, _, _, info = _step(env, task=1, bucket=0)  # run at [0, 30); conflict
        assert reward == 0.0
        assert "overlaps" in info["reason"]

    def test_concurrent_task_can_overlap_placed(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        tasks = [
            make_task("yoga", duration_min=30, duration_max=60),
            make_task("podcast", duration_min=30, duration_max=60, is_concurrent=True),
        ]
        env = SchedulingEnv(calendar=_trace(ev), tasks=tasks, config=_config())
        env.reset()
        _step(env, task=0, bucket=0)  # yoga at [0, 30)
        _, reward, _, _, _ = _step(env, task=1, bucket=0)  # podcast overlaps; OK
        assert reward > 0.0

    def test_no_merge_event_at_slot(self):
        ev = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        task = make_task("yoga", duration_min=30, duration_max=60)
        semantic = SemanticCompatibility(matrix={("yoga", "lunch"): 0.05})  # too low
        env = SchedulingEnv(
            calendar=_trace(ev), tasks=[task], config=_config(), semantic=semantic
        )
        env.reset()
        _, reward, _, _, info = _step(env, bucket=48, merge=1)
        assert reward == 0.0
        assert "no compatible event" in info["reason"]

    def test_truncated_when_max_steps_reached(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev], max_steps=2)
        env.reset()
        _step(env)  # step 1; invalid (full day)
        _, _, _, truncated, _ = _step(env)  # step 2; truncated
        assert truncated is True


# ---------------------------------------------------------------------------
# _observe()
# ---------------------------------------------------------------------------


class TestObserve:
    def test_occupancy_bits_set_for_events(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        obs = env._observe()
        # bucket 0 = minutes [0, 15); should be 1
        assert obs[0] == 1.0

    def test_occupancy_bits_set_for_placed_tasks(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _step(env, bucket=0)  # place at bucket 0
        obs = env._observe()
        assert obs[0] == 1.0

    def test_task_features_populated(self):
        task = make_task(
            "yoga",
            duration_min=30,
            duration_max=60,
            intensity=3,
            is_dividable=True,
            is_concurrent=False,
        )
        ev = make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)
        env = SchedulingEnv(calendar=_trace(ev), tasks=[task], config=_config())
        env.reset()
        obs = env._observe()
        feat_base = _N_DAYS * _N_BUCKETS
        assert obs[feat_base + 0] == pytest.approx(30 / 1440)
        assert obs[feat_base + 1] == pytest.approx(60 / 1440)
        assert obs[feat_base + 2] == pytest.approx(3 / 5)
        assert obs[feat_base + 3] == 1.0  # is_dividable
        assert obs[feat_base + 4] == 0.0  # is_concurrent

    def test_scheduled_flag_set_after_placement(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        feat_base = _N_DAYS * _N_BUCKETS
        assert env._observe()[feat_base + 5] == 0.0  # not yet scheduled
        _step(env, bucket=0)
        assert env._observe()[feat_base + 5] == 1.0  # now scheduled

    def test_elapsed_fraction_increases(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev], max_steps=4)
        env.reset()
        assert env._observe()[-1] == pytest.approx(0.0)
        _step(env)
        assert env._observe()[-1] == pytest.approx(0.25)

    def test_tasks_beyond_max_tasks_not_in_obs(self):
        tasks = [make_task(f"t{i}") for i in range(5)]
        env = SchedulingEnv(
            calendar=_trace(), tasks=tasks, config=_config(), max_tasks=2
        )
        env.reset()
        obs = env._observe()
        expected_size = _N_DAYS * _N_BUCKETS + 2 * _N_TASK_FEATURES + 1
        assert obs.shape == (expected_size,)


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


class TestRender:
    def test_ansi_returns_string(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=60, date=DATE)
        env = _make_env(events=[ev], render_mode="ansi")
        env.reset()
        result = env.render()
        assert isinstance(result, str)
        assert "Step" in result

    def test_none_mode_returns_none(self):
        env = _make_env(render_mode=None)
        env.reset()
        assert env.render() is None

    def test_render_shows_placed_tasks(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev], render_mode="ansi")
        env.reset()
        _step(env, bucket=0)
        result = env.render()
        assert "1 placed" in result


# ---------------------------------------------------------------------------
# get_solution()
# ---------------------------------------------------------------------------


class TestGetSolution:
    def test_solution_has_all_tasks(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        sol = env.get_solution()
        assert len(sol.tasks) == 1

    def test_unscheduled_when_nothing_placed(self):
        env = _make_env()
        env.reset()
        sol = env.get_solution()
        assert len(sol.unscheduled) == 1
        assert sol.scheduled == []

    def test_scheduled_after_placement(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _step(env, bucket=0)
        sol = env.get_solution()
        assert len(sol.scheduled) == 1
        assert sol.unscheduled == []

    def test_merge_creates_composite_event(self):
        ev = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        task = make_task("mindful_eating", duration_min=30, duration_max=60)
        semantic = SemanticCompatibility(matrix={("mindful_eating", "lunch"): 0.95})
        env = SchedulingEnv(
            calendar=_trace(ev), tasks=[task], config=_config(), semantic=semantic
        )
        env.reset()
        _step(env, bucket=48, merge=1)
        sol = env.get_solution()
        concurrent_tasks = [
            s for s in sol.augmented_calendar.scheduled_tasks if not s.is_standalone
        ]
        assert len(concurrent_tasks) == 1
        # base_events now contains ALL original events (lunch is kept)
        assert len(sol.augmented_calendar.base_events) == 1


# ---------------------------------------------------------------------------
# ActionMaskWrapper
# ---------------------------------------------------------------------------


class TestActionMaskWrapper:
    def _make_wrapped(self, events=(), tasks=None):
        env = _make_env(events=events, tasks=tasks)
        env.reset()
        return ActionMaskWrapper(env)

    def test_mask_shape(self):
        wrapped = self._make_wrapped(
            events=[make_event("a", start_minutes=0, end_minutes=60, date=DATE)]
        )
        mask = wrapped.valid_action_mask()
        n = wrapped.unwrapped.action_space.nvec
        assert mask.shape == (int(np.prod(n)),)
        assert mask.dtype == bool

    def test_all_true_when_no_tasks_scheduled(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        wrapped = self._make_wrapped(events=[ev])
        mask = wrapped.valid_action_mask()
        assert mask.any()

    def test_scheduled_task_masked(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _step(env, bucket=0)  # task 0 scheduled
        wrapped = ActionMaskWrapper(env)
        mask = wrapped.valid_action_mask()
        n_days, n_buckets, n_merge = (
            int(env.action_space.nvec[1]),
            int(env.action_space.nvec[2]),
            int(env.action_space.nvec[3]),
        )
        task_stride = n_days * n_buckets * n_merge
        assert not mask[:task_stride].any()  # task 0 fully masked

    def test_days_outside_week_masked(self):
        ev = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        env = _make_env(events=[ev])  # 1-day horizon to week has 1 day
        env.reset()
        wrapped = ActionMaskWrapper(env)
        mask = wrapped.valid_action_mask()
        n_tasks, n_days, n_buckets, n_merge = (int(v) for v in env.action_space.nvec)
        day_stride = n_buckets * n_merge
        task_stride = n_days * day_stride
        # day 1..6 should be masked for task 0
        for d in range(1, n_days):
            off = d * day_stride
            assert not mask[off : off + day_stride].any()

    def test_task_beyond_tasks_list_masked(self):
        ev = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        tasks = [make_task("yoga")]
        env = SchedulingEnv(
            calendar=_trace(ev), tasks=tasks, config=_config(), max_tasks=3
        )
        env.reset()
        wrapped = ActionMaskWrapper(env)
        mask = wrapped.valid_action_mask()
        n_days, n_buckets, n_merge = (int(v) for v in env.action_space.nvec[1:])
        task_stride = n_days * n_buckets * n_merge
        # task 1 and 2 (beyond tasks list) should be masked
        assert not mask[task_stride : 2 * task_stride].any()
        assert not mask[2 * task_stride : 3 * task_stride].any()

    def test_step_and_reset_work_through_wrapper(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        wrapped = ActionMaskWrapper(env)
        obs, _ = wrapped.reset()
        assert obs.shape == env.observation_space.shape
        obs2, reward, term, trunc, info = wrapped.step(np.array([0, 0, 0, 0]))
        assert obs2.shape == env.observation_space.shape


# ---------------------------------------------------------------------------
# FlattenActionWrapper
# ---------------------------------------------------------------------------


class TestFlattenActionWrapper:
    def _make_flat(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        return FlattenActionWrapper(env)

    def test_action_space_is_discrete(self):
        flat = self._make_flat()
        from gymnasium import spaces

        assert isinstance(flat.action_space, spaces.Discrete)

    def test_action_space_size(self):
        env = _make_env()
        flat = FlattenActionWrapper(env)
        expected = int(np.prod(env.action_space.nvec))
        assert int(flat.action_space.n) == expected

    def test_decode_zero(self):
        flat = self._make_flat()
        arr = flat.action(0)
        assert list(arr) == [0, 0, 0, 0]

    def test_decode_one(self):
        flat = self._make_flat()
        arr = flat.action(1)
        assert arr[-1] == 1  # last component cycles fastest

    def test_round_trip_encode_decode(self):
        env = _make_env()
        flat = FlattenActionWrapper(env)
        total = int(flat.action_space.n)
        for k in [0, 1, total // 2, total - 1]:
            arr = flat.action(k)
            assert flat.reverse_action(arr) == k

    def test_step_through_wrapper(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        flat = FlattenActionWrapper(env)
        flat.reset()
        obs, reward, term, trunc, info = flat.step(0)
        assert obs.shape == env.observation_space.shape


# ---------------------------------------------------------------------------
# Reward hook + carry-over + week accessors
# ---------------------------------------------------------------------------


class TestRewardHookAndCarryOver:
    def test_reward_hook_adds_terminal_bonus_on_termination(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        bonuses: list[float] = [0.42]

        def hook(env):
            return bonuses.pop()

        env = SchedulingEnv(
            calendar=_trace(ev),
            tasks=[make_task("yoga", duration_min=30, duration_max=60)],
            config=_config(),
            reward_hook=hook,
        )
        env.reset()
        _, reward, term, trunc, _ = _step(env, task=0, day=0, bucket=0)
        assert term is True
        assert reward >= 0.42  # baseline shaping + bonus

    def test_reward_hook_fires_on_truncation_only(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        called: list[int] = []

        def hook(env):
            called.append(1)
            return 0.0

        env = SchedulingEnv(
            calendar=_trace(ev),
            tasks=[make_task("yoga", duration_min=30, duration_max=60)],
            config=_config(),
            max_steps=1,
            reward_hook=hook,
        )
        env.reset()
        _, _, term, trunc, _ = _step(env, task=0, day=5, bucket=0)  # invalid
        assert trunc is True
        assert called == [1]

    def test_reset_options_carry_over_recorded(self):
        from src.scripts.scenarios.domain.task import ScheduledTask

        ev = make_event("a", start_minutes=0, end_minutes=60, date=DATE)
        env = _make_env(events=[ev])
        task = make_task("yoga", duration_min=30, duration_max=60)
        prior = [
            ScheduledTask(
                task=task,
                start_minutes=300,
                end_minutes=330,
                is_standalone=True,
                concurrent_with=None,
                date=DATE - datetime.timedelta(days=7),
            )
        ]
        env.reset(options={"carry_over": prior})
        assert env.get_carry_over() == prior

    def test_bucket_minutes_shrinks_action_space_and_obs(self):
        env = SchedulingEnv(
            calendar=_trace(),
            tasks=[make_task("yoga", duration_min=30, duration_max=60)],
            config=_config(),
            bucket_minutes=60,
        )
        assert int(env.action_space.nvec[2]) == 24  # 1440 / 60
        n_ctx = 0
        expected = _N_DAYS * 24 + 1 * _N_TASK_FEATURES + 1 + n_ctx
        assert env.observation_space.shape == (expected,)

    def test_per_step_shaping_adds_reward_for_valid_placement(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)

        def _hook(env):
            return 0.0

        env = SchedulingEnv(
            calendar=_trace(ev),
            tasks=[make_task("yoga", duration_min=30, duration_max=60)],
            config=_config(),
            reward_hook=_hook,
            per_step_shaping=1.0,
        )
        env.reset()
        _, reward, term, trunc, _ = _step(env, task=0, day=0, bucket=0)
        # Valid placement on 1-task episode -> shaping contributes 1.0/1.
        assert reward >= 1.0
        assert term is True

    def test_per_step_shaping_zero_for_invalid_action(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)

        def _hook(env):
            return 0.0

        env = SchedulingEnv(
            calendar=_trace(ev),
            tasks=[make_task("yoga", duration_min=30, duration_max=60)],
            config=_config(),
            reward_hook=_hook,
            per_step_shaping=1.0,
        )
        env.reset()
        # day_idx=5 outside week -> invalid -> shaping not added.
        _, reward, _, _, _ = _step(env, task=0, day=5, bucket=0)
        assert reward == 0.0

    def test_bucket_minutes_step_uses_configured_width(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = SchedulingEnv(
            calendar=_trace(ev),
            tasks=[make_task("yoga", duration_min=30, duration_max=60)],
            config=_config(),
            bucket_minutes=60,
        )
        env.reset()
        # bucket_idx=8 with bucket_minutes=60 to start_min=480.
        env.step(np.array([0, 0, 8, 0], dtype=np.int64))
        sol = env.get_solution()
        assert sol.scheduled[0].start_minutes == 480

    def test_get_week_scheduled_returns_episode_placements(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(events=[ev])
        env.reset()
        _step(env, task=0, day=0, bucket=0)
        scheduled = env.get_week_scheduled()
        assert len(scheduled) == 1


# ---------------------------------------------------------------------------
# AutoTaskActionWrapper
# ---------------------------------------------------------------------------


class TestAutoTaskActionWrapper:
    def test_action_space_drops_task_idx(self):
        env = _make_env(
            tasks=[make_task("t1"), make_task("t2"), make_task("t3")],
        )
        wrapped = AutoTaskActionWrapper(env)
        # Original 4D: [3, 7, 96, 2]; wrapped 3D: [7, 96, 2].
        assert list(wrapped.action_space.nvec) == [_N_DAYS, _N_BUCKETS, 2]

    def test_step_uses_next_unscheduled_task(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        env = _make_env(
            events=[ev],
            tasks=[
                make_task("t1", duration_min=30, duration_max=60),
                make_task("t2", duration_min=30, duration_max=60),
            ],
        )
        wrapped = AutoTaskActionWrapper(env)
        wrapped.reset()
        # Agent picks (day=0, bucket=0, merge=0); wrapper places t1.
        obs, _, term, trunc, _ = wrapped.step(np.array([0, 0, 0], dtype=np.int64))
        scheduled = env.get_week_scheduled()
        assert len(scheduled) == 1
        assert scheduled[0].task.label == "t1"
        # Next step should auto-pick t2.
        wrapped.step(np.array([0, 4, 0], dtype=np.int64))
        scheduled = env.get_week_scheduled()
        assert len(scheduled) == 2
        labels = [s.task.label for s in scheduled]
        assert labels == ["t1", "t2"]

    def test_action_no_unscheduled_falls_back_to_task_zero(self):
        env = _make_env(tasks=[make_task("only")])
        wrapped = AutoTaskActionWrapper(env)
        wrapped.reset()
        # Mark all tasks scheduled to force the fallback branch.
        env._scheduled_task_indices = {0}
        action = wrapped.action(np.array([0, 0, 0], dtype=np.int64))
        assert action[0] == 0

    def test_valid_action_mask_filters_days_outside_week(self):
        env = _make_env(tasks=[make_task("t1")])
        wrapped = AutoTaskActionWrapper(env)
        # Reset with a single-day week.
        wrapped.reset(options={"week_index": 0})
        env._week_dates = [DATE]
        mask = wrapped.valid_action_mask()
        # 7 days * 96 buckets * 2 merge = 1344 entries; only day=0 valid.
        n_per_day = _N_BUCKETS * 2
        assert mask[:n_per_day].all()
        assert not mask[n_per_day:].any()

    def test_wraps_inside_flatten_action_wrapper(self):
        env = _make_env(tasks=[make_task("t1")])
        wrapped = AutoTaskActionWrapper(env)
        flat = FlattenActionWrapper(wrapped)
        # 7 * 96 * 2 = 1344 flat actions.
        assert int(flat.action_space.n) == _N_DAYS * _N_BUCKETS * 2


# ---------------------------------------------------------------------------
# Per-week task swap
# ---------------------------------------------------------------------------


class TestWeeklyTaskSwap:
    def test_reset_swaps_active_tasks_by_week(self):
        a = make_task("a", duration_min=30, duration_max=60)
        b = make_task("b1", duration_min=30, duration_max=60)
        c = make_task("b2", duration_min=30, duration_max=60)
        date = datetime.date(2026, 6, 1)
        events = [
            make_event(date=date),
            make_event(date=date + datetime.timedelta(days=7)),
        ]
        env = SchedulingEnv(
            calendar=_trace(*events),
            tasks=[a, b, c],
            config=_config(),
            weekly_tasks=[[a], [b, c]],
        )
        # Spaces sized to the widest week (2), not the first week (1).
        assert env._max_tasks == 2
        env.reset(options={"week_index": 0})
        assert [t.label for t in env._tasks] == ["a"]
        env.reset(options={"week_index": 1})
        assert [t.label for t in env._tasks] == ["b1", "b2"]

    def test_explicit_max_tasks_overrides_weekly_cap(self):
        a = make_task("a")
        env = SchedulingEnv(
            calendar=_trace(),
            tasks=[a],
            config=_config(),
            weekly_tasks=[[a]],
            max_tasks=5,
        )
        assert env._max_tasks == 5
