"""Gymnasium environment for week-by-week task scheduling.

Design principles:
  1. `observation_space` is `Box(float32)`; compatible with any RL algorithm.
  2. `action_space` is `MultiDiscrete([n_tasks, 7, 96, 2])`; precise.
     DQN users should wrap with :class:`FlattenActionWrapper`.
  3. `step()` returns the 5-tuple `(obs, reward, terminated, truncated, info)`
     using the post-0.26 Gymnasium API.  `terminated` (all tasks placed) and
     `truncated` (max_steps exceeded) are kept separate for correct value
     bootstrapping in e.g. PPO.
  4. The class is picklable and vectorizable via `gymnasium.vector`
     (no closures stored as instance attributes; `np_random` handled by
     `super().reset(seed=seed)`).
  5. Reward shaping, action masking, and frame stacking live in Wrapper
     subclasses; the core env stays simple and reusable.

Observation vector layout (float32)::

    [0  : 7*96]           week occupancy bitmap (15-min buckets, 7 days)
    [7*96 : 7*96+T*6]     task feature matrix   (max_tasks × 6 features)
    [-1]                  elapsed fraction       (step / max_steps)

Task features per task::

    0  duration_min / 1440
    1  duration_max / 1440
    2  intensity    / 5
    3  is_dividable   (0 or 1)
    4  is_concurrent  (0 or 1)
    5  is_scheduled   (0 or 1)

Action vector::

    [task_idx, day_idx, bucket_idx, merge_flag]
    task_idx   in [0, max_tasks)
    day_idx    in [0, 7)         ; day within the current week
    bucket_idx in [0, 96)        ; 15-min slot start (bucket × 15 = minutes)
    merge_flag in {0, 1}         ; 0 = standalone, 1 = merge into event
"""

from __future__ import annotations

import datetime
from typing import Any, Callable

import gymnasium
import numpy as np
from gymnasium import spaces

from src.scripts.persona.config.schema import AllenPairRule
from src.scripts.scenarios.augmentation.greedy import (
    _get_horizon_dates,
    _get_weekly_chunks,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, ObservationConfig
from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import (
    _build_rule_index,
    build_admissible_rx,
    compute_allen_relation,
    is_overlapping,
)
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_TASK_FEATURES: int = 6  # features per task in the observation vector
_N_BUCKETS: int = 96  # default 15-min slots per day (1440 / 15)
_N_DAYS: int = 7  # days per week
_DEFAULT_BUCKET_MINUTES: int = 15


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------


class SchedulingEnv(gymnasium.Env):
    """Gymnasium environment for week-by-week task scheduling.

    One episode covers one week of scheduling.  Each step proposes placing
    (or implicitly skipping via an invalid action) one task.

    Args:
        calendar: the person's existing `CalendarTrace`.
        tasks: desired-behavior tasks to place.
        config: augmentation configuration (merge threshold, etc.).
        semantic: `SemanticCompatibility` for merge checks.
        allen_rules: `AllenPairRule` list for admissible-relation lookup.
        max_tasks: fixed observation/action dimension; defaults to
            `len(tasks)`.  Set higher when episodes use task subsets.
        max_steps: episode step limit; defaults to `max(10, 3 × n_tasks)`.
        week_index: which 7-day chunk of the horizon to schedule; can be
            overridden per episode via `reset(options={"week_index": k})`.
        render_mode: `"ansi"` for text output; `None` for no rendering.
    """

    metadata: dict = {"render_modes": ["ansi"]}

    def __init__(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
        config: AugmentationConfig,
        *,
        semantic: SemanticCompatibility | None = None,
        allen_rules: list[AllenPairRule] | None = None,
        max_tasks: int | None = None,
        max_steps: int | None = None,
        week_index: int = 0,
        render_mode: str | None = None,
        observation: ObservationConfig | None = None,
        reward_hook: Callable[["SchedulingEnv"], float] | None = None,
        bucket_minutes: int = _DEFAULT_BUCKET_MINUTES,
        per_step_shaping: float = 0.0,
        max_task_attempts: int = 0,
        weekly_tasks: list[list[RecommendedTask]] | None = None,
    ) -> None:
        super().__init__()

        self._calendar = calendar
        # `weekly_tasks` swaps the active task set per week in `reset`; the
        # active set starts on the first week (or the flat `tasks`).
        self._weekly_tasks = weekly_tasks
        self._tasks = list(weekly_tasks[0]) if weekly_tasks else list(tasks)
        self._config = config
        self._semantic = semantic if semantic is not None else SemanticCompatibility()
        self._allen_rules: list[AllenPairRule] = allen_rules or []
        self._rule_index = _build_rule_index(self._allen_rules)
        # Size the spaces from a horizon-wide cap (the widest week) so they
        # stay fixed when a week's task set is smaller; the feature loop
        # zero-pads the rest.
        if max_tasks is not None:
            cap = max_tasks
        elif weekly_tasks:
            cap = max((len(week) for week in weekly_tasks), default=len(self._tasks))
        else:
            cap = len(self._tasks)
        self._max_tasks = max(1, cap)
        self._max_steps = (
            max_steps if max_steps is not None else max(10, 3 * len(tasks))
        )
        self._week_index = week_index
        self.render_mode = render_mode
        self._reward_hook = reward_hook
        self._carry_over: list[ScheduledTask] = []
        # Time-bucket granularity. 15 keeps the historical action space
        # (96 buckets/day); 60 cuts it 4x to make Q-learning tractable
        # with limited training budget.
        self._bucket_minutes = max(1, int(bucket_minutes))
        self._n_buckets = max(1, 1440 // self._bucket_minutes)
        # Potential-based per-step shaping reward magnitude. Each valid
        # placement adds `per_step_shaping / n_tasks` to env.step's
        # reward, so a fully-placed episode sums to per_step_shaping
        # from shaping alone. Combined with the reward_hook's terminal
        # weighted gain, this gives DQN a denser signal than the
        # terminal-only reward (which suffers from gamma^N decay over
        # the per-week 30-60 step episode).
        self._per_step_shaping = float(per_step_shaping)
        # After this many failed attempts on one outstanding task, give up on
        # it so the episode advances instead of spinning on an unplaceable
        # task until `max_steps`. 0 disables the guard.
        self._max_task_attempts = max(0, int(max_task_attempts))

        # Pre-build events_by_date once; immutable across episodes.
        self._events_by_date: dict[datetime.date, list[CalendarEvent]] = {}
        for ev in calendar.events:
            self._events_by_date.setdefault(ev.date, []).append(ev)

        # Context categories opted into via observation.contexts.
        # The observation vector grows by `_N_DAYS * len(visible)` slots when
        # at least one category is enabled; trained policies that ignore
        # contexts see an all-zero tail when contexts are absent.
        self._visible_context_categories: list[str] = (
            list(observation.contexts) if observation is not None else []
        )
        self._contexts_by_date: dict[datetime.date, dict[str, list[ContextEpisode]]] = (
            self._build_contexts_index(calendar.contexts)
        )

        # Spaces
        n_ctx_slots = _N_DAYS * len(self._visible_context_categories)
        obs_dim = (
            _N_DAYS * self._n_buckets
            + self._max_tasks * _N_TASK_FEATURES
            + 1
            + n_ctx_slots
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [self._max_tasks, _N_DAYS, self._n_buckets, 2]
        )

        # Episode state; reset in reset()
        self._week_dates: list[datetime.date] = []
        self._placed_by_date: dict[datetime.date, list[tuple[int, int]]] = {}
        self._consumed_merges: set[tuple[datetime.date, str]] = set()
        self._scheduled: list[ScheduledTask] = []
        self._scheduled_task_indices: set[int] = set()
        self._skipped_task_indices: set[int] = set()
        self._task_attempts: dict[int, int] = {}
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._placed_by_date = {}
        self._consumed_merges = set()
        self._scheduled = []
        self._scheduled_task_indices = set()
        self._skipped_task_indices = set()
        self._task_attempts = {}
        self._step_count = 0

        opts = options or {}
        # Persist the current week so SB3's internal resets (which call
        # reset() without options between training episodes) stay on the
        # same week as the explicit augmenter rollout. Without this, all
        # SB3 training transitions roll out on the env's __init__ week
        # while the outer loop varies the week, corrupting the buffer.
        week_index = int(opts.get("week_index", self._week_index))
        self._week_index = week_index
        # Swap the active task set to this week's batch. Keyed off the
        # persisted week index so SB3's option-less internal resets stay on
        # the same week's tasks as the outer rollout.
        if self._weekly_tasks:
            idx = min(week_index, len(self._weekly_tasks) - 1)
            self._tasks = list(self._weekly_tasks[idx])
        self._carry_over = list(opts.get("carry_over", []))
        horizon = _get_horizon_dates(self._calendar)
        weeks = _get_weekly_chunks(horizon)

        if weeks and week_index < len(weeks):
            self._week_dates = weeks[week_index]
        elif horizon:
            self._week_dates = horizon[:_N_DAYS]
        else:
            self._week_dates = []

        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        task_idx = int(action[0])
        day_idx = int(action[1])
        bucket_idx = int(action[2])
        merge_flag = int(action[3])

        self._step_count += 1
        info: dict[str, Any] = {}

        valid = self._apply_action(task_idx, day_idx, bucket_idx, merge_flag, info)
        # Count failed attempts on the current task and give up on it after
        # the cap so the episode advances. Only failures on a not-yet-resolved
        # task count.
        if (
            self._max_task_attempts > 0
            and not valid
            and 0 <= task_idx < len(self._tasks)
            and task_idx not in self._scheduled_task_indices
            and task_idx not in self._skipped_task_indices
        ):
            self._task_attempts[task_idx] = self._task_attempts.get(task_idx, 0) + 1
            if self._task_attempts[task_idx] >= self._max_task_attempts:
                self._skipped_task_indices.add(task_idx)
                info["skipped"] = f"gave up on task {task_idx} after cap"
        # When a reward_hook is wired, terminal reward comes from the
        # hook and per-step shaping is controlled by `per_step_shaping`.
        # Without a hook, fall back to the legacy `1/n` per-valid-step
        # reward so older callers keep working.
        if self._reward_hook is None:
            reward = (1.0 / max(1, len(self._tasks))) if valid else 0.0
        else:
            reward = 0.0
            if valid and self._per_step_shaping > 0.0:
                reward = self._per_step_shaping / max(1, len(self._tasks))

        obs = self._observe()
        # Episode ends when every task is resolved (placed or given up on).
        resolved = len(self._scheduled_task_indices) + len(self._skipped_task_indices)
        terminated = resolved == len(self._tasks)
        truncated = self._step_count >= self._max_steps

        if (terminated or truncated) and self._reward_hook is not None:
            reward += float(self._reward_hook(self))

        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        lines = [
            f"Step {self._step_count}/{self._max_steps}",
            f"Scheduled: {len(self._scheduled_task_indices)}/{len(self._tasks)}",
        ]
        for day in self._week_dates:
            n_ev = len(self._events_by_date.get(day, []))
            n_pl = len(self._placed_by_date.get(day, []))
            lines.append(f"  {day}: {n_ev} calendar events, {n_pl} placed tasks")
        return "\n".join(lines)

    def get_solution(self) -> SchedulingSolution:
        """Build a `SchedulingSolution` from the current episode state."""
        base_events = list(self._calendar.events)  # always the full original calendar
        unscheduled = [
            t
            for i, t in enumerate(self._tasks)
            if i not in self._scheduled_task_indices
        ]
        aug_cal = AugmentedCalendar(
            person_id=self._calendar.person_id,
            base_events=base_events,
            scheduled_tasks=list(self._scheduled),  # ALL tasks, not just standalone
        )
        return SchedulingSolution(
            person_id=self._calendar.person_id,
            augmented_calendar=aug_cal,
            tasks=list(self._tasks),
            scheduled=list(self._scheduled),
            unscheduled=unscheduled,
        )

    def get_week_scheduled(self) -> list[ScheduledTask]:
        """Return placements made in the current episode (one week)."""
        return list(self._scheduled)

    def get_carry_over(self) -> list[ScheduledTask]:
        """Return placements carried over from prior weeks."""
        return list(self._carry_over)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_contexts_index(
        self,
        episodes: list[ContextEpisode],
    ) -> dict[datetime.date, dict[str, list[ContextEpisode]]]:
        """Group context episodes by (date, category) once at env construction."""
        out: dict[datetime.date, dict[str, list[ContextEpisode]]] = {}
        for ep in episodes:
            out.setdefault(ep.date, {}).setdefault(ep.category, []).append(ep)
        return out

    def _observe(self) -> np.ndarray:
        n_ctx_slots = _N_DAYS * len(self._visible_context_categories)
        bm = self._bucket_minutes
        nb = self._n_buckets
        obs = np.zeros(
            _N_DAYS * nb + self._max_tasks * _N_TASK_FEATURES + 1 + n_ctx_slots,
            dtype=np.float32,
        )
        for day_idx, day in enumerate(self._week_dates[:_N_DAYS]):
            base = day_idx * nb
            for ev in self._events_by_date.get(day, []):
                s = ev.start_minutes // bm
                e = min(nb, (ev.end_minutes + bm - 1) // bm)
                obs[base + s : base + e] = 1.0
            for s_m, e_m in self._placed_by_date.get(day, []):
                s = s_m // bm
                e = min(nb, (e_m + bm - 1) // bm)
                obs[base + s : base + e] = 1.0

        feat_base = _N_DAYS * nb
        for i, task in enumerate(self._tasks[: self._max_tasks]):
            b = feat_base + i * _N_TASK_FEATURES
            obs[b + 0] = task.duration_min / 1440.0
            obs[b + 1] = task.duration_max / 1440.0
            obs[b + 2] = task.intensity / 5.0
            obs[b + 3] = float(task.is_dividable)
            obs[b + 4] = float(task.is_concurrent)
            obs[b + 5] = 1.0 if i in self._scheduled_task_indices else 0.0

        elapsed_idx = feat_base + self._max_tasks * _N_TASK_FEATURES
        obs[elapsed_idx] = self._step_count / self._max_steps

        if self._visible_context_categories:
            ctx_base = elapsed_idx + 1
            n_cat = len(self._visible_context_categories)
            for day_idx, day in enumerate(self._week_dates[:_N_DAYS]):
                cats_today = self._contexts_by_date.get(day, {})
                for cat_idx, cat in enumerate(self._visible_context_categories):
                    if cats_today.get(cat):
                        obs[ctx_base + day_idx * n_cat + cat_idx] = 1.0
        return obs

    def _apply_action(
        self,
        task_idx: int,
        day_idx: int,
        bucket_idx: int,
        merge_flag: int,
        info: dict,
    ) -> bool:
        """Try to apply one action. Returns True on valid placement."""
        if task_idx >= len(self._tasks):
            info["reason"] = "task_idx beyond available tasks"
            return False
        if task_idx in self._scheduled_task_indices:
            info["reason"] = "task already scheduled"
            return False
        if day_idx >= len(self._week_dates):
            info["reason"] = "day_idx outside current week"
            return False

        task = self._tasks[task_idx]
        day = self._week_dates[day_idx]
        start_min = bucket_idx * self._bucket_minutes
        end_min = start_min + task.duration_min

        if end_min > 1440:
            info["reason"] = "placement extends past midnight"
            return False

        if merge_flag:
            return self._try_merge(task, task_idx, day, start_min, end_min, info)
        return self._try_standalone(task, task_idx, day, start_min, end_min, info)

    def _try_merge(
        self,
        task: RecommendedTask,
        task_idx: int,
        day: datetime.date,
        start_min: int,
        end_min: int,
        info: dict,
    ) -> bool:
        day_events = self._events_by_date.get(day, [])
        target = next(
            (
                e
                for e in day_events
                if e.start_minutes <= start_min < e.end_minutes
                and self._semantic.score(task.label, e.label)
                >= self._config.merge_threshold
                and (day, e.label) not in self._consumed_merges
            ),
            None,
        )
        if target is None:
            info["reason"] = "no compatible event for merge at this slot"
            return False

        st = ScheduledTask(
            task=task,
            start_minutes=start_min,
            end_minutes=min(end_min, target.end_minutes),
            is_standalone=False,
            concurrent_with=target.label,
            date=day,
        )
        self._consumed_merges.add((day, target.label))
        self._scheduled.append(st)
        self._scheduled_task_indices.add(task_idx)
        self._placed_by_date.setdefault(day, []).append((start_min, st.end_minutes))
        info["placement"] = (
            f"co-scheduled {task.label!r} with {target.label!r} on {day}"
        )
        return True

    def _try_standalone(
        self,
        task: RecommendedTask,
        task_idx: int,
        day: datetime.date,
        start_min: int,
        end_min: int,
        info: dict,
    ) -> bool:
        for ev in self._events_by_date.get(day, []):
            admissible = build_admissible_rx(task, ev, self._rule_index)
            rel = compute_allen_relation(
                start_min, end_min, ev.start_minutes, ev.end_minutes
            )
            if rel not in admissible:
                info["reason"] = f"conflicts with existing event {ev.label!r}"
                return False

        for ps, pe in self._placed_by_date.get(day, []):
            if is_overlapping(start_min, end_min, ps, pe) and not task.is_concurrent:
                info["reason"] = "overlaps already-placed task"
                return False

        st = ScheduledTask(
            task=task,
            start_minutes=start_min,
            end_minutes=end_min,
            is_standalone=True,
            concurrent_with=None,
            date=day,
        )
        self._scheduled.append(st)
        self._scheduled_task_indices.add(task_idx)
        self._placed_by_date.setdefault(day, []).append((start_min, end_min))
        info["placement"] = (
            f"placed {task.label!r} at {day} {start_min // 60:02d}:{start_min % 60:02d}"
        )
        return True


# ---------------------------------------------------------------------------
# Wrapper: action masking
# ---------------------------------------------------------------------------


class ActionMaskWrapper(gymnasium.Wrapper):
    """Adds `valid_action_mask()` for masked RL algorithms (e.g., MaskablePPO).

    Returns a flat boolean array of length `prod(action_space.nvec)`
    where `True` means the action is a priori valid (not guaranteed conflict-free,
    but avoids obviously invalid moves such as placing an already-scheduled task
    or targeting a day outside the current week).
    """

    def valid_action_mask(self) -> np.ndarray:
        env: SchedulingEnv = self.unwrapped  # type: ignore[assignment]
        n_tasks, n_days, n_buckets, n_merge = (int(v) for v in env.action_space.nvec)
        total = n_tasks * n_days * n_buckets * n_merge
        mask = np.ones(total, dtype=bool)

        day_stride = n_buckets * n_merge
        task_stride = n_days * day_stride

        for i in range(n_tasks):
            if i in env._scheduled_task_indices or i >= len(env._tasks):
                mask[i * task_stride : (i + 1) * task_stride] = False
                continue
            for d in range(n_days):
                if d >= len(env._week_dates):
                    off = i * task_stride + d * day_stride
                    mask[off : off + day_stride] = False

        return mask


# ---------------------------------------------------------------------------
# Wrapper: hide task_idx and auto-pick the next unscheduled task
# ---------------------------------------------------------------------------


class AutoTaskActionWrapper(gymnasium.ActionWrapper):
    """Drops the `task_idx` dimension from the agent's action space.

    The env always places the next unscheduled task; the agent only
    decides `(day_idx, bucket_idx, merge_flag)`. Shrinks the action
    space by a factor of `n_tasks` so the Q-network has far fewer
    outputs to learn. Also exposes a `valid_action_mask()` over the
    reduced 3D space so the agent's exploration only samples
    in-week (day, bucket, merge) tuples.
    """

    def __init__(self, env: "SchedulingEnv") -> None:
        super().__init__(env)
        nvec = env.action_space.nvec.tolist()
        self.action_space = spaces.MultiDiscrete([int(v) for v in nvec[1:]])

    def action(self, action: np.ndarray) -> np.ndarray:
        """Prepend the next task not yet scheduled or skipped."""
        env: SchedulingEnv = self.env.unwrapped  # type: ignore[assignment]
        n = len(env._tasks)
        task_idx = next(
            (
                i
                for i in range(n)
                if i not in env._scheduled_task_indices
                and i not in env._skipped_task_indices
            ),
            0,
        )
        return np.array([task_idx, *action], dtype=np.int64)

    def valid_action_mask(self) -> np.ndarray:
        """Return the day-validity mask over the reduced 3D action space."""
        env: SchedulingEnv = self.env.unwrapped  # type: ignore[assignment]
        n_days, n_buckets, n_merge = (int(v) for v in self.action_space.nvec)
        total = n_days * n_buckets * n_merge
        mask = np.ones(total, dtype=bool)
        day_stride = n_buckets * n_merge
        for d in range(n_days):
            if d >= len(env._week_dates):
                mask[d * day_stride : (d + 1) * day_stride] = False
        return mask


# ---------------------------------------------------------------------------
# Wrapper: flatten MultiDiscrete to Discrete for DQN
# ---------------------------------------------------------------------------


class FlattenActionWrapper(gymnasium.ActionWrapper):
    """Converts `MultiDiscrete` action to a single `Discrete` integer.

    Enables DQN (which requires a `Discrete` action space) while keeping
    the underlying env's `MultiDiscrete` semantics intact.

    Encoding uses row-major (C-order) linearisation::

        flat = a0 * (n1*n2*n3) + a1 * (n2*n3) + a2 * n3 + a3
    """

    def __init__(self, env: SchedulingEnv) -> None:
        super().__init__(env)
        nvec = env.action_space.nvec.tolist()
        self._nvec: list[int] = nvec
        self.action_space = spaces.Discrete(int(np.prod(nvec)))

    def action(self, action: int) -> np.ndarray:
        """Decode flat integer to MultiDiscrete component array."""
        idx = int(action)
        result = np.zeros(len(self._nvec), dtype=np.int64)
        for i in reversed(range(len(self._nvec))):
            result[i] = idx % self._nvec[i]
            idx //= self._nvec[i]
        return result

    def reverse_action(self, action: np.ndarray) -> int:
        """Encode MultiDiscrete components to flat integer."""
        result = 0
        for a, n in zip(action, self._nvec):
            result = result * n + int(a)
        return result
