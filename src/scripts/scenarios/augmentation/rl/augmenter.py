"""Online DQN augmenter that trains week-by-week per person.

For each person the augmenter creates a fresh agent, optionally warm-
starts it from a checkpoint file, runs one episode per calendar week,
adds the full weighted gain as a terminal reward, runs a configured
number of gradient steps, and carries placements forward into the
next week's environment. The final solution merges placements from
every week into one `SchedulingSolution`; a `weekly_records` attribute
is attached so the writer can emit a per-person weekly-gain sidecar.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

from src.scripts.persona.config.schema import AllenPairRule, DailyWindow, WindowRange
from src.scripts.scenarios.augmentation.base import Augmenter, HorizonHint
from src.scripts.scenarios.augmentation.greedy import (
    _get_horizon_dates,
    _get_weekly_chunks,
)
from src.scripts.scenarios.augmentation.rl.agent import RLAgent
from src.scripts.scenarios.augmentation.rl.env import (
    ActionMaskWrapper,
    AutoTaskActionWrapper,
    FlattenActionWrapper,
    SchedulingEnv,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, LossWeights
from src.scripts.scenarios.domain.calendar import AugmentedCalendar, CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityResolver,
    MetQuartiles,
)
from src.scripts.scenarios.metrics.loss import (
    SchedulingLoss,
    compute_applicability_mask,
)
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

logger = logging.getLogger(__name__)

# Keep a per-person seed inside numpy's 32-bit range.
_SEED_MODULUS = 2**31


def _person_seed(base_seed: int, person_id: str) -> int:
    """Return a stable per-person seed so a person trains the same everywhere.

    Uses a hash of `person_id` (not the base seed plus a position) so the value
    is independent of cohort order and identical across processes.
    """
    digest = hashlib.sha256(person_id.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % _SEED_MODULUS


PersonaConstraintsFor = Callable[[CalendarTrace], Any]
WindowMapFor = Callable[[CalendarTrace], Any]


_GAIN_NAMES: tuple[tuple[str, str], ...] = (
    ("cov", "recommended_task_coverage"),
    ("cal", "task_event_and_task_task_temporal_relations"),
    ("pref", "user_preference_deviation"),
    ("disp", "intensive_task_dispersion"),
    ("merge", "semantic_coscheduling_merge"),
    ("spread", "recommended_task_spread"),
    ("divide", "dividable_task_split_reward"),
    ("context_fit", "user_context_recommendation_fit"),
)


def _safe_epsilon(agent) -> float | None:
    """Return the current DQN exploration rate or `None` for other policies."""
    model = getattr(agent, "model", None)
    if model is None:
        return None
    rate = getattr(model, "exploration_rate", None)
    return None if rate is None else float(rate)


def _components_to_gains(components) -> dict[str, float | None]:
    """Return a mapping of long-name leg to `1 - loss` (or `None`)."""
    out: dict[str, float | None] = {}
    for attr, long_name in _GAIN_NAMES:
        value = getattr(components, attr)
        out[long_name] = None if value is None else 1.0 - float(value)
    return out


class RLAugmenter(Augmenter):
    """Online DQN augmenter with weekly training.

    Args:
        time_windows: named `[start, end]` windows from `environment.yaml`.
        daily_window: waking-hours bracket for placement.
        allen_rules: cross-event temporal rules.
        semantic: `SemanticCompatibility` for merge scoring.
        ruleset: parsed `RuleSet` for `L_cal`.
        matcher: `SelectorMatcher` for `L_cal`.
        resolver: `IntensityResolver` for `L_disp`.
        loss_weights: weights for the eight scheduling-gain legs.
        merge_threshold: minimum sigma for `L_merge`.
        buffer_minutes: ramp width for `L_cal` partial credit.
        half_life_days: lag half-life for `L_disp`.
        max_met: continuous MET normalizer for `L_disp`.
        divide_tolerance_pct: width of the `L_divide` band.
        persona_constraints_for: factory returning per-person `PersonaConstraints`.
        window_map_for: factory returning per-person `WindowMap`.
        context_links_by_uri: ontology `context_links` map for `L_context_fit`.
        max_tasks: cap on the env's task observation slot count.
        seed: integer seed forwarded to the env and agent.
    """

    def __init__(
        self,
        *,
        time_windows: dict[str, WindowRange] | None = None,
        daily_window: DailyWindow | None = None,
        allen_rules: list[AllenPairRule] | None = None,
        semantic: SemanticCompatibility | None = None,
        ruleset: RuleSet | None = None,
        matcher: SelectorMatcher | None = None,
        resolver: IntensityResolver | None = None,
        loss_weights: LossWeights | None = None,
        merge_threshold: float = 0.65,
        buffer_minutes: int = 30,
        half_life_days: float = 2.0,
        max_met: float = 16.8,
        divide_tolerance_pct: float = 0.15,
        persona_constraints_for: PersonaConstraintsFor | None = None,
        window_map_for: WindowMapFor | None = None,
        context_links_by_uri: dict[str, list[str]] | None = None,
        max_tasks: int | None = None,
        seed: int = 0,
    ) -> None:
        self._time_windows: dict[str, WindowRange] = time_windows or {}
        self._daily_window: DailyWindow = daily_window or DailyWindow()
        self._allen_rules: list[AllenPairRule] = allen_rules or []
        self._semantic = semantic if semantic is not None else SemanticCompatibility()
        self._ruleset = ruleset if ruleset is not None else RuleSet(rules=[])
        self._matcher = matcher if matcher is not None else SelectorMatcher()
        self._resolver = (
            resolver
            if resolver is not None
            else IntensityResolver(MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8))
        )
        self._loss_weights = loss_weights if loss_weights is not None else LossWeights()
        self._merge_threshold = merge_threshold
        self._buffer_minutes = buffer_minutes
        self._half_life_days = half_life_days
        self._max_met = max_met
        self._divide_tolerance_pct = divide_tolerance_pct
        self._persona_constraints_for = persona_constraints_for
        self._window_map_for = window_map_for
        self._context_links_by_uri = context_links_by_uri or {}
        self._max_tasks = max_tasks
        self._seed = int(seed)
        self._rng = None

    def set_constraint_factories(
        self,
        persona_constraints_for: PersonaConstraintsFor,
        window_map_for: WindowMapFor,
    ) -> None:
        """Wire per-person preference-constraint factories for the L_pref reward leg."""
        self._persona_constraints_for = persona_constraints_for
        self._window_map_for = window_map_for

    def augment(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
        config: AugmentationConfig,
        horizon: HorizonHint | None = None,
        *,
        weekly_tasks: list[list[RecommendedTask]] | None = None,
    ) -> SchedulingSolution:
        """Train a DQN week-by-week and return the combined solution.

        `weekly_tasks` swaps the active task set per week inside the env;
        when omitted, `tasks` is used for every week.
        """
        import numpy as _np

        self._weekly_tasks = weekly_tasks
        horizon_dates = _get_horizon_dates(calendar, horizon=horizon)
        if not horizon_dates or not tasks:
            return self._empty_solution(calendar, tasks)
        person_seed = _person_seed(self._seed, calendar.person_id)
        self._rng = _np.random.default_rng(person_seed)
        weeks = _get_weekly_chunks(horizon_dates) or [horizon_dates[:7]]
        loss_scorer = self._make_loss()
        persona_constraints = (
            self._persona_constraints_for(calendar)
            if self._persona_constraints_for is not None
            else None
        )
        window_map = (
            self._window_map_for(calendar) if self._window_map_for is not None else None
        )
        observation = getattr(config, "observation", None)
        observed_categories = (
            frozenset(observation.contexts)
            if observation is not None and observation.contexts
            else None
        )

        # Per-person input-applicability mask: the same set of legs the
        # benchmark scores at evaluate time. The embedding oracle stands
        # in for the judge merge signal during training (Q4 proxy).
        mask = compute_applicability_mask(
            tasks,
            calendar,
            self._loss_weights,
            persona_constraints=persona_constraints,
            semantic=self._semantic,
            context_links_by_uri=self._context_links_by_uri,
            observed_categories=observed_categories,
            merge_threshold=self._merge_threshold,
        )

        accumulated: list[ScheduledTask] = []

        def _reward_hook(env_inst) -> float:
            """Score the episode's week slice as the masked scheduling gain.

            Uses the input-applicability mask and zero-placement rule the
            benchmark reports at evaluate time: an in-mask leg with no
            signal, or a placement-free week, is charged full loss. This
            removes the do-nothing loophole the old renormalized gain left
            open, and makes the agent optimize the reported shape.
            """
            episode_scheduled = env_inst.get_week_scheduled()
            scored = self._solution_from(
                accumulated + episode_scheduled, tasks, calendar
            )
            week_loss, _ = loss_scorer.compute_weekly(
                scored,
                calendar,
                _reward_week_dates,
                time_windows=self._time_windows,
                persona_constraints=persona_constraints,
                window_map=window_map,
                observed_categories=observed_categories,
                mask=mask,
            )
            return 1.0 - week_loss

        _reward_week_dates: list[datetime.date] = list(weeks[0])
        env = self._build_env(
            calendar,
            tasks,
            config,
            observation,
            reward_hook=_reward_hook,
            weekly_tasks=self._weekly_tasks,
        )
        total_timesteps = max(1, len(weeks)) * max(1, config.rl.train_steps_per_week)
        agent = RLAgent(
            env,
            policy=config.rl.policy,
            rl_config=config.rl,
            total_timesteps=total_timesteps,
            device=config.rl.device,
            seed=person_seed,
        )
        if config.rl.checkpoint:
            agent.load(config.rl.checkpoint)

        weekly_records: list[dict] = []
        training_records: list[dict] = []
        for week_idx, week_dates in enumerate(weeks):
            _reward_week_dates[:] = list(week_dates)
            # Cross-week epsilon curriculum (opt-in): pin the exploration
            # rate per week so it decays across the whole horizon. SB3's
            # anneal restarts each weekly learn() and never spans weeks.
            if config.rl.cross_week_epsilon:
                agent.set_epsilon(self._epsilon_for_week(week_idx, config.rl))
            obs, _ = env.reset(
                seed=person_seed + week_idx,
                options={"week_index": week_idx, "carry_over": list(accumulated)},
            )
            done = False
            episode_rewards: list[float] = []
            episode_valid_steps = 0
            episode_steps = 0
            while not done:
                action = agent.act(obs, deterministic=False)
                action = self._mask_to_valid_action(env, action)
                obs2, reward, term, trunc, _ = env.step(action)
                agent.remember(obs, action, reward, obs2, term or trunc)
                episode_rewards.append(float(reward))
                episode_steps += 1
                if reward != 0.0:
                    episode_valid_steps += 1
                obs = obs2
                done = term or trunc
            week_scheduled = env.unwrapped.get_week_scheduled()
            candidate_solution = self._solution_from(
                accumulated + week_scheduled, tasks, calendar
            )
            week_gain, week_components = loss_scorer.compute_weekly(
                candidate_solution,
                calendar,
                list(week_dates),
                time_windows=self._time_windows,
                persona_constraints=persona_constraints,
                window_map=window_map,
                observed_categories=observed_categories,
                mask=mask,
            )
            weekly_gain = 1.0 - week_gain
            training_records.append(
                {
                    "week_index": week_idx + 1,
                    "episode_steps": episode_steps,
                    "episode_valid_placements": episode_valid_steps,
                    "episode_total_reward": sum(episode_rewards),
                    "episode_terminal_reward": (
                        float(episode_rewards[-1]) if episode_rewards else 0.0
                    ),
                    "epsilon_before_learn": _safe_epsilon(agent),
                    "tasks_placed_this_week": len(week_scheduled),
                    "tasks_total": len(tasks),
                }
            )
            # SB3 internal training rolls out on Wk_N's dates (env.reset
            # persists week_index now). The reward_hook scores
            # `accumulated + episode_scheduled` sliced to Wk_N, so the
            # committed-state view must still be weeks 1..N-1 during
            # learn(); the current week's placements are added AFTER.
            agent.learn(config.rl.train_steps_per_week)
            accumulated.extend(week_scheduled)
            weekly_records.append(
                {
                    "week_index": week_idx + 1,
                    "week_start": week_dates[0].isoformat(),
                    "weighted_gain": weekly_gain,
                    "gains": _components_to_gains(week_components),
                }
            )
            if not config.repeat_per_week:
                break

        if config.rl.checkpoint:
            target = Path(config.rl.checkpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            agent.save(target)
        solution = self._solution_from(accumulated, tasks, calendar)
        solution.weekly_records = weekly_records  # type: ignore[attr-defined]
        solution.rl_training_records = training_records  # type: ignore[attr-defined]
        return solution

    def _build_env(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
        config: AugmentationConfig,
        observation,
        reward_hook=None,
        weekly_tasks: list[list[RecommendedTask]] | None = None,
    ):
        """Return the wrapped environment ready for DQN action sampling."""
        # The env sizes its spaces from the widest week when `weekly_tasks`
        # is set, so a smaller week's batch zero-pads the rest.
        base = SchedulingEnv(
            calendar=calendar,
            tasks=tasks,
            config=config,
            semantic=self._semantic,
            allen_rules=self._allen_rules,
            max_tasks=self._max_tasks,
            observation=observation,
            reward_hook=reward_hook,
            bucket_minutes=config.rl.bucket_minutes,
            per_step_shaping=config.rl.per_step_shaping,
            max_task_attempts=config.rl.max_task_attempts,
            weekly_tasks=weekly_tasks,
        )
        middle = (
            AutoTaskActionWrapper(base)
            if config.rl.auto_pick_task
            else ActionMaskWrapper(base)
        )
        return FlattenActionWrapper(middle)

    def _mask_to_valid_action(self, env, action):
        """Return `action` if it passes the env's validity mask, else re-sample.

        Re-samples uniformly from the True entries of `valid_action_mask`.
        Falls back to the original action when the wrapper does not expose
        a mask or every entry is invalid.
        """
        import numpy as _np

        mask_fn = getattr(env.env, "valid_action_mask", None)
        if mask_fn is None:
            return action
        try:
            mask = mask_fn()
        except Exception:
            return action
        idx = int(action)
        if 0 <= idx < len(mask) and bool(mask[idx]):
            return action
        valid = _np.flatnonzero(mask)
        if valid.size == 0:
            return action
        rng = getattr(self, "_rng", None) or _np.random.default_rng()
        return int(valid[rng.integers(0, valid.size)])

    def _make_loss(self) -> SchedulingLoss:
        """Construct the per-augment `SchedulingLoss` scorer."""
        return SchedulingLoss(
            self._loss_weights,
            self._semantic,
            self._ruleset,
            self._matcher,
            self._resolver,
            merge_threshold=self._merge_threshold,
            buffer_minutes=self._buffer_minutes,
            half_life_days=self._half_life_days,
            max_met=self._max_met,
            divide_tolerance_pct=self._divide_tolerance_pct,
            context_links_by_uri=self._context_links_by_uri,
        )

    def _epsilon_for_week(self, week_idx: int, rl_cfg) -> float:
        """Linear epsilon decay from `epsilon_start` to `epsilon_end`."""
        start = float(rl_cfg.epsilon_start)
        end = float(rl_cfg.epsilon_end)
        decay = max(1, int(rl_cfg.epsilon_decay_weeks))
        progress = min(1.0, week_idx / decay)
        return start + (end - start) * progress

    def _solution_from(
        self,
        scheduled: list[ScheduledTask],
        tasks: list[RecommendedTask],
        calendar: CalendarTrace,
    ) -> SchedulingSolution:
        """Build a `SchedulingSolution` from accumulated weekly placements."""
        scheduled_labels = {st.task.label for st in scheduled}
        unscheduled = [t for t in tasks if t.label not in scheduled_labels]
        aug_cal = AugmentedCalendar(
            person_id=calendar.person_id,
            base_events=list(calendar.events),
            scheduled_tasks=list(scheduled),
        )
        return SchedulingSolution(
            person_id=calendar.person_id,
            augmented_calendar=aug_cal,
            tasks=list(tasks),
            scheduled=list(scheduled),
            unscheduled=unscheduled,
        )

    def _empty_solution(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
    ) -> SchedulingSolution:
        """Return an empty solution and attach an empty `weekly_records`."""
        aug_cal = AugmentedCalendar(
            person_id=calendar.person_id,
            base_events=list(calendar.events),
        )
        sol = SchedulingSolution(
            person_id=calendar.person_id,
            augmented_calendar=aug_cal,
            tasks=list(tasks),
            scheduled=[],
            unscheduled=list(tasks),
        )
        sol.weekly_records = []  # type: ignore[attr-defined]
        sol.rl_training_records = []  # type: ignore[attr-defined]
        return sol
