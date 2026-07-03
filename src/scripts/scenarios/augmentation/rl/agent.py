"""DQN / PPO / random agent wrapper around `SchedulingEnv`.

Exposes a granular online API (`act`, `remember`, `terminal_bonus`,
`learn`, `save`, `load`) so the augmenter can drive a week-by-week
training loop. `stable-baselines3` is an optional dependency; the
random policy needs nothing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.buffers import ReplayBuffer

    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover  optional dep on CPU image
    _SB3_AVAILABLE = False

if TYPE_CHECKING:
    import gymnasium

    from src.scripts.scenarios.config.schema import RLConfig

_POLICY_CLASSES: dict[str, Any] = {}
if _SB3_AVAILABLE:
    _POLICY_CLASSES = {"ppo": PPO, "dqn": DQN}


class RLAgent:
    """Online RL agent driving a `SchedulingEnv`.

    Args:
        env: gymnasium environment to act in.
        policy: `dqn`, `ppo`, or `random`.
        rl_config: hyperparameters; defaults applied when `None`.
        device: torch device (`"cuda"` / `"cpu"` / `"auto"`).
    """

    def __init__(
        self,
        env: "gymnasium.Env",
        *,
        policy: str = "dqn",
        rl_config: "RLConfig | None" = None,
        device: str = "auto",
        total_timesteps: int | None = None,
        seed: int | None = None,
    ) -> None:
        valid = {"random", "ppo", "dqn"}
        if policy not in valid:
            raise ValueError(
                f"Unknown policy {policy!r}. Choose one of: {sorted(valid)}."
            )
        self._env = env
        self._policy = policy
        self._cfg = rl_config
        self._model: Any = None
        self._last_obs: np.ndarray | None = None
        self._last_action: np.ndarray | int | None = None

        if policy != "random":
            if not _SB3_AVAILABLE:
                raise RuntimeError(
                    f"stable-baselines3 is required for policy={policy!r}. "
                    "Install with: pip install stable-baselines3"
                )
            cls = _POLICY_CLASSES[policy]
            kwargs: dict[str, Any] = {"verbose": 0, "device": device}
            if rl_config is not None:
                kwargs["learning_rate"] = rl_config.learning_rate
                kwargs["gamma"] = rl_config.gamma
                # Per-person seed when given, else the shared config seed.
                kwargs["seed"] = rl_config.seed if seed is None else int(seed)
                if policy == "dqn":
                    kwargs["buffer_size"] = rl_config.buffer_size
                    kwargs["batch_size"] = rl_config.batch_size
                    kwargs["exploration_initial_eps"] = rl_config.epsilon_start
                    kwargs["exploration_final_eps"] = rl_config.epsilon_end
                    # `exploration_fraction` pins the linear anneal to the
                    # caller-declared training horizon so the eps schedule
                    # tracks the whole curriculum rather than restarting
                    # on every `learn()` call.
                    if total_timesteps is not None:
                        decay_weeks = max(1, int(rl_config.epsilon_decay_weeks))
                        kwargs["exploration_fraction"] = max(
                            1e-4,
                            min(
                                1.0,
                                (decay_weeks * rl_config.train_steps_per_week)
                                / max(1, total_timesteps),
                            ),
                        )
                    # Faster target sync and earlier learning so the
                    # per-week budgets actually move Q-values.
                    kwargs["target_update_interval"] = 500
                    kwargs["learning_starts"] = 200
                    kwargs["train_freq"] = 1
                    kwargs["gradient_steps"] = 2
                    kwargs["policy_kwargs"] = {"net_arch": [256, 256, 128]}
            self._model = cls("MlpPolicy", env, **kwargs)
            self._total_timesteps = total_timesteps

    @property
    def policy(self) -> str:
        """Return the policy name."""
        return self._policy

    @property
    def model(self) -> Any:
        """Return the SB3 model; `None` for the random policy."""
        return self._model

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> Any:
        """Return an action for `obs`; random when no model is bound."""
        self._last_obs = np.asarray(obs)
        if self._model is None:
            action = self._env.action_space.sample()
        else:
            action, _ = self._model.predict(obs, deterministic=deterministic)
        self._last_action = action
        return action

    def remember(
        self,
        obs: np.ndarray,
        action: Any,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into the DQN replay buffer; no-op otherwise."""
        if self._model is None or self._policy != "dqn":
            return
        buffer: ReplayBuffer = self._model.replay_buffer
        buffer.add(
            np.asarray(obs, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray([action]),
            np.asarray([float(reward)]),
            np.asarray([bool(done)]),
            [{}],
        )

    def terminal_bonus(self, reward: float) -> None:
        """Add a terminal-step transition carrying the week's gain reward."""
        if self._last_obs is None or self._last_action is None:
            return
        self.remember(
            self._last_obs,
            self._last_action,
            float(reward),
            self._last_obs,
            True,
        )

    def learn(self, total_timesteps: int) -> None:
        """Run `total_timesteps` gradient steps; no-op for the random policy."""
        if self._model is None or total_timesteps <= 0:
            return
        self._model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            log_interval=None,
        )

    def save(self, path: str | os.PathLike) -> None:
        """Save the model to `path`; no-op for the random policy."""
        if self._model is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(target))

    def load(self, path: str | os.PathLike) -> None:
        """Load model weights from `path`; no-op when the file is missing."""
        if self._model is None:
            return
        target = Path(path)
        if not target.exists():
            return
        cls = _POLICY_CLASSES[self._policy]
        self._model = cls.load(str(target), env=self._env)

    def set_epsilon(self, value: float) -> None:
        """Pin the DQN exploration rate; no-op for other policies."""
        if self._model is None or self._policy != "dqn":
            return
        clipped = max(0.0, min(1.0, float(value)))
        self._model.exploration_rate = clipped
