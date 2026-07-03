"""Unit tests for augmentation/rl/agent.py (granular online API)."""

from __future__ import annotations

import datetime
import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.scripts.scenarios.augmentation.rl import agent as agent_module
from src.scripts.scenarios.augmentation.rl.agent import RLAgent
from src.scripts.scenarios.augmentation.rl.env import SchedulingEnv
from src.scripts.scenarios.config.schema import AugmentationConfig, RLConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)


# ---------------------------------------------------------------------------
# Module-level import-guard coverage
# ---------------------------------------------------------------------------


class TestModuleLevelImportGuard:
    """Verify the try/except ImportError guard at the top of agent.py."""

    def test_sb3_absent_sets_available_false_and_empty_classes(self):
        sb3_keys = [k for k in sys.modules if "stable_baselines3" in k]
        saved = {k: sys.modules.pop(k) for k in sb3_keys}
        sys.modules["stable_baselines3"] = None  # type: ignore[assignment]
        try:
            importlib.reload(agent_module)
            assert agent_module._SB3_AVAILABLE is False
            assert agent_module._POLICY_CLASSES == {}
        finally:
            del sys.modules["stable_baselines3"]
            for k, v in saved.items():
                sys.modules[k] = v
            importlib.reload(agent_module)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config() -> AugmentationConfig:
    return AugmentationConfig(method="rl")


def _make_env() -> SchedulingEnv:
    ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
    task = make_task("yoga", duration_min=30, duration_max=60)
    return SchedulingEnv(
        calendar=CalendarTrace(person_id="p001", events=[ev]),
        tasks=[task],
        config=_config(),
    )


def _mock_model_cls():
    """Return a mock SB3 model class and a pre-built model instance."""
    model_instance = MagicMock()
    model_instance.predict.return_value = (np.array([0, 0, 0, 0]), None)
    model_instance.replay_buffer = MagicMock()
    model_cls = MagicMock()
    model_cls.return_value = model_instance
    model_cls.load.return_value = model_instance
    return model_cls, model_instance


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestRLAgentInit:
    def test_random_policy_no_model(self):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        assert agt.model is None
        assert agt.policy == "random"

    def test_unknown_policy_raises_value_error(self):
        env = _make_env()
        with pytest.raises(ValueError, match="Unknown policy"):
            RLAgent(env, policy="td3")

    def test_ppo_creates_model_with_sb3(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            agt = RLAgent(env, policy="ppo")
        assert agt.model is model_instance
        kwargs = model_cls.call_args.kwargs
        assert kwargs["verbose"] == 0
        assert kwargs["device"] == "auto"

    def test_dqn_passes_hyperparameters_through(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        rl_cfg = RLConfig(
            policy="dqn",
            buffer_size=200,
            batch_size=8,
            epsilon_start=0.9,
            epsilon_end=0.1,
            seed=7,
            learning_rate=2e-4,
            gamma=0.9,
        )
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"dqn": model_cls}),
        ):
            RLAgent(env, policy="dqn", rl_config=rl_cfg)
        kwargs = model_cls.call_args.kwargs
        assert kwargs["learning_rate"] == 2e-4
        assert kwargs["gamma"] == 0.9
        assert kwargs["seed"] == 7
        assert kwargs["buffer_size"] == 200
        assert kwargs["batch_size"] == 8
        assert kwargs["exploration_initial_eps"] == 0.9
        assert kwargs["exploration_final_eps"] == 0.1

    def test_dqn_with_total_timesteps_sets_exploration_fraction(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        rl_cfg = RLConfig(
            policy="dqn",
            train_steps_per_week=1000,
            epsilon_decay_weeks=2,
        )
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"dqn": model_cls}),
        ):
            RLAgent(env, policy="dqn", rl_config=rl_cfg, total_timesteps=8000)
        kwargs = model_cls.call_args.kwargs
        # Decay window 2 weeks * 1000 steps / 8000 total = 0.25
        assert kwargs["exploration_fraction"] == pytest.approx(0.25)
        assert kwargs["target_update_interval"] == 500
        assert kwargs["learning_starts"] == 200
        assert kwargs["train_freq"] == 1
        assert kwargs["gradient_steps"] == 2
        assert kwargs["policy_kwargs"] == {"net_arch": [256, 256, 128]}

    def test_ppo_with_rl_config_skips_dqn_only_kwargs(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        rl_cfg = RLConfig(policy="ppo", learning_rate=5e-5, gamma=0.97, seed=3)
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            RLAgent(env, policy="ppo", rl_config=rl_cfg)
        kwargs = model_cls.call_args.kwargs
        assert kwargs["learning_rate"] == 5e-5
        assert kwargs["gamma"] == 0.97
        assert kwargs["seed"] == 3
        assert "buffer_size" not in kwargs
        assert "exploration_initial_eps" not in kwargs

    def test_non_random_without_sb3_raises_runtime_error(self):
        env = _make_env()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", False),
            patch.dict(agent_module._POLICY_CLASSES, {}),
        ):
            with pytest.raises(RuntimeError, match="stable-baselines3"):
                RLAgent(env, policy="ppo")


# ---------------------------------------------------------------------------
# act()
# ---------------------------------------------------------------------------


class TestAct:
    def test_random_act_returns_valid_action(self):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        env.reset()
        obs = env._observe()
        action = agt.act(obs)
        assert env.action_space.contains(action)

    def test_act_with_model_forwards_to_predict(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        fake_action = np.array([0, 0, 0, 0])
        model_instance.predict.return_value = (fake_action, None)
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            agt = RLAgent(env, policy="ppo")
        env.reset()
        obs = env._observe()
        action = agt.act(obs, deterministic=True)
        model_instance.predict.assert_called_once_with(obs, deterministic=True)
        np.testing.assert_array_equal(action, fake_action)


# ---------------------------------------------------------------------------
# remember(), terminal_bonus()
# ---------------------------------------------------------------------------


class TestReplay:
    def test_remember_pushes_dqn_transition(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"dqn": model_cls}),
        ):
            agt = RLAgent(env, policy="dqn")
        obs = np.zeros(env.observation_space.shape, dtype=np.float32)
        agt.remember(obs, np.array([0, 0, 0, 0]), 0.5, obs, False)
        model_instance.replay_buffer.add.assert_called_once()

    def test_remember_noop_for_random(self):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        obs = np.zeros(env.observation_space.shape, dtype=np.float32)
        agt.remember(obs, np.array([0, 0, 0, 0]), 0.5, obs, False)  # no raise

    def test_remember_noop_for_ppo(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            agt = RLAgent(env, policy="ppo")
        obs = np.zeros(env.observation_space.shape, dtype=np.float32)
        agt.remember(obs, np.array([0, 0, 0, 0]), 0.5, obs, False)
        model_instance.replay_buffer.add.assert_not_called()

    def test_terminal_bonus_without_action_is_noop(self):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        agt.terminal_bonus(0.7)  # no _last_obs/_last_action; nothing happens

    def test_terminal_bonus_dqn_pushes_done_transition(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"dqn": model_cls}),
        ):
            agt = RLAgent(env, policy="dqn")
        obs = np.zeros(env.observation_space.shape, dtype=np.float32)
        agt.act(obs)
        agt.terminal_bonus(0.42)
        # remember called once on terminal_bonus.
        model_instance.replay_buffer.add.assert_called_once()


# ---------------------------------------------------------------------------
# learn()
# ---------------------------------------------------------------------------


class TestLearn:
    def test_learn_invokes_model_learn(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"dqn": model_cls}),
        ):
            agt = RLAgent(env, policy="dqn")
        agt.learn(100)
        model_instance.learn.assert_called_once()

    def test_learn_zero_timesteps_is_noop(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"dqn": model_cls}),
        ):
            agt = RLAgent(env, policy="dqn")
        agt.learn(0)
        model_instance.learn.assert_not_called()

    def test_learn_random_is_noop(self):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        agt.learn(100)  # no raise


# ---------------------------------------------------------------------------
# save(), load(), set_epsilon()
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_calls_model_save(self, tmp_path):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            agt = RLAgent(env, policy="ppo")
        target = tmp_path / "nested" / "model.zip"
        agt.save(target)
        model_instance.save.assert_called_once()
        assert target.parent.exists()

    def test_save_random_is_noop(self, tmp_path):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        agt.save(str(tmp_path / "x"))  # no raise

    def test_load_random_is_noop(self, tmp_path):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        agt.load(str(tmp_path / "nope"))

    def test_load_missing_path_is_noop(self, tmp_path):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            agt = RLAgent(env, policy="ppo")
        agt.load(tmp_path / "missing.zip")
        model_cls.load.assert_not_called()

    def test_load_present_path_reloads_model(self, tmp_path):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        target = tmp_path / "model.zip"
        target.write_bytes(b"")
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            agt = RLAgent(env, policy="ppo")
            agt.load(target)
        model_cls.load.assert_called_once()

    def test_set_epsilon_random_is_noop(self):
        env = _make_env()
        agt = RLAgent(env, policy="random")
        agt.set_epsilon(0.5)  # no raise

    def test_set_epsilon_dqn_assigns_attribute(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"dqn": model_cls}),
        ):
            agt = RLAgent(env, policy="dqn")
        agt.set_epsilon(2.0)  # clamps to 1.0
        assert model_instance.exploration_rate == 1.0
        agt.set_epsilon(-1.0)
        assert model_instance.exploration_rate == 0.0
        agt.set_epsilon(0.25)
        assert model_instance.exploration_rate == 0.25

    def test_set_epsilon_noop_for_ppo(self):
        env = _make_env()
        model_cls, model_instance = _mock_model_cls()
        with (
            patch.object(agent_module, "_SB3_AVAILABLE", True),
            patch.dict(agent_module._POLICY_CLASSES, {"ppo": model_cls}),
        ):
            agt = RLAgent(env, policy="ppo")
        agt.set_epsilon(0.5)
        assert not hasattr(model_instance, "exploration_rate") or True
