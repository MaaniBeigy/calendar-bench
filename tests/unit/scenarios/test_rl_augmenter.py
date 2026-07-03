"""Tests for the online DQN RLAugmenter."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from src.scripts.scenarios.augmentation.rl.augmenter import (
    RLAugmenter,
    _components_to_gains,
)
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    LossWeights,
    ObservationConfig,
    RLConfig,
)
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 6, 1)


def _cfg(
    *,
    policy: str = "random",
    repeat_per_week: bool = True,
    checkpoint: str | None = None,
    contexts: list[str] | None = None,
) -> AugmentationConfig:
    return AugmentationConfig(
        method="rl",
        repeat_per_week=repeat_per_week,
        rl=RLConfig(
            policy=policy,
            checkpoint=checkpoint,
            train_steps_per_week=1,
            seed=7,
            epsilon_start=0.5,
            epsilon_end=0.05,
            epsilon_decay_weeks=2,
        ),
        observation=ObservationConfig(contexts=contexts or []),
    )


def _augmenter(seed: int = 0, **kwargs) -> RLAugmenter:
    return RLAugmenter(seed=seed, **kwargs)


def test_empty_horizon_returns_unscheduled():
    aug = _augmenter()
    trace = CalendarTrace(person_id="p1")
    task = make_task(duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg())
    assert sol.scheduled == []
    assert sol.unscheduled == [task]
    assert sol.weekly_records == []


def test_no_tasks_returns_empty_solution():
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    sol = aug.augment(trace, [], _cfg())
    assert sol.scheduled == []
    assert sol.weekly_records == []


def test_random_policy_runs_one_week_per_chunk():
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg(), horizon=(DATE, 14))
    assert len(sol.weekly_records) == 2
    assert sol.weekly_records[0]["week_index"] == 1
    assert sol.weekly_records[1]["week_index"] == 2


def test_repeat_per_week_false_runs_one_week_only():
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg(repeat_per_week=False), horizon=(DATE, 14))
    assert len(sol.weekly_records) == 1


def test_observation_contexts_propagated_to_env(monkeypatch):
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    captured: dict = {}
    real_build = aug._build_env

    def _record(*args, **kwargs):
        env = real_build(*args, **kwargs)
        captured["categories"] = env.unwrapped._visible_context_categories
        return env

    monkeypatch.setattr(aug, "_build_env", _record)
    aug.augment(trace, [task], _cfg(contexts=["mood_emotion"]), horizon=(DATE, 7))
    assert captured["categories"] == ["mood_emotion"]


def test_epsilon_schedule_decays_per_week():
    aug = _augmenter()
    rl_cfg = RLConfig(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_weeks=4)
    assert aug._epsilon_for_week(0, rl_cfg) == pytest.approx(1.0)
    assert aug._epsilon_for_week(2, rl_cfg) == pytest.approx(0.5)
    assert aug._epsilon_for_week(4, rl_cfg) == pytest.approx(0.0)
    assert aug._epsilon_for_week(10, rl_cfg) == pytest.approx(0.0)


def test_weekly_records_carry_gain_components():
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg(), horizon=(DATE, 7))
    assert sol.weekly_records, "expected at least one weekly record"
    row = sol.weekly_records[0]
    assert "weighted_gain" in row
    assert "gains" in row
    assert "recommended_task_coverage" in row["gains"]


def test_solution_from_collects_carried_placements():
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg(), horizon=(DATE, 7))
    assert sol.tasks == [task]
    assert sol.augmented_calendar.person_id == "p1"


def test_components_to_gains_handles_none():
    class _Comp:
        cov = 0.1
        cal = 0.2
        pref = None
        disp = 0.4
        merge = None
        spread = 0.0
        divide = None
        context_fit = None

    gains = _components_to_gains(_Comp())
    assert gains["recommended_task_coverage"] == pytest.approx(0.9)
    assert gains["user_preference_deviation"] is None


def test_checkpoint_load_missing_path_is_noop(tmp_path: Path):
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    sol = aug.augment(
        trace,
        [task],
        _cfg(checkpoint=str(tmp_path / "missing.zip")),
        horizon=(DATE, 7),
    )
    assert sol.weekly_records


def test_loss_weights_default_when_none():
    aug = RLAugmenter(loss_weights=None)
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg(), horizon=(DATE, 7))
    assert sol.weekly_records


def test_persona_constraints_factory_invoked():
    seen: dict = {}

    def constraints_for(trace):
        seen["pid"] = trace.person_id
        return None

    def window_map_for(trace):
        seen["window"] = trace.person_id
        return None

    aug = RLAugmenter(
        persona_constraints_for=constraints_for,
        window_map_for=window_map_for,
    )
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="abc", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    aug.augment(trace, [task], _cfg(), horizon=(DATE, 7))
    assert seen == {"pid": "abc", "window": "abc"}


def test_empty_horizon_carries_weekly_records_attribute():
    aug = _augmenter()
    trace = CalendarTrace(person_id="p1")
    task = make_task(duration_min=30, duration_max=30)
    sol = aug.augment(trace, [task], _cfg(), horizon=(DATE, 0))
    assert hasattr(sol, "weekly_records")
    assert sol.weekly_records == []


def test_safe_epsilon_reads_dqn_exploration_rate():
    from src.scripts.scenarios.augmentation.rl.augmenter import _safe_epsilon

    class _Model:
        exploration_rate = 0.42

    class _Agent:
        model = _Model()

    assert _safe_epsilon(_Agent()) == 0.42


def test_safe_epsilon_returns_none_when_no_rate_attr():
    from src.scripts.scenarios.augmentation.rl.augmenter import _safe_epsilon

    class _Model:
        pass

    class _Agent:
        model = _Model()

    assert _safe_epsilon(_Agent()) is None


def test_mask_to_valid_action_returns_action_when_mask_missing():
    aug = _augmenter()

    class _Inner:
        pass

    class _Wrap:
        env = _Inner()

    assert aug._mask_to_valid_action(_Wrap(), 7) == 7


def test_mask_to_valid_action_swallows_mask_exception():
    aug = _augmenter()

    class _Inner:
        def valid_action_mask(self):
            raise RuntimeError("boom")

    class _Wrap:
        env = _Inner()

    assert aug._mask_to_valid_action(_Wrap(), 3) == 3


def test_mask_to_valid_action_returns_action_when_mask_all_false():
    aug = _augmenter()
    import numpy as np

    class _Inner:
        def valid_action_mask(self):
            return np.zeros(5, dtype=bool)

    class _Wrap:
        env = _Inner()

    assert aug._mask_to_valid_action(_Wrap(), 2) == 2


def test_mask_to_valid_action_resamples_when_picked_action_invalid():
    aug = _augmenter()
    import numpy as np

    class _Inner:
        def valid_action_mask(self):
            mask = np.zeros(8, dtype=bool)
            mask[5] = True
            return mask

    class _Wrap:
        env = _Inner()

    assert aug._mask_to_valid_action(_Wrap(), 1) == 5


def test_checkpoint_write_creates_parent(tmp_path: Path, monkeypatch):
    target = tmp_path / "nested" / "dqn.zip"
    aug = _augmenter()
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    aug.augment(
        trace,
        [task],
        _cfg(checkpoint=str(target)),
        horizon=(DATE, 7),
    )
    # Random policy save is a no-op, but the parent directory should exist.
    assert target.parent.exists()
