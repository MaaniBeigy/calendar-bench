"""Schema validation tests for `PTimeConfig` and the `ptime` augmentation method."""

from __future__ import annotations

import pytest

from src.scripts.scenarios.config.schema import (
    _PTIME_CRITERIA,
    AugmentationConfig,
    AugmentationMethodConfig,
    PTimeConfig,
)


def test_default_ptime_config_is_empty():
    cfg = PTimeConfig()
    assert cfg.candidate_strategy == "windows"
    assert cfg.duration_preference == "minimum"
    assert cfg.importance == {}
    assert cfg.interaction == {}


def test_ptime_config_accepts_known_criteria():
    cfg = PTimeConfig(importance={c: 0.5 for c in _PTIME_CRITERIA})
    assert cfg.importance == {c: 0.5 for c in _PTIME_CRITERIA}


def test_ptime_config_rejects_unknown_importance_key():
    with pytest.raises(ValueError, match="unknown criterion"):
        PTimeConfig(importance={"not_a_criterion": 0.5})


def test_ptime_config_rejects_out_of_range_importance():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PTimeConfig(importance={"time": 1.5})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PTimeConfig(importance={"time": -0.1})


def test_ptime_config_accepts_pair_interaction_keys():
    cfg = PTimeConfig(interaction={"duration|overlap": 0.5, "duration|time": -0.4})
    assert cfg.interaction["duration|overlap"] == 0.5
    assert cfg.interaction["duration|time"] == -0.4


def test_ptime_config_rejects_unsorted_pair_key():
    with pytest.raises(ValueError, match="unknown pair"):
        PTimeConfig(interaction={"overlap|duration": 0.4})


def test_ptime_config_rejects_self_pair_key():
    with pytest.raises(ValueError, match="unknown pair"):
        PTimeConfig(interaction={"time|time": 0.4})


def test_ptime_config_rejects_unknown_pair_member():
    with pytest.raises(ValueError, match="unknown pair"):
        PTimeConfig(interaction={"duration|unknown": 0.4})


def test_ptime_config_rejects_out_of_range_interaction():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        PTimeConfig(interaction={"duration|time": 1.5})


def test_augmentation_config_accepts_ptime_method():
    cfg = AugmentationConfig(method="ptime")
    assert cfg.method == "ptime"
    assert isinstance(cfg.ptime, PTimeConfig)


def test_augmentation_method_config_accepts_ptime_method():
    cfg = AugmentationMethodConfig(method="ptime")
    assert cfg.method == "ptime"
    assert isinstance(cfg.ptime, PTimeConfig)


def test_strategy_literal_rejects_unknown():
    with pytest.raises(ValueError):
        PTimeConfig(candidate_strategy="nope")


def test_duration_preference_literal_rejects_unknown():
    with pytest.raises(ValueError):
        PTimeConfig(duration_preference="random")


def test_solver_literal_rejects_unknown():
    with pytest.raises(ValueError):
        PTimeConfig(solver="bnb")


def test_solver_defaults_to_mcs():
    assert PTimeConfig().solver == "mcs"


def test_mcs_time_budget_seconds_must_be_positive():
    with pytest.raises(ValueError):
        PTimeConfig(mcs_time_budget_seconds=0.0)


def test_ptime_config_only_accepts_four_criteria():
    cfg = PTimeConfig(importance={c: 0.25 for c in _PTIME_CRITERIA})
    assert set(cfg.importance) == {"time", "duration", "overlap", "stability"}
