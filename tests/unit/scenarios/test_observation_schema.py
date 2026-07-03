"""Tests for the new `ObservationConfig` schema and its wiring into AugmentationConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scripts.scenarios.config.schema import (
    _DEFAULT_TASK_FLAGS,
    _KNOWN_HOST_FLAGS,
    _KNOWN_TASK_FLAGS,
    AugmentationConfig,
    AugmentationMethodConfig,
    ObservationConfig,
)


def test_default_observation_reproduces_today_leak_surface() -> None:
    obs = ObservationConfig()
    assert obs.contexts == []
    assert obs.host_flags == []
    assert obs.context_detail == "summary"
    assert tuple(obs.task_flags) == _DEFAULT_TASK_FLAGS


def test_explicit_empty_lists_distinct_from_default_task_flags() -> None:
    """A scenario writer asking for `[]` opts out of every default field."""
    obs = ObservationConfig(task_flags=[])
    assert obs.task_flags == []
    assert tuple(ObservationConfig().task_flags) == _DEFAULT_TASK_FLAGS


def test_unknown_task_flag_rejected_with_clear_message() -> None:
    with pytest.raises(ValidationError, match="unknown fields"):
        ObservationConfig(task_flags=["mystery_field"])


def test_unknown_host_flag_rejected_with_clear_message() -> None:
    with pytest.raises(ValidationError, match="host_flags has unknown"):
        ObservationConfig(host_flags=["calorie_bonus"])


def test_unknown_context_category_rejected_with_clear_message() -> None:
    with pytest.raises(ValidationError, match="contexts has unknown categories"):
        ObservationConfig(contexts=["not_a_category"])


def test_known_context_categories_accepted() -> None:
    obs = ObservationConfig(contexts=["mood_emotion", "location"])
    assert "mood_emotion" in obs.contexts
    assert "location" in obs.contexts


def test_known_host_flags_accepted() -> None:
    for flag in sorted(_KNOWN_HOST_FLAGS):
        obs = ObservationConfig(host_flags=[flag])
        assert obs.host_flags == [flag]


def test_known_task_flags_accepted() -> None:
    for flag in sorted(_KNOWN_TASK_FLAGS):
        obs = ObservationConfig(task_flags=[flag])
        assert obs.task_flags == [flag]


def test_context_recommends_is_a_known_task_flag() -> None:
    """`context_recommends` must parse as a valid task flag."""
    assert "context_recommends" in _KNOWN_TASK_FLAGS


def test_context_recommends_in_default_task_flags() -> None:
    """`context_recommends` ships in the default set so new scenarios pick it up automatically."""
    assert "context_recommends" in _DEFAULT_TASK_FLAGS


def test_model_dump_round_trip_preserves_explicit_values() -> None:
    obs = ObservationConfig(
        contexts=["mood_emotion"],
        host_flags=["is_concurrent"],
        task_flags=["duration_min", "duration_max"],
        context_detail="full",
    )
    payload = obs.model_dump()
    restored = ObservationConfig.model_validate(payload)
    assert restored == obs


def test_augmentation_config_has_default_observation() -> None:
    cfg = AugmentationConfig()
    assert isinstance(cfg.observation, ObservationConfig)
    assert cfg.observation.contexts == []


def test_augmentation_method_config_has_default_observation() -> None:
    cfg = AugmentationMethodConfig()
    assert isinstance(cfg.observation, ObservationConfig)
    assert cfg.observation.contexts == []


def test_augmentation_method_config_accepts_explicit_observation_block() -> None:
    cfg = AugmentationMethodConfig.model_validate(
        {
            "method": "llm_agent",
            "observation": {
                "contexts": ["mood_emotion", "energy_state"],
                "host_flags": ["is_concurrent"],
                "task_flags": ["duration_min", "intensity"],
                "context_detail": "full",
            },
        }
    )
    assert cfg.observation.contexts == ["mood_emotion", "energy_state"]
    assert cfg.observation.context_detail == "full"
    assert cfg.observation.host_flags == ["is_concurrent"]
    assert cfg.observation.task_flags == ["duration_min", "intensity"]


def test_observation_block_rejects_unknown_top_level_field() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ObservationConfig.model_validate(
            {
                "contexts": [],
                "task_flags": [],
                "host_flags": [],
                "context_detail": "summary",
                "nonsense_field": True,
            }
        )


def test_invalid_context_detail_literal_rejected() -> None:
    with pytest.raises(ValidationError):
        ObservationConfig.model_validate({"context_detail": "verbose"})


def test_load_context_categories_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.scripts.scenarios.config import schema as s

    monkeypatch.setattr(
        s._load_context_categories, "_cache", frozenset({"a", "b"}), raising=False
    )
    first = s._load_context_categories()
    second = s._load_context_categories()
    assert first is second
    assert isinstance(first, frozenset)


def test_load_context_categories_queries_neo4j_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call after cache eviction opens a session and reads categories."""
    from unittest.mock import MagicMock

    from src.scripts.scenarios.config import schema as s

    if hasattr(s._load_context_categories, "_cache"):
        delattr(s._load_context_categories, "_cache")

    session = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = session
    cm.__exit__.return_value = False
    driver = MagicMock()
    settings = MagicMock()
    settings.database = "neo4j"

    monkeypatch.setattr("src.graphrag.config.Neo4jSettings.from_env", lambda: settings)
    monkeypatch.setattr("src.graphrag.neo4j_client.make_driver", lambda _s: driver)
    monkeypatch.setattr("src.graphrag.neo4j_client.session_scope", lambda d, db: cm)
    monkeypatch.setattr(
        "src.scripts.scenarios.task_generation.ontology_bridge.fetch_context_categories",
        lambda sess: frozenset({"mood_emotion", "stress"}),
    )

    cats = s._load_context_categories()
    assert cats == frozenset({"mood_emotion", "stress"})
    driver.close.assert_called_once()


def test_observation_with_contexts_when_no_known_categories_does_not_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the loader returns an empty set, the validator falls open."""
    from src.scripts.scenarios.config import schema as s

    monkeypatch.setattr(s, "_load_context_categories", lambda: frozenset())
    obs = ObservationConfig(contexts=["arbitrary_category"])
    assert obs.contexts == ["arbitrary_category"]
