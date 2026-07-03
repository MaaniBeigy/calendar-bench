"""Shared fixtures for end-to-end persona integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "scripts"
    / "persona"
    / "config"
    / "examples"
)


@pytest.fixture
def environment_yaml() -> Path:
    return EXAMPLES_DIR / "environment.yaml"


@pytest.fixture
def persona_yaml() -> Path:
    return EXAMPLES_DIR / "persona_config.yaml"


@pytest.fixture
def event_yaml() -> Path:
    return EXAMPLES_DIR / "event_config.yaml"


@pytest.fixture
def rules_yaml() -> Path:
    return EXAMPLES_DIR / "temporal_relation_rules.yaml"
