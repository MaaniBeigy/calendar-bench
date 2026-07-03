"""Unit tests for src.graphrag.config: env loading + settings classes."""

from __future__ import annotations

import pytest

from src.graphrag.config import LLMSettings, Neo4jSettings, _get


def test_get_returns_default_when_missing(monkeypatch):
    monkeypatch.delenv("CB_TEST_VAR", raising=False)
    assert _get("CB_TEST_VAR", "fallback") == "fallback"


def test_get_returns_empty_when_missing_and_no_default(monkeypatch):
    monkeypatch.delenv("CB_TEST_VAR", raising=False)
    assert _get("CB_TEST_VAR") == ""


def test_get_required_raises_when_missing(monkeypatch):
    monkeypatch.delenv("CB_TEST_VAR", raising=False)
    with pytest.raises(RuntimeError, match="CB_TEST_VAR"):
        _get("CB_TEST_VAR", required=True)


def test_get_reads_existing_value(monkeypatch):
    monkeypatch.setenv("CB_TEST_VAR", "hello")
    assert _get("CB_TEST_VAR") == "hello"


def test_neo4j_settings_from_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://override:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    s = Neo4jSettings.from_env()
    assert s.username == "neo4j"
    assert s.password == "secret"
    assert s.database == "neo4j"
    assert s.uri.startswith("bolt://")


def test_neo4j_settings_password_required(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        Neo4jSettings.from_env()


def test_llm_settings_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    s = LLMSettings.from_env()
    assert s.provider == "openai"
    assert s.openai_model == "gpt-4o-mini"
    assert s.anthropic_model == "claude-sonnet-4-6"
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_llm_provider_lowercased(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    s = LLMSettings.from_env()
    assert s.provider == "openai"
