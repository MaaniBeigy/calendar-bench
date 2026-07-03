"""
Configuration module for the GraphRAG pipeline.

This module handles environment variable loading and validation for Neo4j and LLM
providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get(name: str, default: str | None = None, *, required: bool = False) -> str:
    """ "Gets an environment variable with optional default and requirement check."""
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return value or ""


@dataclass(frozen=True)
class Neo4jSettings:
    """Data class for Neo4j connection parameters."""

    uri: str
    username: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        """Factory method to initialize Neo4jSettings from environment variables."""
        in_container = Path("/.dockerenv").exists() or os.getenv("RUNNING_IN_DOCKER")
        uri = (
            _get("NEO4J_URI")
            if in_container
            else _get("NEO4J_HOST_URI", _get("NEO4J_URI"))
        )
        return cls(
            uri=uri or "bolt://localhost:7687",
            username=_get("NEO4J_USERNAME", "neo4j"),
            password=_get("NEO4J_PASSWORD", required=True),
            database=_get("NEO4J_DATABASE", "neo4j"),
        )


@dataclass(frozen=True)
class LLMSettings:
    """
    Data container for Language Model provider configurations.

    Stores API keys and model identifiers for various supported providers
    (OpenAI, Anthropic, OpenRouter). This allows the LLM factory to switch providers
    seamlessly based on the 'LLM_PROVIDER' environment setting.
    """

    provider: str
    openai_api_key: str
    openai_model: str
    anthropic_api_key: str
    anthropic_model: str
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Factory method to initialize LLMSettings from environment variables."""
        return cls(
            provider=_get("LLM_PROVIDER", "openai").lower(),
            openai_api_key=_get("OPENAI_API_KEY"),
            openai_model=_get("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_api_key=_get("ANTHROPIC_API_KEY"),
            anthropic_model=_get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            openrouter_api_key=_get("OPENROUTER_API_KEY"),
            openrouter_base_url=_get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            openrouter_model=_get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6"),
        )
