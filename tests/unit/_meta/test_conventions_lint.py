"""Static lint that forbids em-dashes, double backticks, and LLM jargon in tracked files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

LINT_PATHS = [
    "src/scripts/generate_healthtasks_ttl.py",
    "tests/unit/scenarios/test_healthtasks_ttl_regen.py",
    "src/scripts/scenarios/domain/context.py",
    "src/scripts/persona/export/timeline_writer.py",
    "tests/unit/scenarios/test_context_loader.py",
    "tests/unit/persona/test_iri_binding_invariant.py",
    "tests/unit/persona/test_timeline_writer.py",
    "src/scripts/scenarios/config/schema.py",
    "tests/unit/scenarios/test_observation_schema.py",
    "tests/unit/scenarios/test_observation_llm_agent.py",
    "tests/unit/scenarios/test_observation_greedy.py",
    "src/scripts/scenarios/metrics/context_fit.py",
    "tests/unit/scenarios/test_l_context_fit.py",
]

EM_DASH = chr(0x2014)
DOUBLE_BACKTICK_PATTERN = re.compile(r"(?<!`)``(?!`)")
BANNED_TOKENS = (
    "leverage",
    "harness",
    "showcase",
    "delve",
    "seamlessly",
    "robust",
    "comprehensive",
    "unleash",
    "empower",
)
_BANNED_WORD = re.compile(r"\b(" + "|".join(BANNED_TOKENS) + r")\b", re.IGNORECASE)

_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")


def _strip_code(text: str) -> str:
    """Drop fenced code blocks and inline single-backtick spans before scanning prose."""
    return _INLINE_CODE.sub("", _FENCED_BLOCK.sub("", text))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", LINT_PATHS)
def test_no_em_dashes(rel: str) -> None:
    prose = _strip_code(_read(rel))
    assert EM_DASH not in prose, (
        f"{rel}: em-dash found in prose; rewrite with parentheses, "
        "semicolons, hyphens or a sentence split."
    )


@pytest.mark.parametrize("rel", LINT_PATHS)
def test_no_double_backticks(rel: str) -> None:
    text = _read(rel)
    hits = list(DOUBLE_BACKTICK_PATTERN.finditer(text))
    assert not hits, (
        f"{rel}: double-backtick code spans at offsets "
        f"{[h.start() for h in hits[:5]]}; use single backticks."
    )


@pytest.mark.parametrize("rel", LINT_PATHS)
def test_no_banned_tokens(rel: str) -> None:
    prose = _strip_code(_read(rel))
    hits = sorted({m.group(0).lower() for m in _BANNED_WORD.finditer(prose)})
    assert not hits, f"{rel}: LLM-jargon banned tokens in prose: {hits}"
