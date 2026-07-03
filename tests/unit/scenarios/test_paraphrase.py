"""Unit tests for src.scripts.scenarios.task_generation.paraphrase."""

from __future__ import annotations

import json
import logging

import pytest

from src.scripts.scenarios.task_generation.gates import GateDecision
from src.scripts.scenarios.task_generation.paraphrase import (
    ParaphraseResult,
    ParaphraseStats,
    apply_gates,
    paraphrase_for_persona,
)

# ---------------------------------------------------------------------------
# Stage 2; paraphrase_for_persona
# ---------------------------------------------------------------------------


def _verified_tasks() -> list[dict]:
    return [
        {
            "label": "drink-a-glass-of-water",
            "display_name": "Drink A Glass of Water 🥤",
            "description": "Make a water drinking ritual.",
        },
        {
            "label": "stretch-for-5-minutes",
            "display_name": "Stretch for 5 Minutes 🤸",
            "description": "Stretch your shoulders, back, legs, or neck.",
        },
    ]


def _ok_invoker():
    """Return an invoker that produces a well-formed paraphrase JSON."""

    def invoke(_prompt: str) -> str:
        return json.dumps(
            [
                {
                    "label": "drink-a-glass-of-water",
                    "personalized_description": "Sip a glass of water.",
                },
                {
                    "label": "stretch-for-5-minutes",
                    "personalized_description": "Stretch your shoulders gently.",
                },
            ]
        )

    return invoke


class TestParaphraseForPersona:
    def test_happy_path_returns_one_paraphrase_per_task(self):
        result = paraphrase_for_persona(
            invoke=_ok_invoker(),
            profile_summary="occupation_status=fulltime",
            verified_tasks=_verified_tasks(),
            length_delta_pct=0.10,
        )
        assert isinstance(result, ParaphraseResult)
        assert len(result.paraphrases) == 2
        assert result.paraphrases[0] == "Sip a glass of water."
        assert result.paraphrases[1] == "Stretch your shoulders gently."
        assert result.stats.parse_failed is False
        assert result.stats.attempts == 1

    def test_prompt_carries_length_delta_pct_and_profile(self):
        captured = {}

        def invoke(prompt: str) -> str:
            captured["prompt"] = prompt
            return _ok_invoker()(prompt)

        paraphrase_for_persona(
            invoke=invoke,
            profile_summary="occupation_status=parttime",
            verified_tasks=_verified_tasks(),
            length_delta_pct=0.15,
        )
        assert "occupation_status=parttime" in captured["prompt"]
        # 0.15 to "±15 %" rendered as integer percent.
        assert "±15%" in captured["prompt"]

    def test_invoker_returning_none_returns_canonical(self):
        def invoke(_prompt: str) -> str:
            return None  # type: ignore[return-value]

        result = paraphrase_for_persona(
            invoke=invoke,
            profile_summary="x",
            verified_tasks=_verified_tasks(),
            length_delta_pct=0.10,
        )
        # Empty-string parser to empty list to all canonical fallbacks.
        assert result.paraphrases == [
            "Make a water drinking ritual.",
            "Stretch your shoulders, back, legs, or neck.",
        ]

    def test_unparseable_response_falls_back_to_canonical(self, caplog):
        def invoke(_prompt: str) -> str:
            return "not json at all {{"

        with caplog.at_level(logging.WARNING):
            result = paraphrase_for_persona(
                invoke=invoke,
                profile_summary="x",
                verified_tasks=_verified_tasks(),
                length_delta_pct=0.10,
            )
        assert result.stats.parse_failed is True
        assert result.paraphrases == [
            "Make a water drinking ritual.",
            "Stretch your shoulders, back, legs, or neck.",
        ]
        assert any("not parseable" in r.getMessage() for r in caplog.records)

    def test_missing_label_in_response_uses_canonical(self, caplog):
        def invoke(_prompt: str) -> str:
            return json.dumps(
                [
                    {
                        "label": "drink-a-glass-of-water",
                        "personalized_description": "Sip a glass of water.",
                    }
                    # second task entry omitted
                ]
            )

        with caplog.at_level(logging.INFO):
            result = paraphrase_for_persona(
                invoke=invoke,
                profile_summary="x",
                verified_tasks=_verified_tasks(),
                length_delta_pct=0.10,
            )
        assert result.paraphrases[0] == "Sip a glass of water."
        # Second entry falls back to canonical.
        assert result.paraphrases[1] == "Stretch your shoulders, back, legs, or neck."
        assert any("matched 1/2 tasks" in r.getMessage() for r in caplog.records)

    def test_blank_personalized_description_falls_back(self):
        def invoke(_prompt: str) -> str:
            return json.dumps(
                [
                    {
                        "label": "drink-a-glass-of-water",
                        "personalized_description": "   ",
                    },
                    {
                        "label": "stretch-for-5-minutes",
                        "personalized_description": "Stretch gently.",
                    },
                ]
            )

        result = paraphrase_for_persona(
            invoke=invoke,
            profile_summary="x",
            verified_tasks=_verified_tasks(),
            length_delta_pct=0.10,
        )
        assert result.paraphrases[0] == "Make a water drinking ritual."
        assert result.paraphrases[1] == "Stretch gently."


# ---------------------------------------------------------------------------
# Stage 3; apply_gates
# ---------------------------------------------------------------------------


class _ConstantEmbedder:
    """Returns the same vector to cosine = 1.0; gate C always passes."""

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0]


class _OrthoEmbedder:
    """Alternating axes to cosine = 0.0; gate C always fails (when
    threshold > 0)."""

    def __init__(self) -> None:
        self._toggle = False

    def embed_query(self, _text: str) -> list[float]:
        self._toggle = not self._toggle
        return [1.0, 0.0] if self._toggle else [0.0, 1.0]


class TestApplyGates:
    def test_returns_one_decision_per_task(self):
        decisions = apply_gates(
            verified_tasks=_verified_tasks(),
            paraphrases=[
                "Make a water drinking ritual.",  # identical to all gates pass
                "Stretch your shoulders, back, legs, or neck.",
            ],
            embedder=_ConstantEmbedder(),
            length_delta_pct=0.10,
            similarity_threshold=0.80,
        )
        assert len(decisions) == 2
        assert all(isinstance(d, GateDecision) for d in decisions)
        assert all(d.used == "paraphrase" for d in decisions)

    def test_failed_decisions_logged_and_canonical_used(self, caplog):
        with caplog.at_level(logging.INFO):
            decisions = apply_gates(
                verified_tasks=_verified_tasks(),
                paraphrases=["", ""],  # empty to length gate fails
                embedder=_ConstantEmbedder(),
                length_delta_pct=0.10,
                similarity_threshold=0.80,
            )
        assert all(d.used == "canonical" for d in decisions)
        assert all(d.failed_gate == "length" for d in decisions)
        # Two INFO log lines, one per failed task.
        infos = [r for r in caplog.records if "paraphrase rejected" in r.getMessage()]
        assert len(infos) == 2

    def test_task_without_display_name_uses_description_as_anchor(self):
        verified = [
            {
                "label": "no-name",
                "description": "Walk every day.",
            }
        ]
        decisions = apply_gates(
            verified_tasks=verified,
            paraphrases=["Walk every day."],
            embedder=_ConstantEmbedder(),
            length_delta_pct=0.10,
            similarity_threshold=0.80,
        )
        assert decisions[0].used == "paraphrase"

    def test_paraphrase_iterable_consumed(self):
        """`apply_gates` accepts any iterable, not just a list."""

        def gen():
            yield "Make a water drinking ritual."
            yield "Stretch your shoulders, back, legs, or neck."

        decisions = apply_gates(
            verified_tasks=_verified_tasks(),
            paraphrases=gen(),
            embedder=_ConstantEmbedder(),
            length_delta_pct=0.10,
            similarity_threshold=0.80,
        )
        assert len(decisions) == 2


# ---------------------------------------------------------------------------
# Dataclass plumbing
# ---------------------------------------------------------------------------


def test_paraphrase_stats_defaults():
    s = ParaphraseStats()
    assert s.attempts == 1
    assert s.parse_failed is False
    assert s.wall_time_seconds == 0.0


def test_paraphrase_result_defaults():
    r = ParaphraseResult()
    assert r.paraphrases == []
    assert isinstance(r.stats, ParaphraseStats)
    assert r.raw_prompt == r.raw_response == ""
