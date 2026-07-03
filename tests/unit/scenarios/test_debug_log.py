"""Unit tests for src.scripts.scenarios.task_generation.debug_log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.task_generation.debug_log import (
    STAGE_PARAPHRASE,
    STAGE_TASK_GEN,
    _resolve_debug_dir,
    write_debug_log,
    write_proposed_tasks_log,
)


class TestResolveDebugDir:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("LLM_DEBUG_DIR", raising=False)
        monkeypatch.delenv("LLM_AGENT_DEBUG_DIR", raising=False)
        assert _resolve_debug_dir() is None

    def test_prefers_new_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path / "new"))
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path / "old"))
        assert _resolve_debug_dir() == str(tmp_path / "new")

    def test_falls_back_to_legacy_env_var(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_DEBUG_DIR", raising=False)
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path / "old"))
        assert _resolve_debug_dir() == str(tmp_path / "old")


class TestWriteDebugLog:
    def test_no_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("LLM_DEBUG_DIR", raising=False)
        monkeypatch.delenv("LLM_AGENT_DEBUG_DIR", raising=False)
        result = write_debug_log(
            "p001",
            STAGE_TASK_GEN,
            attempt=0,
            prompt="prompt body",
            raw_response="response body",
        )
        assert result is None

    def test_writes_file_when_env_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path))
        out = write_debug_log(
            "p001",
            STAGE_TASK_GEN,
            attempt=0,
            prompt="hello prompt",
            raw_response='[{"label": "yoga"}]',
            summary_lines=["proposed=3", "accepted=2"],
        )
        assert out is not None
        text = out.read_text(encoding="utf-8")
        assert "=== person ===\np001" in text
        assert "proposed=3" in text
        assert "accepted=2" in text
        assert "hello prompt" in text
        assert '"label": "yoga"' in text

    def test_filename_encodes_stage_and_attempt(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path))
        out = write_debug_log(
            "a_fulltime_0000",
            STAGE_PARAPHRASE,
            attempt=2,
            prompt="x",
            raw_response="y",
        )
        assert out is not None
        assert out.name == "paraphrase__attempt2.txt"
        assert out.parent.name == "a_fulltime_0000"

    def test_no_summary_lines_renders_placeholder(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path))
        out = write_debug_log(
            "p001",
            STAGE_TASK_GEN,
            attempt=0,
            prompt="p",
            raw_response="r",
        )
        assert out is not None
        text = out.read_text(encoding="utf-8")
        assert "(no summary)" in text

    def test_unwritable_dir_returns_none(self, monkeypatch, tmp_path):
        # Point at an existing FILE so mkdir() raises NotADirectoryError.
        bad = tmp_path / "not_a_dir"
        bad.write_text("blocker", encoding="utf-8")
        monkeypatch.setenv("LLM_DEBUG_DIR", str(bad))
        result = write_debug_log(
            "p001",
            STAGE_TASK_GEN,
            attempt=0,
            prompt="x",
            raw_response="y",
        )
        assert result is None

    @pytest.mark.parametrize(
        "stage_value",
        [STAGE_TASK_GEN, STAGE_PARAPHRASE, "custom_stage"],
    )
    def test_arbitrary_stage_label_lands_in_filename(
        self, monkeypatch, tmp_path, stage_value
    ):
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path))
        out = write_debug_log(
            "p001",
            stage_value,
            attempt=1,
            prompt="x",
            raw_response="y",
        )
        assert out is not None
        assert out.name == f"{stage_value}__attempt1.txt"


class TestWriteProposedTasksLog:
    """`write_proposed_tasks_log` writes one JSON file per persona
    summarising every URI the LLM proposed across all attempts -
    intended for human grep'ing when the LLM is misbehaving."""

    def test_no_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("LLM_DEBUG_DIR", raising=False)
        monkeypatch.delenv("LLM_AGENT_DEBUG_DIR", raising=False)
        out = write_proposed_tasks_log(
            "p001",
            scenario_id="x",
            accepted=[],
            proposals=[],
        )
        assert out is None

    def test_writes_file_with_payload(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path))
        accepted = [
            {
                "label": "yoga",
                "display_name": "Yoga 🤸",
                "ontology_uri": "https://ex.org/task/yoga",
            }
        ]
        proposals = [
            {"attempt": 1, "uri": "https://ex.org/task/yoga", "verdict": "accepted"},
            {
                "attempt": 1,
                "uri": "https://ex.org/health/Foo",
                "verdict": "wrong_branch",
            },
            {"attempt": 2, "uri": "https://ex.org/task/dup", "verdict": "duplicate"},
        ]
        out = write_proposed_tasks_log(
            "p001",
            scenario_id="nutrition_l1",
            accepted=accepted,
            proposals=proposals,
        )
        assert out is not None
        assert out.name == "tasks_proposed.json"
        assert out.parent.name == "p001"
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["person_id"] == "p001"
        assert loaded["scenario_id"] == "nutrition_l1"
        assert loaded["accepted_count"] == 1
        assert loaded["proposed_count"] == 3
        # Verdict counts are sorted alphabetically.
        assert loaded["verdict_counts"] == {
            "accepted": 1,
            "duplicate": 1,
            "wrong_branch": 1,
        }
        assert loaded["accepted"][0]["display_name"] == "Yoga 🤸"
        assert loaded["proposed"][0]["verdict"] == "accepted"

    def test_emoji_round_trip(self, monkeypatch, tmp_path):
        """`ensure_ascii=False` keeps emojis readable in the JSON file."""
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path))
        out = write_proposed_tasks_log(
            "p001",
            scenario_id="x",
            accepted=[
                {
                    "label": "smoothie",
                    "display_name": "Blend a Smoothie 🥤",
                    "ontology_uri": "https://ex.org/task/smoothie",
                }
            ],
            proposals=[],
        )
        assert out is not None
        text = out.read_text(encoding="utf-8")
        assert "🥤" in text  # emoji preserved verbatim

    def test_unwritable_returns_none(self, monkeypatch, tmp_path):
        bad = tmp_path / "blocker"
        bad.write_text("blocker", encoding="utf-8")
        monkeypatch.setenv("LLM_DEBUG_DIR", str(bad))
        out = write_proposed_tasks_log(
            "p001", scenario_id="x", accepted=[], proposals=[]
        )
        assert out is None

    def test_proposals_iterable_consumed_once(self, monkeypatch, tmp_path):
        """An iterable (not a list) is consumed exactly once and
        produces a correct verdict-rollup."""
        monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path))

        def gen():
            yield {"attempt": 1, "verdict": "accepted"}
            yield {"attempt": 1, "verdict": "wrong_branch"}

        out = write_proposed_tasks_log(
            "p001", scenario_id="x", accepted=[], proposals=gen()
        )
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["proposed_count"] == 2
        assert loaded["verdict_counts"] == {"accepted": 1, "wrong_branch": 1}
