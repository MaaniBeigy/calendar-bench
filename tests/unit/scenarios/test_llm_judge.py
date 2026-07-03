"""Unit tests for src.scripts.scenarios.metrics.llm_judge.

Coverage targets:
  LLMJudgeOracle.score: LLM call, caching, symmetric cache key, malformed JSON,
                        partial JSON, value clamping.
  LLMJudgeOracle.score_all_pairs: all combinations, cache usage.
  LLMJudgeOracle._build_task_text: display+description, display only, fallback.
  LLMJudgeOracle._parse_score: valid JSON, partial JSON, invalid to 0.0.
  Cache persistence: writes to JSONL file, loads from JSONL file, skips corrupt lines.

All tests use unittest.mock; no real LLM API calls are made.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from src.scripts.scenarios.metrics.llm_judge import LLMJudgeOracle
from tests.unit.scenarios.conftest import make_event, make_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(response_content: str = '{"score": 0.8, "reason": "ok"}') -> MagicMock:
    """Build a mock OpenAI-compatible client that returns *response_content*."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = response_content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


def _make_oracle(
    client: MagicMock | None = None,
    cache_path: Path | None = None,
    model: str = "gpt-4o-mini",
) -> LLMJudgeOracle:
    return LLMJudgeOracle(
        client or _make_client(),
        model=model,
        cache_path=cache_path,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_calls_llm_and_returns_parsed_result(self):
        client = _make_client('{"score": 0.75, "reason": "good fit"}')
        oracle = _make_oracle(client)
        result = oracle.score("yoga", "gym_session")
        assert result == pytest.approx(0.75)
        client.chat.completions.create.assert_called_once()

    def test_score_caches_result(self):
        client = _make_client('{"score": 0.6, "reason": "ok"}')
        oracle = _make_oracle(client)
        # First call to LLM invoked
        r1 = oracle.score("yoga", "lunch")
        # Second call to cache hit
        r2 = oracle.score("yoga", "lunch")
        assert r1 == pytest.approx(0.6)
        assert r2 == pytest.approx(0.6)
        # LLM called only once
        assert client.chat.completions.create.call_count == 1

    def test_score_symmetric_cache_key(self):
        """(A, B) and (B, A) share the same frozenset cache key."""
        client = _make_client('{"score": 0.7, "reason": "ok"}')
        oracle = _make_oracle(client)
        oracle.score("yoga", "gym")
        # Calling with reversed labels should hit cache without LLM call
        oracle.score("gym", "yoga")
        assert client.chat.completions.create.call_count == 1

    def test_score_malformed_json_returns_zero(self):
        client = _make_client("not json at all")
        oracle = _make_oracle(client)
        result = oracle.score("yoga", "dinner")
        assert result == pytest.approx(0.0)

    def test_score_partial_json_extracts_score(self):
        """Regex fallback: raw text contains 'score': 0.55 but is otherwise bad JSON."""
        client = _make_client('some preamble "score": 0.55 trailing text')
        oracle = _make_oracle(client)
        result = oracle.score("yoga", "lunch")
        assert result == pytest.approx(0.55)

    def test_score_clamps_to_unit_interval_high(self):
        """LLM returns 1.5 to clamped to 1.0."""
        client = _make_client('{"score": 1.5, "reason": "high"}')
        oracle = _make_oracle(client)
        result = oracle.score("yoga", "gym")
        assert result == pytest.approx(1.0)

    def test_score_clamps_to_unit_interval_low(self):
        """LLM returns -0.1 to clamped to 0.0."""
        client = _make_client('{"score": -0.1, "reason": "negative"}')
        oracle = _make_oracle(client)
        result = oracle.score("yoga", "sleep")
        assert result == pytest.approx(0.0)

    def test_score_uses_task_display_and_description_in_prompt(self):
        """score() with task_display and task_description passes them to the prompt."""
        client = _make_client('{"score": 0.9, "reason": "great"}')
        oracle = _make_oracle(client)
        oracle.score(
            "mindful_eating",
            "lunch",
            task_display="Mindful Eating",
            task_description="Eat slowly and mindfully.",
        )
        prompt_arg = client.chat.completions.create.call_args[1]["messages"][0][
            "content"
        ]
        assert "Mindful Eating" in prompt_arg
        assert "Eat slowly and mindfully." in prompt_arg

    def test_score_uses_display_only_when_no_description(self):
        client = _make_client('{"score": 0.5, "reason": "ok"}')
        oracle = _make_oracle(client)
        oracle.score("yoga", "gym", task_display="Yoga Session")
        prompt_arg = client.chat.completions.create.call_args[1]["messages"][0][
            "content"
        ]
        assert "Yoga Session" in prompt_arg

    def test_score_falls_back_to_label_when_no_display(self):
        client = _make_client('{"score": 0.4, "reason": "ok"}')
        oracle = _make_oracle(client)
        oracle.score("yoga_practice", "gym")
        prompt_arg = client.chat.completions.create.call_args[1]["messages"][0][
            "content"
        ]
        assert "yoga practice" in prompt_arg


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


class TestCachePersistence:
    def test_cache_persists_to_file(self, tmp_path):
        cache_file = tmp_path / "cache.jsonl"
        client = _make_client('{"score": 0.8, "reason": "good"}')
        oracle = _make_oracle(client, cache_path=cache_file)
        oracle.score("yoga", "gym")
        assert cache_file.exists()
        lines = [
            l for l in cache_file.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["task_label"] == "yoga"
        assert entry["event_label"] == "gym"
        assert entry["score"] == pytest.approx(0.8)
        assert "ts" in entry
        assert "model" in entry

    def test_cache_loaded_from_file(self, tmp_path):
        """Existing cache JSONL to score returned without LLM call."""
        cache_file = tmp_path / "cache.jsonl"
        # Pre-populate cache file
        entry = json.dumps(
            {
                "task_label": "yoga",
                "event_label": "gym",
                "score": 0.7,
                "model": "gpt-4o-mini",
                "ts": "2026-01-01T00:00:00+00:00",
            }
        )
        cache_file.write_text(entry + "\n", encoding="utf-8")

        client = _make_client()
        oracle = _make_oracle(client, cache_path=cache_file)
        result = oracle.score("yoga", "gym")
        assert result == pytest.approx(0.7)
        # No LLM call needed
        client.chat.completions.create.assert_not_called()

    def test_cache_skips_malformed_lines(self, tmp_path):
        """Corrupt JSONL lines are silently skipped; valid lines are loaded."""
        cache_file = tmp_path / "cache.jsonl"
        good_entry = json.dumps(
            {
                "task_label": "yoga",
                "event_label": "gym",
                "score": 0.9,
                "model": "gpt-4o-mini",
                "ts": "2026-01-01T00:00:00+00:00",
            }
        )
        cache_file.write_text(
            "not json\n" + good_entry + "\n{missing_fields}\n",
            encoding="utf-8",
        )
        client = _make_client()
        oracle = _make_oracle(client, cache_path=cache_file)
        # The valid entry was loaded
        result = oracle.score("yoga", "gym")
        assert result == pytest.approx(0.9)
        client.chat.completions.create.assert_not_called()

    def test_no_file_write_when_cache_path_is_none(self):
        """Without cache_path, no file I/O occurs."""
        client = _make_client('{"score": 0.5, "reason": "ok"}')
        oracle = _make_oracle(client, cache_path=None)
        # Should not raise, even without a file
        result = oracle.score("yoga", "lunch")
        assert result == pytest.approx(0.5)

    def test_cache_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if they don't exist."""
        cache_file = tmp_path / "nested" / "subdir" / "cache.jsonl"
        client = _make_client('{"score": 0.6, "reason": "ok"}')
        oracle = _make_oracle(client, cache_path=cache_file)
        oracle.score("yoga", "gym")
        assert cache_file.exists()

    def test_second_score_appends_to_file(self, tmp_path):
        """Two distinct pairs to two lines in the JSONL file."""
        cache_file = tmp_path / "cache.jsonl"
        client = _make_client('{"score": 0.5, "reason": "ok"}')
        oracle = _make_oracle(client, cache_path=cache_file)
        oracle.score("yoga", "gym")
        oracle.score("running", "park")
        lines = [
            l for l in cache_file.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        assert len(lines) == 2

    def test_cache_os_error_on_read_silently_ignored(self, tmp_path):
        """OSError when reading existing cache file is silently ignored."""
        from unittest.mock import patch

        cache_file = tmp_path / "cache.jsonl"
        cache_file.write_text("", encoding="utf-8")

        client = _make_client('{"score": 0.4, "reason": "ok"}')
        with patch.object(
            type(cache_file), "read_text", side_effect=OSError("permission denied")
        ):
            # Should not raise; cache stays empty
            oracle = LLMJudgeOracle(client, cache_path=cache_file, temperature=0.0)
        # Cache is empty so LLM call is made
        result = oracle.score("yoga", "gym")
        assert result == pytest.approx(0.4)

    def test_cache_skips_blank_lines(self, tmp_path):
        """Blank lines in JSONL file are silently skipped (continue branch)."""
        cache_file = tmp_path / "cache.jsonl"
        good_entry = json.dumps(
            {
                "task_label": "yoga",
                "event_label": "gym",
                "score": 0.85,
                "model": "gpt-4o-mini",
                "ts": "2026-01-01T00:00:00+00:00",
            }
        )
        # File has leading/trailing blank lines
        cache_file.write_text("\n\n" + good_entry + "\n\n", encoding="utf-8")
        client = _make_client()
        oracle = _make_oracle(client, cache_path=cache_file)
        result = oracle.score("yoga", "gym")
        assert result == pytest.approx(0.85)
        client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Cache versioning
# ---------------------------------------------------------------------------


class TestCacheVersioning:
    def _record(self, version: str) -> str:
        entry = {
            "task_label": "yoga",
            "event_label": "gym",
            "score": 0.7,
            "model": "gpt-4o-mini",
            "ts": "2026-01-01T00:00:00+00:00",
        }
        if version is not None:
            entry["prompt_version"] = version
        return json.dumps(entry) + "\n"

    def test_record_from_other_version_is_ignored(self, tmp_path):
        cache_file = tmp_path / "cache.jsonl"
        cache_file.write_text(self._record("v1"), encoding="utf-8")
        client = _make_client('{"score": 0.95, "reason": "rescored"}')
        oracle = LLMJudgeOracle(
            client, cache_path=cache_file, temperature=0.0, cache_version="v2"
        )
        # The v1 score is not reused; the oracle re-scores under v2.
        assert oracle.score("yoga", "gym") == pytest.approx(0.95)
        client.chat.completions.create.assert_called_once()

    def test_record_with_matching_version_is_reused(self, tmp_path):
        cache_file = tmp_path / "cache.jsonl"
        cache_file.write_text(self._record("v2"), encoding="utf-8")
        client = _make_client()
        oracle = LLMJudgeOracle(
            client, cache_path=cache_file, temperature=0.0, cache_version="v2"
        )
        assert oracle.score("yoga", "gym") == pytest.approx(0.7)
        client.chat.completions.create.assert_not_called()

    def test_new_records_carry_the_prompt_version(self, tmp_path):
        cache_file = tmp_path / "cache.jsonl"
        client = _make_client('{"score": 0.8, "reason": "ok"}')
        oracle = LLMJudgeOracle(
            client, cache_path=cache_file, temperature=0.0, cache_version="v2"
        )
        oracle.score("yoga", "gym")
        entry = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
        assert entry["prompt_version"] == "v2"


# ---------------------------------------------------------------------------
# score_all_pairs()
# ---------------------------------------------------------------------------


class TestScoreAllPairs:
    def test_score_all_pairs_returns_all_combinations(self):
        """N tasks × M events to N×M pairs in result."""
        client = _make_client('{"score": 0.5, "reason": "ok"}')
        oracle = _make_oracle(client)
        tasks = [make_task(f"task_{i}") for i in range(2)]
        calendar = MagicMock()
        calendar.events = [make_event(f"event_{j}") for j in range(3)]
        results = oracle.score_all_pairs(tasks, calendar)
        # 2 tasks × 3 events = 6 pairs
        assert len(results) == 6

    def test_score_all_pairs_uses_cache(self):
        """Pairs already cached to no extra LLM calls."""
        client = _make_client('{"score": 0.7, "reason": "ok"}')
        oracle = _make_oracle(client)
        tasks = [make_task("yoga")]
        calendar = MagicMock()
        calendar.events = [make_event("gym")]
        # Pre-score to warm cache
        oracle.score("yoga", "gym")
        call_count_before = client.chat.completions.create.call_count
        # score_all_pairs should use cache for already-scored pairs
        oracle.score_all_pairs(tasks, calendar)
        assert client.chat.completions.create.call_count == call_count_before

    def test_score_all_pairs_uses_actual_label_plus_parent(self):
        """Office_work episodes are scored by their actual label plus parent."""
        client = _make_client('{"score": 0.5, "reason": "ok"}')
        oracle = _make_oracle(client)
        tasks = [make_task("hydrate")]
        calendar = MagicMock()
        calendar.events = [make_event("office_work", display_label="standup")]
        results = oracle.score_all_pairs(tasks, calendar)
        assert ("hydrate", "standup (office_work)") in results

    def test_score_all_pairs_deduplicates_tasks_by_label(self):
        """Duplicate task labels are scored only once."""
        client = _make_client('{"score": 0.6, "reason": "ok"}')
        oracle = _make_oracle(client)
        # Two tasks with the same label
        tasks = [make_task("yoga"), make_task("yoga")]
        calendar = MagicMock()
        calendar.events = [make_event("gym")]
        results = oracle.score_all_pairs(tasks, calendar)
        # Only 1 unique task × 1 event = 1 pair
        assert len(results) == 1
        assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# _build_task_text
# ---------------------------------------------------------------------------


class TestBuildTaskText:
    def test_with_display_and_description(self):
        result = LLMJudgeOracle._build_task_text(
            "yoga", "Morning Yoga", "A gentle yoga session."
        )
        assert result == "Morning Yoga: A gentle yoga session."

    def test_with_display_only(self):
        result = LLMJudgeOracle._build_task_text("yoga", "Morning Yoga", "")
        assert result == "Morning Yoga"

    def test_fallback_to_label(self):
        result = LLMJudgeOracle._build_task_text("yoga_practice", "", "")
        assert result == "yoga practice"


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------


class TestParseScore:
    def test_valid_json(self):
        assert LLMJudgeOracle._parse_score(
            '{"score": 0.9, "reason": "good"}'
        ) == pytest.approx(0.9)

    def test_partial_json_regex_fallback(self):
        assert LLMJudgeOracle._parse_score('"score": 0.55') == pytest.approx(0.55)

    def test_invalid_returns_zero(self):
        assert LLMJudgeOracle._parse_score("no score here") == pytest.approx(0.0)

    def test_json_without_score_key_returns_zero(self):
        assert LLMJudgeOracle._parse_score(
            '{"reason": "no score field"}'
        ) == pytest.approx(0.0)

    def test_clamping_high_value(self):
        assert LLMJudgeOracle._parse_score('{"score": 2.0}') == pytest.approx(1.0)

    def test_clamping_low_value(self):
        assert LLMJudgeOracle._parse_score('{"score": -1.0}') == pytest.approx(0.0)

    def test_regex_clamping_high(self):
        assert LLMJudgeOracle._parse_score('"score": 99.9') == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Construction guards + invoke_fn adapter (post-2026-05 wiring)
# ---------------------------------------------------------------------------


class TestConstructionGuards:
    def test_no_client_no_invoke_fn_raises(self):
        with pytest.raises(ValueError, match="OpenAI-shaped"):
            LLMJudgeOracle()

    def test_invoke_fn_only_works_without_client(self):
        """Pure-callable adapter; used to wrap LLMInterface from GraphRAG."""
        oracle = LLMJudgeOracle(invoke_fn=lambda _p: '{"score": 0.85, "reason": "ok"}')
        assert oracle.score("yoga", "gym") == pytest.approx(0.85)

    def test_invoke_fn_preferred_over_client(self):
        """When both supplied, invoke_fn wins; the OpenAI client is unused."""
        client = MagicMock()
        client.chat.completions.create.side_effect = AssertionError(
            "client must not be called when invoke_fn is supplied"
        )
        oracle = LLMJudgeOracle(client=client, invoke_fn=lambda _p: '{"score": 0.5}')
        assert oracle.score("a", "b") == pytest.approx(0.5)
        client.chat.completions.create.assert_not_called()

    def test_invoke_fn_response_stripped(self):
        oracle = LLMJudgeOracle(invoke_fn=lambda _p: '   {"score": 0.4}   ')
        assert oracle.score("a", "b") == pytest.approx(0.4)
