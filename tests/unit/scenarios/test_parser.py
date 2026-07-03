"""Unit tests for src.scripts.scenarios.task_generation.parser.

Coverage targets:
  _extract_json:
    - plain JSON array
    - markdown-fenced JSON (``json ... ``)
    - markdown fence without language tag (`` ... ``)
    - JSON array embedded in surrounding prose
    - invalid JSON to raises ValueError
    - non-list JSON to raises ValueError

  _task_from_dict:
    - all fields present to taken as-is
    - missing fields to defaults applied
    - empty ontology_uri string to coerced to None
    - legacy epoch keys silently dropped
    - label with spaces to underscored
    - empty label to "unknown_task"

  parse_task_list:
    - empty/whitespace text to []
    - valid text to list of RecommendedTask
    - non-dict items in array to skipped
    - task_overrides override LLM values
    - task_overrides for unknown label to no effect
    - invalid JSON text to raises ValueError
"""

from __future__ import annotations

import json

import pytest

from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.task_generation.parser import (
    _extract_json,
    _task_from_dict,
    extract_raw_task_dicts,
    parse_task_list,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_raw(**overrides: object) -> dict:
    base = {"label": "yoga"}
    base.update(overrides)
    return base


def _json_array(*dicts: dict) -> str:
    return json.dumps(list(dicts))


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json_array(self):
        raw = '[{"label": "yoga"}]'
        result = _extract_json(raw)
        assert result == [{"label": "yoga"}]

    def test_markdown_fence_with_json_tag(self):
        raw = '``json\n[{"label": "running"}]\n``'
        result = _extract_json(raw)
        assert result == [{"label": "running"}]

    def test_markdown_fence_without_language_tag(self):
        raw = '``\n[{"label": "cycling"}]\n``'
        result = _extract_json(raw)
        assert result == [{"label": "cycling"}]

    def test_json_embedded_in_prose(self):
        raw = 'Here are the tasks:\n\n[{"label": "gym"}]\n\nHope that helps!'
        result = _extract_json(raw)
        assert result == [{"label": "gym"}]

    def test_multiple_items_in_array(self):
        raw = '[{"label": "yoga"}, {"label": "running"}]'
        result = _extract_json(raw)
        assert len(result) == 2

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="could not parse JSON"):
            _extract_json("not valid json {{{{")

    def test_non_list_json_raises_value_error(self):
        with pytest.raises(ValueError, match="expected a JSON array"):
            _extract_json('{"label": "yoga"}')

    def test_fence_takes_priority_over_array_search(self):
        """Content inside a fence is used, not the surrounding array."""
        raw = 'outer_text [{"not": "this"}]\n``json\n[{"label": "correct"}]\n``'
        result = _extract_json(raw)
        assert result[0]["label"] == "correct"

    def test_empty_fence_gives_empty_array(self):
        raw = "``json\n[]\n``"
        result = _extract_json(raw)
        assert result == []


# ---------------------------------------------------------------------------
# _task_from_dict
# ---------------------------------------------------------------------------


class TestTaskFromDict:
    def test_all_fields_present(self):
        raw = {
            "label": "morning_yoga",
            "duration_min": 25,
            "duration_max": 45,
            "intensity": 3,
            "is_dividable": True,
            "is_concurrent": False,
            "ontology_uri": "https://example.org/yoga",
            "source_ontology": "HealthTasks",
            "preferred_epoch": "morning",
            "admissible_epochs": ["morning"],
        }
        task = _task_from_dict(raw, None)
        assert task.label == "morning_yoga"
        assert task.duration_min == 25
        assert task.duration_max == 45
        assert task.intensity == 3
        assert task.is_dividable is True
        assert task.is_concurrent is False
        assert task.ontology_uri == "https://example.org/yoga"
        for legacy in ("source_ontology", "preferred_epoch", "admissible_epochs"):
            assert not hasattr(task, legacy)

    def test_missing_fields_get_defaults(self):
        task = _task_from_dict({"label": "walk"}, None)
        assert task.duration_min == 15
        assert task.duration_max == 60
        assert task.intensity == 2
        assert task.is_dividable is False
        assert task.is_concurrent is False
        assert task.ontology_uri is None
        assert task.display_name == ""
        assert task.description == ""

    def test_effective_display_name_uses_set_display_name(self):
        task = _task_from_dict(
            {"label": "yoga", "display_name": "Morning Yoga 🧘"}, None
        )
        assert task.effective_display_name == "Morning Yoga 🧘"

    def test_effective_display_name_falls_back_to_title_label(self):
        task = _task_from_dict({"label": "morning_run"}, None)
        assert task.effective_display_name == "Morning Run"

    def test_effective_description_text_plus_uri(self):
        task = _task_from_dict(
            {
                "label": "yoga",
                "description": "Yoga improves flexibility.",
                "ontology_uri": "http://example.org/yoga",
            },
            None,
        )
        assert (
            task.effective_description
            == "Yoga improves flexibility. [http://example.org/yoga]"
        )

    def test_effective_description_no_duplicate_citation(self):
        task = _task_from_dict(
            {
                "label": "yoga",
                "description": "Yoga is good. [http://example.org/yoga]",
                "ontology_uri": "http://example.org/yoga",
            },
            None,
        )
        assert task.effective_description == "Yoga is good. [http://example.org/yoga]"

    def test_effective_description_empty_description_with_uri(self):
        task = _task_from_dict(
            {"label": "light_walk", "ontology_uri": "http://example.org/walk"},
            None,
        )
        assert task.effective_description == "Light Walk [http://example.org/walk]"

    def test_effective_description_both_empty_returns_empty(self):
        task = _task_from_dict({"label": "yoga"}, None)
        assert task.effective_description == ""

    def test_effective_description_description_only_no_uri(self):
        task = _task_from_dict(
            {"label": "yoga", "description": "Gentle stretch."}, None
        )
        assert task.effective_description == "Gentle stretch."

    def test_display_name_and_description_parsed(self):
        raw = {
            "label": "bodyweight_circuit",
            "display_name": "Do a Bodyweight Circuit 💪",
            "description": "Short circuit with squats and push-ups. [http://example.org/T1]",
        }
        task = _task_from_dict(raw, None)
        assert task.display_name == "Do a Bodyweight Circuit 💪"
        assert "Short circuit" in task.description
        assert "[http://example.org/T1]" in task.description

    def test_missing_display_name_defaults_to_empty_string(self):
        task = _task_from_dict({"label": "walk"}, None)
        assert task.display_name == ""

    def test_missing_description_defaults_to_empty_string(self):
        task = _task_from_dict({"label": "walk"}, None)
        assert task.description == ""

    def test_empty_ontology_uri_coerced_to_none(self):
        task = _task_from_dict({"label": "yoga", "ontology_uri": ""}, None)
        assert task.ontology_uri is None

    def test_legacy_source_ontology_key_is_silently_dropped(self):
        """Legacy task JSON files carry `source_ontology`; parsing them
        must succeed and ignore the field cleanly."""
        task = _task_from_dict(
            {"label": "yoga", "source_ontology": "HealthTasks"}, None
        )
        assert task.label == "yoga"
        assert not hasattr(task, "source_ontology")

    def test_label_spaces_replaced_with_underscores(self):
        task = _task_from_dict({"label": "mindful eating"}, None)
        assert task.label == "mindful_eating"

    def test_missing_label_gives_unknown_task(self):
        task = _task_from_dict({}, None)
        assert task.label == "unknown_task"

    def test_task_overrides_win_over_llm_values(self):
        raw = {"label": "running", "intensity": 2, "duration_min": 20}
        overrides = {"running": {"intensity": 5, "duration_min": 60}}
        task = _task_from_dict(raw, overrides)
        assert task.intensity == 5
        assert task.duration_min == 60

    def test_task_overrides_for_different_label_ignored(self):
        raw = {"label": "yoga", "intensity": 2}
        overrides = {"running": {"intensity": 5}}
        task = _task_from_dict(raw, overrides)
        assert task.intensity == 2

    def test_task_is_desiredtask_instance(self):
        task = _task_from_dict({"label": "swim"}, None)
        assert isinstance(task, RecommendedTask)


# ---------------------------------------------------------------------------
# extract_raw_task_dicts
# ---------------------------------------------------------------------------


class TestExtractRawTaskDicts:
    def test_empty_text_returns_empty_list(self):
        assert extract_raw_task_dicts("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert extract_raw_task_dicts("  \n  ") == []

    def test_valid_array_returns_dicts(self):
        raw = '[{"label": "yoga"}, {"label": "run"}]'
        result = extract_raw_task_dicts(raw)
        assert len(result) == 2
        assert result[0]["label"] == "yoga"

    def test_non_dict_items_filtered_out(self):
        raw = '[{"label": "yoga"}, "not a dict", 42]'
        result = extract_raw_task_dicts(raw)
        assert len(result) == 1
        assert result[0]["label"] == "yoga"


# ---------------------------------------------------------------------------
# parse_task_list
# ---------------------------------------------------------------------------


class TestParseTaskList:
    def test_empty_string_returns_empty_list(self):
        assert parse_task_list("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert parse_task_list("   \n\t  ") == []

    def test_valid_json_returns_tasks(self):
        raw = '[{"label": "yoga"}, {"label": "running"}]'
        tasks = parse_task_list(raw)
        assert len(tasks) == 2
        assert tasks[0].label == "yoga"
        assert tasks[1].label == "running"

    def test_markdown_fenced_json_parsed(self):
        raw = '``json\n[{"label": "swimming"}]\n``'
        tasks = parse_task_list(raw)
        assert len(tasks) == 1
        assert tasks[0].label == "swimming"

    def test_non_dict_items_skipped(self):
        raw = '[{"label": "yoga"}, "not a dict", 42, null, {"label": "running"}]'
        tasks = parse_task_list(raw)
        assert len(tasks) == 2
        labels = {t.label for t in tasks}
        assert labels == {"yoga", "running"}

    def test_empty_json_array_returns_empty_list(self):
        assert parse_task_list("[]") == []

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_task_list("definitely not json")

    def test_task_overrides_applied(self):
        raw = '[{"label": "yoga", "intensity": 1}]'
        overrides = {"yoga": {"intensity": 4}}
        tasks = parse_task_list(raw, task_overrides=overrides)
        assert tasks[0].intensity == 4

    def test_task_overrides_none_accepted(self):
        raw = '[{"label": "yoga"}]'
        tasks = parse_task_list(raw, task_overrides=None)
        assert len(tasks) == 1

    def test_defaults_propagated_for_minimal_task(self):
        raw = '[{"label": "stretch"}]'
        tasks = parse_task_list(raw)
        t = tasks[0]
        assert t.duration_min == 15
        assert t.duration_max == 60
        assert t.intensity == 2
        assert not t.is_dividable
        assert not t.is_concurrent

    def test_full_round_trip_with_all_fields(self):
        data = [
            {
                "label": "morning_run",
                "duration_min": 30,
                "duration_max": 60,
                "intensity": 4,
                "is_dividable": True,
                "is_concurrent": False,
                "ontology_uri": "https://w3id.org/calendar-bench/health/task/run",
                "source_ontology": "HealthTasks",
                "preferred_epoch": "morning",
                "admissible_epochs": ["morning"],
            }
        ]
        tasks = parse_task_list(json.dumps(data))
        t = tasks[0]
        assert t.label == "morning_run"
        assert t.duration_min == 30
        assert t.duration_max == 60
        assert t.intensity == 4
        assert t.is_dividable is True
        assert t.is_concurrent is False
        assert "calendar-bench" in t.ontology_uri
        for legacy in ("source_ontology", "preferred_epoch", "admissible_epochs"):
            assert not hasattr(t, legacy)

    def test_display_name_description_round_trip(self):
        data = [
            {
                "label": "morning_run",
                "display_name": "Morning Run 🏃",
                "description": "A light jog to start the day. [http://example.org/run]",
            }
        ]
        tasks = parse_task_list(json.dumps(data))
        assert tasks[0].display_name == "Morning Run 🏃"
        assert "[http://example.org/run]" in tasks[0].description

    def test_prose_with_embedded_json(self):
        raw = (
            "Based on the profile, I recommend:\n\n"
            '[{"label": "yoga", "intensity": 2}]\n\n'
            "These should fit well into the schedule."
        )
        tasks = parse_task_list(raw)
        assert len(tasks) == 1
        assert tasks[0].label == "yoga"
