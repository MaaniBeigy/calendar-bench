"""Unit tests for src.scripts.scenarios.augmentation.llm_agent.

The augmenter exposes recommended tasks to the LLM via natural display titles
(GraphRAG content is passed through verbatim; emojis preserved) and
identifies them by 1-based `task_index`; events are referred to by
their snake_case label rendered as Title Case in a JSON-array timeline.
Validation is purely structural; no semantic compatibility check -
because `LLMJudgeOracle` scores concurrent placements at evaluation time.
The augmenter also enforces:

* a hard waking window (`DailyWindow`);
* concurrent containment with a deterministic clip-rescue pass on
  partial overlaps; and
* a `host.is_concurrent` guard against placements onto exclusive
  events (this flag is benchmark-internal and the LLM never sees it).
"""

from __future__ import annotations

import datetime
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.scripts.persona.config.schema import DailyWindow
from src.scripts.scenarios.augmentation.llm_agent import (
    DirectLLMPipeline,
    LLMAugmenter,
    LLMCallRecorder,
    _build_calendar_summary,
    _build_decision_lookup,
    _build_tasks_list,
    _coerce_index,
    _DirectResponse,
    _event_display_title,
    _extract_concurrent_ref,
    _extract_task_label,
    _find_host_event,
    _format_date,
    _hhmm_to_minutes,
    _lookup_decision,
    _maybe_dump_call,
    _normalise_title,
    _parse_oneshot_response,
    _resolve_event_label_on_date,
    _ScheduleDecision,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, LLMAgentConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)
DATE2 = datetime.date(2026, 5, 5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_config(
    max_retries: int = 3, repeat_per_week: bool = False
) -> AugmentationConfig:
    return AugmentationConfig(
        method="llm_agent",
        allow_merge=True,
        merge_threshold=0.70,
        repeat_per_week=repeat_per_week,
        llm_agent=LLMAgentConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_retries=max_retries,
            prompt_template="augment_oneshot",
        ),
    )


def _oneshot_json(*entries: dict) -> str:
    return json.dumps(list(entries))


def _placed_entry(
    task_label: str = "yoga",
    *,
    task_index: int | None = None,
    date: str = "2026-05-04",
    start: str = "08:00",
    end: str = "09:00",
    concurrent_flag: bool = False,
    concurrent_with: object = None,
) -> dict:
    entry: dict = {
        "task_label": task_label,
        "scheduled": True,
        "date": date,
        "start": start,
        "end": end,
        "concurrent_flag": concurrent_flag,
        "concurrent_with": concurrent_with,
    }
    if task_index is not None:
        entry["task_index"] = task_index
    return entry


def _skipped_entry(task_label: str = "yoga", *, task_index: int | None = None) -> dict:
    entry: dict = {"task_label": task_label, "scheduled": False}
    if task_index is not None:
        entry["task_index"] = task_index
    return entry


class _MockResponse:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _SeqPipeline:
    """Returns successive answers; repeats the last one once exhausted."""

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self._idx = 0
        self.calls: list[str] = []

    def search(self, query_text: str, **_: Any) -> _MockResponse:
        self.calls.append(query_text)
        ans = self._answers[min(self._idx, len(self._answers) - 1)]
        self._idx += 1
        return _MockResponse(ans)


def _trace(*events) -> CalendarTrace:
    return CalendarTrace(person_id="p001", events=list(events))


def _make_augmenter(pipeline, allen_rules=None, daily_window=None):
    return LLMAugmenter(
        llm_pipeline=pipeline,
        allen_rules=allen_rules,
        daily_window=daily_window,
    )


def _wide_window() -> DailyWindow:
    """A daily window that lets every test slot pass."""
    return DailyWindow(wake_minutes=0, sleep_minutes=1440)


# ---------------------------------------------------------------------------
# _hhmm_to_minutes
# ---------------------------------------------------------------------------


class TestHhmmToMinutes:
    def test_midnight(self):
        assert _hhmm_to_minutes("00:00") == 0

    def test_eight_am(self):
        assert _hhmm_to_minutes("08:00") == 480

    def test_half_past_nine(self):
        assert _hhmm_to_minutes("09:30") == 570

    def test_end_of_day(self):
        assert _hhmm_to_minutes("23:59") == 1439


# ---------------------------------------------------------------------------
# _format_date / _event_display_title
# ---------------------------------------------------------------------------


class TestFormatDate:
    def test_includes_day_name(self):
        assert "Monday" in _format_date(DATE)

    def test_includes_month_name(self):
        assert "May" in _format_date(DATE)

    def test_includes_year(self):
        assert "2026" in _format_date(DATE)

    def test_day_zero_padded(self):
        assert "04" in _format_date(DATE)


class TestEventDisplayTitle:
    def test_snake_case_titlecased(self):
        assert _event_display_title("office_work") == "Office Work"

    def test_single_word(self):
        assert _event_display_title("lunch") == "Lunch"

    def test_multi_underscore(self):
        assert _event_display_title("visit_family") == "Visit Family"


# ---------------------------------------------------------------------------
# _normalise_title and _resolve_event_label_on_date
# ---------------------------------------------------------------------------


class TestNormaliseTitle:
    def test_natural_with_emoji(self):
        assert _normalise_title("Lunch 🥗") == "lunch"

    def test_multi_word_titlecase(self):
        assert _normalise_title("Office Work") == "office_work"

    def test_snake_case_unchanged(self):
        assert _normalise_title("office_work") == "office_work"

    def test_extra_punctuation_stripped(self):
        assert _normalise_title("Visit-Family!") == "visit_family"

    def test_multiple_spaces_collapsed(self):
        assert _normalise_title("Office   Work") == "office_work"

    def test_leading_trailing_underscores_stripped(self):
        assert _normalise_title("  lunch  ") == "lunch"

    def test_empty_string_returns_empty(self):
        assert _normalise_title("") == ""


class TestResolveEventLabelOnDate:
    def test_none_raw_returns_none(self):
        ev = make_event("lunch", date=DATE)
        assert _resolve_event_label_on_date(None, DATE, {DATE: [ev]}) is None

    def test_none_date_returns_none(self):
        assert _resolve_event_label_on_date("Lunch", None, {}) is None

    def test_empty_normalised_returns_none(self):
        ev = make_event("lunch", date=DATE)
        assert _resolve_event_label_on_date("@@@", DATE, {DATE: [ev]}) is None

    def test_natural_title_resolves_to_snake_case(self):
        ev = make_event("office_work", date=DATE)
        assert (
            _resolve_event_label_on_date("Office Work", DATE, {DATE: [ev]})
            == "office_work"
        )

    def test_with_emoji_resolves(self):
        ev = make_event("lunch", date=DATE)
        assert _resolve_event_label_on_date("Lunch 🥗", DATE, {DATE: [ev]}) == "lunch"

    def test_snake_case_resolves(self):
        ev = make_event("office_work", date=DATE)
        assert (
            _resolve_event_label_on_date("office_work", DATE, {DATE: [ev]})
            == "office_work"
        )

    def test_no_match_returns_none(self):
        ev = make_event("lunch", date=DATE)
        assert _resolve_event_label_on_date("Dinner", DATE, {DATE: [ev]}) is None

    def test_no_events_for_date(self):
        assert _resolve_event_label_on_date("Lunch", DATE, {}) is None


# ---------------------------------------------------------------------------
# _coerce_index, _extract_task_label, _extract_concurrent_ref
# ---------------------------------------------------------------------------


class TestCoerceIndex:
    def test_int_passthrough(self):
        assert _coerce_index(3) == 3

    def test_numeric_string(self):
        assert _coerce_index("5") == 5

    def test_none(self):
        assert _coerce_index(None) is None

    def test_garbage_returns_none(self):
        assert _coerce_index("abc") is None

    def test_dict_returns_none(self):
        assert _coerce_index({"a": 1}) is None


class TestExtractTaskLabel:
    def test_label_field(self):
        assert _extract_task_label({"task_label": "yoga"}) == "yoga"

    def test_title_field(self):
        assert _extract_task_label({"task_title": "Light Yoga"}) == "Light Yoga"

    def test_label_preferred_over_title(self):
        assert (
            _extract_task_label({"task_label": "yoga", "task_title": "Light Yoga"})
            == "yoga"
        )

    def test_missing_returns_empty(self):
        assert _extract_task_label({}) == ""

    def test_non_string_value_skipped(self):
        assert _extract_task_label({"task_label": 42}) == ""

    def test_whitespace_stripped(self):
        assert _extract_task_label({"task_label": "  yoga  "}) == "yoga"

    def test_blank_string_skipped(self):
        assert _extract_task_label({"task_label": "   "}) == ""


class TestExtractConcurrentRef:
    def test_natural_field(self):
        assert _extract_concurrent_ref({"concurrent_with_event": "Lunch"}) == "Lunch"

    def test_legacy_field(self):
        assert _extract_concurrent_ref({"concurrent_with": "lunch"}) == "lunch"

    def test_natural_preferred_over_legacy(self):
        item = {"concurrent_with_event": "Lunch", "concurrent_with": "lunch"}
        assert _extract_concurrent_ref(item) == "Lunch"

    def test_none_returns_none(self):
        assert _extract_concurrent_ref({"concurrent_with": None}) is None

    def test_missing_returns_none(self):
        assert _extract_concurrent_ref({}) is None

    def test_blank_returns_none(self):
        assert _extract_concurrent_ref({"concurrent_with": "   "}) is None

    def test_non_string_returns_none(self):
        assert _extract_concurrent_ref({"concurrent_with": 42}) is None


# ---------------------------------------------------------------------------
# _parse_oneshot_response
# ---------------------------------------------------------------------------


class TestParseOneshotResponse:
    def test_empty_string_returns_none(self):
        assert _parse_oneshot_response("") is None

    def test_whitespace_returns_none(self):
        assert _parse_oneshot_response("   ") is None

    def test_malformed_json_returns_none(self):
        assert _parse_oneshot_response("not json {{{{") is None

    def test_non_list_returns_none(self):
        assert (
            _parse_oneshot_response('{"task_label": "yoga", "scheduled": true}') is None
        )

    def test_valid_placed_entry(self):
        raw = _oneshot_json(_placed_entry("yoga"))
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert len(result) == 1
        d = result[0]
        assert d.task_label == "yoga"
        assert d.scheduled is True
        assert d.date == DATE
        assert d.start_minutes == 480
        assert d.end_minutes == 540

    def test_unscheduled_entry(self):
        raw = _oneshot_json(_skipped_entry("walk"))
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert result[0].scheduled is False

    def test_concurrent_entry_legacy(self):
        raw = _oneshot_json(
            _placed_entry(
                "mindful_eating", concurrent_flag=True, concurrent_with="lunch"
            )
        )
        result = _parse_oneshot_response(raw)
        assert result is not None
        d = result[0]
        assert d.concurrent_flag is True
        assert d.concurrent_with == "lunch"

    def test_concurrent_with_event_natural_title(self):
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Mindful Eating",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "12:00",
                    "end": "12:20",
                    "concurrent_with_event": "Lunch",
                }
            ]
        )
        result = _parse_oneshot_response(raw)
        assert result is not None
        d = result[0]
        assert d.task_index == 1
        assert d.concurrent_with == "Lunch"
        assert d.concurrent_flag is True

    def test_concurrent_with_event_null_means_standalone(self):
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Yoga",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "08:00",
                    "end": "08:30",
                    "concurrent_with_event": None,
                }
            ]
        )
        result = _parse_oneshot_response(raw)
        assert result is not None
        d = result[0]
        assert d.concurrent_flag is False
        assert d.concurrent_with is None

    def test_task_index_only_no_label(self):
        raw = json.dumps(
            [
                {
                    "task_index": 2,
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "08:00",
                    "end": "08:30",
                }
            ]
        )
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert result[0].task_index == 2
        assert result[0].task_label == ""

    def test_unscheduled_with_task_index(self):
        raw = json.dumps([{"task_index": 3, "scheduled": False}])
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert result[0].scheduled is False
        assert result[0].task_index == 3

    def test_malformed_date_produces_unscheduled(self):
        raw = _oneshot_json(
            {
                "task_label": "yoga",
                "scheduled": True,
                "date": "not-a-date",
                "start": "08:00",
                "end": "09:00",
            }
        )
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert result[0].scheduled is False

    def test_missing_time_fields_produces_unscheduled(self):
        raw = _oneshot_json(
            {"task_label": "yoga", "scheduled": True, "date": "2026-05-04"}
        )
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert result[0].scheduled is False

    def test_non_dict_items_skipped(self):
        raw = json.dumps([_placed_entry("yoga"), "not a dict", 42])
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert len(result) == 1

    def test_item_missing_label_and_index_skipped(self):
        raw = json.dumps([{"scheduled": False}])
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert len(result) == 0

    def test_markdown_fenced_array(self):
        inner = _oneshot_json(_placed_entry("yoga"))
        raw = f"``json\n{inner}\n``"
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert result[0].task_label == "yoga"

    def test_array_embedded_in_prose(self):
        inner = _oneshot_json(_placed_entry("yoga"))
        raw = f"Here is my plan:\n{inner}\nEnd."
        result = _parse_oneshot_response(raw)
        assert result is not None
        assert result[0].task_label == "yoga"

    def test_concurrent_flag_false_drops_concurrent_with(self):
        raw = json.dumps(
            [
                {
                    "task_label": "yoga",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "08:00",
                    "end": "09:00",
                    "concurrent_flag": False,
                    "concurrent_with": "lunch",
                }
            ]
        )
        result = _parse_oneshot_response(raw)
        d = result[0]
        assert d.concurrent_flag is False
        assert d.concurrent_with is None


# ---------------------------------------------------------------------------
# _maybe_dump_call
# ---------------------------------------------------------------------------


class TestMaybeDumpCall:
    def test_no_dump_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_AGENT_DEBUG_DIR", raising=False)
        _maybe_dump_call([DATE], 0, "prompt", "response", [])
        assert list(tmp_path.iterdir()) == []

    def test_no_dump_when_week_dates_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        _maybe_dump_call([], 0, "prompt", "response", None)
        assert list(tmp_path.iterdir()) == []

    def test_dump_writes_file_under_unknown_when_no_person_id(
        self, tmp_path, monkeypatch
    ):
        """Back-compat: callers that omit `person_id` end up under the
        default `unknown` sub-directory rather than at the top level."""
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        _maybe_dump_call([DATE], 0, "<prompt>", "<resp>", [])
        unknown_dir = tmp_path / "unknown"
        assert unknown_dir.is_dir()
        files = list(unknown_dir.iterdir())
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        assert "<prompt>" in body and "<resp>" in body

    def test_dump_summary_marks_unparseable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        _maybe_dump_call([DATE], 0, "p", "r", None)
        body = next((tmp_path / "unknown").iterdir()).read_text(encoding="utf-8")
        assert "unparseable" in body

    def test_dump_summary_counts_concurrent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        decisions = [
            _ScheduleDecision(
                task_label="x",
                scheduled=True,
                date=DATE,
                start_minutes=480,
                end_minutes=510,
                concurrent_flag=True,
                concurrent_with="lunch",
            ),
            _ScheduleDecision(task_label="y", scheduled=False),
        ]
        _maybe_dump_call([DATE], 0, "p", "r", decisions)
        body = next((tmp_path / "unknown").iterdir()).read_text(encoding="utf-8")
        assert "scheduled=1" in body
        assert "concurrent=1" in body

    def test_dump_swallows_oserror(self, tmp_path, monkeypatch):
        existing_file = tmp_path / "not_a_dir"
        existing_file.write_text("blocker", encoding="utf-8")
        bad_path = existing_file / "subdir"
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(bad_path))
        _maybe_dump_call([DATE], 0, "p", "r", [])  # must not raise

    def test_dump_writes_under_person_id_subdirectory(self, tmp_path, monkeypatch):
        """A supplied `person_id` becomes the dump file's parent
        directory so per-persona logs do not overwrite each other."""
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        _maybe_dump_call(
            [DATE], 0, "<prompt>", "<resp>", [], person_id="a_student_0009"
        )
        person_dir = tmp_path / "a_student_0009"
        assert person_dir.is_dir()
        files = list(person_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == f"{DATE.isoformat()}__attempt0.txt"

    def test_dump_two_personas_do_not_collide(self, tmp_path, monkeypatch):
        """Two personas writing for the same week-start each get their own
        sub-directory; neither overwrites the other."""
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        _maybe_dump_call([DATE], 0, "PA", "RA", [], person_id="alice")
        _maybe_dump_call([DATE], 0, "PB", "RB", [], person_id="bob")
        alice_files = list((tmp_path / "alice").iterdir())
        bob_files = list((tmp_path / "bob").iterdir())
        assert len(alice_files) == 1 and len(bob_files) == 1
        assert "PA" in alice_files[0].read_text(encoding="utf-8")
        assert "PB" in bob_files[0].read_text(encoding="utf-8")

    def test_dump_records_person_id_in_body(self, tmp_path, monkeypatch):
        """The dump body includes the persona id so a file remains
        self-describing if it's separated from its directory."""
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        _maybe_dump_call([DATE], 0, "p", "r", [], person_id="a_fulltime_0001")
        body = next((tmp_path / "a_fulltime_0001").iterdir()).read_text(
            encoding="utf-8"
        )
        assert "=== person ===" in body
        assert "a_fulltime_0001" in body


# ---------------------------------------------------------------------------
# _build_calendar_summary
# ---------------------------------------------------------------------------


class TestBuildCalendarSummary:
    def test_free_day(self):
        summary = _build_calendar_summary([DATE], {})
        assert '"type":"free"' in summary
        assert "TIMELINE" in summary

    def test_day_with_events_uses_natural_title(self):
        ev = make_event("office_work", start_minutes=540, end_minutes=1020, date=DATE)
        summary = _build_calendar_summary([DATE], {DATE: [ev]})
        assert '"label":"Office Work"' in summary
        assert "office_work" not in summary

    def test_busy_entry_omits_accepts_concurrent_for_concurrent_event(self):
        """The LLM must never see `is_concurrent` (rendered as
        `accepts_concurrent`).  Concurrency is a semantic judgement
        scored by the oracle at evaluation."""
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
        )
        summary = _build_calendar_summary([DATE], {DATE: [ev]})
        assert "accepts_concurrent" not in summary

    def test_busy_entry_omits_accepts_concurrent_for_non_concurrent_event(self):
        ev = make_event(
            "office_work",
            start_minutes=540,
            end_minutes=1020,
            date=DATE,
            is_concurrent=False,
        )
        summary = _build_calendar_summary([DATE], {DATE: [ev]})
        assert "accepts_concurrent" not in summary

    def test_busy_entry_carries_only_label_start_end(self):
        """Busy intervals must expose exactly type/label/start/end; no
        extra benchmark-internal fields leak to the LLM."""
        ev = make_event(
            "office_work",
            start_minutes=540,
            end_minutes=1020,
            date=DATE,
            is_concurrent=True,
        )
        summary = _build_calendar_summary([DATE], {DATE: [ev]})
        # Find the busy line and parse it back as JSON.
        busy_lines = [
            ln.strip().rstrip(",")
            for ln in summary.splitlines()
            if '"type":"busy"' in ln
        ]
        assert busy_lines, "expected a busy line"
        payload = json.loads(busy_lines[0])
        assert set(payload.keys()) == {"type", "label", "start", "end"}

    def test_summary_never_exposes_snake_case(self):
        ev = make_event("office_work", start_minutes=540, end_minutes=1020, date=DATE)
        summary = _build_calendar_summary([DATE], {DATE: [ev]})
        assert "office_work" not in summary

    def test_day_with_placed_tasks_shows_done_type(self):
        from src.scripts.scenarios.domain.task import ScheduledTask

        st = ScheduledTask(
            task=make_task("light_yoga", display_name="Do Light Yoga 🧘"),
            start_minutes=480,
            end_minutes=510,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
        )
        summary = _build_calendar_summary([DATE], {}, scheduled_tasks=[st])
        assert '"type":"done"' in summary
        assert "Do Light Yoga" in summary

    def test_done_during_concurrent_carries_host(self):
        from src.scripts.scenarios.domain.task import ScheduledTask

        st = ScheduledTask(
            task=make_task("mindful_eating", display_name="Mindful Eating"),
            start_minutes=720,
            end_minutes=740,
            is_standalone=False,
            concurrent_with="office_work",
            date=DATE,
        )
        summary = _build_calendar_summary([DATE], {}, scheduled_tasks=[st])
        assert '"host":"Office Work"' in summary

    def test_done_standalone_carries_null_host(self):
        from src.scripts.scenarios.domain.task import ScheduledTask

        st = ScheduledTask(
            task=make_task("yoga", display_name="Yoga"),
            start_minutes=480,
            end_minutes=510,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
        )
        summary = _build_calendar_summary([DATE], {}, scheduled_tasks=[st])
        assert '"host":null' in summary

    def test_multiple_days_use_human_format(self):
        summary = _build_calendar_summary([DATE, DATE2], {})
        assert "May" in summary and summary.count("2026") >= 2

    def test_date_format_day_name(self):
        summary = _build_calendar_summary([DATE], {})
        names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        assert any(n in summary for n in names)

    def test_non_overlapping_events_show_free_gap(self):
        ev1 = make_event("breakfast", start_minutes=420, end_minutes=450, date=DATE)
        ev2 = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        summary = _build_calendar_summary([DATE], {DATE: [ev1, ev2]})
        assert '"type":"free"' in summary
        assert '"start":"07:30"' in summary

    def test_overlapping_events_kept_as_separate_busy_entries(self):
        ev1 = make_event("office_work", start_minutes=540, end_minutes=900, date=DATE)
        ev2 = make_event("meeting", start_minutes=720, end_minutes=960, date=DATE)
        summary = _build_calendar_summary([DATE], {DATE: [ev1, ev2]})
        assert '"label":"Office Work"' in summary
        assert '"label":"Meeting"' in summary

    def test_multiple_placed_tasks_show_as_done(self):
        from src.scripts.scenarios.domain.task import ScheduledTask

        st1 = ScheduledTask(
            task=make_task("yoga", display_name="Yoga 🧘"),
            start_minutes=480,
            end_minutes=510,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
        )
        st2 = ScheduledTask(
            task=make_task("walk", display_name="Walk 🚶"),
            start_minutes=600,
            end_minutes=630,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
        )
        summary = _build_calendar_summary([DATE], {}, scheduled_tasks=[st1, st2])
        assert summary.count('"type":"done"') == 2

    def test_done_outside_week_window_omitted(self):
        from src.scripts.scenarios.domain.task import ScheduledTask

        st = ScheduledTask(
            task=make_task("yoga"),
            start_minutes=480,
            end_minutes=510,
            is_standalone=True,
            concurrent_with=None,
            date=DATE2,
        )
        summary = _build_calendar_summary([DATE], {}, scheduled_tasks=[st])
        assert '"type":"done"' not in summary

    def test_end_of_day_clamps_to_23_59(self):
        summary = _build_calendar_summary([DATE], {})
        assert '"end":"23:59"' in summary
        assert "24:00" not in summary

    def test_done_label_passes_through_emoji_verbatim(self):
        """GraphRAG-supplied display names are rendered verbatim; the
        augmenter never modifies task content."""
        from src.scripts.scenarios.domain.task import ScheduledTask

        st = ScheduledTask(
            task=make_task("yoga", display_name="Yoga 🧘"),
            start_minutes=480,
            end_minutes=510,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
        )
        summary = _build_calendar_summary([DATE], {}, scheduled_tasks=[st])
        assert '"label":"Yoga 🧘"' in summary

    def test_done_label_uses_effective_display_name_when_no_explicit_name(self):
        """When GraphRAG omits `display_name` the label snake_case is
        title-cased (a derivation, not a modification)."""
        from src.scripts.scenarios.domain.task import ScheduledTask

        st = ScheduledTask(
            task=make_task("light_yoga"),
            start_minutes=480,
            end_minutes=510,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
        )
        summary = _build_calendar_summary([DATE], {}, scheduled_tasks=[st])
        assert '"label":"Light Yoga"' in summary


# ---------------------------------------------------------------------------
# _build_tasks_list (no recommended_hosts; ontology-agnostic)
# ---------------------------------------------------------------------------


class TestBuildTasksList:
    def test_uses_task_index_heading(self):
        t = make_task("morning_yoga")
        result = _build_tasks_list([t])
        assert "TASK #1" in result
        assert "task_index" in result

    def test_uses_natural_title_not_snake_case(self):
        t = make_task("morning_yoga")
        result = _build_tasks_list([t])
        assert '"Morning Yoga"' in result
        assert "morning_yoga" not in result

    def test_explicit_display_name_used_verbatim(self):
        """GraphRAG-supplied `display_name` (emoji included) flows through
        verbatim into the LLM's task list; the augmenter is content-agnostic."""
        t = make_task("morning_yoga", display_name="Sunrise Yoga 🧘")
        result = _build_tasks_list([t])
        assert '"Sunrise Yoga 🧘"' in result

    def test_contains_duration_min_max_keys(self):
        t = make_task("yoga", duration_min=20, duration_max=40)
        result = _build_tasks_list([t])
        assert "duration_min      : 20" in result
        assert "duration_max      : 40" in result

    def test_contains_intensity(self):
        t = make_task("yoga", intensity=4)
        result = _build_tasks_list([t])
        assert "intensity         : 4" in result

    def test_concurrent_ok_true_for_concurrent_task(self):
        t = make_task("yoga", is_concurrent=True)
        result = _build_tasks_list([t])
        assert "concurrent_ok     : true" in result

    def test_concurrent_ok_false_for_standalone_task(self):
        t = make_task("yoga", is_concurrent=False)
        result = _build_tasks_list([t])
        assert "concurrent_ok     : false" in result

    def test_contains_description_when_set(self):
        t = make_task("yoga", description="Gentle yoga for flexibility.")
        result = _build_tasks_list([t])
        assert "description" in result and "Gentle yoga" in result

    def test_description_omitted_when_empty(self):
        t = make_task("yoga", description="")
        result = _build_tasks_list([t])
        assert "description" not in result

    def test_multiple_tasks_indexed(self):
        tasks = [make_task(f"t{i}") for i in range(3)]
        result = _build_tasks_list(tasks)
        assert "TASK #1" in result and "TASK #2" in result and "TASK #3" in result

    def test_no_recommended_hosts_field(self):
        """Tasks list never carries recommended_hosts; augmenter is
        ontology-agnostic."""
        t = make_task("mindful_eating", is_concurrent=True)
        result = _build_tasks_list([t])
        assert "recommended_hosts" not in result


# ---------------------------------------------------------------------------
# _build_decision_lookup / _lookup_decision
# ---------------------------------------------------------------------------


class TestBuildDecisionLookup:
    def test_indexes_by_task_index(self):
        d1 = _ScheduleDecision(task_label="yoga", scheduled=False, task_index=1)
        d2 = _ScheduleDecision(task_label="walk", scheduled=False, task_index=2)
        by_idx, _ = _build_decision_lookup([d1, d2])
        assert by_idx[1] is d1 and by_idx[2] is d2

    def test_label_queue_preserves_order(self):
        d1 = _ScheduleDecision(task_label="yoga", scheduled=False)
        d2 = _ScheduleDecision(task_label="yoga", scheduled=False)
        _, by_lbl = _build_decision_lookup([d1, d2])
        assert by_lbl["yoga"] == [d1, d2]

    def test_decision_without_label_skipped_for_label_queue(self):
        d = _ScheduleDecision(task_label="", scheduled=False, task_index=5)
        by_idx, by_lbl = _build_decision_lookup([d])
        assert by_idx[5] is d
        assert by_lbl == {}


class TestLookupDecision:
    def test_index_match_preferred(self):
        d_idx = _ScheduleDecision(task_label="yoga", scheduled=False, task_index=1)
        d_lbl = _ScheduleDecision(task_label="yoga", scheduled=False)
        assert (
            _lookup_decision(make_task("yoga"), 1, {1: d_idx}, {"yoga": [d_lbl]})
            is d_idx
        )

    def test_label_fallback_when_no_index(self):
        d = _ScheduleDecision(task_label="yoga", scheduled=False)
        by_lbl = {"yoga": [d]}
        assert _lookup_decision(make_task("yoga"), 1, {}, by_lbl) is d
        assert by_lbl["yoga"] == []

    def test_label_queue_handles_duplicates(self):
        d1 = _ScheduleDecision(task_label="yoga", scheduled=False)
        d2 = _ScheduleDecision(task_label="yoga", scheduled=False)
        by_lbl = {"yoga": [d1, d2]}
        assert _lookup_decision(make_task("yoga"), 1, {}, by_lbl) is d1
        assert _lookup_decision(make_task("yoga"), 2, {}, by_lbl) is d2

    def test_title_fallback_when_label_missing(self):
        d = _ScheduleDecision(task_label="Light Yoga", scheduled=False)
        by_lbl = {"light_yoga": [d]}
        task = make_task("yoga", display_name="Light Yoga")
        assert _lookup_decision(task, 1, {}, by_lbl) is d

    def test_returns_none_when_no_match(self):
        assert _lookup_decision(make_task("yoga"), 1, {}, {}) is None


# ---------------------------------------------------------------------------
# LLMAugmenter.augment; core
# ---------------------------------------------------------------------------


class TestLLMAugmenterAugment:
    _EV = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
    _TASK = make_task("yoga", duration_min=30, duration_max=60)

    def test_empty_calendar_all_unscheduled(self):
        pipeline = _SeqPipeline(_oneshot_json(_placed_entry("yoga")))
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(), [self._TASK], _llm_config())
        assert sol.scheduled == []
        assert len(sol.unscheduled) == 1

    def test_single_task_placed(self):
        pipeline = _SeqPipeline(
            _oneshot_json(_placed_entry("yoga", start="00:00", end="00:30"))
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [self._TASK], _llm_config())
        assert len(sol.scheduled) == 1

    def test_single_task_unscheduled_by_llm(self):
        pipeline = _SeqPipeline(_oneshot_json(_skipped_entry("yoga")))
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [self._TASK], _llm_config())
        assert sol.scheduled == []

    def test_task_missing_from_response_is_unscheduled(self):
        pipeline = _SeqPipeline(_oneshot_json())
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [self._TASK], _llm_config())
        assert len(sol.unscheduled) == 1

    def test_all_tasks_placed(self):
        ev = make_event("sleep", start_minutes=60, end_minutes=1440, date=DATE)
        t1 = make_task("yoga", duration_min=30, duration_max=60)
        t2 = make_task("walk", duration_min=30, duration_max=60)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry("yoga", start="00:00", end="00:30"),
                _placed_entry("walk", start="00:30", end="01:00"),
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [t1, t2], _llm_config())
        assert len(sol.scheduled) == 2

    def test_invalid_date_skips_task(self):
        raw = _oneshot_json(
            _placed_entry("yoga", date="1999-01-01", start="08:00", end="09:00")
        )
        aug = _make_augmenter(_SeqPipeline(raw), daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [self._TASK], _llm_config())
        assert sol.scheduled == []

    def test_invalid_duration_too_short(self):
        task = make_task("yoga", duration_min=60, duration_max=90)
        raw = _oneshot_json(_placed_entry("yoga", start="08:00", end="08:10"))
        aug = _make_augmenter(_SeqPipeline(raw), daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [task], _llm_config())
        assert sol.scheduled == []

    def test_invalid_duration_too_long(self):
        task = make_task("yoga", duration_min=10, duration_max=20)
        raw = _oneshot_json(_placed_entry("yoga", start="08:00", end="09:00"))
        aug = _make_augmenter(_SeqPipeline(raw), daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [task], _llm_config())
        assert sol.scheduled == []

    def test_malformed_json_retried(self):
        pipeline = _SeqPipeline(
            "not json",
            _oneshot_json(_placed_entry("yoga", start="00:00", end="00:30")),
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [self._TASK], _llm_config(max_retries=2))
        assert len(sol.scheduled) == 1
        assert len(pipeline.calls) == 2

    def test_retries_exhausted_all_unscheduled(self):
        pipeline = _SeqPipeline("bad", "still bad", "nope")
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(self._EV), [self._TASK], _llm_config(max_retries=3))
        assert sol.scheduled == []
        assert len(pipeline.calls) == 3

    def test_concurrent_placement_no_semantic_check(self):
        """Concurrent placements pass with NO semantic gate; that's the
        oracle's job at evaluation time."""
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
        )
        task = make_task("anything_at_all", duration_min=20, duration_max=40)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry(
                    "anything_at_all",
                    start="12:00",
                    end="12:20",
                    concurrent_flag=True,
                    concurrent_with="lunch",
                )
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert len(sol.scheduled) == 1
        assert sol.scheduled[0].concurrent_with == "lunch"

    def test_concurrent_placement_natural_title_resolves(self):
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
        )
        task = make_task("anything", duration_min=20, duration_max=40)
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Anything",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "12:00",
                    "end": "12:20",
                    "concurrent_with_event": "Lunch",
                }
            ]
        )
        pipeline = _SeqPipeline(raw)
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert len(sol.scheduled) == 1
        assert sol.scheduled[0].concurrent_with == "lunch"

    def test_concurrent_phantom_event_unscheduled(self):
        ev = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        task = make_task("anything", duration_min=20, duration_max=40)
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Anything",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "12:00",
                    "end": "12:20",
                    "concurrent_with_event": "Phantom",
                }
            ]
        )
        aug = _make_augmenter(_SeqPipeline(raw), daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert sol.scheduled == []

    def test_concurrent_outside_host_interval_rejected(self):
        ev = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        task = make_task("anything", duration_min=10, duration_max=30)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry(
                    "anything",
                    start="13:22",
                    end="13:37",
                    concurrent_flag=True,
                    concurrent_with="lunch",
                )
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert sol.scheduled == []

    def test_within_week_second_task_cannot_reuse_host(self):
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
        )
        t1 = make_task("a", intensity=5, duration_min=20, duration_max=40)
        t2 = make_task("b", intensity=3, duration_min=20, duration_max=40)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry(
                    "a",
                    start="12:00",
                    end="12:20",
                    concurrent_flag=True,
                    concurrent_with="lunch",
                ),
                _placed_entry(
                    "b",
                    start="12:00",
                    end="12:20",
                    concurrent_flag=True,
                    concurrent_with="lunch",
                ),
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [t1, t2], _llm_config())
        assert sum(1 for st in sol.scheduled if st.concurrent_with == "lunch") <= 1

    def test_base_events_always_full_original_calendar(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        pipeline = _SeqPipeline(
            _oneshot_json(_placed_entry("yoga", start="00:00", end="00:30"))
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [self._TASK], _llm_config())
        assert ev in sol.augmented_calendar.base_events

    def test_concurrent_repeat_per_week(self):
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
        )
        task = make_task("anything", duration_min=20, duration_max=40)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry(
                    "anything",
                    start="12:00",
                    end="12:20",
                    concurrent_flag=True,
                    concurrent_with="lunch",
                )
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [task], _llm_config(repeat_per_week=True))
        assert len(sol.scheduled) == 1

    def test_prompt_uses_natural_title_and_human_date(self):
        task = make_task(
            "morning_run", display_name="Morning Run", duration_min=30, duration_max=60
        )
        pipeline = _SeqPipeline(_oneshot_json(_skipped_entry("morning_run")))
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        aug.augment(_trace(self._EV), [task], _llm_config())
        prompt = pipeline.calls[0]
        assert "Morning Run" in prompt
        assert "May" in prompt or "2026" in prompt

    def test_prompt_includes_waking_window(self):
        pipeline = _SeqPipeline(_oneshot_json(_skipped_entry("yoga")))
        aug = _make_augmenter(
            pipeline,
            daily_window=DailyWindow(wake_minutes=420, sleep_minutes=1380),
        )
        aug.augment(_trace(self._EV), [make_task("yoga")], _llm_config())
        prompt = pipeline.calls[0]
        assert "07:00" in prompt
        assert "23:00" in prompt

    def test_same_label_disambiguated_by_index(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        t1 = make_task("active_routine", duration_min=10, duration_max=20)
        t2 = make_task("active_routine", duration_min=10, duration_max=20)
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Active Routine",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "08:00",
                    "end": "08:15",
                },
                {
                    "task_index": 2,
                    "task_title": "Active Routine",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "10:00",
                    "end": "10:15",
                },
            ]
        )
        aug = _make_augmenter(_SeqPipeline(raw), daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [t1, t2], _llm_config())
        assert len(sol.scheduled) == 2

    def test_same_label_disambiguated_by_label_queue(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        t1 = make_task("active_routine", duration_min=10, duration_max=20)
        t2 = make_task("active_routine", duration_min=10, duration_max=20)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry("active_routine", start="08:00", end="08:15"),
                _placed_entry("active_routine", start="10:00", end="10:15"),
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [t1, t2], _llm_config())
        assert len(sol.scheduled) == 2

    # --- Daily window enforcement ---------------------------------------

    def test_placement_before_wake_rejected(self):
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        task = make_task("yoga", duration_min=10, duration_max=20)
        raw = _oneshot_json(_placed_entry("yoga", start="04:34", end="04:44"))
        aug = _make_augmenter(_SeqPipeline(raw))  # default 06:00–22:00
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert sol.scheduled == []

    def test_placement_after_sleep_rejected(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=300, date=DATE)
        task = make_task("yoga", duration_min=10, duration_max=20)
        raw = _oneshot_json(_placed_entry("yoga", start="22:30", end="22:45"))
        aug = _make_augmenter(_SeqPipeline(raw))
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert sol.scheduled == []

    def test_placement_inside_window_accepted(self):
        ev = make_event("sleep", start_minutes=0, end_minutes=300, date=DATE)
        task = make_task("yoga", duration_min=10, duration_max=20)
        raw = _oneshot_json(_placed_entry("yoga", start="08:00", end="08:15"))
        aug = _make_augmenter(_SeqPipeline(raw))
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert len(sol.scheduled) == 1


# ---------------------------------------------------------------------------
# Per-persona debug logging; augment() integration
# ---------------------------------------------------------------------------


class TestAugmentPropagatesPersonId:
    """End-to-end check that `LLMAugmenter.augment` threads
    `calendar.person_id` to `_maybe_dump_call` so debug dumps land
    in a per-persona sub-directory."""

    def test_dump_path_contains_person_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        ev = make_event("sleep", start_minutes=0, end_minutes=300, date=DATE)
        task = make_task("yoga", duration_min=10, duration_max=20)
        raw = _oneshot_json(_placed_entry("yoga", start="08:00", end="08:15"))
        aug = _make_augmenter(_SeqPipeline(raw))
        aug.augment(_trace(ev), [task], _llm_config())
        person_dir = tmp_path / "p001"
        assert person_dir.is_dir()
        files = list(person_dir.iterdir())
        assert len(files) >= 1
        body = files[0].read_text(encoding="utf-8")
        assert "p001" in body

    def test_two_traces_dump_to_separate_dirs(self, tmp_path, monkeypatch):
        """Augmenting two different person-ids in succession produces
        two distinct sub-directories; neither overwrites the other."""
        monkeypatch.setenv("LLM_AGENT_DEBUG_DIR", str(tmp_path))
        ev = make_event("sleep", start_minutes=0, end_minutes=300, date=DATE)
        task = make_task("yoga", duration_min=10, duration_max=20)

        for pid in ("alice", "bob"):
            pipeline = _SeqPipeline(
                _oneshot_json(_placed_entry("yoga", start="08:00", end="08:15"))
            )
            aug = _make_augmenter(pipeline)
            trace = CalendarTrace(person_id=pid, events=[ev])
            aug.augment(trace, [task], _llm_config())

        assert (tmp_path / "alice").is_dir()
        assert (tmp_path / "bob").is_dir()


# ---------------------------------------------------------------------------
# Two-week horizon
# ---------------------------------------------------------------------------


class TestRepeatPerWeek:
    def test_two_weeks_expand_task_instances(self):
        week2 = DATE + datetime.timedelta(days=7)
        ev1 = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        ev2 = make_event("sleep", start_minutes=1380, end_minutes=1440, date=week2)
        task = make_task("yoga", duration_min=30, duration_max=60)
        pipeline = _SeqPipeline(
            _oneshot_json(_placed_entry("yoga", start="08:00", end="08:30")),
            _oneshot_json(
                _placed_entry(
                    "yoga", date=week2.isoformat(), start="08:00", end="08:30"
                )
            ),
        )
        aug = _make_augmenter(pipeline)
        sol = aug.augment(_trace(ev1, ev2), [task], _llm_config(repeat_per_week=True))
        assert len(sol.tasks) == 2
        assert len(sol.scheduled) == 2

    def test_carry_forward_without_repeat(self):
        week2 = DATE + datetime.timedelta(days=7)
        week1_events = [
            make_event(
                f"fill_{i}",
                start_minutes=0,
                end_minutes=1440,
                date=DATE + datetime.timedelta(days=i),
            )
            for i in range(7)
        ]
        ev_w2 = make_event("partial", start_minutes=480, end_minutes=1440, date=week2)
        task = make_task("yoga", duration_min=60, duration_max=60)
        pipeline = _SeqPipeline(
            _oneshot_json(_skipped_entry("yoga")),
            _oneshot_json(
                _placed_entry(
                    "yoga", date=week2.isoformat(), start="06:30", end="07:30"
                )
            ),
        )
        aug = _make_augmenter(pipeline)
        sol = aug.augment(
            _trace(*week1_events, ev_w2),
            [task],
            _llm_config(repeat_per_week=False),
        )
        assert len(sol.scheduled) == 1
        assert sol.scheduled[0].date == week2

    def test_repeat_week1_unscheduled_week2_fresh_attempt(self):
        week2 = DATE + datetime.timedelta(days=7)
        week1_events = [
            make_event(
                f"fill_{i}",
                start_minutes=0,
                end_minutes=1440,
                date=DATE + datetime.timedelta(days=i),
            )
            for i in range(7)
        ]
        ev_w2 = make_event("partial", start_minutes=480, end_minutes=1440, date=week2)
        task = make_task("yoga", duration_min=60, duration_max=60)
        pipeline = _SeqPipeline(
            _oneshot_json(_skipped_entry("yoga")),
            _oneshot_json(
                _placed_entry(
                    "yoga", date=week2.isoformat(), start="06:30", end="07:30"
                )
            ),
        )
        aug = _make_augmenter(pipeline)
        sol = aug.augment(
            _trace(*week1_events, ev_w2),
            [task],
            _llm_config(repeat_per_week=True),
        )
        assert len(sol.tasks) == 2
        assert len(sol.scheduled) == 1
        assert len(sol.unscheduled) == 1


# ---------------------------------------------------------------------------
# _validate_decision (window + duration + dispatch)
# ---------------------------------------------------------------------------


class TestValidateDecision:
    def _aug(self, daily_window=None):
        return LLMAugmenter(llm_pipeline=None, daily_window=daily_window)

    def _decision(self, **kw) -> _ScheduleDecision:
        defaults = dict(
            task_label="yoga",
            scheduled=True,
            date=DATE,
            start_minutes=480,
            end_minutes=540,
            concurrent_flag=False,
            concurrent_with=None,
        )
        defaults.update(kw)
        return _ScheduleDecision(**defaults)

    def test_date_not_in_week_returns_error(self):
        d = self._decision(date=datetime.date(1999, 1, 1))
        task = make_task("yoga", duration_min=30, duration_max=120)
        err = self._aug(_wide_window())._validate_decision(
            d, task, [DATE], {}, {}, set(), {}
        )
        assert err is not None

    def test_window_violation_before_wake_returns_error(self):
        d = self._decision(start_minutes=300, end_minutes=320)
        task = make_task("yoga", duration_min=10, duration_max=60)
        err = self._aug()._validate_decision(d, task, [DATE], {}, {}, set(), {})
        assert err is not None and "outside daily window" in err

    def test_window_violation_after_sleep_returns_error(self):
        d = self._decision(start_minutes=1330, end_minutes=1380)
        task = make_task("yoga", duration_min=10, duration_max=60)
        err = self._aug()._validate_decision(d, task, [DATE], {}, {}, set(), {})
        assert err is not None and "outside daily window" in err

    def test_duration_too_short_returns_error(self):
        d = self._decision(start_minutes=480, end_minutes=490)
        task = make_task("yoga", duration_min=30, duration_max=120)
        err = self._aug(_wide_window())._validate_decision(
            d, task, [DATE], {}, {}, set(), {}
        )
        assert err is not None and "min" in err

    def test_duration_too_long_returns_error(self):
        d = self._decision(start_minutes=480, end_minutes=600)
        task = make_task("yoga", duration_min=10, duration_max=30)
        err = self._aug(_wide_window())._validate_decision(
            d, task, [DATE], {}, {}, set(), {}
        )
        assert err is not None and "max" in err

    def test_valid_standalone_returns_none(self):
        d = self._decision(start_minutes=480, end_minutes=540)
        task = make_task("yoga", duration_min=30, duration_max=120)
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        err = self._aug()._validate_decision(
            d, task, [DATE], {DATE: [ev]}, {}, set(), {}
        )
        assert err is None


# ---------------------------------------------------------------------------
# _validate_concurrent (no semantic check; containment check)
# ---------------------------------------------------------------------------


class TestValidateConcurrent:
    def _aug(self):
        return LLMAugmenter(llm_pipeline=None, daily_window=_wide_window())

    def _decision(self, **kw) -> _ScheduleDecision:
        defaults = dict(
            task_label="any",
            scheduled=True,
            date=DATE,
            start_minutes=720,
            end_minutes=740,
            concurrent_flag=True,
            concurrent_with="lunch",
        )
        defaults.update(kw)
        return _ScheduleDecision(**defaults)

    def _concurrent_host(self, label: str = "lunch", start: int = 720, end: int = 780):
        return make_event(
            label,
            start_minutes=start,
            end_minutes=end,
            date=DATE,
            is_concurrent=True,
        )

    def test_null_concurrent_with_returns_error(self):
        d = self._decision(concurrent_with=None)
        err = self._aug()._validate_concurrent(d, make_task("x"), set(), {})
        assert err is not None

    def test_event_not_found_returns_error(self):
        d = self._decision(concurrent_with="nonexistent")
        err = self._aug()._validate_concurrent(d, make_task("x"), set(), {})
        assert err is not None

    def test_host_not_concurrent_returns_error(self):
        """A busy event with `is_concurrent=False` cannot host any
        concurrent task.  The LLM never sees this flag; the validator
        enforces it post-hoc."""
        d = self._decision()
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=False,
        )
        err = self._aug()._validate_concurrent(d, make_task("x"), set(), {DATE: [ev]})
        assert err is not None
        assert "is_concurrent=False" in err
        assert "lunch" in err

    def test_host_not_concurrent_check_runs_before_consumed(self):
        """The `is_concurrent` check short-circuits before the
        consumed-host check, so the error message is the more
        informative one."""
        d = self._decision()
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=False,
        )
        err = self._aug()._validate_concurrent(
            d, make_task("x"), {(DATE, "lunch")}, {DATE: [ev]}
        )
        assert err is not None and "is_concurrent=False" in err

    def test_already_consumed_returns_error(self):
        d = self._decision()
        ev = self._concurrent_host()
        err = self._aug()._validate_concurrent(
            d, make_task("x"), {(DATE, "lunch")}, {DATE: [ev]}
        )
        assert err is not None and "already" in err

    def test_outside_host_interval_returns_error(self):
        d = self._decision(start_minutes=802, end_minutes=817)
        ev = self._concurrent_host()
        err = self._aug()._validate_concurrent(d, make_task("x"), set(), {DATE: [ev]})
        assert err is not None and "not contained" in err

    def test_valid_concurrent_returns_none(self):
        """Regression: a properly-contained placement against a concurrent
        host is accepted."""
        d = self._decision()
        ev = self._concurrent_host()
        err = self._aug()._validate_concurrent(d, make_task("x"), set(), {DATE: [ev]})
        assert err is None


# ---------------------------------------------------------------------------
# _validate_standalone
# ---------------------------------------------------------------------------


class TestValidateStandalone:
    def _aug(self):
        return LLMAugmenter(llm_pipeline=None, daily_window=_wide_window())

    def _decision(self, start: int = 0, end: int = 60) -> _ScheduleDecision:
        return _ScheduleDecision(
            task_label="yoga",
            scheduled=True,
            date=DATE,
            start_minutes=start,
            end_minutes=end,
            concurrent_flag=False,
            concurrent_with=None,
        )

    def test_allen_conflict_with_event_returns_error(self):
        task = make_task("yoga", is_concurrent=False)
        ev = make_event("sleep", start_minutes=0, end_minutes=1440, date=DATE)
        err = self._aug()._validate_standalone(
            self._decision(start=480, end=540), task, {DATE: [ev]}, {}, {}
        )
        assert err is not None

    def test_placed_overlap_returns_error(self):
        task = make_task("yoga", is_concurrent=False)
        err = self._aug()._validate_standalone(
            self._decision(start=480, end=540), task, {}, {DATE: [(480, 540)]}, {}
        )
        assert err is not None

    def test_concurrent_task_can_overlap_placed(self):
        task = make_task("podcast", is_concurrent=True)
        err = self._aug()._validate_standalone(
            self._decision(start=480, end=540), task, {}, {DATE: [(480, 540)]}, {}
        )
        assert err is None

    def test_no_conflict_returns_none(self):
        task = make_task("yoga", is_concurrent=False)
        ev = make_event("sleep", start_minutes=1380, end_minutes=1440, date=DATE)
        err = self._aug()._validate_standalone(
            self._decision(start=0, end=60), task, {DATE: [ev]}, {}, {}
        )
        assert err is None


# ---------------------------------------------------------------------------
# _build_oneshot_prompt
# ---------------------------------------------------------------------------


class TestBuildOneshotPrompt:
    def _aug(self):
        return LLMAugmenter(llm_pipeline=None, daily_window=_wide_window())

    def test_prompt_contains_task_titles(self):
        tasks = [make_task("yoga"), make_task("walk")]
        prompt = self._aug()._build_oneshot_prompt(
            tasks, [DATE], {}, None, _llm_config()
        )
        assert "Yoga" in prompt and "Walk" in prompt

    def test_prompt_contains_human_dates(self):
        prompt = self._aug()._build_oneshot_prompt(
            [make_task("yoga")], [DATE], {}, None, _llm_config()
        )
        assert "May" in prompt or "2026" in prompt

    def test_prompt_contains_num_tasks(self):
        tasks = [make_task(f"t{i}") for i in range(4)]
        prompt = self._aug()._build_oneshot_prompt(
            tasks, [DATE], {}, None, _llm_config()
        )
        assert "4" in prompt

    def test_prompt_omits_accepts_concurrent_for_concurrent_event(self):
        """The benchmark-internal `is_concurrent` flag must NOT leak into
        the rendered prompt; the LLM has to judge concurrency from titles
        alone."""
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
        )
        prompt = self._aug()._build_oneshot_prompt(
            [make_task("yoga")], [DATE], {DATE: [ev]}, None, _llm_config()
        )
        assert "accepts_concurrent" not in prompt

    def test_prompt_includes_self_verify_block(self):
        """The self-verify guard must appear in every rendered prompt
        so the LLM is reminded to re-check numeric constraints."""
        prompt = self._aug()._build_oneshot_prompt(
            [make_task("yoga")], [DATE], {}, None, _llm_config()
        )
        assert "BEFORE finalizing" in prompt


# ---------------------------------------------------------------------------
# _find_host_event
# ---------------------------------------------------------------------------


class TestFindHostEvent:
    def test_label_none_returns_none(self):
        ev = make_event("lunch", date=DATE)
        assert _find_host_event(None, DATE, {DATE: [ev]}) is None

    def test_date_none_returns_none(self):
        ev = make_event("lunch", date=DATE)
        assert _find_host_event("lunch", None, {DATE: [ev]}) is None

    def test_no_events_for_date_returns_none(self):
        assert _find_host_event("lunch", DATE, {}) is None

    def test_no_label_match_returns_none(self):
        ev = make_event("lunch", date=DATE)
        assert _find_host_event("dinner", DATE, {DATE: [ev]}) is None

    def test_label_match_returns_event(self):
        ev = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        result = _find_host_event("lunch", DATE, {DATE: [ev]})
        assert result is ev

    def test_first_match_wins_when_duplicates(self):
        ev1 = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        ev2 = make_event("lunch", start_minutes=800, end_minutes=860, date=DATE)
        result = _find_host_event("lunch", DATE, {DATE: [ev1, ev2]})
        assert result is ev1


# ---------------------------------------------------------------------------
# Concurrent clip-rescue (deterministic post-LLM clip pass on partial overlaps)
# ---------------------------------------------------------------------------


class TestConcurrentClipRescue:
    """The clip-rescue pass runs in `_plan_week` after the LLM responds
    and before `_validate_decision`.  When a concurrent placement
    partially overlaps its named host, the start/end are clipped back
    into the host's interval; but only when the resulting duration
    still satisfies the task's `[duration_min, duration_max]` band.
    No retry LLM call is made.
    """

    @staticmethod
    def _host(start: int = 720, end: int = 780):
        """Return a concurrent-eligible host event."""
        return make_event(
            "lunch",
            start_minutes=start,
            end_minutes=end,
            date=DATE,
            is_concurrent=True,
        )

    @staticmethod
    def _config():
        return _llm_config()

    def _run(self, host, task, *, start: str, end: str):
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry(
                    task.label,
                    start=start,
                    end=end,
                    concurrent_flag=True,
                    concurrent_with="lunch",
                )
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        return aug.augment(_trace(host), [task], self._config())

    def test_already_contained_unchanged(self):
        """Regression: a fully-contained placement is accepted with start/end
        unchanged."""
        host = self._host(start=1160, end=1190)  # 19:20–19:50
        task = make_task("eat_protein", duration_min=10, duration_max=30)
        sol = self._run(host, task, start="19:20", end="19:50")
        assert len(sol.scheduled) == 1
        st = sol.scheduled[0]
        assert (st.start_minutes, st.end_minutes) == (1160, 1190)

    def test_pre_host_overlap_clipped_and_accepted(self):
        """LLM's start is BEFORE the host's start; clip the start forward."""
        host = self._host(start=1160, end=1190)  # 19:20–19:50
        task = make_task("eat_protein", duration_min=10, duration_max=30)
        sol = self._run(host, task, start="19:10", end="19:30")
        assert len(sol.scheduled) == 1
        st = sol.scheduled[0]
        assert (st.start_minutes, st.end_minutes) == (1160, 1170)  # 19:20–19:30

    def test_post_host_overlap_clipped_and_accepted(self):
        """LLM's end is AFTER the host's end; clip the end backward."""
        host = self._host(start=1160, end=1190)  # 19:20–19:50
        task = make_task("eat_protein", duration_min=10, duration_max=30)
        sol = self._run(host, task, start="19:40", end="20:10")
        assert len(sol.scheduled) == 1
        st = sol.scheduled[0]
        assert (st.start_minutes, st.end_minutes) == (1180, 1190)  # 19:40–19:50

    def test_zero_overlap_at_host_end_not_clipped(self):
        """Zero-overlap (decision starts at host.end) collapses to zero
        duration after clipping to leave decision unchanged to validator
        rejects on containment."""
        host = self._host(start=1160, end=1190)  # 19:20–19:50
        task = make_task("eat_protein", duration_min=10, duration_max=30)
        sol = self._run(host, task, start="19:50", end="20:20")
        assert sol.scheduled == []

    def test_clipped_duration_below_min_not_clipped(self):
        """If the clipped interval is shorter than the task's duration_min,
        leave the decision alone (validator rejects on duration)."""
        host = self._host(start=1160, end=1190)  # 19:20–19:50
        task = make_task("longer_block", duration_min=25, duration_max=40)
        # LLM proposes 19:25–19:45.  Clip to 19:25–19:45 = 20 min < 25.  No clip
        # applies and the original 19:25–19:45 is then duration-rejected by the
        # validator (20 min < duration_min=25).
        sol = self._run(host, task, start="19:25", end="19:45")
        assert sol.scheduled == []

    def test_clipped_duration_above_max_not_clipped(self):
        """If the clipped interval would exceed duration_max, leave the
        decision unchanged (validator rejects on duration)."""
        host = self._host(start=1160, end=1280)  # 19:20–21:20 (long host)
        task = make_task("short_block", duration_min=10, duration_max=20)
        sol = self._run(host, task, start="19:00", end="20:30")
        # Clip would produce 19:20–20:30 = 70 min > 20 to no clip to containment
        # rejects 19:00 < 19:20.
        assert sol.scheduled == []

    def test_pre_host_overlap_clip_below_min_rejected(self):
        """A pre-host overlap whose clipped duration falls below duration_min
        is left alone and the validator rejects on containment."""
        host = self._host(start=1160, end=1190)  # 19:20–19:50
        task = make_task("eat_protein", duration_min=20, duration_max=30)
        sol = self._run(host, task, start="19:15", end="19:30")
        # Clip to 19:20–19:30 = 10 < 20 to no clip to containment rejects.
        assert sol.scheduled == []

    def test_clip_does_not_apply_when_host_not_concurrent(self):
        """If the host has `is_concurrent=False` the validator rejects
        regardless, even when the geometry would otherwise rescue the
        placement."""
        host = make_event(
            "lunch",
            start_minutes=1160,
            end_minutes=1190,
            date=DATE,
            is_concurrent=False,
        )
        task = make_task("eat_protein", duration_min=10, duration_max=30)
        sol = self._run(host, task, start="19:10", end="19:30")
        assert sol.scheduled == []


# ---------------------------------------------------------------------------
# Top-level integration: `host.is_concurrent` guard wired through augment()
# ---------------------------------------------------------------------------


class TestAugmentHostConcurrencyGuard:
    def test_concurrent_against_non_concurrent_host_rejected(self):
        """End-to-end: an LLM concurrent placement against a host with
        `is_concurrent=False` is rejected by `augment`."""
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=False,
        )
        task = make_task("anything", duration_min=10, duration_max=30)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry(
                    "anything",
                    start="12:00",
                    end="12:20",
                    concurrent_flag=True,
                    concurrent_with="lunch",
                )
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert sol.scheduled == []

    def test_concurrent_against_concurrent_host_accepted(self):
        """The same placement against a host with `is_concurrent=True`
        is accepted by `augment`."""
        ev = make_event(
            "lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            is_concurrent=True,
        )
        task = make_task("anything", duration_min=10, duration_max=30)
        pipeline = _SeqPipeline(
            _oneshot_json(
                _placed_entry(
                    "anything",
                    start="12:00",
                    end="12:20",
                    concurrent_flag=True,
                    concurrent_with="lunch",
                )
            )
        )
        aug = _make_augmenter(pipeline, daily_window=_wide_window())
        sol = aug.augment(_trace(ev), [task], _llm_config())
        assert len(sol.scheduled) == 1


# ---------------------------------------------------------------------------
# DirectLLMPipeline; GraphRAG/augmenter separation of concerns
# (added 2026-05-14 after the LLM-agent latency investigation).
# ---------------------------------------------------------------------------


class TestDirectLLMPipeline:
    """The augmenter must not pay for GraphRAG retrieval at augment
    time; placement does not need ontology citations.  The
    :class: adapter exposes the same
    search(query_text) API as neo4j_graphrag.GraphRAG but
    bypasses retrieval entirely, so the augmenter call site is
    untouched while the wire-cost drops by an embedding API call,
    a Neo4j round-trip, and ~500-2000 wasted input tokens per
    week-call.
    """

    def test_search_invokes_llm_with_prompt_verbatim(self):
        """The prompt the augmenter builds reaches the LLM unchanged
        ; no retrieval prefix, no citation template, no re-wrap."""
        captured = []

        class _FakeLLM:
            def invoke(self, prompt):
                captured.append(prompt)
                response = MagicMock()
                response.content = '[{"task_index": 1, "scheduled": false}]'
                return response

        pl = DirectLLMPipeline(_FakeLLM())
        result = pl.search(query_text="schedule these tasks please")
        assert captured == ["schedule these tasks please"]
        assert result.answer == '[{"task_index": 1, "scheduled": false}]'

    def test_search_returns_response_with_answer_field(self):
        """The return shape matches what LLMAugmenter reads
        (response.answer) so the augmenter call site is identical
        for direct and retrieval-augmented pipelines."""
        fake = MagicMock()
        fake.invoke.return_value.content = "hello"
        pl = DirectLLMPipeline(fake)
        result = pl.search("anything")
        assert hasattr(result, "answer")
        assert result.answer == "hello"

    def test_missing_content_attribute_returns_empty_string(self):
        """Defensive: an LLM response object without .content
        produces an empty answer rather than None; keeps the
        augmenter's or "" fallback unused (clearer failure modes)."""
        fake = MagicMock()
        fake.invoke.return_value = object()
        pl = DirectLLMPipeline(fake)
        assert pl.search("x").answer == ""

    def test_none_content_returns_empty_string(self):
        """Defensive: response.content = None (some LLM wrappers
        do this on filter rejections) coerces to "" instead of
        propagating None into the JSON parser."""
        fake = MagicMock()
        fake.invoke.return_value.content = None
        pl = DirectLLMPipeline(fake)
        assert pl.search("x").answer == ""

    def test_direct_response_is_frozen(self):
        """_DirectResponse is frozen so the augmenter cannot
        accidentally mutate the response after parsing."""
        import dataclasses

        r = _DirectResponse(answer="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.answer = "y"


# ---------------------------------------------------------------------------
# LLMCallRecorder + DirectLLMPipeline integration
# ---------------------------------------------------------------------------


class _LLMStubResponse:
    """`neo4j_graphrag` LLM responses expose `.content`;
    `_MockResponse` (used elsewhere in this file) exposes `.answer`
    because it's the GraphRAG-pipeline shape one layer up."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StubLLM:
    """Minimal stand-in for `neo4j_graphrag.llm.LLMInterface`."""

    def __init__(self, content: str = "ok") -> None:
        self._content = content
        self.invocations: list[str] = []

    def invoke(self, prompt: str):
        self.invocations.append(prompt)
        return _LLMStubResponse(self._content)


class TestLLMCallRecorder:
    def test_default_state_is_empty(self):
        r = LLMCallRecorder()
        assert r.calls == []

    def test_record_appends_tuple(self):
        r = LLMCallRecorder()
        r.record(input_tokens=120, output_tokens=30, wall_time_seconds=0.7)
        assert r.calls == [(120, 30, 0.7, 0)]

    def test_repeated_record_accumulates(self):
        r = LLMCallRecorder()
        r.record(input_tokens=10, output_tokens=2, wall_time_seconds=0.1)
        r.record(input_tokens=20, output_tokens=4, wall_time_seconds=0.2)
        assert len(r.calls) == 2

    def test_reset_clears_calls(self):
        r = LLMCallRecorder()
        r.record(input_tokens=10, output_tokens=2, wall_time_seconds=0.1)
        r.reset()
        assert r.calls == []

    def test_record_coerces_floats_and_ints(self):
        """The recorder normalises inputs so a downstream consumer can
        rely on the tuple shape (int, int, float) regardless of caller."""
        r = LLMCallRecorder()
        r.record(input_tokens=10.0, output_tokens=2.7, wall_time_seconds=1)
        in_tok, out_tok, wall, reasoning = r.calls[0]
        assert isinstance(reasoning, int) and reasoning == 0
        assert isinstance(in_tok, int) and in_tok == 10
        assert isinstance(out_tok, int) and out_tok == 2
        assert isinstance(wall, float) and wall == 1.0


class TestDirectLLMPipelineRecorderIntegration:
    def test_no_recorder_runs_silently(self):
        """Backwards-compat: an unwired pipeline still serves the
        augmenter without crashing on missing recorder."""
        llm = _StubLLM(content="hello")
        pipe = DirectLLMPipeline(llm)
        out = pipe.search(query_text="prompt-text")
        assert out.answer == "hello"
        assert llm.invocations == ["prompt-text"]

    def test_recorder_captures_one_entry_per_call(self):
        llm = _StubLLM(content="hello")
        recorder = LLMCallRecorder()
        pipe = DirectLLMPipeline(llm, recorder=recorder)
        pipe.search(query_text="first")
        pipe.search(query_text="second")
        assert len(recorder.calls) == 2

    def test_recorder_token_estimate_grows_with_prompt_length(self):
        """Token estimate is best-effort via `estimate_tokens`; a
        much longer prompt must produce a strictly larger input_tokens."""
        llm = _StubLLM(content="response")
        recorder = LLMCallRecorder()
        pipe = DirectLLMPipeline(llm, recorder=recorder)
        pipe.search(query_text="x")
        pipe.search(query_text="x" * 5000)
        in_tok_short, _, _, _ = recorder.calls[0]
        in_tok_long, _, _, _ = recorder.calls[1]
        assert in_tok_long > in_tok_short

    def test_recorder_records_non_zero_wall_time(self):
        llm = _StubLLM()
        recorder = LLMCallRecorder()
        pipe = DirectLLMPipeline(llm, recorder=recorder)
        pipe.search(query_text="anything")
        _, _, wall, _ = recorder.calls[0]
        assert wall >= 0.0

    def test_response_with_no_content_attr_returns_empty_answer(self):
        """If the LLM returns an object without `.content` (defensive
        path), the pipeline still surfaces an empty-string answer and
        records the round-trip with output_tokens=0."""

        class _NoContent:
            pass

        class _Quirky:
            def invoke(self, prompt):
                return _NoContent()

        recorder = LLMCallRecorder()
        pipe = DirectLLMPipeline(_Quirky(), recorder=recorder)
        out = pipe.search(query_text="prompt")
        assert out.answer == ""
        _, out_tok, _, _ = recorder.calls[0]
        assert out_tok == 0

    def test_non_dict_usage_metadata_falls_back_to_estimates(self):
        """A non-dict `usage_metadata` is ignored and tokens are estimated."""

        class _BadUsage:
            content = "response text"
            usage_metadata = "not a dict"

        class _Quirky:
            def invoke(self, prompt):
                return _BadUsage()

        recorder = LLMCallRecorder()
        pipe = DirectLLMPipeline(_Quirky(), recorder=recorder)
        out = pipe.search(query_text="a prompt")
        assert out.answer == "response text"
        in_tok, out_tok, _, reasoning = recorder.calls[0]
        assert in_tok > 0 and out_tok > 0
        assert reasoning == 0

    def test_dict_usage_metadata_is_recorded_verbatim(self):
        """A dict `usage_metadata` supplies the recorded token counts directly."""

        class _Usage:
            content = "ok"
            usage_metadata = {
                "input_tokens": 17,
                "output_tokens": 4,
                "reasoning_tokens": 2,
            }

        class _Quirky:
            def invoke(self, prompt):
                return _Usage()

        recorder = LLMCallRecorder()
        pipe = DirectLLMPipeline(_Quirky(), recorder=recorder)
        pipe.search(query_text="a prompt")
        in_tok, out_tok, _, reasoning = recorder.calls[0]
        assert (in_tok, out_tok, reasoning) == (17, 4, 2)
