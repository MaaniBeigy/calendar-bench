"""Tests for the `split_dividable_tasks` prompt block, parser path,
and divide-piece planning in `LLMAugmenter`.

Coverage targets:

* `augment_oneshot.j2` `split_dividable_tasks` block content.
* `augment_oneshot_placebos.DEFAULT_PLACEBOS` covers the new block.
* `ABLATABLE_BLOCKS` lists the new block.
* `_build_tasks_list` exposes `dividable_ok`.
* `_parse_oneshot_response` parses `pieces` arrays into
  :class:`_PieceDecision` rows.
* `LLMAugmenter._plan_week` emits one `ScheduledTask` per piece with
  `parent_task_label` set.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

import pytest

from src.scripts.persona.config.schema import DailyWindow
from src.scripts.scenarios.augmentation.llm_agent import (
    LLMAugmenter,
    _build_tasks_list,
    _coerce_pieces_to_single,
    _parse_oneshot_response,
    _PieceDecision,
    _ScheduleDecision,
)
from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
    ABLATABLE_BLOCKS,
    default_placebo_for,
    render,
)
from src.scripts.scenarios.augmentation.prompts.augment_oneshot_placebos import (
    DEFAULT_PLACEBOS,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, LLMAgentConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

DATE_A = datetime.date(2026, 5, 4)
DATE_B = datetime.date(2026, 5, 6)
DATE_C = datetime.date(2026, 5, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oneshot_ctx(**overrides) -> dict:
    base = {
        "week_start": "Monday 04 May 2026",
        "week_end": "Sunday 10 May 2026",
        "wake_start": "06:00",
        "sleep_start": "22:00",
        "calendar_summary": "<<CAL>>",
        "num_tasks": 1,
        "tasks_list": "<<TASKS>>",
    }
    base.update(overrides)
    return base


def _wide_window() -> DailyWindow:
    return DailyWindow(wake_minutes=0, sleep_minutes=1440)


def _llm_config() -> AugmentationConfig:
    return AugmentationConfig(
        method="llm_agent",
        allow_merge=True,
        merge_threshold=0.70,
        repeat_per_week=True,
        llm_agent=LLMAgentConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_retries=1,
            prompt_template="augment_oneshot",
        ),
    )


class _MockResponse:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _ConstPipeline:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[str] = []

    def search(self, query_text: str, **_: Any) -> _MockResponse:
        self.calls.append(query_text)
        return _MockResponse(self.answer)


# ---------------------------------------------------------------------------
# ABLATABLE_BLOCKS registration
# ---------------------------------------------------------------------------


class TestAblatableBlocksRegistration:
    def test_split_dividable_tasks_is_registered(self):
        assert "split_dividable_tasks" in ABLATABLE_BLOCKS

    def test_registered_as_an_always_kept_pb12_column(self):
        """The block sits in the always-kept PB-12 tail (col 11), co-aliased with col 12 (`context_block`) under fold."""
        assert ABLATABLE_BLOCKS[-2] == "split_dividable_tasks"
        assert ABLATABLE_BLOCKS[-1] == "context_block"
        assert len(ABLATABLE_BLOCKS) == 13

    def test_default_placebo_exists(self):
        assert DEFAULT_PLACEBOS["split_dividable_tasks"].strip()

    def test_default_placebo_for_returns_placebo(self):
        assert (
            default_placebo_for("split_dividable_tasks")
            == DEFAULT_PLACEBOS["split_dividable_tasks"]
        )


# ---------------------------------------------------------------------------
# Prompt block content
# ---------------------------------------------------------------------------


class TestSplitDividableTasksBlock:
    def test_block_present_in_baseline_render(self):
        out = render(_oneshot_ctx())
        assert "DIVIDE-FRIENDLY TASKS" in out

    def test_block_names_dividable_ok_gate(self):
        out = render(_oneshot_ctx())
        assert "`dividable_ok: true`" in out
        assert "`dividable_ok == true` is the only gate" in out

    def test_block_recommends_two_or_more_pieces(self):
        out = render(_oneshot_ctx())
        normalised = " ".join(out.split())
        assert "TWO OR MORE shorter easier sibling sessions" in normalised

    def test_block_states_strict_less_than_duration_max(self):
        out = render(_oneshot_ctx())
        assert (
            "duration_min ≤ (piece.end − piece.start) < duration_max" in out
            or "STRICT upper bound" in out
        )
        assert "SHORTER than the" in out and "duration_max" in out

    def test_block_explains_sum_target(self):
        out = render(_oneshot_ctx())
        # Pieces of ONE weekly habit should TOGETHER sum toward the
        # FULL `duration_max`, not merely `duration_min`.
        assert "add up to about" in out
        assert "`duration_max`, not just `duration_min`" in out
        # The prompt qualitatively targets `duration_max`; it must
        # NOT leak the evaluator's tolerance window or any specific
        # pass/fail band (those are scoring constants, not task
        # clarification).
        assert "Aim the total near `duration_max`" in out

    def test_block_does_not_leak_evaluator_tolerance(self):
        """Regression guard: no evaluator threshold may surface in
        the prompt. Pinning the bans explicitly so a future edit
        cannot quietly reintroduce the leak."""
        out = render(_oneshot_ctx())
        forbidden = (
            "±15%",
            "15% tolerance",
            "tolerance band",
            "counts as a valid divide",
            "between 51 and 69",
            "divide_duration_tolerance_pct",
        )
        for needle in forbidden:
            assert (
                needle not in out
            ), f"leaked evaluator constant {needle!r} into the prompt"

    def test_block_describes_pieces_array_output_format(self):
        out = render(_oneshot_ctx())
        assert '"pieces"' in out
        assert "Walk 6,000 Steps" in out

    def test_block_substitutes_wake_and_sleep_into_per_piece_window(self):
        """The per-piece waking-window inequalities use the supplied
        wake/sleep values, not hardcoded literals."""
        out = render(_oneshot_ctx(wake_start="07:00", sleep_start="23:00"))
        assert "`07:00 ≤ piece.start`" in out
        assert "`piece.end ≤" in out and "23:00`" in out

    def test_block_marks_split_as_optional(self):
        out = render(_oneshot_ctx())
        normalised = " ".join(out.split())
        assert "Splitting is OPTIONAL" in normalised

    def test_block_is_individually_ablatable(self):
        baseline = render(_oneshot_ctx())
        ablated = render(_oneshot_ctx(), ablate=["split_dividable_tasks"])
        assert "DIVIDE-FRIENDLY TASKS" in baseline
        assert "DIVIDE-FRIENDLY TASKS" not in ablated

    def test_block_placebo_replacement(self):
        placebo = "Custom divide placebo text for tests."
        out = render(
            _oneshot_ctx(),
            ablate=["split_dividable_tasks"],
            placebos={"split_dividable_tasks": placebo},
        )
        assert placebo in out
        assert "DIVIDE-FRIENDLY TASKS" not in out


# ---------------------------------------------------------------------------
# Output schema example
# ---------------------------------------------------------------------------


class TestOutputSchemaExample:
    def test_schema_lists_three_entry_shapes(self):
        """The STEP 4 example shows single, pieces, and unscheduled."""
        out = render(_oneshot_ctx())
        normalised = " ".join(out.split())
        assert "single placement, split into pieces" in normalised

    def test_schema_example_contains_pieces_field(self):
        out = render(_oneshot_ctx())
        # The skeleton in STEP 4 shows the `pieces` shape.
        assert '"pieces": [' in out


# ---------------------------------------------------------------------------
# `_build_tasks_list` exposes dividable_ok
# ---------------------------------------------------------------------------


class TestTasksListExposesDividableOk:
    def test_dividable_true_renders_true(self):
        t = make_task("walk", is_dividable=True)
        out = _build_tasks_list([t])
        assert "dividable_ok      : true" in out

    def test_dividable_false_renders_false(self):
        t = make_task("walk", is_dividable=False)
        out = _build_tasks_list([t])
        assert "dividable_ok      : false" in out

    def test_concurrent_ok_still_present_alongside(self):
        """Adding `dividable_ok` does not displace `concurrent_ok`."""
        t = make_task("walk", is_dividable=True, is_concurrent=True)
        out = _build_tasks_list([t])
        assert "concurrent_ok     : true" in out
        assert "dividable_ok      : true" in out


# ---------------------------------------------------------------------------
# Parser: `pieces` array
# ---------------------------------------------------------------------------


class TestParserPiecesArray:
    def _entry_with_pieces(self, *pieces: dict) -> str:
        return json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk 6,000 Steps",
                    "scheduled": True,
                    "pieces": list(pieces),
                }
            ]
        )

    def test_two_pieces_parsed(self):
        raw = self._entry_with_pieces(
            {
                "date": "2026-05-04",
                "start": "07:30",
                "end": "07:50",
                "concurrent_with_event": None,
            },
            {
                "date": "2026-05-06",
                "start": "18:00",
                "end": "18:20",
                "concurrent_with_event": None,
            },
        )
        decisions = _parse_oneshot_response(raw)
        assert decisions is not None
        assert len(decisions) == 1
        d = decisions[0]
        assert d.scheduled is True
        assert d.pieces is not None
        assert len(d.pieces) == 2
        assert d.pieces[0].date == DATE_A
        assert d.pieces[0].start_minutes == 7 * 60 + 30
        assert d.pieces[0].end_minutes == 7 * 60 + 50
        assert d.pieces[1].date == DATE_B
        assert d.pieces[1].concurrent_with is None

    def test_piece_with_concurrent_host_parsed(self):
        raw = self._entry_with_pieces(
            {
                "date": "2026-05-04",
                "start": "13:10",
                "end": "13:30",
                "concurrent_with_event": "Lunch",
            }
        )
        decisions = _parse_oneshot_response(raw)
        assert decisions and decisions[0].pieces is not None
        p = decisions[0].pieces[0]
        assert p.concurrent_with == "Lunch"

    def test_pieces_take_priority_over_top_level_date(self):
        """An entry carrying both `pieces` AND top-level
        `date`/`start`/`end` is treated as a split."""
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "08:00",
                    "end": "09:00",
                    "pieces": [
                        {
                            "date": "2026-05-04",
                            "start": "07:30",
                            "end": "07:45",
                        },
                        {
                            "date": "2026-05-06",
                            "start": "18:00",
                            "end": "18:15",
                        },
                    ],
                }
            ]
        )
        decisions = _parse_oneshot_response(raw)
        assert decisions and decisions[0].pieces is not None
        assert len(decisions[0].pieces) == 2
        # Top-level start/end ignored when pieces present.
        assert decisions[0].start_minutes == 0
        assert decisions[0].end_minutes == 0

    def test_empty_pieces_falls_back_to_single_placement(self):
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk",
                    "scheduled": True,
                    "date": "2026-05-04",
                    "start": "08:00",
                    "end": "08:30",
                    "pieces": [],
                }
            ]
        )
        decisions = _parse_oneshot_response(raw)
        assert decisions and decisions[0].pieces is None
        assert decisions[0].start_minutes == 8 * 60
        assert decisions[0].end_minutes == 8 * 60 + 30

    def test_malformed_piece_silently_dropped(self):
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk",
                    "scheduled": True,
                    "pieces": [
                        {
                            "date": "2026-05-04",
                            "start": "07:30",
                            "end": "07:50",
                        },
                        {"date": "not-a-date", "start": "08:00", "end": "08:20"},
                        "not-a-dict",
                    ],
                }
            ]
        )
        decisions = _parse_oneshot_response(raw)
        assert decisions and decisions[0].pieces is not None
        assert len(decisions[0].pieces) == 1


# ---------------------------------------------------------------------------
# `_validate_dividable` + `_coerce_pieces_to_single`
# ---------------------------------------------------------------------------


def _decision_with_pieces(*pieces: _PieceDecision) -> _ScheduleDecision:
    return _ScheduleDecision(
        task_label="walk",
        scheduled=True,
        task_index=1,
        pieces=list(pieces),
    )


class TestValidateDividable:
    def _piece(self, start: int = 480, end: int = 500) -> _PieceDecision:
        return _PieceDecision(date=DATE_A, start_minutes=start, end_minutes=end)

    def _augmenter(self) -> LLMAugmenter:
        return LLMAugmenter(llm_pipeline=None, daily_window=_wide_window())

    def test_returns_none_when_no_pieces(self):
        d = _ScheduleDecision(
            task_label="walk",
            scheduled=True,
            task_index=1,
            date=DATE_A,
            start_minutes=480,
            end_minutes=500,
        )
        task = make_task("walk", is_dividable=False)
        assert self._augmenter()._validate_dividable(d, task) is None

    def test_returns_none_for_dividable_task_with_pieces(self):
        d = _decision_with_pieces(self._piece(), self._piece(start=540, end=560))
        task = make_task("walk", is_dividable=True)
        assert self._augmenter()._validate_dividable(d, task) is None

    def test_returns_error_for_non_dividable_task_with_pieces(self):
        d = _decision_with_pieces(self._piece(), self._piece(start=540, end=560))
        task = make_task("walk", is_dividable=False)
        err = self._augmenter()._validate_dividable(d, task)
        assert err is not None
        assert "is_dividable=False" in err
        assert "2 piece" in err

    def test_error_message_names_the_task(self):
        d = _decision_with_pieces(self._piece())
        task = make_task("specific-task-label", is_dividable=False)
        err = self._augmenter()._validate_dividable(d, task)
        assert err is not None
        assert "specific-task-label" in err


class TestCoercePiecesToSingle:
    def test_returns_none_for_empty_pieces(self):
        d = _ScheduleDecision(
            task_label="walk", scheduled=True, task_index=1, pieces=[]
        )
        assert _coerce_pieces_to_single(d) is None

    def test_promotes_first_piece(self):
        d = _decision_with_pieces(
            _PieceDecision(date=DATE_A, start_minutes=480, end_minutes=500),
            _PieceDecision(date=DATE_B, start_minutes=1080, end_minutes=1100),
        )
        out = _coerce_pieces_to_single(d)
        assert out is not None
        assert out.pieces is None
        assert out.date == DATE_A
        assert out.start_minutes == 480
        assert out.end_minutes == 500
        assert out.concurrent_flag is False
        assert out.concurrent_with is None

    def test_preserves_concurrent_host_from_first_piece(self):
        d = _decision_with_pieces(
            _PieceDecision(
                date=DATE_A,
                start_minutes=780,
                end_minutes=800,
                concurrent_with="Lunch",
            )
        )
        out = _coerce_pieces_to_single(d)
        assert out is not None
        assert out.concurrent_flag is True
        assert out.concurrent_with == "Lunch"

    def test_preserves_task_index_and_label(self):
        d = _decision_with_pieces(
            _PieceDecision(date=DATE_A, start_minutes=480, end_minutes=500)
        )
        out = _coerce_pieces_to_single(d)
        assert out is not None
        assert out.task_index == d.task_index
        assert out.task_label == d.task_label


# ---------------------------------------------------------------------------
# Planning: pieces become ScheduledTasks with parent_task_label
# ---------------------------------------------------------------------------


class TestPlanningEmitsPieces:
    def _plan(self, response_json: str, task) -> list:
        pipeline = _ConstPipeline(response_json)
        augmenter = LLMAugmenter(
            llm_pipeline=pipeline,
            daily_window=_wide_window(),
        )
        # Single-week horizon Monday-Sunday so all DATE_A/B/C land in week.
        trace = CalendarTrace(person_id="p001", events=[])
        # Inject a horizon by stamping events on the first / last days
        # of the desired week. Tests rely on `_get_horizon_dates` which
        # derives the horizon from min/max event date plus 6-day span.
        trace_with_anchors = CalendarTrace(
            person_id="p001",
            events=[
                make_event("anchor_start", date=DATE_A, start_minutes=0, end_minutes=1),
                make_event(
                    "anchor_end",
                    date=DATE_A + datetime.timedelta(days=6),
                    start_minutes=0,
                    end_minutes=1,
                ),
            ],
        )
        solution = augmenter.augment(trace_with_anchors, [task], _llm_config())
        return solution.scheduled

    def test_two_pieces_emit_two_scheduled_with_parent_label(self):
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk 6,000 Steps",
                    "scheduled": True,
                    "pieces": [
                        {
                            "date": DATE_A.isoformat(),
                            "start": "07:30",
                            "end": "07:50",
                        },
                        {
                            "date": DATE_B.isoformat(),
                            "start": "18:00",
                            "end": "18:20",
                        },
                    ],
                }
            ]
        )
        task = make_task(
            "walk-6-000-steps",
            duration_min=15,
            duration_max=60,
            is_dividable=True,
        )
        scheduled = self._plan(raw, task)
        assert len(scheduled) == 2
        for st in scheduled:
            assert st.parent_task_label == "walk-6-000-steps"
            assert st.task is task

    def test_pieces_count_below_two_still_emit_when_valid(self):
        """The augmenter emits whatever pieces validate; the metric
        is responsible for scoring `<2` pieces as `not_divided`."""
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk",
                    "scheduled": True,
                    "pieces": [
                        {
                            "date": DATE_A.isoformat(),
                            "start": "07:30",
                            "end": "07:50",
                        }
                    ],
                }
            ]
        )
        task = make_task("walk", duration_min=15, duration_max=60, is_dividable=True)
        scheduled = self._plan(raw, task)
        assert len(scheduled) == 1
        assert scheduled[0].parent_task_label == "walk"

    def test_invalid_piece_durations_drop_those_pieces(self):
        """Pieces outside [duration_min, duration_max] are dropped."""
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk",
                    "scheduled": True,
                    "pieces": [
                        # 20-minute piece: valid.
                        {
                            "date": DATE_A.isoformat(),
                            "start": "07:30",
                            "end": "07:50",
                        },
                        # 5-minute piece: under duration_min=15.
                        {
                            "date": DATE_B.isoformat(),
                            "start": "18:00",
                            "end": "18:05",
                        },
                        # 90-minute piece: over duration_max=60.
                        {
                            "date": DATE_C.isoformat(),
                            "start": "10:00",
                            "end": "11:30",
                        },
                    ],
                }
            ]
        )
        task = make_task("walk", duration_min=15, duration_max=60, is_dividable=True)
        scheduled = self._plan(raw, task)
        assert len(scheduled) == 1
        assert scheduled[0].date == DATE_A

    def test_non_dividable_with_pieces_coerced_to_single(self):
        """A task with `is_dividable=False` that still came back with a
        `pieces` array is coerced to a single placement using the
        first piece. No `parent_task_label` is stamped (the metric
        must not interpret the result as a split)."""
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk",
                    "scheduled": True,
                    "pieces": [
                        {
                            "date": DATE_A.isoformat(),
                            "start": "07:30",
                            "end": "07:50",
                        },
                        {
                            "date": DATE_B.isoformat(),
                            "start": "18:00",
                            "end": "18:20",
                        },
                    ],
                }
            ]
        )
        task = make_task("walk", duration_min=15, duration_max=60, is_dividable=False)
        scheduled = self._plan(raw, task)
        assert len(scheduled) == 1
        st = scheduled[0]
        assert st.parent_task_label is None
        assert st.date == DATE_A
        assert st.start_minutes == 7 * 60 + 30
        assert st.end_minutes == 7 * 60 + 50

    def test_all_pieces_invalid_marks_task_unscheduled(self):
        raw = json.dumps(
            [
                {
                    "task_index": 1,
                    "task_title": "Walk",
                    "scheduled": True,
                    "pieces": [
                        # both pieces over duration_max
                        {
                            "date": DATE_A.isoformat(),
                            "start": "10:00",
                            "end": "11:30",
                        },
                        {
                            "date": DATE_B.isoformat(),
                            "start": "10:00",
                            "end": "11:30",
                        },
                    ],
                }
            ]
        )
        task = make_task("walk", duration_min=15, duration_max=60, is_dividable=True)
        pipeline = _ConstPipeline(raw)
        augmenter = LLMAugmenter(
            llm_pipeline=pipeline,
            daily_window=_wide_window(),
        )
        trace = CalendarTrace(
            person_id="p001",
            events=[
                make_event("anchor_start", date=DATE_A, start_minutes=0, end_minutes=1),
                make_event(
                    "anchor_end",
                    date=DATE_A + datetime.timedelta(days=6),
                    start_minutes=0,
                    end_minutes=1,
                ),
            ],
        )
        solution = augmenter.augment(trace, [task], _llm_config())
        assert solution.scheduled == []
        assert solution.unscheduled == [task]


# ---------------------------------------------------------------------------
# Piece-path edge cases inside the planning loop
# ---------------------------------------------------------------------------


def _augment_with_response(raw: str, task) -> "object":
    """Run a single-task augment against a stubbed pipeline returning `raw`."""
    pipeline = _ConstPipeline(raw)
    augmenter = LLMAugmenter(llm_pipeline=pipeline, daily_window=_wide_window())
    trace = CalendarTrace(
        person_id="p001",
        events=[
            make_event("anchor_start", date=DATE_A, start_minutes=0, end_minutes=1),
            make_event(
                "anchor_end",
                date=DATE_A + datetime.timedelta(days=6),
                start_minutes=0,
                end_minutes=1,
            ),
        ],
    )
    return augmenter.augment(trace, [task], _llm_config())


def test_non_dividable_task_with_empty_pieces_array_lands_unscheduled():
    """A non-dividable task returned with an empty pieces array falls to unscheduled."""
    raw = json.dumps(
        [
            {
                "task_index": 1,
                "task_title": "walk",
                "scheduled": True,
                "pieces": [],
            }
        ]
    )
    task = make_task("walk", duration_min=15, duration_max=60, is_dividable=False)
    solution = _augment_with_response(raw, task)
    assert solution.scheduled == []
    assert solution.unscheduled == [task]


def test_dividable_piece_with_concurrent_host_is_resolved_and_recorded():
    """A concurrent piece resolves its host label and records the week_concurrent entry."""
    lunch_event = make_event(
        "Lunch",
        date=DATE_A,
        start_minutes=720,
        end_minutes=765,
        is_concurrent=True,
        concurrent_with=["reading"],
    )
    raw = json.dumps(
        [
            {
                "task_index": 1,
                "task_title": "Reading",
                "scheduled": True,
                "pieces": [
                    {
                        "date": DATE_A.isoformat(),
                        "start": "12:00",
                        "end": "12:20",
                        "concurrent_with_event": "Lunch",
                    },
                    {
                        "date": DATE_B.isoformat(),
                        "start": "18:00",
                        "end": "18:20",
                    },
                ],
            }
        ]
    )
    task = make_task(
        "reading",
        duration_min=15,
        duration_max=60,
        is_dividable=True,
        is_concurrent=True,
    )
    pipeline = _ConstPipeline(raw)
    augmenter = LLMAugmenter(llm_pipeline=pipeline, daily_window=_wide_window())
    trace = CalendarTrace(
        person_id="p001",
        events=[
            lunch_event,
            make_event(
                "anchor_end",
                date=DATE_A + datetime.timedelta(days=6),
                start_minutes=0,
                end_minutes=1,
            ),
        ],
    )
    solution = augmenter.augment(trace, [task], _llm_config())
    by_date = {st.date: st for st in solution.scheduled}
    assert DATE_A in by_date
    concurrent_piece = by_date[DATE_A]
    assert concurrent_piece.is_standalone is False
    assert concurrent_piece.concurrent_with == "Lunch"


def test_dividable_piece_with_unknown_concurrent_host_is_dropped():
    """A concurrent piece whose host label is absent on that date is skipped."""
    raw = json.dumps(
        [
            {
                "task_index": 1,
                "task_title": "Reading",
                "scheduled": True,
                "pieces": [
                    {
                        "date": DATE_A.isoformat(),
                        "start": "12:00",
                        "end": "12:20",
                        "concurrent_with_event": "NoSuchEvent",
                    },
                    {
                        "date": DATE_B.isoformat(),
                        "start": "18:00",
                        "end": "18:20",
                    },
                ],
            }
        ]
    )
    task = make_task(
        "reading",
        duration_min=15,
        duration_max=60,
        is_dividable=True,
        is_concurrent=True,
    )
    solution = _augment_with_response(raw, task)
    dates = {st.date for st in solution.scheduled}
    assert DATE_A not in dates
    assert DATE_B in dates
