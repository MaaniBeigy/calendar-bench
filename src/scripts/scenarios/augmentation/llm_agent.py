"""LLM-agent calendar augmenter; one-shot weekly planning.

For each week the augmenter sends ALL the recommended tasks (with human-readable
titles, descriptions, durations, intensity, concurrency flag and preferred
epochs) plus the FULL week calendar (events shown by their natural display
title in a JSON-array timeline) and asks the LLM for a complete placement
plan.  The LLM returns a JSON array; one entry per task, indexed by
`task_index` (1-based).  No snake_case identifier is exposed to the model.

Validation is purely **structural**:

  • `[start, end]` lies inside the configured waking window.
  • `duration_min ≤ end − start ≤ duration_max`.
  • Standalone placement: Allen relations vs every base event on the date are
    in the admissible set; not-overlapping any task already placed (unless
    the task itself is concurrent-able).
  • Concurrent placement: the named host event exists on that date AND the
    task's `[start, end]` is **contained** inside the host's interval AND
    that host is not already used by another concurrent task.

Semantic compatibility (σ between task and event labels) is **not** checked
here; it is the job of `LLMJudgeOracle` at evaluation time to score the
quality of concurrent placements.  The augmenter trusts the LLM's reasoning
to pick a sensible host; bad picks are penalised by `L_concurrent` later.

When `config.repeat_per_week` is `True` (default) each week receives a
fresh copy of all task instances.  When `False` unplaced tasks carry
forward to the next week.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.scripts.persona.config.schema import AllenPairRule, DailyWindow
from src.scripts.scenarios.augmentation.base import Augmenter
from src.scripts.scenarios.augmentation.greedy import (
    _get_horizon_dates,
    _get_weekly_chunks,
    _tasks_for_week,
)
from src.scripts.scenarios.config.schema import AugmentationConfig, ObservationConfig
from src.scripts.scenarios.domain.calendar import (
    AugmentedCalendar,
    CalendarEvent,
    CalendarTrace,
)
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import (
    _build_rule_index,
    build_admissible_rx,
    compute_allen_relation,
    is_overlapping,
)
from src.scripts.scenarios.task_generation.prompt_templates import render

# ---------------------------------------------------------------------------
# Internal regexes
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"``(?:json)?[ \t]*([\s\S]*?)``", re.IGNORECASE)
_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_NORMALISE_RE = re.compile(r"[^a-z0-9]+")


def _event_display_title(label: str) -> str:
    """Render a snake_case calendar event label as a clean human-readable title.

    `office_work`  to `"Office Work"`
    `visit_family` to `"Visit Family"`

    This is a pure derivation from the snake_case `CalendarEvent.label` -
    the augmenter never modifies content, only renders it readably.
    """
    return label.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _hhmm_to_minutes(hhmm: str) -> int:
    """Convert `HH:MM` to integer minutes from midnight."""
    parts = hhmm.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def _format_date(date: datetime.date) -> str:
    """Return a human-friendly date, e.g. `'Monday 04 May 2026'`."""
    return date.strftime("%A %d %B %Y")


def _fmt(minutes: int) -> str:
    """Format integer minutes-from-midnight as `HH:MM`.

    Values at or beyond end-of-day clamp to `23:59` so the LLM never sees an
    impossible `24:00` boundary in the timeline JSON.
    """
    if minutes >= 1440:
        return "23:59"
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# Natural-title resolution (LLM "Lunch 🥗" to calendar event_label "lunch")
# ---------------------------------------------------------------------------


def _normalise_title(text: str) -> str:
    """Normalize a natural title or snake_case label to a comparable key.

    `"Lunch 🥗"`    to `"lunch"`
    `"Office Work"` to `"office_work"`
    `"office_work"` to `"office_work"`
    """
    lowered = text.lower()
    return _NORMALISE_RE.sub("_", lowered).strip("_")


def _resolve_event_label_on_date(
    raw: str | None,
    date: datetime.date | None,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
) -> str | None:
    """Match an LLM-supplied event reference to a calendar event_label.

    Accepts natural titles (`"Lunch"`, `"Office Work"`), snake_case
    (`"lunch"`), or names with emoji.  Returns the canonical `event_label`
    of a matching event on *date*, or `None` if no event matches.
    """
    if not raw or date is None:
        return None
    target = _normalise_title(raw)
    if not target:
        return None
    for ev in events_by_date.get(date, []):
        if target in (_normalise_title(ev.label), _normalise_title(ev.effective_label)):
            return ev.label
    return None


def _find_host_event(
    label: str | None,
    date: datetime.date | None,
    events_by_date: dict[datetime.date, list[CalendarEvent]],
) -> CalendarEvent | None:
    """Return the calendar event with *label* on *date*, or `None`.

    Shared by `_plan_week` (clip-rescue lookup) and
    `_validate_concurrent` so both go through the same matcher.
    """
    if label is None or date is None:
        return None
    for ev in events_by_date.get(date, []):
        if ev.label == label:
            return ev
    return None


def _maybe_clip_rescue(
    d: "_ScheduleDecision",
    task: "RecommendedTask",
    events_by_date: dict[datetime.date, list[CalendarEvent]],
) -> "_ScheduleDecision":
    """Snap a partially-overlapping concurrent placement into the host.

    Only mutates the decision when the clipped duration still satisfies
    the task's `[duration_min, duration_max]` band; otherwise the
    original decision is returned and the validator rejects it.
    Assumes `d.concurrent_with` already resolved to a canonical label.
    """
    host = _find_host_event(d.concurrent_with, d.date, events_by_date)
    if host is None:
        return d
    clipped_start = max(d.start_minutes, host.start_minutes)
    clipped_end = min(d.end_minutes, host.end_minutes)
    clipped_duration = clipped_end - clipped_start
    if (
        clipped_duration > 0
        and task.duration_min <= clipped_duration <= task.duration_max
    ):
        return replace(d, start_minutes=clipped_start, end_minutes=clipped_end)
    return d


def _merge_placed(
    placed_by_date: dict[datetime.date, list[tuple[int, int]]],
    week_placed: dict[datetime.date, list[tuple[int, int]]],
) -> dict[datetime.date, list[tuple[int, int]]]:
    """Per-day union of cross-week placements and same-week placements."""
    return {
        k: placed_by_date.get(k, []) + week_placed.get(k, [])
        for k in set(placed_by_date) | set(week_placed)
    }


def _coerce_pieces_to_single(d: "_ScheduleDecision") -> "_ScheduleDecision | None":
    """Promote the first piece of a divide decision into a single placement.

    Used when `_validate_dividable` rejects a `pieces` array (because the
    task has `is_dividable=False`); the remaining pieces are dropped and
    the result flows through `_try_place_single` as if the LLM had
    returned one placement all along. Returns `None` when the pieces
    list is empty.
    """
    if not d.pieces:
        return None
    first = d.pieces[0]
    return replace(
        d,
        pieces=None,
        date=first.date,
        start_minutes=first.start_minutes,
        end_minutes=first.end_minutes,
        concurrent_flag=first.concurrent_with is not None,
        concurrent_with=first.concurrent_with,
    )


# ---------------------------------------------------------------------------
# Schedule decision dataclass + parser
# ---------------------------------------------------------------------------


@dataclass
class _PieceDecision:
    """One piece of a split (divide) placement.

    `concurrent_with` carries the raw event reference verbatim;
    `_plan_week` resolves it to a canonical event_label before
    validation.
    """

    date: datetime.date
    start_minutes: int
    end_minutes: int
    concurrent_with: str | None = None


@dataclass
class _ScheduleDecision:
    """One entry parsed from the LLM's JSON array.

    `task_index` (1-based) is the primary identifier; `task_label` /
    `task_title` are accepted as fallbacks for older response formats.
    `concurrent_with` holds the raw event reference exactly as the LLM
    wrote it (natural title, possibly with emoji, or snake_case).  Resolution
    to a canonical event_label happens in `_plan_week`.

    `pieces` carries multi-session splits for tasks with
    `is_dividable=true`; when non-empty the loop emits one
    :class:`ScheduledTask` per piece with `parent_task_label` set
    to the original task label so `L_divide` can credit the split.
    """

    task_label: str
    scheduled: bool
    task_index: int | None = None
    date: datetime.date | None = None
    start_minutes: int = 0
    end_minutes: int = 0
    concurrent_flag: bool = False
    concurrent_with: str | None = None
    pieces: list[_PieceDecision] | None = None


def _coerce_index(value: Any) -> int | None:
    """Best-effort int coercion for a `task_index` field."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_task_label(item: dict[str, Any]) -> str:
    """Pull the task identifier from a response item.

    Accepts `task_label` (legacy) or `task_title` (new natural form).
    """
    for key in ("task_label", "task_title"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_concurrent_ref(item: dict[str, Any]) -> str | None:
    """Pull the concurrent-host reference from a response item.

    Accepts `concurrent_with_event` (new natural form) or
    `concurrent_with` (legacy snake_case).
    """
    for key in ("concurrent_with_event", "concurrent_with"):
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _parse_oneshot_response(raw_text: str) -> list[_ScheduleDecision] | None:
    """Parse the LLM's one-shot JSON array.

    Returns `None` if the outer structure cannot be parsed (triggers a
    retry).  Individual malformed entries are silently skipped or marked
    unscheduled (the task remains unplaced).
    """
    text = raw_text.strip()
    if not text:
        return None

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        array_match = _ARRAY_RE.search(text)
        if array_match:
            text = array_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    decisions: list[_ScheduleDecision] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        task_label = _extract_task_label(item)
        task_index = _coerce_index(item.get("task_index"))
        if not task_label and task_index is None:
            continue
        if not item.get("scheduled", True):
            decisions.append(
                _ScheduleDecision(
                    task_label=task_label,
                    task_index=task_index,
                    scheduled=False,
                )
            )
            continue
        pieces = _parse_pieces(item.get("pieces"))
        if pieces:
            decisions.append(
                _ScheduleDecision(
                    task_label=task_label,
                    task_index=task_index,
                    scheduled=True,
                    pieces=pieces,
                )
            )
            continue
        try:
            date = datetime.date.fromisoformat(str(item["date"]))
            start_minutes = _hhmm_to_minutes(str(item["start"]))
            end_minutes = _hhmm_to_minutes(str(item["end"]))
        except (KeyError, ValueError):
            decisions.append(
                _ScheduleDecision(
                    task_label=task_label,
                    task_index=task_index,
                    scheduled=False,
                )
            )
            continue
        concurrent_ref = _extract_concurrent_ref(item)
        concurrent_flag = bool(item.get("concurrent_flag", concurrent_ref is not None))
        decisions.append(
            _ScheduleDecision(
                task_label=task_label,
                task_index=task_index,
                scheduled=True,
                date=date,
                start_minutes=start_minutes,
                end_minutes=end_minutes,
                concurrent_flag=concurrent_flag,
                concurrent_with=concurrent_ref if concurrent_flag else None,
            )
        )
    return decisions


def _parse_pieces(raw: object) -> list[_PieceDecision]:
    """Parse a `pieces` array into validated :class:`_PieceDecision` rows.

    Malformed pieces are silently dropped (the planner treats a piece
    list with `>=1` valid entries as a divide). Returns an empty list
    when `raw` is not a list or every entry is malformed.
    """
    if not isinstance(raw, list):
        return []
    out: list[_PieceDecision] = []
    for piece in raw:
        if not isinstance(piece, dict):
            continue
        try:
            date = datetime.date.fromisoformat(str(piece["date"]))
            start_minutes = _hhmm_to_minutes(str(piece["start"]))
            end_minutes = _hhmm_to_minutes(str(piece["end"]))
        except (KeyError, ValueError):
            continue
        concurrent_ref = _extract_concurrent_ref(piece)
        out.append(
            _PieceDecision(
                date=date,
                start_minutes=start_minutes,
                end_minutes=end_minutes,
                concurrent_with=concurrent_ref,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Optional debug logging
# ---------------------------------------------------------------------------


def _maybe_dump_call(
    week_dates: list[datetime.date],
    attempt: int,
    prompt: str,
    raw_response: str,
    decisions: list[_ScheduleDecision] | None,
    person_id: str = "unknown",
) -> None:
    """Optionally write the prompt + raw response to a debug directory.

    Activated only when the `LLM_AGENT_DEBUG_DIR` environment variable is
    set to a writable directory.  One file per LLM call, scoped per person
    so concurrent runs across personas do not overwrite each other:

        <dir>/<person_id>/<week-start>__attempt<N>.txt

    Args:
        week_dates: the dates in the week the call was issued for.
        attempt: 0-based retry index for the LLM call.
        prompt: the rendered prompt sent to the LLM.
        raw_response: the LLM's raw response string.
        decisions: parsed decisions, or `None` if the response was
            unparseable.
        person_id: the persona this call belongs to.  Defaults to
            `"unknown"` so callers that have no persona context (e.g.
            unit tests) still produce a usable dump path.
    """
    debug_dir = os.environ.get("LLM_AGENT_DEBUG_DIR")
    if not debug_dir or not week_dates:
        return
    try:
        out_dir = Path(debug_dir) / person_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = week_dates[0].isoformat()
        fname = out_dir / f"{ts}__attempt{attempt}.txt"
        if decisions is None:
            summary = "decisions=None  (response unparseable)"
        else:
            placed = sum(1 for d in decisions if d.scheduled)
            concurrent = sum(1 for d in decisions if d.scheduled and d.concurrent_flag)
            summary = (
                f"decisions={len(decisions)}  scheduled={placed}  "
                f"concurrent={concurrent}"
            )
        fname.write_text(
            f"=== person ===\n{person_id}\n\n"
            f"=== summary ===\n{summary}\n\n"
            f"=== prompt ===\n{prompt}\n\n"
            f"=== raw response ===\n{raw_response}\n",
            encoding="utf-8",
        )
    except OSError:
        return


# ---------------------------------------------------------------------------
# Calendar and task summary helpers
# ---------------------------------------------------------------------------


def _merge_busy(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge and sort overlapping busy intervals."""
    if not intervals:
        return []
    srt = sorted(intervals)
    merged = [list(srt[0])]
    for s, e in srt[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def _host_flag_suffix(event: CalendarEvent, host_flags: list[str]) -> str:
    """Return the extra `,key:value` fragments to append to a busy interval JSON."""
    if not host_flags:
        return ""
    parts: list[str] = []
    for flag in host_flags:
        if flag == "is_concurrent":
            parts.append(f',"is_concurrent":{str(event.is_concurrent).lower()}')
        elif flag == "is_dividable":
            parts.append(f',"is_dividable":{str(event.is_dividable).lower()}')
        elif flag == "concurrent_with":
            payload = json.dumps(list(event.concurrent_with))
            parts.append(f',"concurrent_with":{payload}')
        elif flag == "intensity":
            parts.append(f',"intensity":{int(event.intensity)}')
    return "".join(parts)


def _build_calendar_summary(
    week_dates: list[datetime.date],
    events_by_date: dict[datetime.date, list[CalendarEvent]],
    scheduled_tasks: list[ScheduledTask] | None = None,
    *,
    observation: ObservationConfig | None = None,
) -> str:
    """Render each day as a strictly-ordered JSON array of intervals.

    Each interval is one of:

    - `{"type": "busy", "label": "<Title>", "start": "HH:MM", "end": "HH:MM"}`
     ; the LLM judges concurrency suitability from the title alone; the
      benchmark's `is_concurrent` flag is **never** exposed to the LLM.
    - `{"type": "free", "label": "free", "start": "HH:MM", "end": "HH:MM",
       "duration_min": <int>}`
    - `{"type": "done", "label": "<Title>", "start": "HH:MM", "end": "HH:MM",
       "host": "<Title>"|null}`; task placed in an earlier week.
      `label` is the GraphRAG-supplied display name verbatim (emoji
      preserved); the augmenter does not modify task content.
    """
    placed: dict[datetime.date, list[ScheduledTask]] = defaultdict(list)
    week_date_set = set(week_dates)
    for st in scheduled_tasks or []:
        if st.date in week_date_set:
            placed[st.date].append(st)

    host_flags = list(observation.host_flags) if observation is not None else []

    out: list[str] = []
    for day in week_dates:
        day_events = events_by_date.get(day, [])
        day_placed = placed.get(day, [])

        merged_busy = _merge_busy(
            [(e.start_minutes, e.end_minutes) for e in day_events]
            + [(st.start_minutes, st.end_minutes) for st in day_placed]
        )
        free_gaps: list[tuple[int, int]] = []
        prev = 0
        for s, e in merged_busy:
            if s > prev:
                free_gaps.append((prev, s))
            prev = max(prev, e)
        if prev < 1440:
            free_gaps.append((prev, 1440))

        intervals: list[tuple[int, str]] = []

        for ev in day_events:
            title = _event_display_title(ev.effective_label)
            extra = _host_flag_suffix(ev, host_flags)
            intervals.append(
                (
                    ev.start_minutes,
                    f'{{"type":"busy","label":"{title}","start":"{_fmt(ev.start_minutes)}",'
                    f'"end":"{_fmt(ev.end_minutes)}"{extra}}}',
                )
            )

        for st in day_placed:
            label = st.task.effective_display_name
            host = (
                f'"{_event_display_title(st.concurrent_with)}"'
                if st.concurrent_with
                else "null"
            )
            intervals.append(
                (
                    st.start_minutes,
                    f'{{"type":"done","label":"{label}","start":"{_fmt(st.start_minutes)}",'
                    f'"end":"{_fmt(st.end_minutes)}","host":{host}}}',
                )
            )

        for s, e in free_gaps:
            intervals.append(
                (
                    s,
                    f'{{"type":"free","label":"free","start":"{_fmt(s)}",'
                    f'"end":"{_fmt(e)}","duration_min":{e - s}}}',
                )
            )
        intervals.sort(key=lambda x: x[0])

        out.append(f"{_format_date(day)} TIMELINE:")
        out.append("[")
        for i, (_, payload) in enumerate(intervals):
            comma = "," if i < len(intervals) - 1 else ""
            out.append(f"  {payload}{comma}")
        out.append("]")
        out.append("")

    return "\n".join(out).rstrip()


def _build_tasks_list(
    tasks: list[RecommendedTask],
    *,
    observation: ObservationConfig | None = None,
    context_links_by_uri: dict[str, list[str]] | None = None,
) -> str:
    """Render each task as a key:value block; fields gated by `observation.task_flags`.

    When `context_recommends` is enabled in `task_flags` and a task's
    `ontology_uri` carries `context_links`, the rendered field shows
    only the intersection of those links with `observation.contexts`;
    categories outside the augmenter's observation budget are dropped
    at render time and never reach the prompt.
    """
    if observation is not None:
        flags = set(observation.task_flags)
        observed_contexts = list(observation.contexts)
    else:
        flags = {
            "concurrent_ok",
            "dividable_ok",
            "duration_min",
            "duration_max",
            "intensity",
        }
        observed_contexts = []
    observed_set = set(observed_contexts)
    links = context_links_by_uri or {}

    blocks: list[str] = []
    for i, t in enumerate(tasks, 1):
        title = t.effective_display_name
        lines = [
            f"TASK #{i}:",
            f"  task_index        : {i}",
            f'  task_title        : "{title}"',
        ]
        if "duration_min" in flags:
            lines.append(f"  duration_min      : {t.duration_min}")
        if "duration_max" in flags:
            lines.append(f"  duration_max      : {t.duration_max}")
        if "intensity" in flags:
            lines.append(f"  intensity         : {t.intensity}")
        if "concurrent_ok" in flags:
            lines.append(
                f"  concurrent_ok     : {'true' if t.is_concurrent else 'false'}"
            )
        if "dividable_ok" in flags:
            lines.append(
                f"  dividable_ok      : {'true' if t.is_dividable else 'false'}"
            )
        if "context_recommends" in flags and observed_set:
            task_uri = t.ontology_uri or ""
            recommended = set(links.get(task_uri) or ())
            intersection = [c for c in observed_contexts if c in recommended]
            if intersection:
                lines.append(f"  context_recommends: [{', '.join(intersection)}]")
        if t.description:
            lines.append(f'  description       : "{t.description}"')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_context_summary(
    contexts: list[ContextEpisode],
    week_dates: list[datetime.date],
    observation: ObservationConfig | None,
    catalog: Any | None = None,
) -> str:
    """Render context episodes per opt-in category for the week; empty when none apply."""
    if observation is None or not observation.contexts:
        return ""
    visible_categories = list(observation.contexts)
    week_date_set = set(week_dates)

    by_cat_date: dict[tuple[str, datetime.date], list[ContextEpisode]] = defaultdict(
        list
    )
    for ep in contexts:
        if ep.category not in visible_categories:
            continue
        if ep.date not in week_date_set:
            continue
        by_cat_date[(ep.category, ep.date)].append(ep)

    if not by_cat_date:
        return ""

    lines: list[str] = ["## CONTEXT (per-day momentary signals; persona ground truth)"]
    full = observation.context_detail == "full"

    for cat in visible_categories:
        cat_days = sorted({d for (c, d) in by_cat_date if c == cat})
        if not cat_days:
            continue
        lines.append("")
        lines.append(f"# CONTEXT ({cat})")
        for d in cat_days:
            lines.append(f"{d.isoformat()} ({d.strftime('%a')}):")
            for ep in sorted(by_cat_date[(cat, d)], key=lambda e: e.start_minutes):
                display = ep.name
                if catalog is not None and ep.ontology_uri:
                    entry = catalog.get(ep.ontology_uri)
                    if entry is not None and entry.label and entry.label != ep.name:
                        display = f"{ep.name} ({entry.label})"
                start = _fmt(ep.start_minutes)
                end = _fmt(ep.end_minutes)
                row = f"  - {display}  {start}-{end}"
                if full:
                    extras: list[str] = []
                    if ep.ontology_uri:
                        extras.append(f"uri={ep.ontology_uri}")
                    if ep.dimension:
                        extras.append(f"dimension={ep.dimension}")
                    if ep.polarity:
                        extras.append(f"polarity={ep.polarity}")
                    if extras:
                        row += "  [" + ", ".join(extras) + "]"
                lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decision lookup
# ---------------------------------------------------------------------------


def _build_decision_lookup(
    decisions: list[_ScheduleDecision],
) -> tuple[dict[int, _ScheduleDecision], dict[str, list[_ScheduleDecision]]]:
    """Build O(1) lookup maps over the LLM's decisions.

    `by_index` is the primary map (keyed on 1-based `task_index`).  The
    fallback queue `by_label` lists decisions by normalized task label or
    title; popped sequentially when no index match exists, so multiple task
    instances sharing a snake_case label still receive distinct placements.
    """
    by_index: dict[int, _ScheduleDecision] = {}
    by_label: dict[str, list[_ScheduleDecision]] = defaultdict(list)
    for d in decisions:
        if d.task_index is not None:
            by_index[d.task_index] = d
        if d.task_label:
            by_label[_normalise_title(d.task_label)].append(d)
    return by_index, dict(by_label)


def _lookup_decision(
    task: RecommendedTask,
    task_index: int,
    by_index: dict[int, _ScheduleDecision],
    by_label_queue: dict[str, list[_ScheduleDecision]],
) -> _ScheduleDecision | None:
    """Return the LLM decision for *task* (1-based `task_index`)."""
    d = by_index.get(task_index)
    if d is not None:
        return d
    title_keys = (
        _normalise_title(task.effective_display_name),
        _normalise_title(task.label),
    )
    for key in title_keys:
        queue = by_label_queue.get(key)
        if queue:
            return queue.pop(0)
    return None


# ---------------------------------------------------------------------------
# DirectLLMPipeline; separation of concerns from GraphRAG (2026-05-14)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DirectResponse:
    """Minimal stand-in for `neo4j_graphrag.generation.types.RagResultModel`.

    The :class:`LLMAugmenter` only reads `.answer` from the pipeline
    response, so this two-field shape is sufficient to keep the
    pipeline-call site untouched when we swap a retrieval-augmented
    pipeline for a direct LLM client.
    """

    answer: str


class LLMCallRecorder:
    """Per-call telemetry sink shared between the augmenter and the CLI.

    The CLI hands one recorder to :class:`DirectLLMPipeline` for the
    whole augment run; before each per-person augment it calls
    :meth:`reset`, then after the per-person augment returns it copies
    every entry in :attr:`calls` into the person's
    :class:`AugmentTelemetry`.  Token counts come from the provider's
    `usage_metadata` when the LLM client surfaces it (the
    `UsageTrackingLLM` path) and fall back to :func:`estimate_tokens`
    otherwise.  `reasoning_tokens` is the provider-reported hidden
    reasoning share already included in `output_tokens`; 0 when the
    model reports none.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float, int]] = []

    def record(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        wall_time_seconds: float,
        reasoning_tokens: int = 0,
    ) -> None:
        self.calls.append(
            (
                int(input_tokens),
                int(output_tokens),
                float(wall_time_seconds),
                int(reasoning_tokens),
            )
        )

    def reset(self) -> None:
        self.calls.clear()


class DirectLLMPipeline:
    """Drop-in for :class:`neo4j_graphrag.generation.GraphRAG` that bypasses
    retrieval and the citation prompt template.

    **Why bypass.**  The LLM augmenter places already-grounded tasks
    into time slots; it does not need ontology context or URI
    citations.  GraphRAG wrapping costs an extra OpenAI embedding
    call, a Neo4j vector search, and 500-2000 wasted input tokens of
    ontology context per week-call.  Ontology grounding is the *task
    generation* step's job (handled in `task_generation/`); by the
    time a :class:`RecommendedTask` reaches the augmenter it already
    carries the verified `ontology_uri`.  Routing the augmenter
    through `build_graphrag` would re-do work the pipeline has
    already completed.

    Exposes the same `search(query_text)` API the augmenter
    consumes ([llm_agent.py:723](src/scripts/scenarios/augmentation/llm_agent.py))
    so the call site is identical for both retrieval-augmented and
    direct pipelines; only the wiring in `_build_augmenter` flips.

    A *recorder* may be supplied so the CLI can capture per-person
    LLM token counts + wall-time without changing the
    :class:`Augmenter` interface.  When `None` the pipeline acts
    exactly as before (no telemetry overhead).
    """

    def __init__(self, llm: Any, recorder: LLMCallRecorder | None = None) -> None:
        self._llm = llm
        self._recorder = recorder

    def search(self, query_text: str) -> _DirectResponse:
        """Issue a single LLM call and return a response with `.answer`.

        The LLM is expected to expose `.invoke(prompt)` returning an
        object with a `.content` attribute; the `LLMInterface`
        shape from `neo4j_graphrag.llm` (which the project's
        :func:`src.graphrag.llm.make_llm` returns).
        """
        from src.scripts.scenarios.metrics.telemetry import _Timer, estimate_tokens

        with _Timer() as t:
            response = self._llm.invoke(query_text)
        answer = getattr(response, "content", "") or ""
        if self._recorder is not None:
            usage = getattr(response, "usage_metadata", None)
            if not isinstance(usage, dict):
                usage = {}
            self._recorder.record(
                input_tokens=int(usage.get("input_tokens") or 0)
                or estimate_tokens(query_text),
                output_tokens=int(usage.get("output_tokens") or 0)
                or estimate_tokens(answer),
                wall_time_seconds=t.elapsed,
                reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
            )
        return _DirectResponse(answer=answer)


# ---------------------------------------------------------------------------
# LLMAugmenter
# ---------------------------------------------------------------------------


class LLMAugmenter(Augmenter):
    """One-shot LLM-agent augmenter.

    For each calendar week a single LLM call receives all recommended tasks and
    the full week calendar (rendered as a JSON-array timeline) and returns a
    complete placement plan.  Decisions are matched back to tasks by their
    1-based `task_index`.

    Args:
        llm_pipeline: duck-typed pipeline with
            `search(query_text: str) -> response` where `response.answer`
            is the raw LLM string.
        allen_rules: `AllenPairRule` list for admissible-relation derivation.
        daily_window: hard waking-hours bound applied to every placement.
            Defaults to 06:00 to 22:00 to preserve current behavior when
            the caller does not supply one.
        prompt_ablate: optional set of `augment_oneshot` block names
            to drop at render time. Empty by default.
        prompt_placebos: optional per-block replacement text. Wins
            over deletion when both are set.
    """

    def __init__(
        self,
        llm_pipeline: Any,
        allen_rules: list[AllenPairRule] | None = None,
        daily_window: DailyWindow | None = None,
        prompt_ablate: frozenset[str] | None = None,
        prompt_placebos: dict[str, str] | None = None,
        context_catalog: Any | None = None,
    ) -> None:
        self._pipeline = llm_pipeline
        self._allen_rules: list[AllenPairRule] = allen_rules or []
        self._daily_window: DailyWindow = daily_window or DailyWindow()
        self._prompt_ablate: frozenset[str] = prompt_ablate or frozenset()
        self._prompt_placebos: dict[str, str] = dict(prompt_placebos or {})
        # Lazy-loaded context IRI catalog; passed to `_build_context_summary`
        # so context labels render their canonical catalog form alongside
        # the persona-local slug.
        self._context_catalog: Any | None = context_catalog
        self._catalog_load_attempted: bool = False
        # Lazy-loaded HealthTasks context_links map; the renderer
        # intersects each task's recommended categories with the active
        # observation budget before any text reaches the prompt.
        self._context_links_by_uri: dict[str, list[str]] | None = None
        self._links_load_attempted: bool = False

    def _get_context_catalog(self) -> Any | None:
        """Return the ContextIriCatalog, loading from disk on first access."""
        if self._context_catalog is not None or self._catalog_load_attempted:
            return self._context_catalog
        self._catalog_load_attempted = True
        try:
            from src.scripts.persona.context.catalog import load_catalog

            self._context_catalog = load_catalog()
        except Exception:
            self._context_catalog = None
        return self._context_catalog

    def _get_context_links(self) -> dict[str, list[str]]:
        """Return `{task_uri: {category: frozenset(iri)}}`, loaded once per augmenter."""
        if self._context_links_by_uri is not None or self._links_load_attempted:
            return self._context_links_by_uri or {}
        self._links_load_attempted = True
        try:
            from pathlib import Path as _Path

            from src.scripts.scenarios.metrics.context_fit import (
                load_context_categories_by_iri,
                load_context_links_by_uri,
            )

            ttl_path = _Path("src/assets/ontologies/HealthTasks_2026.05.19.ttl")
            iris_path = _Path("src/assets/ontologies/context_iris.json")
            if ttl_path.exists() and iris_path.exists():
                self._context_links_by_uri = load_context_links_by_uri(
                    ttl_path, load_context_categories_by_iri(iris_path)
                )
            else:
                self._context_links_by_uri = {}
        except Exception:
            self._context_links_by_uri = {}
        return self._context_links_by_uri or {}

    # ------------------------------------------------------------------
    # Augmenter interface
    # ------------------------------------------------------------------

    def augment(
        self,
        calendar: CalendarTrace,
        tasks: list[RecommendedTask],
        config: AugmentationConfig,
        horizon: tuple[datetime.date, int] | None = None,
        *,
        weekly_tasks: list[list[RecommendedTask]] | None = None,
    ) -> SchedulingSolution:
        """Place recommended tasks week-by-week using one LLM call per week.

        `weekly_tasks` gives each week its own batch; when omitted, `tasks`
        is broadcast to every week.
        """
        horizon_dates = _get_horizon_dates(calendar, horizon=horizon)

        if not horizon_dates:
            aug_cal = AugmentedCalendar(
                person_id=calendar.person_id,
                base_events=list(calendar.events),
            )
            return SchedulingSolution(
                person_id=calendar.person_id,
                augmented_calendar=aug_cal,
                tasks=list(tasks),
                scheduled=[],
                unscheduled=list(tasks),
            )

        events_by_date: dict[datetime.date, list[CalendarEvent]] = defaultdict(list)
        for e in calendar.events:
            events_by_date[e.date].append(e)

        rule_index = _build_rule_index(self._allen_rules)
        placed_by_date: dict[datetime.date, list[tuple[int, int]]] = defaultdict(list)
        consumed_concurrent: set[tuple[datetime.date, str]] = set()

        tasks_sorted = sorted(tasks, key=lambda t: t.intensity, reverse=True)
        all_scheduled: list[ScheduledTask] = []
        all_unscheduled: list[RecommendedTask] = []
        all_task_instances: list[RecommendedTask] = []
        max_retries = config.llm_agent.max_retries
        weekly_chunks = _get_weekly_chunks(horizon_dates)
        weeks = weekly_tasks if weekly_tasks is not None else [tasks]

        if config.repeat_per_week:
            for week_index, week_dates in enumerate(weekly_chunks):
                week_tasks = sorted(
                    _tasks_for_week(weeks, week_index),
                    key=lambda t: t.intensity,
                    reverse=True,
                )
                all_task_instances.extend(week_tasks)
                scheduled, unscheduled = self._plan_week(
                    week_tasks,
                    week_dates,
                    events_by_date,
                    placed_by_date,
                    consumed_concurrent,
                    config,
                    rule_index,
                    max_retries,
                    scheduled_so_far=list(all_scheduled),
                    person_id=calendar.person_id,
                    contexts=list(calendar.contexts),
                )
                for st in scheduled:
                    placed_by_date[st.date].append((st.start_minutes, st.end_minutes))
                    if st.concurrent_with is not None:
                        consumed_concurrent.add((st.date, st.concurrent_with))
                all_scheduled.extend(scheduled)
                all_unscheduled.extend(unscheduled)
        else:
            all_task_instances = list(tasks_sorted)
            remaining = list(tasks_sorted)
            for week_dates in weekly_chunks:
                scheduled, unplaced = self._plan_week(
                    remaining,
                    week_dates,
                    events_by_date,
                    placed_by_date,
                    consumed_concurrent,
                    config,
                    rule_index,
                    max_retries,
                    scheduled_so_far=list(all_scheduled),
                    person_id=calendar.person_id,
                    contexts=list(calendar.contexts),
                )
                for st in scheduled:
                    placed_by_date[st.date].append((st.start_minutes, st.end_minutes))
                    if st.concurrent_with is not None:
                        consumed_concurrent.add((st.date, st.concurrent_with))
                all_scheduled.extend(scheduled)
                remaining = unplaced
            all_unscheduled = remaining

        aug_cal = AugmentedCalendar(
            person_id=calendar.person_id,
            base_events=list(calendar.events),
            scheduled_tasks=all_scheduled,
        )
        return SchedulingSolution(
            person_id=calendar.person_id,
            augmented_calendar=aug_cal,
            tasks=all_task_instances,
            scheduled=all_scheduled,
            unscheduled=all_unscheduled,
        )

    # ------------------------------------------------------------------
    # One-shot weekly planning
    # ------------------------------------------------------------------

    def _plan_week(
        self,
        tasks: list[RecommendedTask],
        week_dates: list[datetime.date],
        events_by_date: dict,
        placed_by_date: dict,
        consumed_concurrent: set,
        config: AugmentationConfig,
        rule_index: dict,
        max_retries: int,
        scheduled_so_far: list[ScheduledTask] | None = None,
        person_id: str = "unknown",
        contexts: list[ContextEpisode] | None = None,
    ) -> tuple[list[ScheduledTask], list[RecommendedTask]]:
        """Issue one LLM call for the whole week and return `(scheduled, unscheduled)`.

        `person_id` is forwarded to `_maybe_dump_call` so debug dumps
        end up in a per-persona sub-directory and concurrent personas do
        not overwrite each other's logs.
        """
        decisions: list[_ScheduleDecision] | None = None
        prompt = ""
        raw = ""
        for attempt in range(max_retries):
            prompt = self._build_oneshot_prompt(
                tasks,
                week_dates,
                events_by_date,
                scheduled_so_far,
                config,
                contexts=contexts,
            )
            response = self._pipeline.search(query_text=prompt)
            raw = getattr(response, "answer", "") or ""
            decisions = _parse_oneshot_response(raw)
            _maybe_dump_call(
                week_dates, attempt, prompt, raw, decisions, person_id=person_id
            )
            if decisions is not None:
                break

        if decisions is None:
            return [], list(tasks)

        by_index, by_label_queue = _build_decision_lookup(decisions)

        scheduled: list[ScheduledTask] = []
        unscheduled: list[RecommendedTask] = []
        week_placed: dict[datetime.date, list[tuple[int, int]]] = defaultdict(list)
        week_concurrent: set[tuple[datetime.date, str]] = set()

        for idx, task in enumerate(tasks, 1):
            d = _lookup_decision(task, idx, by_index, by_label_queue)
            if d is None or not d.scheduled:
                unscheduled.append(task)
                continue

            if d.pieces is not None:
                if self._validate_dividable(d, task) is None:
                    emitted = self._place_pieces(
                        d.pieces,
                        task,
                        week_dates,
                        events_by_date,
                        placed_by_date,
                        week_placed,
                        consumed_concurrent,
                        week_concurrent,
                        rule_index,
                    )
                    if emitted:
                        scheduled.extend(emitted)
                    else:
                        unscheduled.append(task)
                    continue
                # Non-dividable task came back with a `pieces` array.
                # Coerce to a single placement using the first piece;
                # the remaining pieces are dropped on the floor.
                d = _coerce_pieces_to_single(d)
                if d is None:  # pragma: no cover - parser never produces empty pieces
                    unscheduled.append(task)
                    continue

            st = self._try_place_single(
                d,
                task,
                week_dates,
                events_by_date,
                placed_by_date,
                week_placed,
                consumed_concurrent,
                week_concurrent,
                rule_index,
            )
            if st is None:
                unscheduled.append(task)
                continue
            scheduled.append(st)

        return scheduled, unscheduled

    def _try_place_single(
        self,
        d: _ScheduleDecision,
        task: RecommendedTask,
        week_dates: list[datetime.date],
        events_by_date: dict,
        placed_by_date: dict,
        week_placed: dict,
        consumed_concurrent: set,
        week_concurrent: set,
        rule_index: dict,
    ) -> ScheduledTask | None:
        """Resolve, clip-rescue, validate, and emit one standalone placement.

        Updates `week_placed` / `week_concurrent` on success.
        """
        if d.concurrent_flag and d.concurrent_with is not None:
            resolved = _resolve_event_label_on_date(
                d.concurrent_with, d.date, events_by_date
            )
            if resolved is None:
                return None
            d = replace(d, concurrent_with=resolved)
            d = _maybe_clip_rescue(d, task, events_by_date)

        error = self._validate_decision(
            d,
            task,
            week_dates,
            events_by_date,
            _merge_placed(placed_by_date, week_placed),
            consumed_concurrent | week_concurrent,
            rule_index,
        )
        if error is not None:
            return None

        st = ScheduledTask(
            task=task,
            start_minutes=d.start_minutes,
            end_minutes=d.end_minutes,
            is_standalone=not d.concurrent_flag,
            concurrent_with=d.concurrent_with if d.concurrent_flag else None,
            date=d.date,  # type: ignore[arg-type]
        )
        week_placed[st.date].append((st.start_minutes, st.end_minutes))
        if st.concurrent_with is not None:
            week_concurrent.add((st.date, st.concurrent_with))
        return st

    def _place_pieces(
        self,
        pieces: list[_PieceDecision],
        task: RecommendedTask,
        week_dates: list[datetime.date],
        events_by_date: dict,
        placed_by_date: dict,
        week_placed: dict,
        consumed_concurrent: set,
        week_concurrent: set,
        rule_index: dict,
    ) -> list[ScheduledTask]:
        """Validate every piece independently and emit `ScheduledTask` rows.

        Each emitted row carries `parent_task_label=task.label` so
        `L_divide` credits the split. Invalid pieces are silently
        dropped; an empty return signals the whole split failed and
        the caller marks the task unscheduled.
        """
        emitted: list[ScheduledTask] = []
        for piece in pieces:
            decision = _ScheduleDecision(
                task_label=task.label,
                scheduled=True,
                date=piece.date,
                start_minutes=piece.start_minutes,
                end_minutes=piece.end_minutes,
                concurrent_flag=piece.concurrent_with is not None,
                concurrent_with=piece.concurrent_with,
            )
            if decision.concurrent_flag and decision.concurrent_with is not None:
                resolved = _resolve_event_label_on_date(
                    decision.concurrent_with, decision.date, events_by_date
                )
                if resolved is None:
                    continue
                decision = replace(decision, concurrent_with=resolved)
                decision = _maybe_clip_rescue(decision, task, events_by_date)

            error = self._validate_decision(
                decision,
                task,
                week_dates,
                events_by_date,
                _merge_placed(placed_by_date, week_placed),
                consumed_concurrent | week_concurrent,
                rule_index,
            )
            if error is not None:
                continue
            st = ScheduledTask(
                task=task,
                start_minutes=decision.start_minutes,
                end_minutes=decision.end_minutes,
                is_standalone=not decision.concurrent_flag,
                concurrent_with=(
                    decision.concurrent_with if decision.concurrent_flag else None
                ),
                date=decision.date,  # type: ignore[arg-type]
                parent_task_label=task.label,
            )
            week_placed[st.date].append((st.start_minutes, st.end_minutes))
            if st.concurrent_with is not None:
                week_concurrent.add((st.date, st.concurrent_with))
            emitted.append(st)
        return emitted

    def _build_oneshot_prompt(
        self,
        tasks: list[RecommendedTask],
        week_dates: list[datetime.date],
        events_by_date: dict,
        scheduled_so_far: list[ScheduledTask] | None,
        config: AugmentationConfig,
        *,
        contexts: list[ContextEpisode] | None = None,
    ) -> str:
        """Build the single one-shot prompt for an entire week.

        When `prompt_ablate` is set and the template is
        `augment_oneshot`, those blocks are dropped (or replaced with
        the matching placebo). Other templates ignore the ablation
        parameters.
        """
        observation = getattr(config, "observation", None)
        calendar_summary = _build_calendar_summary(
            week_dates,
            events_by_date,
            scheduled_so_far,
            observation=observation,
        )
        tasks_list = _build_tasks_list(
            tasks,
            observation=observation,
            context_links_by_uri=self._get_context_links(),
        )
        context_summary = _build_context_summary(
            list(contexts or []),
            week_dates,
            observation,
            catalog=(
                self._get_context_catalog()
                if (observation and observation.contexts)
                else None
            ),
        )
        kwargs: dict[str, Any] = dict(
            num_tasks=len(tasks),
            week_start=_format_date(week_dates[0]),
            week_end=_format_date(week_dates[-1]),
            wake_start=_fmt(self._daily_window.wake_minutes),
            sleep_start=_fmt(self._daily_window.sleep_minutes),
            calendar_summary=calendar_summary,
            tasks_list=tasks_list,
            context_summary=context_summary,
        )
        if (
            self._prompt_ablate
            and config.llm_agent.prompt_template == "augment_oneshot"
        ):
            kwargs["_ablate"] = self._prompt_ablate
            if self._prompt_placebos:
                kwargs["_placebos"] = self._prompt_placebos
        return render(config.llm_agent.prompt_template, **kwargs)

    # ------------------------------------------------------------------
    # Validation (purely structural; no semantic compatibility check)
    # ------------------------------------------------------------------

    def _validate_decision(
        self,
        decision: _ScheduleDecision,
        task: RecommendedTask,
        week_dates: list[datetime.date],
        events_by_date: dict,
        placed_by_date: dict,
        consumed_concurrent: set,
        rule_index: dict,
    ) -> str | None:
        """Return an error string if the decision is invalid, else `None`."""
        if decision.date not in week_dates:
            return f"Date {decision.date} is not in the current week."

        # Hard waking-window enforcement.
        if (
            decision.start_minutes < self._daily_window.wake_minutes
            or decision.end_minutes > self._daily_window.sleep_minutes
        ):
            return (
                f"[{_fmt(decision.start_minutes)}, {_fmt(decision.end_minutes)}] "
                f"is outside daily window "
                f"[{_fmt(self._daily_window.wake_minutes)}, "
                f"{_fmt(self._daily_window.sleep_minutes)})."
            )

        duration = decision.end_minutes - decision.start_minutes
        if duration < task.duration_min:
            return f"Duration {duration} min < minimum {task.duration_min} min."
        if duration > task.duration_max:
            return f"Duration {duration} min > maximum {task.duration_max} min."

        if decision.concurrent_flag:
            return self._validate_concurrent(
                decision, task, consumed_concurrent, events_by_date
            )
        return self._validate_standalone(
            decision, task, events_by_date, placed_by_date, rule_index
        )

    def _validate_dividable(
        self,
        decision: _ScheduleDecision,
        task: RecommendedTask,
    ) -> str | None:
        """Server-side guard for the `pieces` (divide) placement path.

        Returns an error string when the decision carries a `pieces`
        array on a task that is not flagged dividable. The caller
        coerces the placement back to a single placement via
        :func:`_coerce_pieces_to_single` rather than dropping the task.

        `task.is_dividable=False` is a benchmark-internal flag; the
        LLM does see it in the tasks_list as `dividable_ok` and the
        `split_dividable_tasks` prompt block tells it to honor the
        gate, but a non-compliant response should not silently inject
        `parent_task_label` rows the metric would then ignore.
        """
        if decision.pieces is None:
            return None
        if not task.is_dividable:
            return (
                f"task {task.label!r} has is_dividable=False but the response "
                f"emitted {len(decision.pieces)} piece(s); coercing back to a "
                "single placement."
            )
        return None

    def _validate_concurrent(
        self,
        decision: _ScheduleDecision,
        task: RecommendedTask,
        consumed_concurrent: set,
        events_by_date: dict,
    ) -> str | None:
        """Validate a concurrent placement structurally.

        Checks (in order):
          1. `concurrent_with` is non-null.
          2. Host event exists on the date.
          3. Host event has `is_concurrent=True`; exclusive events
             may not host concurrent placements.  This flag is
             benchmark-internal; the LLM never sees it and must judge
             concurrency from titles and descriptions alone.
          4. Host event has not already been used by another concurrent task.
          5. Task `[start, end]` is contained inside the host's interval.

        Semantic compatibility (σ) is intentionally not checked here -
        `LLMJudgeOracle` scores quality at evaluation time.
        """
        if decision.concurrent_with is None:
            return "concurrent_flag=true but concurrent_with is null."
        day_events = events_by_date.get(decision.date, [])
        host = next(
            (e for e in day_events if e.label == decision.concurrent_with), None
        )
        if host is None:
            return f"No event {decision.concurrent_with!r} on {decision.date}."
        if not host.is_concurrent:
            return (
                f"Host {decision.concurrent_with!r} on {decision.date} has "
                "is_concurrent=False; concurrent placement is not permitted "
                "against this event."
            )
        if (decision.date, decision.concurrent_with) in consumed_concurrent:
            return (
                f"Event {decision.concurrent_with!r} on {decision.date} "
                "already used by another concurrent task."
            )
        if (
            decision.start_minutes < host.start_minutes
            or decision.end_minutes > host.end_minutes
        ):
            return (
                f"[{_fmt(decision.start_minutes)}, {_fmt(decision.end_minutes)}] "
                f"is not contained inside host {decision.concurrent_with!r} "
                f"[{_fmt(host.start_minutes)}, {_fmt(host.end_minutes)}]."
            )
        return None

    def _validate_standalone(
        self,
        decision: _ScheduleDecision,
        task: RecommendedTask,
        events_by_date: dict,
        placed_by_date: dict,
        rule_index: dict,
    ) -> str | None:
        s, e = decision.start_minutes, decision.end_minutes
        for ev in events_by_date.get(decision.date, []):
            admissible = build_admissible_rx(task, ev, rule_index)
            rel = compute_allen_relation(s, e, ev.start_minutes, ev.end_minutes)
            if rel not in admissible:
                return f"Conflicts with {ev.label!r} via Allen relation {rel.value!r}."
        for ps, pe in placed_by_date.get(decision.date, []):
            if is_overlapping(s, e, ps, pe) and not task.is_concurrent:
                return "Overlaps an already-placed task."
        return None
