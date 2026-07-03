"""Per-person tab-separated timeline export (events + contexts + optional augmented tasks)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from src.scripts.persona.context.catalog import ContextIriCatalog
    from src.scripts.persona.context.schema import ContextEpisode
    from src.scripts.persona.domain.schedule import PersonSchedule
    from src.scripts.scenarios.domain.task import ScheduledTask


TIMELINE_COLUMNS: tuple[str, ...] = (
    "person_id",
    "persona_id",
    "row_type",
    "category",
    "name",
    "label",
    "date",
    "weekday",
    "start",
    "end",
    "duration_min",
    "start_minutes",
    "end_minutes",
    "ontology_uri",
    "is_concurrent",
    "is_dividable",
    "concurrent_with",
    "parent_task_label",
    "intensity",
    "dimension",
    "polarity",
    "source",
)

_ROW_TYPE_RANK: dict[str, int] = {"event": 0, "context": 1, "augmented_task": 2}
_WEEKDAY: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(slots=True)
class TimelineWriteResult:
    """Result of one TSV write call."""

    path: Path
    row_count: int
    tab_replacements: int


def _format_time(minutes: int) -> str:
    """Render minutes-from-midnight as HH:MM, clamped to [0, 24*60)."""
    m = max(0, int(minutes)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def _weekday_short(date: datetime.date) -> str:
    """Return the three-letter weekday code (Mon..Sun)."""
    return _WEEKDAY[date.weekday()]


def _empty_to_blank(value: Any) -> str:
    """Render None as empty string, booleans as `true`/`false`, anything else via str()."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _normalise_cells(row: Mapping[str, Any]) -> tuple[dict[str, str], int]:
    """Stringify every cell value and replace embedded tabs with spaces; return (row, tabs)."""
    cleaned: dict[str, str] = {}
    tabs = 0
    for col, val in row.items():
        s = _empty_to_blank(val)
        if "\t" in s:
            tabs += s.count("\t")
            s = s.replace("\t", " ")
        cleaned[col] = s
    return cleaned, tabs


def _row_sort_key(row: Mapping[str, Any]) -> tuple:
    """Stable sort key: (date, start_minutes, row_type rank)."""
    return (
        row.get("date") or "",
        int(row.get("start_minutes") or 0),
        _ROW_TYPE_RANK.get(str(row.get("row_type") or ""), 99),
    )


def write_timeline_tsv(
    rows: Iterable[Mapping[str, Any]],
    out_path: Path,
) -> TimelineWriteResult:
    """Sort, normalize and write a list of row dicts as a tab-separated file."""
    cleaned_rows: list[dict[str, str]] = []
    tabs = 0
    for raw in rows:
        cleaned, n = _normalise_cells(dict(raw))
        tabs += n
        for col in TIMELINE_COLUMNS:
            cleaned.setdefault(col, "")
        cleaned_rows.append(cleaned)
    cleaned_rows.sort(key=_row_sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(TIMELINE_COLUMNS) + "\n")
        for r in cleaned_rows:
            fh.write("\t".join(r.get(c, "") for c in TIMELINE_COLUMNS) + "\n")
    return TimelineWriteResult(
        path=out_path,
        row_count=len(cleaned_rows),
        tab_replacements=tabs,
    )


def event_row(
    person_id: str,
    persona_id: str,
    *,
    label: str,
    date: datetime.date,
    start_minutes: int,
    end_minutes: int,
    is_concurrent: bool = False,
    is_dividable: bool = False,
    intensity: int | None = None,
) -> dict[str, Any]:
    """Build one `event` row in the canonical schema."""
    return {
        "person_id": person_id,
        "persona_id": persona_id,
        "row_type": "event",
        "category": "",
        "name": label,
        "label": label,
        "date": date.isoformat(),
        "weekday": _weekday_short(date),
        "start": _format_time(start_minutes),
        "end": _format_time(end_minutes),
        "duration_min": int(end_minutes - start_minutes),
        "start_minutes": int(start_minutes),
        "end_minutes": int(end_minutes),
        "ontology_uri": "",
        "is_concurrent": bool(is_concurrent),
        "is_dividable": bool(is_dividable),
        "concurrent_with": "",
        "parent_task_label": "",
        "intensity": intensity if intensity is not None else "",
        "dimension": "",
        "polarity": "",
        "source": "",
    }


def context_row(
    person_id: str,
    persona_id: str,
    *,
    episode: ContextEpisode,
    catalog: ContextIriCatalog | None = None,
) -> dict[str, Any]:
    """Build one `context` row from a persona- or scenarios-side ContextEpisode."""
    display_label = episode.name
    source = ""
    if catalog is not None and episode.ontology_uri:
        entry = catalog.get(episode.ontology_uri)
        if entry is not None:
            display_label = entry.label or episode.name
            source = entry.source or ""
    return {
        "person_id": person_id,
        "persona_id": persona_id,
        "row_type": "context",
        "category": episode.category,
        "name": episode.name,
        "label": display_label,
        "date": episode.date.isoformat(),
        "weekday": _weekday_short(episode.date),
        "start": _format_time(episode.start_minutes),
        "end": _format_time(episode.end_minutes),
        "duration_min": int(episode.end_minutes - episode.start_minutes),
        "start_minutes": int(episode.start_minutes),
        "end_minutes": int(episode.end_minutes),
        "ontology_uri": episode.ontology_uri or "",
        "is_concurrent": "",
        "is_dividable": "",
        "concurrent_with": "",
        "parent_task_label": "",
        "intensity": "",
        "dimension": episode.dimension or "",
        "polarity": episode.polarity or "",
        "source": source,
    }


def augmented_row(
    person_id: str,
    persona_id: str,
    *,
    task: ScheduledTask,
) -> dict[str, Any]:
    """Build one `augmented_task` row from a ScheduledTask."""
    name = task.task.label
    display = task.task.effective_display_name
    return {
        "person_id": person_id,
        "persona_id": persona_id,
        "row_type": "augmented_task",
        "category": "recommended_task",
        "name": name,
        "label": display,
        "date": task.date.isoformat(),
        "weekday": _weekday_short(task.date),
        "start": _format_time(task.start_minutes),
        "end": _format_time(task.end_minutes),
        "duration_min": int(task.end_minutes - task.start_minutes),
        "start_minutes": int(task.start_minutes),
        "end_minutes": int(task.end_minutes),
        "ontology_uri": task.task.ontology_uri or "",
        "is_concurrent": "true" if task.concurrent_with is not None else "false",
        "is_dividable": bool(task.task.is_dividable),
        "concurrent_with": task.concurrent_with or "",
        "parent_task_label": task.parent_task_label or "",
        "intensity": (
            int(task.task.intensity) if task.task.intensity is not None else ""
        ),
        "dimension": "",
        "polarity": "",
        "source": "",
    }


def write_person_timeline(
    person_id: str,
    persona_id: str,
    schedule: PersonSchedule,
    contexts: Iterable[ContextEpisode],
    augmented: Iterable[ScheduledTask] | None,
    catalog: ContextIriCatalog | None,
    out_path: Path,
) -> TimelineWriteResult:
    """Write the per-person timeline TSV (persona-only when augmented is None, augmented otherwise)."""
    rows: list[dict[str, Any]] = []
    for day in schedule.days:
        for label, instances in day.events.items():
            for inst in instances:
                rows.append(
                    event_row(
                        person_id,
                        persona_id,
                        label=label,
                        date=day.date,
                        start_minutes=inst.start,
                        end_minutes=inst.start + inst.duration,
                    )
                )
        for spill in day.spillovers:
            rows.append(
                event_row(
                    person_id,
                    persona_id,
                    label=spill.event_name,
                    date=day.date,
                    start_minutes=spill.start,
                    end_minutes=spill.start + spill.duration,
                )
            )

    for ep in contexts:
        rows.append(context_row(person_id, persona_id, episode=ep, catalog=catalog))

    if augmented is not None:
        for task in augmented:
            rows.append(augmented_row(person_id, persona_id, task=task))

    return write_timeline_tsv(rows, out_path)


__all__ = [
    "TIMELINE_COLUMNS",
    "TimelineWriteResult",
    "augmented_row",
    "context_row",
    "event_row",
    "write_person_timeline",
    "write_timeline_tsv",
]
