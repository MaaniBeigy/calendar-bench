"""Tests for src.scripts.persona.export.timeline_writer."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest

from src.scripts.persona.context.catalog import CatalogEntry, ContextIriCatalog
from src.scripts.persona.context.schema import ContextEpisode as PersonaContext
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule, Spillover
from src.scripts.persona.export.timeline_writer import (
    TIMELINE_COLUMNS,
    augmented_row,
    context_row,
    event_row,
    write_person_timeline,
    write_timeline_tsv,
)
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask


@pytest.fixture
def catalog() -> ContextIriCatalog:
    return ContextIriCatalog(
        entries=[
            CatalogEntry(
                category="energy_state",
                name="feeling_energetic",
                iri="http://purl.obolibrary.org/obo/MFOEM_000109",
                label="feeling energetic",
                source="MFOEM",
                definition=None,
                polarity=None,
                instrument=None,
                theory_mappings=None,
            ),
        ]
    )


@pytest.fixture
def schedule() -> PersonSchedule:
    day = DaySchedule(
        day_index=0,
        date=datetime.date(2026, 5, 4),
        weekday="Mon",
        events={
            "lunch": [EventInstance(event_name="lunch", start=720, duration=45)],
            "office_work": [
                EventInstance(event_name="office_work", start=540, duration=180)
            ],
        },
        spillovers=[
            Spillover(
                event_name="sleep",
                start=0,
                duration=420,
                orig_start=0,
                orig_duration=420,
                event_idx=0,
            )
        ],
    )
    contexts = [
        PersonaContext(
            name="energetic",
            category="energy_state",
            date=datetime.date(2026, 5, 4),
            start_minutes=540,
            end_minutes=720,
            ontology_uri="http://purl.obolibrary.org/obo/MFOEM_000109",
            dimension="extraversion",
            polarity="high",
        ),
    ]
    return PersonSchedule(
        person_id="h_fulltime_0001",
        persona_id="h_fulltime",
        person_seed=42,
        days=[day],
        contexts=contexts,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Parse a TSV by hand to avoid pandas in unit tests."""
    text = path.read_text(encoding="utf-8")
    lines = text.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def test_column_order_pinned(tmp_path: Path) -> None:
    out = tmp_path / "x.tsv"
    write_timeline_tsv([], out)
    header = out.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert tuple(header) == TIMELINE_COLUMNS


def test_sort_order_date_then_start_then_row_type(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {
            "row_type": "augmented_task",
            "date": "2026-05-04",
            "start_minutes": 540,
            "name": "A",
        },
        {
            "row_type": "context",
            "date": "2026-05-04",
            "start_minutes": 540,
            "name": "C",
        },
        {"row_type": "event", "date": "2026-05-04", "start_minutes": 540, "name": "E"},
        {
            "row_type": "event",
            "date": "2026-05-04",
            "start_minutes": 300,
            "name": "early",
        },
        {
            "row_type": "event",
            "date": "2026-05-05",
            "start_minutes": 540,
            "name": "next",
        },
    ]
    out = tmp_path / "x.tsv"
    write_timeline_tsv(rows, out)
    parsed = _read_rows(out)
    seen = [(r["date"], r["start_minutes"], r["row_type"]) for r in parsed]
    assert seen == [
        ("2026-05-04", "300", "event"),
        ("2026-05-04", "540", "event"),
        ("2026-05-04", "540", "context"),
        ("2026-05-04", "540", "augmented_task"),
        ("2026-05-05", "540", "event"),
    ]


def test_embedded_tabs_replaced_with_space_and_counted(tmp_path: Path) -> None:
    rows = [
        {
            "row_type": "event",
            "date": "2026-05-04",
            "start_minutes": 540,
            "label": "field\twith\ttab",
        },
    ]
    out = tmp_path / "x.tsv"
    result = write_timeline_tsv(rows, out)
    assert result.tab_replacements == 2
    parsed = _read_rows(out)
    assert parsed[0]["label"] == "field with tab"


def test_missing_optional_fields_render_as_empty_string(tmp_path: Path) -> None:
    rows = [
        {"row_type": "context", "date": "2026-05-04", "start_minutes": 540, "name": "x"}
    ]
    out = tmp_path / "x.tsv"
    write_timeline_tsv(rows, out)
    parsed = _read_rows(out)
    for col in TIMELINE_COLUMNS:
        cell = parsed[0][col]
        assert "None" not in cell
        assert "nan" not in cell


def test_write_persona_timeline_emits_events_and_contexts_no_augmented(
    tmp_path: Path,
    schedule: PersonSchedule,
    catalog: ContextIriCatalog,
) -> None:
    out = tmp_path / "person.tsv"
    result = write_person_timeline(
        person_id="h_fulltime_0001",
        persona_id="h_fulltime",
        schedule=schedule,
        contexts=schedule.contexts,
        augmented=None,
        catalog=catalog,
        out_path=out,
    )
    parsed = _read_rows(out)
    row_types = sorted({r["row_type"] for r in parsed})
    assert row_types == ["context", "event"]
    # spillover + 2 events + 1 context
    assert result.row_count == 4
    # Context row carries the catalog source + label, not the persona slug.
    ctx_row = next(r for r in parsed if r["row_type"] == "context")
    assert ctx_row["name"] == "energetic"
    assert ctx_row["label"] == "feeling energetic"
    assert ctx_row["source"] == "MFOEM"


def test_write_persona_timeline_with_augmented_appends_rows(
    tmp_path: Path,
    schedule: PersonSchedule,
    catalog: ContextIriCatalog,
) -> None:
    desired = RecommendedTask(
        label="tea-time",
        duration_min=10,
        duration_max=20,
        intensity=1,
        is_dividable=False,
        is_concurrent=True,
        ontology_uri="https://w3id.org/calendar-bench/health/task/tea-time",
        display_name="Tea Time",
    )
    augmented = [
        ScheduledTask(
            task=desired,
            start_minutes=900,
            end_minutes=915,
            is_standalone=False,
            concurrent_with="lunch",
            date=datetime.date(2026, 5, 4),
            parent_task_label=None,
        ),
    ]
    out = tmp_path / "aug.tsv"
    write_person_timeline(
        person_id="h_fulltime_0001",
        persona_id="h_fulltime",
        schedule=schedule,
        contexts=schedule.contexts,
        augmented=augmented,
        catalog=catalog,
        out_path=out,
    )
    parsed = _read_rows(out)
    aug_rows = [r for r in parsed if r["row_type"] == "augmented_task"]
    assert len(aug_rows) == 1
    row = aug_rows[0]
    assert row["name"] == "tea-time"
    assert row["label"] == "Tea Time"
    assert row["concurrent_with"] == "lunch"
    assert row["is_concurrent"] == "true"
    assert row["intensity"] == "1"


def test_context_row_without_catalog_falls_back_to_local_slug() -> None:
    ep = PersonaContext(
        name="mystery",
        category="energy_state",
        date=datetime.date(2026, 5, 4),
        start_minutes=540,
        end_minutes=600,
        ontology_uri="http://example.org/iri",
    )
    row = context_row("p", "tpl", episode=ep, catalog=None)
    assert row["label"] == "mystery"
    assert row["source"] == ""


def test_context_row_with_iri_not_in_catalog_falls_back_to_local_slug(
    catalog: ContextIriCatalog,
) -> None:
    ep = PersonaContext(
        name="mystery",
        category="energy_state",
        date=datetime.date(2026, 5, 4),
        start_minutes=540,
        end_minutes=600,
        ontology_uri="http://example.org/missing",
    )
    row = context_row("p", "tpl", episode=ep, catalog=catalog)
    assert row["label"] == "mystery"
    assert row["source"] == ""


def test_event_row_optional_intensity_renders_as_empty_when_none() -> None:
    row = event_row(
        "p",
        "tpl",
        label="lunch",
        date=datetime.date(2026, 5, 4),
        start_minutes=720,
        end_minutes=765,
    )
    assert row["intensity"] == ""
    assert row["weekday"] == "Mon"
    assert row["start"] == "12:00"
    assert row["end"] == "12:45"


def test_augmented_row_standalone_intensity_and_parent_label() -> None:
    desired = RecommendedTask(
        label="hydration",
        duration_min=5,
        duration_max=10,
        intensity=2,
        is_dividable=True,
        is_concurrent=False,
    )
    task = ScheduledTask(
        task=desired,
        start_minutes=600,
        end_minutes=610,
        is_standalone=True,
        concurrent_with=None,
        date=datetime.date(2026, 5, 4),
        parent_task_label="hydration",
    )
    row = augmented_row("p", "tpl", task=task)
    assert row["is_concurrent"] == "false"
    assert row["concurrent_with"] == ""
    assert row["parent_task_label"] == "hydration"
    assert row["intensity"] == 2
    assert row["is_dividable"] is True


def test_event_row_clamps_minutes_to_24h_for_display(tmp_path: Path) -> None:
    row = event_row(
        "p",
        "tpl",
        label="after_midnight",
        date=datetime.date(2026, 5, 4),
        start_minutes=25 * 60,
        end_minutes=26 * 60,
    )
    out = tmp_path / "x.tsv"
    write_timeline_tsv([row], out)
    text = out.read_text(encoding="utf-8")
    assert "01:00" in text
    assert "02:00" in text


def test_empty_to_blank_renders_none_as_empty_string(tmp_path: Path) -> None:
    """Direct None values in a row pass through _empty_to_blank (None branch)."""
    rows = [
        {
            "row_type": "event",
            "date": "2026-05-04",
            "start_minutes": 540,
            "label": None,
            "ontology_uri": None,
        }
    ]
    out = tmp_path / "x.tsv"
    write_timeline_tsv(rows, out)
    parsed = _read_rows(out)
    assert parsed[0]["label"] == ""
    assert parsed[0]["ontology_uri"] == ""


def test_empty_to_blank_renders_true_bool_as_lowercase(tmp_path: Path) -> None:
    """A True bool in a row exercises the True branch of the bool ternary."""
    rows = [
        {
            "row_type": "event",
            "date": "2026-05-04",
            "start_minutes": 540,
            "is_concurrent": True,
            "is_dividable": True,
        }
    ]
    out = tmp_path / "x.tsv"
    write_timeline_tsv(rows, out)
    parsed = _read_rows(out)
    assert parsed[0]["is_concurrent"] == "true"
    assert parsed[0]["is_dividable"] == "true"
