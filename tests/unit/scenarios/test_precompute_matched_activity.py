"""Tests for `scenarios.precompute_matched_activity`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from src.scripts.scenarios import precompute_matched_activity as mod

HA_PREFIX = mod.HA_INSTANCE_PREFIX
HB_PREFIX = mod.HB_TASK_PREFIX


@pytest.fixture()
def ha_ttl(tmp_path: Path) -> Path:
    """Minimal HumanActivities TTL with three activity instances."""
    body = """
@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .

ha:SportsExerciseWorkoutConditioningExercise a owl:Class ;
    rdfs:label "Conditioning Exercise"@en .
ha:EverydayTasksSelfCare a owl:Class ;
    rdfs:label "Self Care"@en .

ha-act:aerobic-general a ha:SportsExerciseWorkoutConditioningExercise ;
    rdfs:label "Aerobic, general"@en .

ha-act:eating-sitting a ha:EverydayTasksSelfCare ;
    rdfs:label "Eating, sitting"@en .

ha-act:bicycling-general a ha:SportsExerciseWorkoutConditioningExercise ;
    rdfs:label "Bicycling, general"@en .
"""
    path = tmp_path / "ha.ttl"
    path.write_text(body, encoding="utf-8")
    return path


def _write_xlsx(
    path: Path,
    rows: list[tuple[str, str, str]],
    headers: list[str] | None = None,
) -> Path:
    """Write a minimal xlsx with the columns the resolver reads."""
    wb = Workbook()
    sheet = wb.active
    sheet.append(headers or ["task_name", "matched_activity", "matched_activity_note"])
    for triple in rows:
        sheet.append(list(triple))
    wb.save(path.as_posix())
    return path


def test_normalize_collapses_whitespace_and_dashes():
    assert mod._normalize("  Foo,   bar–baz  ") == "foo, bar-baz"


def test_slugify_replaces_punctuation():
    assert mod._slugify("Do a 10-Minute Walk 🚶") == "do-a-10-minute-walk"


def test_strip_emoji_suffix_keeps_inner_punctuation():
    assert (
        mod._strip_emoji_suffix("Plan This Week's Meals 🗓️") == "Plan This Week's Meals"
    )
    assert mod._strip_emoji_suffix("No emoji here") == "No emoji here"


def test_split_multi_no_separator_returns_single():
    assert mod._split_multi("Aerobic, general") == ["Aerobic, general"]


def test_split_multi_double_pipe_separator():
    assert mod._split_multi("Aerobic, general || Bicycling, general") == [
        "Aerobic, general",
        "Bicycling, general",
    ]


def test_split_multi_blank_parts_dropped():
    assert mod._split_multi(" || Aerobic, general || ") == ["Aerobic, general"]


def test_load_label_index_skips_classes(ha_ttl: Path):
    index = mod.load_label_index(ha_ttl)
    assert "aerobic, general" in index
    assert index["aerobic, general"] == f"{HA_PREFIX}aerobic-general"
    # Class-level labels live outside the activity prefix and are dropped.
    assert "conditioning exercise" not in index
    assert "self care" not in index


def test_resolve_rows_happy_path(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(
        tmp_path / "in.xlsx",
        [
            ("Do 10 Minutes of Cardio 💪", "Aerobic, general", "Generic cardio"),
            ("Schedule a 30-Minute Walk 🚶", "Bicycling, general", ""),
        ],
    )
    resolved, unresolved = mod.resolve_rows(xlsx, ha_ttl)
    assert unresolved == []
    assert [r.task_slug for r in resolved] == [
        "do-10-minutes-of-cardio",
        "schedule-a-30-minute-walk",
    ]
    assert resolved[0].matched_activity_iris == (f"{HA_PREFIX}aerobic-general",)
    assert resolved[0].matched_activity_labels == ("Aerobic, general",)
    assert resolved[0].note == "Generic cardio"
    assert resolved[0].task_iri == f"{HB_PREFIX}do-10-minutes-of-cardio"


def test_resolve_rows_multi_target(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(
        tmp_path / "multi.xlsx",
        [("Cardio Choice 💪", "Aerobic, general || Bicycling, general", "")],
    )
    resolved, unresolved = mod.resolve_rows(xlsx, ha_ttl)
    assert unresolved == []
    assert len(resolved) == 1
    assert resolved[0].matched_activity_iris == (
        f"{HA_PREFIX}aerobic-general",
        f"{HA_PREFIX}bicycling-general",
    )


def test_resolve_rows_label_normalization(tmp_path: Path, ha_ttl: Path):
    # Excess whitespace, different case, em-dash in the curated label.
    xlsx = _write_xlsx(
        tmp_path / "norm.xlsx",
        [("Do Cardio", "  aerobic,    general  ", "")],
    )
    resolved, _ = mod.resolve_rows(xlsx, ha_ttl)
    assert resolved[0].matched_activity_iris == (f"{HA_PREFIX}aerobic-general",)


def test_resolve_rows_unresolved_label_collected(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(
        tmp_path / "bad.xlsx",
        [("Run a Mile", "Running fast but not in ontology", "")],
    )
    resolved, unresolved = mod.resolve_rows(xlsx, ha_ttl)
    assert resolved == []
    assert len(unresolved) == 1
    assert unresolved[0].task_slug == "run-a-mile"
    assert "Running fast" in unresolved[0].raw_label


def test_resolve_rows_empty_cell_reported(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(
        tmp_path / "empty.xlsx",
        [("Skip Me", "", "")],
    )
    resolved, unresolved = mod.resolve_rows(xlsx, ha_ttl)
    assert resolved == []
    assert unresolved[0].raw_label == ""


def test_resolve_rows_skips_rows_with_no_task_name(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(
        tmp_path / "no-name.xlsx",
        [("", "Aerobic, general", "")],
    )
    resolved, unresolved = mod.resolve_rows(xlsx, ha_ttl)
    assert resolved == []
    assert unresolved == []


def test_resolve_rows_partial_multi_target_drops_row(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(
        tmp_path / "partial.xlsx",
        [("Mixed", "Aerobic, general || Unknown label", "")],
    )
    resolved, unresolved = mod.resolve_rows(xlsx, ha_ttl)
    assert resolved == []
    assert any("Unknown label" in r.raw_label for r in unresolved)


def test_read_xlsx_rows_handles_empty_sheet(tmp_path: Path):
    wb = Workbook()
    wb.active.title = "empty"
    path = tmp_path / "empty.xlsx"
    wb.save(path.as_posix())
    # Force a truly empty rows iterator path; iter_rows on a fresh sheet
    # returns one all-None row; the loader builds an empty header from it.
    rows = mod._read_xlsx_rows(path)
    assert rows == [] or all(not any(v for v in r.values()) for r in rows)


def test_write_jsonl_round_trip(tmp_path: Path):
    rows = [
        mod.ResolvedRow(
            task_slug="t1",
            task_iri=f"{HB_PREFIX}t1",
            matched_activity_labels=("Aerobic, general",),
            matched_activity_iris=(f"{HA_PREFIX}aerobic-general",),
            note="ok",
        )
    ]
    out = tmp_path / "out.jsonl"
    count = mod.write_jsonl(rows, out)
    assert count == 1
    line = out.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["task_slug"] == "t1"
    assert payload["matched_activity_iris"] == [f"{HA_PREFIX}aerobic-general"]
    assert payload["note"] == "ok"
    assert payload["ts"]


def test_run_writes_file_on_success(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(tmp_path / "ok.xlsx", [("Cardio", "Aerobic, general", "")])
    out = tmp_path / "result.jsonl"
    count = mod.run(xlsx, ha_ttl, out)
    assert count == 1
    assert out.exists()


def test_run_aborts_on_unresolved(tmp_path: Path, ha_ttl: Path):
    xlsx = _write_xlsx(tmp_path / "bad.xlsx", [("Cardio", "Made-up activity", "")])
    out = tmp_path / "result.jsonl"
    with pytest.raises(mod.LabelResolutionError) as exc:
        mod.run(xlsx, ha_ttl, out)
    assert "Made-up activity" in str(exc.value)
    assert not out.exists()


def test_format_unresolved_reports_empty_label():
    msg = mod._format_unresolved(
        [mod.UnresolvedRow(task_slug="t", raw_label="", nearest_matches=())]
    )
    assert "empty cell" in msg


def test_format_unresolved_lists_nearest_matches():
    msg = mod._format_unresolved(
        [
            mod.UnresolvedRow(
                task_slug="t",
                raw_label="aerobix",
                nearest_matches=("aerobic, general",),
            )
        ]
    )
    assert "nearest" in msg
    assert "aerobic, general" in msg


def test_main_exits_zero_on_success(tmp_path: Path, ha_ttl: Path, capsys):
    xlsx = _write_xlsx(tmp_path / "ok.xlsx", [("Cardio", "Aerobic, general", "")])
    out = tmp_path / "out.jsonl"
    rc = mod.main(
        [
            "--xlsx",
            xlsx.as_posix(),
            "--ha-ttl",
            ha_ttl.as_posix(),
            "--out",
            out.as_posix(),
        ]
    )
    assert rc == 0
    assert "resolved 1 tasks" in capsys.readouterr().out


def test_main_exits_two_on_unresolved(tmp_path: Path, ha_ttl: Path, capsys):
    xlsx = _write_xlsx(tmp_path / "bad.xlsx", [("Cardio", "Made-up activity", "")])
    out = tmp_path / "out.jsonl"
    rc = mod.main(
        [
            "--xlsx",
            xlsx.as_posix(),
            "--ha-ttl",
            ha_ttl.as_posix(),
            "--out",
            out.as_posix(),
        ]
    )
    assert rc == 2
    assert "Made-up activity" in capsys.readouterr().err
