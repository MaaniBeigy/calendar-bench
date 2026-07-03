"""Tests for the `divide_verdicts.jsonl` sidecar writer."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.export.json_writer import write_divide_verdicts_sidecar
from src.scripts.scenarios.metrics.loss import DivideVerdictRecord


def _record(
    *,
    label: str = "walk",
    iso_year: int = 2026,
    iso_week: int = 19,
    instances: int = 1,
    original: int = 120,
    pieces: tuple[tuple[str, int, int, int], ...] = (
        ("2026-05-04", 480, 540, 60),
        ("2026-05-06", 900, 960, 60),
    ),
    band: tuple[float, float] = (102.0, 138.0),
    sum_minutes: int = 120,
    verdict: str = "divided_valid",
) -> DivideVerdictRecord:
    return DivideVerdictRecord(
        iso_year=iso_year,
        iso_week=iso_week,
        task_label=label,
        instances_in_week=instances,
        original_duration_minutes=original,
        piece_durations=tuple(d for _, _, _, d in pieces),
        piece_dates=tuple(p[0] for p in pieces),
        piece_starts=tuple(p[1] for p in pieces),
        piece_ends=tuple(p[2] for p in pieces),
        sum_minutes=sum_minutes,
        tolerance_band=band,
        verdict=verdict,
    )


class TestWriteDivideVerdictsSidecar:
    def test_writes_one_line_per_bucket(self, tmp_path: Path):
        out = tmp_path / "person_001_divide_verdicts.jsonl"
        records = [
            _record(label="walk", verdict="divided_valid"),
            _record(
                label="plan_meals",
                instances=1,
                original=45,
                pieces=(("2026-05-04", 540, 555, 15),),
                band=(38.25, 51.75),
                sum_minutes=15,
                verdict="not_divided",
            ),
        ]
        result = write_divide_verdicts_sidecar(
            out, person_id="person_001", records=records
        )
        assert result == out
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        rows = [json.loads(line) for line in lines]
        assert rows[0]["task_label"] == "walk"
        assert rows[0]["verdict"] == "divided_valid"
        assert rows[1]["task_label"] == "plan_meals"
        assert rows[1]["verdict"] == "not_divided"

    def test_omits_file_when_records_empty(self, tmp_path: Path):
        out = tmp_path / "person_001_divide_verdicts.jsonl"
        result = write_divide_verdicts_sidecar(out, person_id="person_001", records=[])
        assert result is None
        assert not out.exists()

    def test_round_trips_via_json_loads_per_line(self, tmp_path: Path):
        out = tmp_path / "p_002.jsonl"
        rec = _record(verdict="divided_invalid_sum_too_high")
        write_divide_verdicts_sidecar(out, person_id="p_002", records=[rec])
        row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert row["person_id"] == "p_002"
        assert row["iso_year"] == rec.iso_year
        assert row["iso_week"] == rec.iso_week
        assert row["instances_in_week"] == rec.instances_in_week
        assert row["original_duration_minutes"] == rec.original_duration_minutes
        assert row["sum_minutes"] == rec.sum_minutes
        assert row["tolerance_band"] == list(rec.tolerance_band)
        assert row["verdict"] == rec.verdict
        assert len(row["pieces"]) == len(rec.piece_dates)
        assert row["pieces"][0] == {
            "date": rec.piece_dates[0],
            "start_minutes": rec.piece_starts[0],
            "end_minutes": rec.piece_ends[0],
            "duration_minutes": rec.piece_durations[0],
        }

    def test_creates_parent_directory_when_missing(self, tmp_path: Path):
        out = tmp_path / "nested" / "deeper" / "p.jsonl"
        write_divide_verdicts_sidecar(out, person_id="p", records=[_record()])
        assert out.exists()

    def test_uses_generator_input(self, tmp_path: Path):
        out = tmp_path / "p.jsonl"
        gen = (r for r in [_record(label="a"), _record(label="b")])
        write_divide_verdicts_sidecar(out, person_id="p", records=gen)
        rows = [
            json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
        ]
        assert [r["task_label"] for r in rows] == ["a", "b"]


class TestSidecarEndToEnd:
    def test_records_from_compute_l_divide_round_trip(self, tmp_path: Path):
        from src.scripts.scenarios.domain.calendar import AugmentedCalendar
        from src.scripts.scenarios.domain.solution import SchedulingSolution
        from src.scripts.scenarios.metrics.loss import compute_l_divide

        date = _dt.date(2026, 5, 4)
        task = RecommendedTask(
            label="walk", duration_min=30, duration_max=120, is_dividable=True
        )
        scheduled = [
            ScheduledTask(
                task=task,
                start_minutes=480,
                end_minutes=540,
                is_standalone=True,
                concurrent_with=None,
                date=date,
            ),
            ScheduledTask(
                task=task,
                start_minutes=900,
                end_minutes=960,
                is_standalone=True,
                concurrent_with=None,
                date=date,
            ),
        ]
        sol = SchedulingSolution(
            person_id="p",
            augmented_calendar=AugmentedCalendar(
                person_id="p", scheduled_tasks=scheduled
            ),
            tasks=[task],
            scheduled=scheduled,
        )
        _, records = compute_l_divide(sol)
        out = tmp_path / "p_divide_verdicts.jsonl"
        write_divide_verdicts_sidecar(out, person_id="p", records=records)
        row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert row["verdict"] == "divided_valid"
        assert row["sum_minutes"] == 120
