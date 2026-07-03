"""Tests for the divide_breakdown aggregator + report writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.export.report_writer import (
    DIVIDE_REPORT_BASENAME,
    divide_breakdown_to_dict,
    format_divide_breakdown_report,
    write_evaluation_reports,
)
from src.scripts.scenarios.metrics.divide_breakdown import read_divide_verdicts


def _row(
    *,
    person_id: str = "p1",
    label: str = "walk",
    iso_year: int = 2026,
    iso_week: int = 19,
    instances: int = 1,
    original: int = 120,
    pieces: tuple[dict, ...] = (
        {
            "date": "2026-05-04",
            "start_minutes": 480,
            "end_minutes": 540,
            "duration_minutes": 60,
        },
        {
            "date": "2026-05-06",
            "start_minutes": 900,
            "end_minutes": 960,
            "duration_minutes": 60,
        },
    ),
    sum_minutes: int = 120,
    band: tuple[float, float] = (102.0, 138.0),
    verdict: str = "divided_valid",
) -> dict:
    return {
        "person_id": person_id,
        "iso_year": iso_year,
        "iso_week": iso_week,
        "task_label": label,
        "instances_in_week": instances,
        "original_duration_minutes": original,
        "pieces": list(pieces),
        "sum_minutes": sum_minutes,
        "tolerance_band": list(band),
        "verdict": verdict,
    }


def _write_sidecar(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return path


class TestReadDivideVerdicts:
    def test_aggregates_per_label_verdict_counts_across_persons(self, tmp_path: Path):
        _write_sidecar(
            tmp_path / "p1.jsonl",
            [
                _row(person_id="p1", label="walk", verdict="divided_valid"),
                _row(
                    person_id="p1",
                    label="plan_meals",
                    pieces=(
                        {
                            "date": "2026-05-04",
                            "start_minutes": 540,
                            "end_minutes": 555,
                            "duration_minutes": 15,
                        },
                    ),
                    sum_minutes=15,
                    original=45,
                    band=(38.25, 51.75),
                    verdict="not_divided",
                ),
            ],
        )
        _write_sidecar(
            tmp_path / "p2.jsonl",
            [
                _row(
                    person_id="p2",
                    label="walk",
                    iso_week=20,
                    verdict="divided_invalid_sum_too_high",
                ),
                _row(
                    person_id="p2", label="walk", iso_week=21, verdict="divided_valid"
                ),
            ],
        )
        out = read_divide_verdicts(list(tmp_path.glob("*.jsonl")))
        assert out["applicable_persons"] == 2
        # weeks: (2026, 19), (2026, 20), (2026, 21).
        assert out["applicable_weeks"] == 3
        assert out["applicable_buckets"] == 4
        counts = out["verdict_counts"]
        assert counts["divided_valid"] == 2
        assert counts["not_divided"] == 1
        assert counts["divided_invalid_sum_too_high"] == 1
        assert counts["divided_invalid_sum_too_low"] == 0
        assert counts["divided_invalid_pieces_too_long"] == 0
        assert out["per_label"]["walk"]["buckets"] == 3
        assert out["per_label"]["walk"]["divided_valid"] == 2
        assert out["per_label"]["walk"]["mean_pieces_per_bucket"] == pytest.approx(2.0)
        assert out["per_label"]["plan_meals"]["buckets"] == 1
        assert out["per_label"]["plan_meals"]["divided_valid"] == 0
        assert out["per_label"]["plan_meals"]["mean_sum_minutes"] == pytest.approx(15.0)

    def test_skips_missing_files(self, tmp_path: Path):
        out = read_divide_verdicts([tmp_path / "does_not_exist.jsonl"])
        assert out["applicable_buckets"] == 0
        assert out["applicable_persons"] == 0

    def test_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "p.jsonl"
        # Blank line sits in the middle so `splitlines()` keeps it.
        path.write_text(
            "\n".join(
                [
                    "{not valid json",
                    "",  # blank line in the middle exercises the blank-skip branch
                    json.dumps([1, 2, 3]),  # not a dict
                    json.dumps(_row()),
                ]
            ),
            encoding="utf-8",
        )
        out = read_divide_verdicts([path])
        assert out["applicable_buckets"] == 1

    def test_skips_unknown_verdict(self, tmp_path: Path):
        path = tmp_path / "p.jsonl"
        bad = _row(verdict="totally_made_up")
        good = _row(verdict="divided_valid")
        _write_sidecar(path, [bad, good])
        out = read_divide_verdicts([path])
        assert out["applicable_buckets"] == 1
        assert out["verdict_counts"]["divided_valid"] == 1

    def test_skips_row_without_task_label(self, tmp_path: Path):
        path = tmp_path / "p.jsonl"
        bad = _row()
        bad.pop("task_label")
        _write_sidecar(path, [bad])
        out = read_divide_verdicts([path])
        assert out["applicable_buckets"] == 0

    def test_handles_row_without_person_id(self, tmp_path: Path):
        path = tmp_path / "p.jsonl"
        row = _row()
        row.pop("person_id")
        _write_sidecar(path, [row])
        out = read_divide_verdicts([path])
        assert out["applicable_buckets"] == 1
        assert out["applicable_persons"] == 0

    def test_handles_row_without_iso_week(self, tmp_path: Path):
        path = tmp_path / "p.jsonl"
        row = _row()
        row.pop("iso_week")
        _write_sidecar(path, [row])
        out = read_divide_verdicts([path])
        assert out["applicable_weeks"] == 0
        assert out["applicable_buckets"] == 1

    def test_handles_row_without_pieces_or_sum(self, tmp_path: Path):
        path = tmp_path / "p.jsonl"
        row = _row()
        row.pop("pieces")
        row.pop("sum_minutes")
        _write_sidecar(path, [row])
        out = read_divide_verdicts([path])
        per_label = out["per_label"]["walk"]
        assert per_label["mean_sum_minutes"] == 0.0
        assert per_label["mean_pieces_per_bucket"] == 0.0


class TestFormatDivideBreakdownReport:
    def test_renders_totals_and_percentages(self):
        breakdown = {
            "applicable_persons": 3,
            "applicable_weeks": 4,
            "applicable_buckets": 4,
            "verdict_counts": {
                "divided_valid": 2,
                "not_divided": 1,
                "divided_invalid_pieces_too_long": 0,
                "divided_invalid_sum_too_low": 0,
                "divided_invalid_sum_too_high": 1,
            },
            "per_label": {
                "walk": {
                    "buckets": 3,
                    "divided_valid": 2,
                    "mean_sum_minutes": 100.0,
                    "mean_pieces_per_bucket": 2.0,
                },
                "plan_meals": {
                    "buckets": 1,
                    "divided_valid": 0,
                    "mean_sum_minutes": 15.0,
                    "mean_pieces_per_bucket": 1.0,
                },
            },
        }
        text = format_divide_breakdown_report(breakdown)
        assert "L_divide Cohort Breakdown" in text
        assert "applicable persons : 3" in text
        assert "applicable weeks   : 4" in text
        assert "applicable buckets : 4" in text
        assert "divided_valid" in text
        assert "( 50.0%)" in text  # 2 / 4
        assert "( 25.0%)" in text  # 1 / 4
        assert "walk" in text
        assert "plan_meals" in text

    def test_renders_without_per_label_section_when_empty(self):
        breakdown = {
            "applicable_persons": 1,
            "applicable_weeks": 1,
            "applicable_buckets": 1,
            "verdict_counts": {
                "divided_valid": 1,
                "not_divided": 0,
                "divided_invalid_pieces_too_long": 0,
                "divided_invalid_sum_too_low": 0,
                "divided_invalid_sum_too_high": 0,
            },
            "per_label": {},
        }
        text = format_divide_breakdown_report(breakdown)
        assert "applicable buckets : 1" in text
        # The per-label sub-section header must NOT appear when per_label is empty.
        assert "per dividable label" not in text

    def test_handles_zero_buckets(self):
        breakdown = {
            "applicable_persons": 0,
            "applicable_weeks": 0,
            "applicable_buckets": 0,
            "verdict_counts": {
                v: 0
                for v in (
                    "divided_valid",
                    "not_divided",
                    "divided_invalid_pieces_too_long",
                    "divided_invalid_sum_too_low",
                    "divided_invalid_sum_too_high",
                )
            },
            "per_label": {},
        }
        text = format_divide_breakdown_report(breakdown)
        assert "n/a" in text

    def test_passthrough_to_dict(self):
        b = {"applicable_buckets": 1, "verdict_counts": {"divided_valid": 1}}
        assert divide_breakdown_to_dict(b) == b


class TestWriteEvaluationReportsDivideSection:
    def test_writes_pair_when_breakdown_supplied(self, tmp_path: Path):
        breakdown = {
            "applicable_persons": 1,
            "applicable_weeks": 1,
            "applicable_buckets": 1,
            "verdict_counts": {
                "divided_valid": 1,
                "not_divided": 0,
                "divided_invalid_pieces_too_long": 0,
                "divided_invalid_sum_too_low": 0,
                "divided_invalid_sum_too_high": 0,
            },
            "per_label": {
                "walk": {
                    "buckets": 1,
                    "divided_valid": 1,
                    "mean_sum_minutes": 120.0,
                    "mean_pieces_per_bucket": 2.0,
                }
            },
        }
        written = write_evaluation_reports(
            results=[], out_dir=tmp_path, divide_breakdown=breakdown
        )
        assert DIVIDE_REPORT_BASENAME in written
        txt, js = written[DIVIDE_REPORT_BASENAME]
        assert txt.exists() and js.exists()
        payload = json.loads(js.read_text(encoding="utf-8"))
        assert payload["applicable_buckets"] == 1

    def test_skipped_when_breakdown_absent(self, tmp_path: Path):
        written = write_evaluation_reports(results=[], out_dir=tmp_path)
        assert DIVIDE_REPORT_BASENAME not in written

    def test_skipped_when_breakdown_empty(self, tmp_path: Path):
        written = write_evaluation_reports(
            results=[], out_dir=tmp_path, divide_breakdown={}
        )
        assert DIVIDE_REPORT_BASENAME not in written
