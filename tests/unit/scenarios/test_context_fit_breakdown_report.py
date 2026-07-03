"""Cohort context_fit_breakdown aggregator + one-row-per-category text renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.metrics.context_fit_breakdown import (
    format_context_fit_breakdown,
    read_context_fit_verdicts,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_zero_applicable_persons_returns_empty_per_category(tmp_path: Path) -> None:
    payload = read_context_fit_verdicts([])
    assert payload == {
        "applicable_persons": 0,
        "applicable_tasks": 0,
        "per_category": {},
    }


def test_reader_skips_missing_files(tmp_path: Path) -> None:
    payload = read_context_fit_verdicts([tmp_path / "missing.jsonl"])
    assert payload["applicable_persons"] == 0
    assert payload["applicable_tasks"] == 0


def test_reader_skips_malformed_lines(tmp_path: Path) -> None:
    bad = tmp_path / "p1_context_fit.jsonl"
    bad.write_text(
        "not json\n"
        + json.dumps(
            {
                "person_id": "p1",
                "task_label": "t",
                "recommended_category": "mood_emotion",
                "overlapped": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = read_context_fit_verdicts([bad])
    assert payload["applicable_persons"] == 1
    assert payload["applicable_tasks"] == 1
    assert payload["per_category"]["mood_emotion"]["pairs"] == 1


def test_reader_skips_non_object_rows_and_non_string_fields(tmp_path: Path) -> None:
    """Rows that are not dicts, lack `recommended_category`, or carry non-string ids are dropped."""
    path = tmp_path / "p_context_fit.jsonl"
    rows = [
        json.dumps([1, 2, 3]),  # not a dict
        json.dumps({"recommended_category": 42, "overlapped": True}),  # cat not str
        json.dumps(
            {
                "person_id": 9,
                "task_label": "t",
                "recommended_category": "mood_emotion",
                "overlapped": True,
            }
        ),  # person_id not str
        json.dumps(
            {
                "person_id": "p",
                "task_label": 5,
                "recommended_category": "mood_emotion",
                "overlapped": True,
            }
        ),  # task_label not str
        json.dumps(
            {
                "person_id": "p",
                "task_label": "t",
                "recommended_category": "mood_emotion",
                "overlapped": True,
            }
        ),  # valid
        "",  # blank line
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = read_context_fit_verdicts([path])
    assert (
        payload["per_category"]["mood_emotion"]["pairs"] == 3
    )  # three valid `recommended_category` rows
    # Only one row provides both person_id (str) and task_label (str).
    assert payload["applicable_tasks"] == 1


def test_aggregates_pairs_overlap_and_percentage_per_category(tmp_path: Path) -> None:
    _write(
        tmp_path / "p1_context_fit.jsonl",
        [
            {
                "person_id": "p1",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "mood_emotion",
                "overlapped": True,
            },
            {
                "person_id": "p1",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "location",
                "overlapped": False,
            },
            {
                "person_id": "p1",
                "task_label": "walk",
                "task_uri": "u2",
                "recommended_category": "mood_emotion",
                "overlapped": False,
            },
        ],
    )
    _write(
        tmp_path / "p2_context_fit.jsonl",
        [
            {
                "person_id": "p2",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "mood_emotion",
                "overlapped": True,
            },
        ],
    )
    payload = read_context_fit_verdicts(list(tmp_path.glob("*.jsonl")))
    assert payload["applicable_persons"] == 2
    assert payload["applicable_tasks"] == 3  # (p1,tea), (p1,walk), (p2,tea)
    mood = payload["per_category"]["mood_emotion"]
    assert mood["pairs"] == 3 and mood["overlapped"] == 2
    assert mood["overlap_pct"] == pytest.approx(200.0 / 3)
    assert payload["per_category"]["location"]["overlap_pct"] == pytest.approx(0.0)


def test_text_report_falls_back_when_no_categories() -> None:
    text = format_context_fit_breakdown(
        {"applicable_persons": 0, "applicable_tasks": 0, "per_category": {}}
    )
    assert "L_context_fit breakdown" in text
    assert "no context_links recommendations" in text


def test_text_report_renders_one_row_per_category_with_one_decimal_pct() -> None:
    payload = {
        "applicable_persons": 2,
        "applicable_tasks": 3,
        "per_category": {
            "mood_emotion": {"pairs": 3, "overlapped": 2, "overlap_pct": 200.0 / 3},
            "location": {"pairs": 1, "overlapped": 0, "overlap_pct": 0.0},
        },
    }
    text = format_context_fit_breakdown(payload)
    assert "mood_emotion" in text and "66.7%" in text
    assert "location" in text and "0.0%" in text


# ---------------------------------------------------------------------------
# Observed-only scoping
# ---------------------------------------------------------------------------


def test_reader_skips_rows_marked_out_of_scope(tmp_path: Path) -> None:
    """Rows with `in_scope: False` (categories outside the observation budget) do not appear in per_category."""
    _write(
        tmp_path / "p1_context_fit.jsonl",
        [
            {
                "person_id": "p1",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "mood_emotion",
                "overlapped": True,
                "in_scope": True,
            },
            {
                "person_id": "p1",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "weather_environment",
                "overlapped": False,
                "in_scope": False,
            },
            {
                "person_id": "p1",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "physiological",
                "overlapped": False,
                "in_scope": False,
            },
        ],
    )
    payload = read_context_fit_verdicts([tmp_path / "p1_context_fit.jsonl"])
    assert "mood_emotion" in payload["per_category"]
    assert "weather_environment" not in payload["per_category"]
    assert "physiological" not in payload["per_category"]
    assert payload["applicable_persons"] == 1
    assert payload["applicable_tasks"] == 1


def test_reader_keeps_legacy_rows_without_in_scope_flag(tmp_path: Path) -> None:
    """Sidecars written before the observed-only scoping change lack `in_scope`; the reader counts every recommended pair (back-compat)."""
    _write(
        tmp_path / "p1_legacy.jsonl",
        [
            {
                "person_id": "p1",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "mood_emotion",
                "overlapped": True,
            },
            {
                "person_id": "p1",
                "task_label": "tea",
                "task_uri": "u1",
                "recommended_category": "weather_environment",
                "overlapped": False,
            },
        ],
    )
    payload = read_context_fit_verdicts([tmp_path / "p1_legacy.jsonl"])
    assert "mood_emotion" in payload["per_category"]
    assert "weather_environment" in payload["per_category"]
