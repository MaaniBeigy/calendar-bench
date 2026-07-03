"""Tests for the per-scenario evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path


def test_write_evaluation_reports_emits_context_fit_pair(tmp_path: Path) -> None:
    """A non-empty context_fit_breakdown writes both txt and json sidecars."""
    from src.scripts.scenarios.export.report_writer import (
        CONTEXT_FIT_REPORT_BASENAME,
        write_evaluation_reports,
    )

    breakdown = {
        "applicable_tasks": 5,
        "by_leg": {"weather_environment": {"applicable_tasks": 5, "fit_mean": 0.6}},
    }
    written = write_evaluation_reports(
        results=[],
        out_dir=tmp_path,
        context_fit_breakdown=breakdown,
    )
    assert CONTEXT_FIT_REPORT_BASENAME in written
    paths = written[CONTEXT_FIT_REPORT_BASENAME]
    assert any(str(p).endswith(".txt") for p in paths)
    assert any(str(p).endswith(".json") for p in paths)


def test_window_line_absent_with_partial_window_dict(tmp_path: Path) -> None:
    """A window dict missing any of start/end/weeks renders no line."""
    from src.scripts.scenarios.export.report_writer import (
        TOTAL_REPORT_BASENAME,
        write_evaluation_reports,
    )

    partial = {"start_date": "2026-06-01", "weeks": 2}  # no end_date_inclusive
    written = write_evaluation_reports(results=[], out_dir=tmp_path, window=partial)
    txt_path = next(
        p for p in written[TOTAL_REPORT_BASENAME] if str(p).endswith(".txt")
    )
    assert "Window:" not in txt_path.read_text(encoding="utf-8")


def test_window_line_absent_without_window(tmp_path: Path) -> None:
    from src.scripts.scenarios.export.report_writer import (
        TOTAL_REPORT_BASENAME,
        write_evaluation_reports,
    )

    written = write_evaluation_reports(results=[], out_dir=tmp_path)
    txt_path = next(
        p for p in written[TOTAL_REPORT_BASENAME] if str(p).endswith(".txt")
    )
    assert "Window:" not in txt_path.read_text(encoding="utf-8")


def test_window_line_present_when_window_given(tmp_path: Path) -> None:
    from src.scripts.scenarios.export.report_writer import (
        PERSON_REPORT_BASENAME,
        TOTAL_REPORT_BASENAME,
        write_evaluation_reports,
    )

    window = {
        "start_date": "2026-06-01",
        "end_date_inclusive": "2026-06-14",
        "weeks": 2,
        "days": 14,
    }
    written = write_evaluation_reports(results=[], out_dir=tmp_path, window=window)
    for basename in (PERSON_REPORT_BASENAME, TOTAL_REPORT_BASENAME):
        txt_path = next(p for p in written[basename] if str(p).endswith(".txt"))
        json_path = next(p for p in written[basename] if str(p).endswith(".json"))
        text = txt_path.read_text(encoding="utf-8")
        assert "Window: 2026-06-01 .. 2026-06-14 (2 weeks)" in text
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["window"] == window
