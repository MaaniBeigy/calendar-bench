"""write_context_fit_sidecar emits one JSONL line per (task, recommended_category) pair."""

from __future__ import annotations

import json
from pathlib import Path

from src.scripts.scenarios.export.json_writer import write_context_fit_sidecar
from src.scripts.scenarios.metrics.context_fit import ContextFitVerdict


def _verdict(
    task_label: str, *, recommended: tuple[str, ...], overlapped: tuple[str, ...]
) -> ContextFitVerdict:
    # No observation budget supplied; scored set equals recommended set
    # (legacy `fit_full` denominator) so the sidecar keeps emitting one
    # row per recommended category exactly as it did before the
    # observed-only scoping change.
    fit_full = len(overlapped) / len(recommended) if recommended else 0.0
    return ContextFitVerdict(
        task_label=task_label,
        task_uri=f"https://w3id.org/calendar-bench/health/task/{task_label}",
        recommended_categories=recommended,
        observed_categories=(),
        scored_categories=recommended,
        overlapped_categories=overlapped,
        fit=fit_full,
        fit_full=fit_full,
    )


def test_writer_returns_none_when_no_verdicts(tmp_path: Path) -> None:
    out = write_context_fit_sidecar(
        tmp_path / "p1_context_fit.jsonl", person_id="p1", verdicts=[]
    )
    assert out is None
    assert not (tmp_path / "p1_context_fit.jsonl").exists()


def test_writer_emits_one_row_per_recommended_category(tmp_path: Path) -> None:
    verdicts = [
        _verdict(
            "tea",
            recommended=("mood_emotion", "location"),
            overlapped=("mood_emotion",),
        ),
        _verdict("walk", recommended=("energy_state",), overlapped=()),
    ]
    out = write_context_fit_sidecar(
        tmp_path / "p1_context_fit.jsonl", person_id="p1", verdicts=verdicts
    )
    assert out is not None and out.exists()
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    by_task_cat = {(r["task_label"], r["recommended_category"]): r for r in rows}
    assert by_task_cat[("tea", "mood_emotion")]["overlapped"] is True
    assert by_task_cat[("tea", "location")]["overlapped"] is False
    assert by_task_cat[("walk", "energy_state")]["overlapped"] is False
    assert all(r["person_id"] == "p1" for r in rows)


def test_writer_preserves_recommended_category_order(tmp_path: Path) -> None:
    verdicts = [
        _verdict("t", recommended=("c", "a", "b"), overlapped=("a",)),
    ]
    out = write_context_fit_sidecar(
        tmp_path / "p_context_fit.jsonl", person_id="p", verdicts=verdicts
    )
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert [r["recommended_category"] for r in rows] == ["c", "a", "b"]
