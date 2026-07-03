"""Write augmented per-person schedule as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask


def _recommended_task_to_dict(task: RecommendedTask) -> dict:
    return {
        "label": task.label,
        "display_name": task.effective_display_name,
        "description": task.effective_description,
        "duration_min": task.duration_min,
        "duration_max": task.duration_max,
        "intensity": task.intensity,
        "is_dividable": task.is_dividable,
        "is_concurrent": task.is_concurrent,
        "ontology_uri": task.ontology_uri,
    }


def _scheduled_task_to_dict(st: ScheduledTask) -> dict:
    # Round-trip ALL task fields that the evaluate-time
    # `_reconstruct_solution_from_json` will need to score correctly.
    # Notably `is_concurrent` and `is_dividable` were previously
    # dropped here, causing every reconstructed task at evaluate time
    # to fall back to the `RecommendedTask` default (`False`); which
    # then fired exclusion source (ii) in `is_excluded_pair` and
    # silently dropped every task from L_merge's Φ.  Fixed 2026-05-14.
    return {
        "label": st.task.label,
        "display_name": st.task.effective_display_name,
        "description": st.task.effective_description,
        "date": st.date.isoformat(),
        "start_minutes": st.start_minutes,
        "end_minutes": st.end_minutes,
        "is_standalone": st.is_standalone,
        "concurrent_with": st.concurrent_with,
        "intensity": st.task.intensity,
        "is_concurrent": st.task.is_concurrent,
        "is_dividable": st.task.is_dividable,
        "duration_min": st.task.duration_min,
        "duration_max": st.task.duration_max,
        "ontology_uri": st.task.ontology_uri,
        "parent_task_label": st.parent_task_label,
    }


def _by_time(st: ScheduledTask) -> tuple:
    """Sort key for a `ScheduledTask`; ascending date, then start minute."""
    return (st.date, st.start_minutes)


def solution_to_dict(solution: SchedulingSolution) -> dict:
    """Serialise a `SchedulingSolution` to a plain dict (JSON-ready).

    All scheduled-task lists are emitted in chronological order
    (ascending `date` then `start_minutes`) so a downstream consumer
    sees a single timeline rather than the augmenter's insertion order.
    """
    ac = solution.augmented_calendar
    scheduled_sorted = sorted(solution.scheduled, key=_by_time)
    standalone_sorted = sorted(
        (st for st in ac.scheduled_tasks if st.is_standalone), key=_by_time
    )
    concurrent_sorted = sorted(
        (st for st in ac.scheduled_tasks if not st.is_standalone), key=_by_time
    )
    return {
        "person_id": solution.person_id,
        "tasks_total": len(solution.tasks),
        "scheduled_count": len(solution.scheduled),
        "unscheduled_count": len(solution.unscheduled),
        "scheduled": [_scheduled_task_to_dict(st) for st in scheduled_sorted],
        "unscheduled": [_recommended_task_to_dict(t) for t in solution.unscheduled],
        "augmented_calendar": {
            "base_events_count": len(ac.base_events),
            "standalone_tasks": [
                _scheduled_task_to_dict(st) for st in standalone_sorted
            ],
            "concurrent_tasks": [
                _scheduled_task_to_dict(st) for st in concurrent_sorted
            ],
        },
    }


def write_solution_json(solution: SchedulingSolution, out_path: Path) -> Path:
    """Write `solution` to *out_path* as JSON and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(solution_to_dict(solution), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# L_pref preference_violations sidecar
# ---------------------------------------------------------------------------


def write_preference_violations_sidecar(
    out_path: Path,
    *,
    person_id: str,
    leg_stats: Iterable,
    pattern_rows: Iterable,
) -> Path:
    """Write one `preference_violations.jsonl` sidecar per persona.

    The file holds two record kinds, one per line:

    * `{"kind": "leg", "name": ..., "mean_loss": ..., "applicable_count": ..., "mape": ...}`
    * `{"kind": "pattern", "mode": ..., "scale": ..., "loss": ..., "mape": ...}`

    Reader: :func:`scenarios.metrics.telemetry.read_preference_violations`
    (added in this same step) aggregates the per-persona files into the
    `preference_breakdown` dict consumed by
    :func:`scenarios.export.report_writer.format_loss_report`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for leg in leg_stats:
        lines.append(
            json.dumps(
                {
                    "kind": "leg",
                    "person_id": person_id,
                    "name": leg.name,
                    "mean_loss": leg.mean_loss,
                    "applicable_count": leg.applicable_count,
                    "mape": leg.mape,
                }
            )
        )
    for row in pattern_rows:
        lines.append(
            json.dumps(
                {
                    "kind": "pattern",
                    "person_id": person_id,
                    "mode": row.mode,
                    "scale": row.scale,
                    "loss": row.loss,
                    "mape": row.mape,
                }
            )
        )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# L_divide verdicts sidecar
# ---------------------------------------------------------------------------


def write_divide_verdicts_sidecar(
    out_path: Path,
    *,
    person_id: str,
    records: Iterable,
) -> Path | None:
    """Write one `divide_verdicts.jsonl` sidecar per persona.

    One line per `(week, dividable label)` bucket. Returns `None` and
    writes nothing when `records` is empty.
    """
    rec_list = list(records)
    if not rec_list:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for rec in rec_list:
        lines.append(
            json.dumps(
                {
                    "person_id": person_id,
                    "iso_year": rec.iso_year,
                    "iso_week": rec.iso_week,
                    "task_label": rec.task_label,
                    "instances_in_week": rec.instances_in_week,
                    "original_duration_minutes": rec.original_duration_minutes,
                    "pieces": [
                        {
                            "date": d,
                            "start_minutes": s,
                            "end_minutes": e,
                            "duration_minutes": dur,
                        }
                        for d, s, e, dur in zip(
                            rec.piece_dates,
                            rec.piece_starts,
                            rec.piece_ends,
                            rec.piece_durations,
                        )
                    ],
                    "sum_minutes": rec.sum_minutes,
                    "tolerance_band": list(rec.tolerance_band),
                    "verdict": rec.verdict,
                }
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_weekly_gain_sidecar(
    out_path: Path,
    *,
    person_id: str,
    weekly_records: Iterable[dict],
) -> Path | None:
    """Write `<pid>_weekly_gain.json` with per-week scheduling-gain rows.

    Returns the written path, or `None` when `weekly_records` is empty.
    """
    rows = list(weekly_records)
    if not rows:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"person_id": person_id, "weeks": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path


def write_rl_training_sidecar(
    out_path: Path,
    *,
    person_id: str,
    training_records: Iterable[dict],
) -> Path | None:
    """Write `<pid>_rl_training.json` with per-week DQN training telemetry.

    Returns the written path, or `None` when `training_records` is empty.
    """
    rows = list(training_records)
    if not rows:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"person_id": person_id, "weeks": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path


def write_context_fit_sidecar(
    out_path: Path,
    *,
    person_id: str,
    verdicts: Iterable,
) -> Path | None:
    """Write one `context_fit.jsonl` line per (task, recommended_category) pair.

    Each row carries the per-task `fit` (observed-only) and `fit_full`
    (full-recommended) values plus an `in_scope` flag marking whether
    the row's `recommended_category` sits in the augmenter's observation
    budget; notebooks can slice on `in_scope` to recover either metric.
    """
    rec_list = list(verdicts)
    if not rec_list:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for v in rec_list:
        overlapped_set = set(v.overlapped_categories)
        scored_set = set(v.scored_categories)
        for cat in v.recommended_categories:
            lines.append(
                json.dumps(
                    {
                        "person_id": person_id,
                        "task_label": v.task_label,
                        "task_uri": v.task_uri,
                        "recommended_category": cat,
                        "overlapped": cat in overlapped_set,
                        "in_scope": cat in scored_set,
                        "observed_categories": list(v.observed_categories),
                        "scored_categories": list(v.scored_categories),
                        "fit": v.fit,
                        "fit_full": v.fit_full,
                    }
                )
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
