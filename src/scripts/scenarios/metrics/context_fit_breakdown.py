"""Aggregator for per-person `context_fit.jsonl` sidecars."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def read_context_fit_verdicts(sidecar_paths: Iterable[Path]) -> dict:
    """Aggregate per-person `context_fit.jsonl` sidecars into a cohort dict.

    Returns
        `{applicable_persons, applicable_tasks, per_category: {<cat>: {pairs, overlapped, overlap_pct}}}`.
        Missing files and malformed lines are dropped silently.
    """
    persons_seen: set[str] = set()
    tasks_seen: set[tuple[str, str]] = set()
    per_cat_pairs: dict[str, int] = defaultdict(int)
    per_cat_overlap: dict[str, int] = defaultdict(int)

    for raw_path in sidecar_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - defensive
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            cat = row.get("recommended_category")
            if not isinstance(cat, str):
                continue
            # When the row carries an explicit `in_scope` flag (sidecars
            # written after the observed-only scoping change), skip categories
            # the augmenter could not see; legacy sidecars without the
            # flag keep counting every recommended pair.
            if "in_scope" in row and not row["in_scope"]:
                continue
            person_id = row.get("person_id")
            task_label = row.get("task_label")
            overlapped = bool(row.get("overlapped"))
            if isinstance(person_id, str):
                persons_seen.add(person_id)
            if isinstance(person_id, str) and isinstance(task_label, str):
                tasks_seen.add((person_id, task_label))
            per_cat_pairs[cat] += 1
            if overlapped:
                per_cat_overlap[cat] += 1

    per_category: dict[str, dict] = {}
    for cat, pairs in per_cat_pairs.items():
        overlapped = per_cat_overlap.get(cat, 0)
        per_category[cat] = {
            "pairs": pairs,
            "overlapped": overlapped,
            "overlap_pct": (overlapped / pairs * 100.0) if pairs else 0.0,
        }

    return {
        "applicable_persons": len(persons_seen),
        "applicable_tasks": len(tasks_seen),
        "per_category": per_category,
    }


def format_context_fit_breakdown(payload: dict) -> str:
    """Render a one-row-per-category text report with one-decimal overlap percentages."""
    lines: list[str] = [
        "-- L_context_fit breakdown --",
        f"applicable_persons : {payload.get('applicable_persons', 0)}",
        f"applicable_tasks   : {payload.get('applicable_tasks', 0)}",
    ]
    per_category = payload.get("per_category") or {}
    if not per_category:
        lines.append("  (no context_links recommendations applied to the cohort)")
        return "\n".join(lines) + "\n"
    width = max((len(c) for c in per_category), default=0)
    for cat in sorted(per_category):
        row = per_category[cat]
        lines.append(
            f"  {cat.ljust(width)}  pairs={row['pairs']:>4d}  "
            f"overlapped={row['overlapped']:>4d}  "
            f"overlap={row['overlap_pct']:>5.1f}%"
        )
    return "\n".join(lines) + "\n"


__all__ = ["format_context_fit_breakdown", "read_context_fit_verdicts"]
