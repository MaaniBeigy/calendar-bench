"""Aggregator for per-person `divide_verdicts.jsonl` sidecars.

Reads the JSONL files and returns the cohort dict:

    {
      "applicable_persons": int,
      "applicable_weeks": int,
      "applicable_buckets": int,
      "verdict_counts": {<verdict>: int, ...},
      "per_label": {
        <task_label>: {
          "buckets": int,
          "divided_valid": int,
          "mean_sum_minutes": float,
          "mean_pieces_per_bucket": float,
        }
      }
    }
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.scripts.scenarios.metrics.loss import DIVIDE_VERDICTS


def read_divide_verdicts(sidecar_paths: Iterable[Path]) -> dict:
    """Aggregate per-person `divide_verdicts.jsonl` sidecars.

    Missing files and malformed lines are dropped silently.
    """
    verdict_counts: dict[str, int] = {v: 0 for v in DIVIDE_VERDICTS}
    persons_seen: set[str] = set()
    weeks_seen: set[tuple[int, int]] = set()
    bucket_total = 0
    per_label_buckets: dict[str, int] = defaultdict(int)
    per_label_valid: dict[str, int] = defaultdict(int)
    per_label_sum_minutes: dict[str, list[int]] = defaultdict(list)
    per_label_piece_counts: dict[str, list[int]] = defaultdict(list)

    for path in sidecar_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - defensive
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            verdict = row.get("verdict")
            if verdict not in verdict_counts:
                continue
            label = row.get("task_label")
            if not isinstance(label, str):
                continue
            person_id = row.get("person_id")
            if isinstance(person_id, str):
                persons_seen.add(person_id)
            iso_year = row.get("iso_year")
            iso_week = row.get("iso_week")
            if isinstance(iso_year, int) and isinstance(iso_week, int):
                weeks_seen.add((iso_year, iso_week))
            bucket_total += 1
            verdict_counts[verdict] += 1
            per_label_buckets[label] += 1
            if verdict == "divided_valid":
                per_label_valid[label] += 1
            sum_minutes = row.get("sum_minutes")
            if isinstance(sum_minutes, (int, float)):
                per_label_sum_minutes[label].append(int(sum_minutes))
            pieces = row.get("pieces")
            if isinstance(pieces, list):
                per_label_piece_counts[label].append(len(pieces))

    per_label: dict[str, dict] = {}
    for label, buckets in per_label_buckets.items():
        sums = per_label_sum_minutes.get(label, [])
        counts = per_label_piece_counts.get(label, [])
        per_label[label] = {
            "buckets": buckets,
            "divided_valid": per_label_valid.get(label, 0),
            "mean_sum_minutes": (sum(sums) / len(sums)) if sums else 0.0,
            "mean_pieces_per_bucket": (sum(counts) / len(counts)) if counts else 0.0,
        }

    return {
        "applicable_persons": len(persons_seen),
        "applicable_weeks": len(weeks_seen),
        "applicable_buckets": bucket_total,
        "verdict_counts": verdict_counts,
        "per_label": per_label,
    }


__all__ = ["read_divide_verdicts"]
