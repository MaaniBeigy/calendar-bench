"""Sweep `run_pool` across worker counts and chunk sizes, report wall time.

The point is empirical tuning: run the same workload several ways and read
the numbers, rather than guessing. The schedules are recomputed each pass,
so deterministic output is preserved (every call uses the same root seed).
A boolean per row records whether the configuration produced byte-identical
schedules to the baseline.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.scripts.persona.concurrency.pool import ExecutorKind, run_pool
from src.scripts.persona.config.schema import EnvironmentConfig, TemporalRelationRules
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One row from a benchmark sweep."""

    workers: int
    executor: ExecutorKind
    chunk_size: int | None
    auto_chunk: bool
    wall_seconds: float
    n_persons: int
    matches_baseline: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    """One configuration to evaluate during a sweep."""

    workers: int
    executor: ExecutorKind = "thread"
    chunk_size: int | None = None
    auto_chunk: bool = False


@dataclass(frozen=True)
class BenchmarkReport:
    """Bundle of every row from a sweep."""

    rows: list[BenchmarkResult] = field(default_factory=list)


def _canonical_schedules(schedules: list[PersonSchedule]) -> str:
    """Just the per-day events and spillovers, hashed across persons."""
    payload = []
    for sched in schedules:
        days = []
        for day in sched.days:
            days.append(
                {
                    "day_index": day.day_index,
                    "date": day.date.isoformat(),
                    "weekday": day.weekday,
                    "events": {
                        name: [
                            {"start": ev.start, "duration": ev.duration}
                            for ev in events
                        ]
                        for name, events in sorted(day.events.items())
                    },
                    "spillovers": [
                        {
                            "type": s.event_name,
                            "start": s.start,
                            "duration": s.duration,
                            "orig_start": s.orig_start,
                            "orig_duration": s.orig_duration,
                            "event_idx": s.event_idx,
                        }
                        for s in day.spillovers
                    ],
                }
            )
        payload.append({"person_id": sched.person_id, "days": days})
    return json.dumps(payload, sort_keys=True)


def benchmark_pool(
    persons: Sequence[Person],
    catalog: Catalog,
    environment: EnvironmentConfig,
    rules: TemporalRelationRules,
    configs: Iterable[BenchmarkConfig],
    *,
    timer: Callable[[], float] = time.perf_counter,
) -> BenchmarkReport:
    """Run `run_pool` once per config, time it, and capture matches-baseline.

    The first config in `configs` is the baseline. Subsequent configs are
    compared against it via canonical-form equality of their schedules.
    `timer` is injectable so tests can be deterministic.
    """
    configs = list(configs)
    if not configs:
        return BenchmarkReport(rows=[])

    rows: list[BenchmarkResult] = []
    baseline_canonical: str | None = None

    for cfg in configs:
        start = timer()
        schedules = run_pool(
            persons,
            catalog,
            environment,
            rules,
            workers=cfg.workers,
            executor=cfg.executor,
            chunk_size=cfg.chunk_size,
            auto_chunk=cfg.auto_chunk,
        )
        elapsed = timer() - start
        canon = _canonical_schedules(schedules)
        if baseline_canonical is None:
            baseline_canonical = canon
            matches = True
        else:
            matches = canon == baseline_canonical
        rows.append(
            BenchmarkResult(
                workers=cfg.workers,
                executor=cfg.executor,
                chunk_size=cfg.chunk_size,
                auto_chunk=cfg.auto_chunk,
                wall_seconds=elapsed,
                n_persons=len(persons),
                matches_baseline=matches,
            )
        )
    return BenchmarkReport(rows=rows)


def render_report(report: BenchmarkReport) -> str:
    """Render the sweep as a fixed-column text table."""
    lines = ["=== Benchmark sweep ==="]
    if not report.rows:
        lines.append("No configurations ran.")
        return "\n".join(lines) + "\n"
    lines.append(
        f"{'workers':>7}  {'executor':>8}  {'chunk':>5}  {'auto':>4}  "
        f"{'persons':>7}  {'wall_s':>7}  matches"
    )
    for r in report.rows:
        chunk = "auto" if r.auto_chunk else (str(r.chunk_size) if r.chunk_size else "-")
        lines.append(
            f"{r.workers:>7}  {r.executor:>8}  {chunk:>5}  "
            f"{'yes' if r.auto_chunk else 'no':>4}  "
            f"{r.n_persons:>7}  {r.wall_seconds:>7.3f}  "
            f"{'yes' if r.matches_baseline else 'no'}"
        )
    return "\n".join(lines) + "\n"


def report_to_dict(report: BenchmarkReport) -> dict[str, object]:
    """Render the report as a JSON-serializable dict."""
    return {
        "rows": [
            {
                "workers": r.workers,
                "executor": r.executor,
                "chunk_size": r.chunk_size,
                "auto_chunk": r.auto_chunk,
                "wall_seconds": r.wall_seconds,
                "n_persons": r.n_persons,
                "matches_baseline": r.matches_baseline,
            }
            for r in report.rows
        ]
    }


def write_report(report: BenchmarkReport, out_dir: Path | str) -> tuple[Path, Path]:
    """Write `benchmark_report.{txt,json}` and return the pair of paths."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    txt = out_dir_path / "benchmark_report.txt"
    js = out_dir_path / "benchmark_report.json"
    txt.write_text(render_report(report), encoding="utf-8")
    js.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return txt, js


__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkResult",
    "benchmark_pool",
    "render_report",
    "report_to_dict",
    "write_report",
]
