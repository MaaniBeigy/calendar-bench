"""Process-pool helpers for fanning the per-person RL augment across workers.

The per-person DQN runs are independent (a fresh agent + replay buffer per
person, seeded from the person id), so they parallelize cleanly. Each worker
rebuilds the unpicklable augmenter in its own process from the picklable
payload, forces the embedder to CPU, and writes the person's artifacts.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RLWorkerPayload:
    """Picklable inputs one worker needs to augment a single person."""

    person_index: int
    n_total: int
    run: Any
    run_dir: str
    cfg: Any
    tasks_dir: str
    persons_dir: str
    ics_dir: str
    aug_dir: str
    horizon_hint: Any
    worker_threads: int


@dataclass
class RLWorkerResult:
    """Per-person outcome returned to the parent."""

    person_id: str
    person_index: int
    written: bool
    error: str | None = None


@dataclass
class RLPoolReport:
    """Roll-up of one parallel RL run across all rounds."""

    written: int = 0
    failed: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)


def _payload_person_id(payload: RLWorkerPayload) -> str:
    """Person id for a payload, falling back to the index when no run is set."""
    run = payload.run
    if run is not None and getattr(run, "traces", None):
        return run.traces[payload.person_index - 1].person_id
    return f"person_{payload.person_index}"


def _rl_augment_worker(payload: RLWorkerPayload) -> RLWorkerResult:
    """Augment one person in a fresh process; never raises, returns the outcome."""
    os.environ["COMPUTE_DEVICE"] = "cpu"
    trace = payload.run.traces[payload.person_index - 1]
    person_id = trace.person_id
    try:
        import torch

        torch.set_num_threads(max(1, int(payload.worker_threads)))
    except Exception:  # pragma: no cover - torch always present in the RL path
        pass
    try:
        from src.scripts.scenarios.cli import (
            _augment_person_to_disk,
            _build_rl_augmenter,
            _LazyCpuEmbedder,
            _wire_rl_constraints,
        )

        augmenter = _build_rl_augmenter(
            run=payload.run,
            cfg=payload.cfg,
            time_windows=getattr(payload.run, "time_windows", {}),
            daily_window=getattr(payload.run, "daily_window", None),
            allen_rules=list(getattr(payload.run, "allen_pair_rules", []) or []),
            met_embedder=_LazyCpuEmbedder(),
        )
        _wire_rl_constraints(
            augmenter, run=payload.run, run_dir=Path(payload.run_dir), cfg=payload.cfg
        )
        written = _augment_person_to_disk(
            trace,
            augmenter,
            idx=payload.person_index,
            n_total=payload.n_total,
            method="rl",
            cfg=payload.cfg,
            tasks_dir=Path(payload.tasks_dir),
            persons_dir=Path(payload.persons_dir),
            ics_dir=Path(payload.ics_dir),
            aug_dir=Path(payload.aug_dir),
            horizon_hint=payload.horizon_hint,
            recorder=None,
        )
        return RLWorkerResult(person_id, payload.person_index, bool(written))
    except Exception as exc:  # keep the pool alive; the parent retries failures
        return RLWorkerResult(person_id, payload.person_index, False, str(exc))


def _ram_worker_cap(per_worker_gb: float) -> int:
    """Workers that fit in MemAvailable at `per_worker_gb` each; at least 1."""
    available_gb = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_gb = int(line.split()[1]) / (1024 * 1024)
                break
    except OSError:  # pragma: no cover - non-Linux fallback
        return 1
    if available_gb is None:  # pragma: no cover - unexpected meminfo shape
        return 1
    return max(1, int(available_gb / max(per_worker_gb, 1e-6)))


def resolve_rl_pool_workers(requested: int, *, per_worker_gb: float) -> tuple[int, str]:
    """Cap requested workers by cores and RAM; return (workers, binding reason)."""
    cpu_cap = max(1, (os.cpu_count() or 1) - 1)
    ram_cap = _ram_worker_cap(per_worker_gb)
    chosen = min(requested, cpu_cap, ram_cap)
    if chosen == ram_cap and ram_cap < cpu_cap and ram_cap < requested:
        reason = "ram"
    elif chosen == cpu_cap and cpu_cap < requested:
        reason = "cpu"
    else:
        reason = "requested"
    return max(1, chosen), reason


def _run_spawn_round(
    payloads: list[RLWorkerPayload], *, workers: int, worker_fn=_rl_augment_worker
) -> list[RLWorkerResult]:
    """Run one round of payloads in a spawn process pool; never raises.

    `worker_fn` defaults to the real per-person worker; a picklable substitute
    lets tests exercise the spawn + crash-survival machinery cheaply.
    """
    results: list[RLWorkerResult] = []
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(payloads)), mp_context=ctx
    ) as pool:
        futures = {pool.submit(worker_fn, p): p for p in payloads}
        for fut in as_completed(futures):
            payload = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # BrokenProcessPool: the subprocess died
                results.append(
                    RLWorkerResult(
                        _payload_person_id(payload),
                        payload.person_index,
                        False,
                        str(exc),
                    )
                )
    return results


def run_rl_process_pool(
    payloads: list[RLWorkerPayload],
    *,
    workers: int,
    max_retries: int,
    round_runner=None,
) -> RLPoolReport:
    """Run payloads in a spawn pool, retrying failed persons in fresh rounds.

    `round_runner(pending) -> list[RLWorkerResult]` runs one round; it defaults
    to the spawn process pool and is injectable for tests.
    """
    if round_runner is None:

        def round_runner(pending):
            return _run_spawn_round(pending, workers=workers)

    report = RLPoolReport()
    pending = list(payloads)
    round_index = 0
    while pending:
        results = round_runner(pending)
        by_index = {r.person_index: r for r in results}
        for res in results:
            report.attempts[res.person_id] = report.attempts.get(res.person_id, 0) + 1
        retry: list[RLWorkerPayload] = []
        for payload in pending:
            res = by_index[payload.person_index]
            if res.written:
                report.written += 1
            elif round_index < max_retries:
                retry.append(payload)
            else:
                report.failed.append(res.person_id)
        pending = retry
        round_index += 1
    return report
