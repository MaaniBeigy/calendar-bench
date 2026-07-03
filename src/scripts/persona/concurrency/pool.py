"""Run per-person tasks across a configurable executor pool."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal

from src.scripts.persona.concurrency.chunking import auto_chunk_size
from src.scripts.persona.concurrency.progress import (
    default_show_progress,
    wrap_progress,
)
from src.scripts.persona.concurrency.tasks import run_person_task
from src.scripts.persona.config.schema import EnvironmentConfig, TemporalRelationRules
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule

ExecutorKind = Literal["thread", "process"]


def _resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    return os.cpu_count() or 1


def _resolve_chunk_size(
    chunk_size: int, *, n_persons: int, workers: int, auto: bool
) -> int:
    """Pick the chunk size used by `Executor.map`.

    `auto=True` ignores `chunk_size` entirely and asks the heuristic for a
    workload-aware value. `auto=False` returns the explicit `chunk_size`.
    """
    if auto:
        return auto_chunk_size(n_persons, workers)
    return chunk_size


def run_pool(
    persons: Sequence[Person],
    catalog: Catalog,
    environment: EnvironmentConfig,
    rules: TemporalRelationRules,
    *,
    workers: int | None = None,
    executor: ExecutorKind | None = None,
    chunk_size: int | None = None,
    auto_chunk: bool = False,
    show_progress: bool | None = None,
) -> list[PersonSchedule]:
    """Submit one task per person, return schedules in input order.

    Defaults are read from `environment.parallelism` when the matching keyword
    is None. `workers=0` resolves to `os.cpu_count()`. `auto_chunk=True`
    asks `auto_chunk_size` for a workload-aware chunk and ignores the
    `chunk_size` argument. `show_progress=None` defers to whether stderr is a
    tty (`default_show_progress`); pass an explicit boolean to override.
    Output mirrors input order regardless of worker count, executor kind,
    or chunk size.
    """
    if not persons:
        return []

    cfg = environment.parallelism
    workers = _resolve_workers(workers if workers is not None else cfg.workers)
    executor_kind: ExecutorKind = executor if executor is not None else cfg.executor
    chunk = _resolve_chunk_size(
        chunk_size if chunk_size is not None else cfg.chunk_size,
        n_persons=len(persons),
        workers=workers,
        auto=auto_chunk,
    )
    progress_on = default_show_progress() if show_progress is None else show_progress

    if workers == 1:
        result_iter = (run_person_task(p, catalog, environment, rules) for p in persons)
        return list(
            wrap_progress(
                result_iter, total=len(persons), desc="Generating:", show=progress_on
            )
        )

    Executor = ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor
    args = [(p, catalog, environment, rules) for p in persons]
    with Executor(max_workers=workers) as ex:
        result_iter = ex.map(_unpack_and_run, args, chunksize=chunk)
        return list(
            wrap_progress(
                result_iter, total=len(persons), desc="Generating:", show=progress_on
            )
        )


def _unpack_and_run(args: tuple) -> PersonSchedule:
    """Module-level helper so `ProcessPoolExecutor` can pickle it."""
    person, catalog, environment, rules = args
    return run_person_task(person, catalog, environment, rules)
