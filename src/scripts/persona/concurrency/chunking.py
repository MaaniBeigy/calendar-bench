"""Pick a sensible `chunksize` for `Executor.map` based on workload shape.

The default `chunk_size: 1` keeps each task on its own future, which is
optimal when tasks vary widely in cost. When every task is roughly the same
cost (which is true for our per-person solve), bigger chunks amortise the
per-task scheduling overhead. Too big, though, and a single straggler can
dominate wall time because workers run out of work near the end.

The heuristic aims for roughly `target_chunks_per_worker` batches per worker
so the late-finish tail is bounded by one chunk.
"""

from __future__ import annotations

DEFAULT_TARGET_CHUNKS_PER_WORKER = 4


def auto_chunk_size(
    n_persons: int,
    workers: int,
    *,
    target_chunks_per_worker: int = DEFAULT_TARGET_CHUNKS_PER_WORKER,
) -> int:
    """Pick a chunksize for an `Executor.map` over `n_persons` items.

    Returns at least 1. With workers <= 1 the chunk size is irrelevant so we
    keep it at 1. With more workers the chunk grows linearly with the per-
    worker share of the population, capped so each worker still gets several
    batches.
    """
    if n_persons <= 0:
        raise ValueError(f"n_persons must be > 0, got {n_persons}")
    if workers <= 0:
        raise ValueError(f"workers must be > 0, got {workers}")
    if target_chunks_per_worker <= 0:
        raise ValueError(
            f"target_chunks_per_worker must be > 0, got {target_chunks_per_worker}"
        )
    if workers == 1:
        return 1
    per_worker = n_persons / workers
    chunk = int(per_worker // target_chunks_per_worker)
    return max(1, chunk)


__all__ = ["DEFAULT_TARGET_CHUNKS_PER_WORKER", "auto_chunk_size"]
