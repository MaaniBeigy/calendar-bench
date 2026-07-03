"""Unit tests for src.scripts.persona.concurrency.chunking."""

from __future__ import annotations

import pytest

from src.scripts.persona.concurrency.chunking import auto_chunk_size


def test_single_worker_always_returns_one():
    assert auto_chunk_size(1, 1) == 1
    assert auto_chunk_size(1000, 1) == 1


def test_small_population_floors_to_one():
    """5 persons across 4 workers: too few items to pre-chunk usefully."""
    assert auto_chunk_size(5, 4) == 1


def test_large_population_grows_with_per_worker_share():
    """100 persons / 4 workers / 4 chunks-per-worker = chunk 6."""
    assert auto_chunk_size(100, 4) == 6


def test_chunk_scales_with_target_chunks_per_worker():
    """Bigger target_chunks_per_worker means smaller chunks (more batches per worker)."""
    big_target = auto_chunk_size(100, 4, target_chunks_per_worker=10)
    small_target = auto_chunk_size(100, 4, target_chunks_per_worker=2)
    assert big_target < small_target


def test_zero_persons_is_rejected():
    with pytest.raises(ValueError, match="n_persons must be"):
        auto_chunk_size(0, 4)


def test_negative_persons_is_rejected():
    with pytest.raises(ValueError, match="n_persons must be"):
        auto_chunk_size(-1, 4)


def test_zero_workers_is_rejected():
    with pytest.raises(ValueError, match="workers must be"):
        auto_chunk_size(10, 0)


def test_negative_workers_is_rejected():
    with pytest.raises(ValueError, match="workers must be"):
        auto_chunk_size(10, -2)


def test_zero_target_chunks_per_worker_is_rejected():
    with pytest.raises(ValueError, match="target_chunks_per_worker"):
        auto_chunk_size(10, 4, target_chunks_per_worker=0)


def test_default_target_chunks_per_worker_is_documented_constant():
    from src.scripts.persona.concurrency.chunking import (
        DEFAULT_TARGET_CHUNKS_PER_WORKER,
    )

    assert DEFAULT_TARGET_CHUNKS_PER_WORKER == 4
