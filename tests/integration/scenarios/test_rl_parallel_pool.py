"""Integration tests for the RL process pool: real spawn, memory, OOM cap, retry.

These run real worker processes (and one drives the live RL augment over the
committed `example_experiment_smoke` fixture), so they are marked `integration` and run
outside the unit gate. Workers must be module-level so spawn can import them.
"""

from __future__ import annotations

import os
import resource
from pathlib import Path

import pytest

from src.scripts.scenarios.augmentation.rl import parallel
from src.scripts.scenarios.augmentation.rl.parallel import (
    RLWorkerPayload,
    RLWorkerResult,
    _run_spawn_round,
    resolve_rl_pool_workers,
    run_rl_process_pool,
)

pytestmark = pytest.mark.integration

SMOKE_DIR = Path("output/example_experiment_smoke")
SMOKE_SCENARIO = Path("src/experiments/persona/example_experiment_smoke/scenarios.yaml")


def _payload(idx: int) -> RLWorkerPayload:
    return RLWorkerPayload(
        person_index=idx,
        n_total=4,
        run=None,
        run_dir="rd",
        cfg=None,
        tasks_dir="t",
        persons_dir="p",
        ics_dir="i",
        aug_dir="a",
        horizon_hint=None,
        worker_threads=1,
    )


# Module-level workers so the spawn context can pickle + import them.


def _ok_worker(payload: RLWorkerPayload) -> RLWorkerResult:
    return RLWorkerResult(f"p{payload.person_index}", payload.person_index, True)


def _fail_one_worker(payload: RLWorkerPayload) -> RLWorkerResult:
    # Real workers catch their own errors and return a failed result rather
    # than raising; person 2 reports a failure the parent must surface.
    if payload.person_index == 2:
        return RLWorkerResult("p2", 2, False, "synthetic failure")
    return RLWorkerResult(f"p{payload.person_index}", payload.person_index, True)


def _hard_crash_worker(payload: RLWorkerPayload) -> RLWorkerResult:
    # An uncaught error kills the subprocess; the pool must still record the
    # other persons and mark this one failed (BrokenProcessPool path).
    if payload.person_index == 2:
        raise RuntimeError("worker crashed in subprocess")
    return RLWorkerResult(f"p{payload.person_index}", payload.person_index, True)


def _rss_worker(payload: RLWorkerPayload) -> RLWorkerResult:
    # Report this child's peak RSS (KB) back through the error field so the
    # parent can assert the per-worker budget without psutil.
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return RLWorkerResult(
        f"p{payload.person_index}", payload.person_index, True, str(rss_kb)
    )


# ---------------------------------------------------------------------------
# 6: real spawn round + crash survival
# ---------------------------------------------------------------------------


def test_real_spawn_round_runs_workers():
    results = _run_spawn_round(
        [_payload(1), _payload(2), _payload(3)], workers=2, worker_fn=_ok_worker
    )
    assert sorted(r.person_index for r in results) == [1, 2, 3]
    assert all(r.written for r in results)


def test_real_spawn_round_records_a_worker_returned_failure():
    results = _run_spawn_round(
        [_payload(1), _payload(2), _payload(3)], workers=2, worker_fn=_fail_one_worker
    )
    by_idx = {r.person_index: r for r in results}
    assert by_idx[2].written is False and by_idx[2].error
    assert by_idx[1].written and by_idx[3].written


def test_real_spawn_round_survives_an_uncaught_crash():
    results = _run_spawn_round(
        [_payload(1), _payload(2), _payload(3)], workers=2, worker_fn=_hard_crash_worker
    )
    by_idx = {r.person_index: r for r in results}
    assert by_idx[2].written is False and "crashed" in (by_idx[2].error or "")
    assert by_idx[1].written and by_idx[3].written


def test_real_pool_retries_a_failed_person():
    # Round 1 fails person 2 (returned failure), round 2 runs only person 2 and
    # succeeds. The parent owns the per-round behaviour, so vary it by round.
    rounds = {"n": 0}

    def _round(pending):
        rounds["n"] += 1
        worker = _fail_one_worker if rounds["n"] == 1 else _ok_worker
        return _run_spawn_round(pending, workers=2, worker_fn=worker)

    report = run_rl_process_pool(
        [_payload(1), _payload(2)], workers=2, max_retries=1, round_runner=_round
    )
    assert report.written == 2 and report.failed == []
    assert report.attempts["p2"] == 2
    assert rounds["n"] == 2


# ---------------------------------------------------------------------------
# 6b: memory smoke for parallel workers
# ---------------------------------------------------------------------------


def test_parallel_worker_rss_under_budget():
    # Each child reports its peak RSS; with the test worker (no torch / Qwen)
    # the per-worker RSS must be far under the configured per-worker budget.
    budget_kb = 1.5 * 1024 * 1024  # parallel_per_worker_gb default
    results = _run_spawn_round(
        [_payload(i) for i in range(1, 5)], workers=4, worker_fn=_rss_worker
    )
    rss = [int(r.error) for r in results]
    assert all(0 < kb < budget_kb for kb in rss)


# ---------------------------------------------------------------------------
# 6c: OOM-safe cap
# ---------------------------------------------------------------------------


def test_resolve_caps_below_requested_on_this_host():
    # Request far more workers than the host has; the resolver must clamp.
    workers, reason = resolve_rl_pool_workers(999, per_worker_gb=1.5)
    assert 1 <= workers <= (os.cpu_count() or 1)
    assert reason in {"cpu", "ram"}


def test_capped_pool_completes_every_person():
    # Even when requested workers exceed the safe cap, every person is run.
    workers, _ = resolve_rl_pool_workers(999, per_worker_gb=1.5)
    payloads = [_payload(i) for i in range(1, 5)]
    report = run_rl_process_pool(
        payloads,
        workers=workers,
        max_retries=0,
        round_runner=lambda pending: _run_spawn_round(
            pending, workers=workers, worker_fn=_ok_worker
        ),
    )
    assert report.written == 4 and report.failed == []


# ---------------------------------------------------------------------------
# 6a: live RL augment over the committed 2-person smoke cohort
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (SMOKE_DIR / "index.json").is_file(),
    reason="example_experiment_smoke fixture not on disk",
)
def test_live_parallel_rl_augment_writes_solutions(tmp_path, monkeypatch):
    from src.scripts.scenarios.cli import _cmd_augment_with_cfg
    from src.scripts.scenarios.config.loader import load_config

    exp = load_config(SMOKE_SCENARIO)
    scenario = next(s for s in exp.scenarios if s.id == "rl_baseline")
    method_cfg = next(m for m in scenario.augmentation if m.method == "rl")
    from src.scripts.scenarios.cli import _method_cfg_to_scenario_config

    cfg = _method_cfg_to_scenario_config(exp, scenario, method_cfg, SMOKE_SCENARIO)
    # Tiny training budget so the live run is seconds, not minutes.
    object.__setattr__(cfg.augmentation.rl, "train_steps_per_week", 50)

    out_dir = tmp_path / "out"
    import argparse

    args = argparse.Namespace(
        scenario=SMOKE_SCENARIO,
        out_dir=out_dir,
        tasks_dir=SMOKE_DIR / "task_generation" / "rl_baseline" / "tasks",
        method="rl",
        workers=2,
        executor="process",
        charts=None,
        log_level="INFO",
    )
    rc = _cmd_augment_with_cfg(cfg, args)
    assert rc == 0
    persons_dir = out_dir / "augmented" / "persons"
    written = sorted(persons_dir.glob("*.json"))
    assert len(written) >= 2
    for p in written:
        if p.stem.endswith(("_rl_training", "_loss", "_weekly_gain")):
            continue
        assert p.stat().st_size > 0
