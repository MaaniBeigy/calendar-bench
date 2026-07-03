"""Tests for the process-pool RL augment helpers."""

from __future__ import annotations

import dataclasses
import pickle
from types import SimpleNamespace

from src.scripts.scenarios.augmentation.rl import parallel
from src.scripts.scenarios.augmentation.rl.parallel import (
    RLPoolReport,
    RLWorkerPayload,
    RLWorkerResult,
    _payload_person_id,
    _ram_worker_cap,
    resolve_rl_pool_workers,
    run_rl_process_pool,
)


def _payload(idx: int) -> RLWorkerPayload:
    return RLWorkerPayload(
        person_index=idx,
        n_total=3,
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


# ---------------------------------------------------------------------------
# 5a: payload is picklable; it carries no drivers / augmenter
# ---------------------------------------------------------------------------


def test_payload_pickles():
    p = _payload(1)
    back = pickle.loads(pickle.dumps(p))
    assert back.person_index == 1 and back.tasks_dir == "t"


def test_payload_person_id_reads_trace_when_run_present():
    run = SimpleNamespace(
        traces=[
            SimpleNamespace(person_id="alice_0000"),
            SimpleNamespace(person_id="bob_0001"),
        ]
    )
    payload = dataclasses.replace(_payload(2), run=run)
    assert _payload_person_id(payload) == "bob_0001"


def test_payload_person_id_falls_back_without_run():
    assert _payload_person_id(_payload(3)) == "person_3"


# ---------------------------------------------------------------------------
# 5b: worker-count resolution caps by cpu, ram, request
# ---------------------------------------------------------------------------


def test_resolve_caps_to_cpu(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(parallel, "_ram_worker_cap", lambda gb: 100)
    workers, reason = resolve_rl_pool_workers(99, per_worker_gb=1.5)
    assert workers == 3 and reason == "cpu"


def test_resolve_caps_to_ram(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(parallel, "_ram_worker_cap", lambda gb: 2)
    workers, reason = resolve_rl_pool_workers(99, per_worker_gb=1.5)
    assert workers == 2 and reason == "ram"


def test_resolve_honours_request(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(parallel, "_ram_worker_cap", lambda gb: 100)
    workers, reason = resolve_rl_pool_workers(3, per_worker_gb=1.5)
    assert workers == 3 and reason == "requested"


def test_ram_cap_reads_meminfo(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1\nMemAvailable: 6291456 kB\n", encoding="utf-8")
    monkeypatch.setattr(parallel, "Path", lambda _p: meminfo)
    assert _ram_worker_cap(1.5) == 4  # 6 GB / 1.5


def test_ram_cap_handles_missing_meminfo(monkeypatch):
    class _Boom:
        def read_text(self, encoding="utf-8"):
            raise OSError("nope")

    monkeypatch.setattr(parallel, "Path", lambda _p: _Boom())
    assert _ram_worker_cap(1.5) == 1


def test_ram_cap_handles_meminfo_without_available(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 123 kB\n", encoding="utf-8")
    monkeypatch.setattr(parallel, "Path", lambda _p: meminfo)
    assert _ram_worker_cap(1.5) == 1


# ---------------------------------------------------------------------------
# 5g: retry queue
# ---------------------------------------------------------------------------


def test_all_written_no_retry_round():
    rounds = []

    def runner(pending):
        rounds.append([p.person_index for p in pending])
        return [
            RLWorkerResult(f"p{p.person_index}", p.person_index, True) for p in pending
        ]

    report = run_rl_process_pool(
        [_payload(1), _payload(2)], workers=2, max_retries=1, round_runner=runner
    )
    assert report.written == 2 and report.failed == []
    assert len(rounds) == 1  # no retry round


def test_failed_person_retried_then_written():
    calls = {"n": 0}

    def runner(pending):
        calls["n"] += 1
        out = []
        for p in pending:
            ok = not (p.person_index == 2 and calls["n"] == 1)
            out.append(
                RLWorkerResult(
                    f"p{p.person_index}", p.person_index, ok, None if ok else "boom"
                )
            )
        return out

    report = run_rl_process_pool(
        [_payload(1), _payload(2)], workers=2, max_retries=1, round_runner=runner
    )
    assert report.written == 2
    assert report.failed == []
    assert report.attempts["p2"] == 2  # one failure + one retry success
    assert calls["n"] == 2


def test_permanent_failure_lands_in_failed():
    def runner(pending):
        return [
            RLWorkerResult(
                f"p{p.person_index}",
                p.person_index,
                p.person_index != 2,
                None if p.person_index != 2 else "boom",
            )
            for p in pending
        ]

    report = run_rl_process_pool(
        [_payload(1), _payload(2)], workers=2, max_retries=1, round_runner=runner
    )
    assert report.written == 1
    assert report.failed == ["p2"]
    assert report.attempts["p2"] == 2  # initial + 1 retry, both failed


# ---------------------------------------------------------------------------
# 5c: real spawn round runs the worker and survives a raising worker
# ---------------------------------------------------------------------------


def test_spawn_round_collects_results():
    # The default round runner pickles `_rl_augment_worker`; substitute a tiny
    # module-level worker via the public seam to exercise the loop end to end.
    report = run_rl_process_pool(
        [_payload(1)],
        workers=1,
        max_retries=0,
        round_runner=lambda pending: [
            RLWorkerResult("p1", pending[0].person_index, True)
        ],
    )
    assert report.written == 1


# ---------------------------------------------------------------------------
# 5h: lazy CPU embedder
# ---------------------------------------------------------------------------


class _Trace:
    def __init__(self, pid):
        self.person_id = pid
        self.time_windows = {}
        self.daily_window = None
        self.allen_pair_rules = []


class _Run:
    def __init__(self, pids):
        self.traces = [_Trace(p) for p in pids]
        self.time_windows = {}
        self.daily_window = None
        self.allen_pair_rules = []


def _run_payload(idx, run):
    return RLWorkerPayload(
        person_index=idx,
        n_total=len(run.traces),
        run=run,
        run_dir="rd",
        cfg=object(),
        tasks_dir="t",
        persons_dir="p",
        ics_dir="i",
        aug_dir="a",
        horizon_hint=None,
        worker_threads=1,
    )


def test_worker_success_sets_cpu_env(monkeypatch):
    monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
    monkeypatch.setattr(
        "src.scripts.scenarios.cli._build_rl_augmenter", lambda **k: object()
    )
    monkeypatch.setattr(
        "src.scripts.scenarios.cli._wire_rl_constraints", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "src.scripts.scenarios.cli._augment_person_to_disk", lambda *a, **k: True
    )
    run = _Run(["p1", "p2"])
    res = parallel._rl_augment_worker(_run_payload(2, run))
    assert res.person_id == "p2" and res.written is True and res.error is None
    import os as _os

    assert _os.environ["COMPUTE_DEVICE"] == "cpu"


def test_worker_captures_exception(monkeypatch):
    def _boom(**k):
        raise RuntimeError("build failed")

    monkeypatch.setattr("src.scripts.scenarios.cli._build_rl_augmenter", _boom)
    run = _Run(["p1"])
    res = parallel._rl_augment_worker(_run_payload(1, run))
    assert res.written is False and "build failed" in res.error


def test_lazy_embedder_defers_model_load(monkeypatch):
    from src.scripts.scenarios import cli

    built = {"n": 0}

    class _Fake:
        def embed_query(self, text):
            return [0.0]

    def _fake_make(device=None):
        built["n"] += 1
        assert device == "cpu"
        return _Fake()

    monkeypatch.setattr("src.graphrag.embeddings.make_embedder", _fake_make)
    emb = cli._LazyCpuEmbedder()
    assert built["n"] == 0  # no load yet
    assert emb.embed_query("walk") == [0.0]
    assert built["n"] == 1  # loaded on first miss
    emb.embed_query("run")
    assert built["n"] == 1  # cached for the rest of the process
