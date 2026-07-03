"""Tests for the augment worker-pool (Fix G): resolution, RL clamp, parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.scripts.scenarios import cli as cli_module
from src.scripts.scenarios.cli import (
    EXIT_OK,
    _build_worker_augmenter,
    _cmd_augment_with_cfg,
    _resolve_augment_pool,
    _run_augment_pool,
)
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    CalendarSourceConfig,
    ScenarioConfig,
    ScenarioOutputConfig,
)
from src.scripts.scenarios.domain.calendar import AugmentedCalendar, CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask

# ---------------------------------------------------------------------------
# _resolve_augment_pool
# ---------------------------------------------------------------------------


def _args(**kw):
    base = dict(workers=1, executor="thread")
    base.update(kw)
    return argparse.Namespace(**base)


def test_resolve_pool_defaults_to_serial():
    workers, executor = _resolve_augment_pool(_args(), "greedy")
    assert workers == 1
    assert executor == "thread"


def test_resolve_pool_honours_explicit_workers():
    workers, executor = _resolve_augment_pool(_args(workers=6), "llm_agent")
    assert workers == 6
    assert executor == "thread"


def test_resolve_pool_clamps_rl_under_threads():
    workers, executor = _resolve_augment_pool(_args(workers=8), "rl")
    assert workers == 1  # rl forced serial under threads
    assert executor == "thread"


def test_resolve_pool_rl_process_not_clamped():
    workers, executor = _resolve_augment_pool(
        _args(workers=4, executor="process"), "rl"
    )
    assert workers == 4
    assert executor == "process"


def test_resolve_pool_floor_is_one():
    workers, _ = _resolve_augment_pool(_args(workers=0), "greedy")
    assert workers == 1


# ---------------------------------------------------------------------------
# _build_worker_augmenter
# ---------------------------------------------------------------------------


def test_build_worker_augmenter_greedy_has_no_recorder():
    aug, recorder = _build_worker_augmenter(
        "greedy", None, run_dir=None, cfg=None, prompt_ablate=set(), prompt_placebos={}
    )
    assert aug is not None
    assert recorder is None


def test_build_worker_augmenter_llm_agent_has_recorder():
    aug, recorder = _build_worker_augmenter(
        "llm_agent",
        None,
        run_dir=None,
        cfg=None,
        prompt_ablate=set(),
        prompt_placebos={},
    )
    assert aug is not None
    assert recorder is not None


# ---------------------------------------------------------------------------
# _run_augment_pool
# ---------------------------------------------------------------------------


def test_run_augment_pool_counts_written_and_calls_each():
    items = [(1, MagicMock()), (2, MagicMock()), (3, MagicMock())]
    seen = []

    def fn(idx, trace, aug, rec):
        seen.append(idx)
        return idx != 2  # person 2 has no task file

    written = _run_augment_pool(
        items,
        fn,
        workers=2,
        executor_kind="thread",
        build_worker=lambda: (object(), None),
        total=3,
        show=False,
    )
    assert written == 2
    assert sorted(seen) == [1, 2, 3]


def test_run_augment_pool_process_falls_back_to_threads(caplog):
    import logging

    items = [(1, MagicMock())]
    with caplog.at_level(logging.WARNING, logger="scenarios.cli"):
        written = _run_augment_pool(
            items,
            lambda idx, trace, aug, rec: True,
            workers=2,
            executor_kind="process",
            build_worker=lambda: (object(), None),
            total=1,
            show=False,
        )
    assert written == 1
    assert any("supported only for rl" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Parallel parity end-to-end
# ---------------------------------------------------------------------------


def _build_layout(tmp_path: Path, n_persons: int) -> dict:
    run_dir = tmp_path / "run"
    (run_dir / "persons").mkdir(parents=True)
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    pids = [f"p{i:03d}" for i in range(n_persons)]
    for pid in pids:
        (tasks_dir / f"{pid}_tasks.json").write_text(
            json.dumps([{"label": "walk", "duration_min": 30, "duration_max": 30}]),
            encoding="utf-8",
        )
    return {
        "run_dir": run_dir,
        "tasks_dir": tasks_dir,
        "out_dir": tmp_path / "out",
        "pids": pids,
    }


def _mock_run(pids):
    run = MagicMock()
    run.traces = [CalendarTrace(person_id=pid, events=[]) for pid in pids]
    run.allen_pair_rules = []
    run.time_windows = {}
    run.horizon_days = 7
    run.horizon_start_date = None
    return run


def _solution_for(trace):
    task = RecommendedTask(label="walk", duration_min=30, duration_max=30)
    import datetime

    sched = ScheduledTask(
        task=task,
        start_minutes=540,
        end_minutes=570,
        is_standalone=True,
        concurrent_with=None,
        date=datetime.date(2026, 6, 1),
    )
    return SchedulingSolution(
        person_id=trace.person_id,
        augmented_calendar=AugmentedCalendar(person_id=trace.person_id),
        tasks=[task],
        scheduled=[sched],
        unscheduled=[],
    )


def _cfg(layout):
    return ScenarioConfig(
        id="parallel_test",
        calendar=CalendarSourceConfig(run_dir=str(layout["run_dir"])),
        output=ScenarioOutputConfig(dir=str(layout["out_dir"])),
        augmentation=AugmentationConfig(method="greedy"),
    )


def _run_augment(layout, workers):
    cfg = _cfg(layout)
    run = _mock_run(layout["pids"])
    args = argparse.Namespace(
        scenario=Path("x.yaml"),
        out_dir=layout["out_dir"],
        tasks_dir=layout["tasks_dir"],
        method=None,
        log_level="INFO",
        workers=workers,
        executor="thread",
    )
    augmenter = MagicMock()
    augmenter.augment.side_effect = lambda trace, *a, **k: _solution_for(trace)
    with (
        patch.object(
            cli_module, "_try_load_persona_run", return_value=(run, layout["run_dir"])
        ),
        patch.object(cli_module, "_build_augmenter", return_value=augmenter),
        patch.object(cli_module, "write_augmented_ics", create=True),
    ):
        rc = _cmd_augment_with_cfg(cfg, args)
    return rc


def test_parallel_augment_writes_all_solutions(tmp_path: Path):
    layout = _build_layout(tmp_path, n_persons=5)
    rc = _run_augment(layout, workers=3)
    assert rc == EXIT_OK
    persons_dir = layout["out_dir"] / "augmented" / "persons"
    for pid in layout["pids"]:
        assert (persons_dir / f"{pid}.json").exists()


def test_parallel_matches_serial_solution_files(tmp_path: Path):
    serial = _build_layout(tmp_path / "serial", n_persons=4)
    parallel = _build_layout(tmp_path / "parallel", n_persons=4)
    assert _run_augment(serial, workers=1) == EXIT_OK
    assert _run_augment(parallel, workers=4) == EXIT_OK
    s_dir = serial["out_dir"] / "augmented" / "persons"
    p_dir = parallel["out_dir"] / "augmented" / "persons"
    for pid in serial["pids"]:
        s = json.loads((s_dir / f"{pid}.json").read_text())
        p = json.loads((p_dir / f"{pid}.json").read_text())
        assert s["scheduled"] == p["scheduled"]
        assert s["person_id"] == p["person_id"]
