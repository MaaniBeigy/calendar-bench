"""Unit tests for src.scripts.persona.concurrency.pool."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.concurrency.pool import run_pool
from src.scripts.persona.config.schema import (
    EnvironmentConfig,
    HorizonConfig,
    OutputConfig,
    ParallelismConfig,
    SolverConfig,
    TemporalRelationRules,
    WindowRange,
)
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.event_config.loader import load_catalog
from tests.unit.persona.conftest import make_person, make_stage


def _env(workers: int = 1, executor: str = "thread") -> EnvironmentConfig:
    return EnvironmentConfig(
        seed=1,
        horizon=HorizonConfig(start_date=_dt.date(2026, 5, 4), weeks=1),
        output=OutputConfig(dir="./out"),
        solver=SolverConfig(),
        time_windows={
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        },
        parallelism=ParallelismConfig(workers=workers, executor=executor),
    )


def _persons(n: int = 3) -> list[Person]:
    return [_person(f"p{i}", seed=10 + i) for i in range(n)]


def _person(persona_id: str, seed: int) -> Person:
    return make_person(
        person_id=f"{persona_id}_0000",
        persona_id=persona_id,
        person_seed=seed,
        occupation_status="student",
        stages=[make_stage("sleep", time="23:00")],
    )


# -------------------------------------------------------------------------------------
# ------------------------------------ basic shape ------------------------------------
# -------------------------------------------------------------------------------------


def test_run_pool_empty_population_returns_empty_list(event_yaml):
    catalog = load_catalog(event_yaml)
    out = run_pool([], catalog, _env(), TemporalRelationRules())
    assert out == []


def test_run_pool_single_thread_returns_one_schedule_per_person(event_yaml):
    catalog = load_catalog(event_yaml)
    out = run_pool(_persons(3), catalog, _env(workers=1), TemporalRelationRules())
    assert len(out) == 3
    assert [s.persona_id for s in out] == ["p0", "p1", "p2"]


def test_run_pool_preserves_input_order(event_yaml):
    catalog = load_catalog(event_yaml)
    persons = _persons(4)
    out = run_pool(persons, catalog, _env(workers=1), TemporalRelationRules())
    assert [s.person_id for s in out] == [p.person_id for p in persons]


# -------------------------------------------------------------------------------------
# ----------------------------- determinism across executors --------------------------
# -------------------------------------------------------------------------------------


def test_run_pool_workers_1_equals_workers_2_thread(event_yaml):
    catalog = load_catalog(event_yaml)
    persons = _persons(3)
    rules = TemporalRelationRules()
    one = run_pool(persons, catalog, _env(workers=1), rules)
    two = run_pool(
        persons, catalog, _env(workers=1), rules, workers=2, executor="thread"
    )
    assert one == two


def test_run_pool_thread_equals_process(event_yaml):
    """Same outputs across thread and process executors."""
    catalog = load_catalog(event_yaml)
    persons = _persons(2)
    rules = TemporalRelationRules()
    thread_out = run_pool(persons, catalog, _env(), rules, workers=2, executor="thread")
    process_out = run_pool(
        persons, catalog, _env(), rules, workers=2, executor="process"
    )
    assert thread_out == process_out


# -------------------------------------------------------------------------------------
# --------------------------------- override resolution -------------------------------
# -------------------------------------------------------------------------------------


def test_run_pool_workers_override_takes_precedence(event_yaml):
    """Explicit `workers=` overrides `environment.parallelism.workers`."""
    catalog = load_catalog(event_yaml)
    out = run_pool(
        _persons(2),
        catalog,
        _env(workers=99),
        TemporalRelationRules(),
        workers=1,
    )
    assert len(out) == 2


def test_run_pool_executor_override_takes_precedence(event_yaml):
    catalog = load_catalog(event_yaml)
    out = run_pool(
        _persons(2),
        catalog,
        _env(executor="process"),
        TemporalRelationRules(),
        workers=2,
        executor="thread",
    )
    assert len(out) == 2


def test_run_pool_zero_workers_resolves_to_cpu_count(event_yaml):
    """workers=0 means 'use os.cpu_count()'; we just verify it runs cleanly."""
    catalog = load_catalog(event_yaml)
    out = run_pool(
        _persons(2),
        catalog,
        _env(workers=0, executor="thread"),
        TemporalRelationRules(),
    )
    assert len(out) == 2


# -------------------------------------------------------------------------------------
# ----------------------------------- chunk sizing ------------------------------------
# -------------------------------------------------------------------------------------


def test_run_pool_chunk_size_does_not_change_output(event_yaml):
    catalog = load_catalog(event_yaml)
    persons = _persons(3)
    rules = TemporalRelationRules()
    chunk1 = run_pool(
        persons,
        catalog,
        _env(),
        rules,
        workers=2,
        executor="thread",
        chunk_size=1,
    )
    chunk2 = run_pool(
        persons,
        catalog,
        _env(),
        rules,
        workers=2,
        executor="thread",
        chunk_size=2,
    )
    assert chunk1 == chunk2


def test_run_pool_auto_chunk_matches_explicit_chunk(event_yaml):
    """`auto_chunk=True` ignores the explicit chunk argument and falls back
    to the heuristic. The output schedules must still be byte-identical
    to the explicit-chunk case."""
    catalog = load_catalog(event_yaml)
    persons = _persons(3)
    rules = TemporalRelationRules()
    explicit = run_pool(
        persons, catalog, _env(), rules, workers=2, executor="thread", chunk_size=1
    )
    auto = run_pool(
        persons,
        catalog,
        _env(),
        rules,
        workers=2,
        executor="thread",
        chunk_size=99,  # ignored under auto_chunk
        auto_chunk=True,
    )
    assert auto == explicit


def test_run_pool_auto_chunk_with_single_worker(event_yaml):
    """`auto_chunk=True` plus workers=1 runs the serial fast path cleanly."""
    catalog = load_catalog(event_yaml)
    out = run_pool(
        _persons(2),
        catalog,
        _env(),
        TemporalRelationRules(),
        workers=1,
        auto_chunk=True,
    )
    assert len(out) == 2


# -------------------------------------------------------------------------------------
# --------------------------- show_progress flag wiring -------------------------------
# -------------------------------------------------------------------------------------


def test_run_pool_show_progress_true_invokes_wrap_progress(event_yaml, monkeypatch):
    catalog = load_catalog(event_yaml)
    captured: dict = {"shows": []}

    from src.scripts.persona.concurrency import pool as pool_module

    real_wrap = pool_module.wrap_progress

    def spy_wrap(iterable, *, total, desc, show):
        captured["shows"].append(show)
        captured["total"] = total
        captured["desc"] = desc
        return real_wrap(iterable, total=total, desc=desc, show=show)

    monkeypatch.setattr(pool_module, "wrap_progress", spy_wrap)
    run_pool(
        _persons(2),
        catalog,
        _env(workers=1),
        TemporalRelationRules(),
        show_progress=True,
    )
    assert True in captured["shows"]
    assert captured["total"] == 2


def test_run_pool_show_progress_false_calls_wrap_with_show_false(
    event_yaml, monkeypatch
):
    catalog = load_catalog(event_yaml)
    seen: dict = {}

    from src.scripts.persona.concurrency import pool as pool_module

    real_wrap = pool_module.wrap_progress

    def spy_wrap(iterable, *, total, desc, show):
        seen["show"] = show
        return real_wrap(iterable, total=total, desc=desc, show=show)

    monkeypatch.setattr(pool_module, "wrap_progress", spy_wrap)
    run_pool(
        _persons(1),
        catalog,
        _env(workers=1),
        TemporalRelationRules(),
        show_progress=False,
    )
    assert seen["show"] is False


def test_run_pool_show_progress_none_uses_tty_default(event_yaml, monkeypatch):
    catalog = load_catalog(event_yaml)
    from src.scripts.persona.concurrency import pool as pool_module

    monkeypatch.setattr(pool_module, "default_show_progress", lambda: True)
    seen: dict = {}
    real_wrap = pool_module.wrap_progress

    def spy_wrap(iterable, *, total, desc, show):
        seen["show"] = show
        return real_wrap(iterable, total=total, desc=desc, show=show)

    monkeypatch.setattr(pool_module, "wrap_progress", spy_wrap)
    run_pool(
        _persons(1),
        catalog,
        _env(workers=1),
        TemporalRelationRules(),
        show_progress=None,
    )
    assert seen["show"] is True


def test_run_pool_progress_path_in_parallel_executor(event_yaml, monkeypatch):
    """The parallel branch must also flow through wrap_progress."""
    catalog = load_catalog(event_yaml)
    from src.scripts.persona.concurrency import pool as pool_module

    real_wrap = pool_module.wrap_progress
    seen: list[bool] = []

    def spy_wrap(iterable, *, total, desc, show):
        seen.append(show)
        return real_wrap(iterable, total=total, desc=desc, show=show)

    monkeypatch.setattr(pool_module, "wrap_progress", spy_wrap)
    run_pool(
        _persons(2),
        catalog,
        _env(),
        TemporalRelationRules(),
        workers=2,
        executor="thread",
        show_progress=True,
    )
    assert True in seen
