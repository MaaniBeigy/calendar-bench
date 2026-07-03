"""Unit tests for src.scripts.persona.analytics.benchmark."""

from __future__ import annotations

import datetime as _dt
import json

from src.scripts.persona.analytics.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkResult,
    benchmark_pool,
    render_report,
    report_to_dict,
    write_report,
)
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


def _env() -> EnvironmentConfig:
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
        parallelism=ParallelismConfig(workers=1, executor="thread"),
    )


def _person(persona_id: str, seed: int = 10) -> Person:
    return make_person(
        person_id=f"{persona_id}_0000",
        persona_id=persona_id,
        person_seed=seed,
        occupation_status="student",
        stages=[make_stage("sleep", time="23:00")],
    )


class _FakeTimer:
    """Returns a strictly increasing sequence of values, two per row.

    perf_counter is called twice per benchmark row (start + stop). Consecutive
    pairs come from `pairs`, so wall_seconds for row k is `pairs[k][1] - pairs[k][0]`.
    """

    def __init__(self, pairs: list[tuple[float, float]]) -> None:
        flat: list[float] = []
        for a, b in pairs:
            flat.append(a)
            flat.append(b)
        self._iter = iter(flat)

    def __call__(self) -> float:
        return next(self._iter)


def test_benchmark_empty_configs_returns_empty_report(event_yaml):
    catalog = load_catalog(event_yaml)
    out = benchmark_pool([_person("a")], catalog, _env(), TemporalRelationRules(), [])
    assert out.rows == []


def test_benchmark_records_wall_seconds_from_injected_timer(event_yaml):
    catalog = load_catalog(event_yaml)
    persons = [_person("a"), _person("b")]
    timer = _FakeTimer([(0.0, 1.5), (10.0, 11.25)])
    out = benchmark_pool(
        persons,
        catalog,
        _env(),
        TemporalRelationRules(),
        [
            BenchmarkConfig(workers=1, executor="thread", chunk_size=1),
            BenchmarkConfig(workers=2, executor="thread", chunk_size=1),
        ],
        timer=timer,
    )
    assert [r.wall_seconds for r in out.rows] == [1.5, 1.25]


def test_benchmark_marks_first_row_as_baseline(event_yaml):
    catalog = load_catalog(event_yaml)
    persons = [_person("a")]
    timer = _FakeTimer([(0.0, 0.1), (1.0, 1.05)])
    out = benchmark_pool(
        persons,
        catalog,
        _env(),
        TemporalRelationRules(),
        [
            BenchmarkConfig(workers=1, executor="thread"),
            BenchmarkConfig(workers=2, executor="thread"),
        ],
        timer=timer,
    )
    assert out.rows[0].matches_baseline is True
    # Second row uses thread executor on the same persons, so output matches.
    assert out.rows[1].matches_baseline is True


def test_benchmark_auto_chunk_does_not_change_outputs(event_yaml):
    catalog = load_catalog(event_yaml)
    persons = [_person("a"), _person("b")]
    timer = _FakeTimer([(0.0, 0.1), (1.0, 1.2)])
    out = benchmark_pool(
        persons,
        catalog,
        _env(),
        TemporalRelationRules(),
        [
            BenchmarkConfig(workers=2, executor="thread", chunk_size=1),
            BenchmarkConfig(workers=2, executor="thread", auto_chunk=True),
        ],
        timer=timer,
    )
    assert all(r.matches_baseline for r in out.rows)


def test_benchmark_records_n_persons_and_workers(event_yaml):
    catalog = load_catalog(event_yaml)
    persons = [_person("a"), _person("b"), _person("c")]
    timer = _FakeTimer([(0.0, 0.1)])
    out = benchmark_pool(
        persons,
        catalog,
        _env(),
        TemporalRelationRules(),
        [BenchmarkConfig(workers=4, executor="thread", chunk_size=1)],
        timer=timer,
    )
    assert out.rows[0].n_persons == 3
    assert out.rows[0].workers == 4
    assert out.rows[0].chunk_size == 1
    assert out.rows[0].auto_chunk is False


def test_render_report_with_no_rows_says_no_configurations():
    text = render_report(BenchmarkReport())
    assert "No configurations ran." in text


def test_render_report_includes_header_and_rows():
    rep = BenchmarkReport(
        rows=[
            BenchmarkResult(
                workers=2,
                executor="thread",
                chunk_size=1,
                auto_chunk=False,
                wall_seconds=1.234,
                n_persons=5,
                matches_baseline=True,
            )
        ]
    )
    text = render_report(rep)
    assert "workers" in text
    assert "executor" in text
    assert "1.234" in text
    assert "yes" in text


def test_render_report_marks_auto_chunk_rows():
    rep = BenchmarkReport(
        rows=[
            BenchmarkResult(
                workers=2,
                executor="thread",
                chunk_size=None,
                auto_chunk=True,
                wall_seconds=0.5,
                n_persons=3,
                matches_baseline=True,
            )
        ]
    )
    assert "auto" in render_report(rep)


def test_report_to_dict_round_trips_through_json():
    rep = BenchmarkReport(
        rows=[
            BenchmarkResult(
                workers=2,
                executor="thread",
                chunk_size=1,
                auto_chunk=False,
                wall_seconds=1.0,
                n_persons=2,
                matches_baseline=True,
            )
        ]
    )
    payload = report_to_dict(rep)
    assert json.dumps(payload)  # serialisable
    assert payload["rows"][0]["wall_seconds"] == 1.0


def test_write_report_creates_both_files(tmp_path):
    rep = BenchmarkReport(
        rows=[
            BenchmarkResult(
                workers=1,
                executor="thread",
                chunk_size=1,
                auto_chunk=False,
                wall_seconds=0.1,
                n_persons=1,
                matches_baseline=True,
            )
        ]
    )
    txt, js = write_report(rep, tmp_path)
    assert txt.name == "benchmark_report.txt"
    assert js.name == "benchmark_report.json"
    assert "wall_s" in txt.read_text(encoding="utf-8")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["rows"][0]["workers"] == 1


def test_perf_counter_default_runs_without_explicit_timer(event_yaml):
    """The default `time.perf_counter` is used when no timer is injected."""
    catalog = load_catalog(event_yaml)
    out = benchmark_pool(
        [_person("a")],
        catalog,
        _env(),
        TemporalRelationRules(),
        [BenchmarkConfig(workers=1, executor="thread", chunk_size=1)],
    )
    assert len(out.rows) == 1
    assert out.rows[0].wall_seconds >= 0.0
