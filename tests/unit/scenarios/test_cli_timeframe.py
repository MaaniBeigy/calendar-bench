"""CLI helpers that resolve, slice, and persist `timeframe` windows."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.scripts.scenarios import cli as cli_module
from src.scripts.scenarios.calendar.loader import LoadedRun
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    CalendarSourceConfig,
    LossWeights,
    ScenarioConfig,
    ScenarioOutputConfig,
    TimeframeSpec,
)
from src.scripts.scenarios.config.timeframe import ResolvedWindow, TimeframeError
from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode

HORIZON_START = datetime.date(2026, 6, 1)
ALL_DATES = [HORIZON_START + datetime.timedelta(days=i) for i in range(28)]


def _run() -> LoadedRun:
    trace = CalendarTrace(
        person_id="p1",
        events=[
            CalendarEvent(label="x", start_minutes=540, end_minutes=600, date=d)
            for d in ALL_DATES
        ],
        contexts=[
            ContextEpisode(
                name="n",
                category="mood_emotion",
                date=d,
                start_minutes=100,
                end_minutes=140,
            )
            for d in ALL_DATES
        ],
    )
    return LoadedRun(
        traces=[trace],
        horizon_days=28,
        horizon_start_date=HORIZON_START,
    )


def _cfg(timeframe: TimeframeSpec | None = None) -> ScenarioConfig:
    return ScenarioConfig(
        id="s1",
        calendar=CalendarSourceConfig(),
        augmentation=AugmentationConfig(),
        loss=LossWeights(),
        output=ScenarioOutputConfig(dir="./output/_t"),
        timeframe=timeframe,
    )


class TestScopeRunToTimeframe:
    def test_no_timeframe_returns_run_unchanged(self):
        run = _run()
        scoped, window = cli_module._scope_run_to_timeframe(run, _cfg())
        assert scoped is run
        assert window is None

    def test_week_form_slices_run(self):
        run = _run()
        cfg = _cfg(TimeframeSpec(scale="week", start=2, end=3))
        scoped, window = cli_module._scope_run_to_timeframe(run, cfg)
        assert window == ResolvedWindow(datetime.date(2026, 6, 8), 2)
        assert scoped.horizon_days == 14
        assert scoped.horizon_start_date == datetime.date(2026, 6, 8)
        assert all(len(t.events) == 14 for t in scoped.traces)

    def test_dates_form_slices_run(self):
        run = _run()
        cfg = _cfg(
            TimeframeSpec(
                scale="dates",
                start=datetime.date(2026, 6, 15),
                end=datetime.date(2026, 6, 28),
            )
        )
        scoped, window = cli_module._scope_run_to_timeframe(run, cfg)
        assert window.start_date == datetime.date(2026, 6, 15)
        assert window.weeks == 2
        assert min(t.events[0].date for t in scoped.traces) == datetime.date(
            2026, 6, 15
        )

    def test_missing_horizon_start_raises(self):
        run = LoadedRun(traces=[], horizon_days=28, horizon_start_date=None)
        cfg = _cfg(TimeframeSpec(scale="week", start=1, end=1))
        with pytest.raises(TimeframeError, match="horizon_start_date"):
            cli_module._scope_run_to_timeframe(run, cfg)


class TestWriteAndReadTimeframeResolved:
    def test_roundtrip_week_form(self, tmp_path: Path):
        cfg = _cfg(TimeframeSpec(scale="week", start=2, end=3))
        window = ResolvedWindow(datetime.date(2026, 6, 8), 2)
        path = tmp_path / "augmented" / "timeframe.json"
        cli_module._write_timeframe_resolved(path, cfg, window)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["scenario_id"] == "s1"
        assert payload["resolved"]["start_date"] == "2026-06-08"
        assert payload["resolved"]["end_date_inclusive"] == "2026-06-21"
        assert payload["resolved"]["weeks"] == 2
        assert payload["declared"] == {"scale": "week", "start": 2, "end": 3}

        resolved = cli_module._read_timeframe_resolved(path)
        assert resolved == payload["resolved"]

    def test_dates_form_serialises_iso_strings(self, tmp_path: Path):
        cfg = _cfg(
            TimeframeSpec(
                scale="dates",
                start=datetime.date(2026, 6, 8),
                end=datetime.date(2026, 6, 21),
            )
        )
        window = ResolvedWindow(datetime.date(2026, 6, 8), 2)
        path = tmp_path / "tf.json"
        cli_module._write_timeframe_resolved(path, cfg, window)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["declared"]["start"] == "2026-06-08"
        assert payload["declared"]["end"] == "2026-06-21"

    def test_read_missing_file_returns_none(self, tmp_path: Path):
        assert cli_module._read_timeframe_resolved(tmp_path / "nope.json") is None

    def test_read_malformed_returns_none(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not-json", encoding="utf-8")
        assert cli_module._read_timeframe_resolved(path) is None


class TestMultiToScenarioConfigPropagation:
    def test_timeframe_carries_into_flat_config(self):
        from src.scripts.scenarios.config.schema import (
            AugmentationMethodConfig,
            ExperimentScenariosConfig,
            ScenarioDefinition,
        )

        spec = TimeframeSpec(scale="week", start=1, end=2)
        scenario = ScenarioDefinition(
            id="s1",
            timeframe=spec,
            augmentation=[AugmentationMethodConfig(method="greedy")],
        )
        exp = ExperimentScenariosConfig(experiment_id="e1", scenarios=[scenario])
        flat = cli_module._method_cfg_to_scenario_config(
            exp, scenario, scenario.augmentation[0]
        )
        assert flat.timeframe is not None
        assert flat.timeframe.scale == "week"
        assert flat.timeframe.start == 1
        assert flat.timeframe.end == 2
