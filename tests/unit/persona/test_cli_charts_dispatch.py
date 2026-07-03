"""Edge cases for `_run_scenarios` argv build and the `charts` subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.scripts.persona import cli


def _capture_scenarios_main(monkeypatch) -> list[list[str]]:
    """Replace `scenarios_main` with a capturing stub that returns 0."""
    captured: list[list[str]] = []

    def fake_main(argv):  # type: ignore[no-untyped-def]
        captured.append(list(argv))
        return 0

    from src.scripts.scenarios import cli as scenarios_cli

    monkeypatch.setattr(scenarios_cli, "main", fake_main)
    return captured


def test_run_scenarios_handles_scenario_id_as_list(monkeypatch):
    """A list-valued `scenario_id` is stringified item-by-item."""
    captured = _capture_scenarios_main(monkeypatch)
    args = argparse.Namespace(
        scenario="path/to/scen.yaml",
        seed=None,
        workers=None,
        scenario_id=["alpha", 7, "beta"],
        scenario_method=None,
        charts=None,
        quiet=False,
    )
    rc = cli._run_scenarios(args)
    assert rc == 0
    [argv] = captured
    assert "--scenario-id" in argv
    idx = argv.index("--scenario-id")
    assert argv[idx + 1 : idx + 4] == ["alpha", "7", "beta"]


def test_run_scenarios_skips_scenario_id_argv_when_string_is_blank(monkeypatch):
    """A comma-only string splits to no ids and the flag is dropped."""
    captured = _capture_scenarios_main(monkeypatch)
    args = argparse.Namespace(
        scenario="path/to/scen.yaml",
        seed=None,
        workers=None,
        scenario_id=" , , ",
        scenario_method=None,
        charts=None,
        quiet=False,
    )
    cli._run_scenarios(args)
    [argv] = captured
    assert "--scenario-id" not in argv


def test_load_event_persona_returns_none_when_snapshot_missing(tmp_path: Path):
    """No `used_configs.json` to `(None, None)`."""
    out = cli._load_event_persona_from_run_dir(tmp_path)
    assert out == (None, None)


def test_load_event_persona_returns_none_when_snapshot_is_malformed(tmp_path: Path):
    """Snapshot present but missing required keys; both configs come back None."""
    snapshot = tmp_path / "used_configs.json"
    snapshot.write_text(json.dumps({"environment": {}}), encoding="utf-8")
    out = cli._load_event_persona_from_run_dir(tmp_path)
    assert out == (None, None)


def _generate(
    tmp_path: Path,
    environment_yaml: Path,
    persona_yaml: Path,
    event_yaml: Path,
    rules_yaml: Path,
) -> Path:
    """Generate a fresh run under `tmp_path/run`."""
    out_dir = tmp_path / "run"
    cli.main(
        [
            "generate",
            "--environment",
            str(environment_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--out-dir",
            str(out_dir),
            "--workers",
            "1",
            "--executor",
            "thread",
            "--seed",
            "1234",
            "--allow-violations",
        ]
    )
    return out_dir


def test_charts_lines_skips_event_without_matching_persona_schedules(
    tmp_path,
    environment_yaml,
    persona_yaml,
    event_yaml,
    rules_yaml,
    monkeypatch,
):
    """An event mapped to a persona that has no schedules is silently skipped."""
    out_dir = _generate(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )

    from src.scripts.persona.analytics import charts as charts_mod

    def fake_variable_events(event_cfg, persona_cfg):
        return {"ghost_persona": {"phantom_event"}}

    monkeypatch.setattr(charts_mod, "variable_heatmap_events", fake_variable_events)

    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "lines",
        ]
    )
    assert code == 0


def test_charts_heatmap_without_snapshot_runs_legacy_aggregate(
    tmp_path,
    environment_yaml,
    persona_yaml,
    event_yaml,
    rules_yaml,
):
    """With `used_configs.json` removed the heatmap branch falls back to the legacy aggregate."""
    out_dir = _generate(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    snapshot = out_dir / "used_configs.json"
    if snapshot.exists():
        snapshot.unlink()
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "heatmap",
            "--event",
            "lunch",
        ]
    )
    assert code == 0


def test_charts_heatmap_per_event_mode_renders_both_metrics(
    tmp_path,
    environment_yaml,
    persona_yaml,
    event_yaml,
    rules_yaml,
):
    """`charts --kind heatmap` without `--event` emits a count and a duration PNG per variable event."""
    out_dir = _generate(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    assert (out_dir / "used_configs.json").is_file()
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "heatmap",
        ]
    )
    assert code == 0
    metric_pngs = sorted((out_dir / "charts").glob("heatmap_*_count_mean.png"))
    duration_pngs = sorted((out_dir / "charts").glob("heatmap_*_duration_mean.png"))
    assert metric_pngs
    assert duration_pngs
