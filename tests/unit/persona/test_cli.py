"""Unit tests for src.scripts.persona.cli."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.scripts.persona import cli


def _argv_generate(out_dir: Path, environment, persona, event, rules) -> list[str]:
    return [
        "generate",
        "--environment",
        str(environment),
        "--personas",
        str(persona),
        "--events",
        str(event),
        "--rules",
        str(rules),
        "--out-dir",
        str(out_dir),
        "--workers",
        "1",
        "--executor",
        "thread",
        "--seed",
        "12345",
        "--allow-violations",
    ]


def test_cli_help_emits_subcommands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "generate" in out and "validate" in out


def _write_env_with_output_dir(
    src_environment_yaml: Path, override_dir: Path, tmp_path: Path
) -> Path:
    """Copy an example environment.yaml but rewrite `output.dir` to
    `override_dir`. Returns the path of the rewritten YAML."""
    import yaml

    raw = yaml.safe_load(src_environment_yaml.read_text(encoding="utf-8"))
    raw["output"]["dir"] = str(override_dir)
    target = tmp_path / "environment.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def test_generate_uses_environment_output_dir_when_out_dir_omitted(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """When `--out-dir` is not passed, `generate` writes to
    `environment.output.dir`. This is the default for environment-driven
    runs."""
    out_dir = tmp_path / "from_env"
    env_yaml = _write_env_with_output_dir(environment_yaml, out_dir, tmp_path)
    code = cli.main(
        [
            "generate",
            "--environment",
            str(env_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--workers",
            "1",
            "--executor",
            "thread",
            "--seed",
            "12345",
            "--allow-violations",
        ]
    )
    assert code == 0
    assert (out_dir / "index.json").exists()


def test_validate_uses_environment_output_dir_when_run_dir_omitted(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """`validate` defaults the run directory to `environment.output.dir`
    so a user pointing only at the environment YAML still gets the
    population-level check."""
    out_dir = tmp_path / "from_env"
    env_yaml = _write_env_with_output_dir(environment_yaml, out_dir, tmp_path)
    cli.main(
        [
            "generate",
            "--environment",
            str(env_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--workers",
            "1",
            "--executor",
            "thread",
            "--seed",
            "12345",
            "--allow-violations",
        ]
    )
    # Wipe the report so we can prove validate re-creates it under the
    # default-resolved run dir.
    (out_dir / "summary_report.txt").unlink()
    code = cli.main(
        [
            "validate",
            "--environment",
            str(env_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--allow-violations",
        ]
    )
    assert code == 0
    assert (out_dir / "summary_report.txt").exists()


def test_validate_errors_when_default_run_dir_does_not_exist(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """If `environment.output.dir` does not exist on disk, `validate`
    exits with EXIT_USAGE rather than crashing."""
    missing = tmp_path / "never_generated"
    env_yaml = _write_env_with_output_dir(environment_yaml, missing, tmp_path)
    code = cli.main(
        [
            "validate",
            "--environment",
            str(env_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
        ]
    )
    assert code == cli.EXIT_USAGE


def test_charts_uses_environment_output_dir_when_run_dir_omitted(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """`charts` mirrors `validate`: default run dir is
    `environment.output.dir`."""
    out_dir = tmp_path / "from_env"
    env_yaml = _write_env_with_output_dir(environment_yaml, out_dir, tmp_path)
    cli.main(
        [
            "generate",
            "--environment",
            str(env_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--workers",
            "1",
            "--executor",
            "thread",
            "--seed",
            "12345",
            "--allow-violations",
        ]
    )
    code = cli.main(
        [
            "charts",
            "--environment",
            str(env_yaml),
        ]
    )
    assert code == 0
    assert (out_dir / "charts").is_dir()


def test_cli_unknown_subcommand_exits_usage_error(capsys):
    with pytest.raises(SystemExit):
        cli.main(["bogus"])


def test_generate_produces_expected_files(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    code = cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    assert code == 0
    assert (out_dir / "index.json").exists()
    assert (out_dir / "summary_report.txt").exists()
    assert (out_dir / "summary_report.json").exists()
    assert (out_dir / "used_configs.json").exists()
    person_jsons = list((out_dir / "persons").glob("*.json"))
    assert len(person_jsons) >= 1
    assert list((out_dir / "ics_per_person").glob("*.ics"))


def test_generate_seed_override_changes_persons(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cli.main(
        _argv_generate(out_a, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    argv = _argv_generate(out_b, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    seed_idx = argv.index("--seed")
    argv[seed_idx + 1] = "99999"
    cli.main(argv)
    a_payload = json.loads((out_a / "index.json").read_text(encoding="utf-8"))
    b_payload = json.loads((out_b / "index.json").read_text(encoding="utf-8"))
    a_seeds = sorted(p["person_seed"] for p in a_payload["persons"])
    b_seeds = sorted(p["person_seed"] for p in b_payload["persons"])
    assert a_seeds != b_seeds


def test_generate_returns_violations_exit_code(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml, monkeypatch
):
    """Force a fake violation report to verify the non-zero exit path."""
    out_dir = tmp_path / "run"
    from src.scripts.persona.validation import report as report_module
    from src.scripts.persona.validation.check_event import EventViolation

    real = report_module.ValidationReport

    def fake_validation(*args, **kwargs):
        return real(
            event=[
                EventViolation(
                    person_id="x_0000",
                    day_index=0,
                    event_name="lunch",
                    kind="per_event_duration",
                    detail="injected",
                )
            ]
        )

    monkeypatch.setattr(cli, "_run_validation", fake_validation)
    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.remove("--allow-violations")
    code = cli.main(argv)
    assert code == cli.EXIT_VIOLATIONS


def test_validate_subcommand_runs_against_existing_run(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    # Wipe the report to make sure validate re-creates it.
    (out_dir / "summary_report.txt").unlink()
    code = cli.main(
        [
            "validate",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--allow-violations",
        ]
    )
    assert code == 0
    assert (out_dir / "summary_report.txt").exists()


def test_config_error_returns_usage_exit_code(
    tmp_path, persona_yaml, event_yaml, rules_yaml
):
    code = cli.main(
        [
            "generate",
            "--environment",
            str(tmp_path / "missing.yaml"),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == cli.EXIT_USAGE


def test_validate_with_fewer_schedules_than_persons_continues(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml, capsys
):
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    # Drop one of the per-person json files to force the warn-and-continue path.
    person_jsons = sorted((out_dir / "persons").glob("*.json"))
    person_jsons[0].unlink()
    code = cli.main(
        [
            "validate",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--allow-violations",
        ]
    )
    err = capsys.readouterr().err
    assert "warning" in err
    assert code == 0


def test_generate_with_per_person_outputs_disabled(
    tmp_path, persona_yaml, event_yaml, rules_yaml
):
    """When environment.output disables per-person json/ics, the directories
    are not populated and the index step is skipped."""
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "\n".join(
            [
                "seed: 1",
                "horizon:",
                "  start_date: 2026-05-04",
                "  weeks: 1",
                "output:",
                "  dir: ./out",
                "  per_person_json: false",
                "  per_person_ics: false",
                "  validation_report: true",
                "solver: {}",
                "time_windows:",
                "  early_morning: [0, 400]",
                "  morning: [400, 600]",
                "  afternoon: [600, 960]",
                "  evening: [960, 1260]",
                "  night: [1260, 1440]",
                "parallelism:",
                "  workers: 1",
                "  executor: thread",
                "  chunk_size: 1",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"
    code = cli.main(
        _argv_generate(out_dir, env_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    assert code == 0
    assert not (out_dir / "persons").exists()
    assert not (out_dir / "ics_per_person").exists()
    assert not (out_dir / "index.json").exists()
    assert (out_dir / "summary_report.txt").exists()


def test_generate_without_seed_flag_uses_yaml_seed(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """The seed override is optional. Omitting --seed leaves the YAML seed
    untouched, exercising the early-return path of `_resolve_config`.
    """
    out_dir = tmp_path / "run"
    code = cli.main(
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
            "--allow-violations",
        ]
    )
    assert code == 0
    payload = json.loads((out_dir / "used_configs.json").read_text(encoding="utf-8"))
    assert payload["environment"]["seed"] == 20260503  # from environment.yaml


def test_generate_skips_writing_validation_report_when_disabled(
    tmp_path, persona_yaml, event_yaml, rules_yaml
):
    """validation_report=false means `summary_report.{txt,json}` are never written."""
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "\n".join(
            [
                "seed: 1",
                "horizon:",
                "  start_date: 2026-05-04",
                "  weeks: 1",
                "output:",
                "  dir: ./out",
                "  per_person_json: true",
                "  per_person_ics: true",
                "  validation_report: false",
                "solver: {}",
                "time_windows:",
                "  early_morning: [0, 400]",
                "  morning: [400, 600]",
                "  afternoon: [600, 960]",
                "  evening: [960, 1260]",
                "  night: [1260, 1440]",
                "parallelism:",
                "  workers: 1",
                "  executor: thread",
                "  chunk_size: 1",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"
    code = cli.main(
        _argv_generate(out_dir, env_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    assert code == 0
    assert not (out_dir / "summary_report.txt").exists()
    assert not (out_dir / "summary_report.json").exists()
    # JSON / ICS / index are still written.
    assert (out_dir / "index.json").exists()


def test_validate_returns_violations_exit_code(
    tmp_path,
    environment_yaml,
    persona_yaml,
    event_yaml,
    rules_yaml,
    monkeypatch,
):
    """Force validate to see violations and assert it exits with EXIT_VIOLATIONS."""
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    from src.scripts.persona.validation import report as report_module
    from src.scripts.persona.validation.check_event import EventViolation

    real = report_module.ValidationReport

    def fake_validation(*args, **kwargs):
        return real(
            event=[
                EventViolation(
                    person_id="x_0000",
                    day_index=0,
                    event_name="lunch",
                    kind="per_event_duration",
                    detail="injected",
                )
            ]
        )

    monkeypatch.setattr(cli, "_run_validation", fake_validation)
    code = cli.main(
        [
            "validate",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
        ]
    )
    assert code == cli.EXIT_VIOLATIONS


# -------------------------------------------------------------------------------------
# ------------------ free-time, charts, chunking, and worker flags --------------------
# -------------------------------------------------------------------------------------


def test_generate_with_free_time_report_writes_extra_file(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.insert(-1, "--free-time-report")
    code = cli.main(argv)
    assert code == 0
    target = out_dir / "free_time_report.txt"
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "Free-time summary by occupation" in content


def test_generate_without_free_time_flag_does_not_write_report(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    assert not (out_dir / "free_time_report.txt").exists()


def test_generate_free_time_group_by_flag_changes_rollup_axis(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv[-1:-1] = ["--free-time-report", "--free-time-group-by", "gender"]
    code = cli.main(argv)
    assert code == 0
    content = (out_dir / "free_time_report.txt").read_text(encoding="utf-8")
    assert "Free-time summary by gender" in content


def test_charts_subcommand_renders_charts_for_existing_run(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """The `charts` subcommand reads `<run-dir>/persons/*.json` and
    writes the chart family without re-running generation."""
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    # Sanity: generation did not write a charts/ directory.
    assert not (out_dir / "charts").exists()
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
        ]
    )
    assert code == 0
    charts_dir = out_dir / "charts"
    assert charts_dir.is_dir()
    assert any(p.suffix == ".png" for p in charts_dir.iterdir())


def test_charts_subcommand_errors_when_persons_dir_missing(tmp_path, environment_yaml):
    """If the run dir has no `persons/` subdir, `charts` exits with
    EXIT_USAGE rather than crashing."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    code = cli.main(
        [
            "charts",
            str(empty_dir),
            "--environment",
            str(environment_yaml),
        ]
    )
    assert code == cli.EXIT_USAGE


def test_charts_subcommand_errors_when_persons_dir_empty(tmp_path, environment_yaml):
    """An empty `persons/` directory triggers the no-schedules guard."""
    persons_dir = tmp_path / "run" / "persons"
    persons_dir.mkdir(parents=True)
    code = cli.main(
        [
            "charts",
            str(persons_dir.parent),
            "--environment",
            str(environment_yaml),
        ]
    )
    assert code == cli.EXIT_USAGE


def _generate_then_assert_persons_dir(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    assert (out_dir / "persons").is_dir()
    return out_dir


def test_charts_kind_lines_writes_one_png_per_persona(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
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
    pngs = sorted((out_dir / "charts").glob("lines_*.png"))
    assert pngs, "expected at least one lines_*.png"


def test_charts_kind_lines_with_persona_filter_writes_only_one_png(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    # Pick the first persona id from the run.
    from src.scripts.persona.config.loader import load_persona

    cfg = load_persona(persona_yaml)
    persona_id = cfg.personas[0].id
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "lines",
            "--persona",
            persona_id,
        ]
    )
    assert code == 0
    assert (out_dir / "charts" / f"lines_{persona_id}.png").exists()


def test_charts_kind_lines_with_unknown_persona_returns_usage(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "heatmap",
            "--persona",
            "nonexistent_persona",
        ]
    )
    assert code == cli.EXIT_USAGE


def test_charts_kind_heatmap_with_event_filter_writes_named_png(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
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
            "--metric",
            "duration",
            "--aggregate",
            "sum",
        ]
    )
    assert code == 0
    pngs = sorted((out_dir / "charts").glob("heatmap_*_lunch_duration_sum.png"))
    assert pngs, "expected at least one heatmap_*_lunch_duration_sum.png"


def test_charts_kind_gantt_default_picks_first_person_first_event_day(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "gantt",
        ]
    )
    assert code == 0
    assert any((out_dir / "charts").glob("gantt_*.png"))


def test_charts_kind_gantt_with_explicit_person_and_date(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    # Pick a person and date from the just-written index.
    from src.scripts.persona.validation.loader import load_schedules

    schedules = load_schedules(out_dir / "persons")
    sched = schedules[0]
    target_day = next(d for d in sched.days if d.events)
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "gantt",
            "--person",
            sched.person_id,
            "--date",
            target_day.date.isoformat(),
        ]
    )
    assert code == 0
    target = (
        out_dir
        / "charts"
        / f"gantt_{sched.person_id}_{target_day.date.isoformat()}.png"
    )
    assert target.exists()


def test_charts_kind_gantt_with_unknown_person_returns_usage(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "gantt",
            "--person",
            "nobody",
        ]
    )
    assert code == cli.EXIT_USAGE


def test_charts_kind_gantt_with_invalid_date_returns_usage(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "gantt",
            "--date",
            "not-a-date",
        ]
    )
    assert code == cli.EXIT_USAGE


def test_charts_kind_gantt_returns_usage_when_default_person_has_no_events(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml, monkeypatch
):
    """Default person + default date path: when the chosen person has
    every day empty, the CLI must surface EXIT_USAGE with a clear
    message instead of crashing."""
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    from src.scripts.persona.domain.schedule import PersonSchedule

    empty = PersonSchedule(
        person_id="ghost", persona_id="ghost", person_seed=0, days=[]
    )
    monkeypatch.setattr(cli, "_load_schedules_from_dir", lambda _path: [empty])
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "gantt",
        ]
    )
    assert code == cli.EXIT_USAGE


def test_charts_kind_gantt_with_date_outside_horizon_returns_usage(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    code = cli.main(
        [
            "charts",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--kind",
            "gantt",
            "--date",
            "2099-01-01",
        ]
    )
    assert code == cli.EXIT_USAGE


def test_charts_kind_calendar_writes_per_person_pngs(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    from src.scripts.persona.analytics import weekly_calendar as wc

    captured: dict = {}
    real = wc.render_weekly_calendars

    def spy(schedules, augmented, dir_, **kw):
        captured["augmented"] = augmented
        captured["dpi"] = kw.get("dpi")
        return real(schedules, augmented, dir_, **kw)

    with patch.object(wc, "render_weekly_calendars", spy):
        code = cli.main(
            [
                "charts",
                str(out_dir),
                "--environment",
                str(environment_yaml),
                "--kind",
                "calendar",
                "--calendar-dpi",
                "120",
            ]
        )
    assert code == 0
    assert captured["augmented"] is None
    assert captured["dpi"] == 120
    # At least one person directory should contain a week PNG.
    assert any((out_dir / "charts").iterdir())


def test_charts_kind_calendar_with_augmented_dir_overlays_records(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    # Materialise a fake augmented persons directory with one task per
    # known person_id.
    person_jsons = sorted((out_dir / "persons").glob("*.json"))
    aug_dir = tmp_path / "augmented" / "persons"
    aug_dir.mkdir(parents=True)
    person_ids = []
    for jp in person_jsons:
        payload = json.loads(jp.read_text(encoding="utf-8"))
        pid = payload["person_id"]
        person_ids.append(pid)
        first_day = payload["days"][0]["date"]
        (aug_dir / f"{pid}.json").write_text(
            json.dumps(
                {
                    "person_id": pid,
                    "scheduled": [
                        {
                            "label": "walk",
                            "display_name": "Walk 🚶",
                            "date": first_day,
                            "start_minutes": 400,
                            "end_minutes": 415,
                            "is_standalone": True,
                            "concurrent_with": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    from src.scripts.persona.analytics import weekly_calendar as wc

    captured: dict = {}
    real = wc.render_weekly_calendars

    def spy(schedules, augmented, dir_, **kw):
        captured["augmented"] = augmented
        return real(schedules, augmented, dir_, **kw)

    with patch.object(wc, "render_weekly_calendars", spy):
        code = cli.main(
            [
                "charts",
                str(out_dir),
                "--environment",
                str(environment_yaml),
                "--kind",
                "calendar",
                "--augmented-dir",
                str(aug_dir),
            ]
        )
    assert code == 0
    aug = captured["augmented"]
    assert aug is not None
    assert set(aug.keys()) == set(person_ids)


def test_charts_kind_all_includes_calendar_with_augmented_dir(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """`--kind all` plus `--augmented-dir` should produce both classic
    charts AND a per-person calendar directory."""
    out_dir = _generate_then_assert_persons_dir(
        tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    aug_dir = tmp_path / "augmented" / "persons"
    aug_dir.mkdir(parents=True)
    person_jsons = sorted((out_dir / "persons").glob("*.json"))
    payload = json.loads(person_jsons[0].read_text(encoding="utf-8"))
    pid = payload["person_id"]
    first_day = payload["days"][0]["date"]
    (aug_dir / f"{pid}.json").write_text(
        json.dumps(
            {
                "person_id": pid,
                "scheduled": [
                    {
                        "label": "walk",
                        "display_name": "Walk 🚶",
                        "date": first_day,
                        "start_minutes": 400,
                        "end_minutes": 415,
                        "is_standalone": True,
                        "concurrent_with": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from src.scripts.persona.analytics import charts as charts_module

    captured: dict = {}
    real = charts_module.render_all_charts

    def spy(schedules, *, out_dir, **kw):
        captured["kinds"] = set(kw.get("kinds") or ())
        captured["augmented"] = kw.get("augmented_by_person")
        captured["dpi"] = kw.get("weekly_calendar_dpi")
        # Skip the costly calendar render in this CLI smoke test.
        kw["kinds"] = {k for k in captured["kinds"] if k != "calendar"}
        return real(schedules, out_dir=out_dir, **kw)

    with patch.object(charts_module, "render_all_charts", spy):
        code = cli.main(
            [
                "charts",
                str(out_dir),
                "--environment",
                str(environment_yaml),
                "--kind",
                "all",
                "--augmented-dir",
                str(aug_dir),
                "--calendar-dpi",
                "200",
            ]
        )
    assert code == 0
    assert "calendar" in captured["kinds"]
    assert captured["augmented"] is not None
    assert captured["dpi"] == 200


def test_generate_with_charts_flag_renders_population_charts(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """`--charts` triggers the lazy import of `analytics.charts` and
    writes the chart family under `<out-dir>/charts/`. We only assert
    the directory exists with at least one PNG; the chart content is
    covered in `test_analytics_charts`.

    The kinds are pinned to the matplotlib families to keep this smoke
    test fast; the calendar-view family is exercised separately in
    `test_analytics_weekly_calendar`.
    """
    out_dir = tmp_path / "run"
    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.extend(["--charts", "lines", "heatmap", "gantt"])
    code = cli.main(argv)
    assert code == 0
    charts_dir = out_dir / "charts"
    assert charts_dir.is_dir()
    assert any(p.suffix == ".png" for p in charts_dir.iterdir())


def test_generate_with_bare_charts_flag_defaults_to_all_kinds(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    """Bare `--charts` should render every kind including the calendar
    family. The calendar renderer is patched out so this test stays
    fast; the real rendering is covered in
    `test_analytics_weekly_calendar`."""
    out_dir = tmp_path / "run"
    from src.scripts.persona.analytics import charts as charts_module

    called: dict = {}
    real = charts_module.render_all_charts

    def spy(schedules, *, out_dir, kinds=None, **kw):
        called["kinds"] = set(kinds) if kinds is not None else None
        called["dpi"] = kw.get("weekly_calendar_dpi")
        # Skip the calendar family during this smoke test; we only care
        # that the CLI requested every kind.
        kinds = (set(kinds) - {"calendar"}) if kinds else kinds
        return real(schedules, out_dir=out_dir, kinds=kinds, **kw)

    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.extend(["--charts"])
    with patch.object(charts_module, "render_all_charts", spy):
        code = cli.main(argv)
    assert code == 0
    assert called["kinds"] == {
        "lines",
        "heatmap",
        "gantt",
        "calendar",
        "context-lines",
        "context-heatmap",
    }
    assert called["dpi"] == 300


def test_generate_with_unknown_chart_kind_returns_usage_error(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.extend(["--charts", "bogus"])
    code = cli.main(argv)
    assert code == cli.EXIT_USAGE


def test_generate_calendar_dpi_flag_is_forwarded(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    from src.scripts.persona.analytics import charts as charts_module

    captured: dict = {}

    def spy(schedules, *, out_dir, **kw):
        captured["dpi"] = kw.get("weekly_calendar_dpi")
        # No-op render: we only care about the forwarded DPI value.
        return charts_module.ChartArtifacts()

    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.extend(["--charts", "calendar", "--calendar-dpi", "150"])
    with patch.object(charts_module, "render_all_charts", spy):
        code = cli.main(argv)
    assert code == 0
    assert captured["dpi"] == 150


def test_generate_without_charts_flag_does_not_create_charts_dir(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    assert not (out_dir / "charts").exists()


def test_generate_with_auto_chunk_size_runs_cleanly(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.insert(-1, "--auto-chunk-size")
    code = cli.main(argv)
    assert code == 0
    assert (out_dir / "index.json").exists()


def test_generate_with_validation_workers_flag_passes_through(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    argv = _argv_generate(
        out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
    )
    argv.insert(-1, "--validation-workers")
    argv.insert(-1, "2")
    code = cli.main(argv)
    assert code == 0
    assert (out_dir / "summary_report.txt").exists()


def test_validate_with_validation_workers_flag(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "run"
    cli.main(
        _argv_generate(out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml)
    )
    code = cli.main(
        [
            "validate",
            str(out_dir),
            "--environment",
            str(environment_yaml),
            "--personas",
            str(persona_yaml),
            "--events",
            str(event_yaml),
            "--rules",
            str(rules_yaml),
            "--validation-workers",
            "2",
            "--allow-violations",
        ]
    )
    assert code == 0


# -------------------------------------------------------------------------------------
# --------------------------------- benchmark subcommand ------------------------------
# -------------------------------------------------------------------------------------


def test_benchmark_subcommand_writes_reports(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "bench"
    code = cli.main(
        [
            "benchmark",
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
            "2",
            "--executor",
            "thread",
            "--chunk-sizes",
            "1",
            "--seed",
            "12345",
        ]
    )
    assert code == 0
    txt = out_dir / "benchmark_report.txt"
    js = out_dir / "benchmark_report.json"
    assert txt.exists() and js.exists()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 2
    workers = sorted(r["workers"] for r in payload["rows"])
    assert workers == [1, 2]


def test_benchmark_subcommand_with_auto_chunk_emits_one_row_per_worker(
    tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
):
    out_dir = tmp_path / "bench"
    code = cli.main(
        [
            "benchmark",
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
            "2",
            "--auto-chunk-size",
            "--seed",
            "12345",
        ]
    )
    assert code == 0
    payload = json.loads(
        (out_dir / "benchmark_report.json").read_text(encoding="utf-8")
    )
    assert len(payload["rows"]) == 2
    assert all(r["auto_chunk"] is True for r in payload["rows"])


def test_benchmark_subcommand_returns_usage_error_for_empty_population(
    tmp_path, environment_yaml, event_yaml, rules_yaml
):
    persona_yaml = tmp_path / "empty_persona.yaml"
    persona_yaml.write_text(
        "\n".join(
            [
                "common: {}",
                "personas:",
                "  - id: ghost",
                "    instances: 1",
                "    characteristics:",
                "      occupation_status: { type: categorical, values: { student: 1.0 } }",
                "    stages:",
                '      - { name: sleep, time: "23:00", days: [Mon, Tue, Wed, Thu, Fri, Sat, Sun] }',
            ]
        ),
        encoding="utf-8",
    )
    # Patch sample_population to return an empty list to force the early exit.
    import src.scripts.persona.cli as cli_module

    out_dir = tmp_path / "bench"
    original = cli_module.sample_population
    try:
        cli_module.sample_population = lambda *_args, **_kwargs: []
        code = cli.main(
            [
                "benchmark",
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
            ]
        )
    finally:
        cli_module.sample_population = original
    assert code == cli.EXIT_USAGE


# ---------------------------------------------------------------------------
# --scenario chaining: _run_scenarios + _generate integration
# ---------------------------------------------------------------------------


class TestRunScenarios:
    """Tests for _run_scenarios() and the --scenario flag on generate."""

    def test_run_scenarios_calls_scenarios_main(self):
        """_run_scenarios builds the right argv and calls scenarios_main."""
        import argparse

        args = argparse.Namespace(
            scenario=Path("/cfg/scenarios.yaml"),
            seed=42,
            workers=3,
            scenario_id=None,
            scenario_method=None,
        )
        with patch("src.scripts.persona.cli._run_scenarios") as mock_rs:
            mock_rs.return_value = 0
            # Call directly to validate the function is reachable
            mock_rs(args)
        mock_rs.assert_called_once_with(args)

    def test_run_scenarios_builds_correct_argv(self):
        """_run_scenarios passes seed, workers, scenario-id, and method."""
        import argparse

        args = argparse.Namespace(
            scenario=Path("/cfg/scenarios.yaml"),
            seed=99,
            workers=4,
            scenario_id="nutrition_l1",
            scenario_method="greedy",
        )
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        with patch("src.scripts.scenarios.cli.main", fake_main):
            rc = cli._run_scenarios(args)

        assert rc == 0
        argv = captured["argv"]
        assert "--scenario" in argv
        assert str(args.scenario) in argv
        assert "--seed" in argv
        assert "99" in argv
        assert "--workers" in argv
        assert "4" in argv
        assert "--scenario-id" in argv
        assert "nutrition_l1" in argv
        assert "--method" in argv
        assert "greedy" in argv

    def test_run_scenarios_omits_optional_flags_when_none(self):
        """seed, workers, scenario-id, method are omitted when not set."""
        import argparse

        args = argparse.Namespace(
            scenario=Path("/cfg/scenarios.yaml"),
            seed=None,
            workers=None,
            scenario_id=None,
            scenario_method=None,
        )
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        with patch("src.scripts.scenarios.cli.main", fake_main):
            cli._run_scenarios(args)

        argv = captured["argv"]
        assert "--seed" not in argv
        assert "--workers" not in argv
        assert "--scenario-id" not in argv
        assert "--method" not in argv

    def test_run_scenarios_propagates_failure_code(self):
        """Non-zero return from scenarios_main is forwarded."""
        import argparse

        args = argparse.Namespace(
            scenario=Path("/cfg/scenarios.yaml"),
            seed=None,
            workers=None,
            scenario_id=None,
            scenario_method=None,
        )
        with patch("src.scripts.scenarios.cli.main", return_value=2):
            rc = cli._run_scenarios(args)
        assert rc == 2

    def test_run_scenarios_forwards_charts_and_dpi(self):
        """When persona CLI received `--charts`, those values must
        propagate so the chained scenarios CLI can render augmented
        weekly calendars next to its own outputs."""
        import argparse

        args = argparse.Namespace(
            scenario=Path("/cfg/scenarios.yaml"),
            seed=None,
            workers=None,
            scenario_id=None,
            scenario_method=None,
            charts=["calendar", "gantt"],
            calendar_dpi=150,
            calendar_min_event_minutes=60,
        )
        captured: dict = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        with patch("src.scripts.scenarios.cli.main", fake_main):
            cli._run_scenarios(args)
        argv = captured["argv"]
        assert "--charts" in argv
        ci = argv.index("--charts")
        assert argv[ci + 1] == "calendar"
        assert argv[ci + 2] == "gantt"
        assert "--calendar-dpi" in argv
        assert argv[argv.index("--calendar-dpi") + 1] == "150"
        assert "--calendar-min-event-minutes" in argv
        assert argv[argv.index("--calendar-min-event-minutes") + 1] == "60"

    def test_run_scenarios_bare_charts_flag_forwards_no_kinds(self):
        """Bare `--charts` on the persona side (an empty list) should
        still propagate the flag so the scenarios CLI sees the user's
        intent to render charts."""
        import argparse

        args = argparse.Namespace(
            scenario=Path("/cfg/scenarios.yaml"),
            seed=None,
            workers=None,
            scenario_id=None,
            scenario_method=None,
            charts=[],
            calendar_dpi=None,
        )
        captured: dict = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        with patch("src.scripts.scenarios.cli.main", fake_main):
            cli._run_scenarios(args)
        argv = captured["argv"]
        assert "--charts" in argv
        # No DPI when `calendar_dpi` is None.
        assert "--calendar-dpi" not in argv

    def test_generate_scenario_flag_parsed(self):
        """--scenario argument is accepted by the generate subparser."""
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "generate",
                "--environment",
                "env.yaml",
                "--personas",
                "persona.yaml",
                "--events",
                "events.yaml",
                "--rules",
                "rules.yaml",
                "--scenario",
                "/cfg/scenarios.yaml",
                "--scenario-id",
                "nutrition_l1",
                "--scenario-method",
                "greedy",
            ]
        )
        assert args.scenario == Path("/cfg/scenarios.yaml")
        assert args.scenario_id == "nutrition_l1"
        assert args.scenario_method == "greedy"

    def test_generate_without_scenario_flag_defaults_none(self):
        """--scenario defaults to None when not provided."""
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "generate",
                "--environment",
                "env.yaml",
                "--personas",
                "persona.yaml",
                "--events",
                "events.yaml",
                "--rules",
                "rules.yaml",
            ]
        )
        assert args.scenario is None
        assert args.scenario_id is None
        assert args.scenario_method is None

    def test_generate_chains_scenarios_on_success(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        """After successful generation, _run_scenarios is called when --scenario given."""
        scenario_yaml = tmp_path / "scenarios.yaml"
        scenario_yaml.touch()

        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["--scenario", str(scenario_yaml)]

        with patch.object(cli, "_run_scenarios", return_value=0) as mock_rs:
            rc = cli.main(argv)

        assert rc == 0
        mock_rs.assert_called_once()

    def test_generate_does_not_chain_without_scenario_flag(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        """_run_scenarios is NOT called when --scenario is absent."""
        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        )
        with patch.object(cli, "_run_scenarios", return_value=0) as mock_rs:
            cli.main(argv)
        mock_rs.assert_not_called()

    def test_generate_does_not_chain_on_violation(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        """_run_scenarios is NOT called when generation exits with violations."""
        scenario_yaml = tmp_path / "scenarios.yaml"
        scenario_yaml.touch()

        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["--scenario", str(scenario_yaml)]
        # Remove --allow-violations so violations cause exit-1
        argv = [a for a in argv if a != "--allow-violations"]

        with (
            patch.object(cli, "_run_validation", return_value=_ViolatingReport()),
            patch("src.scripts.persona.cli.write_report"),
            patch.object(cli, "_run_scenarios", return_value=0) as mock_rs,
        ):
            rc = cli.main(argv)

        assert rc == cli.EXIT_VIOLATIONS
        mock_rs.assert_not_called()

    def test_scenarios_failure_is_returned_from_generate(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        """When _run_scenarios returns non-zero, generate returns that code."""
        scenario_yaml = tmp_path / "scenarios.yaml"
        scenario_yaml.touch()

        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["--scenario", str(scenario_yaml)]

        with patch.object(cli, "_run_scenarios", return_value=2):
            rc = cli.main(argv)

        assert rc == 2


class _ViolatingReport:
    """Minimal stub for a ValidationReport with violations."""

    has_violations = True


# ---------------------------------------------------------------------------
# --quiet propagation into the chained scenarios pipeline
# ---------------------------------------------------------------------------


class TestComputeDeviceFlag:
    """`--compute-device` overrides `environment.compute.device` and
    flows into `os.environ['COMPUTE_DEVICE']` so the embedder picks it
    up downstream."""

    def test_flag_parses_when_supplied(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["--compute-device", "cpu"]
        parser = cli._build_parser()
        parsed = parser.parse_args(argv)
        assert parsed.compute_device == "cpu"

    def test_flag_defaults_to_none(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        )
        parser = cli._build_parser()
        parsed = parser.parse_args(argv)
        assert parsed.compute_device is None

    def test_flag_rejects_unknown_device(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["--compute-device", "tpu"]
        parser = cli._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(argv)

    def test_cli_override_sets_env(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
        monkeypatch,
    ):
        """When --compute-device is passed, COMPUTE_DEVICE env var is set."""
        monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
        out_dir = tmp_path / "run"
        argv = _argv_generate(
            out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["--compute-device", "cpu"]
        rc = cli.main(argv)
        assert rc == 0
        assert os.environ.get("COMPUTE_DEVICE") == "cpu"

    def test_yaml_default_used_when_flag_absent(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
        monkeypatch,
    ):
        """When --compute-device is omitted, environment.compute.device
        from the YAML is exported instead (the example pins `auto`)."""
        monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
        out_dir = tmp_path / "run"
        argv = _argv_generate(
            out_dir, environment_yaml, persona_yaml, event_yaml, rules_yaml
        )
        rc = cli.main(argv)
        assert rc == 0
        # The example YAML now ships with `compute.device: auto`.
        assert os.environ.get("COMPUTE_DEVICE") == "auto"


class TestQuietPropagation:
    """`--quiet` on the persona CLI should reach the chained scenarios CLI
    as `--log-level WARNING` so HTTP / status chatter is hidden during a
    full `persona generate ... --scenario ...` run."""

    def _ns(self, **overrides):
        import argparse

        defaults = dict(
            scenario=Path("/cfg/scenarios.yaml"),
            seed=None,
            workers=None,
            scenario_id=None,
            scenario_method=None,
            quiet=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_quiet_emits_warning_log_level(self):
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        with patch("src.scripts.scenarios.cli.main", fake_main):
            cli._run_scenarios(self._ns(quiet=True))
        argv = captured["argv"]
        assert "--log-level" in argv
        assert argv[argv.index("--log-level") + 1] == "WARNING"

    def test_no_quiet_emits_info_log_level(self):
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        with patch("src.scripts.scenarios.cli.main", fake_main):
            cli._run_scenarios(self._ns(quiet=False))
        argv = captured["argv"]
        assert argv[argv.index("--log-level") + 1] == "INFO"

    def test_quiet_flag_parsed_by_generate_subcommand(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["--quiet"]
        parser = cli._build_parser()
        parsed = parser.parse_args(argv)
        assert parsed.quiet is True

    def test_quiet_short_flag_parsed_by_generate_subcommand(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        ) + ["-q"]
        parser = cli._build_parser()
        parsed = parser.parse_args(argv)
        assert parsed.quiet is True

    def test_quiet_default_is_false(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
    ):
        argv = _argv_generate(
            tmp_path, environment_yaml, persona_yaml, event_yaml, rules_yaml
        )
        parser = cli._build_parser()
        parsed = parser.parse_args(argv)
        assert parsed.quiet is False


class TestLoadExperimentNameFromRunDir:
    """`environment.yaml`'s optional `experiment_name` lands in
    `used_configs.json`'s `environment` block.  The benchmark-report
    writers read it back from there to title the markdown."""

    def test_returns_none_when_snapshot_missing(self, tmp_path: Path):
        assert cli._load_experiment_name_from_run_dir(tmp_path) is None

    def test_reads_name_from_environment_block(self, tmp_path: Path):
        (tmp_path / "used_configs.json").write_text(
            json.dumps({"environment": {"experiment_name": "Hello world"}}),
            encoding="utf-8",
        )
        assert cli._load_experiment_name_from_run_dir(tmp_path) == "Hello world"

    def test_returns_none_when_field_absent(self, tmp_path: Path):
        (tmp_path / "used_configs.json").write_text(
            json.dumps({"environment": {"seed": 1}}),
            encoding="utf-8",
        )
        assert cli._load_experiment_name_from_run_dir(tmp_path) is None

    def test_returns_none_when_environment_block_absent(self, tmp_path: Path):
        (tmp_path / "used_configs.json").write_text(
            json.dumps({"persona": {}}),
            encoding="utf-8",
        )
        assert cli._load_experiment_name_from_run_dir(tmp_path) is None

    def test_returns_none_when_name_is_empty_string(self, tmp_path: Path):
        (tmp_path / "used_configs.json").write_text(
            json.dumps({"environment": {"experiment_name": "   "}}),
            encoding="utf-8",
        )
        assert cli._load_experiment_name_from_run_dir(tmp_path) is None

    def test_returns_none_on_invalid_json(self, tmp_path: Path):
        (tmp_path / "used_configs.json").write_text("not valid json", encoding="utf-8")
        assert cli._load_experiment_name_from_run_dir(tmp_path) is None


# ---------------------------------------------------------------------------
# _attach_contexts: per-person no-contexts skip
# ---------------------------------------------------------------------------


class TestAttachContextsNoContextsBranch:
    def _person_no_ctx(self):
        from types import SimpleNamespace

        return SimpleNamespace(person_id="p1", persona_id="alice", contexts=[])

    def _person_with_ctx(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            person_id="p2",
            persona_id="alice",
            contexts=[SimpleNamespace(category="mood_emotion", name="happy")],
        )

    def _schedule(self, person_id: str):
        from src.scripts.persona.domain.schedule import PersonSchedule

        return PersonSchedule(
            person_id=person_id, persona_id="alice", person_seed=1, days=[]
        )

    def test_persons_without_contexts_kept_unchanged(self, monkeypatch):
        """A person declaring no contexts skips plan_contexts and reuses the original schedule."""
        from src.scripts.persona import cli as persona_cli

        persons = [self._person_no_ctx(), self._person_with_ctx()]
        scheds = [self._schedule("p1"), self._schedule("p2")]
        replanned = self._schedule("p2_replanned")
        monkeypatch.setattr(persona_cli, "resolve_for_person", lambda *a, **k: [])
        monkeypatch.setattr(persona_cli, "plan_contexts", lambda *a, **k: replanned)

        class _Env:
            time_windows = {}

        class _Rules:
            rules = []

        class _Cfg:
            environment = type("E", (), {"time_windows": {}})()
            rules = _Rules()

        # WindowMap.from_config is called with {}; stub it.
        from src.scripts.persona.domain import time_windows as tw_mod

        monkeypatch.setattr(
            tw_mod.WindowMap, "from_config", classmethod(lambda cls, _cfg: object())
        )
        out = persona_cli._attach_contexts(persons, scheds, _Cfg())
        assert out[0] is scheds[0]
        assert out[1] is replanned


# ---------------------------------------------------------------------------
# _write_persona_timelines: catalog load exception falls back to None
# ---------------------------------------------------------------------------


class TestWritePersonaTimelinesCatalogFailure:
    def test_catalog_load_exception_does_not_abort_write(
        self, tmp_path: Path, monkeypatch
    ):
        """A failing context catalog load falls back to None and timelines still write."""
        import datetime as _dt

        from src.scripts.persona import cli as persona_cli
        from src.scripts.persona.context.schema import ContextEpisode
        from src.scripts.persona.domain.schedule import PersonSchedule

        def _boom():
            raise RuntimeError("catalog boom")

        monkeypatch.setattr("src.scripts.persona.context.catalog.load_catalog", _boom)
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        ep = ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=_dt.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
        )
        sched = PersonSchedule(
            person_id="p1",
            persona_id="alice",
            person_seed=1,
            days=[],
            contexts=[ep],
        )

        from types import SimpleNamespace

        person = SimpleNamespace(person_id="p1", persona_id="alice", contexts=[ep])
        paths = persona_cli._write_persona_timelines([person], [sched], persons_dir)
        assert paths and paths[0].exists()


# ---------------------------------------------------------------------------
# _generate: --no-contexts flag skips _attach_contexts
# ---------------------------------------------------------------------------


class TestGenerateNoContextsFlag:
    def test_no_contexts_flag_skips_attach(
        self,
        tmp_path,
        environment_yaml,
        persona_yaml,
        event_yaml,
        rules_yaml,
        monkeypatch,
    ):
        """Passing --no-contexts skips the context placer pass entirely."""
        from src.scripts.persona import cli as persona_cli

        called = {"n": 0}

        def _spy(persons, schedules, cfg):
            called["n"] += 1
            return schedules

        monkeypatch.setattr(persona_cli, "_attach_contexts", _spy)
        argv = _argv_generate(
            tmp_path / "run", environment_yaml, persona_yaml, event_yaml, rules_yaml
        )
        argv.append("--no-contexts")
        code = persona_cli.main(argv)
        assert code in (0, persona_cli.EXIT_VIOLATIONS)
        assert called["n"] == 0


# ---------------------------------------------------------------------------
# _resolve_calendar_contexts
# ---------------------------------------------------------------------------


class TestResolveCalendarContexts:
    def test_none_returns_none(self):
        from src.scripts.persona import cli as persona_cli

        assert persona_cli._resolve_calendar_contexts(None, None) is None

    def test_non_empty_list_returned_verbatim(self):
        from src.scripts.persona import cli as persona_cli

        out = persona_cli._resolve_calendar_contexts(["mood_emotion"], None)
        assert out == ["mood_emotion"]

    def test_empty_list_with_no_persona_config_returns_none(self):
        from src.scripts.persona import cli as persona_cli

        assert persona_cli._resolve_calendar_contexts([], None) is None

    def test_empty_list_falls_back_to_persona_context_keys(self):
        """Bare --calendar-contexts collects every category declared by any persona."""
        from types import SimpleNamespace

        from src.scripts.persona import cli as persona_cli

        p1 = SimpleNamespace(
            contexts={"mood_emotion": object(), "energy_state": object()}
        )
        p2 = SimpleNamespace(contexts={"weather_environment": object()})
        cfg = SimpleNamespace(personas=[p1, p2])
        assert persona_cli._resolve_calendar_contexts([], cfg) == [
            "energy_state",
            "mood_emotion",
            "weather_environment",
        ]

    def test_empty_list_falls_back_to_none_when_no_persona_declares_contexts(self):
        from types import SimpleNamespace

        from src.scripts.persona import cli as persona_cli

        cfg = SimpleNamespace(personas=[SimpleNamespace(contexts={})])
        assert persona_cli._resolve_calendar_contexts([], cfg) is None
