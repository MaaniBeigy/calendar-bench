"""Multi-scenario dispatch, ablation walking, and report filtering in `scenarios.cli`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import src.scripts.scenarios.cli as cli_module
from src.scripts.scenarios.cli import (
    EXIT_OK,
    EXIT_USAGE,
    _augment_ablation_runs_with_disk,
    _cmd_evaluate_with_cfg,
    _cmd_report,
    _load_met_quartiles_or_default,
    _multi_augment,
    _multi_evaluate,
)
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    ExperimentScenariosConfig,
    ScenarioConfig,
    ScenarioOutputConfig,
)


def test_load_met_quartiles_or_default_returns_defaults_when_no_file(
    tmp_path, monkeypatch
):
    """With no quartile file on disk the helper returns the static default."""
    monkeypatch.chdir(tmp_path)
    result = _load_met_quartiles_or_default()
    assert result.q1 > 0
    assert result.q2 > result.q1


def _exp_with_llm_ablation(scenario_id: str = "s1") -> ExperimentScenariosConfig:
    """Single-scenario llm_agent config with a two-variant `custom` ablation."""
    return ExperimentScenariosConfig.model_validate(
        {
            "experiment_id": "exp_ablate",
            "scenarios": [
                {
                    "id": scenario_id,
                    "augmentation": [
                        {
                            "method": "llm_agent",
                            "llm_agent": {
                                "provider": "anthropic",
                                "model": "claude-sonnet-4-6",
                                "prompt_ablation": {
                                    "design": "custom",
                                    "variants": [
                                        {"id": "v1", "ablate": ["fit_examples"]},
                                        {"id": "v2", "ablate": ["self_verify"]},
                                    ],
                                },
                            },
                        }
                    ],
                }
            ],
        }
    )


def _multi_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scenario=tmp_path / "scenarios.yaml",
        out_dir=None,
        seed=None,
        workers=1,
        log_level="INFO",
        tasks_dir=None,
        run_dir=None,
    )


def test_multi_augment_propagates_failing_rc_from_variant_fanout(tmp_path):
    """A failing variant augment short-circuits the dispatch loop."""
    exp = _exp_with_llm_ablation()
    args = _multi_args(tmp_path)
    with patch.object(cli_module, "_cmd_augment_with_cfg", return_value=EXIT_USAGE):
        rc = _multi_augment(exp, args, scenario_id=None, method_filter=None)
    assert rc == EXIT_USAGE


def test_multi_augment_skips_human_coach_fixtures(tmp_path):
    """A sweep skips the fixtures-only human_coach method and continues."""
    exp = ExperimentScenariosConfig.model_validate(
        {
            "experiment_id": "exp_hc",
            "scenarios": [
                {"id": "human_coach", "augmentation": [{"method": "human_coach"}]},
                {"id": "s1", "augmentation": [{"method": "greedy"}]},
            ],
        }
    )
    args = _multi_args(tmp_path)
    args.no_ablation = False
    args.ablation_variant = None
    augmented: list[str] = []

    def fake_aug(cfg, sub_args):
        augmented.append(cfg.augmentation.method)
        return EXIT_OK

    with patch.object(cli_module, "_cmd_augment_with_cfg", side_effect=fake_aug):
        rc = _multi_augment(exp, args, scenario_id=None, method_filter=None)
    assert rc == EXIT_OK
    assert "human_coach" not in augmented
    assert augmented == ["greedy"]


def test_multi_evaluate_skips_variant_excluded_by_filter(tmp_path):
    """`ablation_variant='v1'` runs v1 and skips v2."""
    exp = _exp_with_llm_ablation()
    args = _multi_args(tmp_path)
    args.ablation_variant = "v1"
    args.no_ablation = False
    called: list[str] = []

    def fake_eval(cfg, sub_args):
        called.append(str(sub_args.run_dir))
        return EXIT_OK

    with patch.object(cli_module, "_cmd_evaluate_with_cfg", side_effect=fake_eval):
        rc = _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
    assert rc == EXIT_OK
    assert any("v1" in path for path in called)
    assert not any("v2" in path for path in called)


def test_multi_evaluate_propagates_failing_rc_from_variant_fanout(tmp_path):
    """A failing variant evaluate short-circuits the dispatch loop."""
    exp = _exp_with_llm_ablation()
    args = _multi_args(tmp_path)
    args.no_ablation = False
    args.ablation_variant = None
    with patch.object(cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_USAGE):
        rc = _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
    assert rc == EXIT_USAGE


def test_augment_ablation_runs_with_disk_skips_non_directory_and_seen_keys(tmp_path):
    """Non-directory entries and duplicate keys are both skipped on disk walk."""
    exp = ExperimentScenariosConfig.model_validate(
        {
            "experiment_id": "exp_disk",
            "scenarios": [
                {
                    "id": "s1",
                    "augmentation": [{"method": "greedy"}],
                }
            ],
        }
    )
    ablation_root = tmp_path / "scenarios" / "s1" / "greedy" / "ablation"
    ablation_root.mkdir(parents=True)
    (ablation_root / "stray.txt").write_text("noise", encoding="utf-8")
    variant_dir = ablation_root / "v1"
    variant_dir.mkdir()
    (variant_dir / "evaluation").mkdir()
    (variant_dir / "evaluation" / "total_scheduling_gain.json").write_text(
        "{}", encoding="utf-8"
    )
    seed_runs = [("s1", "greedy", "v1")]
    out = _augment_ablation_runs_with_disk(tmp_path, exp, seed_runs)
    assert out.count(("s1", "greedy", "v1")) == 1


def test_augment_ablation_runs_with_disk_returns_seed_runs_when_no_ablation_dir(
    tmp_path,
):
    """An absent ablation directory is silently skipped for that method."""
    exp = ExperimentScenariosConfig.model_validate(
        {
            "experiment_id": "exp_none",
            "scenarios": [{"id": "s1", "augmentation": [{"method": "greedy"}]}],
        }
    )
    out = _augment_ablation_runs_with_disk(tmp_path, exp, [])
    assert out == []


def test_cmd_evaluate_uses_explicit_tasks_dir(tmp_path):
    """`args.tasks_dir` is prepended to the grounding-candidate list."""
    run_dir = tmp_path / "run"
    persons_dir = run_dir / "augmented" / "persons"
    persons_dir.mkdir(parents=True)
    (persons_dir / "p001.json").write_text(
        json.dumps(
            {
                "person_id": "p001",
                "tasks_total": 0,
                "scheduled_count": 0,
                "unscheduled_count": 0,
                "scheduled": [],
                "unscheduled": [],
            }
        ),
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit_tasks"
    explicit.mkdir()
    (explicit / "p001_tasks.json").write_text(
        json.dumps([{"label": "yoga", "ontology_uri": "https://ex.org/t1"}]),
        encoding="utf-8",
    )

    cfg = ScenarioConfig(
        id="grounding_test",
        output=ScenarioOutputConfig(
            dir=str(run_dir), write_json=False, write_ics=False
        ),
        augmentation=AugmentationConfig(method="greedy"),
    )
    args = argparse.Namespace(
        scenario=tmp_path / "s.yaml",
        run_dir=run_dir,
        tasks_dir=explicit,
        seed=None,
        workers=1,
        log_level="INFO",
    )

    captured: dict = {}

    def fake_writer(
        results,
        out_dir,
        *,
        grounding=None,
        telemetry=None,
        preference_breakdown=None,
        divide_breakdown=None,
        context_fit_breakdown=None,
        window=None,
        weekly_gain=None,
    ):
        captured["grounding"] = grounding
        return out_dir / "report.md", out_dir / "report.json"

    with (
        patch.object(cli_module, "_build_judge_oracle", return_value=None),
        patch(
            "src.scripts.scenarios.export.report_writer.write_evaluation_reports",
            side_effect=fake_writer,
        ),
        patch(
            "src.scripts.scenarios.task_generation.ontology_bridge.uri_resolves",
            return_value=True,
        ),
    ):
        rc = _cmd_evaluate_with_cfg(cfg, args)
    assert rc == EXIT_OK
    assert captured["grounding"] is not None


def test_cmd_report_skips_methods_not_matching_filter(tmp_path, monkeypatch):
    """`--method greedy` walks past the `llm_agent` entry in the same scenario."""
    scenario_yaml = tmp_path / "scenarios.yaml"
    scenario_yaml.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "exp_filter",
                "run_dir": str(tmp_path / "run"),
                "scenarios": [
                    {
                        "id": "s1",
                        "augmentation": [
                            {"method": "greedy"},
                            {
                                "method": "llm_agent",
                                "llm_agent": {
                                    "provider": "anthropic",
                                    "model": "claude-sonnet-4-6",
                                },
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run").mkdir()

    captured: dict = {}

    def fake_write_report(
        experiment_dir,
        experiment_id,
        *,
        scenarios_cfg,
        experiment_name=None,
        scenario_method_pairs=None,
        baseline=None,
        ablation_runs=None,
    ):
        captured["pairs"] = scenario_method_pairs
        return experiment_dir / "report.md"

    monkeypatch.setattr(
        "src.scripts.scenarios.export.benchmark_report.write_benchmark_report",
        fake_write_report,
    )
    args = argparse.Namespace(
        scenario=scenario_yaml,
        scenario_id=None,
        method="greedy",
        baseline=None,
    )
    rc = _cmd_report(args)
    assert rc == EXIT_OK
    assert "pairs" in captured
