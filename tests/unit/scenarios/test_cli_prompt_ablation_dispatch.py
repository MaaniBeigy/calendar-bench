"""Tests for the prompt-ablation CLI dispatch loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.scripts.scenarios import cli
from src.scripts.scenarios.augmentation.prompts.ablation_designs import build_design
from src.scripts.scenarios.augmentation.prompts.augment_oneshot import ABLATABLE_BLOCKS
from src.scripts.scenarios.config.schema import (
    AblationVariantSpec,
    AugmentationMethodConfig,
    ExperimentScenariosConfig,
    LLMAgentConfig,
    PromptAblationConfig,
    ScenarioDefinition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp(
    *,
    design: str,
    fold: bool = True,
    placebos: dict[str, str] | None = None,
    use_default_placebos: bool = True,
    variants: list[AblationVariantSpec] | None = None,
    tmp_path: Path,
):
    """Build a synthetic single-scenario, single-method experiment."""
    ablation = PromptAblationConfig(
        design=design,
        fold=fold,
        placebos=placebos or {},
        use_default_placebos=use_default_placebos,
        variants=variants or [],
    )
    method = AugmentationMethodConfig(
        method="llm_agent",
        llm_agent=LLMAgentConfig(
            provider="openai",
            model="gpt-4o-mini",
            prompt_template="augment_oneshot",
            prompt_ablation=ablation,
        ),
    )
    scenario = ScenarioDefinition(id="scen", augmentation=[method])
    return ExperimentScenariosConfig(
        experiment_id="exp_test",
        run_dir=str(tmp_path),
        output_base=str(tmp_path.parent),
        scenarios=[scenario],
    )


def _capture(monkeypatch):
    """Replace augment + evaluate cmd with recorders returning EXIT_OK."""
    augment_calls: list[argparse.Namespace] = []
    evaluate_calls: list[argparse.Namespace] = []

    def fake_augment(cfg, sub_args):
        augment_calls.append(sub_args)
        return cli.EXIT_OK

    def fake_evaluate(cfg, sub_args):
        evaluate_calls.append(sub_args)
        return cli.EXIT_OK

    def fake_write_report(*_, **__):
        return None

    monkeypatch.setattr(cli, "_cmd_augment_with_cfg", fake_augment)
    monkeypatch.setattr(cli, "_cmd_evaluate_with_cfg", fake_evaluate)
    monkeypatch.setattr(cli, "_write_benchmark_report_if_possible", fake_write_report)
    return augment_calls, evaluate_calls


# ---------------------------------------------------------------------------
# Variant materialisation
# ---------------------------------------------------------------------------


class TestAblationVariantsForMethod:
    def test_single_returns_empty_list(self):
        method = AugmentationMethodConfig(
            method="llm_agent",
            llm_agent=LLMAgentConfig(
                prompt_ablation=PromptAblationConfig(design="single")
            ),
        )
        assert cli._ablation_variants_for_method(method) == []

    def test_absent_ablation_returns_empty_list(self):
        method = AugmentationMethodConfig(
            method="llm_agent",
            llm_agent=LLMAgentConfig(prompt_ablation=None),
        )
        assert cli._ablation_variants_for_method(method) == []

    def test_non_llm_method_returns_empty_list(self):
        method = AugmentationMethodConfig(
            method="greedy",
            llm_agent=LLMAgentConfig(
                prompt_ablation=PromptAblationConfig(design="leave_one_out")
            ),
        )
        assert cli._ablation_variants_for_method(method) == []

    def test_leave_one_out_returns_n_plus_one_variants(self):
        method = AugmentationMethodConfig(
            method="llm_agent",
            llm_agent=LLMAgentConfig(
                prompt_ablation=PromptAblationConfig(design="leave_one_out")
            ),
        )
        variants = cli._ablation_variants_for_method(method)
        assert len(variants) == len(ABLATABLE_BLOCKS) + 1

    def test_plackett_burman_12_matches_build_design(self):
        method = AugmentationMethodConfig(
            method="llm_agent",
            llm_agent=LLMAgentConfig(
                prompt_ablation=PromptAblationConfig(
                    design="plackett_burman_12", fold=False
                )
            ),
        )
        cli_variants = cli._ablation_variants_for_method(method)
        ref = build_design("plackett_burman_12", fold=False)
        assert [v.variant_id for v in cli_variants] == [v.variant_id for v in ref]

    def test_custom_passes_through_user_variants(self):
        method = AugmentationMethodConfig(
            method="llm_agent",
            llm_agent=LLMAgentConfig(
                prompt_ablation=PromptAblationConfig(
                    design="custom",
                    variants=[
                        AblationVariantSpec(id="baseline", ablate=[]),
                        AblationVariantSpec(id="no_rot1", ablate=["rule_of_thumb_1"]),
                    ],
                )
            ),
        )
        variants = cli._ablation_variants_for_method(method)
        assert [v.variant_id for v in variants] == ["baseline", "no_rot1"]


# ---------------------------------------------------------------------------
# Placebo resolution
# ---------------------------------------------------------------------------


class TestResolvePlacebos:
    def _method(self, **kwargs):
        return AugmentationMethodConfig(
            method="llm_agent",
            llm_agent=LLMAgentConfig(
                prompt_ablation=PromptAblationConfig(design="leave_one_out", **kwargs)
            ),
        )

    def test_returns_empty_dict_when_nothing_ablated(self):
        method = self._method()
        assert cli._resolve_placebos(method, frozenset()) == {}

    def test_uses_default_catalog_when_flag_true(self):
        from src.scripts.scenarios.augmentation.prompts.augment_oneshot_placebos import (
            DEFAULT_PLACEBOS,
        )

        method = self._method(use_default_placebos=True)
        out = cli._resolve_placebos(method, frozenset({"rule_of_thumb_1"}))
        assert out["rule_of_thumb_1"] == DEFAULT_PLACEBOS["rule_of_thumb_1"]

    def test_default_off_omits_unmentioned_blocks(self):
        """`use_default_placebos=False` drops unspecified blocks."""
        method = self._method(use_default_placebos=False)
        out = cli._resolve_placebos(method, frozenset({"rule_of_thumb_1"}))
        assert "rule_of_thumb_1" not in out

    def test_user_placebo_wins_over_default(self):
        method = self._method(
            use_default_placebos=True,
            placebos={"rule_of_thumb_1": "custom override"},
        )
        out = cli._resolve_placebos(method, frozenset({"rule_of_thumb_1"}))
        assert out["rule_of_thumb_1"] == "custom override"

    def test_missing_method_returns_empty(self):
        method = AugmentationMethodConfig(method="llm_agent")
        method = method.model_copy(update={"llm_agent": LLMAgentConfig()})
        assert cli._resolve_placebos(method, frozenset({"self_verify"})) == {}


# ---------------------------------------------------------------------------
# Variant directory layout
# ---------------------------------------------------------------------------


class TestVariantPaths:
    def test_variant_dir_layout(self, tmp_path):
        method_dir = tmp_path / "scenarios" / "scen" / "llm_agent"
        out = cli._ablation_variant_dir(method_dir, "10000000000")
        assert out == method_dir / "ablation" / "10000000000"

    def test_write_ablation_index_round_trips(self, tmp_path):
        variants = build_design("leave_one_out")
        cli._write_ablation_index(tmp_path, variants)
        payload = json.loads((tmp_path / "ablation" / "variants.json").read_text())
        ids_on_disk = [v["variant_id"] for v in payload["variants"]]
        assert ids_on_disk == [v.variant_id for v in variants]


# ---------------------------------------------------------------------------
# Variant filter
# ---------------------------------------------------------------------------


class TestVariantFilter:
    def test_none_returns_none(self):
        assert cli._variant_filter_set(None) is None

    def test_string_wrapped(self):
        assert cli._variant_filter_set("10000000000") == {"10000000000"}

    def test_list_pass_through(self):
        out = cli._variant_filter_set(["10000000000", "01000000000"])
        assert out == {"10000000000", "01000000000"}

    def test_empty_list_returns_none(self):
        assert cli._variant_filter_set([]) is None


# ---------------------------------------------------------------------------
# _multi_augment fan-out
# ---------------------------------------------------------------------------


class TestMultiAugmentFanOut:
    def test_no_ablation_returns_one_call(self, monkeypatch, tmp_path):
        exp = _make_exp(design="single", tmp_path=tmp_path)
        augment_calls, _ = _capture(monkeypatch)
        args = argparse.Namespace(scenario=Path("scen.yaml"))
        rc = cli._multi_augment(exp, args, None, None)
        assert rc == cli.EXIT_OK
        assert len(augment_calls) == 1
        assert augment_calls[0].prompt_ablate is None
        # `out_dir is None` causes the caller to fall back to cfg.output.dir.
        assert augment_calls[0].out_dir is None

    def test_leave_one_out_returns_n_plus_one_calls(self, monkeypatch, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        augment_calls, _ = _capture(monkeypatch)
        args = argparse.Namespace(scenario=Path("scen.yaml"))
        rc = cli._multi_augment(exp, args, None, None)
        assert rc == cli.EXIT_OK
        assert len(augment_calls) == len(ABLATABLE_BLOCKS) + 1

    def test_each_variant_call_has_distinct_output_dir(self, monkeypatch, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        augment_calls, _ = _capture(monkeypatch)
        cli._multi_augment(
            exp, argparse.Namespace(scenario=Path("scen.yaml")), None, None
        )
        out_dirs = [a.out_dir for a in augment_calls]
        assert len(set(out_dirs)) == len(out_dirs)
        for out_dir in out_dirs:
            assert "ablation" in str(out_dir)

    def test_each_variant_call_carries_resolved_placebos(self, monkeypatch, tmp_path):
        exp = _make_exp(
            design="leave_one_out",
            placebos={"rule_of_thumb_1": "user override"},
            tmp_path=tmp_path,
        )
        augment_calls, _ = _capture(monkeypatch)
        cli._multi_augment(
            exp, argparse.Namespace(scenario=Path("scen.yaml")), None, None
        )
        match = [
            a
            for a in augment_calls
            if a.prompt_ablate == frozenset({"rule_of_thumb_1"})
        ]
        assert len(match) == 1
        assert match[0].prompt_placebos["rule_of_thumb_1"] == "user override"

    def test_no_ablation_flag_falls_back_to_baseline(self, monkeypatch, tmp_path):
        exp = _make_exp(design="plackett_burman_24", tmp_path=tmp_path)
        augment_calls, _ = _capture(monkeypatch)
        args = argparse.Namespace(scenario=Path("scen.yaml"), no_ablation=True)
        cli._multi_augment(exp, args, None, None)
        assert len(augment_calls) == 1
        assert augment_calls[0].prompt_ablate is None

    def test_variant_filter_subset(self, monkeypatch, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        augment_calls, _ = _capture(monkeypatch)
        baseline_id = "0" * len(ABLATABLE_BLOCKS)
        args = argparse.Namespace(
            scenario=Path("scen.yaml"),
            ablation_variant=[baseline_id],
        )
        cli._multi_augment(exp, args, None, None)
        assert len(augment_calls) == 1
        assert augment_calls[0].prompt_ablate == frozenset()

    def test_writes_variants_index_to_disk(self, monkeypatch, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        _capture(monkeypatch)
        cli._multi_augment(
            exp, argparse.Namespace(scenario=Path("scen.yaml")), None, None
        )
        method_dir = Path(exp.scenario_method_dir("scen", "llm_agent"))
        assert (method_dir / "ablation" / "variants.json").is_file()


# ---------------------------------------------------------------------------
# _multi_evaluate fan-out
# ---------------------------------------------------------------------------


class TestMultiEvaluateFanOut:
    def test_no_ablation_returns_one_eval_call(self, monkeypatch, tmp_path):
        exp = _make_exp(design="single", tmp_path=tmp_path)
        _, evaluate_calls = _capture(monkeypatch)
        rc = cli._multi_evaluate(
            exp, argparse.Namespace(scenario=Path("scen.yaml")), None, None
        )
        assert rc == cli.EXIT_OK
        assert len(evaluate_calls) == 1
        assert evaluate_calls[0].run_dir is None

    def test_leave_one_out_returns_n_plus_one_eval_calls(self, monkeypatch, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        _, evaluate_calls = _capture(monkeypatch)
        cli._multi_evaluate(
            exp, argparse.Namespace(scenario=Path("scen.yaml")), None, None
        )
        assert len(evaluate_calls) == len(ABLATABLE_BLOCKS) + 1
        for call in evaluate_calls:
            assert call.run_dir is not None
            assert "ablation" in str(call.run_dir)

    def test_no_ablation_evaluate_flag(self, monkeypatch, tmp_path):
        exp = _make_exp(design="plackett_burman_24", tmp_path=tmp_path)
        _, evaluate_calls = _capture(monkeypatch)
        args = argparse.Namespace(scenario=Path("scen.yaml"), no_ablation=True)
        cli._multi_evaluate(exp, args, None, None)
        assert len(evaluate_calls) == 1
        assert evaluate_calls[0].run_dir is None


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


class TestAblationFlags:
    def test_parser_accepts_no_ablation_flag(self):
        parser = cli._build_parser()
        args = parser.parse_args(["augment", "--scenario", "x.yaml", "--no-ablation"])
        assert args.no_ablation is True

    def test_parser_accepts_ablation_variant_flag(self):
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "evaluate",
                "--scenario",
                "x.yaml",
                "--ablation-variant",
                "00000000000",
                "10000000000",
            ]
        )
        assert args.ablation_variant == ["00000000000", "10000000000"]

    def test_run_subcommand_also_carries_flags(self):
        parser = cli._build_parser()
        args = parser.parse_args(["run", "--scenario", "x.yaml", "--no-ablation"])
        assert args.no_ablation is True


# ---------------------------------------------------------------------------
# Disk-augmentation helpers used by `_cmd_report` and the auto-write path
# ---------------------------------------------------------------------------


class TestAugmentPairsWithDisk:
    def _seed_pair_with_eval(self, root: Path, sid: str, method: str) -> None:
        eval_dir = root / "scenarios" / sid / method / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "total_scheduling_gain.json").write_text("{}")

    def test_returns_only_pairs_with_top_level_eval_on_disk(self, tmp_path):
        exp = _make_exp(design="single", tmp_path=tmp_path)
        self._seed_pair_with_eval(tmp_path, "scen", "llm_agent")
        out = cli._augment_pairs_with_disk_runs(tmp_path, exp, [])
        assert out == [("scen", "llm_agent")]

    def test_omits_pairs_with_no_top_level_eval(self, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        # Only a per-variant evaluation exists, NOT the top-level one.
        variant_eval = (
            tmp_path
            / "scenarios"
            / "scen"
            / "llm_agent"
            / "ablation"
            / ("0" * 11)
            / "evaluation"
        )
        variant_eval.mkdir(parents=True)
        (variant_eval / "total_scheduling_gain.json").write_text("{}")
        assert cli._augment_pairs_with_disk_runs(tmp_path, exp, []) == []

    def test_dedupes_seed_pairs(self, tmp_path):
        exp = _make_exp(design="single", tmp_path=tmp_path)
        self._seed_pair_with_eval(tmp_path, "scen", "llm_agent")
        out = cli._augment_pairs_with_disk_runs(tmp_path, exp, [("scen", "llm_agent")])
        assert out == [("scen", "llm_agent")]


class TestAugmentAblationRunsWithDisk:
    def test_walks_disk_for_variants(self, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        for vid in ("00000000000", "10000000000"):
            eval_dir = (
                tmp_path
                / "scenarios"
                / "scen"
                / "llm_agent"
                / "ablation"
                / vid
                / "evaluation"
            )
            eval_dir.mkdir(parents=True)
            (eval_dir / "total_scheduling_gain.json").write_text("{}")
        out = cli._augment_ablation_runs_with_disk(tmp_path, exp, [])
        assert set(out) == {
            ("scen", "llm_agent", "00000000000"),
            ("scen", "llm_agent", "10000000000"),
        }

    def test_ignores_variant_dirs_with_no_eval(self, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        (tmp_path / "scenarios" / "scen" / "llm_agent" / "ablation" / ("0" * 11)).mkdir(
            parents=True
        )
        assert cli._augment_ablation_runs_with_disk(tmp_path, exp, []) == []

    def test_preserves_seed_runs_and_appends_new_ones(self, tmp_path):
        exp = _make_exp(design="leave_one_out", tmp_path=tmp_path)
        seed = [("scen", "llm_agent", "00000000000")]
        eval_dir = (
            tmp_path
            / "scenarios"
            / "scen"
            / "llm_agent"
            / "ablation"
            / "10000000000"
            / "evaluation"
        )
        eval_dir.mkdir(parents=True)
        (eval_dir / "total_scheduling_gain.json").write_text("{}")
        out = cli._augment_ablation_runs_with_disk(tmp_path, exp, seed)
        assert ("scen", "llm_agent", "00000000000") in out
        assert ("scen", "llm_agent", "10000000000") in out
        assert len(out) == 2
