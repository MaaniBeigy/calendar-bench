"""Tests for the `## Prompt component ablation` benchmark-report section."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.scripts.scenarios.augmentation.prompts.ablation_designs import build_design
from src.scripts.scenarios.augmentation.prompts.augment_oneshot import ABLATABLE_BLOCKS
from src.scripts.scenarios.config.schema import (
    AugmentationMethodConfig,
    ExperimentScenariosConfig,
    LLMAgentConfig,
    PromptAblationConfig,
    ScenarioDefinition,
)
from src.scripts.scenarios.export.benchmark_report import (
    build_benchmark_markdown,
    write_benchmark_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_pb24_tree(tmp_path: Path, scenario_id: str = "scen") -> Path:
    """Stage a PB-24 ablation tree on disk for the report tests."""
    method_dir = tmp_path / "scenarios" / scenario_id / "llm_agent"
    method_dir.mkdir(parents=True, exist_ok=True)
    variants = build_design("plackett_burman_24")
    (method_dir / "ablation").mkdir(parents=True, exist_ok=True)
    (method_dir / "ablation" / "variants.json").write_text(
        json.dumps(
            {
                "variants": [
                    {"variant_id": v.variant_id, "ablated": sorted(v.ablated)}
                    for v in variants
                ]
            }
        )
    )
    rng = np.random.default_rng(123)
    for v in variants:
        x = [-1 if b in v.ablated else +1 for b in ABLATABLE_BLOCKS]
        y = 0.5 + 0.1 * x[0] + rng.normal(0, 0.001)
        eval_dir = method_dir / "ablation" / v.variant_id / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "total_scheduling_gain.json").write_text(
            json.dumps(
                {
                    "average_total_gain": y,
                    "average_gains": {"recommended_task_coverage": y},
                }
            )
        )
        (eval_dir / "telemetry.json").write_text(
            json.dumps(
                {
                    "by_stage": {
                        "augmentation": {
                            "tokens_total": 1000.0 + 100.0 * x[0],
                            "wall_time_seconds_total": 10.0,
                            "estimated_usd_total": 0.01,
                        }
                    }
                }
            )
        )
    return method_dir


def _ablation_runs_for(scenario_id: str = "scen") -> list[tuple[str, str, str]]:
    return [
        (scenario_id, "llm_agent", v.variant_id)
        for v in build_design("plackett_burman_24")
    ]


def _scenarios_cfg(design: str = "plackett_burman_24") -> ExperimentScenariosConfig:
    return ExperimentScenariosConfig(
        experiment_id="exp_test",
        scenarios=[
            ScenarioDefinition(
                id="scen",
                augmentation=[
                    AugmentationMethodConfig(
                        method="llm_agent",
                        llm_agent=LLMAgentConfig(
                            provider="openai",
                            model="gpt-4o-mini",
                            prompt_template="augment_oneshot",
                            prompt_ablation=PromptAblationConfig(design=design),
                        ),
                    )
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Header presence / shape
# ---------------------------------------------------------------------------


class TestSectionHeader:
    def test_section_absent_when_no_ablation_runs(self, tmp_path):
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            experiment_dir=tmp_path,
            ablation_runs=None,
        )
        assert "## Prompt component ablation" not in md

    def test_section_absent_when_ablation_runs_empty_list(self, tmp_path):
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            experiment_dir=tmp_path,
            ablation_runs=[],
        )
        assert "## Prompt component ablation" not in md

    def test_section_absent_when_disk_has_no_variants(self, tmp_path):
        """No on-disk tree means no section."""
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            experiment_dir=tmp_path,
            ablation_runs=[("scen", "llm_agent", "0" * 11)],
        )
        assert "## Prompt component ablation" not in md

    def test_section_present_with_full_tree(self, tmp_path):
        _build_pb24_tree(tmp_path)
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=_scenarios_cfg(),
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        assert "## Prompt component ablation" in md


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------


class TestSubSections:
    def test_design_summary_subsection_present(self, tmp_path):
        _build_pb24_tree(tmp_path)
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=_scenarios_cfg(),
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        assert "### Design summary" in md
        assert "plackett_burman_24" in md

    def test_main_effects_subsection_present(self, tmp_path):
        _build_pb24_tree(tmp_path)
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=_scenarios_cfg(),
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        assert "### Per-block main effects" in md
        # Every block name must appear in the main-effects table.
        for block in ABLATABLE_BLOCKS:
            assert f"`{block}`" in md

    def test_cost_effects_subsection_present(self, tmp_path):
        _build_pb24_tree(tmp_path)
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=_scenarios_cfg(),
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        assert "### Per-block cost effects" in md
        assert "beta(augment tokens)" in md

    def test_variant_detail_collapsed(self, tmp_path):
        _build_pb24_tree(tmp_path)
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=_scenarios_cfg(),
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        assert "<details>" in md
        assert "</details>" in md


# ---------------------------------------------------------------------------
# Numeric content
# ---------------------------------------------------------------------------


class TestNumericContent:
    def test_planted_effect_visible_in_table(self, tmp_path):
        """Block 0's planted +0.1 effect must surface in the table."""
        _build_pb24_tree(tmp_path)
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=_scenarios_cfg(),
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        # Find the line whose first cell is the first ablatable block.
        target_block = ABLATABLE_BLOCKS[0]
        block_lines = [
            line for line in md.splitlines() if line.startswith(f"| `{target_block}`")
        ]
        assert block_lines, "expected a per-block row for the planted block"
        # At least one of those rows must mention a `+0.1` ± something pattern.
        joined = " ".join(block_lines)
        assert "+0.10" in joined or "+0.1 " in joined or "+0.09" in joined

    def test_unknown_block_default_value_for_scenario_without_match(self, tmp_path):
        _build_pb24_tree(tmp_path)
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=None,
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        # No scenarios_cfg means the design label resolves to "unknown".
        assert "unknown" in md


# ---------------------------------------------------------------------------
# write_benchmark_report end-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_writer_includes_ablation_section(self, tmp_path):
        _build_pb24_tree(tmp_path)
        target = write_benchmark_report(
            tmp_path,
            "exp_test",
            scenarios_cfg=_scenarios_cfg(),
            ablation_runs=_ablation_runs_for(),
        )
        md = target.read_text(encoding="utf-8")
        assert "## Prompt component ablation" in md
        assert "### Design summary" in md

    def test_writer_omits_section_when_ablation_runs_empty(self, tmp_path):
        target = write_benchmark_report(tmp_path, "exp_test")
        md = target.read_text(encoding="utf-8")
        assert "## Prompt component ablation" not in md


# ---------------------------------------------------------------------------
# Defensive branches
# ---------------------------------------------------------------------------


class TestDefensiveBranches:
    def test_section_skipped_when_experiment_dir_is_none(self):
        from src.scripts.scenarios.export.benchmark_report import (
            _format_prompt_ablation,
        )

        out = _format_prompt_ablation(
            experiment_dir=None,
            ablation_runs=[("scen", "llm_agent", "0" * 11)],
        )
        assert out == []

    def test_design_label_skips_non_matching_scenarios(self, tmp_path):
        _build_pb24_tree(tmp_path)
        cfg = ExperimentScenariosConfig(
            experiment_id="exp_test",
            scenarios=[
                ScenarioDefinition(
                    id="other_scen",
                    augmentation=[
                        AugmentationMethodConfig(
                            method="llm_agent",
                            llm_agent=LLMAgentConfig(
                                provider="openai",
                                model="gpt-4o-mini",
                                prompt_template="augment_oneshot",
                            ),
                        )
                    ],
                ),
                ScenarioDefinition(
                    id="scen",
                    augmentation=[
                        AugmentationMethodConfig(method="greedy"),
                        AugmentationMethodConfig(
                            method="llm_agent",
                            llm_agent=LLMAgentConfig(
                                provider="openai",
                                model="gpt-4o-mini",
                                prompt_template="augment_oneshot",
                                prompt_ablation=PromptAblationConfig(
                                    design="plackett_burman_24"
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        )
        md = build_benchmark_markdown(
            experiment_id="exp_test",
            runs=[],
            scenarios_cfg=cfg,
            experiment_dir=tmp_path,
            ablation_runs=_ablation_runs_for(),
        )
        assert "plackett_burman_24" in md

    def test_design_label_falls_back_to_unknown_for_missing_pair(self, tmp_path):
        from src.scripts.scenarios.export.benchmark_report import _design_label_for

        cfg = ExperimentScenariosConfig(
            experiment_id="exp_test",
            scenarios=[
                ScenarioDefinition(
                    id="other_scen",
                    augmentation=[AugmentationMethodConfig(method="greedy")],
                ),
            ],
        )
        assert _design_label_for(cfg, "missing_scen", "llm_agent") == "unknown"

    def test_design_label_falls_back_when_scenario_matches_but_method_does_not(self):
        from src.scripts.scenarios.export.benchmark_report import _design_label_for

        cfg = ExperimentScenariosConfig(
            experiment_id="exp_test",
            scenarios=[
                ScenarioDefinition(
                    id="scen",
                    augmentation=[AugmentationMethodConfig(method="greedy")],
                ),
            ],
        )
        # Scenario id matches but method `rl` does not, so the inner
        # loop exits without returning and we hit the outer fallback.
        assert _design_label_for(cfg, "scen", "rl") == "unknown"

    def test_main_effects_handles_missing_fit_and_zero_effect(self, tmp_path):
        from src.scripts.scenarios.export.ablation_analysis import (
            AblationFit,
            AblationResult,
            BlockEffect,
            VariantRecord,
        )
        from src.scripts.scenarios.export.benchmark_report import (
            _format_ablation_cost_effects,
            _format_ablation_main_effects,
        )

        record = VariantRecord(
            variant_id="0" * 11,
            ablated=frozenset(),
            responses={"total": 0.5},
            augment_tokens=None,
            augment_wall_seconds=None,
            augment_usd=None,
        )
        # Build a fit whose first effect has beta=0, se=0 (zero-effect
        # row) and lacks any effects for the rest of the blocks (so
        # the `next` fallback returns None).
        fit = AblationFit(
            response="total",
            intercept=0.5,
            effects=[BlockEffect(block=ABLATABLE_BLOCKS[0], beta=0.0, se=0.0)],
            n_runs=1,
            residual_std=0.0,
        )
        result = AblationResult(
            scenario_id="scen",
            method="llm_agent",
            design_label="single",
            records=[record],
            fits={"total": fit},
        )
        main_lines = _format_ablation_main_effects([result], scenarios_cfg=None)
        assert any("n/a" in line for line in main_lines)
        cost_lines = _format_ablation_cost_effects([result], scenarios_cfg=None)
        assert any("n/a" in line for line in cost_lines)

        # Now build a fit that targets a cost response with a zero-effect
        # row so the `effect.se == 0 and effect.beta == 0` branch fires
        # inside `_format_ablation_cost_effects`.
        cost_fit = AblationFit(
            response="augment_tokens",
            intercept=0.0,
            effects=[BlockEffect(block=ABLATABLE_BLOCKS[0], beta=0.0, se=0.0)],
            n_runs=1,
            residual_std=0.0,
        )
        cost_result = AblationResult(
            scenario_id="scen",
            method="llm_agent",
            design_label="single",
            records=[record],
            fits={"augment_tokens": cost_fit},
        )
        cost_lines2 = _format_ablation_cost_effects([cost_result], scenarios_cfg=None)
        assert any("n/a" in line for line in cost_lines2)
