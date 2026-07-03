"""Unit tests for src.scripts.scenarios.config (schema + loader).

Coverage targets:
  - All ScenarioConfig sub-models and their field defaults.
  - LossWeights: sum-to-1 validator passes and fails.
  - load_scenario: valid file, missing file, empty file, bad YAML,
    schema violation, missing 'scenario:' key.
  - Example YAML files round-trip correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.scripts.scenarios.config.loader import (
    ScenarioConfigError,
    load_config,
    load_experiment_scenarios,
    load_scenario,
)
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    AugmentationMethodConfig,
    CalendarSourceConfig,
    EvaluationConfig,
    ExperimentScenariosConfig,
    GreedyConfig,
    LLMAgentConfig,
    LLMModelConfig,
    LossWeights,
    RLConfig,
    ScenarioConfig,
    ScenarioDefinition,
    ScenarioOutputConfig,
    TaskFilterGroup,
    TaskFilters,
    TaskGenerationConfig,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EXAMPLES = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "scripts"
    / "scenarios"
    / "config"
    / "examples"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_scenario_dict(**overrides: object) -> dict:
    """Return the minimal valid inner-scenario dict (under 'scenario:' key)."""
    base: dict = {"id": "test_scenario", "output": {"dir": "./output/test"}}
    base.update(overrides)
    return base


def _make_config(**overrides: object) -> ScenarioConfig:
    return ScenarioConfig.model_validate(_minimal_scenario_dict(**overrides))


# ---------------------------------------------------------------------------
# ScenarioConfig top-level
# ---------------------------------------------------------------------------


class TestScenarioConfig:
    def test_minimal_valid_config(self):
        cfg = _make_config()
        assert cfg.id == "test_scenario"
        assert cfg.description == ""
        assert cfg.output.dir == "./output/test"

    def test_id_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            ScenarioConfig.model_validate(_minimal_scenario_dict(id=""))

    def test_description_has_default(self):
        cfg = _make_config()
        assert cfg.description == ""

    def test_description_set(self):
        cfg = _make_config(description="My scenario")
        assert cfg.description == "My scenario"

    def test_weekly_task_generation_wraps_single_block(self):
        cfg = _make_config()
        weekly = cfg.weekly_task_generation()
        assert len(weekly) == 1
        assert weekly[0].num_tasks == cfg.task_generation.num_tasks

    def test_weekly_task_generation_passes_through_list(self):
        cfg = _make_config(
            task_generation=[
                {"week": 1, "num_tasks": 10},
                {"week": 2, "num_tasks": 14},
            ]
        )
        assert [c.num_tasks for c in cfg.weekly_task_generation()] == [10, 14]

    def test_output_is_required(self):
        with pytest.raises(ValidationError):
            ScenarioConfig.model_validate({"id": "x"})

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ScenarioConfig.model_validate(_minimal_scenario_dict(unknown_field="bad"))


# ---------------------------------------------------------------------------
# CalendarSourceConfig
# ---------------------------------------------------------------------------


class TestCalendarSourceConfig:
    def test_all_fields_optional(self):
        cfg = CalendarSourceConfig()
        assert cfg.run_dir is None
        assert cfg.persona_environment is None
        assert cfg.persona_config is None
        assert cfg.persona_events is None
        assert cfg.persona_rules is None

    def test_run_dir_set(self):
        cfg = CalendarSourceConfig(run_dir="./output/example_experiment")
        assert cfg.run_dir == "./output/example_experiment"

    def test_all_persona_paths_set(self):
        cfg = CalendarSourceConfig(
            run_dir="./out",
            persona_environment="env.yaml",
            persona_config="persona.yaml",
            persona_events="events.yaml",
            persona_rules="rules.yaml",
        )
        assert cfg.persona_events == "events.yaml"


# ---------------------------------------------------------------------------
# TaskFilters
# ---------------------------------------------------------------------------


class TestTaskFilters:
    def test_defaults_are_empty(self):
        f = TaskFilters()
        assert f.domains == []
        assert f.difficulty == []

    def test_fields_set(self):
        f = TaskFilters(domains=["PhysicalActivityTask"], difficulty=["Level2"])
        assert "PhysicalActivityTask" in f.domains
        assert "Level2" in f.difficulty


# ---------------------------------------------------------------------------
# TaskGenerationConfig
# ---------------------------------------------------------------------------


class TestTaskGenerationConfig:
    def test_defaults(self):
        cfg = TaskGenerationConfig()
        assert cfg.method == "graphrag"
        assert cfg.num_tasks == 5
        assert "HealthTasks" in cfg.ontologies
        assert cfg.prompt_template == "health_improvement"
        assert cfg.manual_tasks == []
        assert cfg.task_overrides == {}

    def test_method_manual(self):
        cfg = TaskGenerationConfig(method="manual")
        assert cfg.method == "manual"

    def test_method_template(self):
        cfg = TaskGenerationConfig(method="template")
        assert cfg.method == "template"

    def test_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            TaskGenerationConfig(method="unknown")

    def test_num_tasks_ge_one(self):
        with pytest.raises(ValidationError):
            TaskGenerationConfig(num_tasks=0)

    def test_seed_ge_zero(self):
        with pytest.raises(ValidationError):
            TaskGenerationConfig(seed=-1)

    def test_manual_tasks_list_of_dicts(self):
        cfg = TaskGenerationConfig(
            method="manual",
            manual_tasks=[{"label": "walking", "duration_min": 20}],
        )
        assert len(cfg.manual_tasks) == 1
        assert cfg.manual_tasks[0]["label"] == "walking"

    def test_filters_nested(self):
        cfg = TaskGenerationConfig(
            filters={"domains": ["NutritionTask"], "difficulty": ["Level1"]}
        )
        assert cfg.filters.domains == ["NutritionTask"]

    def test_week_defaults_none_and_rejects_zero(self):
        assert TaskGenerationConfig().week is None
        assert TaskGenerationConfig(week=3).week == 3
        with pytest.raises(ValidationError):
            TaskGenerationConfig(week=0)

    def test_cross_week_distinct_defaults_true(self):
        assert TaskGenerationConfig().cross_week_distinct is True
        assert (
            TaskGenerationConfig(cross_week_distinct=False).cross_week_distinct is False
        )

    def test_legacy_filters_normalise_to_one_group(self):
        cfg = TaskGenerationConfig(
            filters={"domains": ["NutritionTask"], "difficulty": ["Level1"]}
        )
        groups = cfg.filter_groups()
        assert len(groups) == 1
        assert groups[0].domains == ["NutritionTask"]
        assert groups[0].difficulty == ["Level1"]

    def test_grouped_filters_parse_and_pass_through(self):
        cfg = TaskGenerationConfig(
            filters=[
                {
                    "domains": ["NutritionTask", "MentalWellbeingTask"],
                    "difficulty": ["Level2", "Level3"],
                },
                {"domains": ["PhysicalActivityTask"], "difficulty": ["Level1"]},
            ]
        )
        groups = cfg.filter_groups()
        assert isinstance(cfg.filters, list)
        assert [g.domains for g in groups] == [
            ["NutritionTask", "MentalWellbeingTask"],
            ["PhysicalActivityTask"],
        ]
        assert isinstance(groups[0], TaskFilterGroup)

    def test_profile_characteristics_defaults_to_none(self):
        assert TaskGenerationConfig().profile_characteristics is None

    def test_profile_characteristics_accepts_axis_list(self):
        cfg = TaskGenerationConfig(
            profile_characteristics=["occupation_status", "age", "has_kids"]
        )
        assert cfg.profile_characteristics == ["occupation_status", "age", "has_kids"]


# ---------------------------------------------------------------------------
# GreedyConfig
# ---------------------------------------------------------------------------


class TestGreedyConfig:
    def test_defaults(self):
        cfg = GreedyConfig()
        assert cfg.strategy == "preference_first"
        assert cfg.retry_on_miss is True
        assert cfg.max_backtrack == 3

    def test_strategy_earliest_fit(self):
        cfg = GreedyConfig(strategy="earliest_fit")
        assert cfg.strategy == "earliest_fit"

    def test_strategy_latest_fit(self):
        cfg = GreedyConfig(strategy="latest_fit")
        assert cfg.strategy == "latest_fit"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValidationError):
            GreedyConfig(strategy="random_fit")

    def test_max_backtrack_ge_zero(self):
        with pytest.raises(ValidationError):
            GreedyConfig(max_backtrack=-1)


# ---------------------------------------------------------------------------
# LLMAgentConfig
# ---------------------------------------------------------------------------


class TestLLMAgentConfig:
    def test_defaults(self):
        cfg = LLMAgentConfig()
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-sonnet-4-6"
        # Lowered from 3 to 1 on 2026-05-14 to cut LLM-augment latency
        # on malformed-JSON paths; bad responses are rare with
        # gpt-4o-mini and the per-week serial design means a single
        # retry already triples the per-week wall time.
        assert cfg.max_retries == 1
        assert cfg.prompt_template == "augment_oneshot"

    def test_provider_openai(self):
        cfg = LLMAgentConfig(provider="openai")
        assert cfg.provider == "openai"

    def test_provider_openrouter(self):
        cfg = LLMAgentConfig(provider="openrouter")
        assert cfg.provider == "openrouter"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            LLMAgentConfig(provider="ollama")

    def test_max_retries_ge_one(self):
        with pytest.raises(ValidationError):
            LLMAgentConfig(max_retries=0)


# ---------------------------------------------------------------------------
# RLConfig
# ---------------------------------------------------------------------------


class TestRLConfig:
    def test_defaults(self):
        cfg = RLConfig()
        assert cfg.policy == "dqn"
        assert cfg.checkpoint is None
        assert cfg.train_steps == 50000
        assert cfg.eval_episodes == 100
        assert cfg.train_steps_per_week == 4000
        assert cfg.gamma == 0.95
        assert cfg.buffer_size == 50000
        assert cfg.batch_size == 64
        assert cfg.epsilon_start == 1.0
        assert cfg.epsilon_end == 0.05
        assert cfg.epsilon_decay_weeks == 4
        assert cfg.bucket_minutes == 15
        assert cfg.device == "auto"
        assert cfg.per_step_shaping == 0.0
        assert cfg.parallel_per_worker_gb == 1.5
        assert cfg.parallel_max_retries == 1
        assert cfg.parallel_worker_threads == 1

    def test_policy_dqn(self):
        cfg = RLConfig(policy="dqn")
        assert cfg.policy == "dqn"

    def test_policy_random(self):
        cfg = RLConfig(policy="random")
        assert cfg.policy == "random"

    def test_invalid_policy_rejected(self):
        with pytest.raises(ValidationError):
            RLConfig(policy="q_learning")

    def test_checkpoint_set(self):
        cfg = RLConfig(checkpoint="./checkpoints/ppo_final.zip")
        assert cfg.checkpoint == "./checkpoints/ppo_final.zip"

    def test_train_steps_ge_one(self):
        with pytest.raises(ValidationError):
            RLConfig(train_steps=0)

    def test_parallel_knobs_set(self):
        cfg = RLConfig(
            parallel_per_worker_gb=2.0,
            parallel_max_retries=3,
            parallel_worker_threads=2,
        )
        assert cfg.parallel_per_worker_gb == 2.0
        assert cfg.parallel_max_retries == 3
        assert cfg.parallel_worker_threads == 2

    def test_parallel_per_worker_gb_must_be_positive(self):
        with pytest.raises(ValidationError):
            RLConfig(parallel_per_worker_gb=0.0)

    def test_parallel_max_retries_non_negative(self):
        with pytest.raises(ValidationError):
            RLConfig(parallel_max_retries=-1)

    def test_parallel_worker_threads_ge_one(self):
        with pytest.raises(ValidationError):
            RLConfig(parallel_worker_threads=0)


# ---------------------------------------------------------------------------
# AugmentationConfig
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    def test_defaults(self):
        cfg = AugmentationConfig()
        assert cfg.method == "greedy"
        assert cfg.allow_merge is True
        assert cfg.merge_threshold == pytest.approx(0.70)
        assert cfg.repeat_per_week is True

    def test_repeat_per_week_false(self):
        cfg = AugmentationConfig(repeat_per_week=False)
        assert cfg.repeat_per_week is False

    def test_method_llm_agent(self):
        cfg = AugmentationConfig(method="llm_agent")
        assert cfg.method == "llm_agent"

    def test_method_rl(self):
        cfg = AugmentationConfig(method="rl")
        assert cfg.method == "rl"

    def test_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            AugmentationConfig(method="solver")

    def test_merge_threshold_bounds(self):
        with pytest.raises(ValidationError):
            AugmentationConfig(merge_threshold=1.5)

    def test_nested_greedy_config(self):
        cfg = AugmentationConfig(greedy={"strategy": "earliest_fit"})
        assert cfg.greedy.strategy == "earliest_fit"

    def test_nested_llm_agent_config(self):
        cfg = AugmentationConfig(llm_agent={"provider": "openai", "model": "gpt-4o"})
        assert cfg.llm_agent.provider == "openai"

    def test_nested_rl_config(self):
        cfg = AugmentationConfig(rl={"policy": "dqn"})
        assert cfg.rl.policy == "dqn"


# ---------------------------------------------------------------------------
# LossWeights
# ---------------------------------------------------------------------------


class TestLossWeights:
    def test_defaults_sum_to_one(self):
        lw = LossWeights()
        total = (
            lw.lambda_cov
            + lw.lambda_cal
            + lw.lambda_pref
            + lw.lambda_disp
            + lw.lambda_merge
            + lw.lambda_spread
            + lw.lambda_divide
        )
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_sum_to_one(self):
        lw = LossWeights(
            lambda_cov=0.40,
            lambda_cal=0.20,
            lambda_pref=0.10,
            lambda_disp=0.10,
            lambda_merge=0.10,
            lambda_spread=0.05,
            lambda_divide=0.05,
        )
        assert lw.lambda_cov == pytest.approx(0.40)

    def test_weights_not_summing_to_one_rejected(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            LossWeights(
                lambda_cov=0.50,
                lambda_cal=0.50,
                lambda_pref=0.10,
                lambda_disp=0.10,
                lambda_merge=0.10,
                lambda_spread=0.10,
                lambda_divide=0.10,
            )

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            LossWeights(lambda_cov=-0.10)

    def test_weight_above_one_rejected(self):
        with pytest.raises(ValidationError):
            LossWeights(lambda_cov=1.10)

    def test_admissible_relation_overrides_default_empty(self):
        lw = LossWeights()
        assert lw.admissible_relation_overrides == {}

    def test_semantic_matrix_default_empty(self):
        lw = LossWeights()
        assert lw.semantic_matrix == {}

    def test_admissible_relation_overrides_set(self):
        lw = LossWeights(
            admissible_relation_overrides={"running::sleep": ["p", "m", "M", "P"]}
        )
        assert "running::sleep" in lw.admissible_relation_overrides

    def test_semantic_matrix_set(self):
        lw = LossWeights(semantic_matrix={"walking::podcast": 0.75})
        assert lw.semantic_matrix["walking::podcast"] == pytest.approx(0.75)

    def test_within_epsilon_passes(self):
        """Values differing by < 1e-6 from 1.0 must pass the validator."""
        lw = LossWeights(
            lambda_cov=0.180000000001,
            lambda_cal=0.18,
            lambda_pref=0.13,
            lambda_disp=0.18,
            lambda_merge=0.13,
            lambda_spread=0.10,
            lambda_divide=0.099999999999,
        )
        assert lw.lambda_cov == pytest.approx(0.180000000001)


# ---------------------------------------------------------------------------
# ScenarioOutputConfig
# ---------------------------------------------------------------------------


class TestScenarioOutputConfig:
    def test_dir_required(self):
        with pytest.raises(ValidationError):
            ScenarioOutputConfig.model_validate({})

    def test_defaults(self):
        cfg = ScenarioOutputConfig(dir="./out")
        assert cfg.write_ics is True
        assert cfg.write_json is True
        assert cfg.write_report is True

    def test_write_flags_overridable(self):
        cfg = ScenarioOutputConfig(dir="./out", write_ics=False, write_report=False)
        assert cfg.write_ics is False
        assert cfg.write_report is False


# ---------------------------------------------------------------------------
# load_scenario
# ---------------------------------------------------------------------------


class TestLoadScenario:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "scenario.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def _valid_yaml(self) -> str:
        return (
            "scenario:\n"
            "  id: my_scenario\n"
            "  output:\n"
            "    dir: ./output/my_scenario\n"
        )

    def test_valid_yaml_loads(self, tmp_path):
        path = self._write_yaml(tmp_path, self._valid_yaml())
        cfg = load_scenario(path)
        assert cfg.id == "my_scenario"
        assert cfg.output.dir == "./output/my_scenario"

    def test_all_defaults_populated(self, tmp_path):
        path = self._write_yaml(tmp_path, self._valid_yaml())
        cfg = load_scenario(path)
        assert cfg.description == ""
        assert cfg.augmentation.method == "greedy"
        assert cfg.task_generation.method == "graphrag"
        assert cfg.loss.lambda_cov == pytest.approx(0.18)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ScenarioConfigError, match="not found"):
            load_scenario(tmp_path / "does_not_exist.yaml")

    def test_empty_yaml_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, "")
        with pytest.raises(ScenarioConfigError, match="empty"):
            load_scenario(path)

    def test_bad_yaml_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, "key: val\n  : bad:\n")
        with pytest.raises(ScenarioConfigError, match="YAML parse error"):
            load_scenario(path)

    def test_missing_scenario_key_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, "id: x\noutput:\n  dir: ./out\n")
        with pytest.raises(ScenarioConfigError, match="top-level 'scenario:'"):
            load_scenario(path)

    def test_schema_violation_raises(self, tmp_path):
        path = self._write_yaml(
            tmp_path, "scenario:\n  id: ''\n  output:\n    dir: ./out\n"
        )
        with pytest.raises(ScenarioConfigError, match="validation error"):
            load_scenario(path)

    def test_unreadable_file_raises(self, tmp_path):
        path = tmp_path / "dir_not_file"
        path.mkdir()
        with pytest.raises(ScenarioConfigError, match="could not read"):
            load_scenario(path)

    def test_non_dict_scenario_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, "scenario: [a, b, c]\n")
        with pytest.raises(ScenarioConfigError, match="validation error"):
            load_scenario(path)


# ---------------------------------------------------------------------------
# Example YAML files round-trip
# ---------------------------------------------------------------------------


class TestExampleYAMLs:
    def test_health_boost_loads(self):
        cfg = load_scenario(_EXAMPLES / "scenario_health_boost.yaml")
        assert cfg.id == "health_boost_a"
        assert cfg.augmentation.method == "greedy"
        assert cfg.loss.lambda_cov == pytest.approx(0.18)
        assert "PhysicalActivityTask" in cfg.task_generation.filters.domains

    def test_senior_nutrition_loads(self):
        cfg = load_scenario(_EXAMPLES / "scenario_senior_nutrition.yaml")
        assert cfg.id == "senior_nutrition_b"
        assert cfg.augmentation.method == "llm_agent"
        assert cfg.loss.lambda_cov == pytest.approx(0.20)
        assert "NutritionTask" in cfg.task_generation.filters.domains

    def test_both_examples_loss_weights_sum_to_one(self):
        for name in ("scenario_health_boost.yaml", "scenario_senior_nutrition.yaml"):
            cfg = load_scenario(_EXAMPLES / name)
            total = (
                cfg.loss.lambda_cov
                + cfg.loss.lambda_cal
                + cfg.loss.lambda_pref
                + cfg.loss.lambda_disp
                + cfg.loss.lambda_merge
                + cfg.loss.lambda_spread
                + cfg.loss.lambda_divide
            )
            assert abs(total - 1.0) < 1e-6, f"{name}: weights sum to {total}"


# ---------------------------------------------------------------------------
# AugmentationMethodConfig
# ---------------------------------------------------------------------------


class TestAugmentationMethodConfig:
    def test_defaults(self):
        cfg = AugmentationMethodConfig()
        assert cfg.method == "greedy"
        assert cfg.allow_merge is True
        assert cfg.merge_threshold == pytest.approx(0.70)
        assert cfg.repeat_per_week is True
        assert cfg.output is None

    def test_repeat_per_week_false(self):
        cfg = AugmentationMethodConfig(repeat_per_week=False)
        assert cfg.repeat_per_week is False

    def test_explicit_method(self):
        cfg = AugmentationMethodConfig(method="llm_agent")
        assert cfg.method == "llm_agent"

    def test_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            AugmentationMethodConfig(method="unsupported")

    def test_output_provided(self):
        cfg = AugmentationMethodConfig(output=ScenarioOutputConfig(dir="./out/a"))
        assert cfg.output is not None
        assert cfg.output.dir == "./out/a"

    def test_loss_weights_default_sum_to_one(self):
        cfg = AugmentationMethodConfig()
        total = sum(
            [
                cfg.loss.lambda_cov,
                cfg.loss.lambda_cal,
                cfg.loss.lambda_pref,
                cfg.loss.lambda_disp,
                cfg.loss.lambda_merge,
                cfg.loss.lambda_spread,
                cfg.loss.lambda_divide,
            ]
        )
        assert abs(total - 1.0) < 1e-6

    def test_merge_threshold_bounds(self):
        with pytest.raises(ValidationError):
            AugmentationMethodConfig(merge_threshold=1.5)
        with pytest.raises(ValidationError):
            AugmentationMethodConfig(merge_threshold=-0.1)


# ---------------------------------------------------------------------------
# ScenarioDefinition
# ---------------------------------------------------------------------------


class TestScenarioDefinition:
    def test_defaults(self):
        sd = ScenarioDefinition(id="my_scenario")
        assert sd.description == ""
        assert len(sd.augmentation) == 1
        assert sd.augmentation[0].method == "greedy"

    def test_label_defaults_empty_and_accepts_display_name(self):
        assert ScenarioDefinition(id="s").label == ""
        sd = ScenarioDefinition(id="dqn_rl_per_person", label="DQN RL per person")
        assert sd.label == "DQN RL per person"

    def test_id_required(self):
        with pytest.raises(ValidationError):
            ScenarioDefinition.model_validate({})

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            ScenarioDefinition(id="")

    def test_multiple_methods(self):
        sd = ScenarioDefinition(
            id="test",
            augmentation=[
                AugmentationMethodConfig(method="greedy"),
                AugmentationMethodConfig(method="llm_agent"),
            ],
        )
        assert len(sd.augmentation) == 2
        assert sd.augmentation[0].method == "greedy"
        assert sd.augmentation[1].method == "llm_agent"

    def test_single_block_wraps_to_one_weekly_entry(self):
        sd = ScenarioDefinition(id="s", task_generation={"num_tasks": 7})
        weekly = sd.weekly_task_generation()
        assert len(weekly) == 1
        assert weekly[0].num_tasks == 7

    def test_single_block_must_not_set_week(self):
        with pytest.raises(ValidationError, match="must not set `week`"):
            ScenarioDefinition(id="s", task_generation={"num_tasks": 7, "week": 1})

    def test_per_week_list_parses_and_normalises(self):
        sd = ScenarioDefinition(
            id="s",
            task_generation=[
                {"week": 1, "num_tasks": 10},
                {"week": 2, "num_tasks": 14},
            ],
        )
        weekly = sd.weekly_task_generation()
        assert [c.week for c in weekly] == [1, 2]
        assert [c.num_tasks for c in weekly] == [10, 14]

    def test_per_week_list_rejects_duplicate_week(self):
        with pytest.raises(ValidationError, match="must not repeat a `week`"):
            ScenarioDefinition(
                id="s",
                task_generation=[
                    {"week": 1, "num_tasks": 10},
                    {"week": 1, "num_tasks": 14},
                ],
            )

    def test_per_week_list_rejects_two_defaults(self):
        with pytest.raises(ValidationError, match="at most one"):
            ScenarioDefinition(
                id="s",
                task_generation=[
                    {"num_tasks": 10},
                    {"num_tasks": 14},
                ],
            )

    def test_per_week_list_requires_repeat_per_week(self):
        with pytest.raises(ValidationError, match="repeat_per_week=True"):
            ScenarioDefinition(
                id="s",
                task_generation=[{"week": 1, "num_tasks": 10}],
                augmentation=[
                    AugmentationMethodConfig(method="greedy", repeat_per_week=False)
                ],
            )

    def test_per_week_list_with_repeat_per_week_passes(self):
        sd = ScenarioDefinition(
            id="s",
            task_generation=[{"week": 1, "num_tasks": 10}],
            augmentation=[
                AugmentationMethodConfig(method="greedy", repeat_per_week=True)
            ],
        )
        assert len(sd.weekly_task_generation()) == 1


# ---------------------------------------------------------------------------
# ExperimentScenariosConfig
# ---------------------------------------------------------------------------


class TestExperimentScenariosConfig:
    def _minimal(self, **overrides) -> dict:
        base = {
            "experiment_id": "exp_a",
            "scenarios": [{"id": "scenario_1"}],
        }
        base.update(overrides)
        return base

    def test_minimal_valid(self):
        cfg = ExperimentScenariosConfig.model_validate(self._minimal())
        assert cfg.experiment_id == "exp_a"
        assert len(cfg.scenarios) == 1
        assert cfg.output_base == "./output"

    def test_run_dir_defaults_empty(self):
        cfg = ExperimentScenariosConfig.model_validate(self._minimal())
        assert cfg.run_dir == ""

    def test_effective_run_dir_from_run_dir(self):
        cfg = ExperimentScenariosConfig.model_validate(
            self._minimal(run_dir="./output/exp_a")
        )
        assert cfg.effective_run_dir() == "./output/exp_a"

    def test_effective_run_dir_derived_when_empty(self):
        cfg = ExperimentScenariosConfig.model_validate(self._minimal())
        assert cfg.effective_run_dir() == "./output/exp_a"

    def test_task_generation_dir(self):
        cfg = ExperimentScenariosConfig.model_validate(self._minimal())
        assert (
            cfg.task_generation_dir("nutrition_l1")
            == "./output/exp_a/task_generation/nutrition_l1"
        )

    def test_scenario_method_dir(self):
        cfg = ExperimentScenariosConfig.model_validate(self._minimal())
        assert (
            cfg.scenario_method_dir("nutrition_l1", "greedy")
            == "./output/exp_a/scenarios/nutrition_l1/greedy"
        )

    def test_custom_output_base(self):
        cfg = ExperimentScenariosConfig.model_validate(
            self._minimal(output_base="/data/runs")
        )
        assert (
            cfg.scenario_method_dir("s1", "llm_agent")
            == "/data/runs/exp_a/scenarios/s1/llm_agent"
        )

    def test_task_generation_dir_uses_custom_output_base(self):
        cfg = ExperimentScenariosConfig.model_validate(
            self._minimal(output_base="/data/runs")
        )
        assert cfg.task_generation_dir("s1") == "/data/runs/exp_a/task_generation/s1"

    def test_path_helpers_isolate_task_generation_from_scenarios(self):
        """task_generation/ and scenarios/ are SIBLINGS under the experiment
        root; `summary_report.txt` and similar task-gen artifacts must
        never accidentally land inside a scenarios/<id>/<method>/ tree."""
        cfg = ExperimentScenariosConfig.model_validate(self._minimal())
        tg = cfg.task_generation_dir("s1")
        sm = cfg.scenario_method_dir("s1", "llm_agent")
        # Both share the same `<output_base>/<experiment_id>/` prefix …
        prefix = f"{cfg.output_base}/{cfg.experiment_id}/"
        assert tg.startswith(prefix) and sm.startswith(prefix)
        # … but live under different stage-grouping subdirs.
        assert "/task_generation/" in tg and "/scenarios/" not in tg
        assert "/scenarios/" in sm and "/task_generation/" not in sm

    def test_empty_experiment_id_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentScenariosConfig.model_validate(self._minimal(experiment_id=""))

    def test_empty_scenarios_list_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentScenariosConfig.model_validate(
                {**self._minimal(), "scenarios": []}
            )

    def test_multiple_scenarios(self):
        cfg = ExperimentScenariosConfig.model_validate(
            {
                **self._minimal(),
                "scenarios": [{"id": "s1"}, {"id": "s2"}],
            }
        )
        assert len(cfg.scenarios) == 2


# ---------------------------------------------------------------------------
# load_experiment_scenarios
# ---------------------------------------------------------------------------


class TestLoadExperimentScenarios:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "scenarios.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def _valid_yaml(self) -> str:
        return (
            "experiment_id: exp_a\n"
            "run_dir: ./output/exp_a\n"
            "scenarios:\n"
            "  - id: nutrition_l1\n"
        )

    def test_valid_yaml_loads(self, tmp_path):
        path = self._write_yaml(tmp_path, self._valid_yaml())
        cfg = load_experiment_scenarios(path)
        assert cfg.experiment_id == "exp_a"
        assert cfg.scenarios[0].id == "nutrition_l1"

    def test_missing_scenarios_key_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, "experiment_id: exp_a\n")
        with pytest.raises(ScenarioConfigError, match="top-level 'scenarios:'"):
            load_experiment_scenarios(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ScenarioConfigError, match="not found"):
            load_experiment_scenarios(tmp_path / "no_file.yaml")

    def test_validation_error_raised(self, tmp_path):
        path = self._write_yaml(tmp_path, "experiment_id: ''\nscenarios:\n  - id: s1\n")
        with pytest.raises(ScenarioConfigError, match="validation error"):
            load_experiment_scenarios(path)


# ---------------------------------------------------------------------------
# load_config; auto-detect
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def _write(self, tmp_path: Path, content: str, name: str = "cfg.yaml") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_detects_legacy_scenario_key(self, tmp_path):
        path = self._write(
            tmp_path,
            "scenario:\n  id: s\n  output:\n    dir: ./out\n",
        )
        cfg = load_config(path)
        assert isinstance(cfg, ScenarioConfig)
        assert cfg.id == "s"

    def test_detects_multi_scenario_key(self, tmp_path):
        path = self._write(
            tmp_path,
            "experiment_id: exp_a\nscenarios:\n  - id: s1\n",
        )
        cfg = load_config(path)
        assert isinstance(cfg, ExperimentScenariosConfig)
        assert cfg.experiment_id == "exp_a"

    def test_neither_key_raises(self, tmp_path):
        path = self._write(tmp_path, "foo: bar\n")
        with pytest.raises(ScenarioConfigError, match="'scenario:' or 'scenarios:'"):
            load_config(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ScenarioConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_non_mapping_root_raises(self, tmp_path):
        path = self._write(tmp_path, "- a\n- b\n")
        with pytest.raises(ScenarioConfigError, match="mapping"):
            load_config(path)

    def test_experiment_a_scenarios_yaml_loads(self):
        """A real scenarios.yaml must parse correctly."""
        real = (
            Path(__file__).resolve().parents[3]
            / "tests"
            / "fixtures"
            / "persona"
            / "experiment_a"
            / "scenarios.yaml"
        )
        cfg = load_config(real)
        assert isinstance(cfg, ExperimentScenariosConfig)
        assert cfg.experiment_id == "experiment_a"
        assert len(cfg.scenarios) >= 2
        # Each scenario has ≥ 1 augmentation method
        for scenario in cfg.scenarios:
            assert len(scenario.augmentation) >= 1
        # Output dirs are auto-derived correctly
        assert "nutrition_l1" in cfg.scenario_method_dir("nutrition_l1", "greedy")


# ---------------------------------------------------------------------------
# LLMModelConfig; per-stage model overrides
# ---------------------------------------------------------------------------


class TestLLMModelConfig:
    def test_minimal_model_only(self):
        """Provider defaults to None (= inherit env); model is the only required field."""
        cfg = LLMModelConfig(model="gpt-4o-mini")
        assert cfg.provider is None
        assert cfg.model == "gpt-4o-mini"
        assert cfg.max_retries == 1

    def test_explicit_provider_and_model(self):
        cfg = LLMModelConfig(provider="openai", model="gpt-4.1-mini", max_retries=3)
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4.1-mini"
        assert cfg.max_retries == 3

    def test_anthropic_provider(self):
        cfg = LLMModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-haiku-4-5-20251001"

    def test_openrouter_provider(self):
        cfg = LLMModelConfig(provider="openrouter", model="some/router-model")
        assert cfg.provider == "openrouter"

    def test_empty_model_rejected(self):
        with pytest.raises(ValidationError):
            LLMModelConfig(model="")

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValidationError):
            LLMModelConfig(provider="bedrock", model="foo")  # type: ignore[arg-type]

    def test_max_retries_must_be_positive(self):
        with pytest.raises(ValidationError):
            LLMModelConfig(model="gpt-4o-mini", max_retries=0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            LLMModelConfig.model_validate({"model": "gpt-4o-mini", "temperature": 0.5})

    def test_frozen(self):
        cfg = LLMModelConfig(model="gpt-4o-mini")
        with pytest.raises(ValidationError):
            cfg.model = "changed"  # type: ignore[misc]

    def test_scenario_definition_carries_overrides(self):
        """Both `task_generator_model` and `evaluator_model` plant on a
        multi-scenario `ScenarioDefinition`; defaults are `None`."""
        s = ScenarioDefinition.model_validate({"id": "s1"})
        assert s.task_generator_model is None
        assert s.evaluator_model is None
        s2 = ScenarioDefinition.model_validate(
            {
                "id": "s2",
                "task_generator_model": {"model": "gpt-4o-mini"},
                "evaluator_model": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "max_retries": 2,
                },
            }
        )
        assert isinstance(s2.task_generator_model, LLMModelConfig)
        assert s2.task_generator_model.model == "gpt-4o-mini"
        assert isinstance(s2.evaluator_model, LLMModelConfig)
        assert s2.evaluator_model.max_retries == 2

    def test_legacy_scenario_config_carries_overrides(self):
        """Single-scenario `ScenarioConfig` exposes the same two
        optional override blocks so legacy YAMLs can also benchmark
        models without migrating to the multi-scenario format."""
        cfg = ScenarioConfig.model_validate(
            {
                "id": "single",
                "output": {"dir": "./out"},
                "task_generator_model": {"model": "gpt-4o-mini"},
                "evaluator_model": {"model": "gpt-4.1-mini"},
            }
        )
        assert cfg.task_generator_model.model == "gpt-4o-mini"
        assert cfg.evaluator_model.model == "gpt-4.1-mini"

    def test_legacy_scenario_overrides_default_to_none(self):
        cfg = ScenarioConfig.model_validate(
            {"id": "single", "output": {"dir": "./out"}}
        )
        assert cfg.task_generator_model is None
        assert cfg.evaluator_model is None


_EXPERIMENT_F_YAML = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "persona"
    / "experiment_f"
    / "scenarios.yaml"
)


_EXPERIMENT_F_BASELINE_ID = "oneshot_gpt_4o_mini"
_EXPERIMENT_F_AUG_AXIS_ID = "oneshot_gpt_4_1_mini"
_EXPERIMENT_F_EVAL_AXIS_ID = "eval_gpt_4_1_mini"
_EXPERIMENT_F_TASKGEN_AXIS_ID = "taskgen_gpt_4_1_mini"
_EXPERIMENT_F_ABLATION_ID = "ablation_pb24_oneshot_gpt_4o_mini"

_EXPERIMENT_F_ALL_IDS = sorted(
    [
        _EXPERIMENT_F_BASELINE_ID,
        _EXPERIMENT_F_AUG_AXIS_ID,
        _EXPERIMENT_F_EVAL_AXIS_ID,
        _EXPERIMENT_F_TASKGEN_AXIS_ID,
        _EXPERIMENT_F_ABLATION_ID,
    ]
)


class TestExperimentFScenariosYaml:
    """Scenarios.yaml round-trip plus per-axis A/B shape."""

    def test_loads(self):
        cfg = load_config(_EXPERIMENT_F_YAML)
        assert isinstance(cfg, ExperimentScenariosConfig)
        assert cfg.experiment_id == "experiment_f"
        ids = sorted(s.id for s in cfg.scenarios)
        assert ids == _EXPERIMENT_F_ALL_IDS

    def test_every_scenario_has_an_llm_agent(self):
        cfg = load_config(_EXPERIMENT_F_YAML)
        for scenario in cfg.scenarios:
            assert any(
                m.method == "llm_agent" for m in scenario.augmentation
            ), f"scenario {scenario.id} is missing an llm_agent method"

    def test_only_baseline_scenario_runs_greedy(self):
        """Greedy is shipped only on the baseline (`oneshot_gpt_4o_mini`)."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        for scenario in cfg.scenarios:
            for method_cfg in scenario.augmentation:
                if method_cfg.method == "greedy":
                    assert scenario.id == "oneshot_gpt_4o_mini"

    def test_shared_task_generation_config(self):
        cfg = load_config(_EXPERIMENT_F_YAML)
        for scenario in cfg.scenarios:
            tg = scenario.task_generation
            assert tg.method == "graphrag_grounded"
            assert tg.num_tasks == 20
            assert tg.max_fetch_retries == 3
            assert tg.prompt_template == "health_improvement"
            assert sorted(tg.filters.domains) == [
                "MentalWellbeingTask",
                "PhysicalActivityTask",
            ]
            assert sorted(tg.filters.difficulty) == ["Level3", "Level4"]

    def test_baseline_pins_every_stage_to_gpt_4o_mini(self):
        """Baseline pins all three stage models."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        baseline = next(s for s in cfg.scenarios if s.id == _EXPERIMENT_F_BASELINE_ID)
        assert baseline.task_generator_model is not None
        assert baseline.task_generator_model.model == "gpt-4o-mini"
        assert baseline.evaluator_model is not None
        assert baseline.evaluator_model.model == "gpt-4o-mini"
        assert baseline.augmentation[0].llm_agent.model == "gpt-4o-mini"

    def test_augmenter_axis_ab_only_swaps_augmenter(self):
        """`oneshot_gpt_4_1_mini` swaps only the augmenter model."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        scenario = next(s for s in cfg.scenarios if s.id == _EXPERIMENT_F_AUG_AXIS_ID)
        assert scenario.task_generator_model is not None
        assert scenario.task_generator_model.model == "gpt-4o-mini"
        assert scenario.evaluator_model is not None
        assert scenario.evaluator_model.model == "gpt-4o-mini"
        assert scenario.augmentation[0].llm_agent.model == "gpt-4.1-mini"

    def test_evaluator_axis_ab_only_swaps_evaluator(self):
        """`eval_gpt_4_1_mini` swaps only the evaluator model."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        scenario = next(s for s in cfg.scenarios if s.id == _EXPERIMENT_F_EVAL_AXIS_ID)
        assert scenario.task_generator_model is not None
        assert scenario.task_generator_model.model == "gpt-4o-mini"
        assert scenario.augmentation[0].llm_agent.model == "gpt-4o-mini"
        assert scenario.evaluator_model is not None
        assert scenario.evaluator_model.model == "gpt-4.1-mini"

    def test_task_generation_axis_ab_only_swaps_task_generator(self):
        """`taskgen_gpt_4_1_mini` swaps only the task-generator model."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        scenario = next(
            s for s in cfg.scenarios if s.id == _EXPERIMENT_F_TASKGEN_AXIS_ID
        )
        assert scenario.augmentation[0].llm_agent.model == "gpt-4o-mini"
        assert scenario.evaluator_model is not None
        assert scenario.evaluator_model.model == "gpt-4o-mini"
        assert scenario.task_generator_model is not None
        assert scenario.task_generator_model.model == "gpt-4.1-mini"

    def test_each_axis_scenario_swaps_exactly_one_model(self):
        """Every model-axis scenario differs from baseline on one stage.

        The ablation scenario is excluded because it varies the prompt
        at constant models.
        """
        cfg = load_config(_EXPERIMENT_F_YAML)
        by_id = {s.id: s for s in cfg.scenarios}
        baseline = by_id[_EXPERIMENT_F_BASELINE_ID]

        def _stage_models(scenario):
            return (
                scenario.task_generator_model.model,
                scenario.augmentation[0].llm_agent.model,
                scenario.evaluator_model.model,
            )

        baseline_triple = _stage_models(baseline)
        for axis_id in (
            _EXPERIMENT_F_AUG_AXIS_ID,
            _EXPERIMENT_F_EVAL_AXIS_ID,
            _EXPERIMENT_F_TASKGEN_AXIS_ID,
        ):
            triple = _stage_models(by_id[axis_id])
            differences = [a != b for a, b in zip(triple, baseline_triple)]
            assert sum(differences) == 1, (
                f"{axis_id} should differ from baseline on exactly one "
                f"stage; got {triple} vs baseline {baseline_triple}"
            )

    def test_ablation_scenario_pins_every_stage_to_baseline_model(self):
        """Ablation scenario uses the baseline model triple."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        scenario = next(s for s in cfg.scenarios if s.id == _EXPERIMENT_F_ABLATION_ID)
        assert scenario.task_generator_model is not None
        assert scenario.task_generator_model.model == "gpt-4o-mini"
        assert scenario.evaluator_model is not None
        assert scenario.evaluator_model.model == "gpt-4o-mini"
        assert scenario.augmentation[0].llm_agent.model == "gpt-4o-mini"

    def test_ablation_scenario_declares_pb24_design(self):
        """Pin the shipped design (folded PB-24) as a regression guard."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        scenario = next(s for s in cfg.scenarios if s.id == _EXPERIMENT_F_ABLATION_ID)
        ablation = scenario.augmentation[0].llm_agent.prompt_ablation
        assert ablation is not None
        assert ablation.design == "plackett_burman_24"
        assert ablation.fold is True
        assert ablation.use_default_placebos is True
        assert ablation.variants == []
        assert ablation.placebos == {}

    def test_ablation_resolves_to_twenty_four_variants(self):
        """The YAML declaration produces 24 variants end-to-end.

        The all-kept extreme over the first 11 testable factors must
        be present. The 12th block (`split_dividable_tasks`) is held
        +1 in PB-12 and -1 in the fold, so the bit at position 11
        flips across variants but never coincides with all 11 other
        factors at the same extreme.
        """
        from src.scripts.scenarios.cli import _ablation_variants_for_method

        cfg = load_config(_EXPERIMENT_F_YAML)
        scenario = next(s for s in cfg.scenarios if s.id == _EXPERIMENT_F_ABLATION_ID)
        variants = _ablation_variants_for_method(scenario.augmentation[0])
        assert len(variants) == 24
        assert any(v.variant_id[:11] == "0" * 11 for v in variants)

    def test_non_ablation_scenarios_have_no_prompt_ablation_block(self):
        """Model-axis scenarios must not declare `prompt_ablation`."""
        cfg = load_config(_EXPERIMENT_F_YAML)
        for sid in (
            _EXPERIMENT_F_BASELINE_ID,
            _EXPERIMENT_F_AUG_AXIS_ID,
            _EXPERIMENT_F_EVAL_AXIS_ID,
            _EXPERIMENT_F_TASKGEN_AXIS_ID,
        ):
            scenario = next(s for s in cfg.scenarios if s.id == sid)
            assert (
                scenario.augmentation[0].llm_agent.prompt_ablation is None
            ), f"{sid!r} unexpectedly declares prompt_ablation"

    def test_pb24_design_covers_split_dividable_tasks(self):
        """The PB-24 ablation exercises the 12th block via the fold.

        Every augmenter run computes the full loss vector (including
        `L_divide`), so PB-24 already returns a main-effect estimate
        for `split_dividable_tasks` on `L_divide`; no separate
        scenario is needed.
        """
        from src.scripts.scenarios.augmentation.prompts.ablation_designs import (
            design_matrix,
        )
        from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
            ABLATABLE_BLOCKS,
        )
        from src.scripts.scenarios.cli import _ablation_variants_for_method

        cfg = load_config(_EXPERIMENT_F_YAML)
        scenario = next(s for s in cfg.scenarios if s.id == _EXPERIMENT_F_ABLATION_ID)
        variants = _ablation_variants_for_method(scenario.augmentation[0])
        assert len(variants) == 24

        col_idx = ABLATABLE_BLOCKS.index("split_dividable_tasks")
        matrix = design_matrix(variants)
        col = [row[col_idx] for row in matrix]
        # 12 +1 and 12 -1 means the column gives a real main-effect
        # estimate, not a frozen always-kept column.
        assert col.count(+1) == 12
        assert col.count(-1) == 12


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.buffer_minutes == 30
        assert cfg.merge_threshold == pytest.approx(0.65)
        assert cfg.disp_half_life_days == pytest.approx(2.0)
        assert cfg.divide_duration_tolerance_pct == pytest.approx(0.15)

    def test_divide_tolerance_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            EvaluationConfig(divide_duration_tolerance_pct=-0.01)

    def test_divide_tolerance_above_one_rejected(self):
        with pytest.raises(ValidationError):
            EvaluationConfig(divide_duration_tolerance_pct=1.01)

    def test_divide_tolerance_zero_accepted(self):
        cfg = EvaluationConfig(divide_duration_tolerance_pct=0.0)
        assert cfg.divide_duration_tolerance_pct == 0.0

    def test_divide_tolerance_one_accepted(self):
        cfg = EvaluationConfig(divide_duration_tolerance_pct=1.0)
        assert cfg.divide_duration_tolerance_pct == 1.0


_EXPERIMENT_H_YAML = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "persona"
    / "experiment_h"
    / "scenarios.yaml"
)


class TestExperimentHScenariosYaml:
    """Scenarios.yaml round-trip plus observation block invariants."""

    def test_loads(self):
        cfg = load_config(_EXPERIMENT_H_YAML)
        assert isinstance(cfg, ExperimentScenariosConfig)
        assert cfg.experiment_id == "experiment_h"
        assert [s.id for s in cfg.scenarios] == ["context_aware_h"]

    def test_three_augmentation_methods_compare_blind_informed_greedy(self):
        cfg = load_config(_EXPERIMENT_H_YAML)
        scenario = cfg.scenarios[0]
        methods = [m.method for m in scenario.augmentation]
        assert methods == ["llm_agent", "llm_agent", "greedy"]

    def test_blind_llm_has_no_context_opt_ins(self):
        cfg = load_config(_EXPERIMENT_H_YAML)
        blind = cfg.scenarios[0].augmentation[0]
        assert blind.observation.contexts == []
        assert blind.observation.host_flags == []

    def test_informed_llm_opts_into_four_categories(self):
        cfg = load_config(_EXPERIMENT_H_YAML)
        informed = cfg.scenarios[0].augmentation[1]
        assert set(informed.observation.contexts) == {
            "mood_emotion",
            "energy_state",
            "location",
            "weather_environment",
        }
        assert informed.observation.context_detail == "summary"

    def test_loss_carries_nonzero_context_fit_lambda(self):
        cfg = load_config(_EXPERIMENT_H_YAML)
        for method in cfg.scenarios[0].augmentation:
            assert method.loss.lambda_context_fit == pytest.approx(0.10)
