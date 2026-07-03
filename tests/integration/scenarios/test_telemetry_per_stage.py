"""Integration tests for the per-stage telemetry pipeline.

Pins the regression discovered 2026-05-15: the legacy aggregator only
read task-generation sidecars from the shared `tasks/` dir, so
greedy and llm_agent reported byte-identical telemetry numbers even
though their actual cost differed by orders of magnitude (greedy: 0
LLM tokens; llm_agent: thousands per person).

The new pipeline writes per-stage sidecars under
`<method>/augmented/_telemetry/` and `<method>/evaluation/_telemetry/`
so `aggregate_full_pipeline` can produce method-distinct numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.metrics.telemetry import (
    AugmentTelemetry,
    EvaluateTelemetry,
    TaskGenTelemetry,
    TaskTelemetryRow,
    aggregate_full_pipeline,
    write_augment_sidecar,
    write_evaluate_sidecar,
    write_taskgen_sidecar,
)

# ---------------------------------------------------------------------------
# Fixture helpers; seed sidecars that mirror what the CLI would write
# during a real `scenarios run` over a 2-persona cohort.
# ---------------------------------------------------------------------------


def _seed_task_generation(tasks_dir: Path) -> None:
    """Two persons in one persona, identical task-gen costs.

    Tasks dir is shared across methods; both greedy and llm_agent see
    the same task-generation telemetry; that's correct because task
    generation runs once per scenario regardless of method.
    """
    for pid in ("b_parttime_morning_0001", "b_parttime_morning_0002"):
        t = TaskGenTelemetry(
            person_id=pid,
            scenario_id="senior_walking_l2",
            method="graphrag_grounded",
            fetch_model="gpt-4.1-mini",
            paraphrase_model="gpt-4.1-mini",
            embedder_model="text-embedding-3-small",
        )
        t.record_llm_call(
            stage="fetch",
            input_tokens=2000,
            output_tokens=400,
            wall_time_seconds=5.0,
        )
        t.record_fetch_summary(
            num_tasks_requested=5,
            num_tasks_accepted=5,
            per_attempt_fabricated=[1],
            per_attempt_duplicate=[0],
            per_attempt_parse_failed=[False],
        )
        for label in ("walk_5000_steps", "morning_yoga"):
            t.record_task(
                TaskTelemetryRow(
                    label=label,
                    ontology_uri=f"https://ex.org/task/{label}",
                    fetch_attempt=1,
                    paraphrase_used="paraphrase",
                    length_delta=0.0,
                    similarity=1.0,
                    missing_emojis={},
                    extra_emojis={},
                    failed_gate=None,
                    canonical_description="C",
                    personalized_description="P",
                )
            )
        write_taskgen_sidecar(t, tasks_dir)


def _seed_greedy_augmentation(aug_dir: Path) -> None:
    """Greedy is a local solver; wall-time only, zero LLM tokens."""
    for pid in ("b_parttime_morning_0001", "b_parttime_morning_0002"):
        t = AugmentTelemetry(
            person_id=pid,
            scenario_id="senior_walking_l2",
            method="greedy",
            wall_time_seconds=0.5,
            n_tasks_attempted=5,
            n_tasks_placed=5,
            n_tasks_dropped=0,
        )
        write_augment_sidecar(t, aug_dir)


def _seed_llm_agent_augmentation(aug_dir: Path) -> None:
    """llm_agent calls the LLM per week; real token + USD spend."""
    for pid in ("b_parttime_morning_0001", "b_parttime_morning_0002"):
        t = AugmentTelemetry(
            person_id=pid,
            scenario_id="senior_walking_l2",
            method="llm_agent",
            model="gpt-4.1-mini",
            wall_time_seconds=45.0,
            n_tasks_attempted=5,
            n_tasks_placed=4,
            n_tasks_dropped=1,
        )
        # 4 weekly LLM calls per person (one per ISO week of the horizon).
        for _ in range(4):
            t.record_llm_call(
                input_tokens=3500,
                output_tokens=400,
                wall_time_seconds=10.0,
            )
        write_augment_sidecar(t, aug_dir)


def _seed_evaluation(eval_dir: Path) -> None:
    """Both methods get the same evaluation cost shape; one semantic
    oracle invocation + ~10 URI validations per person."""
    for pid in ("b_parttime_morning_0001", "b_parttime_morning_0002"):
        t = EvaluateTelemetry(
            person_id=pid,
            scenario_id="senior_walking_l2",
            method="placeholder",  # actual method label is set per-cohort
            wall_time_seconds=2.0,
        )
        t.record_semantic_lookup(wall_time_seconds=1.5)
        for _ in range(10):
            t.record_uri_validation(wall_time_seconds=0.05)
        write_evaluate_sidecar(t, eval_dir)


# ---------------------------------------------------------------------------
# The integration test
# ---------------------------------------------------------------------------


class TestGreedyVsLLMAgentTelemetryDiffer:
    """End-to-end; same task-gen costs, distinct augment costs, identical
    eval costs.  The roll-up must surface the difference instead of
    silently coalescing both methods into the same headline numbers."""

    @pytest.fixture()
    def staged(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        greedy_aug = tmp_path / "greedy" / "augmented"
        greedy_eval = tmp_path / "greedy" / "evaluation"
        llm_aug = tmp_path / "llm_agent" / "augmented"
        llm_eval = tmp_path / "llm_agent" / "evaluation"

        _seed_task_generation(tasks_dir)
        _seed_greedy_augmentation(greedy_aug)
        _seed_evaluation(greedy_eval)
        _seed_llm_agent_augmentation(llm_aug)
        _seed_evaluation(llm_eval)

        greedy_summary = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=greedy_aug,
            evaluation_dir=greedy_eval,
            scenario_id="senior_walking_l2",
            method="greedy",
        )
        llm_summary = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=llm_aug,
            evaluation_dir=llm_eval,
            scenario_id="senior_walking_l2",
            method="llm_agent",
        )
        return greedy_summary, llm_summary

    def test_n_persons_and_n_personas_are_equal(self, staged):
        greedy, llm = staged
        # Identical cohort; both methods see the same 2 persons / 1 persona.
        assert greedy["n_persons"] == llm["n_persons"] == 2
        assert greedy["n_personas"] == llm["n_personas"] == 1

    def test_total_tokens_strictly_greater_for_llm_agent(self, staged):
        """The headline regression: greedy and llm_agent USED to produce
        identical `tokens_total` because only task-gen sidecars were
        aggregated.  After the per-stage rewrite the augmenter's tokens
        flow into the totals so llm_agent must overshoot greedy by the
        augment-stage spend."""
        greedy, llm = staged
        assert greedy["tokens_total"] < llm["tokens_total"]

    def test_total_wall_time_strictly_greater_for_llm_agent(self, staged):
        greedy, llm = staged
        assert greedy["wall_time_seconds_total"] < llm["wall_time_seconds_total"]

    def test_estimated_usd_strictly_greater_for_llm_agent(self, staged):
        greedy, llm = staged
        # Both have known cost (task gen has it).  llm_agent additionally
        # incurs augment-time USD; greedy does not.
        assert greedy["estimated_usd_total"] is not None
        assert llm["estimated_usd_total"] is not None
        assert greedy["estimated_usd_total"] < llm["estimated_usd_total"]

    def test_by_stage_augmentation_diverges(self, staged):
        """The augmentation stage block is where greedy and llm_agent
        actually differ; the test pins both axes (tokens + cost)."""
        greedy_aug = staged[0]["by_stage"]["augmentation"]
        llm_aug = staged[1]["by_stage"]["augmentation"]
        assert greedy_aug["tokens_total"] == 0
        assert llm_aug["tokens_total"] > 0
        assert (
            greedy_aug["estimated_usd_total"] is None
            or greedy_aug["estimated_usd_total"] == 0.0
        )
        assert (llm_aug["estimated_usd_total"] or 0.0) > 0.0

    def test_by_stage_task_generation_is_identical(self, staged):
        """Task generation runs once per scenario, before the augment
        method is chosen; both methods MUST see identical task-gen
        numbers."""
        greedy_tg = staged[0]["by_stage"]["task_generation"]
        llm_tg = staged[1]["by_stage"]["task_generation"]
        assert greedy_tg["wall_time_seconds_total"] == llm_tg["wall_time_seconds_total"]
        assert greedy_tg["tokens_total"] == llm_tg["tokens_total"]
        assert greedy_tg["estimated_usd_total"] == llm_tg["estimated_usd_total"]

    def test_by_stage_evaluation_extras_present(self, staged):
        """The evaluation stage carries oracle counters (semantic +
        URI validations); surface the per-stage cost of the eval pass."""
        for summary in staged:
            ev = summary["by_stage"]["evaluation"]
            assert ev["embedding_lookups_total"] == 2  # one per person
            assert ev["uri_validations_total"] == 20  # 10 per person × 2

    def test_by_persona_carries_total_and_per_person_avg(self, staged):
        """Both summaries should bucket the 2 persons under one persona
        and report cohort totals + within-cohort per-person averages."""
        for summary in staged:
            assert "b_parttime_morning" in summary["by_persona"]
            cohort = summary["by_persona"]["b_parttime_morning"]
            assert cohort["n_persons"] == 2
            # avg_per_person = total / 2 by construction
            assert cohort["wall_time_seconds_avg_per_person"] == pytest.approx(
                cohort["wall_time_seconds_total"] / 2
            )

    def test_by_persona_diverges_between_methods(self, staged):
        """Per-cohort totals must differ between greedy and llm_agent
        because the augment-stage spend bucketed by persona pulls the
        cohort totals apart; even though both cohorts contain the same
        2 persons."""
        greedy_cohort = staged[0]["by_persona"]["b_parttime_morning"]
        llm_cohort = staged[1]["by_persona"]["b_parttime_morning"]
        assert (
            greedy_cohort["wall_time_seconds_total"]
            < llm_cohort["wall_time_seconds_total"]
        )
        assert greedy_cohort["tokens_total"] < llm_cohort["tokens_total"]


class TestSidecarFilesRoundTripAcrossPipeline:
    """Sanity guard; the per-stage sidecars must be readable JSON and
    the aggregator must surface the same numbers we wrote."""

    def test_taskgen_sidecar_is_valid_json(self, tmp_path):
        _seed_task_generation(tmp_path)
        files = list((tmp_path / "_telemetry").glob("*.json"))
        assert len(files) == 2
        for fp in files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            assert data["scenario_id"] == "senior_walking_l2"
            assert data["persona_id"] == "b_parttime_morning"

    def test_augment_sidecar_carries_method_label(self, tmp_path):
        _seed_llm_agent_augmentation(tmp_path)
        for fp in (tmp_path / "_telemetry").glob("*.json"):
            data = json.loads(fp.read_text(encoding="utf-8"))
            assert data["method"] == "llm_agent"
            assert data["model"] == "gpt-4.1-mini"
            assert data["llm_calls"]["total_tokens"] > 0

    def test_evaluate_sidecar_carries_oracle_counters(self, tmp_path):
        _seed_evaluation(tmp_path)
        for fp in (tmp_path / "_telemetry").glob("*.json"):
            data = json.loads(fp.read_text(encoding="utf-8"))
            assert data["semantic_oracle"]["n_calls"] == 1
            assert data["uri_validation"]["n_calls"] == 10
