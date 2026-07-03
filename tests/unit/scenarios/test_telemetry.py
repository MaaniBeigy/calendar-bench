"""Unit tests for src.scripts.scenarios.metrics.telemetry.

Coverage targets:
  estimate_usd          ; USD cost lookup (known / unknown model).
  estimate_tokens       ; tiktoken-or-fallback token count.
  derive_persona_id     ; strip `_NNNN` suffix from a person_id.
  LLMCallStats / EmbedderStats / FetchSummary / OracleStats; defaults.
  TaskGenTelemetry      ; record + serialise + persona_id auto-fill.
  AugmentTelemetry      ; record LLM tokens + serialise + persona_id.
  EvaluateTelemetry     ; record semantic / URI / judge + serialise.
  write_taskgen_sidecar / write_augment_sidecar / write_evaluate_sidecar.
  aggregate_taskgen_sidecars / aggregate_augment_sidecars /
    aggregate_evaluate_sidecars; return raw payload lists.
  aggregate_full_pipeline; by_stage + by_persona roll-up.
  _by_persona_breakdown  ; cohort grouping.
  write_summary_sidecar  ; _summary.json drop.
  _Timer                 ; elapsed-time helper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.scripts.scenarios.metrics.telemetry import (
    PRICING_FILE,
    AugmentTelemetry,
    EmbedderStats,
    EvaluateTelemetry,
    FetchSummary,
    LLMCallStats,
    OracleStats,
    TaskGenTelemetry,
    TaskTelemetryRow,
    _by_persona_breakdown,
    _empty_stage,
    _Timer,
    aggregate_augment_sidecars,
    aggregate_evaluate_sidecars,
    aggregate_full_pipeline,
    aggregate_taskgen_sidecars,
    derive_persona_id,
    estimate_tokens,
    estimate_usd,
    load_pricing,
    write_augment_sidecar,
    write_evaluate_sidecar,
    write_summary_sidecar,
    write_taskgen_sidecar,
)

# ---------------------------------------------------------------------------
# estimate_usd
# ---------------------------------------------------------------------------


class TestEstimateUsd:
    def test_known_model_returns_estimate(self):
        # 1M input tokens at gpt-4o-mini = $0.15 in
        usd = estimate_usd(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert usd == pytest.approx(0.15)

    def test_input_and_output_costs_sum(self):
        usd = estimate_usd(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert usd == pytest.approx(0.15 + 0.60)

    def test_unknown_provider_returns_none(self):
        assert (
            estimate_usd(
                provider="vendor-x",
                model="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
            )
            is None
        )

    def test_unknown_model_returns_none(self):
        assert (
            estimate_usd(
                provider="openai",
                model="not-a-model",
                input_tokens=100,
                output_tokens=50,
            )
            is None
        )

    def test_gpt_5_mini_priced(self):
        usd = estimate_usd(
            provider="openai",
            model="gpt-5-mini",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert usd == pytest.approx(0.25 + 2.00)

    def test_gpt_5_4_mini_priced(self):
        usd = estimate_usd(
            provider="openai",
            model="gpt-5.4-mini",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert usd == pytest.approx(0.75 + 4.50)

    def test_claude_opus_4_8_priced(self):
        usd = estimate_usd(
            provider="anthropic",
            model="claude-opus-4-8",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert usd == pytest.approx(5.00 + 25.00)

    def test_claude_model_under_openai_provider_returns_none(self):
        """The lookup is provider-scoped; a mislabelled provider returns n/a."""
        assert (
            estimate_usd(
                provider="openai",
                model="claude-opus-4-8",
                input_tokens=100,
                output_tokens=50,
            )
            is None
        )

    def test_pricing_table_loaded_from_file(self):
        table = load_pricing(force=True)
        assert "openai" in table
        assert "gpt-4.1-mini" in table["openai"]
        assert "anthropic" in table

    def test_pricing_file_ships_with_repo(self):
        assert PRICING_FILE.is_file()
        assert PRICING_FILE.name == "llm_pricing.json"


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_short_text_returns_at_least_one(self):
        assert estimate_tokens("xyz") >= 1

    def test_longer_text_returns_more_tokens(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 200)
        assert long > short

    def test_tiktoken_happy_path_via_monkeypatch(self, monkeypatch):
        """When `tiktoken` IS importable, estimate_tokens uses it.

        Local CI may not have tiktoken installed (the production docker
        image does); this test injects a fake module so coverage still
        exercises the encoder-loaded branch."""
        import sys

        from src.scripts.scenarios.metrics import telemetry as tel

        class _FakeEncoder:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text) // 2 + 1))

        class _FakeTiktoken:
            @staticmethod
            def get_encoding(name: str):
                assert name == "cl100k_base"
                return _FakeEncoder()

        monkeypatch.setitem(sys.modules, "tiktoken", _FakeTiktoken)
        # Force the lazy initialiser to re-import from the patched module.
        monkeypatch.setattr(tel, "_TIKTOKEN_ENCODER", None)
        try:
            n = tel.estimate_tokens("hello world")
        finally:
            # Restore so other tests get the real (or absent) encoder.
            monkeypatch.setattr(tel, "_TIKTOKEN_ENCODER", None)
        # _FakeEncoder returns len(text)//2 + 1 = 11//2 + 1 = 6 tokens.
        assert n == 6


# ---------------------------------------------------------------------------
# derive_persona_id
# ---------------------------------------------------------------------------


class TestDerivePersonaId:
    def test_strips_four_digit_instance_suffix(self):
        assert derive_persona_id("b_parttime_morning_0001") == "b_parttime_morning"
        assert derive_persona_id("a_fulltime_0042") == "a_fulltime"

    def test_returns_unchanged_when_no_suffix(self):
        assert derive_persona_id("standalone_id") == "standalone_id"

    def test_three_digit_suffix_is_kept(self):
        # Only the canonical 4-digit pattern is stripped.
        assert derive_persona_id("b_parttime_morning_001") == "b_parttime_morning_001"

    def test_empty_input_passes_through(self):
        assert derive_persona_id("") == ""


# ---------------------------------------------------------------------------
# Default-shaped dataclasses
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_llm_call_stats(self):
        s = LLMCallStats()
        assert s.attempts == 0 and s.input_tokens == 0 and s.output_tokens == 0
        assert s.total_tokens == 0 and s.wall_time_seconds == 0.0

    def test_embedder_stats(self):
        s = EmbedderStats()
        assert s.calls == 0 and s.input_tokens == 0 and s.wall_time_seconds == 0.0

    def test_fetch_summary(self):
        s = FetchSummary()
        assert (
            s.num_tasks_requested == 0
            and s.num_tasks_accepted == 0
            and s.fabricated_dropped_per_attempt == []
            and s.duplicate_dropped_per_attempt == []
            and s.parse_failed_per_attempt == []
        )

    def test_oracle_stats(self):
        s = OracleStats()
        assert s.n_calls == 0 and s.wall_time_seconds == 0.0


# ---------------------------------------------------------------------------
# TaskGenTelemetry; collector behaviour + persona_id
# ---------------------------------------------------------------------------


def _make_taskgen(**overrides) -> TaskGenTelemetry:
    defaults = dict(
        person_id="b_parttime_morning_0001",
        scenario_id="nutrition_l1",
        method="graphrag_grounded",
        provider="openai",
        fetch_model="gpt-4o-mini",
        paraphrase_model="gpt-4o-mini",
        embedder_model="text-embedding-3-small",
    )
    defaults.update(overrides)
    return TaskGenTelemetry(**defaults)


class TestTaskGenTelemetryPersonaId:
    def test_persona_id_auto_derived_from_person_id(self):
        t = _make_taskgen()
        assert t.persona_id == "b_parttime_morning"

    def test_explicit_persona_id_is_kept(self):
        t = _make_taskgen(persona_id="custom_cohort")
        assert t.persona_id == "custom_cohort"


class TestRecordLLMCall:
    def test_first_fetch_call_initialises_bucket(self):
        t = _make_taskgen()
        t.record_llm_call(
            stage="fetch",
            input_tokens=100,
            output_tokens=20,
            wall_time_seconds=1.5,
        )
        assert t.fetch_stats.attempts == 1
        assert t.fetch_stats.input_tokens == 100
        assert t.fetch_stats.output_tokens == 20
        assert t.fetch_stats.total_tokens == 120
        assert t.fetch_stats.wall_time_seconds == pytest.approx(1.5)
        assert t.wall_time_by_stage["fetch"] == pytest.approx(1.5)

    def test_repeated_calls_accumulate(self):
        t = _make_taskgen()
        for _ in range(3):
            t.record_llm_call(
                stage="fetch", input_tokens=10, output_tokens=2, wall_time_seconds=0.5
            )
        assert t.fetch_stats.attempts == 3
        assert t.fetch_stats.total_tokens == 36
        assert t.wall_time_by_stage["fetch"] == pytest.approx(1.5)

    def test_paraphrase_routed_separately(self):
        t = _make_taskgen()
        t.record_llm_call(
            stage="paraphrase",
            input_tokens=50,
            output_tokens=10,
            wall_time_seconds=0.7,
        )
        assert t.paraphrase_stats.input_tokens == 50
        assert t.fetch_stats.input_tokens == 0


class TestRecordEmbedderCall:
    def test_embedder_accumulates(self):
        t = _make_taskgen()
        t.record_embedder_call(input_tokens=100, wall_time_seconds=0.2)
        t.record_embedder_call(input_tokens=50, wall_time_seconds=0.1)
        assert t.embedder_stats.calls == 2
        assert t.embedder_stats.input_tokens == 150
        assert t.embedder_stats.wall_time_seconds == pytest.approx(0.3)
        assert t.wall_time_by_stage["gate"] == pytest.approx(0.3)


class TestRecordFetchSummary:
    def test_overrides_existing_summary(self):
        t = _make_taskgen()
        t.record_fetch_summary(
            num_tasks_requested=5,
            num_tasks_accepted=4,
            per_attempt_fabricated=[2, 1],
            per_attempt_duplicate=[0, 0],
            per_attempt_parse_failed=[False, True],
        )
        assert t.fetch_summary.num_tasks_requested == 5
        assert t.fetch_summary.fabricated_dropped_per_attempt == [2, 1]


class TestRecordTask:
    def test_appends_row(self):
        t = _make_taskgen()
        row = TaskTelemetryRow(
            label="x",
            ontology_uri="https://ex.org/task/x",
            fetch_attempt=1,
            paraphrase_used="paraphrase",
            length_delta=0.05,
            similarity=0.9,
            missing_emojis={},
            extra_emojis={},
            failed_gate=None,
            canonical_description="C",
            personalized_description="P",
        )
        t.record_task(row)
        assert t.tasks == [row]


class TestTaskGenToDict:
    def test_round_trip_through_json(self):
        t = _make_taskgen()
        t.record_llm_call(
            stage="fetch", input_tokens=200, output_tokens=50, wall_time_seconds=2.0
        )
        t.record_llm_call(
            stage="paraphrase",
            input_tokens=80,
            output_tokens=20,
            wall_time_seconds=1.0,
        )
        t.record_embedder_call(input_tokens=10, wall_time_seconds=0.1)
        t.record_fetch_summary(
            num_tasks_requested=2,
            num_tasks_accepted=2,
            per_attempt_fabricated=[1],
            per_attempt_duplicate=[0],
            per_attempt_parse_failed=[False],
        )
        t.record_task(
            TaskTelemetryRow(
                label="yoga",
                ontology_uri="https://ex.org/task/yoga",
                fetch_attempt=1,
                paraphrase_used="paraphrase",
                length_delta=0.0,
                similarity=1.0,
                missing_emojis={},
                extra_emojis={},
                failed_gate=None,
                canonical_description="Stretch.",
                personalized_description="Stretch your back.",
            )
        )
        loaded = json.loads(json.dumps(t.to_dict()))
        assert loaded["person_id"] == "b_parttime_morning_0001"
        assert loaded["persona_id"] == "b_parttime_morning"
        assert loaded["wall_time_seconds_total"] == pytest.approx(3.1)
        assert loaded["llm_calls"]["fetch"]["total_tokens"] == 250
        assert loaded["llm_calls"]["paraphrase"]["estimated_usd"] is not None
        assert loaded["embeddings"]["calls"] == 1
        assert len(loaded["tasks"]) == 1

    def test_unknown_model_returns_null_usd(self):
        t = _make_taskgen(fetch_model="unknown-model")
        t.record_llm_call(
            stage="fetch", input_tokens=10, output_tokens=2, wall_time_seconds=0.1
        )
        assert t.to_dict()["llm_calls"]["fetch"]["estimated_usd"] is None


# ---------------------------------------------------------------------------
# AugmentTelemetry
# ---------------------------------------------------------------------------


def _make_augment(**overrides) -> AugmentTelemetry:
    defaults = dict(
        person_id="b_parttime_morning_0001",
        scenario_id="nutrition_l1",
        method="llm_agent",
        model="gpt-4.1-mini",
    )
    defaults.update(overrides)
    return AugmentTelemetry(**defaults)


class TestAugmentTelemetry:
    def test_persona_id_auto_derived(self):
        t = _make_augment()
        assert t.persona_id == "b_parttime_morning"

    def test_explicit_persona_id_kept(self):
        t = _make_augment(persona_id="custom")
        assert t.persona_id == "custom"

    def test_record_llm_call_accumulates(self):
        t = _make_augment()
        t.record_llm_call(input_tokens=100, output_tokens=20, wall_time_seconds=0.5)
        t.record_llm_call(input_tokens=50, output_tokens=10, wall_time_seconds=0.3)
        assert t.llm_calls.attempts == 2
        assert t.llm_calls.input_tokens == 150
        assert t.llm_calls.output_tokens == 30
        assert t.llm_calls.total_tokens == 180
        assert t.llm_calls.wall_time_seconds == pytest.approx(0.8)

    def test_to_dict_round_trips(self):
        t = _make_augment(
            wall_time_seconds=12.5,
            n_tasks_attempted=10,
            n_tasks_placed=8,
            n_tasks_dropped=2,
        )
        t.record_llm_call(input_tokens=1000, output_tokens=200, wall_time_seconds=10.0)
        loaded = json.loads(json.dumps(t.to_dict()))
        assert loaded["person_id"] == "b_parttime_morning_0001"
        assert loaded["persona_id"] == "b_parttime_morning"
        assert loaded["method"] == "llm_agent"
        assert loaded["wall_time_seconds"] == pytest.approx(12.5)
        assert loaded["n_tasks_placed"] == 8
        assert loaded["llm_calls"]["total_tokens"] == 1200
        assert loaded["llm_calls"]["estimated_usd"] is not None

    def test_greedy_method_returns_zero_llm_block(self):
        t = _make_augment(method="greedy", model="")
        t.wall_time_seconds = 0.5
        t.n_tasks_attempted = 5
        t.n_tasks_placed = 5
        d = t.to_dict()
        assert d["method"] == "greedy"
        assert d["llm_calls"]["total_tokens"] == 0
        assert d["llm_calls"]["estimated_usd"] is None  # unknown model


# ---------------------------------------------------------------------------
# EvaluateTelemetry
# ---------------------------------------------------------------------------


def _make_evaluate(**overrides) -> EvaluateTelemetry:
    defaults = dict(
        person_id="b_parttime_morning_0001",
        scenario_id="nutrition_l1",
        method="llm_agent",
    )
    defaults.update(overrides)
    return EvaluateTelemetry(**defaults)


class TestEvaluateTelemetry:
    def test_persona_id_auto_derived(self):
        t = _make_evaluate()
        assert t.persona_id == "b_parttime_morning"

    def test_explicit_persona_id_kept(self):
        t = _make_evaluate(persona_id="custom_cohort")
        assert t.persona_id == "custom_cohort"

    def test_record_semantic_lookup(self):
        t = _make_evaluate()
        t.record_semantic_lookup(wall_time_seconds=0.2)
        t.record_semantic_lookup(wall_time_seconds=0.3)
        assert t.semantic_oracle.n_calls == 2
        assert t.semantic_oracle.wall_time_seconds == pytest.approx(0.5)

    def test_record_uri_validation(self):
        t = _make_evaluate()
        for _ in range(5):
            t.record_uri_validation(wall_time_seconds=0.05)
        assert t.uri_validation.n_calls == 5
        assert t.uri_validation.wall_time_seconds == pytest.approx(0.25)

    def test_record_judge_call(self):
        t = _make_evaluate(judge_model="gpt-4o-mini")
        t.record_judge_call(input_tokens=200, output_tokens=50, wall_time_seconds=1.0)
        assert t.judge_calls.attempts == 1
        assert t.judge_calls.total_tokens == 250

    def test_to_dict_round_trips(self):
        t = _make_evaluate(judge_model="gpt-4o-mini")
        t.wall_time_seconds = 5.5
        t.record_semantic_lookup(wall_time_seconds=2.0)
        t.record_uri_validation(wall_time_seconds=1.0)
        t.record_judge_call(input_tokens=300, output_tokens=100, wall_time_seconds=2.5)
        loaded = json.loads(json.dumps(t.to_dict()))
        assert loaded["person_id"] == "b_parttime_morning_0001"
        assert loaded["semantic_oracle"]["n_calls"] == 1
        assert loaded["uri_validation"]["n_calls"] == 1
        assert loaded["judge_calls"]["total_tokens"] == 400


# ---------------------------------------------------------------------------
# Sidecar writers
# ---------------------------------------------------------------------------


class TestWriteTaskgenSidecar:
    def test_writes_file_into_telemetry_subdir(self, tmp_path):
        t = _make_taskgen()
        out = write_taskgen_sidecar(t, tmp_path)
        assert out.parent.name == "_telemetry"
        assert out.name == "b_parttime_morning_0001.json"
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["person_id"] == "b_parttime_morning_0001"
        assert loaded["persona_id"] == "b_parttime_morning"

    def test_creates_parent_dir_if_missing(self, tmp_path):
        target = tmp_path / "nested" / "tasks"
        out = write_taskgen_sidecar(_make_taskgen(), target)
        assert out.exists()


class TestWriteAugmentSidecar:
    def test_writes_under_augmented_telemetry(self, tmp_path):
        out = write_augment_sidecar(_make_augment(), tmp_path)
        assert out.parent.name == "_telemetry"
        assert out.name == "b_parttime_morning_0001.json"
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["method"] == "llm_agent"


class TestWriteEvaluateSidecar:
    def test_writes_under_evaluation_telemetry(self, tmp_path):
        out = write_evaluate_sidecar(_make_evaluate(), tmp_path)
        assert out.parent.name == "_telemetry"
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["scenario_id"] == "nutrition_l1"


# ---------------------------------------------------------------------------
# Stage aggregators (raw payload lists)
# ---------------------------------------------------------------------------


class TestStageAggregators:
    def test_taskgen_aggregator_returns_payloads(self, tmp_path):
        t1 = _make_taskgen(person_id="a_0001")
        t2 = _make_taskgen(person_id="a_0002")
        write_taskgen_sidecar(t1, tmp_path)
        write_taskgen_sidecar(t2, tmp_path)
        payloads = aggregate_taskgen_sidecars(tmp_path)
        assert len(payloads) == 2
        assert {p["person_id"] for p in payloads} == {"a_0001", "a_0002"}

    def test_augment_aggregator_returns_payloads(self, tmp_path):
        write_augment_sidecar(_make_augment(person_id="a_0001"), tmp_path)
        payloads = aggregate_augment_sidecars(tmp_path)
        assert len(payloads) == 1

    def test_evaluate_aggregator_returns_payloads(self, tmp_path):
        write_evaluate_sidecar(_make_evaluate(person_id="a_0001"), tmp_path)
        payloads = aggregate_evaluate_sidecars(tmp_path)
        assert len(payloads) == 1

    def test_missing_directory_returns_empty_list(self, tmp_path):
        assert aggregate_taskgen_sidecars(tmp_path / "missing") == []

    def test_empty_directory_returns_empty_list(self, tmp_path):
        (tmp_path / "_telemetry").mkdir()
        assert aggregate_taskgen_sidecars(tmp_path) == []

    def test_malformed_json_skipped(self, tmp_path):
        write_taskgen_sidecar(_make_taskgen(person_id="a_0001"), tmp_path)
        bad = tmp_path / "_telemetry" / "broken.json"
        bad.write_text("not json", encoding="utf-8")
        payloads = aggregate_taskgen_sidecars(tmp_path)
        # Bad file is skipped, good file is read.
        assert len(payloads) == 1

    def test_summary_file_excluded_from_iteration(self, tmp_path):
        write_taskgen_sidecar(_make_taskgen(person_id="a_0001"), tmp_path)
        # Drop a _summary.json next to the payload; must not be parsed.
        write_summary_sidecar({"x": 1}, tmp_path)
        payloads = aggregate_taskgen_sidecars(tmp_path)
        assert len(payloads) == 1


# ---------------------------------------------------------------------------
# aggregate_full_pipeline; by_stage + by_persona roll-up
# ---------------------------------------------------------------------------


def _seed_taskgen_pair(tmp_path: Path) -> None:
    """Two persons in the same persona, with non-zero token + wall."""
    for pid in ("a_persona_0001", "a_persona_0002"):
        t = _make_taskgen(person_id=pid)
        t.record_llm_call(
            stage="fetch", input_tokens=100, output_tokens=20, wall_time_seconds=2.0
        )
        t.record_fetch_summary(
            num_tasks_requested=2,
            num_tasks_accepted=2,
            per_attempt_fabricated=[1],
            per_attempt_duplicate=[0],
            per_attempt_parse_failed=[False],
        )
        t.record_task(
            TaskTelemetryRow(
                label="x",
                ontology_uri="https://ex.org/task/x",
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
        write_taskgen_sidecar(t, tmp_path)


def _seed_augment_pair(tmp_path: Path, *, method: str = "llm_agent") -> None:
    for pid in ("a_persona_0001", "a_persona_0002"):
        t = _make_augment(person_id=pid, method=method)
        t.wall_time_seconds = 5.0
        t.n_tasks_attempted = 4
        t.n_tasks_placed = 3
        t.n_tasks_dropped = 1
        if method == "llm_agent":
            t.record_llm_call(input_tokens=400, output_tokens=80, wall_time_seconds=4.0)
        write_augment_sidecar(t, tmp_path)


def _seed_evaluate_pair(tmp_path: Path) -> None:
    for pid in ("a_persona_0001", "a_persona_0002"):
        t = _make_evaluate(person_id=pid, judge_model="gpt-4o-mini")
        t.wall_time_seconds = 1.5
        t.record_semantic_lookup(wall_time_seconds=0.5)
        t.record_uri_validation(wall_time_seconds=0.2)
        t.record_judge_call(input_tokens=50, output_tokens=10, wall_time_seconds=0.3)
        write_evaluate_sidecar(t, tmp_path)


class TestAggregateFullPipeline:
    def test_empty_dirs_return_zero_persons(self, tmp_path):
        out = aggregate_full_pipeline(
            tasks_dir=tmp_path / "tasks",
            augmented_dir=tmp_path / "aug",
            evaluation_dir=tmp_path / "eval",
            scenario_id="x",
            method="greedy",
        )
        assert out["n_persons"] == 0
        assert out["n_personas"] == 0
        assert out["wall_time_seconds_total"] == 0.0
        assert out["estimated_usd_total"] is None
        # Per-stage skeletons present even when empty.
        assert "task_generation" in out["by_stage"]
        assert "augmentation" in out["by_stage"]
        assert "evaluation" in out["by_stage"]
        assert out["by_persona"] == {}

    def test_combines_all_three_stages(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        aug_dir = tmp_path / "aug"
        eval_dir = tmp_path / "eval"
        _seed_taskgen_pair(tasks_dir)
        _seed_augment_pair(aug_dir)
        _seed_evaluate_pair(eval_dir)

        out = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=aug_dir,
            evaluation_dir=eval_dir,
            scenario_id="nutrition_l1",
            method="llm_agent",
        )
        assert out["scenario_id"] == "nutrition_l1"
        assert out["method"] == "llm_agent"
        assert out["n_persons"] == 2
        assert out["n_personas"] == 1  # both persons share one persona
        # Wall = task_gen 2.0×2 + augment 5.0×2 + eval 1.5×2 = 17.0.
        assert out["wall_time_seconds_total"] == pytest.approx(17.0)
        # Per-person avg = 17/2 = 8.5; per-persona avg = 17/1 = 17.0.
        assert out["wall_time_seconds_avg_per_person"] == pytest.approx(8.5)
        assert out["wall_time_seconds_avg_per_persona"] == pytest.approx(17.0)

    def test_by_stage_has_per_stage_extras(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        aug_dir = tmp_path / "aug"
        eval_dir = tmp_path / "eval"
        _seed_taskgen_pair(tasks_dir)
        _seed_augment_pair(aug_dir)
        _seed_evaluate_pair(eval_dir)

        out = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=aug_dir,
            evaluation_dir=eval_dir,
            scenario_id="x",
            method="llm_agent",
        )
        aug_stage = out["by_stage"]["augmentation"]
        # 2 persons × {attempted=4, placed=3, dropped=1}.
        assert aug_stage["n_tasks_attempted"] == 8
        assert aug_stage["n_tasks_placed"] == 6
        assert aug_stage["n_tasks_dropped"] == 2

        eval_stage = out["by_stage"]["evaluation"]
        # 1 semantic lookup + 1 URI validation per person × 2 persons.
        assert eval_stage["embedding_lookups_total"] == 2
        assert eval_stage["uri_validations_total"] == 2

    def test_by_persona_groups_two_persons_into_one_cohort(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _seed_taskgen_pair(tasks_dir)
        out = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=None,
            evaluation_dir=None,
            scenario_id="x",
            method="greedy",
        )
        assert "a_persona" in out["by_persona"]
        cohort = out["by_persona"]["a_persona"]
        assert cohort["n_persons"] == 2
        # Each person spent 2.0s in task gen to cohort total = 4.0s.
        assert cohort["wall_time_seconds_total"] == pytest.approx(4.0)
        assert cohort["wall_time_seconds_avg_per_person"] == pytest.approx(2.0)
        # Both persons have known cost to cohort cost is non-null.
        assert cohort["estimated_usd_total"] is not None

    def test_by_persona_separates_distinct_cohorts(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        # Person from persona A.
        a = _make_taskgen(person_id="cohort_a_0001")
        a.record_llm_call(
            stage="fetch", input_tokens=100, output_tokens=10, wall_time_seconds=1.0
        )
        write_taskgen_sidecar(a, tasks_dir)
        # Person from persona B.
        b = _make_taskgen(person_id="cohort_b_0001")
        b.record_llm_call(
            stage="fetch", input_tokens=200, output_tokens=20, wall_time_seconds=3.0
        )
        write_taskgen_sidecar(b, tasks_dir)

        out = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=None,
            evaluation_dir=None,
            scenario_id="x",
            method="greedy",
        )
        assert set(out["by_persona"]) == {"cohort_a", "cohort_b"}
        assert out["by_persona"]["cohort_a"][
            "wall_time_seconds_total"
        ] == pytest.approx(1.0)
        assert out["by_persona"]["cohort_b"][
            "wall_time_seconds_total"
        ] == pytest.approx(3.0)

    def test_paraphrase_gate_failures_carried_from_taskgen(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        t = _make_taskgen(person_id="a_0001")
        t.record_task(
            TaskTelemetryRow(
                label="x",
                ontology_uri=None,
                fetch_attempt=1,
                paraphrase_used="canonical",
                length_delta=None,
                similarity=None,
                missing_emojis={},
                extra_emojis={},
                failed_gate="emoji",
                canonical_description="",
                personalized_description="",
            )
        )
        write_taskgen_sidecar(t, tasks_dir)
        out = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=None,
            evaluation_dir=None,
            scenario_id="x",
            method="greedy",
        )
        assert out["gate_failures"]["emoji"] == 1
        assert out["paraphrase_acceptance_rate"] == pytest.approx(0.0)

    def test_short_fetched_persons_counted(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        t = _make_taskgen(person_id="a_0001")
        t.record_fetch_summary(
            num_tasks_requested=5,
            num_tasks_accepted=3,
            per_attempt_fabricated=[2],
            per_attempt_duplicate=[0],
            per_attempt_parse_failed=[False],
        )
        write_taskgen_sidecar(t, tasks_dir)
        out = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=None,
            evaluation_dir=None,
            scenario_id="x",
            method="greedy",
        )
        assert out["persons_short_fetched"] == 1

    def test_estimated_usd_null_when_all_unknown(self, tmp_path):
        """Every model unknown (incl. embedder) to top-level cost is null,
        not 0.0; distinguishes "no data" from "$0.00 of data"."""
        tasks_dir = tmp_path / "tasks"
        t = _make_taskgen(
            person_id="a_0001",
            fetch_model="unknown",
            paraphrase_model="unknown",
            embedder_model="unknown",
        )
        t.record_llm_call(
            stage="fetch", input_tokens=10, output_tokens=2, wall_time_seconds=0.1
        )
        write_taskgen_sidecar(t, tasks_dir)
        out = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=None,
            evaluation_dir=None,
            scenario_id="x",
            method="greedy",
        )
        assert out["estimated_usd_total"] is None

    def test_greedy_vs_llm_agent_return_distinct_augment_costs(self, tmp_path):
        """Same task-gen sidecars; different augment sidecars (one with
        LLM tokens, one without) must produce distinct full-pipeline
        token totals; the original `n_personas: 12` regression."""
        tasks_dir = tmp_path / "tasks"
        _seed_taskgen_pair(tasks_dir)

        greedy_aug = tmp_path / "greedy_aug"
        llm_aug = tmp_path / "llm_aug"
        _seed_augment_pair(greedy_aug, method="greedy")
        _seed_augment_pair(llm_aug, method="llm_agent")

        greedy = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=greedy_aug,
            evaluation_dir=None,
            scenario_id="x",
            method="greedy",
        )
        llm = aggregate_full_pipeline(
            tasks_dir=tasks_dir,
            augmented_dir=llm_aug,
            evaluation_dir=None,
            scenario_id="x",
            method="llm_agent",
        )
        assert greedy["tokens_total"] < llm["tokens_total"]
        # Greedy augmentation contributes zero LLM tokens.
        assert greedy["by_stage"]["augmentation"]["tokens_total"] == 0
        assert llm["by_stage"]["augmentation"]["tokens_total"] > 0


# ---------------------------------------------------------------------------
# _by_persona_breakdown; direct
# ---------------------------------------------------------------------------


class TestByPersonaBreakdown:
    def test_empty_inputs_return_empty_dict(self):
        assert _by_persona_breakdown([], [], []) == {}

    def test_persona_id_blank_drops_entry(self):
        # Payloads without persona_id are skipped (the bump function
        # short-circuits on empty persona).
        payloads = [
            {"person_id": "x", "persona_id": "", "wall_time_seconds_total": 1.0}
        ]
        assert _by_persona_breakdown(payloads, [], []) == {}


# ---------------------------------------------------------------------------
# _empty_stage skeletons
# ---------------------------------------------------------------------------


class TestEmptyStage:
    def test_task_generation_skeleton(self):
        s = _empty_stage("task_generation")
        assert s["wall_time_seconds_total"] == 0.0
        assert s["estimated_usd_total"] is None

    def test_augmentation_extra_keys(self):
        s = _empty_stage("augmentation")
        assert s["n_tasks_placed"] == 0
        assert s["n_tasks_dropped"] == 0

    def test_evaluation_extra_keys(self):
        s = _empty_stage("evaluation")
        assert s["embedding_lookups_total"] == 0
        assert s["uri_validations_total"] == 0


# ---------------------------------------------------------------------------
# write_summary_sidecar
# ---------------------------------------------------------------------------


class TestWriteSummarySidecar:
    def test_writes_under_telemetry_dir(self, tmp_path):
        out = write_summary_sidecar({"scenario_id": "x"}, tmp_path)
        assert out.parent.name == "_telemetry"
        assert out.name == "_summary.json"
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["scenario_id"] == "x"


# ---------------------------------------------------------------------------
# _Timer
# ---------------------------------------------------------------------------


def test_timer_records_elapsed():
    with _Timer() as timer:
        time.sleep(0.001)
    assert timer.elapsed > 0.0


class TestReasoningTokenFlow:
    def test_record_llm_call_accumulates_reasoning(self):
        from src.scripts.scenarios.metrics.telemetry import AugmentTelemetry

        telem = AugmentTelemetry(
            person_id="p_0000",
            scenario_id="s",
            method="llm_agent",
            provider="openai",
            model="gpt-5-mini",
        )
        telem.record_llm_call(
            input_tokens=100,
            output_tokens=1_000_000,
            wall_time_seconds=1.0,
            reasoning_tokens=500_000,
        )
        d = telem.to_dict()
        assert d["llm_calls"]["reasoning_tokens"] == 500_000
        # 0.5M reasoning tokens at the gpt-5-mini output rate ($2/M).
        assert d["llm_calls"]["estimated_reasoning_usd"] == pytest.approx(1.0)
        # The full estimate covers reasoning (subset of output tokens).
        assert d["llm_calls"]["estimated_usd"] == pytest.approx(
            (100 / 1e6) * 0.25 + 2.0
        )

    def test_zero_reasoning_renders_none_cost(self):
        from src.scripts.scenarios.metrics.telemetry import AugmentTelemetry

        telem = AugmentTelemetry(
            person_id="p_0000",
            scenario_id="s",
            method="llm_agent",
            provider="openai",
            model="gpt-4o-mini",
        )
        telem.record_llm_call(input_tokens=100, output_tokens=50, wall_time_seconds=0.1)
        d = telem.to_dict()
        assert d["llm_calls"]["reasoning_tokens"] == 0
        assert d["llm_calls"]["estimated_reasoning_usd"] is None

    def test_stage_aggregate_sums_reasoning(self, tmp_path):
        from src.scripts.scenarios.metrics.telemetry import (
            AugmentTelemetry,
            aggregate_full_pipeline,
            write_augment_sidecar,
        )

        for idx, reasoning in enumerate((200_000, 300_000)):
            telem = AugmentTelemetry(
                person_id=f"p_{idx:04d}",
                scenario_id="s",
                method="llm_agent",
                provider="openai",
                model="gpt-5-mini",
            )
            telem.record_llm_call(
                input_tokens=10,
                output_tokens=reasoning + 10,
                wall_time_seconds=0.5,
                reasoning_tokens=reasoning,
            )
            write_augment_sidecar(telem, tmp_path)
        summary = aggregate_full_pipeline(
            tasks_dir=None,
            augmented_dir=tmp_path,
            evaluation_dir=None,
            scenario_id="s",
            method="llm_agent",
        )
        stage = summary["by_stage"]["augmentation"]
        assert stage["reasoning_tokens_total"] == 500_000
        assert stage["estimated_reasoning_usd_total"] == pytest.approx(1.0)

    def test_record_judge_call_accumulates_reasoning(self):
        from src.scripts.scenarios.metrics.telemetry import EvaluateTelemetry

        telem = EvaluateTelemetry(
            person_id="p_0000",
            scenario_id="s",
            method="llm_agent",
            provider="openai",
            judge_model="gpt-4o-mini",
        )
        telem.record_judge_call(
            input_tokens=10,
            output_tokens=20,
            wall_time_seconds=0.1,
            reasoning_tokens=5,
        )
        assert telem.to_dict()["judge_calls"]["reasoning_tokens"] == 5


class TestLoadPricing:
    """`estimate_usd` reads only `PRICING_FILE`; no env knob, no builtin dict."""

    @pytest.fixture(autouse=True)
    def _restore_cache(self):
        """Clear the cache around each test so file swaps take effect."""
        import src.scripts.scenarios.metrics.telemetry as tel

        tel._PRICING = None
        yield
        tel._PRICING = None
        tel.load_pricing(force=True)

    def test_reads_prices_from_the_file(self, tmp_path, monkeypatch):
        import src.scripts.scenarios.metrics.telemetry as tel

        price_file = tmp_path / "prices.json"
        price_file.write_text(
            json.dumps({"openai": {"gpt-4o-mini": {"in": 1.0, "out": 1.0}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(tel, "PRICING_FILE", price_file)
        tel.load_pricing(force=True)
        usd = tel.estimate_usd(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert usd == pytest.approx(2.0)

    def test_force_rereads_after_file_rewrite(self, tmp_path, monkeypatch):
        import src.scripts.scenarios.metrics.telemetry as tel

        price_file = tmp_path / "prices.json"
        price_file.write_text(
            json.dumps({"openai": {"m": {"in": 1.0, "out": 0.0}}}), encoding="utf-8"
        )
        monkeypatch.setattr(tel, "PRICING_FILE", price_file)
        assert tel.load_pricing(force=True)["openai"]["m"]["in"] == 1.0
        price_file.write_text(
            json.dumps({"openai": {"m": {"in": 9.0, "out": 0.0}}}), encoding="utf-8"
        )
        assert tel.load_pricing()["openai"]["m"]["in"] == 1.0  # cached
        assert tel.load_pricing(force=True)["openai"]["m"]["in"] == 9.0

    def test_unknown_model_returns_none(self, tmp_path, monkeypatch):
        import src.scripts.scenarios.metrics.telemetry as tel

        price_file = tmp_path / "prices.json"
        price_file.write_text(
            json.dumps({"openai": {"gpt-4o-mini": {"in": 1.0, "out": 1.0}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(tel, "PRICING_FILE", price_file)
        tel.load_pricing(force=True)
        assert (
            tel.estimate_usd(
                provider="openai",
                model="not-tracked",
                input_tokens=100,
                output_tokens=50,
            )
            is None
        )

    def test_missing_file_returns_empty_table_and_na(self, tmp_path, monkeypatch):
        import src.scripts.scenarios.metrics.telemetry as tel

        monkeypatch.setattr(tel, "PRICING_FILE", tmp_path / "absent.json")
        assert tel.load_pricing(force=True) == {}
        assert (
            tel.estimate_usd(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=1_000_000,
                output_tokens=0,
            )
            is None
        )

    def test_malformed_entry_returns_none(self, tmp_path, monkeypatch):
        import src.scripts.scenarios.metrics.telemetry as tel

        price_file = tmp_path / "prices.json"
        price_file.write_text(
            json.dumps({"openai": {"m": {"in": 1.0}}}), encoding="utf-8"
        )
        monkeypatch.setattr(tel, "PRICING_FILE", price_file)
        tel.load_pricing(force=True)
        assert (
            tel.estimate_usd(
                provider="openai", model="m", input_tokens=1_000_000, output_tokens=0
            )
            is None
        )

    @pytest.mark.parametrize(
        "bad", [{"in": "abc", "out": 2.0}, {"in": None, "out": 2.0}]
    )
    def test_non_numeric_value_returns_none_not_crash(self, tmp_path, monkeypatch, bad):
        """A hand-corrupted price must degrade to n/a, never raise."""
        import src.scripts.scenarios.metrics.telemetry as tel

        price_file = tmp_path / "prices.json"
        price_file.write_text(json.dumps({"openai": {"m": bad}}), encoding="utf-8")
        monkeypatch.setattr(tel, "PRICING_FILE", price_file)
        tel.load_pricing(force=True)
        assert (
            tel.estimate_usd(
                provider="openai", model="m", input_tokens=1_000_000, output_tokens=10
            )
            is None
        )
