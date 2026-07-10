"""Unit tests for the multi-scenario aggregated benchmark report.

The writer is a pure file-in / Markdown-out aggregator, so every test
either drives `build_benchmark_markdown` with synthetic dicts or stages
JSON sidecars on disk and exercises `discover_scenario_runs` /
`write_benchmark_report` end-to-end.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from src.scripts.scenarios.config.schema import (
    AugmentationMethodConfig,
    ExperimentScenariosConfig,
    LLMAgentConfig,
    LLMModelConfig,
    ScenarioDefinition,
)
from src.scripts.scenarios.export.benchmark_report import (
    BENCHMARK_REPORT_BASENAME,
    ONTOLOGY_REPORT_BASENAME,
    PREFERENCE_REPORT_BASENAME,
    TELEMETRY_REPORT_BASENAME,
    TOTAL_REPORT_BASENAME,
    _delta_cell,
    _experiment_badge,
    _format_cross_augmenter_weekly,
    _format_legend,
    _format_per_person_distribution,
    _format_table,
    _gain_cell,
    _int_cell,
    _pct_cell,
    _seconds_cell,
    _shields_escape,
    _usd_cell,
    build_benchmark_markdown,
    discover_scenario_runs,
    read_scenario_artifacts,
    resolve_stage_models,
    write_benchmark_report,
)


def _weekly_artifact(by_person_gains: dict[str, list[float]]) -> dict:
    """Build a `weekly` artifact (weeks aggregate + by_person) for tests."""
    by_person = {
        pid: [
            {
                "week_index": i + 1,
                "week_start": f"2026-06-{1 + 7 * i:02d}",
                "weighted_gain": g,
            }
            for i, g in enumerate(gains)
        ]
        for pid, gains in by_person_gains.items()
    }
    n_weeks = max(len(g) for g in by_person_gains.values())
    weeks = []
    for w in range(1, n_weeks + 1):
        vals = [g[w - 1] for g in by_person_gains.values() if len(g) >= w]
        weeks.append(
            {
                "week_index": w,
                "week_start": f"2026-06-{1 + 7 * (w - 1):02d}",
                "avg_weighted_gain": sum(vals) / len(vals),
                "scored_persons": len(vals),
            }
        )
    return {"weeks": weeks, "by_person": by_person}


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


class TestCellFormatters:
    def test_gain_cell_formats_floats_to_4dp(self):
        assert _gain_cell(0.7351) == "0.7351"

    def test_gain_cell_handles_none(self):
        assert _gain_cell(None) == "n/a"

    def test_usd_cell(self):
        assert _usd_cell(1.234) == "$1.2340"
        assert _usd_cell(None) == "n/a"
        assert _usd_cell("not numeric") == "n/a"  # type: ignore[arg-type]

    def test_int_cell(self):
        assert _int_cell(5) == "5"
        assert _int_cell(5.0) == "5"
        assert _int_cell(None) == "n/a"

    def test_seconds_cell(self):
        assert _seconds_cell(12.34) == "12.3s"
        assert _seconds_cell(None) == "n/a"

    def test_pct_cell(self):
        assert _pct_cell(0.123) == "12.3%"
        assert _pct_cell(None) == "n/a"

    def test_delta_cell_positive_gets_explicit_plus_sign(self):
        assert _delta_cell(0.6, 0.5) == "+0.1000"

    def test_delta_cell_negative_keeps_minus(self):
        assert _delta_cell(0.4, 0.5) == "-0.1000"

    def test_delta_cell_handles_none_either_side(self):
        assert _delta_cell(None, 0.5) == "n/a"
        assert _delta_cell(0.5, None) == "n/a"

    def test_format_table_renders_separator(self):
        lines = _format_table(("col1", "col2"), [("a", "b"), ("c", "d")])
        assert lines[0] == "| col1 | col2 |"
        assert lines[1] == "|---|---|"
        assert lines[2] == "| a | b |"


# ---------------------------------------------------------------------------
# Disk readers
# ---------------------------------------------------------------------------


class TestReadScenarioArtifacts:
    def test_reads_all_four_sidecars(self, tmp_path: Path):
        for name, payload in (
            (TOTAL_REPORT_BASENAME, {"average_total_gain": 0.5}),
            (TELEMETRY_REPORT_BASENAME, {"n_persons": 3}),
            (ONTOLOGY_REPORT_BASENAME, {"total": 9}),
            (PREFERENCE_REPORT_BASENAME, {"start": {"mean_loss": 0.1}}),
        ):
            (tmp_path / f"{name}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        out = read_scenario_artifacts(tmp_path)
        assert out["total"]["average_total_gain"] == 0.5
        assert out["telemetry"]["n_persons"] == 3
        assert out["grounding"]["total"] == 9
        assert out["preference"]["start"]["mean_loss"] == 0.1

    def test_missing_files_return_none(self, tmp_path: Path):
        out = read_scenario_artifacts(tmp_path)
        assert out == {
            "total": None,
            "telemetry": None,
            "grounding": None,
            "preference": None,
            "divide": None,
            "weekly": None,
        }

    def test_unparseable_file_returns_none(self, tmp_path: Path):
        (tmp_path / f"{TOTAL_REPORT_BASENAME}.json").write_text(
            "not json", encoding="utf-8"
        )
        out = read_scenario_artifacts(tmp_path)
        assert out["total"] is None


class TestDiscoverScenarioRuns:
    def _stage(self, root: Path, sid: str, method: str, total_gain: float) -> None:
        eval_dir = root / "scenarios" / sid / method / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / f"{TOTAL_REPORT_BASENAME}.json").write_text(
            json.dumps({"average_total_gain": total_gain}),
            encoding="utf-8",
        )

    def test_returns_empty_when_scenarios_dir_missing(self, tmp_path: Path):
        assert discover_scenario_runs(tmp_path) == []

    def test_walks_every_scenario_method_dir_in_sorted_order(self, tmp_path: Path):
        self._stage(tmp_path, "sB", "llm_agent", 0.6)
        self._stage(tmp_path, "sA", "greedy", 0.5)
        self._stage(tmp_path, "sA", "llm_agent", 0.7)
        runs = discover_scenario_runs(tmp_path)
        ids = [(sid, method) for sid, method, _ in runs]
        assert ids == [("sA", "greedy"), ("sA", "llm_agent"), ("sB", "llm_agent")]

    def test_scenario_method_pairs_filter_restricts_output(self, tmp_path: Path):
        self._stage(tmp_path, "sA", "greedy", 0.5)
        self._stage(tmp_path, "sA", "llm_agent", 0.7)
        self._stage(tmp_path, "sB", "llm_agent", 0.6)
        runs = discover_scenario_runs(
            tmp_path,
            scenario_method_pairs=[("sB", "llm_agent")],
        )
        assert [(sid, method) for sid, method, _ in runs] == [("sB", "llm_agent")]

    def test_pair_with_no_eval_dir_returns_empty_artifacts(self, tmp_path: Path):
        runs = discover_scenario_runs(
            tmp_path,
            scenario_method_pairs=[("ghost", "llm_agent")],
        )
        assert len(runs) == 1
        assert runs[0][2] == {
            "total": None,
            "telemetry": None,
            "grounding": None,
            "preference": None,
            "divide": None,
            "weekly": None,
        }

    def test_skips_non_directory_entries_under_scenarios_root(self, tmp_path: Path):
        """A stray file (e.g. accidentally committed `.DS_Store`) under
        `scenarios/` must not crash the walker; it gets skipped."""
        scenarios_root = tmp_path / "scenarios"
        scenarios_root.mkdir()
        # Stray file at the top level …
        (scenarios_root / ".DS_Store").write_text("noise", encoding="utf-8")
        self._stage(tmp_path, "real", "llm_agent", 0.5)
        # … and a stray file inside a real scenario dir.
        (scenarios_root / "real" / "README.txt").write_text("ignored", encoding="utf-8")
        runs = discover_scenario_runs(tmp_path)
        assert [(sid, method) for sid, method, _ in runs] == [("real", "llm_agent")]


# ---------------------------------------------------------------------------
# resolve_stage_models
# ---------------------------------------------------------------------------


def _make_exp(
    *scenarios: ScenarioDefinition, experiment_id: str = "exp"
) -> ExperimentScenariosConfig:
    return ExperimentScenariosConfig.model_validate(
        {
            "experiment_id": experiment_id,
            "scenarios": [s.model_dump() for s in scenarios],
        }
    )


class TestResolveStageModels:
    def test_returns_env_default_when_cfg_is_none(self):
        out = resolve_stage_models(None, "any", "llm_agent")
        assert out == {
            "task_generator": "env-default",
            "augmenter": "env-default",
            "evaluator": "env-default",
        }

    def test_returns_env_default_when_scenario_id_unknown(self):
        exp = _make_exp(ScenarioDefinition(id="real"))
        assert (
            resolve_stage_models(exp, "ghost", "llm_agent")["task_generator"]
            == "env-default"
        )

    def test_picks_pinned_models(self):
        scenario = ScenarioDefinition(
            id="s1",
            task_generator_model=LLMModelConfig(model="tg-model"),
            evaluator_model=LLMModelConfig(model="eval-model"),
            augmentation=[
                AugmentationMethodConfig(
                    method="llm_agent",
                    llm_agent=LLMAgentConfig(model="aug-model"),
                )
            ],
        )
        exp = _make_exp(scenario)
        out = resolve_stage_models(exp, "s1", "llm_agent")
        assert out["task_generator"] == "tg-model"
        assert out["augmenter"] == "aug-model"
        assert out["evaluator"] == "eval-model"

    def test_unknown_method_keeps_env_default_for_augmenter(self):
        scenario = ScenarioDefinition(
            id="s1",
            augmentation=[AugmentationMethodConfig(method="greedy")],
        )
        exp = _make_exp(scenario)
        out = resolve_stage_models(exp, "s1", "llm_agent")
        # method `llm_agent` not in scenario to augmenter stays env-default.
        assert out["augmenter"] == "env-default"

    def test_greedy_method_does_not_read_llm_agent_model(self):
        """A greedy method has no LLM, so its augmenter label must NOT
        leak the unused `llm_agent` block's default model."""
        scenario = ScenarioDefinition(
            id="s1",
            augmentation=[AugmentationMethodConfig(method="greedy")],
        )
        exp = _make_exp(scenario)
        out = resolve_stage_models(exp, "s1", "greedy")
        assert out["augmenter"] == "env-default"


# ---------------------------------------------------------------------------
# build_benchmark_markdown
# ---------------------------------------------------------------------------


def _full_artifacts(
    *,
    avg_total: float | None,
    cov: float = 0.9,
    cal: float = 0.8,
    usd: float | None = 0.05,
    persons: int = 10,
    grounded: int = 9,
    total_tasks: int = 10,
) -> dict:
    return {
        "total": {
            "average_total_gain": avg_total,
            "scored_persons": persons,
            "empty_plan_persons": 0,
            "average_gains": {
                "recommended_task_coverage": cov,
                "task_event_and_task_task_temporal_relations": cal,
                "user_preference_deviation": 0.7,
                "intensive_task_dispersion": 0.6,
                "semantic_coscheduling_merge": 0.5,
                "recommended_task_spread": 0.4,
                "dividable_task_split_reward": None,
            },
        },
        "telemetry": {
            "n_persons": persons,
            "estimated_usd_total": usd,
            "wall_time_seconds_total": 12.3,
            "by_stage": {
                "task_generation": {
                    "wall_time_seconds_total": 3.0,
                    "tokens_total": 1000,
                    "estimated_usd_total": 0.01,
                },
                "augmentation": {
                    "wall_time_seconds_total": 6.0,
                    "tokens_total": 2000,
                    "estimated_usd_total": 0.03,
                },
                "evaluation": {
                    "wall_time_seconds_total": 3.3,
                    "tokens_total": 500,
                    "estimated_usd_total": 0.01,
                },
            },
        },
        "grounding": {
            "total": total_tasks,
            "grounded": grounded,
            "ratio": grounded / total_tasks,
            "verified": grounded - 1,
            "verified_ratio": (grounded - 1) / total_tasks,
            "persons_short_fetched": 1,
            "persons_with_unverified": 0,
        },
        "preference": {
            "start": {"mean_loss": 0.12, "applicable_tasks": 5},
            "epoch": {"mean_loss": 0.20, "applicable_tasks": 3},
        },
        "divide": None,
    }


class TestBuildBenchmarkMarkdown:
    def test_no_runs_returns_placeholder(self):
        md = build_benchmark_markdown("exp_x", [])
        assert "# Benchmark report: `exp_x`" in md
        assert "No evaluation sidecars" in md

    def test_warns_loudly_when_runs_exist_but_every_total_is_none(self):
        """Regression for the 2026-05-18 silent no-op symptom: the user's
        first `report` after a failed augment produced a wall of n/a
        cells with no explanation. The report now leads with a callout
        when every run is missing its `total_scheduling_gain.json`."""
        runs = [
            (
                "oneshot_gpt_4o_mini_l234",
                "llm_agent",
                {
                    "total": None,
                    "telemetry": None,
                    "grounding": None,
                    "preference": None,
                },
            ),
            (
                "oneshot_gpt_4_1_mini_l234",
                "llm_agent",
                {
                    "total": None,
                    "telemetry": None,
                    "grounding": None,
                    "preference": None,
                },
            ),
        ]
        md = build_benchmark_markdown("exp_x", runs)
        assert "**No evaluation data found" in md
        # The callout must name every affected run so the reader knows
        # which scenarios to re-augment.
        assert "oneshot_gpt_4o_mini_l234 / llm_agent" in md
        assert "oneshot_gpt_4_1_mini_l234 / llm_agent" in md
        # Recovery hint must mention `augment` then `evaluate`; the
        # exact two commands the reader needs to run next.
        assert "re-run `augment` then `evaluate`" in md

    def test_no_warning_when_at_least_one_run_has_total(self):
        """Partial data is still useful; no callout, the n/a cells in
        the missing rows speak for themselves alongside the real ones."""
        runs = [
            ("s1", "llm_agent", _full_artifacts(avg_total=0.65)),
            (
                "s2",
                "llm_agent",
                {
                    "total": None,
                    "telemetry": None,
                    "grounding": None,
                    "preference": None,
                },
            ),
        ]
        md = build_benchmark_markdown("e", runs)
        assert "**No evaluation data found" not in md

    def test_full_report_has_every_top_level_section(self):
        runs = [
            ("s1", "llm_agent", _full_artifacts(avg_total=0.65)),
            ("s2", "llm_agent", _full_artifacts(avg_total=0.70, cov=0.95)),
        ]
        md = build_benchmark_markdown(
            "exp_y",
            runs,
            generated_at=_dt.datetime(2026, 5, 18, tzinfo=_dt.timezone.utc),
        )
        assert "# Benchmark report: `exp_y`" in md
        # Run summary, scheduling gain, cost, grounding, preference, pairwise
        assert "## Run summary" in md
        assert "## Scheduling gain" in md
        assert "## Stage cost & latency" in md
        assert "## Ontology grounding" in md
        assert "## Preference breakdown" in md
        assert "## Pairwise A/B" in md
        assert "_2 `(scenario_id, method)` run(s) aggregated._" in md

    def test_scheduling_gain_table_includes_every_component(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.65))]
        md = build_benchmark_markdown("e", runs)
        for label in (
            "G_cov",
            "G_cal",
            "G_pref",
            "G_disp",
            "G_merge",
            "G_spread",
            "G_divide",
        ):
            assert label in md
        # `divide` was None in the synthetic artifacts to renders as n/a.
        assert "n/a" in md

    def test_stage_cost_table_lists_every_stage(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.65))]
        md = build_benchmark_markdown("e", runs)
        assert "Task generation wall" in md
        assert "Augmentation wall" in md
        assert "Evaluation wall" in md
        assert "Total cost" in md
        # Top-level estimated_usd_total wins for the Total cost cell.
        assert "$0.0500" in md

    def test_ontology_grounding_table_renders_ratios_and_zero_unverified(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.65))]
        md = build_benchmark_markdown("e", runs)
        # 9/10 grounded to 90.0%; verified 8/10 to 80.0%.
        assert "9 (90.0%)" in md
        assert "8 (80.0%)" in md

    def test_preference_breakdown_renders_union_of_legs(self):
        runs = [
            ("s1", "llm_agent", _full_artifacts(avg_total=0.65)),
            (
                "s2",
                "llm_agent",
                {
                    "total": None,
                    "telemetry": None,
                    "grounding": None,
                    # s2 only declares `epoch`; `start` cell must be n/a
                    "preference": {"epoch": {"mean_loss": 0.30}},
                },
            ),
        ]
        md = build_benchmark_markdown("e", runs)
        # `start` column appears even though s2 omitted it.
        assert "| start |" in md or "| start " in md
        # s2 row's `start` is n/a; s1's is the value 0.1200.
        assert "0.1200 (n=5)" in md

    def test_preference_section_placeholder_when_no_breakdowns(self):
        runs = [
            (
                "s1",
                "llm_agent",
                {
                    "total": {"average_total_gain": 0.5, "average_gains": {}},
                    "telemetry": None,
                    "grounding": None,
                    "preference": None,
                },
            )
        ]
        md = build_benchmark_markdown("e", runs)
        assert "_No `preference_breakdown.json` sidecars were found._" in md

    def test_divide_section_placeholder_when_no_signal(self):
        runs = [
            (
                "s1",
                "llm_agent",
                {
                    "total": {"average_total_gain": 0.5, "average_gains": {}},
                    "telemetry": None,
                    "grounding": None,
                    "preference": None,
                    "divide": None,
                },
            )
        ]
        md = build_benchmark_markdown("e", runs)
        assert (
            "_No `divide_breakdown.json` sidecars with dividable signal "
            "were found._" in md
        )

    def test_divide_section_renders_verdict_columns_when_signal(self):
        divide_payload = {
            "applicable_persons": 2,
            "applicable_weeks": 3,
            "applicable_buckets": 4,
            "verdict_counts": {
                "divided_valid": 2,
                "not_divided": 1,
                "divided_invalid_pieces_too_long": 0,
                "divided_invalid_sum_too_low": 0,
                "divided_invalid_sum_too_high": 1,
            },
            "per_label": {
                "walk": {
                    "buckets": 4,
                    "divided_valid": 2,
                    "mean_sum_minutes": 90.0,
                    "mean_pieces_per_bucket": 2.0,
                }
            },
        }
        artifacts = _full_artifacts(avg_total=0.5)
        artifacts["divide"] = divide_payload
        runs = [("s1", "llm_agent", artifacts)]
        md = build_benchmark_markdown("e", runs)
        assert "## Divide breakdown" in md
        assert "divided_valid" in md
        assert "2 (50.0%)" in md  # 2 / 4
        assert "1 (25.0%)" in md  # 1 / 4

    def test_divide_section_uses_na_for_runs_without_buckets(self):
        empty_divide = {
            "applicable_persons": 0,
            "applicable_weeks": 0,
            "applicable_buckets": 0,
            "verdict_counts": {
                "divided_valid": 0,
                "not_divided": 0,
                "divided_invalid_pieces_too_long": 0,
                "divided_invalid_sum_too_low": 0,
                "divided_invalid_sum_too_high": 0,
            },
            "per_label": {},
        }
        nonzero_divide = {
            "applicable_persons": 1,
            "applicable_weeks": 1,
            "applicable_buckets": 2,
            "verdict_counts": {
                "divided_valid": 1,
                "not_divided": 1,
                "divided_invalid_pieces_too_long": 0,
                "divided_invalid_sum_too_low": 0,
                "divided_invalid_sum_too_high": 0,
            },
            "per_label": {},
        }
        art_a = _full_artifacts(avg_total=0.4)
        art_a["divide"] = empty_divide
        art_b = _full_artifacts(avg_total=0.6)
        art_b["divide"] = nonzero_divide
        runs = [("s1", "llm_agent", art_a), ("s2", "llm_agent", art_b)]
        md = build_benchmark_markdown("e", runs)
        assert "## Divide breakdown" in md
        # The s1 row should carry n/a cells.
        assert "| n/a | n/a | n/a | n/a | n/a | n/a |" in md
        # The s2 row carries the verdict counts.
        assert "1 (50.0%)" in md

    def test_pairwise_section_emits_one_block_per_non_baseline(self):
        runs = [
            ("s1", "llm_agent", _full_artifacts(avg_total=0.50)),
            ("s2", "llm_agent", _full_artifacts(avg_total=0.65)),
            ("s3", "llm_agent", _full_artifacts(avg_total=0.70)),
        ]
        md = build_benchmark_markdown("e", runs)
        assert md.count("### `s2 / llm_agent`  vs  `s1 / llm_agent`") == 1
        assert md.count("### `s3 / llm_agent`  vs  `s1 / llm_agent`") == 1
        # Δ for s2 vs s1 on avg total gain is +0.1500.
        assert "+0.1500" in md

    def test_pairwise_baseline_override_swaps_the_anchor(self):
        runs = [
            ("s1", "llm_agent", _full_artifacts(avg_total=0.50)),
            ("s2", "llm_agent", _full_artifacts(avg_total=0.65)),
        ]
        md = build_benchmark_markdown("e", runs, baseline=("s2", "llm_agent"))
        assert "### `s1 / llm_agent`  vs  `s2 / llm_agent`" in md

    def test_pairwise_section_axis_label_lists_only_changed_stages(self):
        scenario_baseline = ScenarioDefinition(
            id="s_base",
            task_generator_model=LLMModelConfig(model="tg-4o"),
            evaluator_model=LLMModelConfig(model="eval-4o"),
            augmentation=[
                AugmentationMethodConfig(
                    method="llm_agent",
                    llm_agent=LLMAgentConfig(model="aug-4o"),
                )
            ],
        )
        scenario_alt = ScenarioDefinition(
            id="s_alt",
            task_generator_model=LLMModelConfig(model="tg-4o"),
            evaluator_model=LLMModelConfig(model="eval-4o"),
            augmentation=[
                AugmentationMethodConfig(
                    method="llm_agent",
                    llm_agent=LLMAgentConfig(model="aug-4_1"),
                )
            ],
        )
        exp = _make_exp(scenario_baseline, scenario_alt)
        runs = [
            ("s_base", "llm_agent", _full_artifacts(avg_total=0.5)),
            ("s_alt", "llm_agent", _full_artifacts(avg_total=0.55)),
        ]
        md = build_benchmark_markdown("e", runs, scenarios_cfg=exp)
        assert "axis: augmenter" in md
        assert "task_generator" not in md.split("axis: augmenter")[1].split("\n", 1)[0]

    def test_pairwise_section_when_no_axis_changed_renders_no_model_changes(self):
        """Two runs with identical model triples (e.g. greedy A vs greedy B
        with no per-stage overrides); axis label says so explicitly."""
        runs = [
            ("s1", "llm_agent", _full_artifacts(avg_total=0.5)),
            ("s2", "llm_agent", _full_artifacts(avg_total=0.6)),
        ]
        md = build_benchmark_markdown("e", runs, scenarios_cfg=None)
        assert "axis: no model changes" in md

    def test_pairwise_placeholder_when_only_one_run(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.5))]
        md = build_benchmark_markdown("e", runs)
        assert "Need at least two `(scenario_id, method)` runs" in md

    def test_pairwise_invalid_baseline_falls_back_to_first(self):
        runs = [
            ("s1", "llm_agent", _full_artifacts(avg_total=0.5)),
            ("s2", "llm_agent", _full_artifacts(avg_total=0.6)),
        ]
        md = build_benchmark_markdown("e", runs, baseline=("does_not", "exist"))
        # Falls back to first run = s1 as baseline.
        assert "### `s2 / llm_agent`  vs  `s1 / llm_agent`" in md


# ---------------------------------------------------------------------------
# write_benchmark_report (end-to-end on disk)
# ---------------------------------------------------------------------------


class TestWriteBenchmarkReport:
    def _stage(self, root: Path, sid: str, method: str, payload: dict) -> None:
        eval_dir = root / "scenarios" / sid / method / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for key, basename in (
            ("total", TOTAL_REPORT_BASENAME),
            ("telemetry", TELEMETRY_REPORT_BASENAME),
            ("grounding", ONTOLOGY_REPORT_BASENAME),
            ("preference", PREFERENCE_REPORT_BASENAME),
        ):
            data = payload.get(key)
            if data is not None:
                (eval_dir / f"{basename}.json").write_text(
                    json.dumps(data), encoding="utf-8"
                )

    def test_writes_under_experiment_dir(self, tmp_path: Path):
        self._stage(tmp_path, "s1", "llm_agent", _full_artifacts(avg_total=0.5))
        target = write_benchmark_report(tmp_path, "exp_z")
        assert target == tmp_path / f"{BENCHMARK_REPORT_BASENAME}.md"
        assert target.is_file()
        text = target.read_text(encoding="utf-8")
        assert "# Benchmark report: `exp_z`" in text
        assert "s1 / llm_agent" in text

    def test_overwrites_existing_report(self, tmp_path: Path):
        target = tmp_path / f"{BENCHMARK_REPORT_BASENAME}.md"
        target.write_text("STALE", encoding="utf-8")
        self._stage(tmp_path, "s1", "llm_agent", _full_artifacts(avg_total=0.5))
        write_benchmark_report(tmp_path, "exp_z")
        assert "STALE" not in target.read_text(encoding="utf-8")

    def test_scenario_method_pairs_restricts_runs(self, tmp_path: Path):
        self._stage(tmp_path, "s1", "llm_agent", _full_artifacts(avg_total=0.5))
        self._stage(tmp_path, "s2", "llm_agent", _full_artifacts(avg_total=0.6))
        write_benchmark_report(
            tmp_path,
            "exp_z",
            scenario_method_pairs=[("s2", "llm_agent")],
        )
        text = (tmp_path / f"{BENCHMARK_REPORT_BASENAME}.md").read_text(
            encoding="utf-8"
        )
        assert "s2 / llm_agent" in text
        assert "s1 / llm_agent" not in text

    def test_baseline_param_threads_through(self, tmp_path: Path):
        self._stage(tmp_path, "s1", "llm_agent", _full_artifacts(avg_total=0.5))
        self._stage(tmp_path, "s2", "llm_agent", _full_artifacts(avg_total=0.6))
        write_benchmark_report(
            tmp_path,
            "exp_z",
            baseline=("s2", "llm_agent"),
        )
        text = (tmp_path / f"{BENCHMARK_REPORT_BASENAME}.md").read_text(
            encoding="utf-8"
        )
        # When s2 is the baseline, the pairwise section anchors on it.
        assert "### `s1 / llm_agent`  vs  `s2 / llm_agent`" in text


# ---------------------------------------------------------------------------
# method_acronym + compact_label + resolve_method_acronym
# ---------------------------------------------------------------------------


from src.scripts.scenarios.export.benchmark_report import (  # noqa: E402
    compact_label,
    method_acronym,
    resolve_method_acronym,
)


class TestMethodAcronym:
    def test_known_method_prompt_pairs(self):
        assert method_acronym("llm_agent", "augment_oneshot") == "SAP"
        assert method_acronym("llm_agent", "augment_agent") == "ITA"
        assert method_acronym("greedy", None) == "GRD"
        assert method_acronym("rl", None) == "RL"

    def test_method_with_unknown_prompt_falls_back_to_method_only(self):
        # `llm_agent` + a brand-new prompt template; fall back to the
        # method-only key. None of the prompt-less methods has its
        # own entry under a stray prompt, so this currently returns the
        # uppercased method (the deepest fallback).
        out = method_acronym("llm_agent", "future_prompt")
        # `llm_agent` doesn't have a None-prompt entry, so deep fallback.
        assert out == "LLMAGENT"

    def test_unknown_method_renders_uppercased(self):
        assert method_acronym("custom_method", None) == "CUSTOMMETHOD"

    def test_known_method_with_stray_prompt_falls_back_to_method_only(self):
        """`greedy` doesn't use prompt templates, but if a caller passes
        one anyway the (method, None) entry must still win."""
        assert method_acronym("greedy", "stray_prompt") == "GRD"
        assert method_acronym("rl", "stray_prompt") == "RL"


class TestCompactLabel:
    def test_all_three_stages_same_model(self):
        out = compact_label(
            "SAP",
            {
                "task_generator": "gpt-4o-mini",
                "augmenter": "gpt-4o-mini",
                "evaluator": "gpt-4o-mini",
            },
        )
        assert out == "M: SAP + TAE: gpt-4o-mini"

    def test_augmenter_differs(self):
        out = compact_label(
            "SAP",
            {
                "task_generator": "gpt-4o-mini",
                "augmenter": "gpt-4.1-mini",
                "evaluator": "gpt-4o-mini",
            },
        )
        assert out == "M: SAP + TE: gpt-4o-mini + A: gpt-4.1-mini"

    def test_evaluator_differs(self):
        out = compact_label(
            "SAP",
            {
                "task_generator": "gpt-4o-mini",
                "augmenter": "gpt-4o-mini",
                "evaluator": "gpt-4.1-mini",
            },
        )
        assert out == "M: SAP + TA: gpt-4o-mini + E: gpt-4.1-mini"

    def test_task_generator_differs(self):
        # Pipeline-order rule: the group whose smallest stage comes
        # first in TtoAtoE renders first. With T alone differing, T (at
        # idx 0) comes before the merged A+E group (min idx 1); even
        # though the merged group is larger.
        out = compact_label(
            "SAP",
            {
                "task_generator": "gpt-4.1-mini",
                "augmenter": "gpt-4o-mini",
                "evaluator": "gpt-4o-mini",
            },
        )
        assert out == "M: SAP + T: gpt-4.1-mini + AE: gpt-4o-mini"

    def test_three_different_models(self):
        out = compact_label(
            "SAP",
            {
                "task_generator": "model-T",
                "augmenter": "model-A",
                "evaluator": "model-E",
            },
        )
        assert out == "M: SAP + T: model-T + A: model-A + E: model-E"

    def test_letters_within_group_sorted_in_pipeline_order(self):
        # T+E share a model, A differs; letters in the shared group must
        # render as `TE`, never `ET`, since T precedes E in pipeline order.
        out = compact_label(
            "SAP",
            {
                "task_generator": "gpt-4o-mini",
                "augmenter": "gpt-4.1-mini",
                "evaluator": "gpt-4o-mini",
            },
        )
        assert "TE: gpt-4o-mini" in out
        assert "ET:" not in out

    def test_groups_ordered_by_earliest_pipeline_stage(self):
        # When augmenter shares a model with task_generator (T+A) and
        # evaluator differs, the shared group's segment starts with the
        # earliest stage letter (T) to ordered first.
        out = compact_label(
            "SAP",
            {
                "task_generator": "gpt-4o-mini",
                "augmenter": "gpt-4o-mini",
                "evaluator": "gpt-4.1-mini",
            },
        )
        assert out.index("TA: gpt-4o-mini") < out.index("E: gpt-4.1-mini")

    def test_env_default_when_stage_model_unset(self):
        out = compact_label(
            "SAP",
            {"task_generator": "env-default"},  # other stages missing to env-default
        )
        # All three stages default to "env-default" to grouped.
        assert out == "M: SAP + TAE: env-default"


class TestResolveMethodAcronym:
    def test_none_scenarios_cfg_uses_method_only_acronym(self):
        # `llm_agent` has no None-prompt mapping to uppercased fallback.
        assert resolve_method_acronym(None, "any", "greedy") == "GRD"

    def test_resolves_from_scenarios_cfg_prompt_template(self):
        scenario = ScenarioDefinition(
            id="s1",
            augmentation=[
                AugmentationMethodConfig(
                    method="llm_agent",
                    llm_agent=LLMAgentConfig(
                        model="gpt-4o-mini",
                        prompt_template="augment_oneshot",
                    ),
                )
            ],
        )
        exp = _make_exp(scenario)
        assert resolve_method_acronym(exp, "s1", "llm_agent") == "SAP"

    def test_resolves_for_greedy_method_in_cfg(self):
        scenario = ScenarioDefinition(
            id="s1",
            augmentation=[AugmentationMethodConfig(method="greedy")],
        )
        exp = _make_exp(scenario)
        assert resolve_method_acronym(exp, "s1", "greedy") == "GRD"

    def test_missing_scenario_id_falls_back_to_method_only(self):
        exp = _make_exp(ScenarioDefinition(id="s1"))
        # The scenario "ghost" isn't in the YAML, so we can't resolve a
        # prompt template; fall back to the method-only acronym.
        assert resolve_method_acronym(exp, "ghost", "greedy") == "GRD"

    def test_missing_method_falls_back_to_method_only(self):
        scenario = ScenarioDefinition(
            id="s1",
            augmentation=[AugmentationMethodConfig(method="greedy")],
        )
        exp = _make_exp(scenario)
        # method `llm_agent` isn't in scenario's augmentation list.
        assert resolve_method_acronym(exp, "s1", "llm_agent") == "LLMAGENT"


# ---------------------------------------------------------------------------
# Compact label flows through every table when scenarios_cfg is supplied
# ---------------------------------------------------------------------------


def _exp_with_aug_axis_pair() -> ExperimentScenariosConfig:
    """Two-scenario fixture: augmenter axis A/B, all other stages pinned
    to gpt-4o-mini.  Matches the fixture shape so the assertions
    below also pin the user-facing convention."""
    baseline = ScenarioDefinition(
        id="oneshot_gpt_4o_mini",
        task_generator_model=LLMModelConfig(model="gpt-4o-mini"),
        evaluator_model=LLMModelConfig(model="gpt-4o-mini"),
        augmentation=[
            AugmentationMethodConfig(
                method="llm_agent",
                llm_agent=LLMAgentConfig(
                    model="gpt-4o-mini", prompt_template="augment_oneshot"
                ),
            )
        ],
    )
    alt = ScenarioDefinition(
        id="oneshot_gpt_4_1_mini",
        task_generator_model=LLMModelConfig(model="gpt-4o-mini"),
        evaluator_model=LLMModelConfig(model="gpt-4o-mini"),
        augmentation=[
            AugmentationMethodConfig(
                method="llm_agent",
                llm_agent=LLMAgentConfig(
                    model="gpt-4.1-mini", prompt_template="augment_oneshot"
                ),
            )
        ],
    )
    return _make_exp(baseline, alt)


class TestCompactLabelInReport:
    def test_run_summary_uses_compact_label_when_cfg_supplied(self):
        runs = [
            ("oneshot_gpt_4o_mini", "llm_agent", _full_artifacts(avg_total=0.5)),
            ("oneshot_gpt_4_1_mini", "llm_agent", _full_artifacts(avg_total=0.6)),
        ]
        md = build_benchmark_markdown(
            "experiment_f", runs, scenarios_cfg=_exp_with_aug_axis_pair()
        )
        # Compact label appears in the run-summary table …
        assert "M: SAP + TAE: gpt-4o-mini" in md
        assert "M: SAP + TE: gpt-4o-mini + A: gpt-4.1-mini" in md
        # … and the scenario id appears only in the dedicated reference
        # column / pre-table block, never as the primary label.
        assert "Scenario id" in md  # header column name

    def test_pairwise_heading_uses_compact_label(self):
        runs = [
            ("oneshot_gpt_4o_mini", "llm_agent", _full_artifacts(avg_total=0.5)),
            ("oneshot_gpt_4_1_mini", "llm_agent", _full_artifacts(avg_total=0.6)),
        ]
        md = build_benchmark_markdown(
            "experiment_f", runs, scenarios_cfg=_exp_with_aug_axis_pair()
        )
        assert (
            "### `M: SAP + TE: gpt-4o-mini + A: gpt-4.1-mini`  vs  "
            "`M: SAP + TAE: gpt-4o-mini`" in md
        )

    def test_legend_documents_only_present_method_families(self):
        # Two runs: an llm_agent one-shot (SAP) and a greedy (GRD).
        cfg = _make_exp(
            ScenarioDefinition(
                id="oneshot_gpt_4o_mini",
                augmentation=[
                    AugmentationMethodConfig(
                        method="llm_agent",
                        llm_agent=LLMAgentConfig(
                            model="gpt-4o-mini", prompt_template="augment_oneshot"
                        ),
                    )
                ],
            ),
            ScenarioDefinition(
                id="fcfs_greedy",
                augmentation=[AugmentationMethodConfig(method="greedy")],
            ),
        )
        runs = [
            ("oneshot_gpt_4o_mini", "llm_agent", _full_artifacts(avg_total=0.5)),
            ("fcfs_greedy", "greedy", _full_artifacts(avg_total=0.6)),
        ]
        md = build_benchmark_markdown("experiment_f", runs, scenarios_cfg=cfg)
        assert "## Legend" in md
        # Only the method families the run contains are spelled out.
        for token in ("**M**", "**T**", "**A**", "**E**", "SAP", "GRD"):
            assert token in md
        # The legacy iterative-turn agent is never run here, so the legend
        # must not document it (nor the unused RL / PTIME families).
        assert "Iterative-Turn" not in md
        assert "`RL`" not in md

    def test_format_legend_lists_only_given_codes(self):
        line = next(x for x in _format_legend({"SAP", "GRD"}) if "**M**" in x)
        assert "`SAP`" in line and "`GRD`" in line
        assert "ITA" not in line and "`RL`" not in line and "`PTIME`" not in line

    def test_format_legend_falls_back_for_unknown_code(self):
        line = next(x for x in _format_legend({"XYZ"}) if "**M**" in x)
        assert "`XYZ` = XYZ" in line

    def test_legend_section_omitted_without_cfg(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.5))]
        md = build_benchmark_markdown("e", runs, scenarios_cfg=None)
        # Without `scenarios_cfg` the report falls back to `<id>/<method>`
        # labels; emitting a Legend explaining acronyms it doesn't use
        # would be dead weight.
        assert "## Legend" not in md


# ---------------------------------------------------------------------------
# experiment_name in the report title
# ---------------------------------------------------------------------------


class TestExperimentNameInTitle:
    def test_experiment_name_titles_the_report(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.5))]
        md = build_benchmark_markdown(
            "exp_f", runs, experiment_name="Fulltime augmenter A/B"
        )
        assert md.splitlines()[0] == "# Benchmark report: Fulltime augmenter A/B"
        # The machine-readable id moves to a subtitle so it stays
        # discoverable for cross-referencing log lines + output paths.
        assert "_Experiment id: `exp_f`_" in md

    def test_falls_back_to_experiment_id_when_name_missing(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.5))]
        md = build_benchmark_markdown("exp_f", runs)
        assert md.splitlines()[0] == "# Benchmark report: `exp_f`"

    def test_experiment_name_threads_through_write_benchmark_report(
        self, tmp_path: Path
    ):
        eval_dir = tmp_path / "scenarios" / "s1" / "llm_agent" / "evaluation"
        eval_dir.mkdir(parents=True)
        (eval_dir / f"{TOTAL_REPORT_BASENAME}.json").write_text(
            json.dumps(
                {
                    "average_total_gain": 0.5,
                    "scored_persons": 1,
                    "empty_plan_persons": 0,
                    "average_gains": {},
                }
            ),
            encoding="utf-8",
        )
        target = write_benchmark_report(
            tmp_path, "exp_f", experiment_name="My experiment"
        )
        text = target.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "# Benchmark report: My experiment"


# ---------------------------------------------------------------------------
# shields.io experiment badge under the report title
# ---------------------------------------------------------------------------


class TestShieldsEscape:
    def test_spaces_become_percent_20(self):
        assert _shields_escape("a b c") == "a%20b%20c"

    def test_parens_are_percent_encoded(self):
        assert _shields_escape("(PHLC)") == "%28PHLC%29"

    def test_literal_dash_is_doubled(self):
        # shields.io reads a single dash as a field separator, so a
        # literal dash in the text must be escaped as `--`.
        assert _shields_escape("gpt-4") == "gpt--4"

    def test_literal_underscore_is_doubled(self):
        assert _shields_escape("a_b") == "a__b"


class TestExperimentBadge:
    def test_coded_name_builds_expected_badge(self):
        out = _experiment_badge(
            "Progressive Healthy Lifestyle Challenge (PHLC.26.07.01)"
        )
        assert out == (
            "![Progressive Healthy Lifestyle Challenge (PHLC) 2026.07]"
            "(https://img.shields.io/badge/"
            "Progressive%20Healthy%20Lifestyle%20Challenge%20%28PHLC%29"
            "-2026.07-22bfda.svg)"
        )

    def test_message_is_year_month_from_the_code(self):
        out = _experiment_badge("Some Study (XYZ.27.12.31)")
        assert out is not None
        assert "Some Study (XYZ) 2027.12" in out
        assert "-2027.12-22bfda.svg)" in out

    def test_name_without_code_returns_none(self):
        assert _experiment_badge("Fulltime augmenter A/B") is None

    def test_trailing_code_wins_over_earlier_parens(self):
        out = _experiment_badge("Weird (inner) Challenge (PHLC.26.07.01)")
        assert out is not None
        assert out.startswith("![Weird (inner) Challenge (PHLC) 2026.07]")


class TestExperimentBadgeInReport:
    def test_badge_rendered_under_title_for_coded_name(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.5))]
        md = build_benchmark_markdown(
            "phlc",
            runs,
            experiment_name="Progressive Healthy Lifestyle Challenge (PHLC.26.07.01)",
        )
        lines = md.splitlines()
        # Title first, badge image on the next non-blank line, before the
        # `_Experiment id:_` subtitle and the `_Generated_` stamp.
        assert lines[0].startswith("# Benchmark report:")
        assert lines[2].startswith(
            "![Progressive Healthy Lifestyle Challenge (PHLC) 2026.07]"
        )
        assert md.index("img.shields.io/badge/") < md.index("_Experiment id:")
        assert md.index("img.shields.io/badge/") < md.index("_Generated")

    def test_no_badge_when_name_lacks_code(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.5))]
        md = build_benchmark_markdown(
            "exp_f", runs, experiment_name="Fulltime augmenter A/B"
        )
        assert "img.shields.io/badge/" not in md

    def test_no_badge_when_name_missing(self):
        runs = [("s1", "llm_agent", _full_artifacts(avg_total=0.5))]
        md = build_benchmark_markdown("exp_f", runs)
        assert "img.shields.io/badge/" not in md

    def test_badge_threads_through_write_benchmark_report(self, tmp_path: Path):
        eval_dir = tmp_path / "scenarios" / "s1" / "llm_agent" / "evaluation"
        eval_dir.mkdir(parents=True)
        (eval_dir / f"{TOTAL_REPORT_BASENAME}.json").write_text(
            json.dumps(
                {
                    "average_total_gain": 0.5,
                    "scored_persons": 1,
                    "empty_plan_persons": 0,
                    "average_gains": {},
                }
            ),
            encoding="utf-8",
        )
        target = write_benchmark_report(
            tmp_path,
            "phlc",
            experiment_name="Progressive Healthy Lifestyle Challenge (PHLC.26.07.01)",
        )
        text = target.read_text(encoding="utf-8")
        assert (
            "https://img.shields.io/badge/"
            "Progressive%20Healthy%20Lifestyle%20Challenge%20%28PHLC%29"
            "-2026.07-22bfda.svg" in text
        )


# ---------------------------------------------------------------------------
# Experiment_f YAML round-trips with new ids + experiment_name
# ---------------------------------------------------------------------------


class TestExperimentFRoundTrip:
    def test_environment_yaml_carries_experiment_name(self):
        from src.scripts.persona.config.loader import load_environment

        env = load_environment(
            Path(__file__).resolve().parents[3]
            / "tests"
            / "fixtures"
            / "persona"
            / "experiment_f"
            / "environment.yaml"
        )
        assert env.experiment_name is not None
        assert "Experiment F" in env.experiment_name

    def test_scenarios_yaml_has_l234_suffix_stripped(self):
        from src.scripts.scenarios.config.loader import load_config

        cfg = load_config(
            Path(__file__).resolve().parents[3]
            / "tests"
            / "fixtures"
            / "persona"
            / "experiment_f"
            / "scenarios.yaml"
        )
        for scenario in cfg.scenarios:
            assert not scenario.id.endswith("_l234"), (
                f"scenario id {scenario.id!r} should drop the legacy " "_l234 suffix"
            )


# ---------------------------------------------------------------------------
# resolve_observation_contexts
# ---------------------------------------------------------------------------


class TestLabelForSection:
    def test_explicit_label_is_used_verbatim(self) -> None:
        """A scenario that declares `label` overrides the compact auto-label."""
        from types import SimpleNamespace

        from src.scripts.scenarios.export.benchmark_report import _label_for_section

        cfg = SimpleNamespace(
            scenarios=[
                SimpleNamespace(id="dqn_rl_per_person", label="DQN RL per person")
            ]
        )
        assert _label_for_section(cfg, "dqn_rl_per_person", "rl") == "DQN RL per person"

    def test_blank_label_falls_back_to_compact_label(self) -> None:
        """An empty `label` keeps the auto-derived acronym/model/context label."""
        from types import SimpleNamespace

        from src.scripts.scenarios.export.benchmark_report import _label_for_section

        cfg = SimpleNamespace(
            scenarios=[
                SimpleNamespace(
                    id="greedy_run",
                    label="",
                    augmentation=[SimpleNamespace(method="greedy")],
                    task_generator_model=None,
                    evaluator_model=None,
                )
            ]
        )
        label = _label_for_section(cfg, "greedy_run", "greedy")
        assert label and label != "greedy_run"


class TestResolveObservationContexts:
    def test_returns_none_when_scenarios_cfg_is_none(self) -> None:
        """No config means the caller falls back to the model-only label."""
        from types import SimpleNamespace

        from src.scripts.scenarios.export.benchmark_report import (
            resolve_observation_contexts,
        )

        assert resolve_observation_contexts(None, "s1", "llm_agent") is None

    def test_returns_none_when_scenario_id_unknown(self) -> None:
        """Unknown scenario id resolves to None."""
        from types import SimpleNamespace

        from src.scripts.scenarios.export.benchmark_report import (
            resolve_observation_contexts,
        )

        cfg = SimpleNamespace(scenarios=[SimpleNamespace(id="other", augmentation=[])])
        assert resolve_observation_contexts(cfg, "s1", "llm_agent") is None

    def test_returns_none_when_method_not_in_scenario(self) -> None:
        """A method missing from the scenario resolves to None."""
        from types import SimpleNamespace

        from src.scripts.scenarios.export.benchmark_report import (
            resolve_observation_contexts,
        )

        cfg = SimpleNamespace(
            scenarios=[
                SimpleNamespace(
                    id="s1",
                    augmentation=[SimpleNamespace(method="greedy")],
                )
            ]
        )
        assert resolve_observation_contexts(cfg, "s1", "llm_agent") is None

    def test_returns_empty_list_when_method_has_no_observation_attr(self) -> None:
        """A method entry without an observation attribute means blind augmenter."""
        from types import SimpleNamespace

        from src.scripts.scenarios.export.benchmark_report import (
            resolve_observation_contexts,
        )

        method_cfg = SimpleNamespace(method="llm_agent")
        cfg = SimpleNamespace(
            scenarios=[SimpleNamespace(id="s1", augmentation=[method_cfg])]
        )
        assert resolve_observation_contexts(cfg, "s1", "llm_agent") == []

    def test_returns_contexts_list_when_method_has_observation(self) -> None:
        """A non-empty observation.contexts list is returned verbatim."""
        from types import SimpleNamespace

        from src.scripts.scenarios.export.benchmark_report import (
            resolve_observation_contexts,
        )

        obs = SimpleNamespace(contexts=["mood_emotion", "energy_state"])
        method_cfg = SimpleNamespace(method="llm_agent", observation=obs)
        cfg = SimpleNamespace(
            scenarios=[SimpleNamespace(id="s1", augmentation=[method_cfg])]
        )
        assert resolve_observation_contexts(cfg, "s1", "llm_agent") == [
            "mood_emotion",
            "energy_state",
        ]


def test_compact_label_emits_obs_segment_when_contexts_present() -> None:
    """compact_label appends obs: only when the observation budget is non-empty."""
    from src.scripts.scenarios.export.benchmark_report import compact_label

    stages = {
        "task_generator": "gpt-4o-mini",
        "augmenter": "gpt-4o-mini",
        "evaluator": "gpt-4o-mini",
    }
    label_blind = compact_label(
        method_code="SAP", stage_models=stages, observation_contexts=[]
    )
    assert "obs:" not in label_blind
    label_informed = compact_label(
        method_code="SAP",
        stage_models=stages,
        observation_contexts=["mood_emotion", "energy_state"],
    )
    assert "obs: mood_emotion,energy_state" in label_informed


# ---------------------------------------------------------------------------
# Learning sections (per-person distribution + cross-augmenter trajectory)
# ---------------------------------------------------------------------------


class TestLearningSections:
    def test_per_person_distribution_renders_with_split_columns(self):
        rl = {
            "weekly": _weekly_artifact(
                {"p0": [0.3, 0.45, 0.6, 0.7], "p1": [0.4, 0.5, 0.62, 0.72]}
            )
        }
        short = {"weekly": _weekly_artifact({"q0": [0.5, 0.5]})}
        empty = {"weekly": {"weeks": [], "by_person": {}}}
        lines = _format_per_person_distribution(
            [("s1", "rl", rl), ("s2", "greedy", short), ("s3", "x", empty)]
        )
        text = "\n".join(lines)
        assert "## Per-person learning distribution" in text
        assert "conv/n" in text and "flat/n" in text and "decl/n" in text
        assert "IQM (last)" in text
        assert "n/a" in text  # the 2-week run has no Wk3/Wk4 median

    def test_cross_augmenter_shows_slope_and_caption(self):
        rl = {"weekly": _weekly_artifact({"p0": [0.3, 0.45, 0.6, 0.7]})}
        short = {"weekly": _weekly_artifact({"q0": [0.5, 0.5]})}
        none_week = {
            "weekly": {
                "weeks": [{"week_index": 1, "avg_weighted_gain": None}],
                "by_person": {},
            }
        }
        lines = _format_cross_augmenter_weekly(
            [("s1", "rl", rl), ("s2", "greedy", short), ("s3", "x", none_week)]
        )
        text = "\n".join(lines)
        assert "## Cross-augmenter weekly trajectory" in text
        assert "slope" in text
        assert "unit-of-analysis" in text
        assert "n/a" in text

    def test_cross_augmenter_rl_slope_is_positive(self):
        rl = {"weekly": _weekly_artifact({"p0": [0.3, 0.45, 0.6, 0.7]})}
        lines = _format_cross_augmenter_weekly([("s1", "rl", rl)])
        rl_row = next(
            ln for ln in lines if ln.startswith("|") and "rl" in ln and "+" in ln
        )
        slope_cell = rl_row.rstrip("|").rsplit("|", 1)[-1].strip()
        assert float(slope_cell) > 0

    def test_sections_absent_without_weekly(self):
        runs = [("s1", "llm_agent", {})]
        assert _format_per_person_distribution(runs) == []
        assert _format_cross_augmenter_weekly(runs) == []

    def test_build_markdown_includes_learning_sections(self):
        rl = _full_artifacts(avg_total=0.6)
        rl["weekly"] = _weekly_artifact(
            {"p0": [0.3, 0.45, 0.6, 0.7], "p1": [0.4, 0.5, 0.6, 0.7]}
        )
        md = build_benchmark_markdown("e", [("s1", "rl", rl)])
        assert "## Per-person learning distribution" in md
        assert "## Cross-augmenter weekly trajectory" in md

    def test_weekly_gain_table_skips_none_metric_rows(self):
        from src.scripts.scenarios.export.benchmark_report import _format_weekly_gain

        weekly = {
            "weeks": [
                {"week_index": 1, "avg_weighted_gain": 0.5},
                {"week_index": 2, "avg_weighted_gain": None},
            ],
            "by_person": {},
        }
        text = "\n".join(_format_weekly_gain([("s1", "rl", {"weekly": weekly})]))
        assert "## Weekly scheduling gain" in text
        assert "Wk1" in text
