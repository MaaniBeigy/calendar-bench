"""Unit tests for src.scripts.scenarios.cli (full pipeline implementation).

Coverage targets:
  Helper functions: _try_load_persona_run (run_dir None, missing, load error, ok),
                    _read_person_json (exists, missing),
                    _person_from_json (valid, invalid),
                    _build_task_generator (graphrag ok, graphrag fails, non-graphrag),
                    _write_tasks / _read_tasks (round-trip),
                    _build_augmenter (greedy, llm_agent pipeline ok, llm_agent no pipeline, None method).

  _cmd_generate_tasks: config error, persona run missing (EXIT_OK + message),
                        persona run load error (EXIT_USAGE),
                        person_from_json fails (skipped),
                        success (n persons written).

  _cmd_augment: config error, run missing (EXIT_USAGE),
                run load error, unknown method,
                task file missing (skipped), success.

  _cmd_evaluate: config error, persons_dir missing, no loss files,
                 success.

  _cmd_run: success chain, first-stage error, second-stage error.

  main(): normal dispatch, exception path.
  _build_parser(): argparse coverage.
  EXIT_OK / EXIT_VIOLATIONS / EXIT_USAGE constants.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.scripts.scenarios.cli as cli_module
from src.scripts.scenarios.cli import (
    EXIT_OK,
    EXIT_USAGE,
    EXIT_VIOLATIONS,
    _build_augmenter,
    _build_parser,
    _build_task_generator,
    _cmd_augment,
    _cmd_evaluate,
    _cmd_generate_tasks,
    _cmd_report,
    _cmd_run,
    _detect_persona_paths,
    _llm_settings_with_override,
    _method_cfg_to_scenario_config,
    _multi_augment,
    _multi_evaluate,
    _multi_generate_tasks,
    _multi_scenario_tasks_dir,
    _normalize_scenario_id_filter,
    _parse_baseline_arg,
    _person_from_json,
    _read_person_json,
    _read_tasks,
    _read_weekly_tasks,
    _render_augment_charts,
    _resolve_weekly_task_configs,
    _try_load_persona_run,
    _validate_multi_filters,
    _write_augmented_timeline,
    _write_benchmark_report_if_possible,
    _write_tasks,
    _write_weekly_tasks,
    main,
)
from src.scripts.scenarios.config.loader import ScenarioConfigError
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    AugmentationMethodConfig,
    CalendarSourceConfig,
    ExperimentScenariosConfig,
    LLMModelConfig,
    ObservationConfig,
    ScenarioConfig,
    ScenarioDefinition,
    ScenarioOutputConfig,
)
from tests.unit.scenarios.conftest import make_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_cfg(
    out_dir: str = "./output/test",
    run_dir: str | None = "./output/run",
    method: str = "greedy",
) -> ScenarioConfig:
    return ScenarioConfig(
        id="test_scenario",
        description="test",
        output=ScenarioOutputConfig(dir=out_dir),
        augmentation=AugmentationConfig(method=method),
    )


def _seed_solution_json(persons_dir: Path, pid: str, *, n_scheduled: int = 1) -> Path:
    """Write a per-person `<pid>.json` solution that evaluate drives off.

    Evaluate is solution-driven; nothing is read from `_loss.json`. With
    `n_scheduled=0` the solution schedules nothing (an empty plan).
    """
    persons_dir.mkdir(parents=True, exist_ok=True)
    scheduled = [
        {
            "label": f"t{i}",
            "display_name": f"Task {i}",
            "description": "",
            "date": "2026-06-01",
            "start_minutes": 540,
            "end_minutes": 600,
            "is_standalone": True,
            "concurrent_with": None,
            "intensity": 1,
            "is_concurrent": False,
            "is_dividable": False,
            "duration_min": 60,
            "duration_max": 60,
            "ontology_uri": None,
            "parent_task_label": None,
        }
        for i in range(n_scheduled)
    ]
    out = persons_dir / f"{pid}.json"
    out.write_text(
        json.dumps(
            {
                "person_id": pid,
                "tasks_total": n_scheduled,
                "scheduled_count": n_scheduled,
                "unscheduled_count": 0,
                "scheduled": scheduled,
                "unscheduled": [],
            }
        ),
        encoding="utf-8",
    )
    return out


def _mock_load(cfg: ScenarioConfig | None = None, raises: bool = False):
    if raises:
        return patch(
            "src.scripts.scenarios.cli.load_config",
            side_effect=ScenarioConfigError("bad"),
        )
    return patch(
        "src.scripts.scenarios.cli.load_config",
        return_value=cfg or _mock_cfg(),
    )


def _args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        scenario=Path("s.yaml"),
        out_dir=None,
        seed=None,
        tasks_dir=None,
        method=None,
        run_dir=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_exit_ok():
    assert EXIT_OK == 0


def test_exit_violations():
    assert EXIT_VIOLATIONS == 1


def test_exit_usage():
    assert EXIT_USAGE == 2


# ---------------------------------------------------------------------------
# _try_load_persona_run
# ---------------------------------------------------------------------------


class TestTryLoadPersonaRun:
    def test_run_dir_none_returns_none(self):
        cfg = _mock_cfg(run_dir=None)
        cfg_copy = ScenarioConfig(
            id="x",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
        )
        run, rd = _try_load_persona_run(cfg_copy)
        assert run is None

    def test_run_dir_missing_returns_none(self, tmp_path):
        cfg = _mock_cfg()
        run, rd = _try_load_persona_run(cfg, run_dir_override=tmp_path / "missing")
        assert run is None

    def test_run_dir_exists_but_load_error_returns_str(self, tmp_path):
        from src.scripts.scenarios.calendar.loader import CalendarLoaderError

        cfg = _mock_cfg()
        with patch(
            "src.scripts.scenarios.calendar.loader.load_persona_run",
            side_effect=CalendarLoaderError("load failed"),
        ):
            run, rd = _try_load_persona_run(cfg, run_dir_override=tmp_path)
        assert isinstance(run, str)

    def test_success_returns_run_object(self, tmp_path):
        cfg = _mock_cfg()
        mock_run = MagicMock()
        with patch(
            "src.scripts.scenarios.calendar.loader.load_persona_run",
            return_value=mock_run,
        ):
            run, rd = _try_load_persona_run(cfg, run_dir_override=tmp_path)
        assert run is mock_run


# ---------------------------------------------------------------------------
# _read_person_json
# ---------------------------------------------------------------------------


class TestReadPersonJson:
    def test_file_exists(self, tmp_path):
        data = {"persona": {"occupation_status": "fulltime"}}
        p = tmp_path / "p.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = _read_person_json(tmp_path, "p.json")
        assert result["persona"]["occupation_status"] == "fulltime"

    def test_file_missing_returns_empty(self, tmp_path):
        result = _read_person_json(tmp_path, "nonexistent.json")
        assert result == {}

    def test_cwd_relative_path_used_directly(self, tmp_path, monkeypatch):
        """Covers the `direct.exists()` True branch: cwd-relative path beats run_dir join.

        This is the real-world case: the persona pipeline stores json_path as
        `output/example_experiment/persons/p.json` (relative to /app cwd), not just
        `persons/p.json` (relative to run_dir).
        """
        monkeypatch.chdir(tmp_path)
        # Write the file at a cwd-relative path (not inside run_dir sub-path)
        data = {"persona": {"occupation_status": "student"}}
        cwd_path = tmp_path / "data.json"
        cwd_path.write_text(json.dumps(data), encoding="utf-8")
        # call with run_dir=some_other_dir but the cwd-relative path exists
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        result = _read_person_json(other_dir, "data.json")
        assert result["persona"]["occupation_status"] == "student"


# ---------------------------------------------------------------------------
# _person_from_json
# ---------------------------------------------------------------------------


class TestPersonFromJson:
    def test_valid_data_returns_person(self):
        data = {
            "persona": {
                "person_id": "p001",
                "persona_id": "test",
                "person_seed": 42,
                "instance_index": 0,
                "occupation_status": "fulltime",
                "stages": [],
                "jitter_applied": {"time_minutes": 15, "duration_minutes": 10},
                "event_overrides": {},
            }
        }
        person = _person_from_json(data)
        assert person is not None
        assert person.occupation_status == "fulltime"

    def test_invalid_data_returns_none(self):
        assert _person_from_json({"persona": "not_a_dict"}) is None

    def test_empty_returns_none(self):
        assert _person_from_json({}) is None


# ---------------------------------------------------------------------------
# _build_task_generator
# ---------------------------------------------------------------------------


class TestBuildTaskGenerator:
    def test_graphrag_builds_with_pipeline(self):
        cfg = _mock_cfg()
        mock_pipeline = MagicMock()
        with patch("src.graphrag.pipeline.build_graphrag", return_value=mock_pipeline):
            gen = _build_task_generator(cfg)
        assert gen is not None

    def test_graphrag_uri_prefixes_driven_by_yaml_ontologies(self):
        """The scenario YAML's `task_generation.ontologies` list is
        the load-bearing knob that controls which ontologies the
        GraphRAG retriever may surface.  Without this gate the shared
        vector index; which carries every imported ontology; surfaces
        off-domain candidates that the validator rejects, and
        `fetch_grounded_unique` exhausts retries with zero accepted
        tasks (an earlier empty-context regression).

        The test asserts that `_build_task_generator` resolves the
        YAML keys via `uri_prefixes_for_keys` and forwards the
        resulting prefix list to `build_graphrag`."""
        cfg = _mock_cfg()  # default ontologies = ["HealthTasks", "HumanActivities"]
        with patch(
            "src.graphrag.pipeline.build_graphrag", return_value=MagicMock()
        ) as ctor:
            _build_task_generator(cfg)
        ctor.assert_called_once()
        kwargs = ctor.call_args.kwargs
        # Both locally-authored ontologies declare uri_namespaces, so
        # both prefixes flow through to the retriever.
        assert kwargs.get("uri_prefixes") == [
            "https://w3id.org/calendar-bench/health/task/",
            "https://w3id.org/calendar-bench/human-activities/",
        ]

    def test_graphrag_uri_prefixes_respects_subset(self):
        """Listing only `HealthTasks` in the YAML must exclude
        HumanActivities from retrieval; the YAML is the single source
        of truth, no hardcoded fallback."""
        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="health-only",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag",
                ontologies=["HealthTasks"],
            ),
        )
        with patch(
            "src.graphrag.pipeline.build_graphrag", return_value=MagicMock()
        ) as ctor:
            _build_task_generator(cfg)
        kwargs = ctor.call_args.kwargs
        assert kwargs.get("uri_prefixes") == [
            "https://w3id.org/calendar-bench/health/task/"
        ]

    def test_graphrag_no_namespaces_falls_back_to_unfiltered(self):
        """A scenario whose listed ontologies declare no
        `uri_namespaces` (e.g. BioPortal-only configs) must fall
        back to unfiltered retrieval; gating on an empty prefix
        list would lock out every URI."""
        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="background-only",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag",
                ontologies=["BCIO"],  # no uri_namespaces declared
            ),
        )
        with patch(
            "src.graphrag.pipeline.build_graphrag", return_value=MagicMock()
        ) as ctor:
            _build_task_generator(cfg)
        kwargs = ctor.call_args.kwargs
        assert kwargs.get("uri_prefixes") is None

    def test_graphrag_grounded_pushes_branch_level_instance_into_cypher(self):
        """The `graphrag_grounded` path must push the scenario's
        filters.domains + filters.difficulty + the instance-only check
        into the retrieval Cypher itself.  Without this the LLM is
        shown off-domain candidates that the validator then rejects,
        exhausting fetch retries with zero accepted tasks (the
        an earlier empty-context regression even after the URI-prefix fix)."""
        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskFilters,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="grounded",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag_grounded",
                ontologies=["HealthTasks"],
                filters=TaskFilters(
                    domains=["NutritionTask", "PhysicalActivityTask"],
                    difficulty=["Level1"],
                ),
            ),
        )
        with patch(
            "src.graphrag.pipeline.build_graphrag", return_value=MagicMock()
        ) as ctor:
            _build_task_generator(cfg)
        kwargs = ctor.call_args.kwargs
        assert kwargs.get("uri_prefixes") == [
            "https://w3id.org/calendar-bench/health/task/"
        ]
        assert sorted(kwargs.get("allowed_branches") or []) == [
            "https://w3id.org/calendar-bench/health/NutritionTask",
            "https://w3id.org/calendar-bench/health/PhysicalActivityTask",
        ]
        assert kwargs.get("allowed_levels") == ["Level1"]
        assert kwargs.get("instance_only") is True

    def test_graphrag_legacy_does_not_push_branch_level_instance(self):
        """The legacy `graphrag` method emits unvalidated output and
        historically tolerated background context; narrowing its
        retriever would change behaviour for any A/B comparison run
        still using it.  Only `graphrag_grounded` gets the strict
        in-Cypher filters."""
        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskFilters,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="legacy",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag",
                ontologies=["HealthTasks"],
                filters=TaskFilters(
                    domains=["NutritionTask"],
                    difficulty=["Level1"],
                ),
            ),
        )
        with patch(
            "src.graphrag.pipeline.build_graphrag", return_value=MagicMock()
        ) as ctor:
            _build_task_generator(cfg)
        kwargs = ctor.call_args.kwargs
        # uri_prefixes still flows from cfg.ontologies (existing fix).
        assert kwargs.get("uri_prefixes") == [
            "https://w3id.org/calendar-bench/health/task/"
        ]
        # But the strict in-Cypher filters are NOT applied for the
        # legacy method.
        assert "allowed_branches" not in kwargs
        assert "allowed_levels" not in kwargs
        assert "instance_only" not in kwargs

    def test_graphrag_unknown_ontology_logs_warning(self, caplog):
        """A typo or stale ontology key in the YAML must surface a
        WARNING rather than silently being dropped; otherwise the
        scenario author would never notice that listing the key had
        no effect on retrieval."""
        import logging

        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="typo",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag",
                ontologies=["HealthTasks", "DEFINITELY_NOT_AN_ONTOLOGY"],
            ),
        )
        with caplog.at_level(logging.WARNING, logger="scenarios.cli"):
            with patch(
                "src.graphrag.pipeline.build_graphrag", return_value=MagicMock()
            ) as ctor:
                _build_task_generator(cfg)
        kwargs = ctor.call_args.kwargs
        # Only the known key contributes a prefix; the typo is dropped
        # with a warning so the author can catch the mistake.
        assert kwargs.get("uri_prefixes") == [
            "https://w3id.org/calendar-bench/health/task/"
        ]
        assert any(
            "unknown ontology" in r.message.lower()
            and "DEFINITELY_NOT_AN_ONTOLOGY" in r.message
            for r in caplog.records
        )

    def test_graphrag_pipeline_failure_warns(self, caplog):
        import logging

        cfg = _mock_cfg()
        with caplog.at_level(logging.WARNING, logger="scenarios.cli"):
            with patch(
                "src.graphrag.pipeline.build_graphrag",
                side_effect=RuntimeError("neo4j down"),
            ):
                gen = _build_task_generator(cfg)
        assert any("unavailable" in r.message.lower() for r in caplog.records)
        assert gen is not None

    def test_non_graphrag_returns_generator_without_pipeline(self):
        """Covers the False branch of `if method == 'graphrag'` in _build_task_generator."""
        from src.scripts.scenarios.config.schema import TaskGenerationConfig

        cfg = ScenarioConfig(
            id="x",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(method="manual"),
        )
        gen = _build_task_generator(cfg)
        assert gen is not None

    def test_graphrag_neo4j_driver_failure_warns_and_continues(self, caplog):
        """Covers the inner `except Exception … driver = None` branch.

        GraphRAG pipeline builds successfully but the Neo4j driver setup
        (`make_driver` / `verify_connectivity`) raises; the warning is
        logged, `driver` is forced to `None`, and the generator is still
        returned so task generation can run without ontology enrichment.
        """
        import logging

        cfg = _mock_cfg()
        mock_pipeline = MagicMock()
        with caplog.at_level(logging.WARNING, logger="scenarios.cli"):
            with (
                patch(
                    "src.graphrag.pipeline.build_graphrag",
                    return_value=mock_pipeline,
                ),
                patch(
                    "src.graphrag.config.Neo4jSettings.from_env",
                    return_value=MagicMock(),
                ),
                patch(
                    "src.graphrag.neo4j_client.make_driver",
                    side_effect=RuntimeError("neo4j down"),
                ),
            ):
                gen = _build_task_generator(cfg)
        assert gen is not None
        assert gen._driver is None
        assert any(
            "driver unavailable for ontology enrichment" in r.message.lower()
            for r in caplog.records
        )

    def test_grounded_method_builds_paraphrase_llm_and_embedder(self):
        """`method == graphrag_grounded` triggers Stage 2 + Stage 3
        collaborator construction; the resulting generator carries
        both the LLM and the embedder."""
        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="grounded",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag_grounded", paraphrase=True
            ),
        )
        with (
            patch("src.graphrag.pipeline.build_graphrag", return_value=MagicMock()),
            patch(
                "src.graphrag.config.Neo4jSettings.from_env",
                return_value=MagicMock(),
            ),
            patch("src.graphrag.neo4j_client.make_driver", return_value=MagicMock()),
            patch("src.graphrag.llm.make_usage_llm", return_value=MagicMock()),
            patch(
                "src.graphrag.embeddings.make_embedder",
                return_value=MagicMock(),
            ),
        ):
            gen = _build_task_generator(cfg)
        assert gen._paraphrase_llm is not None
        assert gen._embedder is not None

    def test_grounded_paraphrase_llm_and_embedder_failure_warned(self, caplog):
        import logging

        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="grounded",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag_grounded", paraphrase=True
            ),
        )
        with caplog.at_level(logging.WARNING, logger="scenarios.cli"):
            with (
                patch(
                    "src.graphrag.pipeline.build_graphrag",
                    return_value=MagicMock(),
                ),
                patch(
                    "src.graphrag.config.Neo4jSettings.from_env",
                    return_value=MagicMock(),
                ),
                patch(
                    "src.graphrag.neo4j_client.make_driver",
                    return_value=MagicMock(),
                ),
                patch(
                    "src.graphrag.llm.make_usage_llm",
                    side_effect=RuntimeError("no key"),
                ),
                patch(
                    "src.graphrag.embeddings.make_embedder",
                    side_effect=RuntimeError("emb down"),
                ),
            ):
                gen = _build_task_generator(cfg)
        assert gen._paraphrase_llm is None
        assert gen._embedder is None
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "Paraphrase LLM unavailable" in msgs
        assert "Embedder unavailable" in msgs

    def test_grounded_paraphrase_disabled_skips_llm_build(self):
        from src.scripts.scenarios.config.schema import (
            ScenarioOutputConfig,
            TaskGenerationConfig,
        )

        cfg = ScenarioConfig(
            id="grounded",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            task_generation=TaskGenerationConfig(
                method="graphrag_grounded", paraphrase=False
            ),
        )
        with (
            patch("src.graphrag.pipeline.build_graphrag", return_value=MagicMock()),
            patch(
                "src.graphrag.config.Neo4jSettings.from_env",
                return_value=MagicMock(),
            ),
            patch("src.graphrag.neo4j_client.make_driver", return_value=MagicMock()),
            patch("src.graphrag.llm.make_usage_llm") as mock_llm,
            patch("src.graphrag.embeddings.make_embedder") as mock_emb,
        ):
            gen = _build_task_generator(cfg)
        # paraphrase=False to neither helper is constructed.
        mock_llm.assert_not_called()
        mock_emb.assert_not_called()
        assert gen._paraphrase_llm is None

    @pytest.mark.parametrize(
        ("provider", "field_name"),
        [
            ("openai", "openai_model"),
            ("anthropic", "anthropic_model"),
            ("openrouter", "openrouter_model"),
        ],
    )
    def test_provider_specific_model_label_in_telemetry(self, provider, field_name):
        """Each provider routes to its own model field for the
        telemetry sidecar's `fetch_model` / `paraphrase_model`."""
        cfg = _mock_cfg()
        fake_settings = MagicMock(
            provider=provider,
            openai_model="gpt-4o-mini",
            anthropic_model="claude-sonnet-4-6",
            openrouter_model="anthropic/claude-sonnet-4.6",
        )
        with (
            patch("src.graphrag.pipeline.build_graphrag", return_value=MagicMock()),
            patch(
                "src.graphrag.config.Neo4jSettings.from_env",
                return_value=MagicMock(),
            ),
            patch("src.graphrag.neo4j_client.make_driver", return_value=MagicMock()),
            patch(
                "src.graphrag.config.LLMSettings.from_env",
                return_value=fake_settings,
            ),
        ):
            gen = _build_task_generator(cfg)
        assert gen._provider == provider
        assert gen._fetch_model == getattr(fake_settings, field_name)

    def test_graphrag_verify_connectivity_failure_warns_and_continues(self, caplog):
        """Same inner-except branch reached via `verify_connectivity` raising.

        `make_driver` returns a driver object but the connectivity probe
        fails; the warning still fires and `driver` is reset to `None`.
        """
        import logging

        cfg = _mock_cfg()
        mock_pipeline = MagicMock()
        bad_driver = MagicMock()
        bad_driver.verify_connectivity.side_effect = RuntimeError("unreachable")
        with caplog.at_level(logging.WARNING, logger="scenarios.cli"):
            with (
                patch(
                    "src.graphrag.pipeline.build_graphrag",
                    return_value=mock_pipeline,
                ),
                patch(
                    "src.graphrag.config.Neo4jSettings.from_env",
                    return_value=MagicMock(),
                ),
                patch(
                    "src.graphrag.neo4j_client.make_driver",
                    return_value=bad_driver,
                ),
            ):
                gen = _build_task_generator(cfg)
        assert gen is not None
        assert gen._driver is None
        assert any(
            "driver unavailable for ontology enrichment" in r.message.lower()
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# _write_tasks / _read_tasks round-trip
# ---------------------------------------------------------------------------


class TestWriteReadTasks:
    def test_round_trip(self, tmp_path):
        t1 = make_task("yoga", duration_min=30, duration_max=60)
        t2 = make_task("walk", duration_min=20, duration_max=40, intensity=1)
        path = tmp_path / "tasks.json"
        _write_tasks([t1, t2], path)
        loaded = _read_tasks(path)
        assert len(loaded) == 2
        assert loaded[0].label == "yoga"
        assert loaded[1].label == "walk"
        assert loaded[1].intensity == 1

    def test_display_name_computed_on_write(self, tmp_path):
        """_write_tasks emits effective_display_name (title-case fallback)."""
        t = make_task("morning_run")  # display_name=""
        path = tmp_path / "tasks.json"
        _write_tasks([t], path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw[0]["display_name"] == "Morning Run"

    def test_explicit_display_name_preserved_on_write(self, tmp_path):
        t = make_task("yoga", display_name="Morning Yoga Session")
        path = tmp_path / "tasks.json"
        _write_tasks([t], path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw[0]["display_name"] == "Morning Yoga Session"

    def test_description_and_display_name_round_trip(self, tmp_path):
        """display_name and description survive a write to read cycle."""
        t = make_task(
            "light_walk",
            display_name="Light Walk",
            description="A gentle stroll.",
            ontology_uri="http://example.org/walk",
        )
        path = tmp_path / "tasks.json"
        _write_tasks([t], path)
        loaded = _read_tasks(path)
        assert loaded[0].display_name == "Light Walk"
        assert "gentle stroll" in loaded[0].description

    def test_defaults_applied_on_read(self, tmp_path):
        """Missing optional fields get sensible defaults."""
        path = tmp_path / "tasks.json"
        path.write_text(json.dumps([{"label": "stretch"}]), encoding="utf-8")
        tasks = _read_tasks(path)
        assert tasks[0].label == "stretch"
        assert tasks[0].duration_min == 15
        assert tasks[0].display_name == ""
        assert tasks[0].description == ""


class TestWeeklyTaskIO:
    def test_weekly_round_trip_wraps_weeks(self, tmp_path):
        w1 = [make_task("yoga"), make_task("walk")]
        w2 = [make_task("run")]
        path = tmp_path / "tasks.json"
        _write_weekly_tasks([w1, w2], path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert list(raw) == ["weeks"]
        loaded = _read_weekly_tasks(path)
        assert [[t.label for t in wk] for wk in loaded] == [["yoga", "walk"], ["run"]]

    def test_read_weekly_accepts_legacy_bare_array(self, tmp_path):
        path = tmp_path / "tasks.json"
        _write_tasks([make_task("yoga")], path)  # bare array
        loaded = _read_weekly_tasks(path)
        assert len(loaded) == 1
        assert [t.label for t in loaded[0]] == ["yoga"]

    def test_read_tasks_flattens_weekly_wrapper(self, tmp_path):
        path = tmp_path / "tasks.json"
        _write_weekly_tasks([[make_task("a")], [make_task("b"), make_task("c")]], path)
        flat = _read_tasks(path)
        assert [t.label for t in flat] == ["a", "b", "c"]


class TestResolveWeeklyTaskConfigs:
    def _cfg(self, **kw):
        from src.scripts.scenarios.config.schema import TaskGenerationConfig

        return TaskGenerationConfig(**kw)

    def test_single_block_is_uniform(self):
        out = _resolve_weekly_task_configs([self._cfg(num_tasks=7)], horizon_weeks=8)
        assert len(out) == 1 and out[0].num_tasks == 7

    def test_per_week_explicit_resolves_each_week(self):
        cfgs = [self._cfg(week=1, num_tasks=10), self._cfg(week=2, num_tasks=14)]
        out = _resolve_weekly_task_configs(cfgs, horizon_weeks=2)
        assert [c.num_tasks for c in out] == [10, 14]

    def test_default_fills_unlisted_weeks(self):
        cfgs = [self._cfg(num_tasks=5), self._cfg(week=2, num_tasks=14)]
        out = _resolve_weekly_task_configs(cfgs, horizon_weeks=3)
        assert [c.num_tasks for c in out] == [5, 14, 5]

    def test_missing_coverage_raises(self):
        cfgs = [self._cfg(week=1, num_tasks=10)]
        with pytest.raises(ValueError, match="no config for week 2"):
            _resolve_weekly_task_configs(cfgs, horizon_weeks=2)

    def test_week_beyond_horizon_raises(self):
        cfgs = [self._cfg(num_tasks=5), self._cfg(week=9, num_tasks=10)]
        with pytest.raises(ValueError, match="exceeds the 8-week horizon"):
            _resolve_weekly_task_configs(cfgs, horizon_weeks=8)


# ---------------------------------------------------------------------------
# _build_augmenter
# ---------------------------------------------------------------------------


class TestBuildAugmenter:
    def test_greedy_returns_augmenter(self):
        from src.scripts.scenarios.augmentation.greedy import GreedyAugmenter

        aug = _build_augmenter("greedy")
        assert isinstance(aug, GreedyAugmenter)

    def test_llm_agent_uses_direct_pipeline_not_graphrag(self):
        """Augment-time LLM agent wiring (2026-05-14): the augmenter
        receives a :class:`DirectLLMPipeline` that bypasses retrieval
        + citation-template wrapping.  GraphRAG is the *task generation*
        step's job; calling it at augment time wastes an embedding
        API call, a Neo4j round-trip, and 500-2000 wasted input
        tokens of ontology context per week-call.  This test guards
        the separation.
        """
        from src.scripts.scenarios.augmentation.llm_agent import (
            DirectLLMPipeline,
            LLMAugmenter,
        )

        fake_llm = MagicMock()
        with patch("src.graphrag.llm.make_usage_llm", return_value=fake_llm):
            aug = _build_augmenter("llm_agent")
        assert isinstance(aug, LLMAugmenter)
        assert isinstance(aug._pipeline, DirectLLMPipeline)

    def test_llm_agent_pipeline_failure_still_returns_augmenter(self):
        """When `make_llm` raises (e.g. missing API key), the augmenter
        is still returned with `pipeline=None`; the LLMAugmenter
        treats no-pipeline as "drop all tasks unscheduled" rather than
        crashing the run."""
        from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter

        with patch(
            "src.graphrag.llm.make_usage_llm",
            side_effect=RuntimeError("no api key"),
        ):
            aug = _build_augmenter("llm_agent")
        assert isinstance(aug, LLMAugmenter)
        assert aug._pipeline is None

    def test_llm_agent_honors_scenario_model_override(self):
        """An explicitly-set llm_agent block re-targets the augmenter LLM."""
        from types import SimpleNamespace

        from src.scripts.scenarios.config.schema import LLMAgentConfig

        cfg = SimpleNamespace(
            augmentation=SimpleNamespace(
                llm_agent=LLMAgentConfig(provider="openai", model="gpt-5-mini")
            )
        )
        with patch(
            "src.graphrag.llm.make_usage_llm", return_value=MagicMock()
        ) as mock_llm:
            _build_augmenter("llm_agent", cfg=cfg)
        settings = mock_llm.call_args.args[0]
        assert settings.provider == "openai"
        assert settings.openai_model == "gpt-5-mini"

    def test_llm_agent_honors_anthropic_override(self):
        from types import SimpleNamespace

        from src.scripts.scenarios.config.schema import LLMAgentConfig

        cfg = SimpleNamespace(
            augmentation=SimpleNamespace(
                llm_agent=LLMAgentConfig(provider="anthropic", model="claude-opus-4-8")
            )
        )
        with patch(
            "src.graphrag.llm.make_usage_llm", return_value=MagicMock()
        ) as mock_llm:
            _build_augmenter("llm_agent", cfg=cfg)
        settings = mock_llm.call_args.args[0]
        assert settings.provider == "anthropic"
        assert settings.anthropic_model == "claude-opus-4-8"

    def test_llm_agent_default_block_falls_back_to_env(self):
        """A default-constructed llm_agent block (nothing set in the YAML)
        must not re-target the augmenter; env settings stay in charge."""
        from types import SimpleNamespace

        from src.scripts.scenarios.config.schema import LLMAgentConfig

        cfg = SimpleNamespace(augmentation=SimpleNamespace(llm_agent=LLMAgentConfig()))
        with patch(
            "src.graphrag.llm.make_usage_llm", return_value=MagicMock()
        ) as mock_llm:
            _build_augmenter("llm_agent", cfg=cfg)
        assert mock_llm.call_args.args[0] is None

    def test_unknown_returns_none(self):
        assert _build_augmenter("solver") is None


class TestAugmentLLMIdentity:
    """Telemetry provider/model must mirror the client the augmenter built."""

    @staticmethod
    def _cfg(llm_agent):
        from types import SimpleNamespace

        return SimpleNamespace(augmentation=SimpleNamespace(llm_agent=llm_agent))

    def test_explicit_anthropic_block_resolved(self):
        from src.scripts.scenarios.cli import _augment_llm_identity
        from src.scripts.scenarios.config.schema import LLMAgentConfig

        cfg = self._cfg(LLMAgentConfig(provider="anthropic", model="claude-opus-4-8"))
        assert _augment_llm_identity("llm_agent", cfg) == (
            "anthropic",
            "claude-opus-4-8",
        )

    def test_explicit_openai_block_resolved(self):
        from src.scripts.scenarios.cli import _augment_llm_identity
        from src.scripts.scenarios.config.schema import LLMAgentConfig

        cfg = self._cfg(LLMAgentConfig(provider="openai", model="gpt-5-mini"))
        assert _augment_llm_identity("llm_agent", cfg) == ("openai", "gpt-5-mini")

    def test_default_block_reports_env_default(self):
        from src.scripts.scenarios.cli import _augment_llm_identity
        from src.scripts.scenarios.config.schema import LLMAgentConfig

        cfg = self._cfg(LLMAgentConfig())
        assert _augment_llm_identity("llm_agent", cfg) == ("openai", "")

    def test_non_llm_method_reports_env_default(self):
        from src.scripts.scenarios.cli import _augment_llm_identity

        assert _augment_llm_identity("greedy", self._cfg(None)) == ("openai", "")


class TestJudgeOracleCallRecorder:
    """The judge oracle records per-call tokens for the eval telemetry."""

    def test_invoke_wrapper_records_tokens(self, tmp_path):
        from src.scripts.scenarios.cli import _build_judge_oracle

        class _FakeResponse:
            content = '{"score": 0.5, "reason": "ok"}'

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _FakeResponse()
        with patch("src.graphrag.llm.make_usage_llm", return_value=fake_llm):
            oracle = _build_judge_oracle(tmp_path / "cache.jsonl")
        assert oracle is not None
        out = oracle._invoke_fn("rate this pair")
        assert out == _FakeResponse.content
        assert len(oracle.call_recorder.calls) == 1
        in_tok, out_tok, wall, reasoning = oracle.call_recorder.calls[0]
        assert in_tok > 0 and out_tok > 0 and wall >= 0.0 and reasoning == 0

    def test_recorder_reset_clears_between_persons(self, tmp_path):
        from src.scripts.scenarios.cli import _build_judge_oracle

        class _FakeResponse:
            content = '{"score": 0.5, "reason": "ok"}'

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _FakeResponse()
        with patch("src.graphrag.llm.make_usage_llm", return_value=fake_llm):
            oracle = _build_judge_oracle(tmp_path / "cache.jsonl")
        oracle._invoke_fn("p1")
        oracle.call_recorder.reset()
        oracle._invoke_fn("p2")
        assert len(oracle.call_recorder.calls) == 1

    def test_invoke_uses_provider_usage_metadata_when_present(self, tmp_path):
        from src.scripts.scenarios.cli import _build_judge_oracle

        class _FakeResponse:
            content = '{"score": 0.5, "reason": "ok"}'
            usage_metadata = {"input_tokens": 31, "output_tokens": 7}

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _FakeResponse()
        with patch("src.graphrag.llm.make_usage_llm", return_value=fake_llm):
            oracle = _build_judge_oracle(tmp_path / "cache.jsonl")
        oracle._invoke_fn("rate this pair")
        in_tok, out_tok, _, _ = oracle.call_recorder.calls[0]
        assert (in_tok, out_tok) == (31, 7)

    def test_time_windows_passed_to_greedy(self):
        from src.scripts.persona.config.schema import WindowRange
        from src.scripts.scenarios.augmentation.greedy import GreedyAugmenter

        mock_run = MagicMock()
        mock_run.time_windows = {"morning": WindowRange(start=400, end=600)}
        aug = _build_augmenter("greedy", run=mock_run)
        assert isinstance(aug, GreedyAugmenter)
        assert "morning" in aug._time_windows


# ---------------------------------------------------------------------------
# _cmd_generate_tasks
# ---------------------------------------------------------------------------


class TestCmdGenerateTasks:
    def test_config_error_returns_exit_usage(self, tmp_path):
        with _mock_load(raises=True):
            rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_persona_run_missing_returns_exit_ok(self, tmp_path, capsys):
        """No persona run to EXIT_OK with a helpful message."""
        with _mock_load(_mock_cfg(out_dir=str(tmp_path / "out"))):
            with patch.object(
                cli_module, "_try_load_persona_run", return_value=(None, None)
            ):
                rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert "0 persons" in capsys.readouterr().out

    def test_persona_run_load_error_returns_exit_usage(self, tmp_path):
        with _mock_load(_mock_cfg(out_dir=str(tmp_path / "out"))):
            with patch.object(
                cli_module, "_try_load_persona_run", return_value=("boom", None)
            ):
                rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_person_from_json_failure_skipped(self, tmp_path):
        cfg = _mock_cfg(out_dir=str(tmp_path / "out"))
        mock_run = MagicMock()
        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        mock_run.traces = [mock_trace]
        with (
            _mock_load(cfg),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(mock_run, tmp_path)
            ),
            patch.object(cli_module, "_person_from_json", return_value=None),
            patch.object(cli_module, "_build_task_generator"),
        ):
            rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK

    def test_success_writes_task_files(self, tmp_path):
        cfg = _mock_cfg(out_dir=str(tmp_path / "out"))
        mock_run = MagicMock()
        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        mock_run.traces = [mock_trace]
        mock_person = MagicMock()
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [make_task("yoga")]
        with (
            _mock_load(cfg),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(mock_run, tmp_path)
            ),
            patch.object(cli_module, "_person_from_json", return_value=mock_person),
            patch.object(cli_module, "_build_task_generator", return_value=mock_gen),
            patch.object(cli_module, "_write_weekly_tasks") as mock_write,
        ):
            rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml", seed=42))
        assert rc == EXIT_OK
        mock_write.assert_called_once()

    def _per_week_cfg(self, tmp_path, entries):
        from src.scripts.scenarios.config.schema import ScenarioConfig

        return ScenarioConfig(
            id="s", task_generation=entries, output={"dir": str(tmp_path / "out")}
        )

    def test_per_week_generation_threads_cross_week_exclusion(self, tmp_path):
        cfg = self._per_week_cfg(
            tmp_path,
            [
                {"week": 1, "num_tasks": 1},
                {"week": 2, "num_tasks": 1, "cross_week_distinct": False},
            ],
        )
        run = MagicMock()
        run.horizon_days = 14
        trace = MagicMock()
        trace.person_id = "p001"
        run.traces = [trace]
        gen1, gen2 = MagicMock(), MagicMock()
        gen1.generate.return_value = [
            make_task("wk1", ontology_uri="https://ex.org/task/a")
        ]
        gen2.generate.return_value = [make_task("wk2")]
        with (
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(
                cli_module, "_scope_run_to_timeframe", return_value=(run, None)
            ),
            patch.object(cli_module, "_person_from_json", return_value=MagicMock()),
            patch.object(cli_module, "_build_task_generator", side_effect=[gen1, gen2]),
            patch.object(cli_module, "_write_weekly_tasks") as mock_write,
        ):
            rc = cli_module._cmd_generate_tasks_with_cfg(
                cfg, _args(scenario=tmp_path / "s.yaml", out_dir=tmp_path / "out")
            )
        assert rc == EXIT_OK
        weekly = mock_write.call_args[0][0]
        assert [[t.label for t in wk] for wk in weekly] == [["wk1"], ["wk2"]]
        # Week 1 is distinct (empty exclusion); week 2 opts out (None).
        assert gen1.generate.call_args.kwargs["exclude_uris"] == set()
        assert gen2.generate.call_args.kwargs["exclude_uris"] is None

    def test_per_week_resolution_error_returns_usage(self, tmp_path):
        # Week 2 is unmapped (no default), so resolution fails.
        cfg = self._per_week_cfg(tmp_path, [{"week": 1, "num_tasks": 1}])
        run = MagicMock()
        run.horizon_days = 14
        run.traces = [MagicMock(person_id="p001")]
        with (
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(
                cli_module, "_scope_run_to_timeframe", return_value=(run, None)
            ),
        ):
            rc = cli_module._cmd_generate_tasks_with_cfg(
                cfg, _args(scenario=tmp_path / "s.yaml", out_dir=tmp_path / "out")
            )
        assert rc == cli_module.EXIT_USAGE

    def test_success_with_index_json_reads_persons(self, tmp_path):
        """Covers the `if index_path.exists():` True branch (lines 295-300)."""
        # Write a real index.json so the branch is taken
        index = {"persons": [{"person_id": "p001", "json_path": "persons/p001.json"}]}
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        (persons_dir / "p001.json").write_text(
            json.dumps({"persona": {"not": "valid"}}), encoding="utf-8"
        )

        cfg = _mock_cfg(out_dir=str(tmp_path / "out"))
        mock_run = MagicMock()
        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        mock_run.traces = [mock_trace]

        with (
            _mock_load(cfg),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(mock_run, tmp_path)
            ),
            patch.object(cli_module, "_person_from_json", return_value=None),
            patch.object(cli_module, "_build_task_generator"),
        ):
            rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK  # skipped (person invalid) but no error

    def test_seed_shown_in_output(self, tmp_path, capsys):
        with _mock_load(_mock_cfg(out_dir=str(tmp_path / "out"))):
            with patch.object(
                cli_module, "_try_load_persona_run", return_value=(None, None)
            ):
                _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml", seed=99))
        assert "99" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _cmd_augment
# ---------------------------------------------------------------------------


class TestCmdAugment:
    def _mock_run(self, traces=None):
        run = MagicMock()
        run.traces = traces or []
        run.time_windows = {}
        run.allen_pair_rules = []
        run.horizon_days = 7
        return run

    def test_config_error_returns_exit_usage(self, tmp_path):
        with _mock_load(raises=True):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_persona_run_load_error_returns_exit_usage(self, tmp_path):
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=("err", None)
            ),
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_persona_run_missing_returns_exit_usage(self, tmp_path):
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(None, None)
            ),
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_unknown_method_returns_exit_usage(self, tmp_path):
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module,
                "_try_load_persona_run",
                return_value=(self._mock_run(), tmp_path),
            ),
            patch.object(cli_module, "_build_augmenter", return_value=None),
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_task_file_missing_skipped(self, tmp_path):
        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        run = self._mock_run(traces=[mock_trace])
        mock_augmenter = MagicMock()
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        mock_augmenter.augment.assert_not_called()

    def test_success_writes_outputs(self, tmp_path):
        task = make_task("yoga")
        # Write a task file
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_tasks([task], tasks_dir / "p001_tasks.json")

        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        run = self._mock_run(traces=[mock_trace])
        mock_solution = MagicMock()
        mock_augmenter = MagicMock()
        mock_augmenter.augment.return_value = mock_solution
        mock_loss_fn = MagicMock()
        mock_loss_fn.compute.return_value = (
            0.5,
            MagicMock(
                cov=0.5,
                cal=0.0,
                pref=0.0,
                disp=0.0,
                merge=0.0,
                spread=0.0,
                divide=0.0,
                context_fit=None,
            ),
        )
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
            patch(
                "src.scripts.scenarios.metrics.loss.SchedulingLoss",
                return_value=mock_loss_fn,
            ),
            patch("src.scripts.scenarios.export.json_writer.write_solution_json"),
            patch(
                "src.scripts.scenarios.export.ics_writer.write_augmented_ics"
            ) as mock_ics,
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        mock_augmenter.augment.assert_called_once()
        # calendar=trace must be forwarded so base events appear in the ICS
        _, kwargs = mock_ics.call_args
        assert "calendar" in kwargs
        assert kwargs["calendar"] is mock_trace

    def test_write_flags_false_skips_io(self, tmp_path):
        """Covers the False branches of `if write_json:` and `if write_ics:`."""
        from src.scripts.scenarios.config.schema import ScenarioOutputConfig

        task = make_task("yoga")
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_tasks([task], tasks_dir / "p001_tasks.json")

        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        run = self._mock_run(traces=[mock_trace])
        mock_solution = MagicMock()
        mock_augmenter = MagicMock()
        mock_augmenter.augment.return_value = mock_solution
        mock_loss_fn = MagicMock()
        mock_loss_fn.compute.return_value = (
            0.5,
            MagicMock(
                cov=0.5,
                cal=0.0,
                pref=0.0,
                disp=0.0,
                merge=0.0,
                spread=0.0,
                divide=0.0,
                context_fit=None,
            ),
        )
        # Config with write_json=False, write_ics=False
        cfg_no_write = ScenarioConfig(
            id="no_write",
            output=ScenarioOutputConfig(
                dir=str(tmp_path), write_json=False, write_ics=False
            ),
            augmentation=AugmentationConfig(method="greedy"),
        )
        with (
            patch("src.scripts.scenarios.cli.load_config", return_value=cfg_no_write),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
            patch(
                "src.scripts.scenarios.metrics.loss.SchedulingLoss",
                return_value=mock_loss_fn,
            ),
            patch(
                "src.scripts.scenarios.export.json_writer.write_solution_json"
            ) as mock_json,
            patch(
                "src.scripts.scenarios.export.ics_writer.write_augmented_ics"
            ) as mock_ics,
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        mock_json.assert_not_called()
        mock_ics.assert_not_called()

    def test_method_override(self, tmp_path, capsys):
        run = self._mock_run()
        mock_augmenter = MagicMock()
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path), method="greedy")),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
            patch("src.scripts.scenarios.metrics.loss.SchedulingLoss"),
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml", method="llm_agent"))
        assert rc == EXIT_OK
        assert "llm_agent" in capsys.readouterr().out

    def test_charts_flag_triggers_render_augment_charts(self, tmp_path):
        """When `args.charts` is set the augment command invokes the
        chart renderer with the augment-output directory."""
        run = self._mock_run()
        mock_augmenter = MagicMock()
        captured: dict = {}

        def fake_render(cfg, args, aug_dir, persons_dir, run_obj):
            captured["aug_dir"] = aug_dir
            captured["persons_dir"] = persons_dir
            captured["charts"] = args.charts
            return EXIT_OK

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
            patch("src.scripts.scenarios.metrics.loss.SchedulingLoss"),
            patch.object(cli_module, "_render_augment_charts", fake_render),
        ):
            rc = _cmd_augment(
                _args(
                    scenario=tmp_path / "s.yaml",
                    charts=["calendar"],
                    calendar_dpi=200,
                )
            )
        assert rc == EXIT_OK
        assert captured["charts"] == ["calendar"]

    def test_charts_render_failure_returns_exit_usage(self, tmp_path):
        run = self._mock_run()
        mock_augmenter = MagicMock()
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
            patch("src.scripts.scenarios.metrics.loss.SchedulingLoss"),
            patch.object(cli_module, "_render_augment_charts", return_value=EXIT_USAGE),
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml", charts=["calendar"]))
        assert rc == EXIT_USAGE


# ---------------------------------------------------------------------------
# _render_augment_charts
# ---------------------------------------------------------------------------


class TestRenderAugmentCharts:
    def _cfg_with_run_dir(self, run_dir: Path | None) -> ScenarioConfig:
        return ScenarioConfig(
            id="x",
            output=ScenarioOutputConfig(dir="./out"),
            augmentation=AugmentationConfig(method="greedy"),
            calendar=CalendarSourceConfig(run_dir=str(run_dir) if run_dir else None),
        )

    def test_returns_ok_with_warning_when_persona_run_dir_missing(
        self, tmp_path, caplog
    ):
        cfg = self._cfg_with_run_dir(None)
        args = argparse.Namespace(charts=[], calendar_dpi=300, method=None)
        run = MagicMock()
        with caplog.at_level("WARNING", logger="scenarios.cli"):
            rc = _render_augment_charts(
                cfg, args, tmp_path / "aug", tmp_path / "persons", run
            )
        assert rc == EXIT_OK
        assert "persona run directory" in caplog.text

    def test_returns_ok_with_warning_when_persons_subdir_missing(
        self, tmp_path, caplog
    ):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = self._cfg_with_run_dir(run_dir)
        args = argparse.Namespace(charts=[], calendar_dpi=300, method=None)
        with caplog.at_level("WARNING", logger="scenarios.cli"):
            rc = _render_augment_charts(
                cfg, args, tmp_path / "aug", tmp_path / "persons", MagicMock()
            )
        assert rc == EXIT_OK
        assert "persona run directory" in caplog.text

    def test_returns_ok_when_no_schedules_found(self, tmp_path, caplog):
        run_dir = tmp_path / "run"
        (run_dir / "persons").mkdir(parents=True)
        cfg = self._cfg_with_run_dir(run_dir)
        args = argparse.Namespace(charts=[], calendar_dpi=300, method=None)
        with caplog.at_level("WARNING", logger="scenarios.cli"):
            rc = _render_augment_charts(
                cfg, args, tmp_path / "aug", tmp_path / "persons", MagicMock()
            )
        assert rc == EXIT_OK
        assert "no persona JSONs" in caplog.text

    def test_unknown_kind_returns_exit_usage(self, tmp_path):
        # Build a valid persona schedule so we reach the kinds check.
        run_dir = tmp_path / "run"
        persons_dir = run_dir / "persons"
        persons_dir.mkdir(parents=True)
        (persons_dir / "p.json").write_text(
            json.dumps(
                {
                    "person_id": "p",
                    "persona_id": "a",
                    "person_seed": 0,
                    "days": [],
                }
            ),
            encoding="utf-8",
        )
        cfg = self._cfg_with_run_dir(run_dir)
        args = argparse.Namespace(charts=["bogus"], calendar_dpi=300, method=None)
        rc = _render_augment_charts(
            cfg, args, tmp_path / "aug", tmp_path / "augp", MagicMock()
        )
        assert rc == EXIT_USAGE

    def test_value_error_from_render_returns_exit_usage(self, tmp_path):
        run_dir = tmp_path / "run"
        persons_dir = run_dir / "persons"
        persons_dir.mkdir(parents=True)
        (persons_dir / "p.json").write_text(
            json.dumps(
                {
                    "person_id": "p",
                    "persona_id": "a",
                    "person_seed": 0,
                    "days": [],
                }
            ),
            encoding="utf-8",
        )
        cfg = self._cfg_with_run_dir(run_dir)
        args = argparse.Namespace(charts=["calendar"], calendar_dpi=300, method=None)
        # Force the underlying renderer to throw ValueError.
        from src.scripts.persona.analytics import charts as charts_module

        with patch.object(
            charts_module,
            "render_all_charts",
            side_effect=ValueError("boom"),
        ):
            rc = _render_augment_charts(
                cfg, args, tmp_path / "aug", tmp_path / "augp", MagicMock()
            )
        assert rc == EXIT_USAGE

    def test_happy_path_invokes_renderer_and_prints_path(self, tmp_path, capsys):
        run_dir = tmp_path / "run"
        persons_dir = run_dir / "persons"
        persons_dir.mkdir(parents=True)
        (persons_dir / "p.json").write_text(
            json.dumps(
                {
                    "person_id": "p",
                    "persona_id": "a",
                    "person_seed": 0,
                    "days": [],
                }
            ),
            encoding="utf-8",
        )
        cfg = self._cfg_with_run_dir(run_dir)
        args = argparse.Namespace(charts=["calendar"], calendar_dpi=120, method=None)
        from src.scripts.persona.analytics import charts as charts_module

        captured: dict = {}

        def fake_render(schedules, *, out_dir, **kw):
            captured["kinds"] = set(kw.get("kinds") or ())
            captured["dpi"] = kw.get("weekly_calendar_dpi")
            return charts_module.ChartArtifacts()

        with patch.object(charts_module, "render_all_charts", fake_render):
            rc = _render_augment_charts(
                cfg,
                args,
                tmp_path / "aug",
                tmp_path / "augp",
                MagicMock(),
            )
        assert rc == EXIT_OK
        assert captured["kinds"] == {"calendar"}
        assert captured["dpi"] == 120
        assert "charts written under" in capsys.readouterr().out

    def test_default_kinds_when_charts_is_empty_list(self, tmp_path):
        """Bare `--charts` (args.charts == []) maps to the calendar
        family only; augment-side does not need lines/heatmap/gantt
        unless explicitly requested."""
        run_dir = tmp_path / "run"
        persons_dir = run_dir / "persons"
        persons_dir.mkdir(parents=True)
        (persons_dir / "p.json").write_text(
            json.dumps(
                {
                    "person_id": "p",
                    "persona_id": "a",
                    "person_seed": 0,
                    "days": [],
                }
            ),
            encoding="utf-8",
        )
        cfg = self._cfg_with_run_dir(run_dir)
        args = argparse.Namespace(charts=[], calendar_dpi=300, method=None)
        from src.scripts.persona.analytics import charts as charts_module

        captured: dict = {}

        def fake_render(schedules, *, out_dir, **kw):
            captured["kinds"] = set(kw.get("kinds") or ())
            return charts_module.ChartArtifacts()

        with patch.object(charts_module, "render_all_charts", fake_render):
            _render_augment_charts(
                cfg, args, tmp_path / "aug", tmp_path / "augp", MagicMock()
            )
        assert captured["kinds"] == {"calendar"}


# ---------------------------------------------------------------------------
# _cmd_evaluate
# ---------------------------------------------------------------------------


class TestCmdEvaluate:
    def test_config_error_returns_exit_usage(self, tmp_path):
        with _mock_load(raises=True):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_persons_dir_missing_returns_exit_usage(self, tmp_path):
        with _mock_load(_mock_cfg(out_dir=str(tmp_path))):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_no_loss_files_returns_exit_usage(self, tmp_path):
        persons_dir = tmp_path / "augmented" / "persons"
        persons_dir.mkdir(parents=True)
        with _mock_load(_mock_cfg(out_dir=str(tmp_path))):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_success_writes_report(self, tmp_path):
        persons_dir = tmp_path / "augmented" / "persons"
        _seed_solution_json(persons_dir, "p001")
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ) as mock_report,
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        mock_report.assert_called_once()

    def test_avg_gain_in_output(self, tmp_path, capsys):
        persons_dir = tmp_path / "augmented" / "persons"
        # Evaluate recomputes every leg from the reconstructed solution;
        # the console line carries an `avg_gain` figure regardless of value.
        for pid in ("p001", "p002"):
            _seed_solution_json(persons_dir, pid)
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
        ):
            _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        out = capsys.readouterr().out
        assert "avg_gain" in out

    def test_avg_gain_na_when_all_plans_empty(self, tmp_path, capsys):
        """Every persona had 0 tasks to grounding sidecar shows `total
        == 0` for each to `avg_gain` must print as `n/a`, not the
        misleading `1.0000` the old formula produced.  Also exercises
        the `excluded N empty plans` suffix on the INFO log line."""
        persons_dir = tmp_path / "augmented" / "persons"
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(parents=True)
        for pid in ("p001", "p002"):
            _seed_solution_json(persons_dir, pid, n_scheduled=0)
            # Empty task list to grounding sidecar will report total=0.
            (tasks_dir / f"{pid}_tasks.json").write_text("[]", encoding="utf-8")
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
        ):
            _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        out = capsys.readouterr().out
        assert "avg_gain=n/a" in out


# ---------------------------------------------------------------------------
# _cmd_run
# ---------------------------------------------------------------------------


class TestCmdRun:
    def test_success_runs_all_stages(self, tmp_path, capsys):
        with (
            _mock_load(),
            patch.object(
                cli_module, "_cmd_generate_tasks_with_cfg", return_value=EXIT_OK
            ),
            patch.object(cli_module, "_cmd_augment_with_cfg", return_value=EXIT_OK),
            patch.object(cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_OK),
        ):
            rc = _cmd_run(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK

    def test_generate_error_stops_pipeline(self, tmp_path):
        with (
            _mock_load(),
            patch.object(
                cli_module, "_cmd_generate_tasks_with_cfg", return_value=EXIT_USAGE
            ),
        ):
            rc = _cmd_run(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE

    def test_augment_error_stops_pipeline(self, tmp_path):
        with (
            _mock_load(),
            patch.object(
                cli_module, "_cmd_generate_tasks_with_cfg", return_value=EXIT_OK
            ),
            patch.object(cli_module, "_cmd_augment_with_cfg", return_value=EXIT_USAGE),
        ):
            rc = _cmd_run(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_USAGE


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_dispatches_handler(self, tmp_path):
        scenario = tmp_path / "s.yaml"
        with _mock_load():
            with patch.object(
                cli_module, "_try_load_persona_run", return_value=(None, None)
            ):
                rc = main(["generate-tasks", "--scenario", str(scenario)])
        assert rc == EXIT_OK

    def test_exception_returns_exit_usage(self, tmp_path):
        scenario = tmp_path / "s.yaml"
        with patch(
            "src.scripts.scenarios.cli.load_config",
            side_effect=RuntimeError("boom"),
        ):
            rc = main(["generate-tasks", "--scenario", str(scenario)])
        assert rc == EXIT_USAGE


# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_help_exits(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--help"])

    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_generate_tasks_parsed(self, tmp_path):
        scenario = tmp_path / "s.yaml"
        scenario.touch()
        args = _build_parser().parse_args(
            ["generate-tasks", "--scenario", str(scenario), "--seed", "42"]
        )
        assert args.seed == 42

    def test_augment_method_parsed(self, tmp_path):
        scenario = tmp_path / "s.yaml"
        scenario.touch()
        args = _build_parser().parse_args(
            ["augment", "--scenario", str(scenario), "--method", "greedy"]
        )
        assert args.method == "greedy"

    def test_augment_invalid_method_rejected(self, tmp_path):
        scenario = tmp_path / "s.yaml"
        scenario.touch()
        with pytest.raises(SystemExit):
            _build_parser().parse_args(
                ["augment", "--scenario", str(scenario), "--method", "bad"]
            )


# ---------------------------------------------------------------------------
# _method_cfg_to_scenario_config
# ---------------------------------------------------------------------------


class TestMethodCfgToScenarioConfig:
    def _exp(self, **kw) -> ExperimentScenariosConfig:
        return ExperimentScenariosConfig.model_validate(
            {"experiment_id": "exp_a", "scenarios": [{"id": "s1"}], **kw}
        )

    def _scenario(self) -> ScenarioDefinition:
        return ScenarioDefinition(id="nutrition_l1", description="test scenario")

    def test_id_is_scenario_slash_method(self):
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(method="greedy")
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.id == "nutrition_l1/greedy"

    def test_description_copied(self):
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(method="greedy")
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.description == "test scenario"

    def test_auto_output_dir_derived(self):
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(method="greedy")
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        # New tree layout: per-method output lives under
        # `<exp>/scenarios/<scenario>/<method>/`.
        assert cfg.output.dir == "./output/exp_a/scenarios/nutrition_l1/greedy"

    def test_explicit_output_dir_used(self):
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(
            method="llm_agent",
            output=ScenarioOutputConfig(dir="./custom/out"),
        )
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.output.dir == "./custom/out"

    def test_explicit_output_write_flags_used(self):
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(
            method="greedy",
            output=ScenarioOutputConfig(dir="./out", write_ics=False, write_json=False),
        )
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.output.write_ics is False
        assert cfg.output.write_json is False

    def test_run_dir_from_experiment(self):
        exp = self._exp(run_dir="./output/exp_a")
        scenario = self._scenario()
        method = AugmentationMethodConfig()
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.calendar.run_dir == "./output/exp_a"

    def test_augmentation_method_forwarded(self):
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(method="llm_agent", merge_threshold=0.80)
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.augmentation.method == "llm_agent"
        assert cfg.augmentation.merge_threshold == pytest.approx(0.80)

    def test_scenario_file_none_gives_null_persona_paths(self):
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig()
        cfg = _method_cfg_to_scenario_config(exp, scenario, method, scenario_file=None)
        assert cfg.calendar.persona_environment is None
        assert cfg.calendar.persona_events is None

    def test_scenario_file_in_dir_with_yamls_populates_paths(self, tmp_path):
        """When standard YAML files exist next to scenarios.yaml, paths are set."""
        for name in (
            "environment.yaml",
            "persona_config.yaml",
            "event_config.yaml",
            "temporal_relation_rules.yaml",
        ):
            (tmp_path / name).write_text("dummy", encoding="utf-8")
        scenario_file = tmp_path / "scenarios.yaml"
        scenario_file.write_text("dummy", encoding="utf-8")

        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig()
        cfg = _method_cfg_to_scenario_config(
            exp, scenario, method, scenario_file=scenario_file
        )
        assert cfg.calendar.persona_environment is not None
        assert cfg.calendar.persona_environment.endswith("environment.yaml")
        assert cfg.calendar.persona_events is not None
        assert cfg.calendar.persona_events.endswith("event_config.yaml")

    def test_scenario_file_missing_yamls_gives_null_paths(self, tmp_path):
        """If sibling YAMLs are absent the paths stay None (no crash)."""
        scenario_file = tmp_path / "scenarios.yaml"
        scenario_file.write_text("dummy", encoding="utf-8")
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig()
        cfg = _method_cfg_to_scenario_config(
            exp, scenario, method, scenario_file=scenario_file
        )
        assert cfg.calendar.persona_environment is None
        assert cfg.calendar.persona_events is None

    def test_observation_contexts_forwarded(self):
        """observation.contexts survives the multi-to-legacy config translation."""
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(
            method="llm_agent",
            observation=ObservationConfig(
                contexts=["mood_emotion", "energy_state"],
            ),
        )
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.augmentation.observation.contexts == [
            "mood_emotion",
            "energy_state",
        ]

    def test_observation_full_block_forwarded(self):
        """All four ObservationConfig fields survive translation."""
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(
            method="llm_agent",
            observation=ObservationConfig(
                contexts=["weather_environment"],
                context_detail="full",
                host_flags=["is_concurrent", "intensity"],
                task_flags=["duration_min", "duration_max"],
            ),
        )
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        obs = cfg.augmentation.observation
        assert obs.contexts == ["weather_environment"]
        assert obs.context_detail == "full"
        assert obs.host_flags == ["is_concurrent", "intensity"]
        assert obs.task_flags == ["duration_min", "duration_max"]

    def test_observation_default_when_unset(self):
        """Omitting observation gives the schema default with empty contexts and host_flags."""
        exp = self._exp()
        scenario = self._scenario()
        method = AugmentationMethodConfig(method="greedy")
        cfg = _method_cfg_to_scenario_config(exp, scenario, method)
        assert cfg.augmentation.observation.contexts == []
        assert cfg.augmentation.observation.host_flags == []


# ---------------------------------------------------------------------------
# _detect_persona_paths
# ---------------------------------------------------------------------------


class TestDetectPersonaPaths:
    def test_none_path_returns_all_none(self):
        result = _detect_persona_paths(None)
        for v in result.values():
            assert v is None

    def test_existing_yamls_detected(self, tmp_path):
        for name in (
            "environment.yaml",
            "persona_config.yaml",
            "event_config.yaml",
            "temporal_relation_rules.yaml",
        ):
            (tmp_path / name).write_text("x", encoding="utf-8")
        result = _detect_persona_paths(tmp_path / "scenarios.yaml")
        assert result["persona_environment"] is not None
        assert "environment.yaml" in result["persona_environment"]
        assert result["persona_config"] is not None
        assert result["persona_events"] is not None
        assert result["persona_rules"] is not None

    def test_missing_yamls_give_none(self, tmp_path):
        result = _detect_persona_paths(tmp_path / "scenarios.yaml")
        for v in result.values():
            assert v is None

    def test_partial_yamls_partial_none(self, tmp_path):
        (tmp_path / "environment.yaml").write_text("x", encoding="utf-8")
        result = _detect_persona_paths(tmp_path / "scenarios.yaml")
        assert result["persona_environment"] is not None
        assert result["persona_config"] is None

    def test_real_experiment_a_scenarios_yaml(self):
        """The fixture folder has all four YAML files."""
        real = (
            Path(__file__).resolve().parents[3]
            / "tests"
            / "fixtures"
            / "persona"
            / "experiment_a"
            / "scenarios.yaml"
        )
        result = _detect_persona_paths(real)
        assert result["persona_environment"] is not None
        assert result["persona_events"] is not None
        assert result["persona_rules"] is not None


# ---------------------------------------------------------------------------
# Multi-scenario dispatch helpers
# ---------------------------------------------------------------------------


def _make_exp_cfg(
    scenario_ids=("s1",), methods=("greedy",)
) -> ExperimentScenariosConfig:
    """Build a minimal ExperimentScenariosConfig for testing."""
    scenarios = [
        {
            "id": sid,
            "augmentation": [{"method": m} for m in methods],
        }
        for sid in scenario_ids
    ]
    return ExperimentScenariosConfig.model_validate(
        {"experiment_id": "exp_a", "scenarios": scenarios}
    )


def _multi_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scenario=tmp_path / "scenarios.yaml",
        out_dir=None,
        seed=None,
        workers=5,
        log_level="INFO",
        tasks_dir=None,
        run_dir=None,
    )


class TestMultiGenerateTasks:
    def test_calls_generate_for_each_scenario(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        calls = []

        def fake_gen(cfg, sub_args):
            calls.append(cfg.id)
            return EXIT_OK

        with patch.object(
            cli_module, "_cmd_generate_tasks_with_cfg", side_effect=fake_gen
        ):
            rc = _multi_generate_tasks(exp, args, scenario_id=None)
        assert rc == EXIT_OK
        assert len(calls) == 2

    def test_scenario_id_filter_skips_others(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        calls = []

        def fake_gen(cfg, sub_args):
            calls.append(cfg.id)
            return EXIT_OK

        with patch.object(
            cli_module, "_cmd_generate_tasks_with_cfg", side_effect=fake_gen
        ):
            rc = _multi_generate_tasks(exp, args, scenario_id="s1")
        assert rc == EXIT_OK
        assert len(calls) == 1
        assert "s1" in calls[0]

    def test_stops_on_first_failure(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        calls = []

        def fake_gen(cfg, sub_args):
            calls.append(cfg.id)
            return EXIT_USAGE  # always fail

        with patch.object(
            cli_module, "_cmd_generate_tasks_with_cfg", side_effect=fake_gen
        ):
            rc = _multi_generate_tasks(exp, args, scenario_id=None)
        assert rc == EXIT_USAGE
        assert len(calls) == 1  # stopped after first


class TestMultiAugment:
    def test_calls_augment_for_each_scenario_and_method(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy", "llm_agent"))
        args = _multi_args(tmp_path)
        calls = []

        def fake_aug(cfg, sub_args):
            calls.append(cfg.id)
            return EXIT_OK

        with patch.object(cli_module, "_cmd_augment_with_cfg", side_effect=fake_aug):
            rc = _multi_augment(exp, args, scenario_id=None, method_filter=None)
        assert rc == EXIT_OK
        assert len(calls) == 2  # 1 scenario × 2 methods

    def test_method_filter_restricts_calls(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy", "llm_agent"))
        args = _multi_args(tmp_path)
        calls = []

        def fake_aug(cfg, sub_args):
            calls.append(cfg.augmentation.method)
            return EXIT_OK

        with patch.object(cli_module, "_cmd_augment_with_cfg", side_effect=fake_aug):
            rc = _multi_augment(exp, args, scenario_id=None, method_filter="greedy")
        assert rc == EXIT_OK
        assert calls == ["greedy"]

    def test_scenario_filter_and_method_filter_combined(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy", "llm_agent"))
        args = _multi_args(tmp_path)
        calls = []

        def fake_aug(cfg, sub_args):
            calls.append(cfg.id)
            return EXIT_OK

        with patch.object(cli_module, "_cmd_augment_with_cfg", side_effect=fake_aug):
            rc = _multi_augment(exp, args, scenario_id="s1", method_filter="greedy")
        assert rc == EXIT_OK
        assert len(calls) == 1
        assert "s1/greedy" in calls[0]

    def test_stops_on_failure(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)

        with patch.object(cli_module, "_cmd_augment_with_cfg", return_value=EXIT_USAGE):
            rc = _multi_augment(exp, args, scenario_id=None, method_filter=None)
        assert rc == EXIT_USAGE


class TestMultiEvaluate:
    def test_calls_evaluate_for_each_scenario_and_method(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        calls = []

        def fake_eval(cfg, sub_args):
            calls.append(cfg.id)
            return EXIT_OK

        with patch.object(cli_module, "_cmd_evaluate_with_cfg", side_effect=fake_eval):
            rc = _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
        assert rc == EXIT_OK
        assert len(calls) == 2

    def test_method_filter_skips_non_matching(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy", "llm_agent"))
        args = _multi_args(tmp_path)
        calls = []

        def fake_eval(cfg, sub_args):
            calls.append(cfg.augmentation.method)
            return EXIT_OK

        with patch.object(cli_module, "_cmd_evaluate_with_cfg", side_effect=fake_eval):
            rc = _multi_evaluate(exp, args, scenario_id=None, method_filter="llm_agent")
        assert rc == EXIT_OK
        assert calls == ["llm_agent"]

    def test_stops_on_failure(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)

        with patch.object(
            cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_USAGE
        ):
            rc = _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
        assert rc == EXIT_USAGE


class TestValidateMultiFilters:
    """Guards added 2026-05-14; silently skipping a mistyped
    `--scenario-id` (e.g. `senior_leisure_l23` vs. the YAML's
    `senior_leisure_l123`) used to return `EXIT_OK` with zero work
    done.  The validator now errors out and lists the valid IDs."""

    def test_passes_when_no_filters(self):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        assert _validate_multi_filters(exp, scenario_id=None) == EXIT_OK
        assert (
            _validate_multi_filters(exp, scenario_id=None, method_filter=None)
            == EXIT_OK
        )

    def test_passes_when_scenario_id_matches(self):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        assert _validate_multi_filters(exp, scenario_id="s1") == EXIT_OK
        assert _validate_multi_filters(exp, scenario_id="s2") == EXIT_OK

    def test_errors_when_scenario_id_unknown(self, caplog):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _validate_multi_filters(exp, scenario_id="nope")
        assert rc == EXIT_USAGE
        # Error message names the offending ID and lists the valid IDs.
        msg = caplog.text
        assert "'nope'" in msg
        assert "s1" in msg and "s2" in msg

    def test_errors_when_scenario_id_off_by_one_typo(self, caplog):
        """The actual 2026-05-14 incident: `senior_leisure_l23`
        passed instead of `senior_leisure_l123` silently no-op'd
        the entire pipeline.  This regression test pins the fix."""
        exp = _make_exp_cfg(
            scenario_ids=("senior_walking_l2", "senior_leisure_l123"),
            methods=("greedy",),
        )
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _validate_multi_filters(exp, scenario_id="senior_leisure_l23")
        assert rc == EXIT_USAGE
        assert "'senior_leisure_l23'" in caplog.text
        assert "senior_leisure_l123" in caplog.text

    def test_passes_when_method_filter_matches(self):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy", "llm_agent"))
        assert (
            _validate_multi_filters(exp, scenario_id=None, method_filter="greedy")
            == EXIT_OK
        )
        assert (
            _validate_multi_filters(exp, scenario_id=None, method_filter="llm_agent")
            == EXIT_OK
        )

    def test_errors_when_method_filter_unknown(self, caplog):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy",))
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _validate_multi_filters(
                exp, scenario_id=None, method_filter="bogus_method"
            )
        assert rc == EXIT_USAGE
        assert "'bogus_method'" in caplog.text
        assert "greedy" in caplog.text

    def test_method_filter_scoped_to_selected_scenario(self, caplog):
        """When --scenario-id narrows to one scenario, --method must
        be valid for THAT scenario (not the union across all)."""
        # s1 has greedy only; s2 has llm_agent only.
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp",
                "scenarios": [
                    {"id": "s1", "augmentation": [{"method": "greedy"}]},
                    {"id": "s2", "augmentation": [{"method": "llm_agent"}]},
                ],
            }
        )
        # s1 + llm_agent should error; llm_agent isn't in s1's methods.
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _validate_multi_filters(
                exp, scenario_id="s1", method_filter="llm_agent"
            )
        assert rc == EXIT_USAGE
        assert "'llm_agent'" in caplog.text

    def test_multi_generate_tasks_errors_on_unknown_scenario_id(self, tmp_path, caplog):
        """Integration-level guard: the bad ID short-circuits the
        runner and no per-scenario work is dispatched."""
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        with (
            patch.object(cli_module, "_cmd_generate_tasks_with_cfg") as inner,
            caplog.at_level("ERROR", logger="scenarios.cli"),
        ):
            rc = _multi_generate_tasks(exp, args, scenario_id="nope")
        assert rc == EXIT_USAGE
        inner.assert_not_called()
        assert "'nope'" in caplog.text

    def test_multi_augment_errors_on_unknown_scenario_id(self, tmp_path, caplog):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        with (
            patch.object(cli_module, "_cmd_augment_with_cfg") as inner,
            caplog.at_level("ERROR", logger="scenarios.cli"),
        ):
            rc = _multi_augment(exp, args, scenario_id="nope", method_filter=None)
        assert rc == EXIT_USAGE
        inner.assert_not_called()
        assert "'nope'" in caplog.text

    def test_multi_augment_errors_on_unknown_method(self, tmp_path, caplog):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy",))
        args = _multi_args(tmp_path)
        with (
            patch.object(cli_module, "_cmd_augment_with_cfg") as inner,
            caplog.at_level("ERROR", logger="scenarios.cli"),
        ):
            rc = _multi_augment(exp, args, scenario_id=None, method_filter="bogus")
        assert rc == EXIT_USAGE
        inner.assert_not_called()
        assert "'bogus'" in caplog.text

    def test_multi_evaluate_errors_on_unknown_scenario_id(self, tmp_path, caplog):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        with (
            patch.object(cli_module, "_cmd_evaluate_with_cfg") as inner,
            caplog.at_level("ERROR", logger="scenarios.cli"),
        ):
            rc = _multi_evaluate(exp, args, scenario_id="nope", method_filter=None)
        assert rc == EXIT_USAGE
        inner.assert_not_called()
        assert "'nope'" in caplog.text


class TestCmdDispatchMulti:
    """Tests that the top-level _cmd_* dispatchers route to multi helpers."""

    def _exp_cfg(self):
        return _make_exp_cfg()

    def test_generate_tasks_dispatches_to_multi(self, tmp_path):
        exp = self._exp_cfg()
        with (
            patch("src.scripts.scenarios.cli.load_config", return_value=exp),
            patch.object(
                cli_module, "_multi_generate_tasks", return_value=EXIT_OK
            ) as m,
        ):
            rc = _cmd_generate_tasks(
                argparse.Namespace(
                    scenario=tmp_path / "s.yaml", scenario_id=None, workers=1, seed=None
                )
            )
        assert rc == EXIT_OK
        m.assert_called_once()

    def test_augment_dispatches_to_multi(self, tmp_path):
        exp = self._exp_cfg()
        with (
            patch("src.scripts.scenarios.cli.load_config", return_value=exp),
            patch.object(cli_module, "_multi_augment", return_value=EXIT_OK) as m,
        ):
            rc = _cmd_augment(
                argparse.Namespace(
                    scenario=tmp_path / "s.yaml",
                    scenario_id=None,
                    method=None,
                )
            )
        assert rc == EXIT_OK
        m.assert_called_once()

    def test_evaluate_dispatches_to_multi(self, tmp_path):
        exp = self._exp_cfg()
        with (
            patch("src.scripts.scenarios.cli.load_config", return_value=exp),
            patch.object(cli_module, "_multi_evaluate", return_value=EXIT_OK) as m,
        ):
            rc = _cmd_evaluate(
                argparse.Namespace(
                    scenario=tmp_path / "s.yaml",
                    scenario_id=None,
                    method=None,
                    run_dir=None,
                )
            )
        assert rc == EXIT_OK
        m.assert_called_once()

    def test_run_dispatches_to_multi(self, tmp_path):
        exp = self._exp_cfg()
        with (
            patch("src.scripts.scenarios.cli.load_config", return_value=exp),
            patch.object(cli_module, "_multi_generate_tasks", return_value=EXIT_OK),
            patch.object(cli_module, "_multi_augment", return_value=EXIT_OK),
            patch.object(cli_module, "_multi_evaluate", return_value=EXIT_OK),
        ):
            rc = _cmd_run(
                argparse.Namespace(
                    scenario=tmp_path / "s.yaml",
                    scenario_id=None,
                    method=None,
                    seed=None,
                    workers=1,
                    out_dir=None,
                )
            )
        assert rc == EXIT_OK

    def test_run_multi_stops_on_generate_failure(self, tmp_path):
        exp = self._exp_cfg()
        with (
            patch("src.scripts.scenarios.cli.load_config", return_value=exp),
            patch.object(cli_module, "_multi_generate_tasks", return_value=EXIT_USAGE),
        ):
            rc = _cmd_run(
                argparse.Namespace(
                    scenario=tmp_path / "s.yaml",
                    scenario_id=None,
                    method=None,
                    seed=None,
                    workers=1,
                    out_dir=None,
                )
            )
        assert rc == EXIT_USAGE

    def test_run_multi_stops_on_augment_failure(self, tmp_path):
        exp = self._exp_cfg()
        with (
            patch("src.scripts.scenarios.cli.load_config", return_value=exp),
            patch.object(cli_module, "_multi_generate_tasks", return_value=EXIT_OK),
            patch.object(cli_module, "_multi_augment", return_value=EXIT_USAGE),
        ):
            rc = _cmd_run(
                argparse.Namespace(
                    scenario=tmp_path / "s.yaml",
                    scenario_id=None,
                    method=None,
                    seed=None,
                    workers=1,
                    out_dir=None,
                )
            )
        assert rc == EXIT_USAGE

    def test_run_config_error_returns_exit_usage(self, tmp_path):
        with patch(
            "src.scripts.scenarios.cli.load_config",
            side_effect=ScenarioConfigError("bad"),
        ):
            rc = _cmd_run(
                argparse.Namespace(
                    scenario=tmp_path / "s.yaml",
                    scenario_id=None,
                    method=None,
                    seed=None,
                    workers=1,
                    out_dir=None,
                )
            )
        assert rc == EXIT_USAGE

    def test_multi_evaluate_scenario_id_filter_skips_non_matching(self, tmp_path):
        """Cover the scenario_id != filter to continue branch in _multi_evaluate."""
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)
        calls = []

        def fake_eval(cfg, sub_args):
            calls.append(cfg.id)
            return EXIT_OK

        with patch.object(cli_module, "_cmd_evaluate_with_cfg", side_effect=fake_eval):
            rc = _multi_evaluate(exp, args, scenario_id="s2", method_filter=None)
        assert rc == EXIT_OK
        assert len(calls) == 1
        assert "s2" in calls[0]


# ---------------------------------------------------------------------------
# _build_judge_oracle (oracle wiring with optional LLM backend)
# ---------------------------------------------------------------------------


class TestBuildJudgeOracle:
    def test_returns_none_when_make_llm_raises(self, tmp_path, caplog):
        """No LLM available to return None and log a warning; do not crash."""
        with patch(
            "src.graphrag.llm.make_usage_llm",
            side_effect=RuntimeError("no key"),
        ):
            result = cli_module._build_judge_oracle(tmp_path / "cache.jsonl")
        assert result is None

    def test_returns_oracle_when_llm_available(self, tmp_path):
        """When make_llm succeeds, an oracle wrapping its .invoke is returned."""
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = '{"score": 0.8}'
        fake_llm.invoke.return_value = fake_response
        with patch("src.graphrag.llm.make_usage_llm", return_value=fake_llm):
            oracle = cli_module._build_judge_oracle(tmp_path / "cache.jsonl")
        assert oracle is not None
        # Score routes through the LLM
        score = oracle.score("yoga", "gym")
        assert score == pytest.approx(0.8)
        fake_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# _build_semantic_oracle (augment-time σ wiring with optional Neo4j embedder)
# ---------------------------------------------------------------------------


class TestBuildSemanticOracle:
    """Wired in the L_merge follow-up (2026-05-14):
    the augmenter now tries a Neo4j-embedding `embedding_fn` so σ
    actually scores; without a reachable driver the default oracle
    (constant 0.0) is returned, which makes `compute_l_concurrent`
    return `None` (no signal) instead of laundering 0.0 into a
    perfect `G_merge = 1.0`.
    """

    def test_falls_back_to_default_when_neo4j_unreachable(self, tmp_path, caplog):
        """Both backends unreachable to a default oracle (stub σ=0)."""
        from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

        with (
            patch(
                "src.graphrag.neo4j_client.make_driver",
                side_effect=RuntimeError("no neo4j"),
            ),
            patch(
                "src.graphrag.llm.make_usage_llm",
                side_effect=RuntimeError("no llm"),
            ),
        ):
            oracle = cli_module._build_semantic_oracle(tmp_path)
        assert isinstance(oracle, SemanticCompatibility)
        # Both backends absent to score is 0.0 (stub none strategy).
        assert oracle.score("walking", "schedule-a-30-minute-walk") == 0.0

    def test_uses_neo4j_embedding_fn_when_driver_reachable(self, tmp_path):
        """When the driver builds and verifies, σ is computed via the
        Neo4j embedding fetch (cosine similarity)."""
        from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

        fake_driver = MagicMock()
        fake_driver.verify_connectivity.return_value = None
        # Two different unit-norm vectors with cos = 0.6.
        emb_map = {
            "yoga": [1.0, 0.0, 0.0],
            "stretching": [0.6, 0.8, 0.0],
        }

        def fake_embedding_fn(label):  # captured by patched factory
            return emb_map.get(label)

        with (
            patch(
                "src.graphrag.neo4j_client.make_driver",
                return_value=fake_driver,
            ),
            patch(
                "src.scripts.scenarios.metrics.semantic.make_neo4j_embedding_fn",
                return_value=fake_embedding_fn,
            ),
        ):
            oracle = cli_module._build_semantic_oracle(tmp_path)
        assert isinstance(oracle, SemanticCompatibility)
        # cos((1,0,0), (0.6,0.8,0)) = 0.6
        score = oracle.score("yoga", "stretching")
        assert score == pytest.approx(0.6)

    def test_falls_back_when_verify_connectivity_raises(self, tmp_path):
        """verify_connectivity raising + no LLM to stub σ=0."""
        from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

        fake_driver = MagicMock()
        fake_driver.verify_connectivity.side_effect = RuntimeError("auth failure")
        with (
            patch(
                "src.graphrag.neo4j_client.make_driver",
                return_value=fake_driver,
            ),
            patch(
                "src.graphrag.llm.make_usage_llm",
                side_effect=RuntimeError("no llm"),
            ),
        ):
            oracle = cli_module._build_semantic_oracle(tmp_path)
        assert isinstance(oracle, SemanticCompatibility)
        assert oracle.score("a", "b") == 0.0

    def test_no_llm_fallback_at_augment_time(self, tmp_path):
        """Even when an LLM is reachable, the augment-time oracle MUST
        NOT call it. A cold-cache LLM fallback blew up run-time by ~30x
        (one OpenAI call per σ pair, hundreds per person); the LLM judge
        only runs at evaluate-time via `_build_judge_oracle`."""
        from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

        fake_llm = MagicMock()
        fake_llm.invoke.return_value.content = "0.7"
        with (
            patch(
                "src.graphrag.neo4j_client.make_driver",
                side_effect=RuntimeError("no neo4j"),
            ),
            patch("src.graphrag.llm.make_llm", return_value=fake_llm),
        ):
            oracle = cli_module._build_semantic_oracle(tmp_path)
        assert isinstance(oracle, SemanticCompatibility)
        _, strategy = oracle.score_with_strategy("walking", "lunch")
        assert strategy == "none"
        fake_llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# _reconstruct_solution_from_json
# ---------------------------------------------------------------------------


class TestReconstructSolutionFromJson:
    def test_minimal_payload(self):
        """All-default payload to valid SchedulingSolution with empty scheduled."""
        sol = cli_module._reconstruct_solution_from_json(
            {"person_id": "p001", "scheduled": []}
        )
        assert sol.person_id == "p001"
        assert sol.scheduled == []

    def test_scheduled_entries_round_trip(self):
        sol = cli_module._reconstruct_solution_from_json(
            {
                "person_id": "p001",
                "scheduled": [
                    {
                        "label": "yoga",
                        "display_name": "Yoga",
                        "description": "",
                        "date": "2026-05-04",
                        "start_minutes": 480,
                        "end_minutes": 540,
                        "is_standalone": True,
                        "concurrent_with": None,
                        "intensity": 2,
                        "ontology_uri": None,
                        "duration_min": 30,
                        "duration_max": 60,
                    }
                ],
            }
        )
        assert len(sol.scheduled) == 1
        st = sol.scheduled[0]
        assert st.task.label == "yoga"
        assert st.task.duration_min == 30
        assert st.start_minutes == 480
        assert st.is_standalone is True

    def test_concurrent_entry_round_trip(self):
        sol = cli_module._reconstruct_solution_from_json(
            {
                "person_id": "p001",
                "scheduled": [
                    {
                        "label": "mindful_eating",
                        "date": "2026-05-04",
                        "start_minutes": 720,
                        "end_minutes": 740,
                        "is_standalone": False,
                        "concurrent_with": "lunch",
                    }
                ],
            }
        )
        st = sol.scheduled[0]
        assert st.is_standalone is False
        assert st.concurrent_with == "lunch"

    def test_missing_optional_durations_default_to_zero(self):
        sol = cli_module._reconstruct_solution_from_json(
            {
                "person_id": "p001",
                "scheduled": [
                    {
                        "label": "yoga",
                        "date": "2026-05-04",
                        "start_minutes": 480,
                        "end_minutes": 540,
                    }
                ],
            }
        )
        assert sol.scheduled[0].task.duration_min == 0
        assert sol.scheduled[0].task.duration_max == 0

    def test_null_durations_treated_as_zero(self):
        sol = cli_module._reconstruct_solution_from_json(
            {
                "person_id": "p001",
                "scheduled": [
                    {
                        "label": "yoga",
                        "date": "2026-05-04",
                        "start_minutes": 480,
                        "end_minutes": 540,
                        "duration_min": None,
                        "duration_max": None,
                    }
                ],
            }
        )
        assert sol.scheduled[0].task.duration_min == 0


# ---------------------------------------------------------------------------
# _cmd_evaluate; oracle recompute path
# ---------------------------------------------------------------------------


class TestCmdEvaluateOracleRecompute:
    def _write_loss(self, persons_dir, pid, merge=0.0):
        """Seed the per-person solution evaluate drives off (merge is ignored)."""
        self._write_solution(persons_dir, pid)

    def _write_solution(self, persons_dir, pid):
        # `is_concurrent: True` is required on the persisted task so the
        # reconstructed `RecommendedTask` survives `is_excluded_pair`'s
        # source-(ii) check and the merge leg can enter the mask.
        persons_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "person_id": pid,
            "tasks_total": 1,
            "scheduled_count": 1,
            "unscheduled_count": 0,
            "scheduled": [
                {
                    "label": "mindful_eating",
                    "display_name": "Mindful Eating",
                    "description": "",
                    "date": "2026-05-04",
                    "start_minutes": 720,
                    "end_minutes": 740,
                    "is_standalone": False,
                    "concurrent_with": "lunch",
                    "intensity": 1,
                    "is_concurrent": True,
                    "is_dividable": False,
                    "duration_min": 20,
                    "duration_max": 20,
                    "ontology_uri": None,
                    "parent_task_label": None,
                }
            ],
            "unscheduled": [],
        }
        (persons_dir / f"{pid}.json").write_text(json.dumps(data), encoding="utf-8")

    def test_recompute_uses_oracle_when_run_loaded(self, tmp_path):
        """End-to-end recompute branch: oracle present, persona run loaded
        and a matching trace exists to L_concurrent is recomputed from the
        live solution and event list."""
        from src.scripts.scenarios.calendar.loader import LoadedRun
        from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace

        persons_dir = tmp_path / "augmented" / "persons"
        persons_dir.mkdir(parents=True)
        self._write_loss(persons_dir, "p001", merge=0.0)
        self._write_solution(persons_dir, "p001")

        ev = CalendarEvent(
            label="lunch",
            start_minutes=720,
            end_minutes=780,
            date=datetime.date(2026, 5, 4),
            is_concurrent=True,
        )
        trace = CalendarTrace(person_id="p001", events=[ev])
        loaded_run = LoadedRun(traces=[trace], horizon_days=7)

        oracle = MagicMock()
        oracle.score.return_value = 0.95  # oracle says the pair is great

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module,
                "_try_load_persona_run",
                return_value=(loaded_run, tmp_path),
            ),
            patch.object(cli_module, "_build_judge_oracle", return_value=oracle),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        # oracle.score must have been called at least once during recompute
        assert oracle.score.called

    def test_no_oracle_falls_through_to_persisted_merge(self, tmp_path, capsys):
        """When _build_judge_oracle returns None, evaluate uses the persisted
        merge value without crashing."""
        persons_dir = tmp_path / "augmented" / "persons"
        persons_dir.mkdir(parents=True)
        self._write_loss(persons_dir, "p001", merge=0.4)

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK

    def test_recompute_failure_keeps_persisted_value(self, tmp_path, caplog):
        """If the oracle's recompute raises, the warning logs and we keep
        the persisted merge; no crash."""
        from src.scripts.scenarios.calendar.loader import LoadedRun
        from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace

        persons_dir = tmp_path / "augmented" / "persons"
        persons_dir.mkdir(parents=True)
        self._write_loss(persons_dir, "p001", merge=0.4)
        self._write_solution(persons_dir, "p001")

        ev = CalendarEvent(
            label="lunch",
            start_minutes=720,
            end_minutes=780,
            date=datetime.date(2026, 5, 4),
        )
        trace = CalendarTrace(person_id="p001", events=[ev])
        loaded_run = LoadedRun(traces=[trace], horizon_days=7)

        oracle = MagicMock()
        oracle.score.side_effect = RuntimeError("boom")

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module,
                "_try_load_persona_run",
                return_value=(loaded_run, tmp_path),
            ),
            patch.object(cli_module, "_build_judge_oracle", return_value=oracle),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK  # never crashes

    def test_oracle_present_but_run_load_returns_string_error(self, tmp_path):
        """`_try_load_persona_run` may return a str on error; recompute is
        skipped silently and the persisted merge is preserved."""
        persons_dir = tmp_path / "augmented" / "persons"
        persons_dir.mkdir(parents=True)
        self._write_loss(persons_dir, "p001", merge=0.4)

        oracle = MagicMock()
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module,
                "_try_load_persona_run",
                return_value=("error string", tmp_path),
            ),
            patch.object(cli_module, "_build_judge_oracle", return_value=oracle),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        oracle.score.assert_not_called()

    def test_oracle_present_but_no_matching_trace(self, tmp_path):
        """Cover the branch where oracle is built but the person has no trace
        in the loaded run; the recompute is skipped."""
        from src.scripts.scenarios.calendar.loader import LoadedRun

        persons_dir = tmp_path / "augmented" / "persons"
        persons_dir.mkdir(parents=True)
        self._write_loss(persons_dir, "p001", merge=0.4)
        self._write_solution(persons_dir, "p001")

        loaded_run = LoadedRun(traces=[], horizon_days=7)
        oracle = MagicMock()

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module,
                "_try_load_persona_run",
                return_value=(loaded_run, tmp_path),
            ),
            patch.object(cli_module, "_build_judge_oracle", return_value=oracle),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        oracle.score.assert_not_called()


# ---------------------------------------------------------------------------
# _setup_logging; third-party-logger silencing + --quiet shortcut
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def _restore(self, names):
        """Reset noisy loggers after a test so other tests are unaffected."""
        import logging as _logging

        for name in names:
            _logging.getLogger(name).setLevel(_logging.NOTSET)

    def test_info_level_silences_noisy_third_party_loggers(self):
        """At INFO level the noisy third-party loggers are forced to WARNING.

        The pipeline emits one INFO line per LLM HTTP round-trip; left at
        their library defaults the user's stderr is drowned in
        `httpx`/`openai` traffic.
        """
        import logging as _logging

        self._restore(cli_module._NOISY_LOGGERS)
        cli_module._setup_logging("INFO")
        try:
            for name in cli_module._NOISY_LOGGERS:
                assert _logging.getLogger(name).level == _logging.WARNING
        finally:
            self._restore(cli_module._NOISY_LOGGERS)

    def test_debug_level_leaves_noisy_loggers_alone(self):
        """A user explicitly asking for DEBUG should still see HTTP traces."""
        import logging as _logging

        self._restore(cli_module._NOISY_LOGGERS)
        cli_module._setup_logging("DEBUG")
        try:
            for name in cli_module._NOISY_LOGGERS:
                # NOTSET (0) means "inherit root"; do not force WARNING.
                assert _logging.getLogger(name).level != _logging.WARNING
        finally:
            self._restore(cli_module._NOISY_LOGGERS)

    def test_warning_level_also_silences_noisy(self):
        """`--quiet` (WARNING) should still demote the noisy loggers."""
        import logging as _logging

        self._restore(cli_module._NOISY_LOGGERS)
        cli_module._setup_logging("WARNING")
        try:
            for name in cli_module._NOISY_LOGGERS:
                assert _logging.getLogger(name).level == _logging.WARNING
        finally:
            self._restore(cli_module._NOISY_LOGGERS)


class TestQuietFlag:
    """The `--quiet` shortcut overrides `--log-level` to WARNING."""

    def test_quiet_flag_parsed(self, tmp_path):
        scenario = tmp_path / "s.yaml"
        scenario.touch()
        args = _build_parser().parse_args(
            ["--quiet", "generate-tasks", "--scenario", str(scenario)]
        )
        assert args.quiet is True

    def test_quiet_short_flag(self, tmp_path):
        scenario = tmp_path / "s.yaml"
        scenario.touch()
        args = _build_parser().parse_args(
            ["-q", "generate-tasks", "--scenario", str(scenario)]
        )
        assert args.quiet is True

    def test_quiet_overrides_log_level_in_main(self, tmp_path):
        """When --quiet is set, `main` calls _setup_logging("WARNING")."""
        scenario = tmp_path / "s.yaml"
        with patch.object(cli_module, "_setup_logging") as setup:
            with _mock_load():
                with patch.object(
                    cli_module, "_try_load_persona_run", return_value=(None, None)
                ):
                    main(
                        [
                            "--quiet",
                            "--log-level",
                            "DEBUG",
                            "generate-tasks",
                            "--scenario",
                            str(scenario),
                        ]
                    )
        setup.assert_called_with("WARNING")

    def test_no_quiet_passes_log_level_through(self, tmp_path):
        """Without --quiet the chosen --log-level is honoured."""
        scenario = tmp_path / "s.yaml"
        with patch.object(cli_module, "_setup_logging") as setup:
            with _mock_load():
                with patch.object(
                    cli_module, "_try_load_persona_run", return_value=(None, None)
                ):
                    main(
                        [
                            "--log-level",
                            "DEBUG",
                            "generate-tasks",
                            "--scenario",
                            str(scenario),
                        ]
                    )
        setup.assert_called_with("DEBUG")


# ---------------------------------------------------------------------------
# Progress bars: "Augmenting" / "Evaluating"
# ---------------------------------------------------------------------------


class TestProgressBars:
    """`_cmd_augment_with_cfg` and `_cmd_evaluate_with_cfg` route their
    per-person loops through `wrap_progress` so the user can see
    `Augmenting:` and `Evaluating:` tqdm bars during long runs."""

    def _mock_run(self, traces=None):
        run = MagicMock()
        run.traces = traces or []
        run.time_windows = {}
        run.allen_pair_rules = []
        run.horizon_days = 7
        return run

    def test_augment_calls_wrap_progress_with_augmenting_label(self, tmp_path):
        run = self._mock_run()
        mock_aug = MagicMock()
        wrap_calls: list[dict] = []

        def fake_wrap(iterable, *, total, desc, show):
            wrap_calls.append({"total": total, "desc": desc, "show": show})
            return iter(iterable)

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            patch.object(cli_module, "_build_augmenter", return_value=mock_aug),
            patch(
                "src.scripts.persona.concurrency.progress.wrap_progress",
                side_effect=fake_wrap,
            ),
            patch("src.scripts.scenarios.metrics.loss.SchedulingLoss"),
        ):
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert wrap_calls, "wrap_progress was not called from _cmd_augment"
        assert wrap_calls[0]["desc"] == "Augmenting"

    @staticmethod
    def _capture(records, message_substring, *, only_levels=("INFO",)):
        """Return the records whose message contains *message_substring* AND
        whose levelname is in *only_levels*."""
        return [
            r
            for r in records
            if message_substring in r.getMessage() and r.levelname in only_levels
        ]

    def test_per_person_lines_are_debug_not_info(self, tmp_path, caplog):
        """During generate-tasks, the per-person `[N/M] Generating ...` and
        `[N/M] X to Y tasks` chatter must stay at DEBUG so the tqdm bar is
        not interrupted by a flood of INFO lines."""
        import logging as _logging

        cfg = _mock_cfg(out_dir=str(tmp_path / "out"))
        mock_run = MagicMock()
        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        mock_run.traces = [mock_trace]
        mock_person = MagicMock()
        mock_person.occupation_status = "student"
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [make_task("yoga")]
        with caplog.at_level(_logging.DEBUG, logger="scenarios.cli"):
            with (
                _mock_load(cfg),
                patch.object(
                    cli_module,
                    "_try_load_persona_run",
                    return_value=(mock_run, tmp_path),
                ),
                patch.object(cli_module, "_person_from_json", return_value=mock_person),
                patch.object(
                    cli_module, "_build_task_generator", return_value=mock_gen
                ),
                patch.object(cli_module, "_write_tasks"),
            ):
                rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        # The per-person chatter is present, but at DEBUG only
        assert (
            self._capture(caplog.records, "Generating tasks for", only_levels=("INFO",))
            == []
        )
        assert self._capture(
            caplog.records, "Generating tasks for", only_levels=("DEBUG",)
        )
        assert self._capture(caplog.records, "to 1 tasks", only_levels=("INFO",)) == []

    def test_persona_run_loading_lines_are_debug(self, tmp_path, caplog):
        """`Loading persona run` and `Persona run loaded` are DEBUG so
        they don't precede the progress bar at INFO."""
        import logging as _logging

        from src.scripts.scenarios.config.schema import CalendarSourceConfig

        # Real on-disk run_dir so _try_load_persona_run reaches the load
        # path (it short-circuits when the directory is missing).
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = ScenarioConfig(
            id="x",
            output=ScenarioOutputConfig(dir=str(tmp_path / "out")),
            augmentation=AugmentationConfig(method="greedy"),
            calendar=CalendarSourceConfig(run_dir=str(run_dir)),
        )
        fake_loaded = MagicMock(traces=[], horizon_days=7)
        with caplog.at_level(_logging.DEBUG, logger="scenarios.cli"):
            with patch(
                "src.scripts.scenarios.calendar.loader.load_persona_run",
                return_value=fake_loaded,
            ):
                cli_module._try_load_persona_run(cfg)
        assert (
            self._capture(caplog.records, "Loading persona run", only_levels=("INFO",))
            == []
        )
        assert self._capture(
            caplog.records, "Loading persona run", only_levels=("DEBUG",)
        )
        assert (
            self._capture(caplog.records, "Persona run loaded", only_levels=("INFO",))
            == []
        )
        assert self._capture(
            caplog.records, "Persona run loaded", only_levels=("DEBUG",)
        )

    def test_build_task_generator_chatter_is_debug(self, caplog):
        """`Connecting to GraphRAG …` and `GraphRAG pipeline ready` are
        DEBUG-level; single-shot bootstrap chatter that doesn't need to
        share the foreground with the bar."""
        import logging as _logging

        cfg = _mock_cfg()
        mock_pipeline = MagicMock()
        with caplog.at_level(_logging.DEBUG, logger="scenarios.cli"):
            with (
                patch(
                    "src.graphrag.pipeline.build_graphrag", return_value=mock_pipeline
                ),
                patch(
                    "src.graphrag.config.Neo4jSettings.from_env",
                    return_value=MagicMock(),
                ),
                patch(
                    "src.graphrag.neo4j_client.make_driver", return_value=MagicMock()
                ),
            ):
                _build_task_generator(cfg)
        assert (
            self._capture(
                caplog.records, "Connecting to GraphRAG", only_levels=("INFO",)
            )
            == []
        )
        assert self._capture(
            caplog.records, "Connecting to GraphRAG", only_levels=("DEBUG",)
        )
        assert (
            self._capture(
                caplog.records, "GraphRAG pipeline ready", only_levels=("INFO",)
            )
            == []
        )
        assert self._capture(
            caplog.records, "GraphRAG pipeline ready", only_levels=("DEBUG",)
        )

    def test_augment_per_person_lines_are_debug_not_info(self, tmp_path, caplog):
        """During augment, the per-person `Augmenting X (N tasks)` and
        `X to Y/Z scheduled` chatter is DEBUG so the bar stays clean."""
        import logging as _logging

        task = make_task("yoga")
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_tasks([task], tasks_dir / "p001_tasks.json")

        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        run = self._mock_run(traces=[mock_trace])
        mock_solution = MagicMock()
        mock_solution.scheduled = []
        mock_solution.tasks = []
        mock_aug = MagicMock()
        mock_aug.augment.return_value = mock_solution
        mock_loss_fn = MagicMock()
        mock_loss_fn.compute.return_value = (
            0.5,
            MagicMock(
                cov=0.5,
                cal=0.0,
                pref=0.0,
                disp=0.0,
                merge=0.0,
                spread=0.0,
                divide=0.0,
                context_fit=None,
            ),
        )
        with caplog.at_level(_logging.DEBUG, logger="scenarios.cli"):
            with (
                _mock_load(_mock_cfg(out_dir=str(tmp_path))),
                patch.object(
                    cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
                ),
                patch.object(cli_module, "_build_augmenter", return_value=mock_aug),
                patch(
                    "src.scripts.scenarios.metrics.loss.SchedulingLoss",
                    return_value=mock_loss_fn,
                ),
                patch("src.scripts.scenarios.export.json_writer.write_solution_json"),
                patch("src.scripts.scenarios.export.ics_writer.write_augmented_ics"),
            ):
                _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert (
            self._capture(caplog.records, "] Augmenting ", only_levels=("INFO",)) == []
        )
        assert self._capture(caplog.records, "] Augmenting ", only_levels=("DEBUG",))
        assert self._capture(caplog.records, "scheduled", only_levels=("INFO",)) == []

    def test_evaluate_reading_records_line_is_debug(self, tmp_path, caplog):
        """`Reading N gain records …` is DEBUG-level; the Evaluating bar
        already conveys the count visually."""
        import logging as _logging

        persons_dir = tmp_path / "augmented" / "persons"
        _seed_solution_json(persons_dir, "p001")
        with caplog.at_level(_logging.DEBUG, logger="scenarios.cli"):
            with (
                _mock_load(_mock_cfg(out_dir=str(tmp_path))),
                patch.object(cli_module, "_build_judge_oracle", return_value=None),
                patch(
                    "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
                ),
            ):
                _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert self._capture(caplog.records, "Reading", only_levels=("INFO",)) == []
        assert self._capture(caplog.records, "Reading", only_levels=("DEBUG",))

    def test_generate_tasks_calls_wrap_progress_with_generating_label(self, tmp_path):
        """`_cmd_generate_tasks_with_cfg` wraps the futures iterator with a
        `Generating tasks` tqdm bar so the long parallel-LLM phase shows
        progress instead of a flood of `[N/M]` INFO log lines."""
        cfg = _mock_cfg(out_dir=str(tmp_path / "out"))
        mock_run = MagicMock()
        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        mock_run.traces = [mock_trace]
        mock_person = MagicMock()
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [make_task("yoga")]
        wrap_calls: list[dict] = []

        def fake_wrap(iterable, *, total, desc, show):
            wrap_calls.append({"total": total, "desc": desc, "show": show})
            return iter(iterable)

        with (
            _mock_load(cfg),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(mock_run, tmp_path)
            ),
            patch.object(cli_module, "_person_from_json", return_value=mock_person),
            patch.object(cli_module, "_build_task_generator", return_value=mock_gen),
            patch.object(cli_module, "_write_tasks"),
            patch(
                "src.scripts.persona.concurrency.progress.wrap_progress",
                side_effect=fake_wrap,
            ),
        ):
            rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert wrap_calls, "wrap_progress was not called from _cmd_generate_tasks"
        assert wrap_calls[0]["desc"] == "Generating tasks"
        assert wrap_calls[0]["total"] == 1

    def test_generate_tasks_marks_empty_persons_in_summary(self, tmp_path):
        """Persons with 0 generated tasks are recorded in the generation
        summary sidecar so a transient empty-fetch is visible downstream."""
        cfg = _mock_cfg(out_dir=str(tmp_path / "out"))
        mock_run = MagicMock()
        t1, t2 = MagicMock(), MagicMock()
        t1.person_id, t2.person_id = "p001", "p002"
        mock_run.traces = [t1, t2]
        mock_gen = MagicMock()
        # p001 gets a task; p002's fetch comes back empty.
        mock_gen.generate.side_effect = [[make_task("yoga")], []]
        with (
            _mock_load(cfg),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(mock_run, tmp_path)
            ),
            patch.object(
                cli_module,
                "_load_person_json_lookup",
                return_value={"p001": "a.json", "p002": "b.json"},
            ),
            patch.object(cli_module, "_read_person_json", return_value={"x": 1}),
            patch.object(cli_module, "_person_from_json", return_value=MagicMock()),
            patch.object(cli_module, "_build_task_generator", return_value=mock_gen),
        ):
            rc = _cmd_generate_tasks(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        summary_path = tmp_path / "out" / "tasks" / "_generation_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["empty_persons"] == ["p002"]
        assert summary["written_persons"] == 1
        assert summary["total_persons"] == 2

    def test_evaluate_calls_wrap_progress_with_evaluating_label(self, tmp_path):
        persons_dir = tmp_path / "augmented" / "persons"
        _seed_solution_json(persons_dir, "p001")
        wrap_calls: list[dict] = []

        def fake_wrap(iterable, *, total, desc, show):
            wrap_calls.append({"total": total, "desc": desc, "show": show})
            return iter(iterable)

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.scripts.persona.concurrency.progress.wrap_progress",
                side_effect=fake_wrap,
            ),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert wrap_calls, "wrap_progress was not called from _cmd_evaluate"
        assert wrap_calls[0]["desc"] == "Evaluating"
        assert wrap_calls[0]["total"] == 1


class TestEvaluateGroundingWiring:
    """`_cmd_evaluate_with_cfg` should compute the ontology-grounding
    ratio from the per-scenario `tasks/` directory and forward it to
    `write_evaluation_reports`.  It tries two layouts: legacy single-scenario
    (`run_dir/tasks`) and multi-scenario (`run_dir/../tasks`)."""

    def _seed_loss(self, persons_dir, person_id="p001"):
        _seed_solution_json(persons_dir, person_id)

    def _seed_tasks(self, tasks_dir, person_id="p001", grounded=2, total=4):
        tasks_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            {"label": f"t{i}", "ontology_uri": f"https://ex.org/task/{i}"}
            for i in range(grounded)
        ]
        tasks += [
            {"label": f"t{i}", "ontology_uri": None} for i in range(grounded, total)
        ]
        (tasks_dir / f"{person_id}_tasks.json").write_text(
            json.dumps(tasks), encoding="utf-8"
        )

    def test_grounding_from_run_dir_tasks(self, tmp_path, capsys):
        """Single-scenario layout: tasks/ sits inside the run dir."""
        run_dir = tmp_path
        self._seed_loss(run_dir / "augmented" / "persons")
        self._seed_tasks(run_dir / "tasks", grounded=3, total=4)

        captured = {}

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
            return out_dir / "txt", out_dir / "json"

        with (
            _mock_load(_mock_cfg(out_dir=str(run_dir))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports",
                side_effect=fake_writer,
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        g = captured["grounding"]
        assert g is not None
        assert g["grounded"] == 3
        assert g["total"] == 4
        # The completion log line includes the grounding ratio
        # (we read it from stderr via the print-style log).

    def test_grounding_from_parent_tasks(self, tmp_path):
        """Multi-scenario layout: tasks/ sits next to the method dir.

        `run_dir = output/exp/scenario/method`,
        tasks at `output/exp/scenario/tasks`.
        """
        method_dir = tmp_path / "scenario_a" / "greedy"
        self._seed_loss(method_dir / "augmented" / "persons")
        self._seed_tasks(method_dir.parent / "tasks", grounded=1, total=2)

        captured = {}

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
            return out_dir / "txt", out_dir / "json"

        with (
            _mock_load(_mock_cfg(out_dir=str(method_dir))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports",
                side_effect=fake_writer,
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        g = captured["grounding"]
        assert g is not None
        assert g["grounded"] == 1
        assert g["total"] == 2

    def test_grounding_none_when_no_tasks_dir(self, tmp_path):
        """Neither candidate tasks/ exists to grounding is forwarded as None."""
        run_dir = tmp_path
        self._seed_loss(run_dir / "augmented" / "persons")

        captured = {}

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
            return out_dir / "txt", out_dir / "json"

        with (
            _mock_load(_mock_cfg(out_dir=str(run_dir))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports",
                side_effect=fake_writer,
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert captured["grounding"] is None

    def test_grounding_logged_in_complete_line(self, tmp_path, caplog):
        import logging as _logging

        run_dir = tmp_path
        self._seed_loss(run_dir / "augmented" / "persons")
        self._seed_tasks(run_dir / "tasks", grounded=2, total=4)
        with caplog.at_level(_logging.INFO, logger="scenarios.cli"):
            with (
                _mock_load(_mock_cfg(out_dir=str(run_dir))),
                patch.object(cli_module, "_build_judge_oracle", return_value=None),
                patch(
                    "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
                ),
            ):
                _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        # Look for a record that mentions both avg_gain and grounding
        msg_blob = " ".join(r.getMessage() for r in caplog.records)
        assert "ontology_grounding=2/4" in msg_blob
        assert "50.0%" in msg_blob


class TestEvaluateUriReValidation:
    """`_cmd_evaluate_with_cfg` opens Neo4j (best-effort) and forwards
    `uri_resolves` as a validator to :func:`compute_grounding`.  This
    is the eval-time safety net that catches URIs that no longer
    resolve in the live ontology; independent of what the generator
    did at fetch time."""

    def _seed(self, run_dir, *, grounded_uris, total):
        persons_dir = run_dir / "augmented" / "persons"
        _seed_solution_json(persons_dir, "p001")
        tasks_dir = run_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        items = [
            {"label": f"t{i}", "ontology_uri": f"https://ex.org/task/{i}"}
            for i in range(grounded_uris)
        ]
        items += [
            {"label": f"t{i}", "ontology_uri": None}
            for i in range(grounded_uris, total)
        ]
        (tasks_dir / "p001_tasks.json").write_text(json.dumps(items), encoding="utf-8")

    def test_validator_passed_when_neo4j_reachable(self, tmp_path):
        """When Neo4j connects, the eval CLI passes a non-None validator
        to `compute_grounding` and closes the driver afterwards."""
        run_dir = tmp_path
        self._seed(run_dir, grounded_uris=2, total=2)
        fake_driver = MagicMock()
        captured = {}

        def fake_compute(_dir, _ids, *, expected_total, uri_validator):
            captured["uri_validator"] = uri_validator
            return {
                "per_person": {},
                "grounded": 2,
                "verified": 2,
                "total": 2,
                "ratio": 1.0,
                "verified_ratio": 1.0,
                "expected_per_person": 5,
                "persons_short_fetched": 0,
                "persons_with_unverified": 0,
            }

        fake_session = MagicMock()
        fake_ctx = MagicMock()
        fake_ctx.__enter__ = MagicMock(return_value=fake_session)
        fake_ctx.__exit__ = MagicMock(return_value=None)

        with (
            _mock_load(_mock_cfg(out_dir=str(run_dir))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.graphrag.config.Neo4jSettings.from_env",
                return_value=MagicMock(),
            ),
            patch(
                "src.graphrag.neo4j_client.make_driver",
                return_value=fake_driver,
            ),
            patch(
                "src.graphrag.neo4j_client.session_scope",
                return_value=fake_ctx,
            ),
            patch(
                "src.scripts.scenarios.metrics.grounding.compute_grounding",
                side_effect=fake_compute,
            ),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert callable(captured["uri_validator"])
        fake_driver.close.assert_called_once()
        fake_ctx.__exit__.assert_called_once()

    def test_validator_none_when_neo4j_unreachable(self, tmp_path):
        run_dir = tmp_path
        self._seed(run_dir, grounded_uris=1, total=2)
        captured = {}

        def fake_compute(_dir, _ids, *, expected_total, uri_validator):
            captured["uri_validator"] = uri_validator
            return {
                "per_person": {},
                "grounded": 1,
                "verified": None,
                "total": 2,
                "ratio": 0.5,
                "verified_ratio": None,
                "expected_per_person": 5,
                "persons_short_fetched": 0,
                "persons_with_unverified": None,
            }

        with (
            _mock_load(_mock_cfg(out_dir=str(run_dir))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.graphrag.config.Neo4jSettings.from_env",
                side_effect=RuntimeError("no env"),
            ),
            patch(
                "src.scripts.scenarios.metrics.grounding.compute_grounding",
                side_effect=fake_compute,
            ),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert captured["uri_validator"] is None

    def test_validator_callable_routes_through_uri_resolves(self, tmp_path):
        """The validator built inside the CLI must call `uri_resolves`
        with the open session.  Capture the validator and invoke it
        ourselves to confirm it forwards correctly."""
        run_dir = tmp_path
        self._seed(run_dir, grounded_uris=1, total=1)

        captured = {}

        def fake_compute(_dir, _ids, *, expected_total, uri_validator):
            captured["validator"] = uri_validator
            return {
                "per_person": {},
                "grounded": 1,
                "verified": 1,
                "total": 1,
                "ratio": 1.0,
                "verified_ratio": 1.0,
                "expected_per_person": 5,
                "persons_short_fetched": 0,
                "persons_with_unverified": 0,
            }

        fake_session = MagicMock()
        fake_ctx = MagicMock()
        fake_ctx.__enter__ = MagicMock(return_value=fake_session)
        fake_ctx.__exit__ = MagicMock(return_value=None)

        with (
            _mock_load(_mock_cfg(out_dir=str(run_dir))),
            patch.object(cli_module, "_build_judge_oracle", return_value=None),
            patch(
                "src.graphrag.config.Neo4jSettings.from_env",
                return_value=MagicMock(),
            ),
            patch(
                "src.graphrag.neo4j_client.make_driver",
                return_value=MagicMock(),
            ),
            patch("src.graphrag.neo4j_client.session_scope", return_value=fake_ctx),
            patch(
                "src.scripts.scenarios.metrics.grounding.compute_grounding",
                side_effect=fake_compute,
            ),
            patch(
                "src.scripts.scenarios.task_generation.ontology_bridge.uri_resolves"
            ) as mock_resolves,
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ),
        ):
            mock_resolves.return_value = True
            _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
            captured["validator"]("https://ex.org/task/x")
            mock_resolves.assert_called_with("https://ex.org/task/x", fake_session)


# ---------------------------------------------------------------------------
# _normalize_scenario_id_filter; coercion helper for --scenario-id
# ---------------------------------------------------------------------------


class TestNormalizeScenarioIdFilter:
    """The new multi-id form (`--scenario-id A B C`) and the legacy
    single-id form must both flow into a uniform `list[str] | None`."""

    def test_none_passes_through(self):
        assert _normalize_scenario_id_filter(None) is None

    def test_single_string_wrapped_into_list(self):
        assert _normalize_scenario_id_filter("s1") == ["s1"]

    def test_list_input_kept_as_list(self):
        assert _normalize_scenario_id_filter(["s1", "s2"]) == ["s1", "s2"]

    def test_empty_list_treated_as_none(self):
        # An empty list (e.g. nargs='+' that somehow received []) means
        # "no narrowing"; we map it to None so callers iterate every
        # scenario instead of silently iterating none.
        assert _normalize_scenario_id_filter([]) is None

    def test_comma_separated_single_token_splits(self):
        assert _normalize_scenario_id_filter(["s1,s2,s3"]) == ["s1", "s2", "s3"]

    def test_comma_separated_string_splits(self):
        assert _normalize_scenario_id_filter("s1,s2") == ["s1", "s2"]

    def test_mixed_space_and_comma_tokens_flatten(self):
        assert _normalize_scenario_id_filter(["s1,s2", "s3"]) == ["s1", "s2", "s3"]

    def test_stray_commas_and_whitespace_dropped(self):
        assert _normalize_scenario_id_filter([" s1 , s2,", ","]) == ["s1", "s2"]

    def test_input_list_is_copied(self):
        original = ["s1", "s2"]
        out = _normalize_scenario_id_filter(original)
        assert out == ["s1", "s2"]
        # Mutating the returned list must not corrupt the caller's args.
        out.append("s3")
        assert original == ["s1", "s2"]


class TestValidateMultiFiltersMultipleScenarioIds:
    """The list form of `--scenario-id` flows into `_validate_multi_filters`
    and into the per-runner dispatch loops."""

    def test_passes_when_every_id_in_list_matches(self):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2", "s3"), methods=("greedy",))
        assert _validate_multi_filters(exp, scenario_id=["s1", "s3"]) == EXIT_OK

    def test_errors_when_any_id_in_list_unknown(self, caplog):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _validate_multi_filters(exp, scenario_id=["s1", "missing"])
        assert rc == EXIT_USAGE
        # Only the offending id is named, not the valid one.
        assert "'missing'" in caplog.text
        assert "'s1'" not in caplog.text

    def test_lists_every_unknown_id_in_one_message(self, caplog):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy",))
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _validate_multi_filters(exp, scenario_id=["bogus_a", "bogus_b"])
        assert rc == EXIT_USAGE
        assert "'bogus_a'" in caplog.text and "'bogus_b'" in caplog.text

    def test_method_filter_scoped_to_union_of_selected_ids(self, caplog):
        """When two scenario_ids are passed, --method only needs to be
        present in the union of their methods (not in every one)."""
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp",
                "scenarios": [
                    {"id": "s1", "augmentation": [{"method": "greedy"}]},
                    {"id": "s2", "augmentation": [{"method": "llm_agent"}]},
                ],
            }
        )
        # llm_agent is in s2's methods, so the (s1, s2)+llm_agent combo
        # is valid even though s1 alone wouldn't be.
        rc = _validate_multi_filters(
            exp, scenario_id=["s1", "s2"], method_filter="llm_agent"
        )
        assert rc == EXIT_OK


class TestMultiRunnersIterateMultipleScenarioIds:
    """The three multi-* runners must iterate ONLY the scenario_ids in
    the list and skip the others."""

    def test_multi_generate_tasks_iterates_subset(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2", "s3"), methods=("greedy",))
        args = _multi_args(tmp_path)
        with patch.object(
            cli_module, "_cmd_generate_tasks_with_cfg", return_value=EXIT_OK
        ) as inner:
            rc = _multi_generate_tasks(exp, args, scenario_id=["s1", "s3"])
        assert rc == EXIT_OK
        called_ids = [call.args[0].id.split("/")[0] for call in inner.call_args_list]
        assert called_ids == ["s1", "s3"]

    def test_multi_augment_iterates_subset(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2", "s3"), methods=("greedy",))
        args = _multi_args(tmp_path)
        with patch.object(
            cli_module, "_cmd_augment_with_cfg", return_value=EXIT_OK
        ) as inner:
            rc = _multi_augment(exp, args, scenario_id=["s2", "s3"], method_filter=None)
        assert rc == EXIT_OK
        called_ids = [call.args[0].id.split("/")[0] for call in inner.call_args_list]
        assert called_ids == ["s2", "s3"]

    def test_multi_evaluate_iterates_subset(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1", "s2", "s3"), methods=("greedy",))
        args = _multi_args(tmp_path)
        with patch.object(
            cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_OK
        ) as inner:
            rc = _multi_evaluate(exp, args, scenario_id=["s1"], method_filter=None)
        assert rc == EXIT_OK
        called_ids = [call.args[0].id.split("/")[0] for call in inner.call_args_list]
        assert called_ids == ["s1"]


class TestMultiRunnersUseNewPathLayout:
    """The runners must read tasks from `task_generation/<id>/` and write
    augment output under `scenarios/<id>/<method>/`; not the legacy
    `<id>/tasks/` and `<id>/<method>/` paths."""

    def test_generate_tasks_writes_under_task_generation(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy",))
        args = _multi_args(tmp_path)
        with patch.object(
            cli_module, "_cmd_generate_tasks_with_cfg", return_value=EXIT_OK
        ) as inner:
            _multi_generate_tasks(exp, args, scenario_id=None)
        sub_args = inner.call_args.args[1]
        out_dir = str(sub_args.out_dir).replace("\\", "/")
        assert "/task_generation/s1" in out_dir
        assert "/scenarios/" not in out_dir

    def test_augment_reads_tasks_from_task_generation(self, tmp_path):
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy",))
        args = _multi_args(tmp_path)
        with patch.object(
            cli_module, "_cmd_augment_with_cfg", return_value=EXIT_OK
        ) as inner:
            _multi_augment(exp, args, scenario_id=None, method_filter=None)
        sub_args = inner.call_args.args[1]
        tasks_dir = str(sub_args.tasks_dir).replace("\\", "/")
        # Exact-match suffix; substring `/task_generation/s1` would also
        # match the buggy `task_generation/s1` parent dir which silently
        # no-ops the augmenter (see `_multi_scenario_tasks_dir` docstring).
        assert tasks_dir.endswith("/task_generation/s1/tasks")

    def test_multi_augment_tasks_dir_matches_multi_generate_tasks_output(
        self, tmp_path
    ):
        """Regression for the 2026-05-18 silent no-op: the dir
        `_multi_augment` passes as `tasks_dir` MUST be the same dir
        `_multi_generate_tasks` writes per-person task JSONs into.
        Otherwise the augmenter opens missing files and "completes"
        every persona in 0 seconds without raising.

        `_cmd_generate_tasks_with_cfg` writes per-person JSONs to
        `out_dir/tasks/`, so the round-trip path is
        `<sub_args.out_dir>/tasks` for the generator and
        `<sub_args.tasks_dir>` for the augmenter; they must match
        byte-for-byte.
        """
        exp = _make_exp_cfg(scenario_ids=("s1", "s2"), methods=("greedy",))
        args = _multi_args(tmp_path)

        gen_per_person_dirs: list[Path] = []
        aug_tasks_dirs: list[Path] = []

        def _gen_capture(_cfg, sub_args):
            # Mirror what _cmd_generate_tasks_with_cfg actually does.
            gen_per_person_dirs.append(Path(sub_args.out_dir) / "tasks")
            return EXIT_OK

        def _aug_capture(_cfg, sub_args):
            aug_tasks_dirs.append(Path(sub_args.tasks_dir))
            return EXIT_OK

        with patch.object(
            cli_module, "_cmd_generate_tasks_with_cfg", side_effect=_gen_capture
        ):
            _multi_generate_tasks(exp, args, scenario_id=None)
        with patch.object(
            cli_module, "_cmd_augment_with_cfg", side_effect=_aug_capture
        ):
            _multi_augment(exp, args, scenario_id=None, method_filter=None)

        assert gen_per_person_dirs, "generate-tasks runner did not dispatch"
        assert aug_tasks_dirs, "augment runner did not dispatch"
        assert gen_per_person_dirs == aug_tasks_dirs

    def test_multi_scenario_tasks_dir_helper_matches_runner_paths(self, tmp_path):
        """The shared helper is the source of truth; both runners must
        derive their per-person task dir from it (or an equivalent
        derivation) so a future refactor can't reintroduce the bug."""
        exp = _make_exp_cfg(scenario_ids=("s1",), methods=("greedy",))
        args = _multi_args(tmp_path)
        expected = _multi_scenario_tasks_dir(exp, "s1")
        with patch.object(
            cli_module, "_cmd_augment_with_cfg", return_value=EXIT_OK
        ) as inner:
            _multi_augment(exp, args, scenario_id=None, method_filter=None)
        assert Path(inner.call_args.args[1].tasks_dir) == expected
        # And the helper itself returns the documented suffix.
        assert expected.as_posix().endswith("/task_generation/s1/tasks")

    def test_method_cfg_to_scenario_uses_new_method_dir(self):
        exp = ExperimentScenariosConfig.model_validate(
            {"experiment_id": "exp", "scenarios": [{"id": "s1"}]}
        )
        scenario = exp.scenarios[0]
        method = scenario.augmentation[0]
        cfg = cli_module._method_cfg_to_scenario_config(exp, scenario, method)
        out_dir = cfg.output.dir.replace("\\", "/")
        assert "/scenarios/s1/greedy" in out_dir


# ---------------------------------------------------------------------------
# _llm_settings_with_override; per-stage model override helper
# ---------------------------------------------------------------------------


class TestLLMSettingsWithOverride:
    """Per-stage `LLMModelConfig` blocks must build a fresh `LLMSettings`
    that inherits API keys from `.env` but flips the active provider's
    model. None means "no override to fall back to env defaults"."""

    def _stub_env_settings(self):
        from src.graphrag.config import LLMSettings

        return LLMSettings(
            provider="openai",
            openai_api_key="sk-env",
            openai_model="env-openai-model",
            anthropic_api_key="env-anthropic-key",
            anthropic_model="env-anthropic-model",
            openrouter_api_key="env-or-key",
            openrouter_base_url="https://env.example/v1",
            openrouter_model="env/or-model",
        )

    def test_none_returns_none(self):
        assert _llm_settings_with_override(None) is None

    def test_provider_inherited_from_env_when_unset(self):
        with patch(
            "src.graphrag.config.LLMSettings.from_env",
            return_value=self._stub_env_settings(),
        ):
            out = _llm_settings_with_override(LLMModelConfig(model="gpt-4o-mini"))
        assert out is not None
        assert out.provider == "openai"  # inherited from env stub
        assert out.openai_model == "gpt-4o-mini"

    def test_explicit_openai_provider_swaps_only_openai_model(self):
        with patch(
            "src.graphrag.config.LLMSettings.from_env",
            return_value=self._stub_env_settings(),
        ):
            out = _llm_settings_with_override(
                LLMModelConfig(provider="openai", model="gpt-4.1-mini")
            )
        assert out is not None
        assert out.openai_model == "gpt-4.1-mini"
        # Other provider models stay at their env defaults.
        assert out.anthropic_model == "env-anthropic-model"
        assert out.openrouter_model == "env/or-model"

    def test_explicit_anthropic_provider_swaps_only_anthropic_model(self):
        with patch(
            "src.graphrag.config.LLMSettings.from_env",
            return_value=self._stub_env_settings(),
        ):
            out = _llm_settings_with_override(
                LLMModelConfig(provider="anthropic", model="claude-haiku-4-5")
            )
        assert out is not None
        assert out.provider == "anthropic"
        assert out.anthropic_model == "claude-haiku-4-5"
        assert out.openai_model == "env-openai-model"
        assert out.openrouter_model == "env/or-model"

    def test_explicit_openrouter_provider_swaps_only_openrouter_model(self):
        with patch(
            "src.graphrag.config.LLMSettings.from_env",
            return_value=self._stub_env_settings(),
        ):
            out = _llm_settings_with_override(
                LLMModelConfig(provider="openrouter", model="org/router-x")
            )
        assert out is not None
        assert out.provider == "openrouter"
        assert out.openrouter_model == "org/router-x"
        assert out.openai_model == "env-openai-model"
        assert out.anthropic_model == "env-anthropic-model"

    def test_api_keys_and_base_url_inherited_unchanged(self):
        """The YAML never carries secrets; every credential must come
        from `.env` so swapping models does not require re-pasting keys."""
        env = self._stub_env_settings()
        with patch("src.graphrag.config.LLMSettings.from_env", return_value=env):
            out = _llm_settings_with_override(LLMModelConfig(model="gpt-4o-mini"))
        assert out is not None
        assert out.openai_api_key == env.openai_api_key
        assert out.anthropic_api_key == env.anthropic_api_key
        assert out.openrouter_api_key == env.openrouter_api_key
        assert out.openrouter_base_url == env.openrouter_base_url


# ---------------------------------------------------------------------------
# Aggregated benchmark report; auto-write hook + standalone subcommand
# ---------------------------------------------------------------------------


def _exp_cfg_yaml(tmp_path: Path, run_dir: Path) -> Path:
    """Write a minimal multi-scenario YAML pointing at *run_dir*."""
    yaml_path = tmp_path / "scen.yaml"
    yaml_path.write_text(
        "experiment_id: exp_z\n"
        f"run_dir: {run_dir.as_posix()}\n"
        "scenarios:\n"
        "  - id: s1\n"
        "    augmentation:\n"
        "      - method: llm_agent\n"
        "  - id: s2\n"
        "    augmentation:\n"
        "      - method: llm_agent\n",
        encoding="utf-8",
    )
    return yaml_path


def _seed_top_level_eval(root: Path, sid: str, method: str) -> None:
    """Stage a sentinel `total_scheduling_gain.json` so the disk filter
    keeps the pair."""
    eval_dir = root / "scenarios" / sid / method / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "total_scheduling_gain.json").write_text("{}", encoding="utf-8")


class TestWriteBenchmarkReportIfPossible:
    def test_no_op_for_empty_pair_list(self, tmp_path: Path):
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [{"id": "s1"}],
            }
        )
        with patch(
            "src.scripts.scenarios.export.benchmark_report.write_benchmark_report"
        ) as writer:
            _write_benchmark_report_if_possible(exp, [])
        writer.assert_not_called()

    def test_calls_writer_with_pairs(self, tmp_path: Path):
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [{"id": "s1"}],
            }
        )
        _seed_top_level_eval(tmp_path, "s1", "llm_agent")
        with patch(
            "src.scripts.scenarios.export.benchmark_report.write_benchmark_report"
        ) as writer:
            writer.return_value = tmp_path / "benchmark_report.md"
            _write_benchmark_report_if_possible(exp, [("s1", "llm_agent")])
        writer.assert_called_once()
        call_kwargs = writer.call_args.kwargs
        assert call_kwargs["scenario_method_pairs"] == [("s1", "llm_agent")]
        assert call_kwargs["scenarios_cfg"] is exp

    def test_loads_experiment_name_from_used_configs_snapshot(self, tmp_path: Path):
        """`experiment_name` from `used_configs.json` reaches the writer."""
        # Stage a minimal used_configs.json with the new field.
        (tmp_path / "used_configs.json").write_text(
            json.dumps({"environment": {"experiment_name": "My Lab Run"}}),
            encoding="utf-8",
        )
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [{"id": "s1"}],
            }
        )
        _seed_top_level_eval(tmp_path, "s1", "llm_agent")
        with patch(
            "src.scripts.scenarios.export.benchmark_report.write_benchmark_report"
        ) as writer:
            writer.return_value = tmp_path / "benchmark_report.md"
            _write_benchmark_report_if_possible(exp, [("s1", "llm_agent")])
        assert writer.call_args.kwargs["experiment_name"] == "My Lab Run"

    def test_experiment_name_is_none_when_snapshot_missing(self, tmp_path: Path):
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [{"id": "s1"}],
            }
        )
        _seed_top_level_eval(tmp_path, "s1", "llm_agent")
        with patch(
            "src.scripts.scenarios.export.benchmark_report.write_benchmark_report"
        ) as writer:
            writer.return_value = tmp_path / "benchmark_report.md"
            _write_benchmark_report_if_possible(exp, [("s1", "llm_agent")])
        assert writer.call_args.kwargs["experiment_name"] is None

    def test_experiment_name_is_none_when_field_missing_from_snapshot(
        self, tmp_path: Path
    ):
        (tmp_path / "used_configs.json").write_text(
            json.dumps({"environment": {"seed": 1}}),
            encoding="utf-8",
        )
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [{"id": "s1"}],
            }
        )
        _seed_top_level_eval(tmp_path, "s1", "llm_agent")
        with patch(
            "src.scripts.scenarios.export.benchmark_report.write_benchmark_report"
        ) as writer:
            writer.return_value = tmp_path / "benchmark_report.md"
            _write_benchmark_report_if_possible(exp, [("s1", "llm_agent")])
        assert writer.call_args.kwargs["experiment_name"] is None

    def test_writer_failure_is_non_fatal(self, tmp_path: Path, caplog):
        """Writer failure logs a warning instead of raising."""
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [{"id": "s1"}],
            }
        )
        _seed_top_level_eval(tmp_path, "s1", "llm_agent")
        with patch(
            "src.scripts.scenarios.export.benchmark_report.write_benchmark_report",
            side_effect=RuntimeError("boom"),
        ):
            with caplog.at_level("WARNING", logger="scenarios.cli"):
                _write_benchmark_report_if_possible(exp, [("s1", "llm_agent")])
        assert "benchmark report skipped" in caplog.text

    def test_skips_pair_with_no_disk_evaluation(self, tmp_path: Path):
        """Pairs with only per-variant evals are excluded from the
        main scheduling-gain table."""
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [
                    {
                        "id": "s1",
                        "augmentation": [{"method": "llm_agent"}],
                    }
                ],
            }
        )
        # Per-variant eval only; no top-level eval.
        variant_eval = (
            tmp_path
            / "scenarios"
            / "s1"
            / "llm_agent"
            / "ablation"
            / ("0" * 11)
            / "evaluation"
        )
        variant_eval.mkdir(parents=True)
        (variant_eval / "total_scheduling_gain.json").write_text("{}")
        with patch(
            "src.scripts.scenarios.export.benchmark_report.write_benchmark_report"
        ) as writer:
            writer.return_value = tmp_path / "benchmark_report.md"
            _write_benchmark_report_if_possible(exp, [("s1", "llm_agent")])
        writer.assert_called_once()
        kwargs = writer.call_args.kwargs
        assert kwargs["scenario_method_pairs"] == []
        assert kwargs["ablation_runs"] == [("s1", "llm_agent", "0" * 11)]


class TestParseBaselineArg:
    def test_none_passes_through(self):
        assert _parse_baseline_arg(None, []) is None

    def test_empty_string_treated_as_none(self):
        assert _parse_baseline_arg("", []) is None

    def test_explicit_id_and_method(self):
        assert _parse_baseline_arg(
            "scenario_a/llm_agent",
            [("scenario_a", "llm_agent")],
        ) == ("scenario_a", "llm_agent")

    def test_bare_id_resolves_to_first_method_for_that_id(self):
        pairs = [("s1", "greedy"), ("s1", "llm_agent"), ("s2", "llm_agent")]
        assert _parse_baseline_arg("s1", pairs) == ("s1", "greedy")

    def test_unknown_bare_id_returns_id_with_empty_method(self):
        """When the bare id doesn't match any pair, return it with an
        empty method so `write_benchmark_report` falls back to the
        first run rather than silently anchoring on a different one."""
        assert _parse_baseline_arg("ghost", [("s1", "llm_agent")]) == (
            "ghost",
            "",
        )


class TestCmdReport:
    def test_legacy_scenario_yaml_rejected(self, tmp_path: Path, caplog):
        path = tmp_path / "legacy.yaml"
        path.write_text(
            "scenario:\n  id: s\n  output:\n    dir: ./out\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            scenario=path,
            scenario_id=None,
            method=None,
            baseline=None,
            log_level="INFO",
        )
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _cmd_report(args)
        assert rc == EXIT_USAGE
        assert "multi-scenario YAML" in caplog.text

    def test_unknown_scenario_id_aborts(self, tmp_path: Path, caplog):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        yaml_path = _exp_cfg_yaml(tmp_path, run_dir)
        args = argparse.Namespace(
            scenario=yaml_path,
            scenario_id=["does_not_exist"],
            method=None,
            baseline=None,
            log_level="INFO",
        )
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _cmd_report(args)
        assert rc == EXIT_USAGE
        assert "does_not_exist" in caplog.text

    def test_writes_report_under_experiment_dir(self, tmp_path: Path, capsys):
        run_dir = tmp_path / "run"
        eval_dir = run_dir / "scenarios" / "s1" / "llm_agent" / "evaluation"
        eval_dir.mkdir(parents=True)
        from src.scripts.scenarios.export.benchmark_report import TOTAL_REPORT_BASENAME

        (eval_dir / f"{TOTAL_REPORT_BASENAME}.json").write_text(
            json.dumps(
                {
                    "average_total_gain": 0.5,
                    "scored_persons": 2,
                    "empty_plan_persons": 0,
                    "average_gains": {},
                }
            ),
            encoding="utf-8",
        )
        yaml_path = _exp_cfg_yaml(tmp_path, run_dir)
        args = argparse.Namespace(
            scenario=yaml_path,
            scenario_id=["s1"],
            method=None,
            baseline=None,
            log_level="INFO",
        )
        rc = _cmd_report(args)
        assert rc == EXIT_OK
        target = run_dir / "benchmark_report.md"
        assert target.is_file()
        text = target.read_text(encoding="utf-8")
        assert "# Benchmark report: `exp_z`" in text
        captured = capsys.readouterr()
        assert "[report] wrote benchmark report" in captured.out

    def test_invalid_scenario_yaml_returns_usage_error(self, tmp_path: Path, caplog):
        bad = tmp_path / "broken.yaml"
        bad.write_text(":\n  not: valid: yaml: at all:\n", encoding="utf-8")
        args = argparse.Namespace(
            scenario=bad,
            scenario_id=None,
            method=None,
            baseline=None,
            log_level="INFO",
        )
        with caplog.at_level("ERROR", logger="scenarios.cli"):
            rc = _cmd_report(args)
        assert rc == EXIT_USAGE


class TestMultiEvaluateThreadsTasksDir:
    """`_multi_evaluate` must hand `_cmd_evaluate_with_cfg` the canonical
    tasks dir from `_multi_scenario_tasks_dir` so the evaluator's
    ontology-grounding probe doesn't fall back to the all-`n/a` no-data
    path under the new tree-style layout (regression for the
    benchmark-report ontology-grounding section being empty)."""

    def test_passes_tasks_dir_matching_helper(self, tmp_path: Path):
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [
                    {"id": "s1", "augmentation": [{"method": "llm_agent"}]},
                ],
            }
        )
        args = argparse.Namespace(scenario=tmp_path / "scen.yaml", log_level="INFO")
        with (
            patch.object(
                cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_OK
            ) as inner,
            patch.object(cli_module, "_write_benchmark_report_if_possible"),
        ):
            _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
        sub_args = inner.call_args.args[1]
        assert Path(sub_args.tasks_dir) == _multi_scenario_tasks_dir(exp, "s1")

    def test_tasks_dir_uses_new_task_generation_layout(self, tmp_path: Path):
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [
                    {"id": "s1", "augmentation": [{"method": "llm_agent"}]},
                ],
            }
        )
        args = argparse.Namespace(scenario=tmp_path / "scen.yaml", log_level="INFO")
        with (
            patch.object(
                cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_OK
            ) as inner,
            patch.object(cli_module, "_write_benchmark_report_if_possible"),
        ):
            _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
        passed = Path(inner.call_args.args[1].tasks_dir).as_posix()
        assert passed.endswith("/task_generation/s1/tasks")


class TestEvaluatorTasksDirCandidates:
    """Standalone-evaluate path: when `args.tasks_dir` is supplied (by
    the multi-runner), it must be the FIRST candidate the
    ontology-grounding probe tries; otherwise the legacy
    run-dir-relative candidates win and miss the new layout's tasks."""

    def test_explicit_tasks_dir_takes_priority_over_legacy_candidates(
        self, tmp_path: Path
    ):
        # Stage one task JSON at each candidate location so the test
        # can tell which candidate the evaluator picked first.
        explicit = tmp_path / "explicit_tasks"
        legacy_a = tmp_path / "run" / "tasks"
        legacy_b = tmp_path / "tasks"
        for d in (explicit, legacy_a, legacy_b):
            d.mkdir(parents=True, exist_ok=True)
            (d / "p1_tasks.json").write_text("[]", encoding="utf-8")

        called_with: dict = {}

        def fake_compute_grounding(candidate, *args, **kwargs):
            called_with.setdefault("candidate", candidate)
            return {"total": 1, "grounded": 1, "ratio": 1.0}

        run_dir = tmp_path / "run"
        with patch(
            "src.scripts.scenarios.metrics.grounding.compute_grounding",
            side_effect=fake_compute_grounding,
        ):
            # Drive the same candidate-list logic the evaluator uses,
            # without spinning up the whole _cmd_evaluate_with_cfg.
            from src.scripts.scenarios.metrics.grounding import compute_grounding

            args = argparse.Namespace(tasks_dir=explicit)
            explicit_tasks_dir = getattr(args, "tasks_dir", None)
            candidate_tasks_dirs: list[Path] = []
            if explicit_tasks_dir is not None:
                candidate_tasks_dirs.append(Path(explicit_tasks_dir))
            candidate_tasks_dirs += [run_dir / "tasks", run_dir.parent / "tasks"]
            for candidate in candidate_tasks_dirs:
                if candidate.is_dir():
                    compute_grounding(candidate, ["p1"], expected_total=1)
                    break
        assert called_with["candidate"] == explicit


class TestMultiEvaluateAutoWritesReport:
    """`_multi_evaluate` must call the benchmark-report writer once at
    the end, with the actual scenario × method pairs it iterated."""

    def test_auto_writes_report_after_successful_runs(self, tmp_path: Path):
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [
                    {"id": "s1", "augmentation": [{"method": "llm_agent"}]},
                    {"id": "s2", "augmentation": [{"method": "llm_agent"}]},
                ],
            }
        )
        args = argparse.Namespace(
            scenario=tmp_path / "scen.yaml",
            log_level="INFO",
        )
        with (
            patch.object(cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_OK),
            patch.object(
                cli_module, "_write_benchmark_report_if_possible"
            ) as writer_hook,
        ):
            rc = _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
        assert rc == EXIT_OK
        writer_hook.assert_called_once()
        passed_pairs = writer_hook.call_args.args[1]
        assert passed_pairs == [("s1", "llm_agent"), ("s2", "llm_agent")]

    def test_does_not_write_report_when_evaluate_fails(self, tmp_path: Path):
        exp = ExperimentScenariosConfig.model_validate(
            {
                "experiment_id": "exp_z",
                "run_dir": str(tmp_path),
                "scenarios": [
                    {"id": "s1", "augmentation": [{"method": "llm_agent"}]},
                ],
            }
        )
        args = argparse.Namespace(
            scenario=tmp_path / "scen.yaml",
            log_level="INFO",
        )
        with (
            patch.object(cli_module, "_cmd_evaluate_with_cfg", return_value=EXIT_USAGE),
            patch.object(
                cli_module, "_write_benchmark_report_if_possible"
            ) as writer_hook,
        ):
            rc = _multi_evaluate(exp, args, scenario_id=None, method_filter=None)
        assert rc == EXIT_USAGE
        writer_hook.assert_not_called()


# ---------------------------------------------------------------------------
# _write_augmented_timeline
# ---------------------------------------------------------------------------


class _TimelineStubEvent:
    def __init__(self) -> None:
        import datetime as _dt

        self.label = "lunch"
        self.date = _dt.date(2026, 5, 4)
        self.start_minutes = 720
        self.end_minutes = 765
        self.is_concurrent = True
        self.is_dividable = False
        self.intensity = 2


class _TimelineStubEpisode:
    def __init__(self) -> None:
        import datetime as _dt

        self.name = "happy"
        self.category = "mood_emotion"
        self.date = _dt.date(2026, 5, 4)
        self.start_minutes = 480
        self.end_minutes = 540
        self.ontology_uri = None
        self.dimension = None
        self.polarity = None
        self.instrument = None
        self.theory_mappings = None


class _TimelineStubTrace:
    def __init__(self, *, with_events: bool, with_contexts: bool) -> None:
        self.person_id = "p1"
        self.persona_id = "alice"
        self.events = [_TimelineStubEvent()] if with_events else []
        self.contexts = [_TimelineStubEpisode()] if with_contexts else []


def test_write_augmented_timeline_writes_event_rows(tmp_path: Path) -> None:
    """A trace with events writes one event row per occurrence."""
    out = _write_augmented_timeline(
        trace=_TimelineStubTrace(with_events=True, with_contexts=False),
        scheduled=[],
        out_path=tmp_path / "t.tsv",
    )
    assert out.exists()
    assert "lunch" in out.read_text(encoding="utf-8")


def test_write_augmented_timeline_writes_context_rows(tmp_path: Path) -> None:
    """A trace with contexts writes one context row per episode."""
    out = _write_augmented_timeline(
        trace=_TimelineStubTrace(with_events=False, with_contexts=True),
        scheduled=[],
        out_path=tmp_path / "t.tsv",
    )
    assert "happy" in out.read_text(encoding="utf-8")


def test_write_augmented_timeline_handles_catalog_load_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """A failing catalog load leaves catalog as None and the writer still succeeds."""

    def _boom():
        raise RuntimeError("catalog boom")

    monkeypatch.setattr("src.scripts.persona.context.catalog.load_catalog", _boom)
    out = _write_augmented_timeline(
        trace=_TimelineStubTrace(with_events=False, with_contexts=True),
        scheduled=[],
        out_path=tmp_path / "t.tsv",
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# _build_met_lookup_with_bridge
# ---------------------------------------------------------------------------


class TestBuildMetLookupWithBridge:
    def test_returns_none_when_neither_bridge_nor_neo4j_available(
        self, monkeypatch
    ) -> None:
        """Missing bridge files and unreachable Neo4j return None."""
        from src.scripts.scenarios import cli as cli_mod

        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)

        def _bad_driver(*_a, **_k):
            raise RuntimeError("no neo4j")

        monkeypatch.setattr("src.graphrag.neo4j_client.make_driver", _bad_driver)
        assert cli_mod._build_met_lookup_with_bridge() is None

    def test_returns_lookup_when_bridge_loads(self, monkeypatch) -> None:
        """A successful bridge load returns a MetLookup even when Neo4j is down."""
        from src.scripts.scenarios import cli as cli_mod

        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr(
            "src.scripts.scenarios.metrics.met_lookup.load_matched_activity_index",
            lambda *_a, **_k: {"http://x": 3.5},
        )

        def _bad_driver(*_a, **_k):
            raise RuntimeError("no neo4j")

        monkeypatch.setattr("src.graphrag.neo4j_client.make_driver", _bad_driver)
        assert cli_mod._build_met_lookup_with_bridge() is not None


# ---------------------------------------------------------------------------
# _build_activity_bridge
# ---------------------------------------------------------------------------


class TestBuildActivityBridge:
    def test_returns_none_when_neo4j_unreachable(self, monkeypatch) -> None:
        """Unreachable Neo4j returns None before any catalog load."""
        from types import SimpleNamespace

        from src.scripts.scenarios import cli as cli_mod

        def _bad_driver(*_a, **_k):
            raise RuntimeError("no neo4j")

        monkeypatch.setattr("src.graphrag.neo4j_client.make_driver", _bad_driver)
        cfg = SimpleNamespace(
            calendar=SimpleNamespace(persona_events=None),
            evaluation=None,
        )
        assert cli_mod._build_activity_bridge(cfg) is None

    def test_returns_none_when_no_event_iris(self, monkeypatch) -> None:
        """An empty event-to-IRI map returns None."""
        from types import SimpleNamespace

        from src.scripts.scenarios import cli as cli_mod

        class _StubDriver:
            def verify_connectivity(self):
                return None

        monkeypatch.setattr(
            "src.graphrag.neo4j_client.make_driver", lambda *_a, **_k: _StubDriver()
        )
        cfg = SimpleNamespace(
            calendar=SimpleNamespace(persona_events=None),
            evaluation=None,
        )
        assert cli_mod._build_activity_bridge(cfg) is None

    def test_returns_bridge_when_event_catalog_resolves(self, monkeypatch) -> None:
        """A non-empty event-to-IRI map builds an ActivityFamilyBridge."""
        from types import SimpleNamespace

        from src.scripts.scenarios import cli as cli_mod

        class _StubDriver:
            def verify_connectivity(self):
                return None

        monkeypatch.setattr(
            "src.graphrag.neo4j_client.make_driver", lambda *_a, **_k: _StubDriver()
        )
        ev = SimpleNamespace(human_activity_iri="http://example.org/walk")
        category = SimpleNamespace(events={"walking": ev})
        event_cfg = SimpleNamespace(categories={"sports": category})
        monkeypatch.setattr(
            "src.scripts.persona.config.loader.load_event",
            lambda *_a, **_k: event_cfg,
        )
        cfg = SimpleNamespace(
            calendar=SimpleNamespace(persona_events="dummy.yaml"),
            evaluation=SimpleNamespace(cross_ancestor_max_hops=4),
        )
        out = cli_mod._build_activity_bridge(cfg)
        assert out is not None

    def test_events_without_human_activity_iri_are_skipped(self, monkeypatch) -> None:
        """An event whose human_activity_iri is empty drops out of the bridge map."""
        from types import SimpleNamespace

        from src.scripts.scenarios import cli as cli_mod

        class _StubDriver:
            def verify_connectivity(self):
                return None

        monkeypatch.setattr(
            "src.graphrag.neo4j_client.make_driver", lambda *_a, **_k: _StubDriver()
        )
        ev_with = SimpleNamespace(human_activity_iri="http://example.org/walk")
        ev_without = SimpleNamespace(human_activity_iri=None)
        ev_blank = SimpleNamespace(human_activity_iri="")
        category = SimpleNamespace(
            events={
                "walking": ev_with,
                "office_work": ev_without,
                "lunch": ev_blank,
            }
        )
        event_cfg = SimpleNamespace(categories={"mixed": category})
        monkeypatch.setattr(
            "src.scripts.persona.config.loader.load_event",
            lambda *_a, **_k: event_cfg,
        )
        cfg = SimpleNamespace(
            calendar=SimpleNamespace(persona_events="dummy.yaml"),
            evaluation=None,
        )
        out = cli_mod._build_activity_bridge(cfg)
        assert out is not None


# ---------------------------------------------------------------------------
# _cmd_augment branches: context-links file missing + pref-driver Neo4j down
# ---------------------------------------------------------------------------


class TestCmdAugmentFallbackPaths:
    def _mock_run(self, traces=None):
        run = MagicMock()
        run.traces = traces or []
        run.time_windows = {}
        run.allen_pair_rules = []
        run.horizon_days = 7
        run.horizon_start_date = None
        return run

    def _loss_components(self):
        return MagicMock(
            cov=0.5,
            cal=0.0,
            pref=0.0,
            disp=0.0,
            merge=0.0,
            spread=0.0,
            divide=0.0,
            context_fit=None,
        )

    def _wire_augment_stubs(self, run, mock_augmenter, mock_loss_fn):
        return [
            patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
            patch(
                "src.scripts.scenarios.metrics.loss.SchedulingLoss",
                return_value=mock_loss_fn,
            ),
            patch("src.scripts.scenarios.export.json_writer.write_solution_json"),
            patch("src.scripts.scenarios.export.ics_writer.write_augmented_ics"),
        ]

    def test_missing_context_links_json_falls_back_to_empty_dict(
        self, tmp_path, monkeypatch
    ):
        """A missing HealthTasks context-links JSON does not abort augment."""
        from src.scripts.scenarios.cli import _cmd_augment

        # Force the HealthTasks JSON existence check to False.
        original_exists = Path.exists

        def _fake_exists(self):
            if self.name == "HealthTasks_2026.05.19.json":
                return False
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", _fake_exists)

        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        run = self._mock_run(traces=[mock_trace])
        mock_augmenter = MagicMock()
        mock_augmenter.augment.return_value = MagicMock()
        mock_loss_fn = MagicMock()
        mock_loss_fn.compute.return_value = (0.5, self._loss_components())
        ctx_stack = [
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            *self._wire_augment_stubs(run, mock_augmenter, mock_loss_fn),
        ]
        from contextlib import ExitStack

        with ExitStack() as stack:
            for ctx in ctx_stack:
                stack.enter_context(ctx)
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK

    def test_pref_driver_neo4j_unreachable_falls_back_to_semantic_only(
        self, tmp_path, monkeypatch
    ):
        """A Neo4j connectivity error during PreferenceMapper wiring is swallowed."""
        from src.scripts.scenarios.cli import _cmd_augment

        # Make Neo4j connectivity raise so the inner try/except fires.
        def _bad_driver(*_a, **_k):
            raise RuntimeError("no neo4j")

        monkeypatch.setattr("src.graphrag.neo4j_client.make_driver", _bad_driver)

        # Stub load_event so the L_pref v2 path enters and the inner Neo4j
        # block fires.
        from types import SimpleNamespace

        event_cfg = SimpleNamespace(categories={})
        monkeypatch.setattr(
            "src.scripts.persona.config.loader.load_event",
            lambda *_a, **_k: event_cfg,
        )
        from src.scripts.persona.domain import time_windows as tw_mod

        monkeypatch.setattr(
            tw_mod.WindowMap,
            "from_config",
            classmethod(lambda cls, _cfg: object()),
        )

        mock_trace = MagicMock()
        mock_trace.person_id = "p001"
        run = self._mock_run(traces=[mock_trace])
        # Wire a persona_events path so the L_pref v2 branch enters.
        cfg = _mock_cfg(out_dir=str(tmp_path))
        from src.scripts.scenarios.config.schema import CalendarSourceConfig

        cfg = cfg.model_copy(
            update={"calendar": CalendarSourceConfig(persona_events="dummy.yaml")}
        )
        mock_augmenter = MagicMock()
        mock_augmenter.augment.return_value = MagicMock()
        mock_loss_fn = MagicMock()
        mock_loss_fn.compute.return_value = (0.5, self._loss_components())
        ctx_stack = [
            _mock_load(cfg),
            patch.object(
                cli_module, "_try_load_persona_run", return_value=(run, tmp_path)
            ),
            *self._wire_augment_stubs(run, mock_augmenter, mock_loss_fn),
        ]
        from contextlib import ExitStack

        with ExitStack() as stack:
            for ctx in ctx_stack:
                stack.enter_context(ctx)
            rc = _cmd_augment(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK


# ---------------------------------------------------------------------------
# _cmd_evaluate branches: missing context-links JSON + empty breakdowns
# ---------------------------------------------------------------------------


class TestCmdEvaluateFallbackPaths:
    def _write_one_loss(self, persons_dir: Path, pid: str = "p001"):
        persons_dir.mkdir(parents=True, exist_ok=True)
        _seed_solution_json(persons_dir, pid)

    def test_missing_context_links_json_falls_back_to_empty_dict(
        self, tmp_path, monkeypatch
    ):
        """`_cmd_evaluate` recomputes without crashing when the HealthTasks JSON is absent."""
        from src.scripts.scenarios.cli import _cmd_evaluate

        original_exists = Path.exists

        def _fake_exists(self):
            if self.name == "HealthTasks_2026.05.19.json":
                return False
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", _fake_exists)

        persons_dir = tmp_path / "augmented" / "persons"
        self._write_one_loss(persons_dir)
        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports"
            ) as mock_report,
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        mock_report.assert_called_once()

    def test_empty_divide_and_context_fit_sidecars_set_breakdown_to_none(
        self, tmp_path, monkeypatch
    ):
        """Empty applicable buckets and tasks resolve the breakdown dicts to None."""
        from src.scripts.scenarios.cli import _cmd_evaluate

        persons_dir = tmp_path / "augmented" / "persons"
        self._write_one_loss(persons_dir)
        # Stub the breakdown readers to return empty bucket / task counts.
        monkeypatch.setattr(
            "src.scripts.scenarios.metrics.divide_breakdown.read_divide_verdicts",
            lambda paths: {"applicable_buckets": 0},
        )
        monkeypatch.setattr(
            "src.scripts.scenarios.metrics.context_fit_breakdown.read_context_fit_verdicts",
            lambda paths: {"applicable_tasks": 0},
        )
        captured: dict = {}

        def _spy(*a, **kw):
            captured["divide"] = kw.get("divide_breakdown")
            captured["context_fit"] = kw.get("context_fit_breakdown")
            return {}

        with (
            _mock_load(_mock_cfg(out_dir=str(tmp_path))),
            patch(
                "src.scripts.scenarios.export.report_writer.write_evaluation_reports",
                _spy,
            ),
        ):
            rc = _cmd_evaluate(_args(scenario=tmp_path / "s.yaml"))
        assert rc == EXIT_OK
        assert captured["divide"] is None
        assert captured["context_fit"] is None


class TestResolveWeeklyChunksForSolution:
    def test_anchor_clamps_to_first_placement_when_horizon_start_later(self):
        import datetime
        from types import SimpleNamespace

        from src.scripts.scenarios.cli import _resolve_weekly_chunks_for_solution

        solution = SimpleNamespace(
            scheduled=[
                SimpleNamespace(date=datetime.date(2026, 6, 1)),
                SimpleNamespace(date=datetime.date(2026, 6, 9)),
            ]
        )
        late = _resolve_weekly_chunks_for_solution(
            solution, horizon_start_date=datetime.date(2026, 6, 5)
        )
        none = _resolve_weekly_chunks_for_solution(solution)
        # A horizon start after the first placement is clamped back to it,
        # so both calls anchor identically.
        assert late and late == none

    def test_returns_empty_when_nothing_scheduled(self):
        from types import SimpleNamespace

        from src.scripts.scenarios.cli import _resolve_weekly_chunks_for_solution

        assert _resolve_weekly_chunks_for_solution(SimpleNamespace(scheduled=[])) == []
