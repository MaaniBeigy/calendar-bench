"""End-to-end CLI integration test for the L_pref v2 wiring.

Drives `_cmd_augment_with_cfg` with mocked augmenter + persona-run
infrastructure but real:

* `event_config.yaml` (with ontology IRIs declared);
* per-person JSON snapshot in the run dir;
* :class:`PreferenceCache` (fakeredis fallback);
* :class:`PreferenceMapper` (literal-tier match);
* sidecar writer.

Verifies the augment flow writes `<person_id>_pref_violations.jsonl`
sidecars next to `_loss.json`, with a non-zero `per_occurrence_duration`
leg row (the placement deliberately violates the duration band).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scripts.scenarios import cli as cli_module
from src.scripts.scenarios.cli import _cmd_augment_with_cfg, _cmd_evaluate_with_cfg
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    CalendarSourceConfig,
    ScenarioConfig,
    ScenarioOutputConfig,
)
from src.scripts.scenarios.domain.calendar import AugmentedCalendar, CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask

HEALTH_PREFIX = "https://w3id.org/calendar-bench/health/task/"


@pytest.fixture()
def scenario_layout(tmp_path: Path) -> dict:
    """Build a minimal run dir + tasks dir + event_config.yaml on disk."""
    run_dir = tmp_path / "run"
    persons_dir_in_run = run_dir / "persons"
    persons_dir_in_run.mkdir(parents=True)

    # Per-person snapshot.
    person_id = "p001"
    person_json = run_dir / "persons" / f"{person_id}.json"
    person_json.write_text(
        json.dumps(
            {
                "person_id": person_id,
                "persona": {
                    "person_id": person_id,
                    "persona_id": "parttime_morning",
                    "occupation_status": "parttime",
                    "stages": [
                        {
                            "name": "walking",
                            "time": "morning",
                            "duration_minutes": 30,
                            "days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                        }
                    ],
                    "event_overrides": {},
                },
                "days": [],
            }
        ),
        encoding="utf-8",
    )

    # index.json: tells PersonaConstraints where to find the persona snapshot.
    (run_dir / "index.json").write_text(
        json.dumps(
            {
                "persons": [
                    {
                        "person_id": person_id,
                        "json_path": str(person_json.relative_to(run_dir)),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # event_config.yaml; walking with a literal HealthTasks IRI.
    event_config_path = tmp_path / "event_config.yaml"
    event_config_path.write_text(
        """
categories:
  sports:
    events:
      walking:
        health_task_iri: https://w3id.org/calendar-bench/health/task/schedule-a-30-minute-walk
        per_event_duration: { min: 30, max: 75, unit: minutes }
        total_event_duration: { scale: day, min: 30, max: 75, unit: minutes }
        total_event_episodes: { scale: day, min: 0, max: 1 }
""",
        encoding="utf-8",
    )

    # Tasks dir; used by the augment loop to look up task JSON per persona.
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / f"{person_id}_tasks.json").write_text(
        json.dumps(
            [
                {
                    "label": "schedule-a-30-minute-walk",
                    "duration_min": 30,
                    "duration_max": 30,
                    "ontology_uri": HEALTH_PREFIX + "schedule-a-30-minute-walk",
                }
            ]
        ),
        encoding="utf-8",
    )

    return {
        "tmp_path": tmp_path,
        "run_dir": run_dir,
        "tasks_dir": tasks_dir,
        "event_config_path": event_config_path,
        "person_id": person_id,
        "out_dir": tmp_path / "out",
    }


def _build_mock_run(layout: dict):
    """Construct a LoadedRun-shaped MagicMock the augment flow expects."""
    from src.scripts.persona.config.schema import WindowRange

    trace = CalendarTrace(person_id=layout["person_id"], events=[])
    run = MagicMock()
    run.traces = [trace]
    run.time_windows = {
        "early_morning": WindowRange(start=0, end=400),
        "morning": WindowRange(start=400, end=600),
        "afternoon": WindowRange(start=600, end=960),
        "evening": WindowRange(start=960, end=1260),
        "night": WindowRange(start=1260, end=1440),
    }
    run.allen_pair_rules = []
    run.horizon_days = 28
    run.horizon_start_date = _dt.date(2026, 6, 1)
    return run, trace


def _make_solution(trace, *, bad_duration: bool = True) -> SchedulingSolution:
    task = RecommendedTask(
        label="schedule-a-30-minute-walk",
        duration_min=30,
        duration_max=30,
        ontology_uri=HEALTH_PREFIX + "schedule-a-30-minute-walk",
    )
    # Bad augmenter: 10-min walk to below per_event_min=30 to triggers
    # per_occurrence_duration violation.
    end = 490 if bad_duration else 510
    sched = ScheduledTask(
        task=task,
        start_minutes=480,
        end_minutes=end,
        is_standalone=True,
        concurrent_with=None,
        date=_dt.date(2026, 6, 1),
    )
    return SchedulingSolution(
        person_id=trace.person_id,
        augmented_calendar=AugmentedCalendar(person_id=trace.person_id),
        tasks=[task],
        scheduled=[sched],
        unscheduled=[],
    )


def _build_cfg(layout: dict) -> ScenarioConfig:
    return ScenarioConfig(
        id="health_l1",
        description="L_pref v2 CLI smoke",
        calendar=CalendarSourceConfig(
            run_dir=str(layout["run_dir"]),
            persona_events=str(layout["event_config_path"]),
        ),
        output=ScenarioOutputConfig(dir=str(layout["out_dir"])),
        augmentation=AugmentationConfig(method="greedy"),
    )


def _argv(layout: dict):
    import argparse

    return argparse.Namespace(
        scenario=Path("nonexistent.yaml"),
        out_dir=layout["out_dir"],
        tasks_dir=layout["tasks_dir"],
        seed=None,
        method=None,
        run_dir=layout["run_dir"],
    )


def test_evaluate_writes_preference_violations_sidecar(scenario_layout):
    """Running augment then evaluate must:

    * augment writes the per-person solution JSON (placements only);
    * evaluate writes `<person>_pref_violations.jsonl` from the
      authoritative recompute;
    * the sidecar's `per_occurrence_duration` leg row reflects the
      duration violation in the reconstructed solution.
    """
    layout = scenario_layout
    cfg = _build_cfg(layout)
    args = _argv(layout)

    run, trace = _build_mock_run(layout)
    solution = _make_solution(trace, bad_duration=True)
    mock_augmenter = MagicMock()
    mock_augmenter.augment.return_value = solution

    with (
        patch.object(
            cli_module, "_try_load_persona_run", return_value=(run, layout["run_dir"])
        ),
        patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
        patch.object(cli_module, "write_augmented_ics", create=True),
    ):
        rc = _cmd_augment_with_cfg(cfg, args)
    assert rc == 0

    persons_dir = layout["out_dir"] / "augmented" / "persons"
    # Augment writes the solution JSON and never an augment-time _loss.json.
    assert (persons_dir / f"{layout['person_id']}.json").exists()
    assert not (persons_dir / f"{layout['person_id']}_loss.json").exists()

    eval_args = argparse.Namespace(
        scenario=None,
        run_dir=layout["out_dir"],
        tasks_dir=layout["tasks_dir"],
        method=None,
        log_level="INFO",
    )
    with (
        patch.object(
            cli_module, "_try_load_persona_run", return_value=(run, layout["run_dir"])
        ),
        patch.object(cli_module, "_build_judge_oracle", return_value=None),
    ):
        rc = _cmd_evaluate_with_cfg(cfg, eval_args)
    assert rc == 0

    sidecar = persons_dir / f"{layout['person_id']}_pref_violations.jsonl"
    assert (
        sidecar.exists()
    ), f"sidecar {sidecar} not written; evaluate did not wire L_pref v2"
    rows = [
        json.loads(line) for line in sidecar.read_text("utf-8").splitlines() if line
    ]
    legs = {r["name"]: r for r in rows if r.get("kind") == "leg"}
    assert "per_occurrence_duration" in legs
    assert legs["per_occurrence_duration"]["applicable_count"] == 1
    assert legs["per_occurrence_duration"]["mean_loss"] > 0.0


def test_augment_writes_solution_without_loss_json(scenario_layout, tmp_path):
    """Augment writes the solution JSON and no augment-time `_loss.json`.

    All scoring now happens at evaluate, so the augment step emits
    placements only regardless of whether `persona_events` is wired.
    """
    layout = scenario_layout
    cfg = ScenarioConfig(
        id="health_l1",
        description="No event_config wired",
        calendar=CalendarSourceConfig(
            run_dir=str(layout["run_dir"]),
            persona_events=None,
        ),
        output=ScenarioOutputConfig(dir=str(layout["out_dir"])),
        augmentation=AugmentationConfig(method="greedy"),
    )
    args = _argv(layout)
    run, trace = _build_mock_run(layout)
    solution = _make_solution(trace, bad_duration=False)
    mock_augmenter = MagicMock()
    mock_augmenter.augment.return_value = solution
    with (
        patch.object(
            cli_module, "_try_load_persona_run", return_value=(run, layout["run_dir"])
        ),
        patch.object(cli_module, "_build_augmenter", return_value=mock_augmenter),
        patch.object(cli_module, "write_augmented_ics", create=True),
    ):
        rc = _cmd_augment_with_cfg(cfg, args)
    assert rc == 0
    persons_dir = layout["out_dir"] / "augmented" / "persons"
    assert (persons_dir / f"{layout['person_id']}.json").exists()
    assert not (persons_dir / f"{layout['person_id']}_loss.json").exists()
    assert not (persons_dir / f"{layout['person_id']}_pref_violations.jsonl").exists()
