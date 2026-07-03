"""Integration: per-episode task labels flow persona JSON to augmented ICS."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from src.scripts.persona.config.loader import load_event
from src.scripts.persona.domain.event import Catalog, EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.export.json_writer import write_person_json
from src.scripts.persona.export.task_labels import assign_task_labels
from src.scripts.scenarios.augmentation.greedy import GreedyAugmenter
from src.scripts.scenarios.calendar.loader import load_persona_run
from src.scripts.scenarios.config.schema import AugmentationConfig
from src.scripts.scenarios.export.ics_writer import solution_to_ical_bytes
from tests.unit.persona.conftest import make_person, make_stage
from tests.unit.scenarios.conftest import make_task

pytestmark = pytest.mark.integration

_EXP = Path("src/experiments/persona/example_experiment")
_TASK_LABELS = {"standup", "doing research", "external meeting"}
_TITLES = {"Standup", "Doing Research", "External Meeting"}


def _schedule() -> PersonSchedule:
    """Three weekdays with one office_work episode plus a label-less lunch."""
    days = []
    for offset in range(3):
        date = datetime.date(2026, 6, 1) + datetime.timedelta(days=offset)
        days.append(
            DaySchedule(
                day_index=offset,
                date=date,
                weekday=date.strftime("%a"),
                events={
                    "office_work": [
                        EventInstance(event_name="office_work", start=540, duration=120)
                    ],
                    "lunch": [
                        EventInstance(event_name="lunch", start=720, duration=45)
                    ],
                },
                spillovers=[],
            )
        )
    return PersonSchedule(
        person_id="i_fulltime_0000", persona_id="fulltime", person_seed=7, days=days
    )


def _person():
    return make_person(
        person_id="i_fulltime_0000",
        persona_id="fulltime",
        person_seed=7,
        occupation_status="fulltime",
        stages=[make_stage("sleep", time="23:00")],
    )


def test_task_label_round_trips_into_augmented_ics(tmp_path):
    catalog = Catalog.from_event_config(load_event(_EXP / "event_config.yaml"))
    labelled = assign_task_labels(_schedule(), catalog)

    run_dir = tmp_path / "run"
    persons_dir = run_dir / "persons"
    write_person_json(_person(), labelled, persons_dir)

    # The persisted JSON carries one rotating label per office_work episode.
    raw = json.loads((persons_dir / "i_fulltime_0000.json").read_text())
    office_labels = [
        inst["label"] for day in raw["days"] for inst in day["events"]["office_work"]
    ]
    assert set(office_labels) == _TASK_LABELS
    assert "label" not in raw["days"][0]["events"]["lunch"][0]

    (run_dir / "index.json").write_text(
        json.dumps(
            {
                "persons": [
                    {
                        "person_id": "i_fulltime_0000",
                        "json_path": "persons/i_fulltime_0000.json",
                    }
                ]
            }
        )
    )
    run = load_persona_run(
        run_dir,
        event_config_path=_EXP / "event_config.yaml",
        rules_path=_EXP / "temporal_relation_rules.yaml",
        environment_path=_EXP / "environment.yaml",
    )
    trace = run.traces[0]
    work_events = [e for e in trace.events if e.label == "office_work"]
    assert work_events
    for ev in work_events:
        assert ev.label == "office_work"  # scoring identity is preserved
        assert ev.display_label in _TASK_LABELS
        assert ev.intensity == 3  # office_work metadata still resolves
        # The evaluation judge scores the actual label plus its parent.
        assert ev.oracle_label == f"{ev.display_label} (office_work)"

    solution = GreedyAugmenter().augment(
        trace, [make_task("hydrate")], AugmentationConfig()
    )
    ics = solution_to_ical_bytes(solution, calendar=trace).decode("utf-8")
    assert any(f"SUMMARY:{title}" in ics for title in _TITLES)
    assert "Office Work" not in ics
