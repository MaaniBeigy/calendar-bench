"""Integration: the per-week task file round-trips through disk into placement.

Exercises the seam that ties generation to augmentation without Neo4j or an
LLM: per-week batches are written as the `{"weeks": [...]}` wrapper, read back
into per-week lists, and a real greedy augmenter places each week's batch into
its own calendar week.
"""

from __future__ import annotations

import datetime

import pytest

from src.scripts.scenarios.augmentation.greedy import GreedyAugmenter
from src.scripts.scenarios.cli import _read_weekly_tasks, _write_weekly_tasks
from src.scripts.scenarios.config.schema import AugmentationConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

pytestmark = pytest.mark.integration

_DATE = datetime.date(2026, 6, 1)  # Monday


def test_weekly_batches_round_trip_into_distinct_weeks(tmp_path):
    week1 = [make_task("nutrition_a", duration_min=30, duration_max=30)]
    week2 = [
        make_task("nutrition_b", duration_min=30, duration_max=30),
        make_task("physical_b", duration_min=30, duration_max=30),
    ]
    path = tmp_path / "p001_tasks.json"
    _write_weekly_tasks([week1, week2], path)

    weekly_tasks = _read_weekly_tasks(path)
    assert [[t.label for t in wk] for wk in weekly_tasks] == [
        ["nutrition_a"],
        ["nutrition_b", "physical_b"],
    ]

    trace = CalendarTrace(
        person_id="p001",
        events=[
            make_event(date=_DATE),
            make_event(date=_DATE + datetime.timedelta(days=7)),
        ],
    )
    flat = [t for wk in weekly_tasks for t in wk]
    solution = GreedyAugmenter().augment(
        trace, flat, AugmentationConfig(), weekly_tasks=weekly_tasks
    )

    placed = {st.task.label: st.date for st in solution.scheduled}
    # Each week's batch lands in its own calendar week.
    assert placed["nutrition_a"] < _DATE + datetime.timedelta(days=7)
    assert placed["nutrition_b"] >= _DATE + datetime.timedelta(days=7)
    assert placed["physical_b"] >= _DATE + datetime.timedelta(days=7)
    # The instance total is the sum of the per-week counts, not a repeat.
    assert len(solution.tasks) == 3
