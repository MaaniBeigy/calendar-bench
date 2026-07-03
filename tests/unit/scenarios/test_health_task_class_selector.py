"""SelectorPredicate.health_task_class / health_task_uri match against the SUBCLASSOF closure."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import SelectorPredicate
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import SelectorMatcher
from src.scripts.scenarios.metrics.health_task_hierarchy import (
    HB_TASK_PREFIX,
    load_class_closure,
)

DATE = datetime.date(2026, 5, 4)


def _build_fixture_closure(tmp_path: Path) -> dict[str, frozenset[str]]:
    """Tiny JSON fixture so the test does not depend on the shipped 05.19 ontology."""
    payload = {
        "Nutrition": {
            "BetterBeverageBalance": {
                "Level1": {
                    "tea-time": {"title": "Tea Time", "description": ""},
                },
            },
        },
        "PhysicalActivity": {
            "CardioAndStamina": {
                "Level2": {
                    "run-10-minutes": {"title": "Run 10 Minutes", "description": ""},
                },
            },
        },
    }
    path = tmp_path / "HealthTasks_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_class_closure(path)


def _scheduled_task(uri: str) -> ScheduledTask:
    return ScheduledTask(
        task=RecommendedTask(
            label="x", duration_min=10, duration_max=20, ontology_uri=uri
        ),
        start_minutes=540,
        end_minutes=560,
        is_standalone=True,
        concurrent_with=None,
        date=DATE,
    )


def test_load_class_closure_emits_one_entry_per_leaf(tmp_path: Path) -> None:
    closure = _build_fixture_closure(tmp_path)
    tea_uri = HB_TASK_PREFIX + "tea-time"
    run_uri = HB_TASK_PREFIX + "run-10-minutes"
    assert tea_uri in closure
    assert run_uri in closure
    assert "tea-time" in closure[tea_uri]
    assert "NutritionBetterBeverageBalanceLevel1" in closure[tea_uri]
    assert "NutritionBetterBeverageBalanceTask" in closure[tea_uri]
    assert "NutritionTask" in closure[tea_uri]
    assert "HealthTask" in closure[tea_uri]


def test_health_task_class_matches_branch_ancestor(tmp_path: Path) -> None:
    closure = _build_fixture_closure(tmp_path)
    matcher = SelectorMatcher(class_closure=closure)
    sel = SelectorPredicate(
        kind="task", health_task_class="NutritionBetterBeverageBalanceTask"
    )
    tea = _scheduled_task(HB_TASK_PREFIX + "tea-time")
    run = _scheduled_task(HB_TASK_PREFIX + "run-10-minutes")
    assert matcher.matches(sel, tea) is True
    assert matcher.matches(sel, run) is False


def test_health_task_class_leaf_slug_pins_to_one_instance(tmp_path: Path) -> None:
    closure = _build_fixture_closure(tmp_path)
    matcher = SelectorMatcher(class_closure=closure)
    sel = SelectorPredicate(kind="task", health_task_class="tea-time")
    assert matcher.matches(sel, _scheduled_task(HB_TASK_PREFIX + "tea-time")) is True
    assert (
        matcher.matches(sel, _scheduled_task(HB_TASK_PREFIX + "run-10-minutes"))
        is False
    )


def test_health_task_class_list_requires_all_to_match(tmp_path: Path) -> None:
    closure = _build_fixture_closure(tmp_path)
    matcher = SelectorMatcher(class_closure=closure)
    sel = SelectorPredicate(
        kind="task",
        health_task_class=["NutritionTask", "NutritionBetterBeverageBalanceTask"],
    )
    assert matcher.matches(sel, _scheduled_task(HB_TASK_PREFIX + "tea-time")) is True
    assert (
        matcher.matches(sel, _scheduled_task(HB_TASK_PREFIX + "run-10-minutes"))
        is False
    )


def test_health_task_uri_exact_match(tmp_path: Path) -> None:
    closure = _build_fixture_closure(tmp_path)
    matcher = SelectorMatcher(class_closure=closure)
    tea_uri = HB_TASK_PREFIX + "tea-time"
    sel = SelectorPredicate(kind="task", health_task_uri=tea_uri)
    assert matcher.matches(sel, _scheduled_task(tea_uri)) is True
    assert (
        matcher.matches(sel, _scheduled_task(HB_TASK_PREFIX + "run-10-minutes"))
        is False
    )


def test_context_kind_rejects_health_task_class_field() -> None:
    with pytest.raises(ValidationError, match="event/task-only"):
        SelectorPredicate(kind="context", name="tired", health_task_class="X")


def test_class_closure_missing_means_no_match() -> None:
    """When no closure is wired, `health_task_class` cannot match anything."""
    matcher = SelectorMatcher()  # no class_closure
    sel = SelectorPredicate(kind="task", health_task_class="NutritionTask")
    assert matcher.matches(sel, _scheduled_task(HB_TASK_PREFIX + "tea-time")) is False
