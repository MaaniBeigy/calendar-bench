"""Tests for `metrics/ontology_source.py::infer_source`."""

from __future__ import annotations

import pytest

from src.graphrag.retriever import HEALTH_TASK_INSTANCE_PREFIX
from src.scripts.scenarios.metrics.met_lookup import HUMAN_ACTIVITIES_PREFIX
from src.scripts.scenarios.metrics.ontology_source import infer_source


class TestInferSource:
    @pytest.mark.parametrize(
        "uri",
        [
            HEALTH_TASK_INSTANCE_PREFIX + "schedule-a-30-minute-walk",
            HEALTH_TASK_INSTANCE_PREFIX + "cook-meal",
            HEALTH_TASK_INSTANCE_PREFIX,  # exact prefix matches
        ],
    )
    def test_health_tasks_prefix_resolves_to_HealthTasks(self, uri: str) -> None:
        assert infer_source(uri) == "HealthTasks"

    @pytest.mark.parametrize(
        "uri",
        [
            HUMAN_ACTIVITIES_PREFIX + "activity/race-walking-3-1-m-s-6-9-mph",
            HUMAN_ACTIVITIES_PREFIX + "activity/yoga-hatha",
            HUMAN_ACTIVITIES_PREFIX,
        ],
    )
    def test_human_activities_prefix_resolves_to_HumanActivities(
        self, uri: str
    ) -> None:
        assert infer_source(uri) == "HumanActivities"

    def test_none_input_returns_none(self) -> None:
        assert infer_source(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert infer_source("") is None

    @pytest.mark.parametrize(
        "uri",
        [
            "https://example.org/some/unknown",
            "https://w3id.org/calendar-bench/other-namespace/foo",
            "ftp://internal/uri",
            "not-even-a-url",
        ],
    )
    def test_unknown_prefix_returns_none(self, uri: str) -> None:
        assert infer_source(uri) is None

    def test_case_sensitive_prefix(self) -> None:
        """Prefix match is case-sensitive; capitalised variants miss."""
        upper = HEALTH_TASK_INSTANCE_PREFIX.upper() + "schedule-a-walk"
        assert infer_source(upper) is None
