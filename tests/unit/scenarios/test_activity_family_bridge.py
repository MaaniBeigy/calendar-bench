"""Tests for `ActivityFamilyBridge` and its wiring into `compute_l_concurrent`."""

from __future__ import annotations

import datetime
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityResolver,
    MetQuartiles,
)
from src.scripts.scenarios.metrics.loss import _bridged_score, compute_l_concurrent
from src.scripts.scenarios.metrics.preference_mapping import ActivityFamilyBridge
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

HEALTH = "https://w3id.org/calendar-bench/health/task/"
ACTIVITY = "https://w3id.org/calendar-bench/human-activities/activity/"
DATE = datetime.date(2026, 5, 4)


# ---------------------------------------------------------------------------
# Driver mocks
# ---------------------------------------------------------------------------


def _driver(
    matched_by_task: dict[str, list[str]],
    hops_by_pair: dict[tuple[str, str], int | None],
):
    drv = MagicMock()

    @contextmanager
    def session_cm():
        sess = MagicMock()

        def run(cypher, **kw):
            result = MagicMock()
            if ":MATCHEDACTIVITY" in cypher:
                targets = matched_by_task.get(kw.get("task_iri", ""), [])
                result.__iter__ = lambda self: iter([{"uri": t} for t in targets])
                return result
            key = (kw.get("iri_a", ""), kw.get("iri_b", ""))
            hops = hops_by_pair.get(key)
            result.single.return_value = None if hops is None else {"hops": hops}
            return result

        sess.run.side_effect = run
        yield sess

    drv.session = session_cm
    return drv


# ---------------------------------------------------------------------------
# ActivityFamilyBridge
# ---------------------------------------------------------------------------


class TestActivityFamilyBridge:
    def test_shares_family_false_when_driver_missing(self):
        bridge = ActivityFamilyBridge(driver=None)
        assert bridge.shares_family(HEALTH + "task", "lunch") is False

    def test_shares_family_false_when_task_iri_blank(self):
        bridge = ActivityFamilyBridge(driver=MagicMock())
        assert bridge.shares_family("", "lunch") is False

    def test_shares_family_false_when_host_label_blank(self):
        bridge = ActivityFamilyBridge(driver=MagicMock())
        assert bridge.shares_family(HEALTH + "task", "") is False

    def test_shares_family_false_when_host_has_no_iri(self):
        bridge = ActivityFamilyBridge(
            driver=MagicMock(), event_ha_iri_by_label={"other": ACTIVITY + "x"}
        )
        assert bridge.shares_family(HEALTH + "task", "lunch") is False

    def test_shares_family_true_via_literal_match(self):
        bridge = ActivityFamilyBridge(
            driver=_driver(
                matched_by_task={HEALTH + "task": [ACTIVITY + "aerobic-general"]},
                hops_by_pair={},
            ),
            event_ha_iri_by_label={"cycling": ACTIVITY + "aerobic-general"},
        )
        assert bridge.shares_family(HEALTH + "task", "cycling") is True

    def test_shares_family_true_via_ancestor_walk(self):
        bridge = ActivityFamilyBridge(
            driver=_driver(
                matched_by_task={HEALTH + "task": [ACTIVITY + "aerobic-general"]},
                hops_by_pair={
                    (
                        ACTIVITY + "aerobic-general",
                        ACTIVITY + "bicycling-general",
                    ): 1
                },
            ),
            event_ha_iri_by_label={"cycling": ACTIVITY + "bicycling-general"},
        )
        assert bridge.shares_family(HEALTH + "task", "cycling") is True

    def test_shares_family_false_when_no_path(self):
        bridge = ActivityFamilyBridge(
            driver=_driver(
                matched_by_task={HEALTH + "task": [ACTIVITY + "aerobic-general"]},
                hops_by_pair={},
            ),
            event_ha_iri_by_label={"cycling": ACTIVITY + "bicycling-general"},
        )
        assert bridge.shares_family(HEALTH + "task", "cycling") is False

    def test_shares_family_caches_result(self):
        # Counter the driver's MATCHEDACTIVITY calls; second lookup must hit
        # the bridge's cache without re-querying.
        call_count = {"n": 0}
        drv = MagicMock()

        @contextmanager
        def session_cm():
            sess = MagicMock()

            def run(cypher, **kw):
                result = MagicMock()
                if ":MATCHEDACTIVITY" in cypher:
                    call_count["n"] += 1
                    result.__iter__ = lambda self: iter(
                        [{"uri": ACTIVITY + "aerobic-general"}]
                    )
                    return result
                result.single.return_value = {"hops": 1}
                return result

            sess.run.side_effect = run
            yield sess

        drv.session = session_cm
        bridge = ActivityFamilyBridge(
            driver=drv,
            event_ha_iri_by_label={"cycling": ACTIVITY + "bicycling-general"},
        )
        assert bridge.shares_family(HEALTH + "task", "cycling") is True
        assert bridge.shares_family(HEALTH + "task", "cycling") is True
        # MATCHEDACTIVITY runs once; result cache prevents re-query.
        assert call_count["n"] == 1

    def test_matched_activities_returns_empty_for_blank_task(self):
        bridge = ActivityFamilyBridge(driver=MagicMock())
        assert bridge.matched_activities("") == []

    def test_matched_activities_returns_empty_when_driver_none(self):
        bridge = ActivityFamilyBridge(driver=None)
        assert bridge.matched_activities(HEALTH + "task") == []

    def test_matched_activities_caches_lookup(self):
        # Direct calls to matched_activities should hit the per-task cache.
        call_count = {"n": 0}
        drv = MagicMock()

        @contextmanager
        def session_cm():
            sess = MagicMock()

            def run(_cypher, **kw):
                call_count["n"] += 1
                result = MagicMock()
                result.__iter__ = lambda self: iter(
                    [{"uri": ACTIVITY + "aerobic-general"}]
                )
                return result

            sess.run.side_effect = run
            yield sess

        drv.session = session_cm
        bridge = ActivityFamilyBridge(driver=drv)
        first = bridge.matched_activities(HEALTH + "task")
        second = bridge.matched_activities(HEALTH + "task")
        assert first == second == [ACTIVITY + "aerobic-general"]
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# _bridged_score helper
# ---------------------------------------------------------------------------


def _semantic_matrix(value: float) -> SemanticCompatibility:
    return SemanticCompatibility(matrix={("read", "lunch"): value})


def _host(label: str) -> CalendarEvent:
    return CalendarEvent(
        label=label,
        start_minutes=720,
        end_minutes=780,
        date=datetime.date(2026, 6, 1),
    )


class TestBridgedScore:
    def test_no_bridge_falls_back_to_semantic(self):
        task = RecommendedTask(
            label="read",
            duration_min=30,
            duration_max=60,
            ontology_uri=HEALTH + "task",
        )
        score = _bridged_score(None, task, _host("lunch"), _semantic_matrix(0.42))
        assert score == pytest.approx(0.42)

    def test_bridge_hit_returns_one(self):
        task = RecommendedTask(
            label="read",
            duration_min=30,
            duration_max=60,
            ontology_uri=HEALTH + "task",
        )
        bridge = ActivityFamilyBridge(
            driver=_driver(
                matched_by_task={HEALTH + "task": [ACTIVITY + "x"]},
                hops_by_pair={},
            ),
            event_ha_iri_by_label={"lunch": ACTIVITY + "x"},
        )
        score = _bridged_score(bridge, task, _host("lunch"), _semantic_matrix(0.42))
        assert score == 1.0

    def test_bridge_miss_falls_back_to_semantic(self):
        task = RecommendedTask(
            label="read",
            duration_min=30,
            duration_max=60,
            ontology_uri=HEALTH + "task",
        )
        bridge = ActivityFamilyBridge(
            driver=_driver(
                matched_by_task={HEALTH + "task": [ACTIVITY + "x"]},
                hops_by_pair={},
            ),
            event_ha_iri_by_label={"lunch": ACTIVITY + "y"},
        )
        score = _bridged_score(bridge, task, _host("lunch"), _semantic_matrix(0.42))
        assert score == pytest.approx(0.42)

    def test_bridge_skipped_when_task_has_no_ontology_uri(self):
        task = RecommendedTask(
            label="read",
            duration_min=30,
            duration_max=60,
            ontology_uri=None,
        )
        bridge = ActivityFamilyBridge(driver=MagicMock())
        score = _bridged_score(bridge, task, _host("lunch"), _semantic_matrix(0.42))
        assert score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# compute_l_concurrent with bridge wiring
# ---------------------------------------------------------------------------


def _scheduled(task: RecommendedTask, **kw) -> ScheduledTask:
    return ScheduledTask(
        task=task,
        date=kw.get("date", DATE),
        start_minutes=kw.get("start", 720),
        end_minutes=kw.get("end", 780),
        is_standalone=kw.get("is_standalone", False),
        concurrent_with=kw.get("concurrent_with"),
    )


def _event(**kw) -> CalendarEvent:
    return CalendarEvent(
        label=kw.get("label", "lunch"),
        start_minutes=kw.get("start", 720),
        end_minutes=kw.get("end", 780),
        date=kw.get("date", DATE),
        is_concurrent=kw.get("is_concurrent", True),
    )


def _solution(tasks, scheduled, person="p001") -> SchedulingSolution:
    return SchedulingSolution(
        person_id=person,
        tasks=list(tasks),
        scheduled=list(scheduled),
        unscheduled=[],
        augmented_calendar=None,
    )


def _trace(events) -> CalendarTrace:
    return CalendarTrace(person_id="p001", events=list(events))


def test_compute_l_concurrent_bridge_replaces_semantic_for_match():
    task = RecommendedTask(
        label="cardio",
        duration_min=10,
        duration_max=10,
        ontology_uri=HEALTH + "do-10-minutes-of-cardio",
        is_concurrent=True,
    )
    event = _event(label="cycling")
    st = _scheduled(task, is_standalone=False, concurrent_with="cycling")
    sol = _solution([task], [st])
    bridge = ActivityFamilyBridge(
        driver=_driver(
            matched_by_task={
                HEALTH + "do-10-minutes-of-cardio": [ACTIVITY + "aerobic-general"]
            },
            hops_by_pair={
                (
                    ACTIVITY + "aerobic-general",
                    ACTIVITY + "bicycling-general",
                ): 1
            },
        ),
        event_ha_iri_by_label={"cycling": ACTIVITY + "bicycling-general"},
    )
    # Stub the semantic oracle to return a low score that would otherwise
    # disqualify the pair; the bridge must override to 1.0.
    semantic = SemanticCompatibility(matrix={("cardio", "cycling"): 0.10})
    value = compute_l_concurrent(
        sol,
        _trace([event]),
        semantic,
        ruleset=RuleSet([]),
        matcher=SelectorMatcher(
            resolver=IntensityResolver(MetQuartiles(1.8, 3.0, 6.0, 23.0))
        ),
        merge_threshold=0.5,
        bridge=bridge,
    )
    # σ_actual = σ_best = 1.0 to L_merge = 1 - 1.0/1.0 = 0.0
    assert value == pytest.approx(0.0)


def test_compute_l_concurrent_without_bridge_uses_semantic_only():
    task = RecommendedTask(
        label="cardio",
        duration_min=10,
        duration_max=10,
        ontology_uri=HEALTH + "do-10-minutes-of-cardio",
        is_concurrent=True,
    )
    event = _event(label="cycling")
    st = _scheduled(task, is_standalone=False, concurrent_with="cycling")
    sol = _solution([task], [st])
    semantic = SemanticCompatibility(matrix={("cardio", "cycling"): 0.80})
    value = compute_l_concurrent(
        sol,
        _trace([event]),
        semantic,
        ruleset=RuleSet([]),
        matcher=SelectorMatcher(
            resolver=IntensityResolver(MetQuartiles(1.8, 3.0, 6.0, 23.0))
        ),
        merge_threshold=0.5,
    )
    # σ_actual = σ_best = 0.80 to L_merge = 0.0; sanity check that the
    # original (bridge-less) path still works.
    assert value == pytest.approx(0.0)
