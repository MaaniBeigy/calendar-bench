"""Tests for `metrics/preference_mapping.py`."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import fakeredis
import pytest

from src.scripts.persona.config.schema import (
    DurationRange,
    EpisodeRange,
    EventDefinition,
    TotalDuration,
)
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
from src.scripts.scenarios.metrics.preference_mapping import (
    MappedEvent,
    MapperConfig,
    PreferenceMapper,
    _bounded_ancestor_hops,
)
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

HEALTH_PREFIX = "https://w3id.org/calendar-bench/health/task/"
ACTIVITY_PREFIX = "https://w3id.org/calendar-bench/human-activities/activity/"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache() -> PreferenceCache:
    return PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))


def _make_event(
    name: str,
    *,
    category: str = "sports",
    health_task_iri: str | None = None,
    human_activity_iri: str | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category=category,
        per_event_duration=DurationRange(min=15, max=120, unit="minutes"),
        total_event_duration=TotalDuration(
            min=15, max=240, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        health_task_iri=health_task_iri,
        human_activity_iri=human_activity_iri,
    )


def _make_task(
    label: str = "schedule-a-30-minute-walk",
    *,
    ontology_uri: str | None = None,
    display_name: str = "",
    description: str = "",
) -> RecommendedTask:
    return RecommendedTask(
        label=label,
        duration_min=30,
        duration_max=30,
        ontology_uri=ontology_uri,
        display_name=display_name,
        description=description,
    )


def _mock_driver_hops(hops_by_iri: dict[tuple[str, str], int | None]) -> object:
    """A Neo4j-driver mock whose `run` returns a fixed hop count per IRI pair."""
    driver = MagicMock()

    @contextmanager
    def session_cm():
        sess = MagicMock()

        def run(_cypher, iri_a: str, iri_b: str):
            result = MagicMock()
            key = (iri_a, iri_b)
            hops = hops_by_iri.get(key)
            if hops is None:
                result.single.return_value = None
            else:
                result.single.return_value = {"hops": hops}
            return result

        sess.run.side_effect = lambda q, iri_a, iri_b: run(q, iri_a, iri_b)
        yield sess

    driver.session = session_cm
    return driver


def _stub_semantic(scores: dict[tuple[str, str], float]) -> SemanticCompatibility:
    """A SemanticCompatibility seeded with an explicit matrix."""
    return SemanticCompatibility(matrix=scores)


# ---------------------------------------------------------------------------
# _bounded_ancestor_hops
# ---------------------------------------------------------------------------


class TestBoundedAncestorHops:
    def test_returns_none_when_driver_is_none(self):
        assert _bounded_ancestor_hops(None, "a", "b", max_hops=2) is None

    def test_returns_none_for_empty_iri(self):
        driver = MagicMock()
        assert _bounded_ancestor_hops(driver, "", "b", max_hops=2) is None
        assert _bounded_ancestor_hops(driver, "a", "", max_hops=2) is None

    def test_returns_zero_when_iris_are_equal(self):
        driver = MagicMock()
        assert _bounded_ancestor_hops(driver, "iri", "iri", max_hops=2) == 0

    def test_returns_hops_from_driver(self):
        driver = _mock_driver_hops({("a", "b"): 2})
        assert _bounded_ancestor_hops(driver, "a", "b", max_hops=3) == 2

    def test_returns_none_when_no_path(self):
        driver = _mock_driver_hops({("a", "b"): None})
        assert _bounded_ancestor_hops(driver, "a", "b", max_hops=2) is None

    def test_driver_exception_swallowed_to_none(self):
        driver = MagicMock()

        @contextmanager
        def bad_session():
            sess = MagicMock()
            sess.run.side_effect = RuntimeError("network down")
            yield sess

        driver.session = bad_session
        assert _bounded_ancestor_hops(driver, "a", "b", max_hops=2) is None


# ---------------------------------------------------------------------------
# PreferenceMapper; tier 1 (literal)
# ---------------------------------------------------------------------------


class TestLiteralTier:
    def test_literal_health_task_iri_match_locks_weight_1(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "schedule-a-30-minute-walk")
        event = _make_event(
            "walking",
            health_task_iri=HEALTH_PREFIX + "schedule-a-30-minute-walk",
        )
        mapper = PreferenceMapper(cache=cache)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert len(matched) == 1
        assert matched[0].weight == 1.0
        assert matched[0].source_tier == "literal"
        assert matched[0].evidence["kind"] == "health_task"

    def test_literal_human_activity_iri_match_locks_weight_1(
        self, cache: PreferenceCache
    ):
        task = _make_task(ontology_uri=ACTIVITY_PREFIX + "walking-3-5-mph-mod-pace")
        event = _make_event(
            "walking",
            human_activity_iri=ACTIVITY_PREFIX + "walking-3-5-mph-mod-pace",
        )
        mapper = PreferenceMapper(cache=cache)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert len(matched) == 1
        assert matched[0].weight == 1.0
        assert matched[0].evidence["kind"] == "human_activity"

    def test_literal_no_uri_on_task_returns_empty(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=None)
        event = _make_event(
            "walking", health_task_iri=HEALTH_PREFIX + "schedule-a-30-minute-walk"
        )
        mapper = PreferenceMapper(cache=cache)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []


# ---------------------------------------------------------------------------
# PreferenceMapper; tier 2 (ancestor)
# ---------------------------------------------------------------------------


class TestAncestorTier:
    def test_ancestor_walk_at_two_hops_gives_decayed_weight(
        self, cache: PreferenceCache
    ):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "schedule-a-30-minute-walk")
        event = _make_event(
            "cycling", health_task_iri=HEALTH_PREFIX + "cycle-for-30-minutes"
        )
        driver = _mock_driver_hops(
            {
                (
                    HEALTH_PREFIX + "schedule-a-30-minute-walk",
                    HEALTH_PREFIX + "cycle-for-30-minutes",
                ): 2
            }
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        # Default: 0.6 * 0.7**2 = 0.294
        assert len(matched) == 1
        assert matched[0].source_tier == "ancestor"
        assert matched[0].weight == pytest.approx(0.294, rel=1e-3)
        assert matched[0].evidence == {
            "hops": 2,
            "via_iri": HEALTH_PREFIX + "cycle-for-30-minutes",
            "kind": "health_task",
        }

    def test_ancestor_walk_one_hop_gives_higher_weight(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event("event-b", health_task_iri=HEALTH_PREFIX + "task-b")
        driver = _mock_driver_hops(
            {(HEALTH_PREFIX + "task-a", HEALTH_PREFIX + "task-b"): 1}
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        # 0.6 * 0.7**1 = 0.42
        assert matched[0].weight == pytest.approx(0.42, rel=1e-3)

    def test_ancestor_no_path_returns_empty(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event("event-b", health_task_iri=HEALTH_PREFIX + "task-b")
        driver = _mock_driver_hops({})  # no entries to None
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_ancestor_no_driver_returns_no_ancestor_match(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event("event-b", health_task_iri=HEALTH_PREFIX + "task-b")
        mapper = PreferenceMapper(cache=cache, neo4j_driver=None)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_ancestor_skips_when_task_has_no_ontology_uri(self, cache: PreferenceCache):
        """Tier 2 must short-circuit when the task carries no URI even
        with a live driver."""
        task = _make_task(ontology_uri=None)
        event = _make_event("event", health_task_iri=HEALTH_PREFIX + "task-b")
        driver = _mock_driver_hops({})
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_ancestor_keeps_first_iri_when_second_is_longer(
        self, cache: PreferenceCache
    ):
        """When both IRIs reach the task, second iteration with longer
        path must NOT overwrite the shorter one from the first
        iteration."""
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event(
            "event",
            health_task_iri=HEALTH_PREFIX + "task-b",
            human_activity_iri=ACTIVITY_PREFIX + "activity-c",
        )
        driver = _mock_driver_hops(
            {
                (HEALTH_PREFIX + "task-a", HEALTH_PREFIX + "task-b"): 1,
                (HEALTH_PREFIX + "task-a", ACTIVITY_PREFIX + "activity-c"): 2,
            }
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].evidence["hops"] == 1
        assert matched[0].evidence["kind"] == "health_task"

    def test_ancestor_prefers_shorter_path_across_both_iri_kinds(
        self, cache: PreferenceCache
    ):
        """When both IRIs reach the task, take whichever has fewer hops."""
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event(
            "event",
            health_task_iri=HEALTH_PREFIX + "task-b",
            human_activity_iri=ACTIVITY_PREFIX + "activity-c",
        )
        driver = _mock_driver_hops(
            {
                (HEALTH_PREFIX + "task-a", HEALTH_PREFIX + "task-b"): 2,
                (HEALTH_PREFIX + "task-a", ACTIVITY_PREFIX + "activity-c"): 1,
            }
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].evidence["hops"] == 1
        assert matched[0].evidence["kind"] == "human_activity"


# ---------------------------------------------------------------------------
# PreferenceMapper; tier 3 (semantic)
# ---------------------------------------------------------------------------


class TestSemanticTier:
    def test_semantic_above_threshold_returns_sigma_weight(
        self, cache: PreferenceCache
    ):
        task = _make_task(
            ontology_uri=None,
            display_name="Take a half-hour walk",
            description="A leisurely walk around the block.",
        )
        event = _make_event("walking", category="sports")
        task_text = "Take a half-hour walk; A leisurely walk around the block."
        event_text = "walking; sports"
        semantic = _stub_semantic({(task_text, event_text): 0.85})
        mapper = PreferenceMapper(cache=cache, semantic=semantic)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert len(matched) == 1
        assert matched[0].source_tier == "semantic"
        assert matched[0].weight == pytest.approx(0.85)
        assert matched[0].evidence["sigma"] == pytest.approx(0.85)

    def test_semantic_below_threshold_returns_empty(self, cache: PreferenceCache):
        task = _make_task(
            ontology_uri=None,
            display_name="Talk to a stranger",
            description="",
        )
        event = _make_event("walking", category="sports")
        semantic = _stub_semantic({("Talk to a stranger", "walking; sports"): 0.4})
        mapper = PreferenceMapper(cache=cache, semantic=semantic)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_semantic_uses_label_title_case_when_display_name_empty(
        self, cache: PreferenceCache
    ):
        """Empty display_name falls back to `label.title()` via
        :attr:`RecommendedTask.effective_display_name`."""
        task = _make_task(label="walk_2000_steps", ontology_uri=None)
        event = _make_event("walking", category="sports")
        # effective_display_name == "Walk 2000 Steps"
        semantic = _stub_semantic({("Walk 2000 Steps", "walking; sports"): 0.9})
        mapper = PreferenceMapper(cache=cache, semantic=semantic)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].weight == pytest.approx(0.9)

    def test_semantic_no_oracle_returns_no_semantic_match(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=None, display_name="Walk", description="")
        event = _make_event("walking")
        mapper = PreferenceMapper(cache=cache, semantic=None)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []


# ---------------------------------------------------------------------------
# PreferenceMapper; tier composition
# ---------------------------------------------------------------------------


class TestTierComposition:
    def test_literal_beats_ancestor_and_semantic(self, cache: PreferenceCache):
        """When a literal IRI match exists for an event, lower tiers are
        not even evaluated for that event."""
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event("event", health_task_iri=HEALTH_PREFIX + "task-a")
        # Drive both lower tiers; they must be ignored.
        driver = _mock_driver_hops(
            {(HEALTH_PREFIX + "task-a", HEALTH_PREFIX + "task-a"): 0}
        )
        semantic = _stub_semantic({})
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver, semantic=semantic)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].source_tier == "literal"
        assert matched[0].weight == 1.0

    def test_ancestor_beats_semantic_when_higher_weight(self, cache: PreferenceCache):
        task = _make_task(
            ontology_uri=HEALTH_PREFIX + "task-a",
            display_name="walk",
        )
        event = _make_event(
            "walking", category="sports", health_task_iri=HEALTH_PREFIX + "task-b"
        )
        driver = _mock_driver_hops(
            {(HEALTH_PREFIX + "task-a", HEALTH_PREFIX + "task-b"): 1}
        )  # ancestor weight = 0.42
        semantic = _stub_semantic(
            {("walk", "walking; sports"): 0.75}
        )  # semantic = 0.75
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver, semantic=semantic)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].source_tier == "semantic"
        assert matched[0].weight == pytest.approx(0.75)

    def test_semantic_beats_ancestor_when_higher_weight(self, cache: PreferenceCache):
        task = _make_task(
            ontology_uri=HEALTH_PREFIX + "task-a",
            display_name="walk",
        )
        event = _make_event(
            "walking", category="sports", health_task_iri=HEALTH_PREFIX + "task-b"
        )
        driver = _mock_driver_hops(
            {(HEALTH_PREFIX + "task-a", HEALTH_PREFIX + "task-b"): 1}
        )  # ancestor = 0.42
        semantic = _stub_semantic({("walk", "walking; sports"): 0.30})  # below 0.7
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver, semantic=semantic)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].source_tier == "ancestor"
        assert matched[0].weight == pytest.approx(0.42, rel=1e-3)


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    def test_first_call_writes_to_cache_second_reads_back(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event("event", health_task_iri=HEALTH_PREFIX + "task-a")
        mapper = PreferenceMapper(cache=cache)
        m1 = mapper.matched_events(task, [event], experiment="e", scenario="s")
        # Cache now has the entry.
        all_ = cache.all_fields("e", "s")
        assert (task.label, event.name) in all_
        # Second call hits the cache (we don't have visibility into Redis
        # call count via fakeredis here, but the result must match).
        m2 = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert m1 == m2

    def test_cache_with_stale_hash_recomputes(self, cache: PreferenceCache):
        """When knobs change, the content_hash changes to old entry stale."""
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-a")
        event = _make_event("event", health_task_iri=HEALTH_PREFIX + "task-a")
        mapper_a = PreferenceMapper(cache=cache, cfg=MapperConfig(ancestor_max_hops=2))
        mapper_a.matched_events(task, [event], experiment="e", scenario="s")
        # Change a knob to content hash differs to re-compute.
        mapper_b = PreferenceMapper(cache=cache, cfg=MapperConfig(ancestor_max_hops=5))
        matched = mapper_b.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].source_tier == "literal"

    def test_warm_up_writes_full_matrix_in_one_pass(self, cache: PreferenceCache):
        tasks = [
            _make_task(label="task-a", ontology_uri=HEALTH_PREFIX + "task-a"),
            _make_task(label="task-b", ontology_uri=HEALTH_PREFIX + "task-b"),
        ]
        events = [
            _make_event("event-1", health_task_iri=HEALTH_PREFIX + "task-a"),
            _make_event("event-2", health_task_iri=HEALTH_PREFIX + "other"),
        ]
        mapper = PreferenceMapper(cache=cache)
        computed = mapper.warm_up(tasks, events, experiment="e", scenario="s")
        assert computed == 4  # 2 tasks × 2 events
        all_ = cache.all_fields("e", "s")
        # task-a × event-1 is literal hit.
        assert all_[("task-a", "event-1")].source_tier == "literal"
        assert all_[("task-a", "event-1")].weight == 1.0
        # task-b × event-2 has neither literal IRI nor driver/semantic to weight 0.
        assert all_[("task-b", "event-2")].weight == 0.0

    def test_warm_up_skips_pairs_with_fresh_cache_entry(self, cache: PreferenceCache):
        tasks = [_make_task(label="task-a", ontology_uri=HEALTH_PREFIX + "task-a")]
        events = [_make_event("event-1", health_task_iri=HEALTH_PREFIX + "task-a")]
        mapper = PreferenceMapper(cache=cache)
        first = mapper.warm_up(tasks, events, experiment="e", scenario="s")
        second = mapper.warm_up(tasks, events, experiment="e", scenario="s")
        assert first == 1
        assert second == 0  # all cached

    def test_warm_up_with_no_tasks_or_no_events_returns_zero(
        self, cache: PreferenceCache
    ):
        mapper = PreferenceMapper(cache=cache)
        assert mapper.warm_up([], [_make_event("e")], experiment="e", scenario="s") == 0
        assert mapper.warm_up([_make_task()], [], experiment="e", scenario="s") == 0

    def test_matched_events_empty_event_list_returns_empty(
        self, cache: PreferenceCache
    ):
        mapper = PreferenceMapper(cache=cache)
        assert (
            mapper.matched_events(_make_task(), [], experiment="e", scenario="s") == []
        )

    def test_cached_zero_weight_entry_does_not_appear_in_results(
        self, cache: PreferenceCache
    ):
        """When a previous call cached a no-match (weight 0) the next
        call must NOT surface it in the matched-events list."""
        task = _make_task(label="task", ontology_uri=None)  # no URI to no match
        event = _make_event("event")
        mapper = PreferenceMapper(cache=cache)
        first = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert first == []
        # The cache has a sentinel weight=0 entry; a second call must
        # honour it (no recompute, no leak into results).
        second = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert second == []
        # Sanity: the sentinel was actually written.
        assert (task.label, event.name) in cache.all_fields("e", "s")
        assert cache.all_fields("e", "s")[(task.label, event.name)].weight == 0.0


# ---------------------------------------------------------------------------
# Misc / equality
# ---------------------------------------------------------------------------


class TestMappedEvent:
    def test_default_evidence_empty_dict(self):
        m = MappedEvent(event_name="x", weight=0.5, source_tier="ancestor")
        assert m.evidence == {}

    def test_frozen_so_equality_works(self):
        m1 = MappedEvent(
            event_name="x", weight=0.5, source_tier="ancestor", evidence={"k": 1}
        )
        m2 = MappedEvent(
            event_name="x", weight=0.5, source_tier="ancestor", evidence={"k": 1}
        )
        assert m1 == m2


# ---------------------------------------------------------------------------
# fetch_matched_activities
# ---------------------------------------------------------------------------


from src.scripts.scenarios.metrics.preference_mapping import fetch_matched_activities


def _mock_driver_matched(matched_by_task: dict[str, list[str]]):
    """Mock driver whose MATCHEDACTIVITY query returns the configured targets."""
    driver = MagicMock()

    @contextmanager
    def session_cm():
        sess = MagicMock()

        def run(_cypher, **kw):
            task_iri = kw.get("task_iri", "")
            targets = matched_by_task.get(task_iri, [])
            result = MagicMock()
            result.__iter__ = lambda self: iter([{"uri": t} for t in targets])
            return result

        sess.run.side_effect = run
        yield sess

    driver.session = session_cm
    return driver


def _mock_driver_combined(
    matched_by_task: dict[str, list[str]],
    hops_by_iri: dict[tuple[str, str], int | None],
):
    """Mock driver that answers both MATCHEDACTIVITY and ancestor queries."""
    driver = MagicMock()

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
            hops = hops_by_iri.get(key)
            result.single.return_value = None if hops is None else {"hops": hops}
            return result

        sess.run.side_effect = run
        yield sess

    driver.session = session_cm
    return driver


class TestFetchMatchedActivities:
    def test_returns_empty_when_driver_none(self):
        assert fetch_matched_activities(None, "iri") == []

    def test_returns_empty_when_task_iri_blank(self):
        driver = MagicMock()
        assert fetch_matched_activities(driver, "") == []

    def test_returns_uris_from_driver(self):
        driver = _mock_driver_matched(
            {"task-x": ["ha-act:aerobic-general", "ha-act:bicycling-general"]}
        )
        out = fetch_matched_activities(driver, "task-x")
        assert out == ["ha-act:aerobic-general", "ha-act:bicycling-general"]


# ---------------------------------------------------------------------------
# PreferenceMapper; tier 1.5 (cross-ontology bridge)
# ---------------------------------------------------------------------------


class TestCrossOntologyTier:
    def test_cross_literal_match_returns_weight_1(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-10-minutes-of-cardio")
        event = _make_event(
            "cycling",
            human_activity_iri=ACTIVITY_PREFIX + "bicycling-general",
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX
                + "do-10-minutes-of-cardio": [ACTIVITY_PREFIX + "bicycling-general"]
            },
            hops_by_iri={},
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert len(matched) == 1
        assert matched[0].weight == 1.0
        assert matched[0].source_tier == "cross_literal"
        assert (
            matched[0].evidence["source_iri"] == ACTIVITY_PREFIX + "bicycling-general"
        )
        assert matched[0].evidence["hops"] == 0

    def test_cross_ancestor_walk_gives_decayed_weight(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-10-minutes-of-cardio")
        event = _make_event(
            "cycling",
            human_activity_iri=ACTIVITY_PREFIX + "bicycling-general",
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX
                + "do-10-minutes-of-cardio": [ACTIVITY_PREFIX + "aerobic-general"]
            },
            hops_by_iri={
                (
                    ACTIVITY_PREFIX + "aerobic-general",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 2
            },
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert len(matched) == 1
        assert matched[0].source_tier == "cross_ancestor"
        assert matched[0].weight == pytest.approx(0.294, rel=1e-3)
        assert matched[0].evidence["hops"] == 2

    def test_cross_returns_none_without_human_activity_iri(
        self, cache: PreferenceCache
    ):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event("e", health_task_iri=HEALTH_PREFIX + "task-b")
        driver = _mock_driver_combined(
            matched_by_task={HEALTH_PREFIX + "do-cardio": [ACTIVITY_PREFIX + "a"]},
            hops_by_iri={},
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_cross_returns_none_without_driver(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event("e", human_activity_iri=ACTIVITY_PREFIX + "bicycling")
        mapper = PreferenceMapper(cache=cache, neo4j_driver=None)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_cross_picks_best_of_multi_targets(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event(
            "cycling",
            human_activity_iri=ACTIVITY_PREFIX + "bicycling-general",
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX
                + "do-cardio": [
                    ACTIVITY_PREFIX + "aerobic-general",
                    ACTIVITY_PREFIX + "bicycling-general",
                ]
            },
            hops_by_iri={
                (
                    ACTIVITY_PREFIX + "aerobic-general",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 2
            },
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].weight == 1.0
        assert matched[0].source_tier == "cross_literal"
        candidates = matched[0].evidence["candidates"]
        assert {c["source_iri"] for c in candidates} == {
            ACTIVITY_PREFIX + "aerobic-general",
            ACTIVITY_PREFIX + "bicycling-general",
        }

    def test_cross_picks_shortest_path_among_ancestors(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event(
            "cycling", human_activity_iri=ACTIVITY_PREFIX + "bicycling-general"
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX
                + "do-cardio": [
                    ACTIVITY_PREFIX + "far-target",
                    ACTIVITY_PREFIX + "near-target",
                ]
            },
            hops_by_iri={
                (
                    ACTIVITY_PREFIX + "far-target",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 2,
                (
                    ACTIVITY_PREFIX + "near-target",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 1,
            },
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].source_tier == "cross_ancestor"
        assert matched[0].evidence["hops"] == 1
        assert matched[0].evidence["source_iri"] == ACTIVITY_PREFIX + "near-target"

    def test_cross_defers_to_literal_tier_when_present(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event(
            "cycling",
            health_task_iri=HEALTH_PREFIX + "do-cardio",
            human_activity_iri=ACTIVITY_PREFIX + "bicycling-general",
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX + "do-cardio": [ACTIVITY_PREFIX + "aerobic-general"]
            },
            hops_by_iri={
                (
                    ACTIVITY_PREFIX + "aerobic-general",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 1
            },
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].source_tier == "literal"
        assert matched[0].weight == 1.0

    def test_cross_returns_zero_when_no_targets_on_task(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "task-x")
        event = _make_event("e", human_activity_iri=ACTIVITY_PREFIX + "bicycling")
        driver = _mock_driver_combined(matched_by_task={}, hops_by_iri={})
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_matched_activities_cached_across_calls(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event_a = _make_event("a", human_activity_iri=ACTIVITY_PREFIX + "x")
        event_b = _make_event("b", human_activity_iri=ACTIVITY_PREFIX + "y")
        driver = MagicMock()
        call_count = {"n": 0}

        @contextmanager
        def session_cm():
            sess = MagicMock()

            def run(cypher, **kw):
                result = MagicMock()
                if ":MATCHEDACTIVITY" in cypher:
                    call_count["n"] += 1
                    result.__iter__ = lambda self: iter(
                        [{"uri": ACTIVITY_PREFIX + "src"}]
                    )
                    return result
                result.single.return_value = None
                return result

            sess.run.side_effect = run
            yield sess

        driver.session = session_cm
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        mapper.matched_events(task, [event_a, event_b], experiment="e", scenario="s")
        assert call_count["n"] == 1

    def test_cross_returns_none_without_task_ontology_uri(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=None)
        event = _make_event(
            "cycling", human_activity_iri=ACTIVITY_PREFIX + "bicycling-general"
        )
        driver = _mock_driver_combined(matched_by_task={}, hops_by_iri={})
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched == []

    def test_cross_second_literal_does_not_overwrite_first(
        self, cache: PreferenceCache
    ):
        # Two targets both equal the event IRI; the first sets best=1.0,
        # the second hits the same branch but does not overwrite.
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event(
            "cycling", human_activity_iri=ACTIVITY_PREFIX + "bicycling-general"
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX
                + "do-cardio": [
                    ACTIVITY_PREFIX + "bicycling-general",
                    ACTIVITY_PREFIX + "bicycling-general",
                ]
            },
            hops_by_iri={},
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].weight == 1.0
        assert matched[0].source_tier == "cross_literal"
        assert len(matched[0].evidence["candidates"]) == 2

    def test_cross_lower_ancestor_does_not_overwrite_higher(
        self, cache: PreferenceCache
    ):
        # First target reaches the event at 1 hop (weight 0.42); the
        # second target at 2 hops (0.294) must NOT overwrite the best.
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event(
            "cycling", human_activity_iri=ACTIVITY_PREFIX + "bicycling-general"
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX
                + "do-cardio": [
                    ACTIVITY_PREFIX + "near-target",
                    ACTIVITY_PREFIX + "far-target",
                ]
            },
            hops_by_iri={
                (
                    ACTIVITY_PREFIX + "near-target",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 1,
                (
                    ACTIVITY_PREFIX + "far-target",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 2,
            },
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].evidence["hops"] == 1
        assert matched[0].evidence["source_iri"] == ACTIVITY_PREFIX + "near-target"

    def test_cross_ancestor_decayed_weight_beats_lower_ancestor_match(
        self, cache: PreferenceCache
    ):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "do-cardio")
        event = _make_event(
            "cycling",
            health_task_iri=HEALTH_PREFIX + "cardio-task",
            human_activity_iri=ACTIVITY_PREFIX + "bicycling-general",
        )
        driver = _mock_driver_combined(
            matched_by_task={
                HEALTH_PREFIX + "do-cardio": [ACTIVITY_PREFIX + "aerobic-general"]
            },
            hops_by_iri={
                # Tier 2 ancestor walk is far (2 hops to weight 0.294).
                (HEALTH_PREFIX + "do-cardio", HEALTH_PREFIX + "cardio-task"): 2,
                # Cross-ancestor is close (1 hop to weight 0.42).
                (
                    ACTIVITY_PREFIX + "aerobic-general",
                    ACTIVITY_PREFIX + "bicycling-general",
                ): 1,
            },
        )
        mapper = PreferenceMapper(cache=cache, neo4j_driver=driver)
        matched = mapper.matched_events(task, [event], experiment="e", scenario="s")
        assert matched[0].source_tier == "cross_ancestor"
        assert matched[0].weight == pytest.approx(0.42, rel=1e-3)


# ---------------------------------------------------------------------------
# Cache invalidation under new knobs and ontology_version
# ---------------------------------------------------------------------------


class TestHashStability:
    def test_hash_changes_when_cross_knobs_change(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "t")
        event = _make_event("e", human_activity_iri=ACTIVITY_PREFIX + "a")
        m1 = PreferenceMapper(cache=cache)
        m2 = PreferenceMapper(
            cache=cache,
            cfg=MapperConfig(cross_ancestor_max_hops=4),
        )
        assert m1._hash_for(task, event) != m2._hash_for(task, event)

    def test_hash_changes_when_ontology_version_changes(self, cache: PreferenceCache):
        task = _make_task(ontology_uri=HEALTH_PREFIX + "t")
        event = _make_event("e", human_activity_iri=ACTIVITY_PREFIX + "a")
        m1 = PreferenceMapper(cache=cache, ontology_version="2026.05.06")
        m2 = PreferenceMapper(cache=cache, ontology_version="2026.05.18")
        assert m1._hash_for(task, event) != m2._hash_for(task, event)
