"""Tests for the MET-bridge extensions in `metrics.met_lookup`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityResolver,
    MetQuartiles,
)
from src.scripts.scenarios.metrics.met_lookup import (
    HUMAN_ACTIVITIES_PREFIX,
    MatchedActivityMetIndex,
    MetLookup,
    MetMatch,
    load_activity_met_index,
    load_matched_activity_index,
)

HA_ACT_PREFIX = HUMAN_ACTIVITIES_PREFIX + "activity/"
HB_TASK_PREFIX = "https://w3id.org/calendar-bench/health/task/"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ha_ttl(tmp_path: Path) -> Path:
    body = """
@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .

ha:metValue a owl:DatatypeProperty .

ha-act:aerobic-general a ha:SportsExerciseWorkoutConditioningExercise ;
    rdfs:label "Aerobic, general"@en ;
    ha:metValue "7.0"^^xsd:decimal .

ha-act:bicycling-general a ha:SportsExerciseWorkoutBicycling ;
    rdfs:label "Bicycling, general"@en ;
    ha:metValue "7.5"^^xsd:decimal .

ha-act:eating-sitting a ha:EverydayTasksSelfCare ;
    rdfs:label "Eating, sitting"@en ;
    ha:metValue "1.5"^^xsd:decimal .

ha-act:weird-no-met a ha:EverydayTasksSelfCare ;
    rdfs:label "Weird, no MET"@en .

ha-act:weird-bad-met a ha:EverydayTasksSelfCare ;
    rdfs:label "Weird, bad MET"@en ;
    ha:metValue "not-a-number" .
"""
    path = tmp_path / "ha.ttl"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def matched_jsonl(tmp_path: Path) -> Path:
    records = [
        {
            "task_iri": HB_TASK_PREFIX + "do-10-minutes-of-cardio",
            "matched_activity_iris": [HA_ACT_PREFIX + "aerobic-general"],
        },
        {
            "task_iri": HB_TASK_PREFIX + "ride-easy",
            "matched_activity_iris": [
                HA_ACT_PREFIX + "aerobic-general",
                HA_ACT_PREFIX + "bicycling-general",
            ],
        },
        {
            "task_iri": HB_TASK_PREFIX + "no-targets",
            "matched_activity_iris": [],
        },
    ]
    path = tmp_path / "matched.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_activity_met_index
# ---------------------------------------------------------------------------


def test_load_activity_met_index_returns_numeric_values(ha_ttl: Path):
    idx = load_activity_met_index(ha_ttl)
    assert idx[HA_ACT_PREFIX + "aerobic-general"] == pytest.approx(7.0)
    assert idx[HA_ACT_PREFIX + "bicycling-general"] == pytest.approx(7.5)
    assert idx[HA_ACT_PREFIX + "eating-sitting"] == pytest.approx(1.5)


def test_load_activity_met_index_skips_missing_and_non_numeric(ha_ttl: Path):
    idx = load_activity_met_index(ha_ttl)
    assert HA_ACT_PREFIX + "weird-no-met" not in idx
    assert HA_ACT_PREFIX + "weird-bad-met" not in idx


def test_load_activity_met_index_skips_off_prefix_subjects(tmp_path: Path):
    # A metValue triple on a non-activity subject should be dropped.
    ttl = tmp_path / "off.ttl"
    ttl.write_text(
        """
@prefix ha:      <https://w3id.org/calendar-bench/human-activities/> .
@prefix ha-act:  <https://w3id.org/calendar-bench/human-activities/activity/> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .

ha:SomeClass ha:metValue "5.0"^^xsd:decimal .

ha-act:walking ha:metValue "3.0"^^xsd:decimal .
""",
        encoding="utf-8",
    )
    idx = load_activity_met_index(ttl)
    assert idx == {HA_ACT_PREFIX + "walking": 3.0}


# ---------------------------------------------------------------------------
# MatchedActivityMetIndex
# ---------------------------------------------------------------------------


class TestMatchedActivityMetIndex:
    def test_met_for_task_empty_iri(self):
        idx = MatchedActivityMetIndex({}, {})
        assert idx.met_for_task("") == (None, None)

    def test_met_for_task_no_entry(self):
        idx = MatchedActivityMetIndex({}, {})
        assert idx.met_for_task("task-x") == (None, None)

    def test_met_for_task_single_target(self):
        idx = MatchedActivityMetIndex(
            {"task-x": ("ha-act:aerobic-general",)},
            {"ha-act:aerobic-general": 7.0},
        )
        assert idx.met_for_task("task-x") == (7.0, "ha-act:aerobic-general")

    def test_met_for_task_picks_highest_met_across_targets(self):
        idx = MatchedActivityMetIndex(
            {"task-x": ("ha-act:eating-sitting", "ha-act:bicycling-general")},
            {
                "ha-act:eating-sitting": 1.5,
                "ha-act:bicycling-general": 7.5,
            },
        )
        assert idx.met_for_task("task-x") == (7.5, "ha-act:bicycling-general")

    def test_met_for_task_skips_targets_without_met(self):
        idx = MatchedActivityMetIndex(
            {"task-x": ("ha-act:unknown", "ha-act:aerobic-general")},
            {"ha-act:aerobic-general": 7.0},
        )
        assert idx.met_for_task("task-x") == (7.0, "ha-act:aerobic-general")

    def test_met_for_task_lower_met_target_does_not_overwrite(self):
        # Order: high MET first, then low. The second iteration should
        # NOT update the best when its MET is below the current top.
        idx = MatchedActivityMetIndex(
            {"task-x": ("ha-act:bicycling-general", "ha-act:eating-sitting")},
            {
                "ha-act:bicycling-general": 7.5,
                "ha-act:eating-sitting": 1.5,
            },
        )
        assert idx.met_for_task("task-x") == (7.5, "ha-act:bicycling-general")

    def test_met_for_task_all_targets_missing_returns_none(self):
        idx = MatchedActivityMetIndex(
            {"task-x": ("ha-act:unknown-a", "ha-act:unknown-b")},
            {},
        )
        assert idx.met_for_task("task-x") == (None, None)


# ---------------------------------------------------------------------------
# load_matched_activity_index
# ---------------------------------------------------------------------------


def test_load_matched_activity_index_round_trip(matched_jsonl: Path, ha_ttl: Path):
    idx = load_matched_activity_index(matched_jsonl, ha_ttl)
    assert idx.task_iri_to_activities[HB_TASK_PREFIX + "do-10-minutes-of-cardio"] == (
        HA_ACT_PREFIX + "aerobic-general",
    )
    assert idx.task_iri_to_activities[HB_TASK_PREFIX + "ride-easy"] == (
        HA_ACT_PREFIX + "aerobic-general",
        HA_ACT_PREFIX + "bicycling-general",
    )
    assert HB_TASK_PREFIX + "no-targets" not in idx.task_iri_to_activities
    met, uri = idx.met_for_task(HB_TASK_PREFIX + "ride-easy")
    assert met == pytest.approx(7.5)
    assert uri == HA_ACT_PREFIX + "bicycling-general"


def test_load_matched_activity_index_skips_blank_lines(tmp_path: Path, ha_ttl: Path):
    path = tmp_path / "blank.jsonl"
    path.write_text(
        "\n\n"
        + json.dumps(
            {
                "task_iri": HB_TASK_PREFIX + "x",
                "matched_activity_iris": [HA_ACT_PREFIX + "aerobic-general"],
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    idx = load_matched_activity_index(path, ha_ttl)
    assert len(idx.task_iri_to_activities) == 1


# ---------------------------------------------------------------------------
# MetLookup.met_for_task
# ---------------------------------------------------------------------------


def _empty_lookup() -> MetLookup:
    return MetLookup()


class TestMetLookupForTask:
    def test_empty_task_iri_without_fallback_returns_none_source(self):
        lookup = _empty_lookup()
        match = lookup.met_for_task("")
        assert match == MetMatch(met=None, uri=None, cosine=None, source="none")

    def test_empty_task_iri_with_fallback_calls_query(self):
        lookup = MetLookup()
        # Without a driver/embedder, met() returns an empty match.
        match = lookup.met_for_task("", fallback_query="walking")
        assert match.source == "none"

    def test_bridge_hit_returns_bridge_source(self):
        bridge = MatchedActivityMetIndex(
            {"task-x": ("ha-act:aerobic-general",)},
            {"ha-act:aerobic-general": 7.0},
        )
        lookup = MetLookup(bridge=bridge)
        match = lookup.met_for_task("task-x")
        assert match.met == pytest.approx(7.0)
        assert match.uri == "ha-act:aerobic-general"
        assert match.source == "bridge"

    def test_bridge_miss_falls_back_to_query(self):
        bridge = MatchedActivityMetIndex({}, {})
        lookup = MetLookup(bridge=bridge)
        match = lookup.met_for_task("task-x", fallback_query="walking")
        # No driver wired to fallback returns empty source=none.
        assert match.source == "none"

    def test_no_bridge_wired_falls_through_to_embedding_path(self):
        # No bridge attached; met_for_task degrades to met(fallback_query).
        lookup = MetLookup()  # no bridge
        match = lookup.met_for_task("task-x", fallback_query="walking")
        # Without driver/embedder, met() returns empty match.
        assert match.source == "none"

    def test_per_task_cache_short_circuits_second_call(self):
        bridge = MatchedActivityMetIndex(
            {"task-x": ("ha-act:aerobic-general",)},
            {"ha-act:aerobic-general": 7.0},
        )
        lookup = MetLookup(bridge=bridge)
        first = lookup.met_for_task("task-x")
        # Mutate the bridge map after the first lookup; the per-task
        # cache must NOT re-read it.
        bridge.task_iri_to_activities.clear()
        second = lookup.met_for_task("task-x")
        assert first == second


# ---------------------------------------------------------------------------
# Cache round-trip with `source` field
# ---------------------------------------------------------------------------


def test_cache_round_trip_preserves_source(tmp_path: Path):
    cache_path = tmp_path / "cache.jsonl"
    lookup = MetLookup(cache_path=cache_path)
    lookup._save_to_cache(
        "walking",
        MetMatch(met=3.0, uri="ha-act:walking", cosine=0.9, source="embedding"),
    )
    reborn = MetLookup(cache_path=cache_path)
    assert reborn._cache["walking"].source == "embedding"


def test_cache_round_trip_defaults_to_none_when_missing(tmp_path: Path):
    cache_path = tmp_path / "legacy.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "query_text": "walking",
                "met": 3.0,
                "uri": "ha-act:walking",
                "cosine": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lookup = MetLookup(cache_path=cache_path)
    assert lookup._cache["walking"].source == "none"


# ---------------------------------------------------------------------------
# Integration: IntensityResolver reads bridge MET
# ---------------------------------------------------------------------------


def test_intensity_resolver_uses_bridge_via_met_for_task():
    bridge = MatchedActivityMetIndex(
        {HB_TASK_PREFIX + "do-cardio": ("ha-act:aerobic-general",)},
        {"ha-act:aerobic-general": 7.0},
    )
    lookup = MetLookup(bridge=bridge)
    quartiles = MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=23.0)
    resolver = IntensityResolver(quartiles, met_lookup=lookup)
    task = RecommendedTask(
        label="do-cardio",
        duration_min=10,
        duration_max=20,
        ontology_uri=HB_TASK_PREFIX + "do-cardio",
    )
    record = resolver.record(task)
    assert record.met == pytest.approx(7.0)
    assert record.matched_uri == "ha-act:aerobic-general"
    # MET 7.0 > q3 6.0 to bucket 4.
    assert record.met_bucket == 4
