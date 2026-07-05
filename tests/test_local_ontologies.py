"""Exact-content tests for the locally-developed ontologies.

Unlike test_imports.py (which only checks "some sample URIs landed in Neo4j"),
these tests pin specific URIs, labels, parent classes, and numeric data
properties from real entries in the author's TTL files. They fail loudly if
n10s ever loses fidelity on a future Turtle/RDF/XML round-trip.

Skipped automatically if the local TTL files aren't on disk (e.g., a fresh
checkout that hasn't received the author's local ontology files).
"""

from __future__ import annotations

from pathlib import Path

import pytest

ONTOLOGIES_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "assets" / "ontologies"
)

HEALTHTASKS_FILE = ONTOLOGIES_DIR / "HealthTasks_2026.05.19.ttl"
HUMANACTIVITIES_FILE = ONTOLOGIES_DIR / "HumanActivities_2026.05.03.ttl"

HEALTH_NS = "https://w3id.org/calendar-bench/health/"
HUMAN_NS = "https://w3id.org/calendar-bench/human-activities/"


# ---------------------------------------------------------------------------
# HealthTasks
# ---------------------------------------------------------------------------


@pytest.fixture()
def require_healthtasks_imported(neo4j_session):
    if not HEALTHTASKS_FILE.exists():
        pytest.skip(f"Local file {HEALTHTASKS_FILE.name} not present on disk")
    rec = neo4j_session.run(
        "MATCH (n:Resource) WHERE n.uri STARTS WITH $ns RETURN count(n) AS c",
        ns=HEALTH_NS,
    ).single()
    if rec["c"] == 0:
        pytest.skip("HealthTasks ontology not imported into Neo4j yet")


def test_healthtasks_root_class(neo4j_session, require_healthtasks_imported):
    """The root :HealthTask class with its label and human-readable comment."""
    rec = neo4j_session.run(
        "MATCH (n:Resource {uri: $uri}) "
        "RETURN n.label AS label, n.comment AS comment",
        uri=HEALTH_NS + "HealthTask",
    ).single()
    assert rec is not None, "Root HealthTask URI missing"
    assert "Health Task" in (rec["label"] or [""])[0]
    assert "schedulable health behaviour" in (rec["comment"] or [""])[0]


def test_healthtasks_specific_difficulty_level(
    neo4j_session, require_healthtasks_imported
):
    """Pick one specific difficulty subclass and verify label + parent."""
    target = HEALTH_NS + "MentalWellbeingCalmBreathingAndMindfulnessLevel1"
    rec = neo4j_session.run(
        """
        MATCH (n:Resource {uri: $uri})
        OPTIONAL MATCH (n)-[:SUBCLASSOF]->(parent:Resource)
        RETURN n.label AS label, collect(parent.uri) AS parents
        """,
        uri=target,
    ).single()
    assert rec is not None, f"{target} not found"
    # HealthTasks 05.19 uses Title Case ("And"); earlier versions used "and".
    # Compare case-insensitively so the test rides through either capitalisation.
    expected = "mental wellbeing - calm breathing and mindfulness - level 1"
    assert any(expected in lbl.lower() for lbl in (rec["label"] or [])), rec["label"]
    assert (
        HEALTH_NS + "MentalWellbeingCalmBreathingAndMindfulnessTask" in rec["parents"]
    )


def test_healthtasks_three_branches_present(
    neo4j_session, require_healthtasks_imported
):
    """Verify the three top-level intervention branches are all imported."""
    branches = [
        HEALTH_NS + "NutritionTask",
        HEALTH_NS + "MentalWellbeingTask",
        HEALTH_NS + "PhysicalActivityTask",
    ]
    rec = neo4j_session.run(
        "MATCH (n:Resource) WHERE n.uri IN $uris RETURN count(n) AS c",
        uris=branches,
    ).single()
    assert rec["c"] == 3, f"Expected 3 top-level branches, found {rec['c']}"


# ---------------------------------------------------------------------------
# HumanActivities
# ---------------------------------------------------------------------------


@pytest.fixture()
def require_humanactivities_imported(neo4j_session):
    if not HUMANACTIVITIES_FILE.exists():
        pytest.skip(f"Local file {HUMANACTIVITIES_FILE.name} not present on disk")
    rec = neo4j_session.run(
        "MATCH (n:Resource) WHERE n.uri STARTS WITH $ns RETURN count(n) AS c",
        ns=HUMAN_NS,
    ).single()
    if rec["c"] == 0:
        pytest.skip("HumanActivities ontology not imported into Neo4j yet")


def test_humanactivities_root_class(neo4j_session, require_humanactivities_imported):
    rec = neo4j_session.run(
        "MATCH (n:Resource {uri: $uri}) RETURN n.label AS label",
        uri=HUMAN_NS + "HumanActivity",
    ).single()
    assert rec is not None
    assert any("Human Activity" in lbl for lbl in (rec["label"] or []))


def test_humanactivities_bicycling_general_instance(
    neo4j_session, require_humanactivities_imported
):
    """The 'bicycling-general' individual must round-trip with code=1014, met=7.0."""
    target = HUMAN_NS + "activity/bicycling-general"
    rec = neo4j_session.run(
        "MATCH (n:Resource {uri: $uri}) "
        "RETURN n.label AS label, n.activityCode AS code, n.metValue AS met",
        uri=target,
    ).single()
    assert rec is not None, f"{target} not imported"
    assert any("Bicycling, general" in lbl for lbl in (rec["label"] or [])), rec[
        "label"
    ]
    # n10s with handleMultival=ARRAY wraps datatype properties as arrays.
    code = rec["code"][0] if isinstance(rec["code"], list) else rec["code"]
    met = rec["met"][0] if isinstance(rec["met"], list) else rec["met"]
    assert int(code) == 1014, f"activityCode mismatch: {rec['code']!r}"
    assert float(met) == pytest.approx(7.0), f"metValue mismatch: {rec['met']!r}"


def test_humanactivities_seven_top_level_categories(
    neo4j_session, require_humanactivities_imported
):
    """Compendium has seven event categories; verify each has its class."""
    expected = [
        HUMAN_NS + "SportsExerciseWorkout",
        HUMAN_NS + "Work",
        HUMAN_NS + "SocialFamilyFriends",
        HUMAN_NS + "Education",
        HUMAN_NS + "EverydayTasks",
        HUMAN_NS + "Health",
        HUMAN_NS + "LeisureEntertainmentTravel",
    ]
    rec = neo4j_session.run(
        "MATCH (n:Resource) WHERE n.uri IN $uris RETURN count(n) AS c",
        uris=expected,
    ).single()
    assert rec["c"] == len(
        expected
    ), f"Expected {len(expected)} top-level categories, found {rec['c']}"


# ---------------------------------- local embeddings----------------------------------


def test_local_ontologies_have_embedded_concepts(
    neo4j_session, require_healthtasks_imported, require_humanactivities_imported
):
    """At least one node from each local ontology must carry an embedding."""
    rec = neo4j_session.run(
        """
        MATCH (n:EmbeddedConcept)
        WHERE n.uri STARTS WITH $h_ns OR n.uri STARTS WITH $a_ns
        WITH
          sum(CASE WHEN n.uri STARTS WITH $h_ns THEN 1 ELSE 0 END) AS health,
          sum(CASE WHEN n.uri STARTS WITH $a_ns THEN 1 ELSE 0 END) AS human
        RETURN health, human
        """,
        h_ns=HEALTH_NS,
        a_ns=HUMAN_NS,
    ).single()
    assert (
        rec["health"] > 0
    ), "No HealthTasks nodes are embedded yet: re-run build_embeddings"
    assert (
        rec["human"] > 0
    ), "No HumanActivities nodes are embedded yet: re-run build_embeddings"


# -------------------------------------------------------------------------------------
# --- Retrieval: verify the GraphRAG brings these nodes for a domain-specific query. --
# -------------------------------------------------------------------------------------


def test_humanactivities_bicycling_general_retrievable(
    retriever, require_humanactivities_imported
):
    """Vector retrieval must surface the bicycling-general URI for a Compendium query."""
    target = HUMAN_NS + "activity/bicycling-general"
    result = retriever.search(
        query_text=(
            "Bicycling general activity from the Compendium of Physical Activities "
            "with its MET metabolic equivalent value and activity code."
        ),
        top_k=20,
    )
    contexts = [item.content for item in result.items]
    assert any(
        target in c for c in contexts
    ), f"Did not surface {target}. Top context heads:\n" + "\n---\n".join(
        c[:200] for c in contexts[:3]
    )


def test_healthtasks_calm_breathing_retrievable(
    retriever, require_healthtasks_imported
):
    """Vector retrieval must surface a calm-breathing-mindfulness Level node."""
    result = retriever.search(
        query_text=(
            "Calendar-bench mental wellbeing calm breathing and mindfulness "
            "graded difficulty levels Level1 Level2."
        ),
        top_k=20,
    )
    contexts = [item.content for item in result.items]
    assert any("MentalWellbeingCalmBreathingAndMindfulness" in c for c in contexts), (
        "Did not surface any CalmBreathingAndMindfulness node. Top contexts:\n"
        + "\n---\n".join(c[:200] for c in contexts[:3])
    )
