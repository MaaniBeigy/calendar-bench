"""Verify each ontology was imported into Neo4j by n10s."""

from __future__ import annotations

import pytest


def test_ontology_metadata_node_exists(neo4j_session, ontology_spec):
    """Each ontology should have an :Ontology root node tagged by import_ontologies.py."""
    rec = neo4j_session.run(
        "MATCH (o:Ontology {key: $key}) RETURN o.title AS title, o.file AS file",
        key=ontology_spec.key,
    ).single()
    assert rec is not None, f"No :Ontology node for {ontology_spec.key}"
    assert rec["title"] == ontology_spec.title
    assert rec["file"] == ontology_spec.file


def test_ontology_resources_imported(
    neo4j_session, ontology_spec, ontology_uri_samples
):
    """Sample subject URIs from the ontology file should exist as :Resource nodes."""
    samples = ontology_uri_samples.get(ontology_spec.key, [])
    if not samples:
        pytest.skip(f"Could not parse sample URIs from {ontology_spec.file}")
    rec = neo4j_session.run(
        "MATCH (n:Resource) WHERE n.uri IN $uris RETURN count(n) AS c",
        uris=samples,
    ).single()
    found = rec["c"]
    assert found > 0, (
        f"None of {len(samples)} sample URIs from {ontology_spec.key} found in Neo4j. "
        f"Sample URI: {samples[0]}"
    )


def test_total_resource_count_nonzero(neo4j_session):
    rec = neo4j_session.run("MATCH (n:Resource) RETURN count(n) AS c").single()
    assert rec["c"] > 1000, f"Only {rec['c']} :Resource nodes : import looks broken"
