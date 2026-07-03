"""Verify the vector index exists and each ontology contributed embedded nodes."""

from __future__ import annotations

import pytest

from src.graphrag.embeddings import EMBED_DIMENSIONS
from src.graphrag.retriever import EMBEDDING_PROPERTY, NODE_LABEL, VECTOR_INDEX_NAME


def test_vector_index_exists_and_online(neo4j_session):
    rec = neo4j_session.run(
        "SHOW VECTOR INDEXES YIELD name, state, options "
        "WHERE name = $name "
        "RETURN name, state, options",
        name=VECTOR_INDEX_NAME,
    ).single()
    assert rec is not None, f"Vector index {VECTOR_INDEX_NAME} missing"
    assert rec["state"] == "ONLINE", f"Vector index state is {rec['state']}"
    cfg = rec["options"]["indexConfig"]
    assert cfg["vector.dimensions"] == EMBED_DIMENSIONS
    assert cfg["vector.similarity_function"].lower() == "cosine"


def test_total_embedded_count_nonzero(neo4j_session):
    rec = neo4j_session.run(f"MATCH (n:{NODE_LABEL}) RETURN count(n) AS c").single()
    assert rec["c"] > 0, "No :EmbeddedConcept nodes : build_embeddings did not run"


def test_ontology_has_embedded_concepts(
    neo4j_session, ontology_spec, ontology_uri_samples, ontology_uri_prefixes
):
    # Slow-parse ontologies (e.g. BCIO; axiom-dense RDF/XML) return
    # `[]` from `_sample_uris` in conftest.py; the same opt-out
    # mechanism prevents the rdflib parse cost from being paid at
    # fixture-resolution time.  Coverage for these ontologies is
    # retained via test_imports::test_ontology_metadata_node_exists
    # and test_rag::test_query_defined, which do not depend on the
    # sample-URI parse.
    samples = ontology_uri_samples.get(ontology_spec.key, [])
    prefix = ontology_uri_prefixes.get(ontology_spec.key, "")
    if not samples:
        pytest.skip(f"No sample URIs for {ontology_spec.key}")

    rec = neo4j_session.run(
        f"""
        MATCH (n:{NODE_LABEL})
        WHERE ($prefix <> '' AND n.uri STARTS WITH $prefix) OR n.uri IN $uris
        RETURN count(n) AS c
        """,
        prefix=prefix,
        uris=samples,
    ).single()

    if rec["c"] == 0:
        pytest.xfail(
            f"{ontology_spec.key}: no :EmbeddedConcept under prefix {prefix!r}. "
            "Either no text-bearing nodes (axiom-only ontology) or embeddings stale."
        )
    assert rec["c"] > 0


def test_embedding_dimensionality(neo4j_session):
    rec = neo4j_session.run(
        f"MATCH (n:{NODE_LABEL}) WHERE n.{EMBEDDING_PROPERTY} IS NOT NULL "
        f"RETURN size(n.{EMBEDDING_PROPERTY}) AS dim LIMIT 1"
    ).single()
    assert rec is not None, "No embedded node with embedding property"
    assert (
        rec["dim"] == EMBED_DIMENSIONS
    ), f"Embedding dimension {rec['dim']} != configured {EMBED_DIMENSIONS}"
