"""Shared fixtures for ontology / embedding / RAG verification.

We parse each ontology file once with rdflib to extract a sample of subject URIs.
Those URIs are the ground truth as they should have been imported into Neo4j and
(for text-bearing ones) embedded.

All tests are integration tests, they require a running Neo4j with ontologies already
imported and embeddings built.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from rdflib import Graph

from src.graphrag.config import Neo4jSettings
from src.graphrag.embeddings import make_embedder
from src.graphrag.neo4j_client import make_driver, session_scope
from src.graphrag.ontology_metadata import ONTOLOGIES, OntologySpec
from src.graphrag.retriever import make_retriever

log = logging.getLogger(__name__)

ONTOLOGIES_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "assets" / "ontologies"
)
SAMPLE_SIZE = 25


@pytest.fixture(scope="session")
def neo4j_settings() -> Neo4jSettings:
    return Neo4jSettings.from_env()


@pytest.fixture(scope="session")
def neo4j_driver(neo4j_settings):
    driver = make_driver(neo4j_settings)
    driver.verify_connectivity()
    yield driver
    driver.close()


@pytest.fixture()
def neo4j_session(neo4j_driver, neo4j_settings) -> Iterator:
    with session_scope(neo4j_driver, neo4j_settings.database) as session:
        yield session


@pytest.fixture(scope="session")
def retriever(neo4j_driver):
    return make_retriever(neo4j_driver, make_embedder())


def _parse_graph(path: Path) -> Graph | None:
    """Parse an RDF file, trying common formats. Return None on hard failure."""
    if not path.exists():
        return None
    g = Graph()
    for fmt in ("xml", "turtle", "n3", "json-ld", "nt"):
        try:
            g.parse(str(path), format=fmt)
            return g
        except Exception:
            continue
    return None


_SAMPLE_URI_SKIP: set[str] = {"BCIO", "HSPO", "OCHV", "FOODON"}

# rdflib parse cost grows non-linearly with axiom count; the 205 MB
# HSPO.xrdf will OOM-kill a 4 GB worker on first `ontology_uri_samples`
# build (session-scoped fixture eagerly iterates every `OntologySpec`
# in :data:`ONTOLOGIES`).  20 MB picks every known offender (HSPO, OCHV,
# FoodOn) without touching the smaller ontologies whose URI samples
# remain useful for `test_ontology_has_embedded_concepts`.  The
# corresponding test self-skips when `samples == []`.
_SAMPLE_URI_SIZE_LIMIT_BYTES = 20 * 1024 * 1024


def _sample_uris(spec: OntologySpec) -> list[str]:
    if spec.key in _SAMPLE_URI_SKIP:
        return []
    path = ONTOLOGIES_DIR / spec.file
    try:
        if path.stat().st_size > _SAMPLE_URI_SIZE_LIMIT_BYTES:
            return []
    except OSError:
        return []
    g = _parse_graph(path)
    if g is None:
        return []
    seen: list[str] = []
    for s in g.subjects():
        u = str(s)
        if u.startswith("http") and u not in seen:
            seen.append(u)
            if len(seen) >= SAMPLE_SIZE:
                break
    return seen


@pytest.fixture(scope="session")
def ontology_uri_samples() -> dict[str, list[str]]:
    return {spec.key: _sample_uris(spec) for spec in ONTOLOGIES}


def _common_prefix(uris: list[str]) -> str:
    """Longest common prefix, truncated to the last namespace separator.

    URIs share a namespace ending in '#' or '/'; cutting there prevents
    over-narrow prefixes like '...BCTT#00' that exclude valid '...BCTT#01x' URIs.
    """
    if not uris:
        return ""
    common = uris[0]
    for u in uris[1:]:
        while common and not u.startswith(common):
            common = common[:-1]
        if not common:
            return ""
    for sep in ("#", "/"):
        idx = common.rfind(sep)
        if idx > 0:
            return common[: idx + 1]
    return common


@pytest.fixture(scope="session")
def ontology_uri_prefixes(ontology_uri_samples) -> dict[str, str]:
    return {key: _common_prefix(uris) for key, uris in ontology_uri_samples.items()}


def pytest_generate_tests(metafunc):
    if "ontology_spec" in metafunc.fixturenames:
        metafunc.parametrize(
            "ontology_spec",
            ONTOLOGIES,
            ids=[s.key for s in ONTOLOGIES],
        )
