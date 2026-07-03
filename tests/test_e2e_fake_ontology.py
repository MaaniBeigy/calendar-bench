"""End-to-end honesty test using a synthetic ontology fixture.

Imports `tests/fixtures/fake_ontology.xrdf` into Neo4j via n10s, embeds its
text-bearing nodes via OpenAI, then asks a question that's only answerable
from the fake namespace using a free OpenRouter model. Asserts the answer
cites a URI from `http://calendar-bench.test/fake#`.

This is the strongest grounding check we have: the fake fixture defines
fictional surgical / hospital concepts (`Quibble bypass surgery`, `Throckmorton
operating theatre`, `Vexil-7 clamp`, `Zorblax registered nurse`) that sit
completely outside the behaviour-change / nutrition / physical-activity domain
covered by every imported public ontology, so the LLM cannot fall back to its
training prior or to a sibling ontology.

Skipped when OPENROUTER_API_KEY is missing.
"""

from __future__ import annotations

import os

import pytest
from tenacity import RetryError

try:
    from neo4j_graphrag.exceptions import RateLimitError as Neo4jRateLimitError
except ImportError:  # older neo4j-graphrag versions
    Neo4jRateLimitError = Exception  # type: ignore[assignment, misc]

from src.graphrag.config import LLMSettings
from src.graphrag.embeddings import EMBED_DIMENSIONS, make_embedder
from src.graphrag.llm import make_llm
from src.graphrag.pipeline import CITATION_TEMPLATE
from src.graphrag.retriever import (
    EMBEDDING_PROPERTY,
    NODE_LABEL,
    VECTOR_INDEX_NAME,
    make_retriever,
)


def _looks_like_rate_limit(exc: BaseException) -> bool:
    """True if the exception (or its tenacity-wrapped cause) is a 429."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "ratelimit" in msg:
        return True
    if isinstance(exc, RetryError) and exc.last_attempt is not None:
        inner = exc.last_attempt.exception()
        if inner is not None:
            return _looks_like_rate_limit(inner)
    return False


FAKE_NS = "http://calendar-bench.test/fake#"
FAKE_FILE_URI = "file:///import/test_fixtures/fake_ontology.xrdf"
FAKE_ONTOLOGY_KEY = "FAKE_E2E"
# Cheap, reliable, instruction-following model. Override with OPENROUTER_TEST_MODEL.
# Cost per test run: ~$0.001. Free models are unusable due to rate limits.
TEST_MODEL_DEFAULT = "openai/gpt-4o-mini"


def _cleanup(session) -> None:
    session.run(
        "MATCH (n:Resource) WHERE n.uri STARTS WITH $ns DETACH DELETE n",
        ns=FAKE_NS,
    ).consume()
    session.run(
        "MATCH (o:Ontology {key: $k}) DETACH DELETE o",
        k=FAKE_ONTOLOGY_KEY,
    ).consume()


@pytest.fixture()
def openrouter_settings():
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set; skipping fake-ontology e2e test")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; embeddings still need it")
    base = LLMSettings.from_env()
    return LLMSettings(
        provider="openrouter",
        openai_api_key=base.openai_api_key,
        openai_model=base.openai_model,
        anthropic_api_key=base.anthropic_api_key,
        anthropic_model=base.anthropic_model,
        openrouter_api_key=base.openrouter_api_key,
        openrouter_base_url=base.openrouter_base_url,
        openrouter_model=os.getenv("OPENROUTER_TEST_MODEL", TEST_MODEL_DEFAULT),
    )


def test_e2e_fake_ontology_grounds_answer(
    neo4j_driver, neo4j_session, openrouter_settings
):
    """Ask about the made-up Zorblax routine; verify the answer references a fake URI."""
    _cleanup(neo4j_session)

    # 1. Import the fake ontology via n10s.
    fetch_record = neo4j_session.run(
        "CALL n10s.rdf.import.fetch($url, 'RDF/XML') "
        "YIELD terminationStatus, triplesLoaded "
        "RETURN terminationStatus, triplesLoaded",
        url=FAKE_FILE_URI,
    ).single()
    assert fetch_record is not None, "n10s did not return a result"
    assert fetch_record["terminationStatus"] == "OK", fetch_record
    assert fetch_record["triplesLoaded"] > 0

    neo4j_session.run(
        "MERGE (o:Ontology {key: $k}) SET o.title = $t, o.sourceUri = $u",
        k=FAKE_ONTOLOGY_KEY,
        t="Calendar-Bench Fake Ontology",
        u=FAKE_FILE_URI,
    ).consume()

    # 2. Embed every text-bearing fake node with the configured embedder.
    rows = neo4j_session.run(
        """
        MATCH (n:Resource) WHERE n.uri STARTS WITH $ns
          AND (size(coalesce(n.label, [])) > 0 OR size(coalesce(n.comment, [])) > 0)
        WITH n,
             [v IN coalesce(n.label, [])   | toString(v)] +
             [v IN coalesce(n.comment, []) | toString(v)] AS parts
        WITH n, apoc.text.join(parts, ' | ') AS text
        WHERE text <> ''
        RETURN elementId(n) AS id, text
        """,
        ns=FAKE_NS,
    ).data()
    assert rows, "No text-bearing nodes found in the imported fake ontology"

    # Use the same embedder the retriever will query with, so the fake
    # nodes' vectors live in the same space as the production index.
    # Picking different embedders here (e.g. hardcoded OpenAI when the
    # retriever uses local Qwen3) produces a silent dimension mismatch and
    # the retrieval never surfaces the fake nodes.
    fake_embedder = make_embedder()
    fake_vectors = [fake_embedder.embed_query(r["text"]) for r in rows]
    payload = [
        {"id": rows[i]["id"], "embedding": fake_vectors[i]} for i in range(len(rows))
    ]
    neo4j_session.run(
        f"""
        UNWIND $rows AS row
        MATCH (n) WHERE elementId(n) = row.id
        SET n.{EMBEDDING_PROPERTY} = row.embedding
        SET n:{NODE_LABEL}
        """,
        rows=payload,
    ).consume()

    # 3. Ensure the vector index exists (idempotent).
    existing = neo4j_session.run(
        "SHOW INDEXES YIELD name WHERE name = $n RETURN name",
        n=VECTOR_INDEX_NAME,
    ).single()
    if not existing:
        neo4j_session.run(
            "CALL db.index.vector.createNodeIndex($name, $label, $prop, $dim, $sim)",
            name=VECTOR_INDEX_NAME,
            label=NODE_LABEL,
            prop=EMBEDDING_PROPERTY,
            dim=EMBED_DIMENSIONS,
            sim="cosine",
        ).consume()

    # 4. Build a GraphRAG using OpenRouter for the answer LLM.
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.generation.prompts import RagTemplate

    retriever = make_retriever(neo4j_driver, make_embedder())
    llm = make_llm(openrouter_settings)
    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
        prompt_template=RagTemplate(
            template=CITATION_TEMPLATE,
            expected_inputs=["context", "query_text"],
        ),
    )

    try:
        response = rag.search(
            query_text=(
                "What is the Quibble bypass surgery and what surgical instrument "
                "does it require?"
            ),
            # top_k=20 keeps the assertion stable across OpenAI vs local
            # Qwen embeddings: the fake-ontology entries land in the top-20
            # under both backends, but not always in the top-5 under Qwen3
            # when OCHV `*bypass` concepts crowd the first positions.
            retriever_config={"top_k": 20},
            return_context=True,
        )
    except (RetryError, Neo4jRateLimitError) as exc:
        if _looks_like_rate_limit(exc):
            _cleanup(neo4j_session)
            pytest.skip(
                "OpenRouter test model rate-limited (429). Try again later, or "
                "set OPENROUTER_TEST_MODEL to a different cheap model. "
                f"Detail: {exc}"
            )
        raise

    # 5a. Retrieval must surface the fake namespace.
    contexts = [item.content for item in response.retriever_result.items]
    assert any(
        FAKE_NS in c for c in contexts
    ), "Retrieval did not surface any fake-ontology URI. Contexts:\n" + "\n---\n".join(
        c[:300] for c in contexts[:3]
    )

    # 5b. The answer must cite a URI from the fake namespace: proves grounding.
    assert FAKE_NS in response.answer, (
        "Answer did not cite the fake namespace, so it isn't grounded in the "
        f"fake ontology. Answer: {response.answer}"
    )

    # Cleanup so subsequent runs start clean.
    _cleanup(neo4j_session)
