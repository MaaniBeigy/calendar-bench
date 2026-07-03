"""GraphRAG pipeline."""

from __future__ import annotations

from neo4j import Driver
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.llm import LLMInterface

from .embeddings import make_embedder
from .llm import make_llm
from .neo4j_client import make_driver
from .retrieval_filters import AttributeFilter
from .retriever import make_retriever

CITATION_TEMPLATE = """You are answering a question using a biomedical / behavior-change ontology graph.

Use ONLY the information in the context below. Each context block represents one graph node and ends with a line `uri: <URI>`. that URI is the citation handle for that block.

Rules:
- For every factual claim, append the supporting node URI(s) in square brackets, e.g. "Walking is an aerobic activity [http://...EmpowerBP#T8]."
- If multiple nodes support a claim, list each URI in its own brackets: [URI1] [URI2].
- If the context does not contain enough information to answer, say so explicitly. Do not fall back to general knowledge.
- Never fabricate URIs.

Context:
{context}

Question: {query_text}

Answer (with [URI] citations after each claim):"""


def build_graphrag(
    driver: Driver | None = None,
    *,
    uri_prefixes: list[str] | None = None,
    allowed_branches: list[str] | None = None,
    allowed_levels: list[str] | None = None,
    instance_only: bool = False,
    attribute_filters: list[AttributeFilter] | None = None,
    exclude_prefixes: list[str] | None = None,
    llm: LLMInterface | None = None,
) -> GraphRAG:
    """Build the project's GraphRAG pipeline.

    When *uri_prefixes* is supplied with 2+ entries, the underlying
    retriever runs one independent vector search per prefix and merges
    the results; guarantees per-ontology representation regardless of
    relative corpus size (see
    :class:`src.graphrag.retriever.MultiPrefixVectorCypherRetriever`).
    With 0–1 prefixes, a single :class:`VectorCypherRetriever` is used.

    The optional `allowed_branches` / `allowed_levels` /
    `instance_only` filters bake the scenario's
    `filters.domains` / `filters.difficulty` and the validator's
    instance-only check directly into the retrieval Cypher, so the LLM
    only ever sees URIs that would also pass
    :func:`validate_task_uri`.  Without these the retriever surfaces
    cross-domain candidates the LLM cites and the validator then
    rejects, exhausting fetch retries with zero accepted tasks (an earlier
    empty-context regression).
    """
    driver = driver or make_driver()
    retriever = make_retriever(
        driver,
        make_embedder(),
        uri_prefixes=uri_prefixes,
        allowed_branches=allowed_branches,
        allowed_levels=allowed_levels,
        instance_only=instance_only,
        attribute_filters=attribute_filters,
        exclude_prefixes=exclude_prefixes,
    )
    # Caller-supplied `llm` lets a scenario YAML target a specific model
    # (`task_generator_model.model: gpt-4.1-mini`) without mutating env
    # vars. When omitted we fall back to the env-default settings.
    llm = llm or make_llm()
    template = RagTemplate(
        template=CITATION_TEMPLATE,
        expected_inputs=["context", "query_text"],
    )
    return GraphRAG(retriever=retriever, llm=llm, prompt_template=template)
