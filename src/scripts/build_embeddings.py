"""Compute and store embeddings for ontology concept nodes.

It runs after `import_ontologies.py` and is idempotent, so that already-embedded nodes
will be skipped.

Strategy:
  1. Mark every text-bearing `:Resource` with a transient `:UnembeddedConcept` label.
  2. Loop: pull a batch of `:UnembeddedConcept` nodes, build a text snippet from
     prefLabel | label | altLabel | title | comment | description | definition,
     embed via the provider selected by `EMBEDDING_PROVIDER`
     (`openai` or `local` sentence-transformers), write back, swap
     labels to `:EmbeddedConcept`.
  3. Create the vector index on (:EmbeddedConcept).embedding.

`title` and `description` come from `dcterms:title` and
`dcterms:description` and are the only text the local HealthTasks
instance nodes (`hb-tk:*`) carry; without these in the predicate
list those instances would never reach the vector index.

The label-queue pattern keeps each fetch as a fast label scan instead of re-scanning
all resource nodes per batch.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Callable

from neo4j.exceptions import ClientError
from openai import OpenAI

from src.graphrag.config import Neo4jSettings
from src.graphrag.embeddings import EMBED_DIMENSIONS, EMBED_MODEL, resolve_device
from src.graphrag.neo4j_client import make_driver, session_scope
from src.graphrag.retriever import EMBEDDING_PROPERTY, NODE_LABEL, VECTOR_INDEX_NAME

EmbedBatch = Callable[[list[str]], list[list[float]]]
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build_embeddings")

BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
PENDING_LABEL = "UnembeddedConcept"
MAX_TEXT_CHARS = 4000  # text-embedding-3-small handles 8k tokens; 4000 chars is safe


PRE_PASS = f"""
MATCH (n:Resource)
WHERE n.{EMBEDDING_PROPERTY} IS NULL
  AND NOT n:{NODE_LABEL}
  AND NOT n.uri STARTS WITH 'bnode://'
  AND (size(coalesce(n.label, []))             > 0
    OR size(coalesce(n.prefLabel, []))         > 0
    OR size(coalesce(n.altLabel, []))          > 0
    OR size(coalesce(n.title, []))             > 0
    OR size(coalesce(n.comment, []))           > 0
    OR size(coalesce(n.description, []))       > 0
    OR size(coalesce(n.definition, []))        > 0
    OR size(coalesce(n.iAO_0000115, []))       > 0
    OR size(coalesce(n.hasExactSynonym, []))   > 0
    OR size(coalesce(n.hasRelatedSynonym, [])) > 0
    OR size(coalesce(n.hasSynonym, []))        > 0)
SET n:{PENDING_LABEL}
RETURN count(n) AS marked
"""

FETCH = f"""
MATCH (n:{PENDING_LABEL})
WITH n LIMIT $limit
WITH n,
     [v IN coalesce(n.prefLabel, [])         | toString(v)] +
     [v IN coalesce(n.label, [])             | toString(v)] +
     [v IN coalesce(n.altLabel, [])          | toString(v)] +
     [v IN coalesce(n.title, [])             | toString(v)] +
     [v IN coalesce(n.hasExactSynonym, [])   | toString(v)] +
     [v IN coalesce(n.hasRelatedSynonym, []) | toString(v)] +
     [v IN coalesce(n.hasSynonym, [])        | toString(v)] +
     [v IN coalesce(n.comment, [])           | toString(v)] +
     [v IN coalesce(n.description, [])       | toString(v)] +
     [v IN coalesce(n.definition, [])        | toString(v)] +
     [v IN coalesce(n.iAO_0000115, [])       | toString(v)] AS parts
WITH n, apoc.text.join(parts, ' | ') AS text
WHERE text <> ''
RETURN elementId(n) AS id, substring(text, 0, $max_chars) AS text
"""

STORE = f"""
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.id
SET n.{EMBEDDING_PROPERTY} = row.embedding
SET n:{NODE_LABEL}
REMOVE n:{PENDING_LABEL}
"""

CHECK_INDEX = "SHOW INDEXES YIELD name WHERE name = $name RETURN name"

CREATE_INDEX_PROC = (
    "CALL db.index.vector.createNodeIndex($name, $label, $prop, $dim, $sim)"
)


def _make_openai_batch(model: str) -> EmbedBatch:
    """Return an embed-batch callable backed by the OpenAI embeddings endpoint."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    def _embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]

    return _embed


def _make_local_batch(model_name: str, batch_size: int) -> EmbedBatch:
    """Return an embed-batch callable backed by a local sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=local requires sentence-transformers; "
            "install via the GPU image (docker-compose.gpu.yml)."
        ) from exc

    device = resolve_device()
    model = SentenceTransformer(model_name, device=device)

    def _embed(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [row.tolist() for row in vectors]

    return _embed


def make_embed_batch(provider: str | None = None) -> tuple[EmbedBatch, str, int]:
    """Resolve `EMBEDDING_PROVIDER` to a batch embedder, its model name, and its dim."""
    name = (provider or os.getenv("EMBEDDING_PROVIDER", "openai")).strip().lower()
    if name == "openai":
        return _make_openai_batch(EMBED_MODEL), EMBED_MODEL, EMBED_DIMENSIONS
    if name == "local":
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL)
        return _make_local_batch(model_name, BATCH_SIZE), model_name, EMBED_DIMENSIONS
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {name!r}")


def ensure_vector_index(session, dim: int) -> None:
    existing = session.run(CHECK_INDEX, name=VECTOR_INDEX_NAME).single()
    if existing:
        log.info("Vector index %s already exists", VECTOR_INDEX_NAME)
        return
    try:
        session.run(
            CREATE_INDEX_PROC,
            name=VECTOR_INDEX_NAME,
            label=NODE_LABEL,
            prop=EMBEDDING_PROPERTY,
            dim=dim,
            sim="cosine",
        ).consume()
        log.info("Created vector index %s (%d dims, cosine)", VECTOR_INDEX_NAME, dim)
    except ClientError as exc:
        if "EquivalentSchemaRule" in str(exc) or "already exists" in str(exc).lower():
            log.info("Vector index %s already exists (race)", VECTOR_INDEX_NAME)
            return
        raise


def main() -> int:
    """Bake embeddings into Neo4j; return 0 on success or 2 when config is missing."""
    try:
        embed_batch, model_name, dim = make_embed_batch()
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2

    settings = Neo4jSettings.from_env()
    log.info("Neo4j: %s db=%s", settings.uri, settings.database)
    log.info("Embedder: %s (%d dims)", model_name, dim)

    driver = make_driver(settings)
    total = 0
    start = time.time()

    try:
        driver.verify_connectivity()
        with session_scope(driver, settings.database) as session:
            marked = session.run(PRE_PASS).single()["marked"]
            log.info("Marked %d nodes as %s", marked, PENDING_LABEL)

            ensure_vector_index(session, dim)

            while True:
                rows = session.run(
                    FETCH, limit=BATCH_SIZE, max_chars=MAX_TEXT_CHARS
                ).data()
                if not rows:
                    break

                texts = [r["text"] for r in rows]
                embeddings = embed_batch(texts)
                payload = [
                    {"id": rows[i]["id"], "embedding": embeddings[i]}
                    for i in range(len(rows))
                ]
                session.run(STORE, rows=payload).consume()

                total += len(rows)
                if total % (BATCH_SIZE * 10) == 0 or len(rows) < BATCH_SIZE:
                    elapsed = time.time() - start
                    rate = total / elapsed if elapsed > 0 else 0
                    log.info("Embedded %d nodes (%.1f/s)", total, rate)
    finally:
        driver.close()

    log.info("Done. Embedded %d nodes in %.1fs.", total, time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
