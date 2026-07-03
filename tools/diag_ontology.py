"""Diagnostic — verify the HealthTasks instance node landed in Neo4j with
``dcterms:title`` and ``dcterms:description`` exposed under the property
names the bridge expects.

Usage (inside the docker app container):

    docker compose run --rm app python tools/diag_ontology.py

The script answers four questions:

    1. Does the canonical instance URI have a Resource node?
    2. What property names does the node actually carry?
    3. Do ``n.title`` / ``n.description`` resolve to the authored
       emoji-bearing values?
    4. Is the node embedded (``EmbeddedConcept`` label + non-null
       ``embedding`` vector)?

Run this BEFORE re-running the experiment if you suspect the import
or the embedding pipeline didn't land the dcterms data.  All four
answers are printed so the failure mode is obvious at a glance.
"""

from __future__ import annotations

import sys

from src.graphrag.config import Neo4jSettings
from src.graphrag.neo4j_client import make_driver, session_scope

URI = "https://w3id.org/calendar-bench/health/task/drink-a-glass-of-water"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    settings = Neo4jSettings.from_env()
    driver = make_driver(settings)
    try:
        driver.verify_connectivity()
        with session_scope(driver, settings.database) as session:
            print(f"=== Looking up Resource {{uri: {URI!r}}} ===\n")

            # 1. Does the node exist?
            row = session.run(
                "MATCH (n:Resource {uri: $uri}) RETURN count(n) AS n",
                uri=URI,
            ).single()
            node_count = row["n"] if row else 0
            print(f"Q1. Node count: {node_count}")
            if node_count == 0:
                print(
                    "    ❌ Node not in graph. Run "
                    "`docker compose run --rm app python -m src.scripts.import_ontologies`."
                )
                return 1
            print("    ✓ Node present.\n")

            # 2. What property keys does it carry?
            row = session.run(
                "MATCH (n:Resource {uri: $uri}) RETURN keys(n) AS keys",
                uri=URI,
            ).single()
            keys = sorted(row["keys"]) if row else []
            print(f"Q2. Property keys on the node ({len(keys)}):")
            for k in keys:
                print(f"      {k}")
            print()

            # 3. Do n.title / n.description resolve to the authored values?
            row = session.run(
                """
                MATCH (n:Resource {uri: $uri})
                RETURN
                  head(coalesce(n.displayName, n.title, n.label)) AS display_name,
                  head(coalesce(n.description, n.comment))        AS description,
                  head(n.isConcurrent)                            AS is_concurrent,
                  head(n.isDividable)                             AS is_dividable,
                  head(n.estimatedDurationMinutes)                AS duration_minutes
                """,
                uri=URI,
            ).single()
            print("Q3. Bridge-equivalent coalesce result:")
            print(f"      display_name      = {row['display_name']!r}")
            print(f"      description       = {row['description']!r}")
            print(f"      is_concurrent     = {row['is_concurrent']!r}")
            print(f"      is_dividable      = {row['is_dividable']!r}")
            print(f"      duration_minutes  = {row['duration_minutes']!r}")
            print()
            if not row["display_name"]:
                print(
                    "    ❌ display_name is empty — n.title / n.label / n.displayName "
                    "are all null on this node.\n"
                    "       The dcterms:title triple did not land.  Re-run "
                    "`docker compose run --rm app python -m src.scripts.import_ontologies`."
                )
                return 2

            # 4. Is the node embedded?
            row = session.run(
                """
                MATCH (n:Resource {uri: $uri})
                RETURN
                  ('EmbeddedConcept' IN labels(n))  AS is_embedded,
                  size(coalesce(n.embedding, []))   AS embedding_dim
                """,
                uri=URI,
            ).single()
            print(
                f"Q4. Embedded? {row['is_embedded']}  "
                f"(embedding vector dim = {row['embedding_dim']})"
            )
            if not row["is_embedded"]:
                print(
                    "    ❌ Node is not in the vector index.  Run "
                    "`docker compose run --rm app python -m src.scripts.build_embeddings`."
                )
                return 3
            print("    ✓ Node is in the vector index.\n")

        print("All four checks passed.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
