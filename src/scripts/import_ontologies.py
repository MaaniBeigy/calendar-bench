"""Import every ontology from src/assets/ontologies into Neo4j via neosemantics."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from neo4j.exceptions import ClientError

from src.graphrag.config import Neo4jSettings
from src.graphrag.neo4j_client import (
    ensure_n10s_initialized,
    make_driver,
    session_scope,
)
from src.graphrag.ontology_metadata import ONTOLOGIES, OntologySpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("import_ontologies")


# Inside the neo4j container the host directory src/assets/ontologies is mounted at /import/ontologies.
NEO4J_IMPORT_DIR = "file:///import/ontologies"


def import_one(session, spec: OntologySpec) -> None:
    file_uri = f"{NEO4J_IMPORT_DIR}/{spec.file}"
    log.info("Importing %s (%s) from %s", spec.key, spec.title, file_uri)

    try:
        result = session.run(
            "CALL n10s.rdf.import.fetch($url, $format) "
            "YIELD terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo "
            "RETURN terminationStatus, triplesLoaded, triplesParsed, extraInfo",
            url=file_uri,
            format=spec.rdf_format,
        ).single()
    except ClientError as exc:
        log.error("n10s import failed for %s: %s", spec.key, exc)
        return

    if result is None:
        log.error("No result returned for %s", spec.key)
        return

    log.info(
        "  %s -> status=%s parsed=%s loaded=%s",
        spec.key,
        result["terminationStatus"],
        result["triplesParsed"],
        result["triplesLoaded"],
    )
    if result["terminationStatus"] != "OK":
        log.warning("  extraInfo: %s", result["extraInfo"])

    # Tag every node loaded from this file with the ontology key + metadata.
    session.run(
        """
        MERGE (o:Ontology {key: $key})
        SET o.title = $title,
            o.file = $file,
            o.format = $format,
            o.description = $description,
            o.sourceUri = $sourceUri
        """,
        key=spec.key,
        title=spec.title,
        file=spec.file,
        format=spec.rdf_format,
        description=spec.description,
        sourceUri=file_uri,
    )


def main() -> int:
    settings = Neo4jSettings.from_env()
    log.info(
        "Connecting to Neo4j at %s as %s (db=%s)",
        settings.uri,
        settings.username,
        settings.database,
    )

    driver = make_driver(settings)
    try:
        driver.verify_connectivity()
        with session_scope(driver, settings.database) as session:
            ensure_n10s_initialized(session)
            for spec in ONTOLOGIES:
                # The app container has src/assets/ontologies bind-mounted at
                # /app/ontologies; neo4j sees the same files at /import/ontologies.
                app_path = Path("/app/ontologies") / spec.file
                if spec.local and not app_path.exists():
                    log.warning(
                        "Skipping %s: local ontology missing at %s",
                        spec.key,
                        app_path,
                    )
                    continue
                hint = (
                    ""
                    if app_path.exists()
                    else " (file path unverified from app container, n10s reads from neo4j container)"
                )
                log.info("Queueing %s%s", spec.file, hint)
                import_one(session, spec)
    finally:
        driver.close()

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
