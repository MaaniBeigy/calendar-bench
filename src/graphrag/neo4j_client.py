"""Neo4j client utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import ClientError

from .config import Neo4jSettings

log = logging.getLogger(__name__)


def make_driver(settings: Neo4jSettings | None = None) -> Driver:
    settings = settings or Neo4jSettings.from_env()
    # `UNRECOGNIZED` is silenced because retrieval Cypher unions
    # multiple ontology-specific relationship types (e.g. SUBCLASSOF,
    # BROADER) and Neo4j fires a notification for every relationship
    # type that does not exist in the loaded graph. The query result is
    # still correct; the absent type simply matches nothing.
    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )


@contextmanager
def session_scope(driver: Driver, database: str | None = None) -> Iterator[Session]:
    db = database or Neo4jSettings.from_env().database
    with driver.session(database=db) as session:
        yield session


def ensure_n10s_initialized(session: Session) -> None:
    """Create the URI uniqueness constraint and initialize the n10s graph config.

    Safe to call repeatedly: both operations short-circuit if already present.
    """
    session.run(
        "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS "
        "FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
    ).consume()

    try:
        session.run(
            "CALL n10s.graphconfig.init({"
            "handleVocabUris: 'IGNORE',"
            "handleMultival: 'ARRAY',"
            "keepLangTag: false,"
            "applyNeo4jNaming: true"
            "})"
        ).consume()
        log.info("n10s graph config initialized.")
    except ClientError as exc:
        msg = str(exc).lower()
        # init() refuses on a non-empty graph or when a config already exists.
        if "non-empty" in msg or "already" in msg or "existing config" in msg:
            log.info("n10s graph config already present; continuing.")
            return
        raise
