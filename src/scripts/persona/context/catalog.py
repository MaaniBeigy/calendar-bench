"""Load the curated context IRI catalog from Neo4j."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One curated IRI with its category and metadata."""

    category: str
    name: str
    iri: str
    label: str
    source: str
    definition: str | None
    polarity: str | None
    instrument: str | None
    theory_mappings: dict | None


class ContextIriCatalog:
    """In-memory view of the IRI catalog with fast IRI lookup."""

    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        self._by_iri: dict[str, CatalogEntry] = {e.iri: e for e in entries}
        self._by_category: dict[str, list[CatalogEntry]] = {}
        for entry in self._by_iri.values():
            self._by_category.setdefault(entry.category, []).append(entry)

    def has(self, iri: str) -> bool:
        return iri in self._by_iri

    def get(self, iri: str) -> CatalogEntry | None:
        return self._by_iri.get(iri)

    def categories(self) -> list[str]:
        return sorted(self._by_category)

    def members(self, category: str) -> list[CatalogEntry]:
        return list(self._by_category.get(category, []))

    def __len__(self) -> int:
        return len(self._by_iri)


def load_catalog(driver: Any) -> ContextIriCatalog:
    """Build the catalog by querying the Neo4j Context ontology."""
    if driver is None:
        raise ValueError("load_catalog requires a Neo4j driver")
    from src.graphrag.config import Neo4jSettings
    from src.graphrag.neo4j_client import session_scope
    from src.scripts.scenarios.task_generation.ontology_bridge import (
        fetch_all_context_entries,
    )

    settings = Neo4jSettings.from_env()
    entries: list[CatalogEntry] = []
    with session_scope(driver, settings.database) as session:
        for row in fetch_all_context_entries(session):
            slug = str(row.get("slug") or "")
            entries.append(
                CatalogEntry(
                    category=str(row.get("category") or ""),
                    name=slug,
                    iri=str(row["iri"]),
                    label=str(row.get("label") or slug),
                    source=str(row.get("source") or ""),
                    definition=row.get("definition"),
                    polarity=row.get("polarity"),
                    instrument=row.get("instrument"),
                    theory_mappings=None,
                )
            )
    return ContextIriCatalog(entries)


def validate_iri(catalog: ContextIriCatalog, iri: str, *, where: str) -> None:
    """Raise ValueError when an IRI is not in the catalog."""
    if not catalog.has(iri):
        raise ValueError(f"{where}: ontology_uri {iri!r} not in Context dictionary")


__all__ = [
    "CatalogEntry",
    "ContextIriCatalog",
    "load_catalog",
    "validate_iri",
]
