"""Catalog of ontologies imported into the GraphRAG system.

The catalog data lives in the sibling `ontology_metadata.yaml`; this module
loads it into `OntologySpec` records. Each entry is downloaded on demand by
`src/scripts/download_ontologies.py` into `src/assets/ontologies/`, then
imported into Neo4j by `src/scripts/import_ontologies.py` via the n10s plugin.

`download_url` may contain the placeholder `{apikey}`, that is substituted with
`BIOPORTAL_API_KEY` from .env at download time. URLs without that placeholder
(e.g. OBO Foundry purls) need no credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CATALOG_PATH = Path(__file__).with_name("ontology_metadata.yaml")


@dataclass(frozen=True)
class OntologySpec:
    key: str
    file: str
    title: str
    rdf_format: str  # n10s format hint: "RDF/XML", "Turtle", "JSON-LD", "N-Triples"
    description: str
    download_url: str  # may include literal `{apikey}` for BioPortal; "" for local
    local: bool = (
        False  # True = author-defined, local in src/assets/ontologies/, not downloaded
    )
    # URI prefixes that identify this ontology's nodes in the shared
    # Neo4j graph.  Used by the task-generation pipeline to constrain
    # GraphRAG retrieval to the ontologies the scenario YAML actually
    # asked for (`task_generation.ontologies`): when a scenario lists
    # `HealthTasks`, the retriever is restricted to URIs starting
    # with HealthTasks' prefix(es), so the LLM is not tempted by
    # off-domain candidates from co-loaded ontologies (HumanActivities,
    # BCTT, OCHV, …).  Empty tuple means "no scenario-level URI gate
    # for this ontology"; it stays available as background context but
    # never filters retrieval; appropriate for BioPortal references
    # that we do not draw task instances from.
    uri_namespaces: tuple[str, ...] = ()


def _load_catalog() -> tuple[OntologySpec, ...]:
    """Load the ontology catalog from the sibling YAML file."""
    raw = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    return tuple(
        OntologySpec(
            key=entry["key"],
            file=entry["file"],
            title=entry["title"],
            rdf_format=entry["rdf_format"],
            description=entry["description"],
            download_url=entry.get("download_url", ""),
            local=entry.get("local", False),
            uri_namespaces=tuple(entry.get("uri_namespaces", ())),
        )
        for entry in raw["ontologies"]
    )


ONTOLOGIES: tuple[OntologySpec, ...] = _load_catalog()


def by_key(key: str) -> OntologySpec | None:
    for spec in ONTOLOGIES:
        if spec.key == key:
            return spec
    return None


def uri_prefixes_for_keys(keys: list[str] | tuple[str, ...]) -> list[str]:
    """Resolve a list of ontology keys to the union of their URI prefixes.

    Unknown keys and keys whose `OntologySpec` declares no namespaces
    are silently dropped; caller-side validation decides whether that
    is an error (e.g. scenario YAML asked for an ontology we do not
    know how to gate) or expected (e.g. BioPortal background-only
    ontology with no instance namespace).

    Returns a deduplicated list preserving caller order so the
    resulting Cypher filter is stable across runs.
    """
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        spec = by_key(k)
        if spec is None:
            continue
        for ns in spec.uri_namespaces:
            if ns not in seen:
                seen.add(ns)
                out.append(ns)
    return out
