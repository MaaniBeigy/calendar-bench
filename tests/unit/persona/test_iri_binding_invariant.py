"""Persona context members bind to the catalog by IRI, not by slug."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.scripts.persona.config.loader import ConfigError, _validate_context_iris
from src.scripts.persona.config.schema import (
    ContextCategory,
    ContextMember,
    DurationRange,
    EpisodeRange,
    Persona,
    PersonaConfig,
    TotalDuration,
)
from src.scripts.persona.context.catalog import CatalogEntry, ContextIriCatalog

FEELING_ENERGETIC_IRI = "http://purl.obolibrary.org/obo/MFOEM_000109"


def _context_member(*, ontology_uri: str) -> ContextMember:
    return ContextMember(
        per_event_duration=DurationRange(min=15, max=60),
        total_event_duration=TotalDuration(min=30, max=180),
        total_event_episodes=EpisodeRange(min=1, max=3),
        ontology_uri=ontology_uri,
    )


def _persona_with_local_slug(local_slug: str, *, iri: str) -> PersonaConfig:
    return PersonaConfig(
        personas=[
            Persona(
                id="p",
                instances=1,
                contexts={
                    "energy_state": ContextCategory(
                        members={local_slug: _context_member(ontology_uri=iri)},
                    ),
                },
            )
        ]
    )


def _catalog_with_energetic() -> ContextIriCatalog:
    return ContextIriCatalog(
        [
            CatalogEntry(
                category="energy_state",
                name="feeling_energetic",
                iri=FEELING_ENERGETIC_IRI,
                label="feeling energetic",
                source="MFOEM",
                definition=None,
                polarity=None,
                instrument=None,
                theory_mappings=None,
            )
        ]
    )


def _patch(monkeypatch, catalog: ContextIriCatalog) -> None:
    import src.scripts.persona.context.catalog as catalog_module

    monkeypatch.setattr(catalog_module, "load_catalog", lambda driver: catalog)


def test_persona_local_slug_differs_from_catalog_slug_validates_via_iri(
    monkeypatch,
) -> None:
    _patch(monkeypatch, _catalog_with_energetic())
    cfg = _persona_with_local_slug("energetic", iri=FEELING_ENERGETIC_IRI)
    _validate_context_iris(cfg, driver=MagicMock())


def test_catalog_lookup_by_iri_returns_canonical_slug_and_label() -> None:
    catalog = _catalog_with_energetic()
    entry = catalog.get(FEELING_ENERGETIC_IRI)
    assert entry is not None
    assert entry.name == "feeling_energetic"
    assert entry.category == "energy_state"
    assert "energetic" in (entry.label or "").lower()


def test_persona_member_keeps_local_slug_after_load() -> None:
    cfg = _persona_with_local_slug("energetic", iri=FEELING_ENERGETIC_IRI)
    persona = cfg.personas[0]
    members = persona.contexts["energy_state"].members
    assert list(members) == ["energetic"]
    assert members["energetic"].ontology_uri == FEELING_ENERGETIC_IRI


def test_unknown_iri_still_rejected_regardless_of_slug(monkeypatch) -> None:
    _patch(monkeypatch, _catalog_with_energetic())
    cfg = _persona_with_local_slug(
        "feeling_energetic",
        iri="http://example.org/nonexistent_iri",
    )
    with pytest.raises(ConfigError, match="not in Context dictionary"):
        _validate_context_iris(cfg, driver=MagicMock())
