"""Loader-side strict check that context `ontology_uri` exists in the catalog."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.scripts.persona.config.loader import (
    ConfigError,
    _validate_context_iris,
    load_persona,
)
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


def _member(ontology_uri: str | None) -> ContextMember:
    return ContextMember(
        per_event_duration=DurationRange(min=15, max=60),
        total_event_duration=TotalDuration(min=30, max=180),
        total_event_episodes=EpisodeRange(min=1, max=3),
        ontology_uri=ontology_uri,
    )


def _persona_cfg(*, ontology_uri: str | None) -> PersonaConfig:
    return PersonaConfig(
        personas=[
            Persona(
                id="p",
                instances=1,
                contexts={
                    "mood_emotion": ContextCategory(
                        mutually_exclusive=True,
                        members={"happy": _member(ontology_uri)},
                    )
                },
            )
        ]
    )


def _catalog(*iris: str) -> ContextIriCatalog:
    return ContextIriCatalog(
        CatalogEntry(
            category="mood_emotion",
            name="x",
            iri=iri,
            label="x",
            source="TEST",
            definition=None,
            polarity=None,
            instrument=None,
            theory_mappings=None,
        )
        for iri in iris
    )


def _patch_loader(monkeypatch, catalog: ContextIriCatalog) -> None:
    """Make `load_catalog` return our fake catalog without hitting Neo4j."""
    import src.scripts.persona.context.catalog as catalog_module

    monkeypatch.setattr(catalog_module, "load_catalog", lambda driver: catalog)


def test_no_op_when_no_persona_declares_contexts():
    cfg = PersonaConfig(personas=[Persona(id="p", instances=1)])
    _validate_context_iris(cfg, driver=None)


def test_iri_in_catalog_passes(monkeypatch):
    _patch_loader(monkeypatch, _catalog("http://purl.obolibrary.org/obo/MFOEM_000042"))
    cfg = _persona_cfg(ontology_uri="http://purl.obolibrary.org/obo/MFOEM_000042")
    _validate_context_iris(cfg, driver=MagicMock())


def test_iri_none_is_accepted(monkeypatch):
    _patch_loader(monkeypatch, _catalog())
    cfg = _persona_cfg(ontology_uri=None)
    _validate_context_iris(cfg, driver=MagicMock())


def test_iri_not_in_catalog_rejected(monkeypatch):
    _patch_loader(monkeypatch, _catalog("http://example.org/known"))
    cfg = _persona_cfg(ontology_uri="http://example.org/bogus")
    with pytest.raises(ConfigError, match="not in Context dictionary"):
        _validate_context_iris(cfg, driver=MagicMock())


def test_persona_yaml_with_unknown_iri_raises(tmp_path: Path, monkeypatch):
    _patch_loader(monkeypatch, _catalog("http://example.org/known"))
    payload = {
        "personas": [
            {
                "id": "p",
                "instances": 1,
                "contexts": {
                    "mood_emotion": {
                        "mutually_exclusive": True,
                        "members": {
                            "happy": {
                                "per_event_duration": {"min": 15, "max": 60},
                                "total_event_duration": {
                                    "min": 30,
                                    "max": 180,
                                    "scale": "day",
                                },
                                "total_event_episodes": {
                                    "min": 1,
                                    "max": 3,
                                    "scale": "day",
                                },
                                "ontology_uri": "http://example.org/bogus",
                            }
                        },
                    }
                },
            }
        ]
    }
    path = tmp_path / "persona.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_persona(path)
    with pytest.raises(ConfigError, match="not in Context dictionary"):
        _validate_context_iris(cfg, driver=MagicMock())
