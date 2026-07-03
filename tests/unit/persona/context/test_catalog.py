"""Tests for the IRI catalog loader (Neo4j-backed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scripts.persona.context.catalog import (
    CatalogEntry,
    ContextIriCatalog,
    load_catalog,
    validate_iri,
)

_FAKE_ROWS = [
    {
        "iri": "http://purl.obolibrary.org/obo/MFOEM_000042",
        "slug": "happiness",
        "source": "MFOEM",
        "label": "happiness",
        "definition": "A positive emotion.",
        "polarity": None,
        "instrument": None,
        "comment": None,
        "category": "mood_emotion",
    },
    {
        "iri": "http://example.org/b",
        "slug": "thing_two",
        "source": "TEST",
        "label": "thing two",
        "definition": None,
        "polarity": "high",
        "instrument": "test",
        "comment": None,
        "category": "category_a",
    },
]


def _stub_load(rows):
    """Build the catalog with stubbed Neo4j session + bridge response."""
    driver = MagicMock()
    session = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = session
    cm.__exit__.return_value = False
    settings = MagicMock()
    settings.database = "neo4j"
    with patch(
        "src.graphrag.config.Neo4jSettings.from_env", return_value=settings
    ), patch(
        "src.graphrag.neo4j_client.session_scope", return_value=cm
    ) as mock_scope, patch(
        "src.scripts.scenarios.task_generation.ontology_bridge.fetch_all_context_entries",
        return_value=rows,
    ):
        cat = load_catalog(driver)
    return cat, driver, mock_scope


def test_load_catalog_requires_driver():
    with pytest.raises(ValueError, match="requires a Neo4j driver"):
        load_catalog(None)


def test_load_catalog_returns_known_categories():
    cat, *_ = _stub_load(_FAKE_ROWS)
    assert len(cat) == 2
    assert set(cat.categories()) == {"mood_emotion", "category_a"}


def test_has_and_get_round_trip():
    cat, *_ = _stub_load(_FAKE_ROWS)
    iri = "http://purl.obolibrary.org/obo/MFOEM_000042"
    assert cat.has(iri)
    entry = cat.get(iri)
    assert entry is not None
    assert entry.category == "mood_emotion"
    assert entry.label == "happiness"
    assert entry.source == "MFOEM"


def test_get_returns_none_for_unknown_iri():
    cat, *_ = _stub_load(_FAKE_ROWS)
    assert cat.get("http://example.org/not-in-catalog") is None
    assert not cat.has("http://example.org/not-in-catalog")


def test_members_returns_entries_for_category():
    cat, *_ = _stub_load(_FAKE_ROWS)
    members = cat.members("mood_emotion")
    assert members
    assert all(e.category == "mood_emotion" for e in members)


def test_members_unknown_category_returns_empty_list():
    cat, *_ = _stub_load(_FAKE_ROWS)
    assert cat.members("does_not_exist") == []


def test_validate_iri_raises_on_missing():
    cat, *_ = _stub_load(_FAKE_ROWS)
    with pytest.raises(ValueError, match="not in Context dictionary"):
        validate_iri(cat, "http://example.org/bogus", where="test")


def test_validate_iri_silent_on_present():
    cat, *_ = _stub_load(_FAKE_ROWS)
    validate_iri(
        cat,
        "http://purl.obolibrary.org/obo/MFOEM_000042",
        where="test",
    )


def test_load_catalog_passes_db_setting_to_session_scope():
    rows = [_FAKE_ROWS[0]]
    cat, _driver, mock_scope = _stub_load(rows)
    assert len(cat) == 1
    mock_scope.assert_called_once()


def test_load_catalog_carries_optional_fields_through():
    cat, *_ = _stub_load(_FAKE_ROWS)
    entry = cat.get("http://example.org/b")
    assert entry is not None
    assert entry.polarity == "high"
    assert entry.instrument == "test"
    # theory_mappings is not loaded from Neo4j; downstream code reads it from
    # the persona schema, not from the catalog.
    assert entry.theory_mappings is None


def test_load_catalog_falls_back_to_slug_when_label_missing():
    rows = [
        {
            "iri": "http://example.org/no-label",
            "slug": "fallback_slug",
            "source": "TEST",
            "label": None,
            "definition": None,
            "polarity": None,
            "instrument": None,
            "comment": None,
            "category": "category_a",
        }
    ]
    cat, *_ = _stub_load(rows)
    entry = cat.get("http://example.org/no-label")
    assert entry is not None
    assert entry.label == "fallback_slug"


def test_catalog_class_directly_constructed():
    entries = [
        CatalogEntry(
            category="x",
            name="y",
            iri="iri:1",
            label="y",
            source="src",
            definition=None,
            polarity=None,
            instrument=None,
            theory_mappings=None,
        )
    ]
    cat = ContextIriCatalog(entries)
    assert len(cat) == 1
    assert cat.categories() == ["x"]
