"""Unit tests for src.scripts.persona.event_config.loader."""

from __future__ import annotations

import pytest

from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.event_config.loader import load_catalog


def test_load_catalog_indexes_events(event_yaml):
    catalog = load_catalog(event_yaml)
    assert isinstance(catalog, Catalog)
    assert len(catalog) > 0
    assert "sleep" in catalog
    assert "office_work" in catalog


def test_catalog_get_returns_event_definition(event_yaml):
    catalog = load_catalog(event_yaml)
    sleep = catalog.get("sleep")
    assert sleep.name == "sleep"
    assert sleep.category == "sleep"


def test_catalog_get_unknown_event_raises(event_yaml):
    catalog = load_catalog(event_yaml)
    with pytest.raises(KeyError):
        catalog.get("not_a_real_event")


def test_catalog_by_category(event_yaml):
    catalog = load_catalog(event_yaml)
    sports = catalog.by_category("sports")
    names = {e.name for e in sports}
    assert {"padel", "running", "cycling", "swimming", "gym"} <= names


def test_catalog_by_category_unknown_raises(event_yaml):
    catalog = load_catalog(event_yaml)
    with pytest.raises(KeyError):
        catalog.by_category("not_a_category")


def test_catalog_membership(event_yaml):
    catalog = load_catalog(event_yaml)
    assert "running" in catalog
    assert "no_such_event" not in catalog
