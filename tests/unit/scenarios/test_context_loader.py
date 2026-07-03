"""Tests for scenarios-side context loading (CalendarTrace.contexts)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from src.scripts.persona.domain.event import Catalog
from src.scripts.scenarios.calendar.loader import (
    _parse_context_episode,
    _parse_person_json,
)
from src.scripts.scenarios.domain.context import ContextEpisode


@pytest.fixture
def empty_catalog() -> Catalog:
    """Catalog with no events; the loader uses it for event metadata only."""
    return Catalog(events_by_name={}, categories={})


def _person_data(
    contexts: list[dict] | None,
    *,
    person_id: str = "p1",
    persona_id: str = "tpl",
) -> dict:
    body: dict = {
        "person_id": person_id,
        "persona_id": persona_id,
        "persona": {"id": persona_id, "event_overrides": {}},
        "days": [],
    }
    if contexts is not None:
        body["contexts"] = contexts
    return body


def test_parse_context_episode_round_trips_all_fields() -> None:
    raw = {
        "name": "tired",
        "category": "energy_state",
        "date": "2026-05-04",
        "start_minutes": 720,
        "end_minutes": 780,
        "ontology_uri": "http://example.org/iri",
        "dimension": "extraversion",
        "polarity": "low",
        "instrument": "MFOEM",
        "theory_mappings": {"comb": "Motivation"},
    }
    ep = _parse_context_episode(raw)
    assert ep == ContextEpisode(
        name="tired",
        category="energy_state",
        date=datetime.date(2026, 5, 4),
        start_minutes=720,
        end_minutes=780,
        ontology_uri="http://example.org/iri",
        dimension="extraversion",
        polarity="low",
        instrument="MFOEM",
        theory_mappings={"comb": "Motivation"},
    )


def test_parse_context_episode_optional_fields_default_to_none() -> None:
    raw = {
        "name": "happy",
        "category": "mood_emotion",
        "date": "2026-05-04",
        "start_minutes": 480,
        "end_minutes": 540,
    }
    ep = _parse_context_episode(raw)
    assert ep.ontology_uri is None
    assert ep.dimension is None
    assert ep.polarity is None
    assert ep.instrument is None
    assert ep.theory_mappings is None
    assert ep.duration == 60


def test_parse_person_json_missing_contexts_key_returns_empty_list(
    empty_catalog: Catalog,
) -> None:
    data = _person_data(contexts=None)
    trace = _parse_person_json(data, empty_catalog)
    assert trace.contexts == []
    assert trace.person_id == "p1"


def test_parse_person_json_empty_contexts_list_returns_empty(
    empty_catalog: Catalog,
) -> None:
    data = _person_data(contexts=[])
    trace = _parse_person_json(data, empty_catalog)
    assert trace.contexts == []


def test_parse_person_json_sorts_contexts_by_date_start_category_name(
    empty_catalog: Catalog,
) -> None:
    rows = [
        {
            "name": "energetic",
            "category": "energy_state",
            "date": "2026-05-05",
            "start_minutes": 480,
            "end_minutes": 540,
        },
        {
            "name": "happy",
            "category": "mood_emotion",
            "date": "2026-05-04",
            "start_minutes": 540,
            "end_minutes": 600,
        },
        {
            "name": "tired",
            "category": "energy_state",
            "date": "2026-05-04",
            "start_minutes": 540,
            "end_minutes": 600,
        },
    ]
    data = _person_data(contexts=rows)
    trace = _parse_person_json(data, empty_catalog)

    assert [
        (c.date.isoformat(), c.start_minutes, c.category, c.name)
        for c in trace.contexts
    ] == [
        ("2026-05-04", 540, "energy_state", "tired"),
        ("2026-05-04", 540, "mood_emotion", "happy"),
        ("2026-05-05", 480, "energy_state", "energetic"),
    ]


def test_parse_person_json_via_tmpdir(tmp_path: Path, empty_catalog: Catalog) -> None:
    data = _person_data(
        contexts=[
            {
                "name": "home",
                "category": "location",
                "date": "2026-05-04",
                "start_minutes": 0,
                "end_minutes": 1440,
                "ontology_uri": "http://purl.obolibrary.org/obo/ENVO_01000744",
            }
        ]
    )
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace = _parse_person_json(payload, empty_catalog)
    assert len(trace.contexts) == 1
    assert (
        trace.contexts[0].ontology_uri == "http://purl.obolibrary.org/obo/ENVO_01000744"
    )
