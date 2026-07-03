"""Tests for the `contexts:` schema on `Persona`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import (
    ContextCategory,
    ContextMember,
    DurationRange,
    EpisodeRange,
    Persona,
    RolePredicate,
    TemporalPattern,
    TotalDuration,
)


def _member(**overrides) -> ContextMember:
    payload: dict = {
        "per_event_duration": DurationRange(min=15, max=60),
        "total_event_duration": TotalDuration(min=30, max=180),
        "total_event_episodes": EpisodeRange(min=1, max=3),
    }
    payload.update(overrides)
    return ContextMember(**payload)


def test_context_member_defaults():
    m = _member()
    assert m.temporal_patterns == []
    assert m.requires == {}
    assert m.ontology_uri is None
    assert m.dimension is None
    assert m.polarity is None
    assert m.instrument is None
    assert m.theory_mappings is None


def test_context_member_passthrough_metadata():
    m = _member(
        ontology_uri="http://x/y",
        dimension="extraversion",
        polarity="high",
        instrument="BFI",
        theory_mappings={"comb": "physical_capability"},
    )
    assert m.dimension == "extraversion"
    assert m.polarity == "high"
    assert m.instrument == "BFI"
    assert m.theory_mappings == {"comb": "physical_capability"}


def test_context_member_polarity_value_restricted():
    with pytest.raises(ValidationError):
        _member(polarity="invalid")


def test_context_member_requires_predicate_round_trip():
    m = _member(requires={"neuroticism": RolePredicate(ge=0.6)})
    assert m.requires["neuroticism"].ge == 0.6


def test_context_member_temporal_pattern_passthrough():
    pattern = TemporalPattern(
        mode="seasonality",
        details={"within": ["morning"], "amount": 30, "direction": "increasing"},
    )
    m = _member(temporal_patterns=[pattern])
    assert m.temporal_patterns[0].mode == "seasonality"


def test_context_category_default_mutually_exclusive_none():
    cat = ContextCategory(members={"a": _member()})
    assert cat.mutually_exclusive is None


def test_context_category_members_required_non_empty():
    with pytest.raises(ValidationError):
        ContextCategory(mutually_exclusive=True, members={})


def test_persona_contexts_default_empty():
    p = Persona(id="p", instances=1)
    assert p.contexts == {}


def test_persona_with_contexts_block():
    cat = ContextCategory(mutually_exclusive=True, members={"happy": _member()})
    p = Persona(id="p", instances=1, contexts={"mood_emotion": cat})
    assert "mood_emotion" in p.contexts
    assert p.contexts["mood_emotion"].mutually_exclusive is True


def test_persona_with_contexts_round_trips_via_model_dump():
    cat = ContextCategory(mutually_exclusive=True, members={"happy": _member()})
    p = Persona(id="p", instances=1, contexts={"mood_emotion": cat})
    reloaded = Persona.model_validate(p.model_dump())
    assert "mood_emotion" in reloaded.contexts


def test_context_member_extra_field_rejected():
    with pytest.raises(ValidationError):
        ContextMember.model_validate(
            {
                "per_event_duration": {"min": 15, "max": 60},
                "total_event_duration": {"min": 30, "max": 180, "scale": "day"},
                "total_event_episodes": {"min": 1, "max": 3, "scale": "day"},
                "unknown_field": True,
            }
        )
