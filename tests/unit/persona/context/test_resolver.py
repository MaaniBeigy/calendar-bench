"""Tests for `ContextResolver.resolve_for_person`."""

from __future__ import annotations

from src.scripts.persona.config.schema import (
    ContextCategory,
    ContextMember,
    DurationRange,
    EpisodeRange,
    JitterConfig,
    RolePredicate,
    TotalDuration,
)
from src.scripts.persona.context.resolver import resolve_for_person
from src.scripts.persona.domain.persona import Person


def _person(**characteristics) -> Person:
    return Person(
        person_id="p_0001",
        persona_id="p",
        person_seed=1,
        instance_index=0,
        characteristics=characteristics,
        jitter_applied=JitterConfig(),
    )


def _member(**overrides) -> ContextMember:
    payload: dict = {
        "per_event_duration": DurationRange(min=15, max=60),
        "total_event_duration": TotalDuration(min=30, max=180),
        "total_event_episodes": EpisodeRange(min=1, max=3),
    }
    payload.update(overrides)
    return ContextMember(**payload)


def test_all_members_enabled_when_no_requires():
    cat = ContextCategory(members={"happy": _member(), "sad": _member()})
    out = resolve_for_person(_person(), {"mood_emotion": cat})
    assert set(out["mood_emotion"].members) == {"happy", "sad"}


def test_requires_gates_members():
    cat = ContextCategory(
        members={
            "happy": _member(),
            "sad": _member(requires={"neuroticism": RolePredicate(ge=0.6)}),
        }
    )
    p = _person(neuroticism=0.3)
    out = resolve_for_person(p, {"mood_emotion": cat})
    assert set(out["mood_emotion"].members) == {"happy"}


def test_requires_enables_when_predicate_passes():
    cat = ContextCategory(
        members={
            "sad": _member(requires={"neuroticism": RolePredicate(ge=0.6)}),
        }
    )
    p = _person(neuroticism=0.8)
    out = resolve_for_person(p, {"mood_emotion": cat})
    assert "sad" in out["mood_emotion"].members


def test_category_dropped_when_no_member_enabled():
    cat = ContextCategory(
        members={
            "happy": _member(requires={"occupation_status": RolePredicate(eq="other")}),
        }
    )
    p = _person(occupation_status="fulltime")
    out = resolve_for_person(p, {"mood_emotion": cat})
    assert out == {}


def test_explicit_mutually_exclusive_true_honoured():
    cat = ContextCategory(mutually_exclusive=True, members={"happy": _member()})
    out = resolve_for_person(_person(), {"mood_emotion": cat})
    assert out["mood_emotion"].mutually_exclusive is True


def test_explicit_mutually_exclusive_false_honoured():
    # mood_emotion defaults to True, but explicit False must win.
    cat = ContextCategory(mutually_exclusive=False, members={"happy": _member()})
    out = resolve_for_person(_person(), {"mood_emotion": cat})
    assert out["mood_emotion"].mutually_exclusive is False


def test_default_used_when_flag_unset():
    cat = ContextCategory(members={"happy": _member()})
    out = resolve_for_person(_person(), {"mood_emotion": cat})
    assert out["mood_emotion"].mutually_exclusive is True


def test_default_for_unknown_category_falls_to_false():
    cat = ContextCategory(members={"foo": _member()})
    out = resolve_for_person(_person(), {"weird_category": cat})
    assert out["weird_category"].mutually_exclusive is False


def test_disjoint_resolution_across_two_persons():
    cat = ContextCategory(
        members={
            "anxious": _member(requires={"neuroticism": RolePredicate(ge=0.7)}),
        }
    )
    high = _person(neuroticism=0.9)
    low = _person(neuroticism=0.1)
    out_high = resolve_for_person(high, {"mood_emotion": cat})
    out_low = resolve_for_person(low, {"mood_emotion": cat})
    assert "mood_emotion" in out_high
    assert out_low == {}
