"""`SelectorPredicate.kind` accepts `event` (default) and `context`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import AllenPairRule, SelectorPredicate


def test_kind_defaults_to_event():
    s = SelectorPredicate(name="walking")
    assert s.kind == "event"


def test_kind_context_accepted_with_name():
    s = SelectorPredicate(kind="context", name="anxious")
    assert s.kind == "context"


def test_kind_context_rejects_event_only_fields():
    for forbidden, value in (
        ("intensity", [2]),
        ("domain", "PhysicalActivityTask"),
        ("met_min", 1.0),
        ("met_max", 5.0),
        ("health_task_class", "PhysicalActivityTask"),
        ("health_task_uri", "https://example.org/x"),
    ):
        with pytest.raises(ValidationError, match="event/task-only"):
            SelectorPredicate(kind="context", name="x", **{forbidden: value})


def test_kind_invalid_value_rejected():
    with pytest.raises(ValidationError):
        SelectorPredicate(kind="something_else", name="x")


def test_kind_round_trips_through_allen_pair_rule():
    rule = AllenPairRule(
        id="r",
        event_a="lunch",
        event_b={"kind": "context", "name": "anxious"},
        admissible_relations=["p", "P"],
    )
    assert rule.event_a.kind == "event"
    assert rule.event_b.kind == "context"
    assert rule.event_b.name == "anxious"
