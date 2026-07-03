"""Validation of literal-name endpoints in `AllenPairRule`."""

from __future__ import annotations

import pytest

from src.scripts.persona.config.loader import (
    ConfigError,
    _validate_allen_pair_rules,
)
from src.scripts.persona.config.schema import (
    AllenPairRule,
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    SelectorPredicate,
    TemporalRelationRules,
    TotalDuration,
)


def _event_cfg_with(events: dict[str, EventDefinition]) -> EventConfig:
    return EventConfig(categories={"test": Category(name="test", events=events)})


def _event_def(name: str) -> EventDefinition:
    return EventDefinition(
        name=name,
        category="test",
        per_event_duration=DurationRange(min=10, max=30, unit="minutes"),
        total_event_duration=TotalDuration(scale="day", min=10, max=30, unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
    )


def test_predicate_only_selector_is_skipped():
    """A selector with no `name` field is accepted without cross-checking."""
    rule = AllenPairRule(
        id="predicate_only",
        event_a=SelectorPredicate(intensity=[3, 4]),
        event_b=SelectorPredicate(domain="NutritionTask"),
        admissible_relations=["p", "P"],
    )
    cfg = _event_cfg_with({"walking": _event_def("walking")})
    rules = TemporalRelationRules(allen_pair_rules=[rule])
    _validate_allen_pair_rules(cfg, rules)


def test_mixed_selector_with_unknown_name_does_not_raise():
    """A selector that mixes a literal name with predicate fields is not
    flagged even when the name is absent from the catalog."""
    rule = AllenPairRule(
        id="mixed_selector",
        event_a=SelectorPredicate(name="ghost_event", intensity=[2]),
        event_b=SelectorPredicate(name="walking"),
        admissible_relations=["p", "P"],
    )
    cfg = _event_cfg_with({"walking": _event_def("walking")})
    rules = TemporalRelationRules(allen_pair_rules=[rule])
    _validate_allen_pair_rules(cfg, rules)


def test_purely_literal_unknown_name_raises():
    """Two literal endpoints and one name absent from the catalog raise."""
    rule = AllenPairRule(
        id="purely_literal",
        event_a=SelectorPredicate(name="ghost"),
        event_b=SelectorPredicate(name="walking"),
        admissible_relations=["p", "P"],
    )
    cfg = _event_cfg_with({"walking": _event_def("walking")})
    rules = TemporalRelationRules(allen_pair_rules=[rule])
    with pytest.raises(ConfigError, match="not found in event catalog"):
        _validate_allen_pair_rules(cfg, rules)


def test_two_unknown_purely_literal_names_raise_against_non_empty_catalog():
    """Negative control: empty selector intersection still raises when
    the catalog is non-empty."""
    rule = AllenPairRule(
        id="literal_pair_unknown",
        event_a=SelectorPredicate(name="ghost_a"),
        event_b=SelectorPredicate(name="ghost_b"),
        admissible_relations=["p"],
    )
    cfg = _event_cfg_with({"sentinel": _event_def("sentinel")})
    rules = TemporalRelationRules(allen_pair_rules=[rule])
    with pytest.raises(ConfigError):
        _validate_allen_pair_rules(cfg, rules)
