"""Unit tests for SelectorPredicate + AllenPairRule selector grammar.

Covers the selector grammar contract:
* SelectorPredicate validates the five keys (name, intensity, domain,
  met_min, met_max) with at-least-one-set and intensity ∈ {1..4}.
* AllenPairRule.event_a / event_b accept either a literal string or a
  selector dict; strings auto-promote to `{"name": "<value>"}`.
* Per-rule `buffer` is optional.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.scripts.persona.config.schema import AllenPairRule, SelectorPredicate

# ---------------------------------------------------------------------------
# SelectorPredicate
# ---------------------------------------------------------------------------


class TestSelectorPredicate:
    def test_at_least_one_key_required(self):
        with pytest.raises(ValidationError):
            SelectorPredicate()

    def test_intensity_must_be_1_to_4(self):
        with pytest.raises(ValidationError):
            SelectorPredicate(intensity=[0])
        with pytest.raises(ValidationError):
            SelectorPredicate(intensity=[5])

    def test_intensity_list_ok(self):
        s = SelectorPredicate(intensity=[1, 2, 3, 4])
        assert s.intensity == [1, 2, 3, 4]

    def test_name_string(self):
        s = SelectorPredicate(name="lunch")
        assert s.name == "lunch"

    def test_name_list(self):
        s = SelectorPredicate(name=["lunch", "dinner"])
        assert s.name == ["lunch", "dinner"]

    def test_domain(self):
        s = SelectorPredicate(domain="NutritionTask")
        assert s.domain == "NutritionTask"

    def test_met_range(self):
        s = SelectorPredicate(met_min=2.0, met_max=6.0)
        assert s.met_min == pytest.approx(2.0)
        assert s.met_max == pytest.approx(6.0)

    def test_met_max_below_min_rejected(self):
        with pytest.raises(ValidationError):
            SelectorPredicate(met_min=6.0, met_max=2.0)

    def test_met_only_min_ok(self):
        s = SelectorPredicate(met_min=2.0)
        assert s.met_min == pytest.approx(2.0)
        assert s.met_max is None

    def test_negative_met_rejected(self):
        with pytest.raises(ValidationError):
            SelectorPredicate(met_min=-1.0)
        with pytest.raises(ValidationError):
            SelectorPredicate(met_max=-1.0)


# ---------------------------------------------------------------------------
# AllenPairRule; selector / literal grammar
# ---------------------------------------------------------------------------


class TestAllenPairRuleGrammar:
    def test_two_literals_promote(self):
        rule = AllenPairRule(
            id="r1",
            event_a="lunch",
            event_b="dinner",
            admissible_relations=["p", "P"],
        )
        assert rule.event_a.name == "lunch"
        assert rule.event_b.name == "dinner"

    def test_two_predicates(self):
        rule = AllenPairRule(
            id="r2",
            event_a={"intensity": [2, 3, 4]},
            event_b={"intensity": [2, 3, 4]},
            admissible_relations=["p", "P"],
        )
        assert rule.event_a.intensity == [2, 3, 4]
        assert rule.event_b.intensity == [2, 3, 4]

    def test_mixed_literal_and_predicate(self):
        rule = AllenPairRule(
            id="r3",
            event_a={"name": "lunch"},
            event_b={"intensity": [2, 3, 4]},
            admissible_relations=["p", "P"],
        )
        assert rule.event_a.name == "lunch"
        assert rule.event_b.intensity == [2, 3, 4]

    def test_list_of_names_promotes(self):
        rule = AllenPairRule(
            id="r4",
            event_a=["lunch", "dinner"],
            event_b="reading",
            admissible_relations=["p", "m", "M", "P"],
        )
        assert rule.event_a.name == ["lunch", "dinner"]
        assert rule.event_b.name == "reading"

    def test_admissible_relations_required(self):
        with pytest.raises(ValidationError):
            AllenPairRule(
                id="r",
                event_a="lunch",
                event_b="dinner",
                admissible_relations=[],
            )

    def test_buffer_optional(self):
        rule = AllenPairRule(
            id="r",
            event_a="lunch",
            event_b="dinner",
            admissible_relations=["p", "P"],
        )
        assert rule.buffer is None

    def test_buffer_set(self):
        rule = AllenPairRule(
            id="r",
            event_a="lunch",
            event_b="dinner",
            admissible_relations=["p", "P"],
            buffer=60,
        )
        assert rule.buffer == 60

    def test_buffer_negative_rejected(self):
        with pytest.raises(ValidationError):
            AllenPairRule(
                id="r",
                event_a="lunch",
                event_b="dinner",
                admissible_relations=["p", "P"],
                buffer=-5,
            )

    def test_invalid_relation_token_rejected(self):
        with pytest.raises(ValidationError):
            AllenPairRule(
                id="r",
                event_a="lunch",
                event_b="dinner",
                admissible_relations=["x"],
            )

    def test_d1_example_rules(self):
        """The three example selector rules must parse verbatim."""
        r1 = AllenPairRule(
            id="intensive_far_from_intensive",
            event_a={"intensity": [2, 3, 4]},
            event_b={"intensity": [2, 3, 4]},
            admissible_relations=["p", "P"],
        )
        r2 = AllenPairRule(
            id="lunch_far_from_intensive",
            event_a={"name": "lunch"},
            event_b={"intensity": [2, 3, 4]},
            admissible_relations=["p", "P"],
        )
        r3 = AllenPairRule(
            id="reading_adjacent_to_lunch",
            event_a={"name": "reading"},
            event_b={"name": "lunch"},
            admissible_relations=["p", "m", "M", "P"],
        )
        # Sanity:
        assert r1.event_a.intensity == [2, 3, 4]
        assert r2.event_b.intensity == [2, 3, 4]
        assert r3.event_a.name == "reading"
