"""Unit tests for the admissible-relation-set extension of metrics/allen.py.

Coverage targets:
  - _build_rule_index: empty, single rule, multiple rules, symmetric key.
  - build_admissible_rx: rule-index hit, concurrent_with hit, task.is_concurrent,
    event.is_concurrent, default R_SEP.
"""

from __future__ import annotations

import pytest

from src.scripts.persona.config.schema import AllenPairRule
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.allen import (
    R_ALL,
    R_SEP,
    AllenRelation,
    _build_rule_index,
    build_admissible_rx,
)
from tests.unit.scenarios.conftest import make_event, make_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rule(
    id_: str,
    event_a: str,
    event_b: str,
    relations: list[str],
) -> AllenPairRule:
    return AllenPairRule(
        id=id_,
        event_a=event_a,
        event_b=event_b,
        admissible_relations=relations,
    )


def _empty_index() -> dict:
    return _build_rule_index([])


# ---------------------------------------------------------------------------
# _build_rule_index
# ---------------------------------------------------------------------------


class TestBuildRuleIndex:
    def test_empty_rules_gives_empty_index(self):
        assert _build_rule_index([]) == {}

    def test_single_rule_adds_frozenset_key(self):
        rule = _rule("r1", "sleep", "running", ["p", "m", "M", "P"])
        index = _build_rule_index([rule])
        key = frozenset({"sleep", "running"})
        assert key in index

    def test_relation_tokens_converted_to_enum(self):
        rule = _rule("r1", "a", "b", ["p", "e"])
        index = _build_rule_index([rule])
        key = frozenset({"a", "b"})
        assert AllenRelation.p in index[key]
        assert AllenRelation.e in index[key]

    def test_key_is_symmetric(self):
        rule = _rule("r1", "x", "y", ["p"])
        index = _build_rule_index([rule])
        assert frozenset({"x", "y"}) in index
        assert frozenset({"y", "x"}) in index  # same frozenset

    def test_multiple_rules(self):
        rules = [
            _rule("r1", "a", "b", ["p"]),
            _rule("r2", "c", "d", ["e"]),
        ]
        index = _build_rule_index(rules)
        assert len(index) == 2
        assert frozenset({"a", "b"}) in index
        assert frozenset({"c", "d"}) in index

    def test_all_thirteen_tokens_accepted(self):
        tokens = ["p", "m", "o", "s", "d", "f", "e", "P", "M", "O", "S", "D", "F"]
        rule = _rule("r1", "a", "b", tokens)
        index = _build_rule_index([rule])
        assert index[frozenset({"a", "b"})] == R_ALL


# ---------------------------------------------------------------------------
# build_admissible_rx
# ---------------------------------------------------------------------------


class TestBuildAdmissibleRx:
    def test_rule_index_hit_returns_rule_relations(self):
        rule = _rule("r1", "walking", "sleep", ["p", "m", "M", "P"])
        index = _build_rule_index([rule])
        task = make_task(label="walking")
        event = make_event(label="sleep")
        result = build_admissible_rx(task, event, index)
        assert result == frozenset(
            {AllenRelation.p, AllenRelation.m, AllenRelation.M, AllenRelation.P}
        )

    def test_rule_index_hit_reversed_labels(self):
        """(event_a, event_b) order in rule doesn't matter; key is a frozenset."""
        rule = _rule("r1", "sleep", "walking", ["p"])
        index = _build_rule_index([rule])
        task = make_task(label="walking")
        event = make_event(label="sleep")
        result = build_admissible_rx(task, event, index)
        assert AllenRelation.p in result

    def test_concurrent_with_hit_gives_r_all(self):
        task = make_task(label="reading")
        event = make_event(label="lunch", concurrent_with=["reading"])
        result = build_admissible_rx(task, event, _empty_index())
        assert result == R_ALL

    def test_task_is_concurrent_gives_r_all(self):
        task = make_task(label="podcast", is_concurrent=True)
        event = make_event(label="walking")
        result = build_admissible_rx(task, event, _empty_index())
        assert result == R_ALL

    def test_event_is_concurrent_gives_r_all(self):
        task = make_task(label="yoga")
        event = make_event(label="lunch", is_concurrent=True)
        result = build_admissible_rx(task, event, _empty_index())
        assert result == R_ALL

    def test_default_returns_r_sep(self):
        task = make_task(label="running", is_concurrent=False)
        event = make_event(label="sleep", is_concurrent=False, concurrent_with=[])
        result = build_admissible_rx(task, event, _empty_index())
        assert result == R_SEP

    def test_rule_index_takes_priority_over_concurrent_with(self):
        """An explicit rule overrides the concurrent_with flag."""
        rule = _rule("r1", "task_a", "event_x", ["p"])  # only 'p' allowed
        index = _build_rule_index([rule])
        task = make_task(label="task_a", is_concurrent=True)
        event = make_event(
            label="event_x", is_concurrent=True, concurrent_with=["task_a"]
        )
        result = build_admissible_rx(task, event, index)
        assert result == frozenset({AllenRelation.p})

    def test_concurrent_with_takes_priority_over_flags(self):
        """concurrent_with hit is checked before is_concurrent flags."""
        task = make_task(label="mindful_eating", is_concurrent=False)
        event = make_event(
            label="lunch", is_concurrent=False, concurrent_with=["mindful_eating"]
        )
        result = build_admissible_rx(task, event, _empty_index())
        assert result == R_ALL
