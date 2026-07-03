"""Helper functions in `src.scripts.scenarios.metrics.allen`.

Covers `SelectorMatcher.resolve` edge cases, `RuleSet.rules` /
`buffer_for`, `_fallback_for`, `_get_concurrent_flag`, and the legacy
`_build_rule_index`.
"""

from __future__ import annotations

import datetime

import pytest

from src.scripts.persona.config.schema import AllenPairRule, SelectorPredicate
from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import (
    R_ALL,
    R_SEP,
    AllenRelation,
    ResolvedActivity,
    RuleSet,
    SelectorMatcher,
    _build_rule_index,
    _fallback_for,
    _get_concurrent_flag,
)

DATE = datetime.date(2026, 5, 1)


def test_resolve_calendar_event_with_zero_intensity_returns_none_intensity():
    """`CalendarEvent.intensity == 0` is treated as no signal."""
    ev = CalendarEvent(
        label="lunch",
        start_minutes=720,
        end_minutes=750,
        date=DATE,
        intensity=0,
    )
    resolved = SelectorMatcher().resolve(ev)
    assert resolved.label == "lunch"
    assert resolved.intensity is None


def test_ruleset_rules_property_returns_copy():
    """`RuleSet.rules` returns a fresh list per call."""
    r = AllenPairRule(
        id="r0",
        event_a=SelectorPredicate(name="a"),
        event_b=SelectorPredicate(name="b"),
        admissible_relations=["p"],
    )
    rs = RuleSet([r])
    out = rs.rules
    assert out == [r]
    out.clear()
    assert rs.rules == [r]


def test_buffer_for_skips_rules_without_buffer():
    """Rules with `buffer=None` are walked past and the buffered rule wins."""
    no_buffer = AllenPairRule(
        id="no_buf",
        event_a=SelectorPredicate(name="a"),
        event_b=SelectorPredicate(name="b"),
        admissible_relations=["p"],
    )
    with_buffer = AllenPairRule(
        id="buf",
        event_a=SelectorPredicate(name="a"),
        event_b=SelectorPredicate(name="b"),
        admissible_relations=["p"],
        buffer=30,
    )
    rs = RuleSet([no_buffer, with_buffer])
    matcher = SelectorMatcher()
    a = ResolvedActivity(label="a")
    b = ResolvedActivity(label="b")
    assert rs.buffer_for(a, b, matcher, default=10) == 30


def test_buffer_for_returns_default_when_only_buffer_none_rules_present():
    """No rule has a buffer; the caller default wins."""
    no_buffer = AllenPairRule(
        id="no_buf",
        event_a=SelectorPredicate(name="a"),
        event_b=SelectorPredicate(name="b"),
        admissible_relations=["p"],
    )
    rs = RuleSet([no_buffer])
    matcher = SelectorMatcher()
    a = ResolvedActivity(label="a")
    b = ResolvedActivity(label="b")
    assert rs.buffer_for(a, b, matcher, default=7) == 7


def test_buffer_for_skips_buffered_rule_when_selectors_dont_match():
    """A buffered rule whose selectors do not match the pair is skipped."""
    non_matching = AllenPairRule(
        id="non_match_buf",
        event_a=SelectorPredicate(name="x"),
        event_b=SelectorPredicate(name="y"),
        admissible_relations=["p"],
        buffer=99,
    )
    rs = RuleSet([non_matching])
    matcher = SelectorMatcher()
    a = ResolvedActivity(label="a")
    b = ResolvedActivity(label="b")
    assert rs.buffer_for(a, b, matcher, default=5) == 5


class _StubActivity:
    """Minimal duck-typed object exposing the fields `_fallback_for` reads."""

    def __init__(self, label: str, concurrent_with, is_concurrent: bool = False):
        self.label = label
        self.concurrent_with = concurrent_with
        self.is_concurrent = is_concurrent


def test_fallback_for_returns_r_all_when_a_lists_b_in_concurrent_with():
    """`a.concurrent_with` includes b's label, so the pair gets `R_ALL`."""
    a = _StubActivity("walking", concurrent_with=["reading"])
    b = _StubActivity("reading", concurrent_with=[])
    assert _fallback_for(a, b) == R_ALL


def test_fallback_for_returns_r_sep_when_neither_side_links_to_other():
    """No `concurrent_with` link and no `is_concurrent` flag; `R_SEP`."""
    a = _StubActivity("walking", concurrent_with=[])
    b = _StubActivity("reading", concurrent_with=[])
    assert _fallback_for(a, b) == R_SEP


def test_get_concurrent_flag_reads_scheduled_task_inner_task():
    """For a `ScheduledTask`, the inner `task.is_concurrent` is returned."""
    inner = RecommendedTask(
        label="walk",
        duration_min=10,
        duration_max=20,
        difficulty_level=2,
        is_concurrent=True,
    )
    st = ScheduledTask(
        task=inner,
        start_minutes=0,
        end_minutes=10,
        is_standalone=True,
        concurrent_with=None,
        date=DATE,
    )
    assert _get_concurrent_flag(st) is True


def test_get_concurrent_flag_returns_false_for_object_without_flag():
    """Objects with no `is_concurrent` attribute return False."""

    class _Bare:
        pass

    assert _get_concurrent_flag(_Bare()) is False


def test_build_rule_index_skips_predicate_only_rule():
    """A rule with a predicate-only `event_a` is omitted from the legacy index."""
    keep = AllenPairRule(
        id="literal",
        event_a=SelectorPredicate(name="lunch"),
        event_b=SelectorPredicate(name="walking"),
        admissible_relations=["p"],
    )
    drop = AllenPairRule(
        id="predicate_only",
        event_a=SelectorPredicate(intensity=[3, 4]),
        event_b=SelectorPredicate(name="walking"),
        admissible_relations=["p"],
    )
    index = _build_rule_index([keep, drop])
    assert frozenset({"lunch", "walking"}) in index
    assert all(
        key != frozenset({None, "walking"}) for key in index  # type: ignore[arg-type]
    )
