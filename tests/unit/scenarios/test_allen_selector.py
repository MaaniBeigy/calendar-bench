"""Unit tests for the selector-aware admissible-set machinery in allen.py.

Covers the selector matching and admissible-set rules:
* SelectorMatcher matches predicates against ResolvedActivity / RecommendedTask /
  CalendarEvent / ScheduledTask using the unified intensity scale.
* RuleSet.build_admissible intersects matching rules and respects the
  no-rule fallback chain (concurrent_with, is_concurrent, R_SEP).
* compute_buffer_violation handles all three regimes (admissible, hard,
  partial).
"""

from __future__ import annotations

import datetime

import pytest

from src.scripts.persona.config.schema import AllenPairRule, SelectorPredicate
from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import (
    R_ALL,
    R_BUFFER,
    R_MERGE,
    R_SEP,
    AllenRelation,
    ResolvedActivity,
    RuleSet,
    SelectorMatcher,
    build_admissible_pair,
    compute_buffer_violation,
)
from src.scripts.scenarios.metrics.intensity_resolver import (
    IntensityResolver,
    MetQuartiles,
)

DEFAULT_QUARTILES = MetQuartiles(q1=1.8, q2=3.0, q3=6.0, max_met=16.8)
DATE = datetime.date(2026, 5, 1)


def _resolved(label, intensity=None, met=None, domain=None) -> ResolvedActivity:
    return ResolvedActivity(label=label, intensity=intensity, met=met, domain=domain)


# ---------------------------------------------------------------------------
# SelectorMatcher.matches
# ---------------------------------------------------------------------------


class TestSelectorMatcherMatches:
    def test_name_match(self):
        m = SelectorMatcher()
        assert m.matches(SelectorPredicate(name="lunch"), _resolved("lunch"))
        assert not m.matches(SelectorPredicate(name="lunch"), _resolved("dinner"))

    def test_name_list(self):
        m = SelectorMatcher()
        sel = SelectorPredicate(name=["lunch", "dinner"])
        assert m.matches(sel, _resolved("lunch"))
        assert m.matches(sel, _resolved("dinner"))
        assert not m.matches(sel, _resolved("breakfast"))

    def test_intensity_in_list(self):
        m = SelectorMatcher()
        sel = SelectorPredicate(intensity=[2, 3, 4])
        assert m.matches(sel, _resolved("x", intensity=3))
        assert not m.matches(sel, _resolved("x", intensity=1))

    def test_intensity_none_fails(self):
        m = SelectorMatcher()
        sel = SelectorPredicate(intensity=[2, 3, 4])
        assert not m.matches(sel, _resolved("x", intensity=None))

    def test_domain_match(self):
        m = SelectorMatcher()
        sel = SelectorPredicate(domain="NutritionTask")
        assert m.matches(sel, _resolved("x", domain="NutritionTask"))
        assert not m.matches(sel, _resolved("x", domain="PhysicalActivityTask"))

    def test_met_min(self):
        m = SelectorMatcher()
        sel = SelectorPredicate(met_min=3.0)
        assert m.matches(sel, _resolved("x", met=3.5))
        assert m.matches(sel, _resolved("x", met=3.0))
        assert not m.matches(sel, _resolved("x", met=2.5))
        assert not m.matches(sel, _resolved("x", met=None))

    def test_met_max(self):
        m = SelectorMatcher()
        sel = SelectorPredicate(met_max=3.0)
        assert m.matches(sel, _resolved("x", met=2.5))
        assert m.matches(sel, _resolved("x", met=3.0))
        assert not m.matches(sel, _resolved("x", met=3.5))

    def test_combined_predicates_all_must_match(self):
        m = SelectorMatcher()
        sel = SelectorPredicate(name="lunch", intensity=[1])
        assert m.matches(sel, _resolved("lunch", intensity=1))
        assert not m.matches(sel, _resolved("lunch", intensity=2))


# ---------------------------------------------------------------------------
# SelectorMatcher.resolve; projection from heterogeneous activities
# ---------------------------------------------------------------------------


class TestSelectorMatcherResolve:
    def test_resolved_activity_passthrough(self):
        m = SelectorMatcher()
        ra = _resolved("x", intensity=2)
        assert m.resolve(ra) is ra

    def test_calendar_event_picks_intensity_field(self):
        m = SelectorMatcher()
        ev = CalendarEvent(
            label="run", start_minutes=0, end_minutes=60, date=DATE, intensity=4
        )
        ra = m.resolve(ev)
        assert ra.label == "run"
        assert ra.intensity == 4

    def test_calendar_event_intensity_5_collapses_to_4(self):
        m = SelectorMatcher()
        ev = CalendarEvent(
            label="run", start_minutes=0, end_minutes=60, date=DATE, intensity=5
        )
        assert m.resolve(ev).intensity == 4

    def test_recommended_task_with_resolver(self):
        t = RecommendedTask(
            label="cook",
            duration_min=10,
            duration_max=20,
            difficulty_level=3,
        )
        resolver = IntensityResolver(DEFAULT_QUARTILES)
        m = SelectorMatcher(resolver=resolver)
        ra = m.resolve(t)
        assert ra.label == "cook"
        assert ra.intensity == 3

    def test_scheduled_task_resolves(self):
        t = RecommendedTask(
            label="walk",
            duration_min=10,
            duration_max=20,
            difficulty_level=2,
        )
        st = ScheduledTask(
            task=t,
            start_minutes=0,
            end_minutes=10,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
        )
        resolver = IntensityResolver(DEFAULT_QUARTILES)
        m = SelectorMatcher(resolver=resolver)
        assert m.resolve(st).label == "walk"
        assert m.resolve(st).intensity == 2

    def test_unknown_object_returns_empty_label(self):
        m = SelectorMatcher()

        class _X:
            pass

        ra = m.resolve(_X())
        assert ra.label == ""
        assert ra.intensity is None


# ---------------------------------------------------------------------------
# RuleSet.build_admissible
# ---------------------------------------------------------------------------


def _rule(id_, a, b, relations, buffer=None) -> AllenPairRule:
    return AllenPairRule(
        id=id_,
        event_a=a,
        event_b=b,
        admissible_relations=list(relations),
        buffer=buffer,
    )


class TestRuleSetBuildAdmissible:
    def test_no_rules_returns_fallback(self):
        rs = RuleSet([])
        m = SelectorMatcher()
        adm = rs.build_admissible(_resolved("a"), _resolved("b"), m, fallback=R_SEP)
        assert adm == R_SEP

    def test_literal_rule_match_forward(self):
        rs = RuleSet([_rule("r", "a", "b", ["p", "P"])])
        m = SelectorMatcher()
        adm = rs.build_admissible(_resolved("a"), _resolved("b"), m)
        assert adm == frozenset({AllenRelation.p, AllenRelation.P})

    def test_literal_rule_match_backward(self):
        # Rule says (a, b) but we ask (b, a); should still match.
        rs = RuleSet([_rule("r", "a", "b", ["p", "P"])])
        m = SelectorMatcher()
        adm = rs.build_admissible(_resolved("b"), _resolved("a"), m)
        assert adm == frozenset({AllenRelation.p, AllenRelation.P})

    def test_predicate_rule_match(self):
        # intensive vs intensive; predicate-only.
        rs = RuleSet(
            [
                _rule(
                    "intensive",
                    {"intensity": [3, 4]},
                    {"intensity": [3, 4]},
                    ["p", "P"],
                )
            ]
        )
        m = SelectorMatcher()
        adm = rs.build_admissible(
            _resolved("run", intensity=4),
            _resolved("gym", intensity=3),
            m,
        )
        assert adm == frozenset({AllenRelation.p, AllenRelation.P})

    def test_multiple_rule_intersection(self):
        # rule1 admits {p, P, m}; rule2 admits {p, P, M}; intersection is {p, P}.
        rs = RuleSet(
            [
                _rule("r1", "a", "b", ["p", "P", "m"]),
                _rule("r2", "a", "b", ["p", "P", "M"]),
            ]
        )
        m = SelectorMatcher()
        adm = rs.build_admissible(_resolved("a"), _resolved("b"), m)
        assert adm == frozenset({AllenRelation.p, AllenRelation.P})

    def test_no_match_uses_supplied_fallback(self):
        rs = RuleSet([_rule("r", "x", "y", ["p", "P"])])
        m = SelectorMatcher()
        adm = rs.build_admissible(
            _resolved("a"),
            _resolved("b"),
            m,
            fallback=R_ALL,
        )
        assert adm == R_ALL

    def test_caches_repeat_lookups(self):
        rs = RuleSet([_rule("r", "a", "b", ["p", "P"])])
        m = SelectorMatcher()
        first = rs.build_admissible(_resolved("a"), _resolved("b"), m)
        second = rs.build_admissible(_resolved("a"), _resolved("b"), m)
        assert first is second


# ---------------------------------------------------------------------------
# build_admissible_pair; public wrapper with no-rule fallback chain
# ---------------------------------------------------------------------------


class TestBuildAdmissiblePairFallback:
    def test_concurrent_with_hit_gives_r_all(self):
        rs = RuleSet([])
        m = SelectorMatcher()
        ev = CalendarEvent(
            label="lunch",
            start_minutes=720,
            end_minutes=780,
            date=DATE,
            concurrent_with=["reading"],
        )
        t = RecommendedTask(label="reading", duration_min=10, duration_max=20)
        adm = build_admissible_pair(t, ev, rs, m)
        assert adm == R_ALL

    def test_is_concurrent_flag_gives_r_all(self):
        rs = RuleSet([])
        m = SelectorMatcher()
        t = RecommendedTask(
            label="podcast", duration_min=10, duration_max=20, is_concurrent=True
        )
        ev = CalendarEvent(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
        adm = build_admissible_pair(t, ev, rs, m)
        assert adm == R_ALL

    def test_default_returns_r_sep(self):
        rs = RuleSet([])
        m = SelectorMatcher()
        t = RecommendedTask(label="x", duration_min=10, duration_max=20)
        ev = CalendarEvent(label="y", start_minutes=0, end_minutes=10, date=DATE)
        adm = build_admissible_pair(t, ev, rs, m)
        assert adm == R_SEP

    def test_rule_overrides_fallback(self):
        rs = RuleSet([_rule("r", "x", "y", ["p", "P"])])
        m = SelectorMatcher()
        t = RecommendedTask(
            label="x", duration_min=10, duration_max=20, is_concurrent=True
        )
        ev = CalendarEvent(label="y", start_minutes=0, end_minutes=10, date=DATE)
        # Rule fires to return its relation set, not R_ALL via is_concurrent.
        adm = build_admissible_pair(t, ev, rs, m)
        assert adm == frozenset({AllenRelation.p, AllenRelation.P})


# ---------------------------------------------------------------------------
# compute_buffer_violation
# ---------------------------------------------------------------------------


class TestComputeBufferViolation:
    def test_admissible_returns_zero(self):
        adm = frozenset({AllenRelation.p, AllenRelation.P})
        v = compute_buffer_violation(
            AllenRelation.p, gap=100.0, buffer_minutes=30, admissible=adm
        )
        assert v == pytest.approx(0.0)

    def test_meets_is_one(self):
        adm = R_SEP - frozenset({AllenRelation.m, AllenRelation.M})
        v = compute_buffer_violation(
            AllenRelation.m, gap=0.0, buffer_minutes=30, admissible=adm
        )
        assert v == pytest.approx(1.0)

    def test_met_by_is_one(self):
        adm = R_SEP - frozenset({AllenRelation.m, AllenRelation.M})
        v = compute_buffer_violation(
            AllenRelation.M, gap=0.0, buffer_minutes=30, admissible=adm
        )
        assert v == pytest.approx(1.0)

    def test_p_with_gap_above_buffer_zero(self):
        adm = frozenset({AllenRelation.m, AllenRelation.M})  # p NOT admissible
        v = compute_buffer_violation(
            AllenRelation.p, gap=60.0, buffer_minutes=30, admissible=adm
        )
        assert v == pytest.approx(0.0)

    def test_p_with_gap_below_buffer_partial(self):
        adm = frozenset()
        v = compute_buffer_violation(
            AllenRelation.p, gap=15.0, buffer_minutes=30, admissible=adm
        )
        # 1 - 15/30 = 0.5
        assert v == pytest.approx(0.5)

    def test_p_with_gap_at_buffer_zero(self):
        adm = frozenset()
        v = compute_buffer_violation(
            AllenRelation.p, gap=30.0, buffer_minutes=30, admissible=adm
        )
        assert v == pytest.approx(0.0)

    def test_buffer_none_disables_partial(self):
        adm = frozenset()
        v = compute_buffer_violation(
            AllenRelation.p, gap=0.0, buffer_minutes=None, admissible=adm
        )
        assert v == pytest.approx(0.0)

    def test_overlap_not_admissible_is_one(self):
        adm = R_SEP
        v = compute_buffer_violation(
            AllenRelation.o, gap=0.0, buffer_minutes=30, admissible=adm
        )
        assert v == pytest.approx(1.0)

    def test_overlap_admissible_zero(self):
        v = compute_buffer_violation(
            AllenRelation.o, gap=0.0, buffer_minutes=30, admissible=R_ALL
        )
        assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RuleSet.buffer_for
# ---------------------------------------------------------------------------


class TestRuleSetBufferFor:
    def test_default_when_no_match(self):
        rs = RuleSet([])
        m = SelectorMatcher()
        b = rs.buffer_for(_resolved("a"), _resolved("b"), m, default=30)
        assert b == 30

    def test_rule_buffer_wins(self):
        rs = RuleSet([_rule("r", "a", "b", ["p"], buffer=60)])
        m = SelectorMatcher()
        b = rs.buffer_for(_resolved("a"), _resolved("b"), m, default=30)
        assert b == 60

    def test_smallest_buffer_wins_when_multi_match(self):
        rs = RuleSet(
            [
                _rule("r1", "a", "b", ["p"], buffer=60),
                _rule("r2", "a", "b", ["p"], buffer=15),
            ]
        )
        m = SelectorMatcher()
        b = rs.buffer_for(_resolved("a"), _resolved("b"), m, default=30)
        assert b == 15

    def test_rule_without_buffer_falls_back_to_default(self):
        rs = RuleSet([_rule("r", "a", "b", ["p"], buffer=None)])
        m = SelectorMatcher()
        b = rs.buffer_for(_resolved("a"), _resolved("b"), m, default=45)
        assert b == 45

    def test_caches(self):
        rs = RuleSet([_rule("r", "a", "b", ["p"], buffer=60)])
        m = SelectorMatcher()
        first = rs.buffer_for(_resolved("a"), _resolved("b"), m, default=30)
        second = rs.buffer_for(_resolved("a"), _resolved("b"), m, default=30)
        assert first == second
