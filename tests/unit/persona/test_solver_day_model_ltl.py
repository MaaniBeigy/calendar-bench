"""LTL rules thread into build_day_model as Z3 constraints."""

from __future__ import annotations

import z3

from src.scripts.persona.config.schema import (
    DurationRange,
    EpisodeRange,
    EventDefinition,
    JitterConfig,
    TemporalPattern,
    TemporalRule,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.solver.day_model import (
    _add_ltl_constraints,
    _concurrent_with_pairs,
    _ltl_rule_applies_to,
    _parse_event_ltl,
    build_day_model,
)


def _wm() -> WindowMap:
    return WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        }
    )


def _event(
    name: str,
    *,
    per_min: int = 30,
    per_max: int = 60,
    total_min: int = 30,
    total_max: int = 60,
    eps_min: int = 1,
    eps_max: int = 1,
    windows: list[str] | None = None,
    concurrent_with: list[str] | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category="x",
        per_event_duration=DurationRange(min=per_min, max=per_max, unit="minutes"),
        total_event_duration=TotalDuration(
            min=total_min, max=total_max, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=eps_min, max=eps_max),
        concurrent_with=concurrent_with or [],
        temporal_patterns=(
            [TemporalPattern(mode="fix", details={"within": windows})]
            if windows
            else []
        ),
    )


def _person(persona_id: str = "p1", **chars) -> Person:
    return Person(
        person_id="p_0001",
        persona_id=persona_id,
        person_seed=1,
        instance_index=0,
        characteristics=dict(chars),
        jitter_applied=JitterConfig(),
    )


def _read(model: z3.ModelRef, ev_vars: dict, name: str) -> list[tuple[int, int]]:
    return [(model[s].as_long(), model[d].as_long()) for s, d in ev_vars[name]]


# -------------------------------------------------------------------------------------
# ----------------------------------- parser tests ------------------------------------
# -------------------------------------------------------------------------------------


def test_parse_event_ltl_no_overlap_shape():
    assert _parse_event_ltl("G ¬(a ∧ b)") == ("no_overlap", "a", "b")


def test_parse_event_ltl_min_overlap_shape():
    assert _parse_event_ltl("G (a → F (a ∧ b))") == ("min_overlap", "a", "b")


def test_parse_event_ltl_min_overlap_with_head_mismatch_rejected():
    assert _parse_event_ltl("G (a → F (c ∧ b))") is None


def test_parse_event_ltl_implies_future_shape():
    assert _parse_event_ltl("G (a → F b)") == ("implies_future", "a", "b")


def test_parse_event_ltl_eventual_co_occur_shape():
    """Shape (2): `F (A AND B)` parses to the eventual-co-occur kind."""
    assert _parse_event_ltl("F (a ∧ b)") == ("eventual_co_occur", "a", "b")


def test_parse_event_ltl_returns_none_for_other_shapes():
    assert _parse_event_ltl("weekly_count(a, ≥, 1)") is None
    assert _parse_event_ltl("garbage") is None


# -------------------------------------------------------------------------------------
# -------------------- concurrent_with overlap permission -----------------------------
# -------------------------------------------------------------------------------------


def _slot_wm() -> WindowMap:
    """Window map with a 10-min `slot` so two events share the only start minute."""
    return WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
            "slot": WindowRange(start=400, end=410),
        }
    )


def test_concurrent_with_pairs_symmetric_and_ignores_self():
    events = {
        "a": _event("a"),
        "b": _event("b", concurrent_with=["a"]),
        "c": _event("c", concurrent_with=["c"]),
    }
    assert _concurrent_with_pairs(events) == {frozenset({"a", "b"})}


def test_non_overlap_forced_without_concurrent_with():
    """Two 120-min events sharing the only start minute cannot be placed disjoint."""
    events = {
        "a": _event("a", per_min=120, per_max=120, total_min=120, total_max=120, windows=["slot"]),
        "b": _event("b", per_min=120, per_max=120, total_min=120, total_max=120, windows=["slot"]),
    }
    solver, _ = build_day_model(
        day_idx=0,
        total_days=1,
        events=events,
        counts={"a": 1, "b": 1},
        window_map=_slot_wm(),
        maximize_event=None,
    )
    assert solver.check() == z3.unsat


def test_concurrent_with_permits_overlap():
    """Listing the other in concurrent_with exempts the pair from non-overlap."""
    events = {
        "a": _event("a", per_min=120, per_max=120, total_min=120, total_max=120, windows=["slot"]),
        "b": _event(
            "b",
            per_min=120,
            per_max=120,
            total_min=120,
            total_max=120,
            windows=["slot"],
            concurrent_with=["a"],
        ),
    }
    solver, _ = build_day_model(
        day_idx=0,
        total_days=1,
        events=events,
        counts={"a": 1, "b": 1},
        window_map=_slot_wm(),
        maximize_event=None,
    )
    assert solver.check() == z3.sat


# -------------------------------------------------------------------------------------
# -------------------------------- applies_to filter ----------------------------------
# -------------------------------------------------------------------------------------


def test_applies_to_empty_matches_everyone():
    assert _ltl_rule_applies_to(_person(), {}) is True


def test_applies_to_personas_admits_listed_id():
    assert (
        _ltl_rule_applies_to(_person(persona_id="p1"), {"personas": ["p1", "p2"]})
        is True
    )


def test_applies_to_personas_rejects_unlisted_id():
    assert _ltl_rule_applies_to(_person(persona_id="p3"), {"personas": ["p1"]}) is False


def test_applies_to_characteristic_matches_value():
    person = _person(occupation_status="fulltime")
    assert _ltl_rule_applies_to(person, {"occupation_status": ["fulltime"]}) is True


def test_applies_to_characteristic_rejects_other_value():
    person = _person(occupation_status="student")
    assert _ltl_rule_applies_to(person, {"occupation_status": ["fulltime"]}) is False


def test_applies_to_characteristic_value_compared_case_insensitive():
    person = _person(weekday="MONDAY")
    assert _ltl_rule_applies_to(person, {"weekday": ["monday"]}) is True


def test_applies_to_missing_characteristic_rejects():
    assert _ltl_rule_applies_to(_person(), {"age": [30]}) is False


def test_applies_to_with_none_person_rejects_filtered_rule():
    assert _ltl_rule_applies_to(None, {"personas": ["p1"]}) is False


# -------------------------------------------------------------------------------------
# ------------------------------- _add_ltl_constraints --------------------------------
# -------------------------------------------------------------------------------------


def test_add_ltl_skips_when_rule_does_not_match_person():
    person = _person(persona_id="p1")
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": _event("a", windows=["morning"])},
        counts={"a": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rule = TemporalRule(
        id="r", formula="G ¬(a ∧ a)", applies_to={"personas": ["other"]}
    )
    _add_ltl_constraints(solver, ev, [rule], person=person)
    assert len(list(solver.assertions())) == before


def test_add_ltl_skips_unparseable_formula():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": _event("a", windows=["morning"])},
        counts={"a": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rule = TemporalRule(id="r", formula="not a valid formula")
    _add_ltl_constraints(solver, ev, [rule])
    assert len(list(solver.assertions())) == before


def test_add_ltl_skips_context_atom_rules():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": _event("a", windows=["morning"])},
        counts={"a": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rule = TemporalRule(id="r", formula="G ¬(a ∧ context:tired)")
    _add_ltl_constraints(solver, ev, [rule])
    assert len(list(solver.assertions())) == before


def test_add_ltl_eventual_co_occur_emits_pair_overlap_or():
    """`F (A AND B)` adds an Or-of-overlap clause across every (a, b) pair on the day."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={
            "a": _event("a", windows=["morning"]),
            "b": _event("b", windows=["morning"]),
        },
        counts={"a": 1, "b": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rule = TemporalRule(id="r", formula="F (a ∧ b)")
    _add_ltl_constraints(solver, ev, [rule])
    after = list(solver.assertions())
    assert len(after) > before
    # The added clause names the overlap predicate for the single (a, b) pair.
    new_text = "\n".join(str(a) for a in after[before:])
    assert "And(" in new_text
    assert "a_start" in new_text and "b_start" in new_text


def test_add_ltl_eventual_co_occur_followed_by_another_rule():
    """A rule after `F (A AND B)` is still processed, so the loop continues."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={
            "a": _event("a", windows=["morning"]),
            "b": _event("b", windows=["morning"]),
            "c": _event("c", windows=["morning"]),
        },
        counts={"a": 1, "b": 1, "c": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rules = [
        TemporalRule(id="r1", formula="F (a ∧ b)"),
        TemporalRule(id="r2", formula="G ¬(b ∧ c)"),
    ]
    _add_ltl_constraints(solver, ev, rules)
    new_text = "\n".join(str(x) for x in list(solver.assertions())[before:])
    assert "a_start" in new_text and "c_start" in new_text


def test_overlap_allowed_pairs_empty_when_rules_is_none():
    """`_ltl_overlap_allowed_pairs(None, ...)` returns an empty set."""
    from src.scripts.persona.solver.day_model import _ltl_overlap_allowed_pairs

    assert _ltl_overlap_allowed_pairs(None, _person()) == set()


def test_overlap_allowed_pairs_skips_rule_filtered_by_persona():
    """An applies_to mismatch drops the rule from the overlap-allowed set."""
    from src.scripts.persona.solver.day_model import _ltl_overlap_allowed_pairs

    rule = TemporalRule(
        id="r",
        formula="G (a → F (a ∧ b))",
        applies_to={"personas": ["other"]},
    )
    assert _ltl_overlap_allowed_pairs([rule], _person(persona_id="me")) == set()


def test_overlap_allowed_pairs_skips_unparseable_formula():
    """Garbage formulas don't enter the overlap-allowed set."""
    from src.scripts.persona.solver.day_model import _ltl_overlap_allowed_pairs

    rule = TemporalRule(id="r", formula="bogus")
    assert _ltl_overlap_allowed_pairs([rule], _person()) == set()


def test_overlap_allowed_pairs_skips_non_min_overlap_rules():
    """no_overlap and implies_future rules are NOT added to the overlap set."""
    from src.scripts.persona.solver.day_model import _ltl_overlap_allowed_pairs

    rule_no = TemporalRule(id="r1", formula="G ¬(a ∧ b)")
    rule_if = TemporalRule(id="r2", formula="G (a → F b)")
    assert _ltl_overlap_allowed_pairs([rule_no, rule_if], _person()) == set()


def test_overlap_allowed_pairs_includes_context_atom_pair():
    """Event/context overlap rules now feed the unified atom-pair set."""
    from src.scripts.persona.solver.day_model import _ltl_overlap_allowed_pairs

    rule = TemporalRule(id="r", formula="G (a → F (a ∧ context:happy))")
    assert _ltl_overlap_allowed_pairs([rule], _person()) == {
        frozenset({"a", "context:happy"})
    }


def test_overlap_allowed_pairs_includes_eventual_co_occur_pair():
    """Shape (2) `F (A AND B)` adds the pair to the overlap-allowed set."""
    from src.scripts.persona.solver.day_model import _ltl_overlap_allowed_pairs

    rule = TemporalRule(id="r", formula="F (a ∧ b)")
    assert _ltl_overlap_allowed_pairs([rule], _person()) == {frozenset({"a", "b"})}


def test_overlap_allowed_pairs_includes_min_overlap_pair():
    """A valid `min_overlap` rule adds its two event names as an unordered pair."""
    from src.scripts.persona.solver.day_model import _ltl_overlap_allowed_pairs

    rule = TemporalRule(id="r", formula="G (a → F (a ∧ b))")
    assert _ltl_overlap_allowed_pairs([rule], _person()) == {frozenset({"a", "b"})}


def test_add_ltl_constraints_iterates_multiple_rules():
    """Loop body exercises rule-N continuation back to the outer for."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={
            "a": _event("a", windows=["morning"]),
            "b": _event("b", windows=["afternoon"]),
        },
        counts={"a": 1, "b": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rules = [
        TemporalRule(id="r1", formula="G ¬(a ∧ b)"),
        TemporalRule(id="r2", formula="G (a → F b)"),
    ]
    _add_ltl_constraints(solver, ev, rules)
    assert len(list(solver.assertions())) > before


def test_add_ltl_constraints_min_overlap_skipped_when_context_atom_involved():
    """A context-atom min_overlap rule emits no Z3 constraints inside _add_ltl_constraints."""
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": _event("a", windows=["morning"])},
        counts={"a": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rule = TemporalRule(id="r", formula="G (a → F (a ∧ context:happy))")
    _add_ltl_constraints(solver, ev, [rule])
    assert len(list(solver.assertions())) == before


def test_add_ltl_skips_when_referenced_event_is_absent():
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": _event("a", windows=["morning"])},
        counts={"a": 1},
        window_map=_wm(),
        maximize_event=None,
    )
    before = len(list(solver.assertions()))
    rule = TemporalRule(id="r", formula="G ¬(a ∧ ghost)")
    _add_ltl_constraints(solver, ev, [rule])
    assert len(list(solver.assertions())) == before


# -------------------------------------------------------------------------------------
# --------------------------- end-to-end behavioural tests ----------------------------
# -------------------------------------------------------------------------------------


def test_no_overlap_rule_separates_events_via_z3():
    """`G ¬(a ∧ b)` keeps two co-located events apart inside the same window."""
    morning = _event("a", windows=["morning"])
    afternoon = _event(
        "b",
        per_min=30,
        per_max=60,
        total_min=30,
        total_max=60,
        windows=["morning"],
    )
    rule = TemporalRule(id="ab", formula="G ¬(a ∧ b)")
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": morning, "b": afternoon},
        counts={"a": 1, "b": 1},
        window_map=_wm(),
        maximize_event=None,
        ltl_rules=[rule],
    )
    assert solver.check() == z3.sat
    m = solver.model()
    a_start, a_dur = _read(m, ev, "a")[0]
    b_start, b_dur = _read(m, ev, "b")[0]
    assert a_start + a_dur <= b_start or b_start + b_dur <= a_start


def test_implies_future_rule_orders_events():
    """`G (a -> F b)` forces b to start after a ends."""
    a = _event(
        "a",
        per_min=30,
        per_max=30,
        total_min=30,
        total_max=30,
        windows=["morning"],
    )
    b = _event(
        "b",
        per_min=30,
        per_max=30,
        total_min=30,
        total_max=30,
        windows=["morning"],
    )
    rule = TemporalRule(id="ab", formula="G (a → F b)")
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": a, "b": b},
        counts={"a": 1, "b": 1},
        window_map=_wm(),
        maximize_event=None,
        ltl_rules=[rule],
    )
    assert solver.check() == z3.sat
    m = solver.model()
    a_start, a_dur = _read(m, ev, "a")[0]
    b_start, _ = _read(m, ev, "b")[0]
    assert b_start >= a_start + a_dur


def test_min_overlap_rule_forces_overlap():
    """`G (a -> F (a AND b))` forces a and b to overlap."""
    a = _event(
        "a",
        per_min=30,
        per_max=30,
        total_min=30,
        total_max=30,
        windows=["morning"],
    )
    b = _event(
        "b",
        per_min=60,
        per_max=60,
        total_min=60,
        total_max=60,
        windows=["morning"],
    )
    rule = TemporalRule(id="ab", formula="G (a → F (a ∧ b))")
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": a, "b": b},
        counts={"a": 1, "b": 1},
        window_map=_wm(),
        maximize_event=None,
        ltl_rules=[rule],
    )
    assert solver.check() == z3.sat
    m = solver.model()
    a_start, a_dur = _read(m, ev, "a")[0]
    b_start, b_dur = _read(m, ev, "b")[0]
    assert a_start < b_start + b_dur and a_start + a_dur > b_start


def test_infeasible_ltl_returns_unsat_so_caller_can_relax():
    """An impossible LTL pair returns unsat; horizon.py's retry-relax handles fallback."""
    sleep = _event(
        "sleep",
        per_min=480,
        per_max=480,
        total_min=480,
        total_max=480,
        windows=["night"],
    )
    work = _event(
        "work",
        per_min=120,
        per_max=120,
        total_min=120,
        total_max=120,
        windows=["morning"],
    )
    # Both windows are disjoint; an overlap requirement is unsatisfiable.
    rule = TemporalRule(id="imp", formula="G (work → F (work ∧ sleep))")
    solver, _ = build_day_model(
        day_idx=0,
        total_days=1,
        events={"sleep": sleep, "work": work},
        counts={"sleep": 1, "work": 1},
        window_map=_wm(),
        maximize_event=None,
        ltl_rules=[rule],
    )
    assert solver.check() == z3.unsat


def test_rule_filtered_by_persona_does_not_constrain_solver():
    """A rule scoped to other personas leaves placement unchanged."""
    a = _event(
        "a",
        per_min=30,
        per_max=30,
        total_min=30,
        total_max=30,
        windows=["morning"],
    )
    b = _event(
        "b",
        per_min=120,
        per_max=120,
        total_min=120,
        total_max=120,
        windows=["morning"],
    )
    rule = TemporalRule(
        id="imp",
        formula="G (a → F (a ∧ b))",
        applies_to={"personas": ["someone_else"]},
    )
    person = _person(persona_id="active_person")
    solver, ev = build_day_model(
        day_idx=0,
        total_days=1,
        events={"a": a, "b": b},
        counts={"a": 1, "b": 1},
        window_map=_wm(),
        maximize_event=None,
        ltl_rules=[rule],
        person=person,
    )
    # The day is sat without the overlap constraint; the rule filtered out.
    assert solver.check() == z3.sat


def test_horizon_retry_relax_drops_ltl_when_day_is_infeasible():
    """An infeasible LTL set falls through to the no-LTL attempt and the day still solves."""
    from src.scripts.persona.planner.horizon import _solve_one_day

    events = {
        "sleep": _event(
            "sleep",
            per_min=420,
            per_max=420,
            total_min=420,
            total_max=420,
            windows=["night"],
        ),
        "work": _event(
            "work",
            per_min=120,
            per_max=120,
            total_min=120,
            total_max=120,
            windows=["morning"],
        ),
    }
    impossible = TemporalRule(id="imp", formula="G (work → F (work ∧ sleep))")
    day_events, _ = _solve_one_day(
        day_idx=0,
        horizon_days=1,
        events=events,
        counts={"sleep": 1, "work": 1},
        window_map=_wm(),
        occupied=[],
        step_minutes=10,
        maximize=None,
        ltl_rules=[impossible],
        person=_person(),
    )
    # The first three attempts return unsat under the impossible LTL;
    # the fourth attempt drops LTL and produces a schedule containing
    # both events.
    assert day_events.get("sleep")
    assert day_events.get("work")
