"""z3 second-pass placer for contexts: LTL rules span events and contexts."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.config.schema import (
    ContextCategory,
    ContextMember,
    DurationRange,
    EpisodeRange,
    JitterConfig,
    TemporalPattern,
    TemporalRule,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.context.resolver import ResolvedCategory
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.solver.context_day_model import (
    build_context_day_model,
    extract_context_episodes,
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


def _person(persona_id: str = "p1", **chars) -> Person:
    return Person(
        person_id="p_0001",
        persona_id=persona_id,
        person_seed=1,
        instance_index=0,
        characteristics=dict(chars),
        jitter_applied=JitterConfig(),
    )


def _member(
    *,
    per_min: int = 30,
    per_max: int = 60,
    total_min: int = 30,
    total_max: int = 60,
    eps_min: int = 1,
    eps_max: int = 1,
    windows: list[str] | None = None,
    dimension: str | None = None,
    polarity: str | None = None,
) -> ContextMember:
    return ContextMember(
        per_event_duration=DurationRange(min=per_min, max=per_max, unit="minutes"),
        total_event_duration=TotalDuration(
            min=total_min, max=total_max, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=eps_min, max=eps_max),
        temporal_patterns=(
            [TemporalPattern(mode="fix", details={"within": windows})]
            if windows
            else []
        ),
        dimension=dimension,
        polarity=polarity,
    )


def _resolve(
    name: str,
    members: dict[str, ContextMember],
    *,
    mutually_exclusive: bool = False,
) -> ResolvedCategory:
    return ResolvedCategory(
        name=name, mutually_exclusive=mutually_exclusive, members=members
    )


# -------------------------------------------------------------------------------------
# ------------------------------ baseline placement -----------------------------------
# -------------------------------------------------------------------------------------


def test_z3_places_single_context_member_in_its_fix_window():
    resolved = {
        "mood": _resolve(
            "mood", {"calm": _member(per_min=30, per_max=30, windows=["evening"])}
        )
    }
    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={},
        window_map=_wm(),
        person=_person(),
    )
    import z3

    assert solver.check() == z3.sat
    episodes = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"calm": "mood"},
        day_date=_dt.date(2026, 6, 1),
    )
    assert len(episodes) == 1
    ep = episodes[0]
    assert 960 <= ep.start_minutes
    assert ep.end_minutes <= 1260


def test_z3_pairwise_non_overlap_among_same_member_episodes():
    """Two episodes of the same member must not overlap."""
    resolved = {
        "mood": _resolve(
            "mood",
            {
                "calm": _member(
                    per_min=30,
                    per_max=60,
                    total_min=60,
                    total_max=180,
                    eps_min=2,
                    eps_max=2,
                    windows=["evening"],
                )
            },
        )
    }
    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={},
        window_map=_wm(),
        person=_person(),
    )
    import z3

    assert solver.check() == z3.sat
    eps = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"calm": "mood"},
        day_date=_dt.date(2026, 6, 1),
    )
    a, b = sorted(eps, key=lambda e: e.start_minutes)
    assert a.end_minutes <= b.start_minutes


def test_z3_mutual_exclusion_within_category():
    """`mutually_exclusive: true` forbids overlap between members in one category."""
    resolved = {
        "energy": _resolve(
            "energy",
            {
                "energetic": _member(per_min=30, per_max=30, windows=["evening"]),
                "tired": _member(per_min=30, per_max=30, windows=["evening"]),
            },
            mutually_exclusive=True,
        )
    }
    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={},
        window_map=_wm(),
        person=_person(),
    )
    import z3

    assert solver.check() == z3.sat
    eps = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"energetic": "energy", "tired": "energy"},
        day_date=_dt.date(2026, 6, 1),
    )
    a, b = sorted(eps, key=lambda e: e.start_minutes)
    assert a.end_minutes <= b.start_minutes


def test_z3_dimension_exclusion_across_members():
    """Same-dimension trait pairs do not overlap."""
    resolved = {
        "trait": _resolve(
            "trait",
            {
                "extra": _member(
                    per_min=60,
                    per_max=60,
                    windows=["evening"],
                    dimension="extraversion",
                    polarity="high",
                ),
                "intro": _member(
                    per_min=60,
                    per_max=60,
                    windows=["evening"],
                    dimension="extraversion",
                    polarity="low",
                ),
            },
        )
    }
    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={},
        window_map=_wm(),
        person=_person(),
    )
    import z3

    assert solver.check() == z3.sat
    eps = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"extra": "trait", "intro": "trait"},
        day_date=_dt.date(2026, 6, 1),
    )
    a, b = sorted(eps, key=lambda e: e.start_minutes)
    assert a.end_minutes <= b.start_minutes


# -------------------------------------------------------------------------------------
# ------------------------------ LTL across event + context ---------------------------
# -------------------------------------------------------------------------------------


def test_no_overlap_event_context_enforced_by_z3():
    """`G ¬(swimming ∧ context:home)` keeps the home context off the swim block."""
    resolved = {
        "location": _resolve(
            "location",
            {"home": _member(per_min=60, per_max=60, total_min=60, total_max=60)},
        )
    }
    rule = TemporalRule(id="r", formula="G ¬(swimming ∧ context:home)")
    import z3

    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Sat",
        day_date=_dt.date(2026, 6, 6),
        resolved=resolved,
        event_placements={"swimming": [(450, 60)]},
        window_map=_wm(),
        person=_person(),
        ltl_rules=[rule],
    )
    assert solver.check() == z3.sat
    eps = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"home": "location"},
        day_date=_dt.date(2026, 6, 6),
    )
    ep = eps[0]
    # Either ends before 450 or starts at/after 510.
    assert ep.end_minutes <= 450 or ep.start_minutes >= 510


def test_min_overlap_event_context_enforced_by_z3():
    """`G (gym → F (gym ∧ context:self_efficacy))` forces overlap with the gym block."""
    resolved = {
        "belief": _resolve(
            "belief",
            {
                "self_efficacy": _member(
                    per_min=30, per_max=60, total_min=30, total_max=60
                )
            },
        )
    }
    rule = TemporalRule(id="r", formula="G (gym → F (gym ∧ context:self_efficacy))")
    import z3

    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={"gym": [(1000, 90)]},
        window_map=_wm(),
        person=_person(),
        ltl_rules=[rule],
    )
    assert solver.check() == z3.sat
    eps = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"self_efficacy": "belief"},
        day_date=_dt.date(2026, 6, 1),
    )
    ep = eps[0]
    # Must overlap [1000, 1090).
    assert ep.start_minutes < 1090 and ep.end_minutes > 1000


def test_eventual_co_occur_event_context_enforced_by_z3():
    """`F (running ∧ context:intention)` forces a same-time overlap on this day."""
    resolved = {
        "intent": _resolve(
            "intent",
            {"intention": _member(per_min=15, per_max=30, total_min=15, total_max=30)},
        )
    }
    rule = TemporalRule(id="r", formula="F (running ∧ context:intention)")
    import z3

    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Tue",
        day_date=_dt.date(2026, 6, 2),
        resolved=resolved,
        event_placements={"running": [(470, 30)]},
        window_map=_wm(),
        person=_person(),
        ltl_rules=[rule],
    )
    assert solver.check() == z3.sat
    eps = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"intention": "intent"},
        day_date=_dt.date(2026, 6, 2),
    )
    ep = eps[0]
    assert ep.start_minutes < 500 and ep.end_minutes > 470


def test_implies_future_event_context_enforced_by_z3():
    """`G (gym → F context:tired)` makes tired start after gym ends."""
    resolved = {
        "energy": _resolve(
            "energy",
            {"tired": _member(per_min=30, per_max=30, total_min=30, total_max=30)},
        )
    }
    rule = TemporalRule(id="r", formula="G (gym → F context:tired)")
    import z3

    solver, ctx_vars, members = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={"gym": [(1000, 90)]},
        window_map=_wm(),
        person=_person(),
        ltl_rules=[rule],
    )
    assert solver.check() == z3.sat
    eps = extract_context_episodes(
        solver.model(),
        ctx_vars,
        members,
        category_by_member={"tired": "energy"},
        day_date=_dt.date(2026, 6, 1),
    )
    ep = eps[0]
    assert ep.start_minutes >= 1090


def test_infeasible_ltl_returns_unsat_so_caller_can_relax():
    """A context restricted to evening cannot overlap a morning event."""
    resolved = {
        "energy": _resolve(
            "energy",
            {
                "energetic": _member(
                    per_min=30,
                    per_max=30,
                    total_min=30,
                    total_max=30,
                    windows=["night"],
                )
            },
        )
    }
    rule = TemporalRule(id="r", formula="G (running → F (running ∧ context:energetic))")
    import z3

    solver, _, _ = build_context_day_model(
        day_idx=0,
        weekday="Tue",
        day_date=_dt.date(2026, 6, 2),
        resolved=resolved,
        event_placements={"running": [(470, 30)]},
        window_map=_wm(),
        person=_person(),
        ltl_rules=[rule],
    )
    assert solver.check() == z3.unsat


def test_plan_contexts_for_day_falls_back_to_greedy_when_z3_unsat():
    """When Z3 cannot satisfy LTL, plan_contexts_for_day uses the greedy placer."""
    from src.scripts.persona.context.planner import plan_contexts_for_day
    from src.scripts.persona.domain.event import EventInstance
    from src.scripts.persona.domain.schedule import DaySchedule

    resolved = {
        "energy": _resolve(
            "energy",
            {
                "energetic": _member(
                    per_min=30,
                    per_max=30,
                    total_min=30,
                    total_max=30,
                    windows=["night"],
                )
            },
        )
    }
    impossible = TemporalRule(
        id="r", formula="G (running → F (running ∧ context:energetic))"
    )
    day = DaySchedule(
        day_index=0,
        date=_dt.date(2026, 6, 2),
        weekday="Tue",
        events={"running": [EventInstance("running", 470, 30)]},
    )
    out = plan_contexts_for_day(_person(), day, resolved, _wm(), ltl_rules=[impossible])
    # Greedy fallback still produces an episode in the night window.
    assert len(out) == 1
    assert out[0].start_minutes >= 1260


def test_plan_contexts_for_day_uses_z3_when_ltl_satisfiable():
    """With a satisfiable LTL, plan_contexts_for_day produces a Z3-aligned placement."""
    from src.scripts.persona.context.planner import plan_contexts_for_day
    from src.scripts.persona.domain.event import EventInstance
    from src.scripts.persona.domain.schedule import DaySchedule

    resolved = {
        "belief": _resolve(
            "belief",
            {
                "self_efficacy": _member(
                    per_min=30, per_max=60, total_min=30, total_max=60
                )
            },
        )
    }
    rule = TemporalRule(id="r", formula="G (gym → F (gym ∧ context:self_efficacy))")
    day = DaySchedule(
        day_index=0,
        date=_dt.date(2026, 6, 1),
        weekday="Mon",
        events={"gym": [EventInstance("gym", 1000, 90)]},
    )
    out = plan_contexts_for_day(_person(), day, resolved, _wm(), ltl_rules=[rule])
    assert len(out) == 1
    ep = out[0]
    assert ep.start_minutes < 1090 and ep.end_minutes > 1000


def test_zero_episode_count_skipped_silently():
    """An eps_min=0 / eps_max=0 member produces no Z3 vars at all."""
    resolved = {
        "mood": _resolve(
            "mood",
            {"calm": _member(per_min=30, per_max=30, eps_min=0, eps_max=0)},
        )
    }
    import z3

    solver, ctx_vars, _ = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={},
        window_map=_wm(),
        person=_person(),
    )
    assert solver.check() == z3.sat
    assert ctx_vars == {}


def test_zero_max_duration_skipped_silently():
    """A member with per_event_duration max==0 is skipped."""
    resolved = {
        "mood": _resolve(
            "mood",
            {"calm": _member(per_min=0, per_max=0, total_min=0, total_max=0)},
        )
    }
    import z3

    solver, ctx_vars, _ = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={},
        window_map=_wm(),
        person=_person(),
    )
    assert solver.check() == z3.sat
    assert ctx_vars == {}


def test_disabled_weekday_skipped():
    """A `fix: {within: [Tue]}` member produces no vars on Monday."""
    resolved = {
        "mood": _resolve(
            "mood",
            {
                "calm": _member(
                    per_min=30,
                    per_max=30,
                    windows=["Tue"],
                )
            },
        )
    }
    import z3

    solver, ctx_vars, _ = build_context_day_model(
        day_idx=0,
        weekday="Mon",
        day_date=_dt.date(2026, 6, 1),
        resolved=resolved,
        event_placements={},
        window_map=_wm(),
        person=_person(),
    )
    assert solver.check() == z3.sat
    assert ctx_vars == {}


# -------------------------------------------------------------------------------------
# ------------------------------ shape (2) in event Z3 phase --------------------------
# -------------------------------------------------------------------------------------


def test_eventual_co_occur_between_two_events_in_build_day_model():
    """`F (A AND B)` placed in the event Z3 phase forces an overlap on this day."""
    import z3

    from src.scripts.persona.solver.day_model import build_day_model

    a = _ev_def(name="a", windows=["morning"])
    b = _ev_def(name="b", windows=["morning"])
    rule = TemporalRule(id="r", formula="F (a ∧ b)")
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
    model = solver.model()
    a_s, a_d = ev["a"][0]
    b_s, b_d = ev["b"][0]
    a_start = model[a_s].as_long()
    a_dur = model[a_d].as_long()
    b_start = model[b_s].as_long()
    b_dur = model[b_d].as_long()
    assert a_start < b_start + b_dur and a_start + a_dur > b_start


def _ev_def(
    *,
    name: str,
    per_min: int = 30,
    per_max: int = 60,
    total_min: int = 30,
    total_max: int = 60,
    eps_min: int = 1,
    eps_max: int = 1,
    windows: list[str] | None = None,
):
    from src.scripts.persona.config.schema import EventDefinition

    return EventDefinition(
        name=name,
        category="x",
        per_event_duration=DurationRange(min=per_min, max=per_max, unit="minutes"),
        total_event_duration=TotalDuration(
            min=total_min, max=total_max, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=eps_min, max=eps_max),
        temporal_patterns=(
            [TemporalPattern(mode="fix", details={"within": windows})]
            if windows
            else []
        ),
    )
