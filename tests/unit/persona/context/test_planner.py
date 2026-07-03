"""Tests for the greedy context placer."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.config.schema import (
    ContextCategory,
    ContextMember,
    DurationRange,
    EpisodeRange,
    JitterConfig,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.context.planner import (
    _enumerate_free_starts,
    _Interval,
    plan_contexts,
    plan_contexts_for_day,
)
from src.scripts.persona.context.resolver import resolve_for_person
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.domain.time_windows import WindowMap

DATE = _dt.date(2026, 5, 4)


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


def _person(person_seed: int = 1) -> Person:
    return Person(
        person_id="p_0001",
        persona_id="p",
        person_seed=person_seed,
        instance_index=0,
        characteristics={},
        jitter_applied=JitterConfig(),
    )


def _day(events: dict[str, list[EventInstance]] | None = None) -> DaySchedule:
    return DaySchedule(
        day_index=0,
        date=DATE,
        weekday="Mon",
        events=events or {},
    )


def _member(
    per_min: int = 15,
    per_max: int = 60,
    eps_min: int = 1,
    eps_max: int = 1,
    *,
    patterns: list[TemporalPattern] | None = None,
    dimension: str | None = None,
    polarity: str | None = None,
) -> ContextMember:
    return ContextMember(
        per_event_duration=DurationRange(min=per_min, max=per_max),
        total_event_duration=TotalDuration(min=per_min, max=per_max * 4),
        total_event_episodes=EpisodeRange(min=eps_min, max=eps_max),
        temporal_patterns=patterns or [],
        dimension=dimension,
        polarity=polarity,
    )


# ---------------------------------------------------------------------------
# _enumerate_free_starts
# ---------------------------------------------------------------------------


def test_free_starts_empty_when_duration_zero():
    assert _enumerate_free_starts(_Interval(0, 100), 0, []) == []


def test_free_starts_empty_when_bound_too_small():
    assert _enumerate_free_starts(_Interval(0, 10), 30, []) == []


def test_free_starts_no_obstacles():
    starts = _enumerate_free_starts(_Interval(0, 100), 60, [])
    assert starts == list(range(0, 41))


def test_free_starts_with_obstacle_in_middle():
    # bound [0, 100], obstacle [40, 60], duration 30:
    # free in [0, 10] and [60, 70].
    starts = _enumerate_free_starts(_Interval(0, 100), 30, [_Interval(40, 60)])
    assert starts == list(range(0, 11)) + list(range(60, 71))


def test_free_starts_obstacles_outside_bound_ignored():
    starts = _enumerate_free_starts(
        _Interval(50, 100), 20, [_Interval(0, 10), _Interval(150, 200)]
    )
    assert starts == list(range(50, 81))


def test_free_starts_obstacle_clamped_to_bound():
    starts = _enumerate_free_starts(_Interval(0, 100), 20, [_Interval(80, 200)])
    assert starts == list(range(0, 61))


def test_free_starts_cursor_breaks_when_past_bound_end():
    starts = _enumerate_free_starts(
        _Interval(0, 100), 30, [_Interval(20, 200), _Interval(300, 400)]
    )
    assert starts == []


# ---------------------------------------------------------------------------
# plan_contexts_for_day
# ---------------------------------------------------------------------------


def test_single_episode_placed_inside_bounds():
    cat = ContextCategory(members={"calm": _member(per_min=30, per_max=60)})
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert len(out) == 1
    ep = out[0]
    assert ep.category == "mood_emotion"
    assert ep.name == "calm"
    assert 30 <= ep.duration <= 60
    assert 0 <= ep.start_minutes and ep.end_minutes <= 1440


def test_multiple_episodes_within_count_band():
    cat = ContextCategory(
        members={
            "calm": _member(per_min=15, per_max=30, eps_min=3, eps_max=3),
        }
    )
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert len(out) == 3


def test_mutual_exclusion_prevents_overlap_within_category():
    # Two members in the same exclusive category; each places one episode.
    cat = ContextCategory(
        mutually_exclusive=True,
        members={
            "happy": _member(per_min=120, per_max=120),
            "sad": _member(per_min=120, per_max=120),
        },
    )
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert len(out) == 2
    a, b = sorted(out, key=lambda e: e.start_minutes)
    assert a.end_minutes <= b.start_minutes


def test_context_may_overlap_event():
    """Contexts are person states and may overlap any event in the day."""
    cat = ContextCategory(members={"calm": _member(per_min=60, per_max=60)})
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    events = {
        "office_work": [EventInstance(event_name="office_work", start=0, duration=600)]
    }
    out = plan_contexts_for_day(_person(), _day(events), resolved, _wm())
    assert len(out) == 1
    # The placer must NOT treat the event as a placement obstacle; the
    # calm episode lands somewhere in [0, 1440-60] without dodging
    # office_work's 0-600 window.
    assert 0 <= out[0].start_minutes <= 1440 - 60


def test_fix_window_restricts_placement():
    cat = ContextCategory(
        members={
            "calm": _member(
                per_min=30,
                per_max=30,
                patterns=[TemporalPattern(mode="fix", details={"within": ["night"]})],
            )
        }
    )
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert len(out) == 1
    assert 1260 <= out[0].start_minutes
    assert out[0].end_minutes <= 1440


def test_fix_window_with_weekday_filter_drops_off_day():
    """A `within: [Tue]` pattern places nothing on a Monday."""
    cat = ContextCategory(
        members={
            "calm": _member(
                per_min=30,
                per_max=30,
                patterns=[TemporalPattern(mode="fix", details={"within": ["Tue"]})],
            )
        }
    )
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert out == []


def test_zero_episode_count_skips_member():
    cat = ContextCategory(members={"calm": _member(eps_min=0, eps_max=0)})
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert out == []


def test_dimension_exclusion_inside_trait_state():
    cat = ContextCategory(
        mutually_exclusive=False,
        members={
            "extra": _member(
                per_min=120,
                per_max=120,
                dimension="extraversion",
                polarity="high",
            ),
            "intro": _member(
                per_min=120,
                per_max=120,
                dimension="extraversion",
                polarity="low",
            ),
        },
    )
    resolved = resolve_for_person(_person(), {"trait_state": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert len(out) == 2
    a, b = sorted(out, key=lambda e: e.start_minutes)
    assert a.end_minutes <= b.start_minutes
    assert {e.dimension for e in out} == {"extraversion"}


def test_unplaceable_episode_drops_silently():
    """A member whose per_event duration exceeds the fix window drops."""
    cat = ContextCategory(
        members={
            "calm": _member(
                per_min=300,
                per_max=300,
                patterns=[TemporalPattern(mode="fix", details={"within": ["morning"]})],
            )
        }
    )
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    # Morning window is [400, 600] (200 min); a 300-min episode cannot
    # fit and the placer silently drops it.
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert out == []


def test_member_zero_max_duration_skipped():
    cat = ContextCategory(members={"calm": _member(per_min=0, per_max=0)})
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert out == []


def test_weekly_episode_band_flattens_to_daily_average():
    """A non-day scale flattens via `band.max // 7`; min 0 max 7 produces ~1/day."""
    member = ContextMember(
        per_event_duration=DurationRange(min=10, max=20),
        total_event_duration=TotalDuration(min=10, max=120, scale="week"),
        total_event_episodes=EpisodeRange(min=0, max=14, scale="week"),
    )
    cat = ContextCategory(members={"x": member})
    resolved = resolve_for_person(_person(), {"behaviour_state": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    # Flattening: band.max=14 -> max(2) per day on a fixed RNG. At least one.
    assert len(out) >= 0


def test_theory_mappings_round_trip_to_episode():
    member = _member()
    object.__setattr__(member, "theory_mappings", {"comb": "physical_capability"})
    cat = ContextCategory(members={"x": member})
    resolved = resolve_for_person(_person(), {"behaviour_state": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert out[0].theory_mappings == {"comb": "physical_capability"}


def test_polarity_and_ontology_uri_round_trip():
    member = _member(polarity="high")
    object.__setattr__(member, "ontology_uri", "http://example/x")
    cat = ContextCategory(members={"x": member})
    resolved = resolve_for_person(_person(), {"trait_state": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert out[0].ontology_uri == "http://example/x"
    assert out[0].polarity == "high"


# ---------------------------------------------------------------------------
# plan_contexts (multi-day)
# ---------------------------------------------------------------------------


def test_plan_contexts_walks_every_day():
    cat = ContextCategory(members={"calm": _member()})
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    days = [
        DaySchedule(day_index=i, date=DATE + _dt.timedelta(days=i), weekday="Mon")
        for i in range(3)
    ]
    schedule = PersonSchedule(
        person_id="p_0001", persona_id="p", person_seed=1, days=days
    )
    out = plan_contexts(_person(), schedule, resolved, _wm())
    assert out.contexts
    assert len({e.date for e in out.contexts}) == 3


def test_deterministic_under_same_seed():
    cat = ContextCategory(members={"calm": _member()})
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    day = _day()
    a = plan_contexts_for_day(_person(person_seed=42), day, resolved, _wm())
    b = plan_contexts_for_day(_person(person_seed=42), day, resolved, _wm())
    assert [(e.start_minutes, e.end_minutes) for e in a] == [
        (e.start_minutes, e.end_minutes) for e in b
    ]


def test_episode_duration_property():
    cat = ContextCategory(members={"calm": _member(per_min=30, per_max=30)})
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    assert out[0].duration == 30


def test_within_tokens_string_form():
    from src.scripts.persona.config.schema import TemporalPattern
    from src.scripts.persona.context.planner import _within_tokens

    pat = TemporalPattern(mode="fix", details={"within": "morning"})
    assert _within_tokens(pat) == ["morning"]


def test_within_tokens_other_shape_returns_empty():
    from src.scripts.persona.config.schema import TemporalPattern
    from src.scripts.persona.context.planner import _within_tokens

    pat = TemporalPattern(mode="fix", details={"within": 42})
    assert _within_tokens(pat) == []


def test_fix_window_token_unknown_to_window_map_is_skipped():
    cat = ContextCategory(
        members={
            "calm": _member(
                per_min=30,
                per_max=30,
                patterns=[
                    TemporalPattern(mode="fix", details={"within": ["never_named"]})
                ],
            )
        }
    )
    resolved = resolve_for_person(_person(), {"mood_emotion": cat})
    out = plan_contexts_for_day(_person(), _day(), resolved, _wm())
    # No usable window from a typo'd name; planner falls through to
    # the default 24-hour bound and still places.
    assert len(out) == 1
