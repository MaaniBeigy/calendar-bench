"""Tests for `check_context_distributions`."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.config.schema import (
    ContextCategory,
    ContextMember,
    DurationRange,
    EpisodeRange,
    JitterConfig,
    TotalDuration,
)
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.validation.check_contexts import (
    ContextViolation,
    check_context_distributions,
)

DATE = _dt.date(2026, 5, 4)


def _member(
    per_min: int = 15,
    per_max: int = 60,
    eps_min: int = 1,
    eps_max: int = 3,
    *,
    eps_scale: str = "day",
    eps_unit: str = "minutes",
) -> ContextMember:
    return ContextMember(
        per_event_duration=DurationRange(min=per_min, max=per_max, unit=eps_unit),
        total_event_duration=TotalDuration(min=per_min, max=per_max * 4),
        total_event_episodes=EpisodeRange(min=eps_min, max=eps_max, scale=eps_scale),
    )


def _person(contexts: dict[str, ContextCategory]) -> Person:
    return Person(
        person_id="p_0001",
        persona_id="p",
        person_seed=0,
        instance_index=0,
        characteristics={},
        jitter_applied=JitterConfig(),
        contexts=contexts,
    )


def _schedule(episodes: list[ContextEpisode], n_days: int = 1) -> PersonSchedule:
    return PersonSchedule(
        person_id="p_0001",
        persona_id="p",
        person_seed=0,
        days=[
            DaySchedule(
                day_index=i,
                date=DATE + _dt.timedelta(days=i),
                weekday="Mon",
            )
            for i in range(n_days)
        ],
        contexts=episodes,
    )


def _ep(
    name: str = "happy",
    category: str = "mood_emotion",
    *,
    date: _dt.date = DATE,
    start: int = 540,
    end: int = 600,
) -> ContextEpisode:
    return ContextEpisode(
        name=name,
        category=category,
        date=date,
        start_minutes=start,
        end_minutes=end,
    )


def test_returns_empty_when_person_has_no_contexts():
    p = _person({})
    out = check_context_distributions([p], {"p_0001": _schedule([])})
    assert out == []


def test_returns_empty_when_schedule_missing():
    p = _person({"mood_emotion": ContextCategory(members={"happy": _member()})})
    out = check_context_distributions([p], {})
    assert out == []


def test_count_within_band_no_violation():
    cat = ContextCategory(members={"happy": _member(eps_min=1, eps_max=3)})
    p = _person({"mood_emotion": cat})
    s = _schedule([_ep(start=540, end=600), _ep(start=700, end=760)])
    assert check_context_distributions([p], {"p_0001": s}) == []


def test_count_below_band_emits_violation():
    cat = ContextCategory(members={"happy": _member(eps_min=5, eps_max=10)})
    p = _person({"mood_emotion": cat})
    s = _schedule([_ep()])
    out = check_context_distributions([p], {"p_0001": s}, tolerance=0)
    assert len(out) == 1
    assert out[0].kind == "episode_count_mismatch"
    assert ">= 5" in out[0].detail


def test_count_above_band_emits_violation():
    cat = ContextCategory(members={"happy": _member(eps_min=0, eps_max=1)})
    p = _person({"mood_emotion": cat})
    s = _schedule(
        [
            _ep(start=100, end=130),
            _ep(start=200, end=230),
            _ep(start=300, end=330),
        ]
    )
    out = check_context_distributions([p], {"p_0001": s}, tolerance=0)
    counts = [v for v in out if v.kind == "episode_count_mismatch"]
    assert len(counts) == 1
    assert "<= 1" in counts[0].detail


def test_tolerance_absorbs_near_misses():
    cat = ContextCategory(members={"happy": _member(eps_min=2, eps_max=2)})
    p = _person({"mood_emotion": cat})
    s = _schedule([_ep()])  # one short
    assert check_context_distributions([p], {"p_0001": s}, tolerance=1) == []


def test_weekly_scale_expanded_via_horizon():
    cat = ContextCategory(
        members={
            "walk": _member(eps_min=5, eps_max=7, eps_scale="week"),
        }
    )
    p = _person({"behaviour_state": cat})
    s = _schedule([_ep(name="walk", category="behaviour_state")], n_days=14)
    # horizon=14 -> 2 weeks; expected [10, 14]; realized=1 -> violation.
    out = check_context_distributions([p], {"p_0001": s}, tolerance=0)
    assert len(out) == 1
    assert ">= 10" in out[0].detail


def test_monthly_scale_expanded_via_horizon():
    cat = ContextCategory(
        members={
            "rare": _member(eps_min=1, eps_max=2, eps_scale="month"),
        }
    )
    p = _person({"mood_emotion": cat})
    s = _schedule([], n_days=60)  # 2 months; expected [2, 4]
    out = check_context_distributions([p], {"p_0001": s}, tolerance=0)
    assert len(out) == 1
    assert ">= 2" in out[0].detail


def test_season_scale_expanded_via_horizon():
    cat = ContextCategory(
        members={"rare": _member(eps_min=1, eps_max=2, eps_scale="season")}
    )
    p = _person({"mood_emotion": cat})
    s = _schedule([], n_days=180)  # 2 seasons; expected [2, 4]
    out = check_context_distributions([p], {"p_0001": s}, tolerance=0)
    assert len(out) == 1


def test_weekday_scale_returns_band_unchanged():
    cat = ContextCategory(
        members={"x": _member(eps_min=0, eps_max=1, eps_scale="weekday")}
    )
    p = _person({"mood_emotion": cat})
    s = _schedule([])
    assert check_context_distributions([p], {"p_0001": s}, tolerance=0) == []


def test_duration_outside_band_emits_violation():
    cat = ContextCategory(members={"happy": _member(per_min=30, per_max=60)})
    p = _person({"mood_emotion": cat})
    s = _schedule([_ep(start=540, end=550)])  # 10 min, below 30
    out = check_context_distributions([p], {"p_0001": s})
    out = [v for v in out if v.kind == "duration_band_violation"]
    assert len(out) == 1
    assert "outside band" in out[0].detail


def test_duration_hours_unit_resolved():
    cat = ContextCategory(
        members={
            "happy": _member(per_min=1, per_max=2, eps_unit="hours"),
        }
    )
    p = _person({"mood_emotion": cat})
    s = _schedule([_ep(start=540, end=540 + 90)])  # 90 min within [60, 120]
    out = check_context_distributions([p], {"p_0001": s})
    assert [v for v in out if v.kind == "duration_band_violation"] == []


def test_mutually_exclusive_overlap_emits_violation():
    cat = ContextCategory(
        mutually_exclusive=True,
        members={"happy": _member(), "sad": _member()},
    )
    p = _person({"mood_emotion": cat})
    s = _schedule([_ep("happy", start=540, end=620), _ep("sad", start=600, end=700)])
    out = check_context_distributions([p], {"p_0001": s})
    overlaps = [v for v in out if v.kind == "mutually_exclusive_overlap"]
    assert len(overlaps) == 1
    assert "happy" in overlaps[0].detail and "sad" in overlaps[0].detail


def test_mutually_exclusive_silent_when_disjoint():
    cat = ContextCategory(
        mutually_exclusive=True,
        members={"happy": _member(), "sad": _member()},
    )
    p = _person({"mood_emotion": cat})
    s = _schedule([_ep("happy", start=540, end=600), _ep("sad", start=700, end=760)])
    out = check_context_distributions([p], {"p_0001": s})
    assert [v for v in out if v.kind == "mutually_exclusive_overlap"] == []


def test_non_exclusive_category_does_not_emit_overlap_violation():
    cat = ContextCategory(
        mutually_exclusive=False,
        members={"a": _member(), "b": _member()},
    )
    p = _person({"behaviour_state": cat})
    s = _schedule(
        [
            _ep("a", category="behaviour_state", start=540, end=600),
            _ep("b", category="behaviour_state", start=550, end=610),
        ]
    )
    out = check_context_distributions([p], {"p_0001": s})
    assert [v for v in out if v.kind == "mutually_exclusive_overlap"] == []


def test_violation_dataclass_is_hashable():
    v = ContextViolation(person_id="p", category="c", member="m", kind="k", detail="d")
    assert hash(v)
