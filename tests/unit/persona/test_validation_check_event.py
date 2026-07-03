"""Unit tests for src.scripts.persona.validation.check_event."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    EventOverride,
    TotalDuration,
)
from src.scripts.persona.domain.event import Catalog, EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.validation.check_event import check_event_constraints
from tests.unit.persona.conftest import make_person


def _catalog(event_def: EventDefinition) -> Catalog:
    cfg = EventConfig(
        categories={
            event_def.category: Category(
                name=event_def.category, events={event_def.name: event_def}
            )
        }
    )
    return Catalog.from_event_config(cfg)


def _person(
    person_id: str = "x_0000",
    *,
    event_overrides: dict[str, EventOverride] | None = None,
) -> Person:
    return make_person(
        person_id=person_id,
        persona_id="x",
        person_seed=1,
        occupation_status="student",
        stages=[],
        event_overrides=event_overrides or {},
    )


def _sched(
    events: dict[str, list[EventInstance]],
    weekday: str = "Mon",
    *,
    person_id: str = "x_0000",
) -> PersonSchedule:
    return PersonSchedule(
        person_id=person_id,
        persona_id="x",
        person_seed=1,
        days=[
            DaySchedule(
                day_index=0,
                date=_dt.date(2026, 5, 4),
                weekday=weekday,
                events=events,
                spillovers=[],
            )
        ],
    )


def _lunch_def(**overrides) -> EventDefinition:
    base = dict(
        name="lunch",
        category="eat",
        per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
        total_event_duration=TotalDuration(min=30, max=60, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )
    base.update(overrides)
    return EventDefinition(**base)


def test_no_violations_when_event_within_bounds():
    cat = _catalog(_lunch_def())
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]}
    )
    assert check_event_constraints([sched], [_person()], cat) == []


def test_per_event_duration_below_min_is_flagged():
    cat = _catalog(_lunch_def())
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=10)]}
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert len(out) >= 1
    assert any(v.kind == "per_event_duration" and "< min" in v.detail for v in out)


def test_per_event_duration_above_max_is_flagged():
    cat = _catalog(_lunch_def())
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=120)]}
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert any(v.kind == "per_event_duration" and "> max" in v.detail for v in out)


def test_per_day_total_under_min_flagged():
    e = _lunch_def(
        per_event_duration=DurationRange(min=10, max=60),
        total_event_duration=TotalDuration(min=60, max=90, scale="day", unit="minutes"),
    )
    cat = _catalog(e)
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=30)]}
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert any(
        v.kind == "per_day" and "total duration" in v.detail and "< min" in v.detail
        for v in out
    )


def test_per_day_total_over_max_flagged():
    e = _lunch_def(
        per_event_duration=DurationRange(min=10, max=200),
        total_event_duration=TotalDuration(min=10, max=30, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=2),
    )
    cat = _catalog(e)
    sched = _sched(
        {
            "lunch": [
                EventInstance(event_name="lunch", start=600, duration=30),
                EventInstance(event_name="lunch", start=700, duration=30),
            ]
        }
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert any(v.kind == "per_day" and "> max" in v.detail for v in out)


def test_per_day_episodes_under_min_flagged():
    e = _lunch_def(
        total_event_episodes=EpisodeRange(scale="day", min=2, max=2),
        total_event_duration=TotalDuration(
            min=30, max=240, scale="day", unit="minutes"
        ),
    )
    cat = _catalog(e)
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]}
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert any(
        v.kind == "per_day" and "episodes" in v.detail and "< min" in v.detail
        for v in out
    )


def test_per_day_episodes_over_max_flagged():
    e = _lunch_def(
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        total_event_duration=TotalDuration(
            min=10, max=240, scale="day", unit="minutes"
        ),
    )
    cat = _catalog(e)
    sched = _sched(
        {
            "lunch": [
                EventInstance(event_name="lunch", start=600, duration=30),
                EventInstance(event_name="lunch", start=700, duration=30),
            ]
        }
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert any(
        v.kind == "per_day" and "episodes" in v.detail and "> max" in v.detail
        for v in out
    )


def test_weekday_violation_flagged():
    e = _lunch_def(weekdays=["Tue"])
    cat = _catalog(e)
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]},
        weekday="Mon",
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert any(v.kind == "weekday" for v in out)


def test_weekday_no_violation_when_allowed():
    e = _lunch_def(weekdays=["Mon", "Tue"])
    cat = _catalog(e)
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]},
        weekday="Mon",
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert all(v.kind != "weekday" for v in out)


def test_per_day_skipped_when_scale_is_not_day():
    e = _lunch_def(
        total_event_duration=TotalDuration(
            min=60, max=120, scale="week", unit="minutes"
        )
    )
    cat = _catalog(e)
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=15)]}
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert all(v.kind != "per_day" for v in out)


def test_per_day_skipped_when_episode_scale_is_not_day():
    e = _lunch_def(
        total_event_episodes=EpisodeRange(scale="week", min=1, max=2),
    )
    cat = _catalog(e)
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=15)]}
    )
    out = check_event_constraints([sched], [_person()], cat)
    assert all(v.kind != "per_day" for v in out)


def test_hours_unit_converted_to_minutes_for_bounds():
    e = _lunch_def(
        per_event_duration=DurationRange(min=1, max=2, unit="hours"),
        total_event_duration=TotalDuration(min=1, max=2, scale="day", unit="hours"),
    )
    cat = _catalog(e)
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=600, duration=30)]}
    )
    out = check_event_constraints([sched], [_person()], cat)
    # 30 minutes is below the 60-minute min.
    assert any("< min 60" in v.detail for v in out)


def test_no_events_skips_per_day_check():
    e = _lunch_def(total_event_episodes=EpisodeRange(scale="day", min=1, max=1))
    cat = _catalog(e)
    sched = _sched({})
    out = check_event_constraints([sched], [_person()], cat)
    assert out == []


# -------------------------------------------------------------------------------------
# ----------------------- per-person event_overrides drive validation ------------------
# -------------------------------------------------------------------------------------


def test_event_override_narrows_duration_and_flags_previously_legal_event():
    """A persona that overrides lunch's duration to [10, 20] sees a 45-min
    lunch as a violation, even though the global catalog allows 30-60 min."""
    cat = _catalog(_lunch_def())  # global allows 30-60
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]}
    )
    overrides = {
        "lunch": EventOverride(
            per_event_duration=DurationRange(min=10, max=20, unit="minutes")
        )
    }
    out = check_event_constraints([sched], [_person(event_overrides=overrides)], cat)
    assert any(v.kind == "per_event_duration" and "> max 20" in v.detail for v in out)


def test_event_override_widens_bounds_and_clears_previously_illegal_event():
    """A persona that overrides lunch up to 200 min sees a 120-min event as
    legal, even though the global catalog flags it."""
    cat = _catalog(_lunch_def())  # global flags > 60 min
    sched = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=600, duration=120)]}
    )
    base_violations = check_event_constraints([sched], [_person()], cat)
    assert any(v.kind == "per_event_duration" for v in base_violations)
    overrides = {
        "lunch": EventOverride(
            per_event_duration=DurationRange(min=10, max=200, unit="minutes"),
            total_event_duration=TotalDuration(
                min=10, max=200, scale="day", unit="minutes"
            ),
        )
    }
    out = check_event_constraints([sched], [_person(event_overrides=overrides)], cat)
    assert all(v.kind != "per_event_duration" for v in out)


def test_two_persons_validated_against_their_own_overrides():
    """Two persons with different overrides on the same event each see their
    own bounds; one passes, one fails on the same emitted duration."""
    cat = _catalog(_lunch_def())  # global 30-60 min
    sched_a = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]},
        person_id="a_0000",
    )
    sched_b = _sched(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]},
        person_id="b_0000",
    )
    person_a = _person(
        "a_0000",
        event_overrides={
            "lunch": EventOverride(
                per_event_duration=DurationRange(min=30, max=60, unit="minutes"),
                total_event_duration=TotalDuration(
                    min=30, max=60, scale="day", unit="minutes"
                ),
            )
        },
    )
    person_b = _person(
        "b_0000",
        event_overrides={
            "lunch": EventOverride(
                per_event_duration=DurationRange(min=10, max=20, unit="minutes"),
                total_event_duration=TotalDuration(
                    min=10, max=20, scale="day", unit="minutes"
                ),
            )
        },
    )
    out = check_event_constraints([sched_a, sched_b], [person_a, person_b], cat)
    person_b_violations = [v for v in out if v.person_id == "b_0000"]
    person_a_violations = [v for v in out if v.person_id == "a_0000"]
    assert person_a_violations == []
    assert any(v.kind == "per_event_duration" for v in person_b_violations)


def test_persons_and_schedules_length_mismatch_raises():
    cat = _catalog(_lunch_def())
    sched = _sched({})
    import pytest

    with pytest.raises(ValueError, match="lengths must match"):
        check_event_constraints([sched, sched], [_person()], cat)
