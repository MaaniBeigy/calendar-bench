"""LTL checker now resolves `context:<name>` atoms against `PersonSchedule.contexts`."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.config.schema import JitterConfig, TemporalRule
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.validation.check_ltl import (
    _atom_count_in_window,
    _atom_instances,
    check_ltl_rules,
)

DATE = _dt.date(2026, 5, 4)


def _person(pid: str = "p_0001") -> Person:
    return Person(
        person_id=pid,
        persona_id="p",
        person_seed=0,
        instance_index=0,
        jitter_applied=JitterConfig(),
    )


def _schedule(
    *,
    events: dict[str, list[EventInstance]] | None = None,
    contexts: list[ContextEpisode] | None = None,
) -> PersonSchedule:
    day = DaySchedule(day_index=0, date=DATE, weekday="Mon", events=events or {})
    return PersonSchedule(
        person_id="p_0001",
        persona_id="p",
        person_seed=0,
        days=[day],
        contexts=contexts or [],
    )


def _ctx(name: str, start: int, end: int) -> ContextEpisode:
    return ContextEpisode(
        name=name,
        category="mood_emotion",
        date=DATE,
        start_minutes=start,
        end_minutes=end,
    )


def test_atom_instances_returns_event_when_no_prefix():
    s = _schedule(events={"office_work": [EventInstance("office_work", 540, 60)]})
    out = _atom_instances(s, s.days[0], "office_work")
    assert len(out) == 1
    assert out[0].start == 540 and out[0].duration == 60


def test_atom_instances_returns_context_when_prefixed():
    s = _schedule(contexts=[_ctx("happy", 540, 600)])
    out = _atom_instances(s, s.days[0], "context:happy")
    assert len(out) == 1
    assert out[0].start == 540 and out[0].duration == 60


def test_atom_instances_filters_context_by_date():
    other = _dt.date(2026, 5, 5)
    s = _schedule(
        contexts=[
            _ctx("happy", 540, 600),
            ContextEpisode(
                name="happy",
                category="mood_emotion",
                date=other,
                start_minutes=720,
                end_minutes=780,
            ),
        ]
    )
    out = _atom_instances(s, s.days[0], "context:happy")
    assert len(out) == 1


def test_atom_instances_unknown_event_returns_empty():
    s = _schedule()
    assert _atom_instances(s, s.days[0], "missing") == []


def test_atom_instances_unknown_context_returns_empty():
    s = _schedule()
    assert _atom_instances(s, s.days[0], "context:absent") == []


def test_atom_count_in_window_sums_per_day_for_contexts():
    day1 = DaySchedule(day_index=0, date=DATE, weekday="Mon")
    day2 = DaySchedule(day_index=1, date=DATE + _dt.timedelta(days=1), weekday="Tue")
    s = PersonSchedule(
        person_id="p",
        persona_id="p",
        person_seed=0,
        days=[day1, day2],
        contexts=[
            _ctx("walk", 540, 600),
            ContextEpisode(
                name="walk",
                category="behaviour_state",
                date=DATE + _dt.timedelta(days=1),
                start_minutes=720,
                end_minutes=780,
            ),
        ],
    )
    assert _atom_count_in_window(s, [day1, day2], "context:walk") == 2


def test_no_overlap_fires_on_event_vs_context_conflict():
    s = _schedule(
        events={"office_work": [EventInstance("office_work", 540, 60)]},
        contexts=[_ctx("happy", 550, 610)],
    )
    rule = TemporalRule(
        id="no_office_happy", formula="G ¬(office_work ∧ context:happy)"
    )
    persons = {"p_0001": _person()}
    violations = check_ltl_rules([s], [rule], persons)
    assert len(violations) == 1
    assert "office_work" in violations[0].detail
    assert "context:happy" in violations[0].detail


def test_no_overlap_silent_when_no_conflict():
    s = _schedule(
        events={"office_work": [EventInstance("office_work", 540, 60)]},
        contexts=[_ctx("happy", 700, 730)],
    )
    rule = TemporalRule(
        id="no_office_happy", formula="G ¬(office_work ∧ context:happy)"
    )
    assert check_ltl_rules([s], [rule], {"p_0001": _person()}) == []


def test_min_overlap_demands_context_overlap_with_event():
    s = _schedule(
        events={"lunch": [EventInstance("lunch", 720, 60)]},
        contexts=[_ctx("happy", 200, 300)],
    )
    rule = TemporalRule(
        id="lunch_happy", formula="G (lunch → F (lunch ∧ context:happy))"
    )
    out = check_ltl_rules([s], [rule], {"p_0001": _person()})
    assert len(out) == 1
    assert "no overlapping context:happy" in out[0].detail


def test_min_overlap_passes_when_context_overlaps_event():
    """The persona placer may now overlap state contexts with events."""
    s = _schedule(
        events={"gym": [EventInstance("gym", 960, 90)]},
        contexts=[_ctx("happy", 990, 1030)],
    )
    rule = TemporalRule(
        id="gym_overlaps_happy", formula="G (gym → F (gym ∧ context:happy))"
    )
    out = check_ltl_rules([s], [rule], {"p_0001": _person()})
    assert out == []


def test_eventually_co_occur_passes_when_event_overlaps_context():
    """`F (event AND context:X)` is satisfied by a single overlap anywhere."""
    s = _schedule(
        events={"gym": [EventInstance("gym", 960, 90)]},
        contexts=[_ctx("self_efficacy", 1000, 1040)],
    )
    rule = TemporalRule(
        id="exists_gym_with_belief",
        formula="F (gym ∧ context:self_efficacy)",
    )
    out = check_ltl_rules([s], [rule], {"p_0001": _person()})
    assert out == []


def test_implies_future_works_with_context_consequent():
    s = _schedule(
        events={"office_work": [EventInstance("office_work", 540, 60)]},
    )
    rule = TemporalRule(
        id="office_to_tired", formula="G (office_work → F context:tired)"
    )
    out = check_ltl_rules([s], [rule], {"p_0001": _person()})
    assert len(out) == 1
    assert "no context:tired that day" in out[0].detail


def test_weekly_count_atom_can_be_context():
    days = [
        DaySchedule(day_index=i, date=DATE + _dt.timedelta(days=i), weekday="Mon")
        for i in range(7)
    ]
    s = PersonSchedule(
        person_id="p_0001",
        persona_id="p",
        person_seed=0,
        days=days,
        contexts=[
            ContextEpisode(
                name="walk",
                category="behaviour_state",
                date=DATE + _dt.timedelta(days=i),
                start_minutes=600,
                end_minutes=630,
            )
            for i in range(3)
        ],
    )
    rule = TemporalRule(id="walk_target", formula="weekly_count(context:walk, ≥, 5)")
    out = check_ltl_rules([s], [rule], {"p_0001": _person()})
    assert len(out) == 1
    assert "weekly_count(context:walk) = 3" in out[0].detail
