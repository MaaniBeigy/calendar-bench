"""Unit tests for src.scripts.persona.validation.check_continuity."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule, Spillover
from src.scripts.persona.validation.check_continuity import check_continuity

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _build(
    *,
    person_id: str = "x_0000",
    days_data: list[tuple[dict, list[Spillover]]],
    base: _dt.date | None = None,
    weekday_overrides: dict[int, str] | None = None,
) -> PersonSchedule:
    base = base or _dt.date(2026, 5, 4)
    weekday_overrides = weekday_overrides or {}
    days: list[DaySchedule] = []
    for i, (events, spillovers) in enumerate(days_data):
        d = base + _dt.timedelta(days=i)
        evs = {
            name: [
                EventInstance(event_name=name, start=s, duration=du) for s, du in pairs
            ]
            for name, pairs in events.items()
        }
        weekday = weekday_overrides.get(i, WEEKDAY_NAMES[d.weekday()])
        days.append(
            DaySchedule(
                day_index=i,
                date=d,
                weekday=weekday,
                events=evs,
                spillovers=list(spillovers),
            )
        )
    return PersonSchedule(
        person_id=person_id,
        persona_id="x",
        person_seed=1,
        days=days,
    )


def test_clean_horizon_emits_no_violations():
    sched = _build(days_data=[({}, []), ({}, [])])
    assert check_continuity([sched]) == []


def test_day0_with_spillover_is_flagged():
    spill = Spillover(
        event_name="sleep",
        start=0,
        duration=60,
        orig_start=1380,
        orig_duration=120,
        event_idx=0,
    )
    sched = _build(days_data=[({}, [spill])])
    out = check_continuity([sched])
    assert any(v.kind == "day0_spillover" for v in out)


def test_correct_spillover_propagation_no_violation():
    spill = Spillover(
        event_name="sleep",
        start=0,
        duration=60,
        orig_start=1380,
        orig_duration=120,
        event_idx=0,
    )
    sched = _build(
        days_data=[
            ({"sleep": [(1380, 120)]}, []),
            ({}, [spill]),
        ]
    )
    out = check_continuity([sched])
    assert all(v.kind != "spillover_mismatch" for v in out)


def test_missing_spillover_flagged_as_mismatch():
    sched = _build(
        days_data=[
            ({"sleep": [(1380, 120)]}, []),
            ({}, []),  # spillover dropped
        ]
    )
    out = check_continuity([sched])
    assert any(v.kind == "spillover_mismatch" for v in out)


def test_event_overlapping_spillover_flagged():
    spill = Spillover(
        event_name="sleep",
        start=0,
        duration=60,
        orig_start=1380,
        orig_duration=120,
        event_idx=0,
    )
    sched = _build(
        days_data=[
            ({"sleep": [(1380, 120)]}, []),
            ({"first_eat": [(30, 30)]}, [spill]),
        ]
    )
    out = check_continuity([sched])
    assert any(v.kind == "overlap" for v in out)


def test_event_after_spillover_no_overlap_violation():
    spill = Spillover(
        event_name="sleep",
        start=0,
        duration=60,
        orig_start=1380,
        orig_duration=120,
        event_idx=0,
    )
    sched = _build(
        days_data=[
            ({"sleep": [(1380, 120)]}, []),
            ({"first_eat": [(120, 30)]}, [spill]),
        ]
    )
    out = check_continuity([sched])
    assert all(v.kind != "overlap" for v in out)


def test_date_jump_flagged():
    sched = _build(days_data=[({}, []), ({}, [])])
    bad_day1 = sched.days[1]
    sched_bad = PersonSchedule(
        person_id=sched.person_id,
        persona_id=sched.persona_id,
        person_seed=sched.person_seed,
        days=[
            sched.days[0],
            DaySchedule(
                day_index=bad_day1.day_index,
                date=_dt.date(2026, 5, 6),  # off-by-one
                weekday=bad_day1.weekday,
                events=bad_day1.events,
                spillovers=bad_day1.spillovers,
            ),
        ],
    )
    out = check_continuity([sched_bad])
    assert any(v.kind == "date" for v in out)


def test_weekday_inconsistent_with_date_flagged():
    sched = _build(
        days_data=[({}, [])],
        weekday_overrides={0: "Sun"},  # 2026-05-04 is a Monday
    )
    out = check_continuity([sched])
    assert any(v.kind == "weekday" for v in out)


def test_empty_schedule_returns_no_violations():
    sched = PersonSchedule(person_id="x_0000", persona_id="x", person_seed=1, days=[])
    assert check_continuity([sched]) == []


def test_multiple_schedules_all_checked():
    spill = Spillover(
        event_name="sleep",
        start=0,
        duration=60,
        orig_start=1380,
        orig_duration=120,
        event_idx=0,
    )
    a = _build(person_id="a_0000", days_data=[({}, [spill])])
    b = _build(person_id="b_0000", days_data=[({}, [])])
    out = check_continuity([a, b])
    persons = {v.person_id for v in out}
    assert persons == {"a_0000"}
