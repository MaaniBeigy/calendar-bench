"""Unit tests for src.scripts.persona.analytics.free_time."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.scripts.persona.analytics.free_time import (
    DAY_MINUTES,
    UNKNOWN_GROUP,
    FreeTimeReport,
    aggregate_by_group,
    build_free_time_report,
    free_fraction_in_day,
    free_minutes_in_day,
    person_free_time,
    population_free_time,
    render_report,
    write_report,
)
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from tests.unit.persona.conftest import make_person


def _person(pid: str, occupation: str = "student") -> Person:
    return make_person(
        person_id=pid,
        persona_id="alice",
        person_seed=42,
        occupation_status=occupation,
        stages=[],
    )


def _day(events: dict[str, list[tuple[int, int]]]) -> DaySchedule:
    return DaySchedule(
        day_index=0,
        date=_dt.date(2026, 5, 4),
        weekday="Mon",
        events={
            name: [
                EventInstance(event_name=name, start=s, duration=d) for s, d in pairs
            ]
            for name, pairs in events.items()
        },
        spillovers=[],
    )


def _sched(person_id: str, days: list[DaySchedule]) -> PersonSchedule:
    return PersonSchedule(
        person_id=person_id,
        persona_id="alice",
        person_seed=42,
        days=days,
    )


def test_free_minutes_with_no_events_is_full_day():
    assert free_minutes_in_day(_day({})) == DAY_MINUTES


def test_free_minutes_with_a_60_minute_event():
    day = _day({"lunch": [(750, 60)]})
    assert free_minutes_in_day(day) == DAY_MINUTES - 60


def test_free_minutes_clamps_at_zero_for_overscheduled_day():
    """An overnight event of 1500 min consumes the whole day plus extra; the
    helper still reports 0 free minutes (no negatives)."""
    day = _day({"sleep": [(0, 1500)]})
    assert free_minutes_in_day(day) == 0


def test_free_fraction_returns_minutes_over_1440():
    day = _day({"lunch": [(0, 720)]})
    assert free_fraction_in_day(day) == pytest.approx(0.5)


def test_person_free_time_rolls_up_avg_min_max():
    days = [
        _day({"lunch": [(0, 720)]}),  # 0.5 free
        _day({}),  # 1.0 free
        _day({"lunch": [(0, 360)]}),  # 0.75 free
    ]
    sched = _sched("alice_0000", days)
    stats = person_free_time(sched, "student")
    assert stats.avg_free == pytest.approx((0.5 + 1.0 + 0.75) / 3)
    assert stats.min_free == pytest.approx(0.5)
    assert stats.max_free == pytest.approx(1.0)
    assert stats.daily_free_fractions == pytest.approx((0.5, 1.0, 0.75))


def test_person_free_time_with_empty_schedule_returns_zero_rollup():
    stats = person_free_time(_sched("alice_0000", []), "student")
    assert stats.daily_free_fractions == ()
    assert stats.avg_free == 0.0
    assert stats.min_free == 0.0
    assert stats.max_free == 0.0


def test_population_length_mismatch_raises():
    with pytest.raises(ValueError, match="lengths must match"):
        population_free_time([_person("a")], [_sched("a", []), _sched("b", [])])


def test_population_pairs_persons_to_schedules_by_index():
    persons = [_person("alice_0000", "student"), _person("bob_0000", "fulltime")]
    schedules = [
        _sched("alice_0000", [_day({})]),
        _sched("bob_0000", [_day({"lunch": [(0, 720)]})]),
    ]
    out = population_free_time(persons, schedules)
    assert out[0].group_value == "student"
    assert out[0].avg_free == pytest.approx(1.0)
    assert out[1].group_value == "fulltime"
    assert out[1].avg_free == pytest.approx(0.5)


def test_population_groups_by_any_characteristic_axis():
    persons = [
        make_person(person_id="a_0000", stages=[]).model_copy(
            update={"characteristics": {"has_kids": True}}
        ),
        make_person(person_id="b_0000", stages=[]).model_copy(
            update={"characteristics": {"has_kids": False}}
        ),
    ]
    schedules = [
        _sched("a_0000", [_day({})]),
        _sched("b_0000", [_day({"lunch": [(0, 720)]})]),
    ]
    out = population_free_time(persons, schedules, group_by="has_kids")
    assert out[0].group_value == "true"
    assert out[1].group_value == "false"


def test_population_unknown_axis_falls_back_to_unknown_group():
    persons = [_person("a_0000", "student")]
    schedules = [_sched("a_0000", [_day({})])]
    out = population_free_time(persons, schedules, group_by="field_of_study")
    assert out[0].group_value == UNKNOWN_GROUP


def test_population_float_axis_renders_two_decimals():
    persons = [
        make_person(person_id="a_0000", stages=[]).model_copy(
            update={"characteristics": {"neuroticism": 0.3888}}
        )
    ]
    schedules = [_sched("a_0000", [_day({})])]
    out = population_free_time(persons, schedules, group_by="neuroticism")
    assert out[0].group_value == "0.39"


def test_mixed_population_buckets_carriers_and_unknowns_together():
    """Persons missing the grouped axis land in the unknown bucket beside real groups."""
    persons = [
        make_person(person_id="a_0000", stages=[]).model_copy(
            update={"characteristics": {"gender": "female"}}
        ),
        make_person(person_id="b_0000", stages=[]).model_copy(
            update={"characteristics": {}}
        ),
    ]
    schedules = [
        _sched("a_0000", [_day({})]),
        _sched("b_0000", [_day({})]),
    ]
    report = build_free_time_report(persons, schedules, group_by="gender")
    groups = {g.group_value: g.person_count for g in report.by_group}
    assert groups == {"female": 1, UNKNOWN_GROUP: 1}


def test_aggregate_by_group_groups_and_sorts():
    persons = [
        _person("a_0000", "student"),
        _person("b_0000", "fulltime"),
        _person("c_0000", "student"),
    ]
    schedules = [
        _sched("a_0000", [_day({})]),  # 1.0
        _sched("b_0000", [_day({"lunch": [(0, 720)]})]),  # 0.5
        _sched("c_0000", [_day({"lunch": [(0, 360)]})]),  # 0.75
    ]
    out = aggregate_by_group(population_free_time(persons, schedules))
    groups = [o.group_value for o in out]
    assert groups == ["fulltime", "student"]  # sorted alphabetically
    student = next(o for o in out if o.group_value == "student")
    assert student.person_count == 2
    assert student.avg_free == pytest.approx((1.0 + 0.75) / 2)
    assert student.min_free == pytest.approx(0.75)
    assert student.max_free == pytest.approx(1.0)


def test_aggregate_by_group_empty_input_returns_empty():
    assert aggregate_by_group([]) == []


def test_build_free_time_report_combines_persons_and_aggregate():
    persons = [_person("a_0000", "student")]
    schedules = [_sched("a_0000", [_day({})])]
    report = build_free_time_report(persons, schedules)
    assert report.group_by == "occupation_status"
    assert len(report.persons) == 1
    assert len(report.by_group) == 1
    assert report.by_group[0].group_value == "student"


def test_build_free_time_report_with_custom_group_by():
    persons = [
        make_person(person_id="a_0000", stages=[]).model_copy(
            update={"characteristics": {"gender": "female"}}
        )
    ]
    schedules = [_sched("a_0000", [_day({})])]
    report = build_free_time_report(persons, schedules, group_by="gender")
    assert report.group_by == "gender"
    assert report.by_group[0].group_value == "female"


def test_render_report_contains_every_section_header():
    persons = [_person("a_0000", "student")]
    schedules = [_sched("a_0000", [_day({})])]
    text = render_report(build_free_time_report(persons, schedules))
    assert "=== Free-time summary by occupation_status ===" in text
    assert "=== Free-time summary per person ===" in text
    assert "a_0000" in text
    assert "student" in text


def test_render_report_header_names_the_group_axis():
    persons = [
        make_person(person_id="a_0000", stages=[]).model_copy(
            update={"characteristics": {"gender": "female"}}
        )
    ]
    schedules = [_sched("a_0000", [_day({})])]
    text = render_report(build_free_time_report(persons, schedules, group_by="gender"))
    assert "=== Free-time summary by gender ===" in text
    assert "- female: persons=1" in text


def test_render_report_handles_empty_population_per_section():
    text = render_report(FreeTimeReport())
    assert "No persons in the population." in text
    # Both sections should display the empty message.
    assert text.count("No persons in the population.") == 2


def test_write_report_creates_text_file(tmp_path):
    persons = [_person("a_0000", "student")]
    schedules = [_sched("a_0000", [_day({})])]
    target = write_report(build_free_time_report(persons, schedules), tmp_path)
    assert target.name == "free_time_report.txt"
    assert "a_0000" in target.read_text(encoding="utf-8")


def test_render_report_orders_persons_by_id():
    persons = [
        _person("z_0000", "student"),
        _person("a_0000", "student"),
        _person("m_0000", "student"),
    ]
    schedules = [_sched(p.person_id, [_day({})]) for p in persons]
    text = render_report(build_free_time_report(persons, schedules))
    pos = [text.index(p.person_id) for p in persons]
    # 'a' appears before 'm', which appears before 'z' regardless of input order.
    a_pos = text.index("a_0000")
    m_pos = text.index("m_0000")
    z_pos = text.index("z_0000")
    assert a_pos < m_pos < z_pos
    del pos  # keep the variable referenced for clarity


def test_render_report_formats_percentages_to_one_decimal():
    persons = [_person("a_0000", "student")]
    schedules = [_sched("a_0000", [_day({"lunch": [(0, 720)]})])]
    text = render_report(build_free_time_report(persons, schedules))
    assert "50.0%" in text
