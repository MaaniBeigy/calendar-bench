"""Unit tests for src.scripts.scenarios.domain types."""

from __future__ import annotations

import datetime

from src.scripts.scenarios.domain.calendar import AugmentedCalendar, CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from tests.unit.scenarios.conftest import DATE, make_event, make_scheduled, make_task


class TestRecommendedTask:
    def test_required_fields(self):
        t = make_task()
        assert t.label == "running"
        assert t.duration_min == 30
        assert t.duration_max == 60
        assert t.intensity == 3

    def test_flag_defaults(self):
        t = make_task()
        assert not t.is_dividable
        assert not t.is_concurrent

    def test_optional_ontology_fields_default_none(self):
        t = make_task()
        assert t.ontology_uri is None

    def test_concurrent_flag(self):
        t = make_task(is_concurrent=True)
        assert t.is_concurrent

    def test_dividable_flag(self):
        t = make_task(is_dividable=True)
        assert t.is_dividable

    def test_with_ontology_uri(self):
        uri = "https://w3id.org/calendar-bench/health/task/running"
        t = make_task(ontology_uri=uri)
        assert t.ontology_uri == uri

    def test_intensity_range(self):
        for intensity in range(1, 6):
            t = make_task(intensity=intensity)
            assert t.intensity == intensity

    def test_fields_are_mutable(self):
        t = make_task()
        t.is_concurrent = True
        assert t.is_concurrent is True


class TestScheduledTask:
    def test_required_fields(self):
        st = make_scheduled()
        assert st.start_minutes == 480
        assert st.end_minutes == 540
        assert st.date == DATE

    def test_standalone_default(self):
        st = make_scheduled()
        assert st.is_standalone
        assert st.concurrent_with is None

    def test_concurrent_task(self):
        st = make_scheduled(is_standalone=False, concurrent_with="lunch")
        assert not st.is_standalone
        assert st.concurrent_with == "lunch"

    def test_task_reference_preserved(self):
        t = make_task(label="yoga")
        st = make_scheduled(task=t)
        assert st.task.label == "yoga"
        assert st.task is t

    def test_duration_via_start_end(self):
        st = make_scheduled(start_minutes=480, end_minutes=540)
        assert st.end_minutes - st.start_minutes == 60

    def test_different_date(self):
        other_date = datetime.date(2026, 5, 11)
        st = make_scheduled(date=other_date)
        assert st.date == other_date


class TestCalendarEvent:
    def test_required_fields(self):
        ev = make_event()
        assert ev.label == "lunch"
        assert ev.start_minutes == 720
        assert ev.end_minutes == 780
        assert ev.date == DATE

    def test_concurrency_defaults(self):
        ev = make_event()
        assert not ev.is_concurrent
        assert not ev.is_dividable
        assert ev.concurrent_with == []
        assert ev.intensity == 1

    def test_concurrent_event(self):
        ev = make_event(is_concurrent=True, concurrent_with=["reading", "podcast"])
        assert ev.is_concurrent
        assert "reading" in ev.concurrent_with
        assert "podcast" in ev.concurrent_with

    def test_dividable_event(self):
        ev = make_event(is_dividable=True)
        assert ev.is_dividable

    def test_intensity_override(self):
        ev = make_event(intensity=4)
        assert ev.intensity == 4

    def test_concurrent_with_is_mutable(self):
        ev = make_event()
        ev.concurrent_with.append("walking")
        assert "walking" in ev.concurrent_with

    def test_concurrent_with_independence(self):
        """Two events with default concurrent_with do not share the same list."""
        ev1 = make_event(label="lunch")
        ev2 = make_event(label="dinner")
        ev1.concurrent_with.append("running")
        assert "running" not in ev2.concurrent_with

    def test_duration_implied_by_start_end(self):
        ev = make_event(start_minutes=720, end_minutes=780)
        assert ev.end_minutes - ev.start_minutes == 60


class TestCalendarTrace:
    def test_empty_trace(self, empty_trace):
        assert empty_trace.person_id == "p001"
        assert empty_trace.events == []

    def test_trace_with_events(self):
        events = [
            make_event(label="lunch", start_minutes=720, end_minutes=780),
            make_event(label="dinner", start_minutes=1140, end_minutes=1200),
        ]
        ct = CalendarTrace(person_id="p001", events=events)
        assert len(ct.events) == 2
        assert ct.events[0].label == "lunch"
        assert ct.events[1].label == "dinner"

    def test_trace_default_empty_events(self):
        ct = CalendarTrace(person_id="alice")
        assert ct.events == []

    def test_trace_person_id(self):
        ct = CalendarTrace(person_id="a_student_0005")
        assert ct.person_id == "a_student_0005"


class TestAugmentedCalendar:
    def test_empty_augmented_calendar(self, empty_augmented):
        assert empty_augmented.person_id == "p001"
        assert empty_augmented.base_events == []
        assert empty_augmented.scheduled_tasks == []

    def test_with_base_events(self):
        events = [make_event()]
        ac = AugmentedCalendar(person_id="p001", base_events=events)
        assert len(ac.base_events) == 1

    def test_with_scheduled_tasks(self):
        tasks = [make_scheduled()]
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=tasks)
        assert len(ac.scheduled_tasks) == 1

    def test_list_independence(self):
        """Default lists for two different instances are independent."""
        ac1 = AugmentedCalendar(person_id="p001")
        ac2 = AugmentedCalendar(person_id="p002")
        ac1.base_events.append(make_event())
        assert len(ac2.base_events) == 0


class TestSchedulingSolution:
    def test_empty_solution(self, empty_augmented):
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=empty_augmented,
        )
        assert sol.person_id == "p001"
        assert sol.tasks == []
        assert sol.scheduled == []
        assert sol.unscheduled == []

    def test_augmented_calendar_stored(self, empty_augmented):
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=empty_augmented,
        )
        assert sol.augmented_calendar is empty_augmented

    def test_solution_with_scheduled_and_unscheduled(self):
        t1 = make_task(label="running")
        t2 = make_task(label="yoga")
        st1 = make_scheduled(task=t1)
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st1])

        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[t1, t2],
            scheduled=[st1],
            unscheduled=[t2],
        )

        assert len(sol.tasks) == 2
        assert len(sol.scheduled) == 1
        assert len(sol.unscheduled) == 1
        assert sol.scheduled[0].task.label == "running"
        assert sol.unscheduled[0].label == "yoga"

    def test_list_defaults_are_independent(self):
        """Two solutions do not share list objects."""
        ac1 = AugmentedCalendar(person_id="p001")
        ac2 = AugmentedCalendar(person_id="p002")
        sol1 = SchedulingSolution(person_id="p001", augmented_calendar=ac1)
        sol2 = SchedulingSolution(person_id="p002", augmented_calendar=ac2)
        sol1.tasks.append(make_task())
        assert len(sol2.tasks) == 0

    def test_partially_scheduled(self):
        tasks = [make_task(label=f"task_{i}") for i in range(4)]
        scheduled_tasks = [make_scheduled(task=t) for t in tasks[:2]]
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=scheduled_tasks)

        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=tasks,
            scheduled=scheduled_tasks,
            unscheduled=tasks[2:],
        )

        assert len(sol.scheduled) + len(sol.unscheduled) == len(sol.tasks)
