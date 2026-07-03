"""Unit tests for export/json_writer.py, export/ics_writer.py, export/report_writer.py.

Coverage targets:
  json_writer.py:  solution_to_dict, write_solution_json.
  ics_writer.py:   _minutes_to_datetime, solution_to_ical_bytes, write_augmented_ics.
  report_writer.py: format_person_gains_report, format_total_gain_report,
                    format_ontology_grounding_report, format_telemetry_report,
                    format_preference_breakdown_report, *_to_dict counterparts,
                    write_evaluation_reports (5 isolated .txt+.json pairs).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from src.scripts.scenarios.domain.calendar import AugmentedCalendar, CalendarTrace
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import ScheduledTask
from src.scripts.scenarios.export.ics_writer import (
    _minutes_to_datetime,
    solution_to_ical_bytes,
    write_augmented_ics,
)
from src.scripts.scenarios.export.json_writer import (
    solution_to_dict,
    write_solution_json,
)
from src.scripts.scenarios.export.report_writer import (
    ONTOLOGY_REPORT_BASENAME,
    PERSON_REPORT_BASENAME,
    PREFERENCE_REPORT_BASENAME,
    TELEMETRY_REPORT_BASENAME,
    TOTAL_REPORT_BASENAME,
    format_ontology_grounding_report,
    format_person_gains_report,
    format_preference_breakdown_report,
    format_telemetry_report,
    format_total_gain_report,
    ontology_grounding_to_dict,
    person_gains_to_dict,
    preference_breakdown_to_dict,
    telemetry_to_dict,
    total_gain_to_dict,
    write_evaluation_reports,
)
from src.scripts.scenarios.metrics.loss import LossComponents
from tests.unit.scenarios.conftest import make_event, make_scheduled, make_task

DATE = datetime.date(2026, 5, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_solution(person_id: str = "p001") -> SchedulingSolution:
    ac = AugmentedCalendar(person_id=person_id)
    return SchedulingSolution(
        person_id=person_id,
        augmented_calendar=ac,
    )


def _solution_with_scheduled(person_id: str = "p001") -> SchedulingSolution:
    task = make_task("yoga")
    st = make_scheduled(task=task, start_minutes=480, end_minutes=540, date=DATE)
    ac = AugmentedCalendar(
        person_id=person_id,
        scheduled_tasks=[st],
    )
    return SchedulingSolution(
        person_id=person_id,
        augmented_calendar=ac,
        tasks=[task],
        scheduled=[st],
        unscheduled=[],
    )


def _solution_with_unscheduled(person_id: str = "p001") -> SchedulingSolution:
    task = make_task("running")
    ac = AugmentedCalendar(person_id=person_id)
    return SchedulingSolution(
        person_id=person_id,
        augmented_calendar=ac,
        tasks=[task],
        scheduled=[],
        unscheduled=[task],
    )


# ---------------------------------------------------------------------------
# json_writer
# ---------------------------------------------------------------------------


class TestSolutionToDict:
    def test_empty_solution_structure(self):
        d = solution_to_dict(_empty_solution())
        assert d["person_id"] == "p001"
        assert d["tasks_total"] == 0
        assert d["scheduled_count"] == 0
        assert d["unscheduled_count"] == 0
        assert d["scheduled"] == []
        assert d["unscheduled"] == []
        assert d["augmented_calendar"]["base_events_count"] == 0

    def test_scheduled_task_in_dict(self):
        d = solution_to_dict(_solution_with_scheduled())
        assert d["scheduled_count"] == 1
        s = d["scheduled"][0]
        assert s["label"] == "yoga"
        assert s["date"] == DATE.isoformat()
        assert s["start_minutes"] == 480
        assert s["end_minutes"] == 540
        assert s["is_standalone"] is True
        assert s["display_name"] == "Yoga"  # effective_display_name fallback
        assert "description" in s
        assert "ontology_uri" in s

    def test_parent_task_label_round_trips_through_augmented_json(self):
        """`parent_task_label` must survive serialize then reload for `L_divide`."""
        from dataclasses import replace

        task = make_task("walk_10000_steps", duration_max=120, is_dividable=True)
        morning = ScheduledTask(
            task=replace(task, label="walk_morning"),
            start_minutes=420,
            end_minutes=480,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
            parent_task_label="walk_10000_steps",
        )
        evening = ScheduledTask(
            task=replace(task, label="walk_evening"),
            start_minutes=1080,
            end_minutes=1140,
            is_standalone=True,
            concurrent_with=None,
            date=DATE,
            parent_task_label="walk_10000_steps",
        )
        ac = AugmentedCalendar(person_id="p", scheduled_tasks=[morning, evening])
        sol = SchedulingSolution(
            person_id="p",
            augmented_calendar=ac,
            tasks=[task],
            scheduled=[morning, evening],
        )
        d = solution_to_dict(sol)
        for row in d["scheduled"]:
            assert row["parent_task_label"] == "walk_10000_steps"

    def test_scheduled_task_round_trips_is_concurrent_and_is_dividable(self):
        """Regression guard for the 2026-05-14 L_merge fix.

        `_scheduled_task_to_dict` previously dropped `is_concurrent`,
        `is_dividable`, `duration_min`, and `duration_max` on the
        task; so at evaluate time `_reconstruct_solution_from_json`
        rebuilt every `RecommendedTask` with `is_concurrent=False` (the
        dataclass default).  That fired exclusion source (ii) inside
        `is_excluded_pair` for every reconstructed task and dropped
        every task from L_merge's opportunity set Φ, producing
        `G_semantic_coscheduling_merge = n/a` for every persona in
        production runs.  The fields must round-trip faithfully.
        """
        task = make_task(
            "hydrate",
            duration_min=5,
            duration_max=10,
            is_concurrent=True,
            is_dividable=True,
        )
        st = make_scheduled(task=task, start_minutes=720, end_minutes=730, date=DATE)
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[task],
            scheduled=[st],
        )
        s = solution_to_dict(sol)["scheduled"][0]
        assert s["is_concurrent"] is True
        assert s["is_dividable"] is True
        assert s["duration_min"] == 5
        assert s["duration_max"] == 10

    def test_scheduled_task_display_name_and_description(self):
        task = make_task(
            "bodyweight_circuit",
            display_name="Do a Bodyweight Circuit 💪",
            description="Short circuit with squats and push-ups.",
            ontology_uri="http://example.org/T1",
        )
        st = make_scheduled(task=task, start_minutes=480, end_minutes=540, date=DATE)
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[task],
            scheduled=[st],
        )
        d = solution_to_dict(sol)
        s = d["scheduled"][0]
        assert s["display_name"] == "Do a Bodyweight Circuit 💪"
        assert "Short circuit" in s["description"]
        assert "[http://example.org/T1]" in s["description"]
        assert s["ontology_uri"] == "http://example.org/T1"

    def test_scheduled_task_display_name_fallback_and_uri_only(self):
        task = make_task("light_yoga", ontology_uri="http://example.org/yoga")
        st = make_scheduled(task=task, start_minutes=480, end_minutes=540, date=DATE)
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st])
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        d = solution_to_dict(sol)
        s = d["scheduled"][0]
        assert s["display_name"] == "Light Yoga"  # title-case fallback
        assert "[http://example.org/yoga]" in s["description"]

    def test_unscheduled_task_in_dict(self):
        d = solution_to_dict(_solution_with_unscheduled())
        assert d["unscheduled_count"] == 1
        u = d["unscheduled"][0]
        assert u["label"] == "running"
        assert u["display_name"] == "Running"  # effective_display_name fallback
        assert "description" in u

    def test_is_json_serialisable(self):
        d = solution_to_dict(_solution_with_scheduled())
        text = json.dumps(d)  # must not raise
        loaded = json.loads(text)
        assert loaded["person_id"] == "p001"

    def test_concurrent_tasks_in_dict(self):
        """Concurrent tasks appear under 'concurrent_tasks' key in augmented_calendar."""
        task = make_task("mindful_eating")
        st = make_scheduled(
            task=task,
            start_minutes=720,
            end_minutes=760,
            is_standalone=False,
            concurrent_with="lunch",
            date=DATE,
        )
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[task],
            scheduled=[st],
        )
        d = solution_to_dict(sol)
        assert len(d["augmented_calendar"]["concurrent_tasks"]) == 1
        ct = d["augmented_calendar"]["concurrent_tasks"][0]
        assert ct["label"] == "mindful_eating"
        assert ct["concurrent_with"] == "lunch"
        assert len(d["augmented_calendar"]["standalone_tasks"]) == 0

    def test_scheduled_sorted_chronologically(self):
        """`scheduled` is sorted by `date` then `start_minutes` regardless
        of insertion order."""
        t_morning = make_task("yoga")
        t_evening = make_task("walk")
        t_next_day = make_task("run")
        st_evening = make_scheduled(
            task=t_evening, start_minutes=1080, end_minutes=1140, date=DATE
        )
        st_morning = make_scheduled(
            task=t_morning, start_minutes=420, end_minutes=480, date=DATE
        )
        st_next = make_scheduled(
            task=t_next_day,
            start_minutes=480,
            end_minutes=540,
            date=DATE + datetime.timedelta(days=1),
        )
        # Insert in non-chronological order
        ac = AugmentedCalendar(
            person_id="p001",
            scheduled_tasks=[st_evening, st_next, st_morning],
        )
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[t_morning, t_evening, t_next_day],
            scheduled=[st_evening, st_next, st_morning],
        )
        d = solution_to_dict(sol)
        labels_in_order = [s["label"] for s in d["scheduled"]]
        assert labels_in_order == ["yoga", "walk", "run"]
        # standalone_tasks is also sorted
        standalone_labels = [
            s["label"] for s in d["augmented_calendar"]["standalone_tasks"]
        ]
        assert standalone_labels == ["yoga", "walk", "run"]

    def test_concurrent_section_sorted_chronologically(self):
        """Concurrent tasks under `augmented_calendar` are also time-ordered."""
        a = make_task("mindful_eating_a")
        b = make_task("mindful_eating_b")
        st_late = make_scheduled(
            task=b,
            start_minutes=1200,
            end_minutes=1230,
            is_standalone=False,
            concurrent_with="dinner",
            date=DATE,
        )
        st_early = make_scheduled(
            task=a,
            start_minutes=720,
            end_minutes=750,
            is_standalone=False,
            concurrent_with="lunch",
            date=DATE,
        )
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st_late, st_early])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[a, b],
            scheduled=[st_late, st_early],
        )
        d = solution_to_dict(sol)
        labels = [c["label"] for c in d["augmented_calendar"]["concurrent_tasks"]]
        assert labels == ["mindful_eating_a", "mindful_eating_b"]


class TestWriteSolutionJson:
    def test_writes_file(self, tmp_path):
        sol = _solution_with_scheduled()
        out = tmp_path / "p001.json"
        result = write_solution_json(sol, out)
        assert result == out
        assert out.exists()

    def test_content_is_valid_json(self, tmp_path):
        sol = _solution_with_scheduled()
        out = tmp_path / "p001.json"
        write_solution_json(sol, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["person_id"] == "p001"

    def test_creates_parent_dirs(self, tmp_path):
        sol = _empty_solution()
        out = tmp_path / "subdir" / "p001.json"
        write_solution_json(sol, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# ics_writer
# ---------------------------------------------------------------------------


class TestMinutesToDatetime:
    def test_zero_minutes(self):
        dt = _minutes_to_datetime(DATE, 0)
        assert dt == datetime.datetime(2026, 5, 4, 0, 0)

    def test_480_minutes(self):
        dt = _minutes_to_datetime(DATE, 480)
        assert dt == datetime.datetime(2026, 5, 4, 8, 0)

    def test_minutes_overflow_next_day(self):
        dt = _minutes_to_datetime(DATE, 1500)  # 1440 + 60 = next day 01:00
        assert dt == datetime.datetime(2026, 5, 5, 1, 0)


class TestSolutionToIcalBytes:
    def test_empty_solution_produces_valid_ical(self):
        b = solution_to_ical_bytes(_empty_solution())
        assert b.startswith(b"BEGIN:VCALENDAR")
        assert b"PRODID" in b

    def test_scheduled_task_produces_vevent(self):
        sol = _solution_with_scheduled()
        b = solution_to_ical_bytes(sol)
        assert b"BEGIN:VEVENT" in b
        assert b"YOGA" in b.upper()

    def test_concurrent_task_includes_during_in_summary(self):
        task = make_task("mindful_eating")
        st = make_scheduled(
            task=task,
            start_minutes=720,
            end_minutes=760,
            is_standalone=False,
            concurrent_with="lunch",
            date=DATE,
        )
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[task],
            scheduled=[st],
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace").lower()
        assert "during lunch" in text

    def test_display_name_used_as_summary_when_set(self):
        task = make_task("bodyweight_circuit", display_name="Do a Bodyweight Circuit")
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace")
        assert "Do a Bodyweight Circuit" in text

    def test_label_used_as_fallback_summary_when_no_display_name(self):
        task = make_task("morning_yoga")  # display_name="" to effective_display_name
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace")
        assert "Morning Yoga" in text

    def test_description_with_uri_appended_in_ics(self):
        # description set without embedded citation to URI appended
        task = make_task(
            "walking",
            description="Walking at moderate pace supports cardiovascular health.",
            ontology_uri="http://example.org/HealthTasks#T8",
        )
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace").replace("\r\n ", "")
        assert "Walking at moderate pace" in text
        assert "[http://example.org/HealthTasks#T8]" in text

    def test_description_with_embedded_citation_not_duplicated_in_ics(self):
        uri = "http://example.org/yoga"
        task = make_task(
            "yoga",
            description=f"Yoga improves flexibility. [{uri}]",
            ontology_uri=uri,
        )
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace").replace("\r\n ", "")
        assert text.count(f"[{uri}]") == 1  # citation appears exactly once

    def test_description_only_no_uri_in_ics(self):
        task = make_task("stretching", description="Gentle stretching reduces tension.")
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace")
        assert "Gentle stretching reduces tension." in text

    def test_uri_only_no_description_shows_display_name_and_citation_in_ics(self):
        # empty description + URI to effective_display_name + [URI]
        task = make_task("light_yoga", ontology_uri="http://example.org/yoga")
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace").replace("\r\n ", "")
        assert "Light Yoga" in text
        assert "[http://example.org/yoga]" in text

    def test_no_description_no_uri_falls_back_to_metadata(self):
        task = make_task("running")  # no description, no ontology_uri
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace")
        assert "intensity" in text

    def test_base_calendar_events_included_when_calendar_provided(self):
        """When a CalendarTrace is passed, base events appear in the ICS."""
        task = make_task("yoga")
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        base_event = make_event(
            "office_work", start_minutes=540, end_minutes=1020, date=DATE
        )
        calendar = CalendarTrace(person_id="p001", events=[base_event])
        text = solution_to_ical_bytes(sol, calendar=calendar).decode(errors="replace")
        assert "Office Work" in text
        assert "BASE_CALENDAR" in text
        assert "AUGMENTED" in text

    def test_base_event_display_label_used_in_summary(self):
        """A per-episode display label renders instead of the catalog label."""
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[], scheduled=[]
        )
        base_event = make_event(
            "office_work",
            start_minutes=540,
            end_minutes=1020,
            date=DATE,
            display_label="external meeting",
        )
        calendar = CalendarTrace(person_id="p001", events=[base_event])
        text = solution_to_ical_bytes(sol, calendar=calendar).decode(errors="replace")
        assert "External Meeting" in text
        assert "Office Work" not in text

    def test_no_base_events_when_calendar_not_provided(self):
        """Without a CalendarTrace, only augmented events appear."""
        task = make_task("yoga")
        st = make_scheduled(task=task, date=DATE)
        ac = AugmentedCalendar(person_id="p001")
        sol = SchedulingSolution(
            person_id="p001", augmented_calendar=ac, tasks=[task], scheduled=[st]
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace")
        assert "BASE_CALENDAR" not in text

    def test_base_events_sorted_by_date_and_start(self):
        """Base events are emitted in chronological order."""
        ev_late = make_event("dinner", start_minutes=1080, end_minutes=1140, date=DATE)
        ev_early = make_event(
            "breakfast", start_minutes=420, end_minutes=480, date=DATE
        )
        cal = CalendarTrace(person_id="p001", events=[ev_late, ev_early])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=AugmentedCalendar(person_id="p001"),
        )
        text = solution_to_ical_bytes(sol, calendar=cal).decode(errors="replace")
        breakfast_pos = text.index("Breakfast")
        dinner_pos = text.index("Dinner")
        assert breakfast_pos < dinner_pos

    def test_base_and_augmented_interleaved_chronologically(self):
        """Base events and augmented tasks share one ascending timeline.

        Previously augmented entries were appended after every base event
        regardless of clock time; the writer now interleaves them by
        `(date, start_minutes)` so a calendar app shows a coherent day.
        """
        # Base events: morning breakfast, evening dinner.
        breakfast = make_event(
            "breakfast", start_minutes=420, end_minutes=480, date=DATE
        )
        dinner = make_event("dinner", start_minutes=1080, end_minutes=1140, date=DATE)
        # Augmented task at noon; should land between the two base events.
        midday_task = make_task("walk")
        st_midday = make_scheduled(
            task=midday_task, start_minutes=720, end_minutes=750, date=DATE
        )
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st_midday])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[midday_task],
            scheduled=[st_midday],
        )
        cal = CalendarTrace(person_id="p001", events=[breakfast, dinner])
        text = solution_to_ical_bytes(sol, calendar=cal).decode(errors="replace")
        breakfast_pos = text.index("Breakfast")
        walk_pos = text.index("Walk")
        dinner_pos = text.index("Dinner")
        assert breakfast_pos < walk_pos < dinner_pos

    def test_tied_start_minute_orders_base_before_augmented(self):
        """When a base event and an augmented task share the exact start
        minute, the base event is emitted first (deterministic tiebreak)."""
        base = make_event("lunch", start_minutes=720, end_minutes=780, date=DATE)
        task = make_task("mindful_eating")
        st = make_scheduled(task=task, start_minutes=720, end_minutes=740, date=DATE)
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[task],
            scheduled=[st],
        )
        cal = CalendarTrace(person_id="p001", events=[base])
        text = solution_to_ical_bytes(sol, calendar=cal).decode(errors="replace")
        assert text.index("Lunch") < text.index("Mindful Eating")

    def test_augmented_tasks_alone_are_sorted(self):
        """Without any calendar, the scheduled tasks themselves still sort
        by `(date, start_minutes)`."""
        early = make_task("yoga")
        late = make_task("walk")
        st_late = make_scheduled(
            task=late, start_minutes=1080, end_minutes=1140, date=DATE
        )
        st_early = make_scheduled(
            task=early, start_minutes=420, end_minutes=480, date=DATE
        )
        ac = AugmentedCalendar(person_id="p001", scheduled_tasks=[st_late, st_early])
        sol = SchedulingSolution(
            person_id="p001",
            augmented_calendar=ac,
            tasks=[early, late],
            scheduled=[st_late, st_early],
        )
        text = solution_to_ical_bytes(sol).decode(errors="replace")
        assert text.index("Yoga") < text.index("Walk")


class TestWriteAugmentedIcs:
    def test_writes_ics_file(self, tmp_path):
        sol = _solution_with_scheduled()
        out = tmp_path / "p001.ics"
        result = write_augmented_ics(sol, out)
        assert result == out
        assert out.exists()

    def test_file_content_is_valid_ical(self, tmp_path):
        sol = _solution_with_scheduled()
        out = tmp_path / "p001.ics"
        write_augmented_ics(sol, out)
        content = out.read_bytes()
        assert content.startswith(b"BEGIN:VCALENDAR")

    def test_creates_parent_dirs(self, tmp_path):
        sol = _empty_solution()
        out = tmp_path / "nested" / "p001.ics"
        write_augmented_ics(sol, out)
        assert out.exists()

    def test_write_with_calendar_includes_base_events(self, tmp_path):
        sol = _solution_with_scheduled()
        base_event = make_event(
            "sleep", start_minutes=1380, end_minutes=1440, date=DATE
        )
        calendar = CalendarTrace(person_id="p001", events=[base_event])
        out = tmp_path / "p001.ics"
        write_augmented_ics(sol, out, calendar=calendar)
        text = out.read_bytes().decode(errors="replace")
        assert "Sleep" in text
        assert "BASE_CALENDAR" in text


# ---------------------------------------------------------------------------
# report_writer
# ---------------------------------------------------------------------------


def _components(
    cov=0.2, cal=0.1, pref=0.1, disp=0.1, merge=0.1, spread=0.1, divide=0.1, **_legacy
) -> LossComponents:
    """Build a LossComponents for tests; ignores legacy `tt` kwarg."""
    return LossComponents(
        cov=cov,
        cal=cal,
        pref=pref,
        disp=disp,
        merge=merge,
        spread=spread,
        divide=divide,
    )


# ---- Section 1: per-person gain breakdown ---------------------------------


class TestFormatPersonGainsReport:
    """`person_instance_scheduling_gain_report.txt` content."""

    def test_empty_results_renders_header(self):
        text = format_person_gains_report([])
        assert "Person Instance Scheduling Gain Report" in text

    def test_single_person_renders_gains_with_full_names(self):
        text = format_person_gains_report([("p001", 0.4, _components())])
        assert "p001" in text
        # Total gain = 1 − 0.4 = 0.6.
        assert "Gain : 0.6000" in text
        # Long-form per-component labels with `G_` prefix.
        for full in (
            "G_recommended_task_coverage",
            "G_task_event_and_task_task_temporal_relations",
            "G_user_preference_deviation",
            "G_intensive_task_dispersion",
            "G_semantic_coscheduling_merge",
            "G_recommended_task_spread",
            "G_dividable_task_split_reward",
        ):
            assert full in text
        # Math-internal short names must NOT leak.
        assert "L_cov" not in text
        assert "  cov " not in text

    def test_per_component_gains_are_complement_of_loss(self):
        text = format_person_gains_report(
            [("p001", 0.0, _components(cov=0.2, cal=0.4))]
        )
        assert "0.8000" in text  # 1 − 0.2
        assert "0.6000" in text  # 1 − 0.4

    def test_multiple_persons_each_appear(self):
        results = [
            ("p001", 0.4, _components(cov=0.4)),
            ("p002", 0.6, _components(cov=0.6)),
        ]
        text = format_person_gains_report(results)
        assert "p001" in text
        assert "p002" in text
        # Cohort aggregates live in a SEPARATE report; not in this one.
        assert "Average total gain" not in text
        assert "Average per-component gain" not in text

    def test_empty_plan_person_renders_na_gain(self):
        """A persona with 0 generated tasks must have `Gain : n/a`."""
        grounding = {
            "per_person": {"p001": {"total": 5}, "p002": {"total": 0}},
            "grounded": 0,
            "total": 5,
            "ratio": 0.0,
        }
        text = format_person_gains_report(
            [("p001", 0.4, _components()), ("p002", 0.0, _components(cov=0.0))],
            grounding=grounding,
        )
        # p001 has 5 tasks to real gain.
        assert "Gain : 0.6000" in text
        # p002 has 0 tasks to n/a.
        assert "Gain : n/a" in text

    def test_none_merge_renders_na_in_row(self):
        no_signal = _components(merge=None)  # type: ignore[arg-type]
        text = format_person_gains_report([("p001", 0.4, no_signal)])
        assert "G_semantic_coscheduling_merge                 : n/a" in text

    def test_none_divide_renders_na_in_row(self):
        no_signal = _components(divide=None)  # type: ignore[arg-type]
        text = format_person_gains_report([("p001", 0.4, no_signal)])
        assert "G_dividable_task_split_reward                 : n/a" in text


class TestPersonGainsToDict:
    """`person_instance_scheduling_gain_report.json` shape."""

    def test_empty_results_returns_empty_persons(self):
        d = person_gains_to_dict([])
        assert d == {"persons": []}

    def test_single_result_emits_gains_with_full_names(self):
        d = person_gains_to_dict([("p001", 0.35, _components(cov=0.5))])
        assert len(d["persons"]) == 1
        person = d["persons"][0]
        assert person["person_id"] == "p001"
        assert person["total_gain"] == pytest.approx(0.65)
        assert person["gains"]["recommended_task_coverage"] == pytest.approx(0.5)
        for full in (
            "task_event_and_task_task_temporal_relations",
            "user_preference_deviation",
            "intensive_task_dispersion",
            "semantic_coscheduling_merge",
            "recommended_task_spread",
            "dividable_task_split_reward",
        ):
            assert full in person["gains"]
        # Math-internal vocabulary must NOT leak.
        assert "total_loss" not in person
        assert "components" not in person
        assert "cov" not in person.get("gains", {})

    def test_no_cohort_aggregate_keys_in_person_section(self):
        """Cohort aggregates live in a SEPARATE report; this one carries
        only per-person rows."""
        d = person_gains_to_dict([("p001", 0.4, _components())])
        for forbidden in (
            "average_total_gain",
            "average_gains",
            "scored_persons",
            "empty_plan_persons",
            "ontology_grounding",
            "telemetry",
            "preference_breakdown",
        ):
            assert forbidden not in d

    def test_is_json_serialisable(self):
        d = person_gains_to_dict([("p001", 0.5, _components())])
        loaded = json.loads(json.dumps(d))
        assert loaded["persons"][0]["person_id"] == "p001"

    def test_empty_plan_person_gets_null_gain(self):
        grounding = {
            "per_person": {
                "p001": {"grounded": 3, "total": 5, "ratio": 0.6},
                "p002": {"grounded": 0, "total": 0, "ratio": 0.0},
            },
            "grounded": 3,
            "total": 5,
            "ratio": 0.6,
        }
        d = person_gains_to_dict(
            [("p001", 0.4, _components()), ("p002", 0.0, _components(cov=0.0))],
            grounding=grounding,
        )
        by_id = {p["person_id"]: p for p in d["persons"]}
        assert by_id["p001"]["total_gain"] == pytest.approx(0.6)
        assert by_id["p002"]["total_gain"] is None

    def test_none_merge_surfaces_as_null(self):
        no_signal = _components(merge=None)  # type: ignore[arg-type]
        d = person_gains_to_dict([("p001", 0.4, no_signal)])
        assert d["persons"][0]["gains"]["semantic_coscheduling_merge"] is None

    def test_none_divide_surfaces_as_null(self):
        no_signal = _components(divide=None)  # type: ignore[arg-type]
        d = person_gains_to_dict([("p001", 0.4, no_signal)])
        assert d["persons"][0]["gains"]["dividable_task_split_reward"] is None


# ---- Section 2: cohort total + per-component averages ---------------------


class TestFormatTotalGainReport:
    """`total_scheduling_gain.txt` content."""

    def test_empty_results_renders_na(self):
        text = format_total_gain_report([])
        assert "Total Scheduling Gain" in text
        assert "Average total gain : n/a" in text

    def test_single_person_renders_average(self):
        text = format_total_gain_report([("p001", 0.4, _components())])
        # avg loss = 0.4 to avg gain = 0.6.
        assert "Average total gain : 0.6000" in text
        # No per-person rows in this section; those live in their own file.
        assert "Person : p001" not in text
        assert "  Gain : " not in text

    def test_cohort_total_gain_computed(self):
        results = [("p001", 0.4, _components()), ("p002", 0.6, _components())]
        text = format_total_gain_report(results)
        assert "Average total gain : 0.5000" in text
        # No loss vocabulary in the new report.
        assert "Average total loss" not in text

    def test_cohort_per_component_gains_rendered(self):
        results = [
            ("p001", 0.0, _components(cov=0.2, cal=0.4)),
            ("p002", 0.0, _components(cov=0.4, cal=0.0)),
        ]
        text = format_total_gain_report(results)
        assert "Average per-component gain" in text
        assert "G_recommended_task_coverage" in text
        # cov average loss = 0.3 to average gain 0.7000.
        assert "0.7000" in text
        # cal average loss = 0.2 to average gain 0.8000.
        assert "0.8000" in text

    def test_all_empty_plans_renders_na_average(self):
        grounding = {
            "per_person": {"p001": {"total": 0}, "p002": {"total": 0}},
            "grounded": 0,
            "total": 0,
            "ratio": 0.0,
        }
        text = format_total_gain_report(
            [("p001", 0.0, _components(cov=0.0)), ("p002", 0.0, _components(cov=0.0))],
            grounding=grounding,
        )
        assert "Average total gain : n/a  (every persona had 0 tasks)" in text
        assert "Empty plans excluded from average : 2" in text

    def test_partial_empty_plans_excluded(self):
        grounding = {
            "per_person": {"p001": {"total": 5}, "p002": {"total": 0}},
            "grounded": 0,
            "total": 5,
            "ratio": 0.0,
        }
        text = format_total_gain_report(
            [("p001", 0.4, _components()), ("p002", 0.0, _components(cov=0.0))],
            grounding=grounding,
        )
        # Average is over the one scored person to loss=0.4, gain=0.6.
        assert "Average total gain : 0.6000  (over 1 scored persons)" in text
        assert "Empty plans excluded from average : 1" in text

    def test_none_merge_excluded_from_average(self):
        """When `compute_l_concurrent` returns None for some personas,
        the average for that component skips them; must NOT inflate
        the cohort by treating `None` as `1.0` or `0.0`."""
        no_signal = _components(merge=None)  # type: ignore[arg-type]
        has_signal = _components(merge=1.0)  # had opportunity, missed it
        text = format_total_gain_report(
            [("p001", 0.4, no_signal), ("p002", 0.4, has_signal)]
        )
        avg_block = text.split("── Average per-component gain ──")[1]
        for line in avg_block.splitlines():
            if "G_semantic_coscheduling_merge" in line:
                # Average over the one signalled person to 0.0000.
                assert line.strip().endswith(": 0.0000"), line


class TestTotalGainToDict:
    """`total_scheduling_gain.json` shape."""

    def test_empty_results_returns_null_aggregates(self):
        d = total_gain_to_dict([])
        assert d["average_total_gain"] is None
        assert d["average_gains"] is None
        assert d["empty_plan_persons"] == 0
        assert d["scored_persons"] == 0
        # Cohort section never carries per-person rows.
        assert "persons" not in d

    def test_gain_is_complement_of_loss(self):
        d = total_gain_to_dict(
            [("p001", 0.3, _components()), ("p002", 0.7, _components())]
        )
        assert d["average_total_gain"] == pytest.approx(0.5)
        assert d["scored_persons"] == 2
        assert d["empty_plan_persons"] == 0

    def test_average_gains_per_component(self):
        d = total_gain_to_dict(
            [
                ("p001", 0.0, _components(cov=0.2, cal=0.4)),
                ("p002", 0.0, _components(cov=0.4, cal=0.0)),
            ]
        )
        assert d["average_gains"]["recommended_task_coverage"] == pytest.approx(0.7)
        assert d["average_gains"][
            "task_event_and_task_task_temporal_relations"
        ] == pytest.approx(0.8)

    def test_empty_plan_persons_excluded_from_average(self):
        grounding = {
            "per_person": {
                "p001": {"grounded": 3, "total": 5, "ratio": 0.6},
                "p002": {"grounded": 0, "total": 0, "ratio": 0.0},
            },
            "grounded": 3,
            "total": 5,
            "ratio": 0.6,
        }
        d = total_gain_to_dict(
            [("p001", 0.4, _components()), ("p002", 0.0, _components(cov=0.0))],
            grounding=grounding,
        )
        assert d["average_total_gain"] == pytest.approx(0.6)
        assert d["empty_plan_persons"] == 1
        assert d["scored_persons"] == 1

    def test_all_empty_plans_returns_null_aggregates(self):
        grounding = {
            "per_person": {
                "p001": {"grounded": 0, "total": 0, "ratio": 0.0},
                "p002": {"grounded": 0, "total": 0, "ratio": 0.0},
            },
            "grounded": 0,
            "total": 0,
            "ratio": 0.0,
        }
        d = total_gain_to_dict(
            [("p001", 0.0, _components(cov=0.0)), ("p002", 0.0, _components(cov=0.0))],
            grounding=grounding,
        )
        assert d["average_total_gain"] is None
        assert d["average_gains"] is None
        assert d["empty_plan_persons"] == 2
        assert d["scored_persons"] == 0

    def test_all_none_merge_returns_null_average_gain(self):
        d = total_gain_to_dict(
            [
                ("p001", 0.4, _components(merge=None)),  # type: ignore[arg-type]
                ("p002", 0.4, _components(merge=None)),  # type: ignore[arg-type]
            ]
        )
        assert d["average_gains"]["semantic_coscheduling_merge"] is None

    def test_no_per_person_or_section_keys(self):
        """Cohort section never carries the other reports' fields."""
        d = total_gain_to_dict([("p001", 0.4, _components())])
        for forbidden in (
            "persons",
            "ontology_grounding",
            "telemetry",
            "preference_breakdown",
        ):
            assert forbidden not in d


# ---- Section 3: ontology grounding ---------------------------------------


class TestFormatOntologyGroundingReport:
    """`ontology_grounding.txt` content."""

    _GROUNDING = {
        "per_person": {
            "p001": {"grounded": 3, "total": 5, "ratio": 0.6},
            "p002": {"grounded": 4, "total": 5, "ratio": 0.8},
        },
        "grounded": 7,
        "total": 10,
        "ratio": 0.7,
    }

    def test_renders_header(self):
        text = format_ontology_grounding_report(self._GROUNDING)
        assert "Ontology Grounding" in text

    def test_renders_grounding_line(self):
        text = format_ontology_grounding_report(self._GROUNDING)
        assert "Ontology grounding" in text
        assert "7/10" in text
        assert "70.0%" in text

    def test_zero_total_renders_na(self):
        empty = {"per_person": {}, "grounded": 0, "total": 0, "ratio": 0.0}
        text = format_ontology_grounding_report(empty)
        assert "Ontology grounding : n/a" in text

    def test_persons_short_fetched_line_rendered(self):
        grounding = {
            **self._GROUNDING,
            "persons_short_fetched": 2,
            "expected_per_person": 5,
        }
        text = format_ontology_grounding_report(grounding)
        assert "Persons short-fetched" in text
        assert "(< 5 grounded tasks)" in text

    def test_verified_line_rendered_when_validator_ran(self):
        grounding = {
            **self._GROUNDING,
            "verified": 7,
            "verified_ratio": 0.7,
            "persons_with_unverified": 0,
        }
        text = format_ontology_grounding_report(grounding)
        assert "Verified in ontology" in text
        assert "7/10" in text
        assert "70.0%" in text

    def test_verified_line_omitted_when_no_validator(self):
        text = format_ontology_grounding_report(self._GROUNDING)
        assert "Verified in ontology" not in text

    def test_unverified_persons_count_rendered(self):
        grounding = {
            **self._GROUNDING,
            "verified": 6,
            "verified_ratio": 0.6,
            "persons_with_unverified": 3,
        }
        text = format_ontology_grounding_report(grounding)
        assert "Persons with unverified URIs : 3" in text


class TestOntologyGroundingToDict:
    """`ontology_grounding.json` shape."""

    def test_passes_through_unchanged(self):
        grounding = {"grounded": 7, "total": 10, "ratio": 0.7, "per_person": {}}
        d = ontology_grounding_to_dict(grounding)
        assert d == grounding

    def test_returns_independent_copy(self):
        grounding = {"grounded": 7, "total": 10}
        d = ontology_grounding_to_dict(grounding)
        d["grounded"] = 0
        # Mutating the writer's output must not bleed back into the caller.
        assert grounding["grounded"] == 7

    def test_is_json_serialisable(self):
        d = ontology_grounding_to_dict(
            {"grounded": 7, "total": 10, "ratio": 0.7, "per_person": {"p001": {}}}
        )
        loaded = json.loads(json.dumps(d))
        assert loaded["grounded"] == 7


# ---- Section 4: telemetry ------------------------------------------------


class TestFormatTelemetryReport:
    """`telemetry.txt` content with by_stage + by_persona blocks."""

    _TELEM = {
        "scenario_id": "nutrition_l1",
        "method": "llm_agent",
        "n_persons": 12,
        "n_personas": 3,
        "wall_time_seconds_total": 553.2,
        "wall_time_seconds_avg_per_person": 46.1,
        "wall_time_seconds_avg_per_persona": 184.4,
        "tokens_total": 175200,
        "tokens_avg_per_person": 14600,
        "tokens_avg_per_persona": 58400,
        "tokens_avg_per_task": 1168,
        "estimated_usd_total": 0.029,
        "estimated_usd_avg_per_person": 0.0024,
        "estimated_usd_avg_per_persona": 0.0097,
        "fetch_attempts_avg": 1.7,
        "fetch_attempts_max": 3,
        "persons_short_fetched": 0,
        "paraphrase_acceptance_rate": 0.92,
        "gate_failures": {"length": 4, "emoji": 1, "similarity": 7},
        "by_stage": {
            "task_generation": {
                "wall_time_seconds_total": 400.0,
                "wall_time_seconds_avg_per_person": 33.3,
                "wall_time_seconds_avg_per_persona": 133.3,
                "tokens_total": 100000,
                "tokens_avg_per_person": 8333,
                "estimated_usd_total": 0.020,
            },
            "augmentation": {
                "wall_time_seconds_total": 100.0,
                "wall_time_seconds_avg_per_person": 8.3,
                "wall_time_seconds_avg_per_persona": 33.3,
                "tokens_total": 75200,
                "tokens_avg_per_person": 6266,
                "estimated_usd_total": 0.009,
                "n_tasks_attempted": 120,
                "n_tasks_placed": 100,
                "n_tasks_dropped": 20,
            },
            "evaluation": {
                "wall_time_seconds_total": 53.2,
                "wall_time_seconds_avg_per_person": 4.4,
                "wall_time_seconds_avg_per_persona": 17.7,
                "tokens_total": 0,
                "tokens_avg_per_person": 0,
                "estimated_usd_total": None,
                "embedding_lookups_total": 36,
                "uri_validations_total": 240,
            },
        },
        "by_persona": {
            "b_parttime_morning": {
                "n_persons": 4,
                "wall_time_seconds_total": 180.0,
                "wall_time_seconds_avg_per_person": 45.0,
                "tokens_total": 60000,
                "tokens_avg_per_person": 15000,
                "estimated_usd_total": 0.010,
                "estimated_usd_avg_per_person": 0.0025,
            },
            "b_parttime_evening": {
                "n_persons": 4,
                "wall_time_seconds_total": 200.0,
                "wall_time_seconds_avg_per_person": 50.0,
                "tokens_total": 65000,
                "tokens_avg_per_person": 16250,
                "estimated_usd_total": 0.011,
                "estimated_usd_avg_per_person": 0.00275,
            },
            "b_fulltime_self_employed": {
                "n_persons": 4,
                "wall_time_seconds_total": 173.2,
                "wall_time_seconds_avg_per_person": 43.3,
                "tokens_total": 50200,
                "tokens_avg_per_person": 12550,
                "estimated_usd_total": 0.008,
                "estimated_usd_avg_per_person": 0.002,
            },
        },
    }

    def test_renders_header(self):
        text = format_telemetry_report(self._TELEM)
        assert "Telemetry" in text

    def test_renders_cost_and_latency_block(self):
        text = format_telemetry_report(self._TELEM)
        assert "── Cost & Latency ──" in text
        assert "persons               : 12" in text
        assert "personas              : 3" in text
        assert "tokens_total          : 175200" in text
        assert "estimated_cost        : $0.0290" in text

    def test_renders_per_stage_block(self):
        text = format_telemetry_report(self._TELEM)
        assert "── Per-stage ──" in text
        # Each stage label appears as a sub-heading.
        assert "── Task Generation ──" in text
        assert "── Augmentation ──" in text
        assert "── Evaluation ──" in text
        # Stage-specific extras for augmentation + evaluation surface.
        assert "tasks_placed          : 100" in text
        assert "embedding_lookups     : 36" in text
        assert "uri_validations       : 240" in text

    def test_renders_per_persona_block(self):
        text = format_telemetry_report(self._TELEM)
        assert "── Per-persona ──" in text
        assert "b_parttime_morning" in text
        assert "b_parttime_evening" in text
        assert "b_fulltime_self_employed" in text
        # Per-persona row carries per-person averages, totals, cost.
        assert "4 persons" in text
        assert "$0.0100" in text or "$0.0110" in text or "$0.0080" in text

    def test_renders_paraphrase_gates_block(self):
        text = format_telemetry_report(self._TELEM)
        assert "── Paraphrase gates ──" in text
        assert "acceptance_rate       : 92.0%" in text
        assert "failed_emoji_gate     : 1" in text

    def test_estimated_cost_renders_na_when_none(self):
        telemetry = {**self._TELEM, "estimated_usd_total": None}
        text = format_telemetry_report(telemetry)
        assert "estimated_cost        : n/a" in text

    def test_zero_persons_renders_na(self):
        text = format_telemetry_report({**self._TELEM, "n_persons": 0})
        assert "Telemetry : n/a" in text
        # When zero persons no stage / persona block is rendered.
        assert "── Per-stage ──" not in text
        assert "── Per-persona ──" not in text

    def test_missing_stage_skipped_in_per_stage_block(self):
        """The renderer must skip a stage when its key is absent or None
        from `by_stage`; covers the `if stage:` False branch."""
        telem = {
            **self._TELEM,
            "by_stage": {
                "task_generation": self._TELEM["by_stage"]["task_generation"],
                "augmentation": None,
                # evaluation key absent entirely
            },
        }
        text = format_telemetry_report(telem)
        assert "── Task Generation ──" in text
        assert "── Augmentation ──" not in text
        assert "── Evaluation ──" not in text


class TestTelemetryToDict:
    """`telemetry.json` shape."""

    def test_passes_through_unchanged(self):
        telem = {"n_persons": 30, "tokens_total": 1000}
        d = telemetry_to_dict(telem)
        assert d == telem

    def test_returns_independent_copy(self):
        telem = {"n_persons": 30}
        d = telemetry_to_dict(telem)
        d["n_persons"] = 0
        assert telem["n_persons"] == 30

    def test_is_json_serialisable(self):
        d = telemetry_to_dict({"n_persons": 30, "gate_failures": {"length": 4}})
        loaded = json.loads(json.dumps(d))
        assert loaded["n_persons"] == 30


# ---- Section 5: preference breakdown -------------------------------------


class TestFormatPreferenceBreakdownReport:
    """`preference_breakdown.txt` content."""

    _BREAKDOWN = {
        "per_occurrence_duration": {
            "applicable_tasks": 12,
            "mean_loss": 0.18,
            "mape": 0.07,
        },
        "temporal_pattern_seasonality": {
            "applicable_tasks": 8,
            "mean_loss": 0.42,
        },
    }

    def test_renders_header(self):
        text = format_preference_breakdown_report(self._BREAKDOWN)
        assert "L_pref Preference Breakdown" in text

    def test_renders_each_leg(self):
        text = format_preference_breakdown_report(self._BREAKDOWN)
        assert "per_occurrence_duration" in text
        assert "temporal_pattern_seasonality" in text
        assert "applicable_tasks=12" in text
        assert "applicable_tasks=8" in text
        # 0.07 to 7.0% MAPE.
        assert "mape=7.0%" in text

    def test_none_loss_renders_na(self):
        breakdown = {
            "per_occurrence_duration": {
                "applicable_tasks": 0,
                "mean_loss": None,
            }
        }
        text = format_preference_breakdown_report(breakdown)
        assert "loss=n/a" in text

    def test_empty_breakdown_renders_na(self):
        text = format_preference_breakdown_report({})
        assert "Preference breakdown : n/a" in text


class TestPreferenceBreakdownToDict:
    """`preference_breakdown.json` shape."""

    def test_passes_through_unchanged(self):
        breakdown = {"per_occurrence_duration": {"applicable_tasks": 12}}
        d = preference_breakdown_to_dict(breakdown)
        assert d == breakdown

    def test_returns_independent_copy(self):
        breakdown = {"per_occurrence_duration": {"applicable_tasks": 12}}
        d = preference_breakdown_to_dict(breakdown)
        d.pop("per_occurrence_duration")
        # Mutating the writer's output must not bleed back into the caller.
        assert "per_occurrence_duration" in breakdown

    def test_is_json_serialisable(self):
        d = preference_breakdown_to_dict(
            {"per_occurrence_duration": {"applicable_tasks": 12, "mean_loss": 0.18}}
        )
        loaded = json.loads(json.dumps(d))
        assert "per_occurrence_duration" in loaded


# ---- Top-level writer: 5 isolated file pairs -----------------------------


class TestWriteEvaluationReports:
    """`write_evaluation_reports` writes one `.txt` + `.json` per
    section so a downstream consumer can fetch exactly the slice they
    need without parsing a combined report.
    """

    def test_always_writes_person_and_total_pairs(self, tmp_path):
        results = [("p001", 0.3, _components())]
        written = write_evaluation_reports(results, tmp_path)
        # Always-written pair keys.
        assert PERSON_REPORT_BASENAME in written
        assert TOTAL_REPORT_BASENAME in written
        # Both files exist on disk for each pair.
        for basename in (PERSON_REPORT_BASENAME, TOTAL_REPORT_BASENAME):
            txt, jsn = written[basename]
            assert txt == tmp_path / f"{basename}.txt"
            assert jsn == tmp_path / f"{basename}.json"
            assert txt.exists()
            assert jsn.exists()

    def test_optional_pairs_only_when_kwarg_supplied(self, tmp_path):
        write_evaluation_reports([("p001", 0.3, _components())], tmp_path)
        # No grounding / telemetry / preference_breakdown kwargs to those
        # files MUST NOT exist.
        for basename in (
            ONTOLOGY_REPORT_BASENAME,
            TELEMETRY_REPORT_BASENAME,
            PREFERENCE_REPORT_BASENAME,
        ):
            assert not (tmp_path / f"{basename}.txt").exists()
            assert not (tmp_path / f"{basename}.json").exists()

    def test_grounding_pair_written_when_supplied(self, tmp_path):
        grounding = {
            "per_person": {"p001": {"grounded": 7, "total": 10}},
            "grounded": 7,
            "total": 10,
            "ratio": 0.7,
        }
        written = write_evaluation_reports(
            [("p001", 0.3, _components())], tmp_path, grounding=grounding
        )
        assert ONTOLOGY_REPORT_BASENAME in written
        text = written[ONTOLOGY_REPORT_BASENAME][0].read_text(encoding="utf-8")
        assert "70.0%" in text
        data = json.loads(
            written[ONTOLOGY_REPORT_BASENAME][1].read_text(encoding="utf-8")
        )
        assert data["grounded"] == 7

    def test_telemetry_pair_written_when_supplied(self, tmp_path):
        telemetry = {
            "n_persons": 5,
            "n_personas": 2,
            "wall_time_seconds_total": 100.0,
            "wall_time_seconds_avg_per_person": 20.0,
            "wall_time_seconds_avg_per_persona": 50.0,
            "tokens_total": 1000,
            "tokens_avg_per_person": 200,
            "tokens_avg_per_persona": 500,
            "tokens_avg_per_task": 50,
            "estimated_usd_total": 0.01,
            "fetch_attempts_avg": 1.0,
            "fetch_attempts_max": 1,
            "persons_short_fetched": 0,
            "paraphrase_acceptance_rate": 1.0,
            "gate_failures": {},
        }
        written = write_evaluation_reports(
            [("p001", 0.3, _components())], tmp_path, telemetry=telemetry
        )
        assert TELEMETRY_REPORT_BASENAME in written
        text = written[TELEMETRY_REPORT_BASENAME][0].read_text(encoding="utf-8")
        assert "── Cost & Latency ──" in text
        data = json.loads(
            written[TELEMETRY_REPORT_BASENAME][1].read_text(encoding="utf-8")
        )
        assert data["n_persons"] == 5
        assert data["n_personas"] == 2

    def test_preference_breakdown_pair_written_when_supplied(self, tmp_path):
        breakdown = {
            "per_occurrence_duration": {
                "applicable_tasks": 12,
                "mean_loss": 0.18,
            }
        }
        written = write_evaluation_reports(
            [("p001", 0.3, _components())],
            tmp_path,
            preference_breakdown=breakdown,
        )
        assert PREFERENCE_REPORT_BASENAME in written
        text = written[PREFERENCE_REPORT_BASENAME][0].read_text(encoding="utf-8")
        assert "per_occurrence_duration" in text
        data = json.loads(
            written[PREFERENCE_REPORT_BASENAME][1].read_text(encoding="utf-8")
        )
        assert "per_occurrence_duration" in data

    def test_empty_preference_breakdown_does_not_write_pair(self, tmp_path):
        """`preference_breakdown={}` is treated as 'no signal'; the
        writer skips the file entirely (mirrors the falsy-truthy check
        used by the eval CLI which passes `or None`)."""
        write_evaluation_reports(
            [("p001", 0.3, _components())], tmp_path, preference_breakdown={}
        )
        assert not (tmp_path / f"{PREFERENCE_REPORT_BASENAME}.txt").exists()

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "new_dir"
        write_evaluation_reports([], out)
        assert out.is_dir()

    def test_combined_legacy_files_no_longer_written(self, tmp_path):
        """Sanity guard against regression; the old combined files
        (`scheduling_loss.txt` / `scheduling_loss.json`) must NOT
        be produced, otherwise a downstream consumer that switched to
        the per-section files would silently see two competing report
        formats side-by-side.
        """
        write_evaluation_reports([("p001", 0.3, _components())], tmp_path)
        assert not (tmp_path / "scheduling_loss.txt").exists()
        assert not (tmp_path / "scheduling_loss.json").exists()

    def test_section_files_are_independent(self, tmp_path):
        """Each section's JSON must NOT contain the other sections'
        data (the whole point of the refactor; isolation)."""
        grounding = {"grounded": 7, "total": 10, "ratio": 0.7, "per_person": {}}
        telemetry = {
            "n_personas": 1,
            "wall_time_seconds_total": 10.0,
            "wall_time_seconds_avg_per_persona": 10.0,
            "tokens_total": 100,
            "tokens_avg_per_persona": 100,
            "tokens_avg_per_task": 50,
            "estimated_usd_total": 0.001,
            "fetch_attempts_avg": 1.0,
            "fetch_attempts_max": 1,
            "personas_short_fetched": 0,
            "paraphrase_acceptance_rate": 1.0,
            "gate_failures": {},
        }
        breakdown = {"per_occurrence_duration": {"applicable_tasks": 1}}
        written = write_evaluation_reports(
            [("p001", 0.3, _components())],
            tmp_path,
            grounding=grounding,
            telemetry=telemetry,
            preference_breakdown=breakdown,
        )
        person_data = json.loads(written[PERSON_REPORT_BASENAME][1].read_text("utf-8"))
        total_data = json.loads(written[TOTAL_REPORT_BASENAME][1].read_text("utf-8"))
        for forbidden in ("ontology_grounding", "telemetry", "preference_breakdown"):
            assert forbidden not in person_data
            assert forbidden not in total_data
        # Person file carries persons but not the cohort aggregates.
        assert "persons" in person_data
        assert "average_total_gain" not in person_data
        # Total file carries cohort aggregates but not persons.
        assert "average_total_gain" in total_data
        assert "persons" not in total_data
