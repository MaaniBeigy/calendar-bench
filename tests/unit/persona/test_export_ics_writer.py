"""Unit tests for src.scripts.persona.export.ics_writer."""

from __future__ import annotations

import datetime as _dt

from src.scripts.persona.config.schema import (
    CalendarVariations,
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    TotalDuration,
)
from src.scripts.persona.domain.event import Catalog, EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule
from src.scripts.persona.export.ics_writer import (
    catalog_from_event_config,
    schedule_to_ics_bytes,
    write_person_ics,
    write_population_ics,
)


def _schedule(events: dict[str, list[EventInstance]]) -> PersonSchedule:
    return PersonSchedule(
        person_id="alice_0000",
        persona_id="alice",
        person_seed=42,
        days=[
            DaySchedule(
                day_index=0,
                date=_dt.date(2026, 5, 4),
                weekday="Mon",
                events=events,
                spillovers=[],
            )
        ],
    )


def _catalog_with_labels() -> Catalog:
    office_event = EventDefinition(
        name="office_work",
        category="work",
        per_event_duration=DurationRange(min=60, max=480, unit="minutes"),
        total_event_duration=TotalDuration(
            min=60, max=480, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=2),
        calendar_variations=CalendarVariations(task_labels=["standup", "meeting"]),
    )
    cfg = EventConfig(
        categories={
            "work": Category(name="work", events={"office_work": office_event}),
        }
    )
    return catalog_from_event_config(cfg)


def test_schedule_to_ics_bytes_returns_valid_ics_envelope():
    sched = _schedule(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]}
    )
    blob = schedule_to_ics_bytes(sched)
    assert blob.startswith(b"BEGIN:VCALENDAR")
    assert blob.rstrip().endswith(b"END:VCALENDAR")
    assert b"BEGIN:VEVENT" in blob
    assert b"SUMMARY:lunch (Person alice_0000)" in blob


def test_sleep_emits_three_events_in_ics():
    sched = _schedule(
        {"sleep": [EventInstance(event_name="sleep", start=1380, duration=420)]}
    )
    blob = schedule_to_ics_bytes(sched).decode("utf-8")
    # Block, go_to_bed, wake_up: three VEVENTs.
    assert blob.count("BEGIN:VEVENT") == 3
    assert "go_to_bed" in blob
    assert "wake_up" in blob


def test_task_label_rotates_per_event_via_catalog():
    sched = _schedule(
        {
            "office_work": [
                EventInstance(event_name="office_work", start=540, duration=180),
                EventInstance(event_name="office_work", start=780, duration=180),
            ],
        }
    )
    blob = schedule_to_ics_bytes(sched, catalog=_catalog_with_labels()).decode("utf-8")
    assert "SUMMARY:standup (Person alice_0000)" in blob
    assert "SUMMARY:meeting (Person alice_0000)" in blob


def test_stored_instance_label_wins_over_catalog_rotation():
    sched = _schedule(
        {
            "office_work": [
                EventInstance(
                    event_name="office_work",
                    start=540,
                    duration=180,
                    label="external meeting",
                ),
            ],
        }
    )
    blob = schedule_to_ics_bytes(sched, catalog=_catalog_with_labels()).decode("utf-8")
    assert "SUMMARY:external meeting (Person alice_0000)" in blob
    assert "standup" not in blob


def test_no_catalog_falls_back_to_event_name_summary():
    sched = _schedule(
        {
            "office_work": [
                EventInstance(event_name="office_work", start=540, duration=180),
            ],
        }
    )
    blob = schedule_to_ics_bytes(sched, catalog=None).decode("utf-8")
    assert "SUMMARY:office_work (Person alice_0000)" in blob


def test_event_unknown_to_catalog_falls_back_to_event_name():
    sched = _schedule(
        {"reading": [EventInstance(event_name="reading", start=1100, duration=60)]}
    )
    blob = schedule_to_ics_bytes(sched, catalog=_catalog_with_labels()).decode("utf-8")
    assert "SUMMARY:reading (Person alice_0000)" in blob


def test_catalog_event_without_labels_uses_event_name():
    sleep_event = EventDefinition(
        name="sleep",
        category="sleep",
        per_event_duration=DurationRange(min=360, max=540, unit="minutes"),
        total_event_duration=TotalDuration(
            min=360, max=540, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
    )
    cfg = EventConfig(
        categories={
            "sleep": Category(name="sleep", events={"sleep": sleep_event}),
        }
    )
    cat = catalog_from_event_config(cfg)
    sched = _schedule(
        {"sleep": [EventInstance(event_name="sleep", start=0, duration=300)]}
    )
    blob = schedule_to_ics_bytes(sched, catalog=cat).decode("utf-8")
    assert "SUMMARY:sleep (Person alice_0000)" in blob


def test_write_person_ics_creates_file(tmp_path):
    sched = _schedule(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]}
    )
    target = write_person_ics(sched, tmp_path)
    assert target.exists() and target.name == "alice_0000.ics"
    assert target.read_bytes().startswith(b"BEGIN:VCALENDAR")


def test_write_population_ics_writes_one_per_schedule(tmp_path):
    s1 = _schedule(
        {"lunch": [EventInstance(event_name="lunch", start=750, duration=45)]}
    )
    s2 = PersonSchedule(
        person_id="alice_0001",
        persona_id="alice",
        person_seed=43,
        days=s1.days,
    )
    paths = write_population_ics([s1, s2], tmp_path)
    assert sorted(p.name for p in paths) == ["alice_0000.ics", "alice_0001.ics"]


def test_event_crossing_midnight_rolls_date_forward():
    """An event whose start+duration > 1440 keeps a valid HH:MM in the ICS."""
    sched = _schedule(
        {"sleep": [EventInstance(event_name="sleep", start=1380, duration=600)]}
    )
    blob = schedule_to_ics_bytes(sched).decode("utf-8")
    # End time is 1380 + 600 = 1980 minutes to day+1 at 09:00.
    assert "DTEND:20260505T090000" in blob
