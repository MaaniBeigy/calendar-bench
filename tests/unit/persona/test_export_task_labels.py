"""Unit tests for src.scripts.persona.export.task_labels."""

from __future__ import annotations

import collections
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
from src.scripts.persona.export.ics_writer import catalog_from_event_config
from src.scripts.persona.export.task_labels import (
    _LabelDealer,
    _labels_for,
    assign_population_task_labels,
    assign_task_labels,
)

LABELS = ["standup", "doing research", "external meeting"]


def _office_event(*, task_labels: list[str] | None) -> EventDefinition:
    return EventDefinition(
        name="office_work",
        category="work",
        per_event_duration=DurationRange(min=60, max=480, unit="minutes"),
        total_event_duration=TotalDuration(
            min=60, max=480, scale="day", unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=3),
        calendar_variations=(
            CalendarVariations(task_labels=task_labels)
            if task_labels is not None
            else None
        ),
    )


def _catalog(office: EventDefinition) -> Catalog:
    cfg = EventConfig(
        categories={"work": Category(name="work", events={"office_work": office})}
    )
    return catalog_from_event_config(cfg)


def _schedule(person_id: str, per_day_counts: list[int]) -> PersonSchedule:
    """Build a schedule with `office_work` episodes per day plus a fixed lunch."""
    days = []
    for day_index, count in enumerate(per_day_counts):
        events: dict[str, list[EventInstance]] = {
            "lunch": [EventInstance(event_name="lunch", start=720, duration=30)],
            "office_work": [
                EventInstance(event_name="office_work", start=540 + 30 * i, duration=20)
                for i in range(count)
            ],
        }
        days.append(
            DaySchedule(
                day_index=day_index,
                date=_dt.date(2026, 6, 1) + _dt.timedelta(days=day_index),
                weekday="Mon",
                events=events,
                spillovers=[],
            )
        )
    return PersonSchedule(person_id=person_id, persona_id="p", person_seed=1, days=days)


def _office_labels(schedule: PersonSchedule) -> list[str | None]:
    return [inst.label for day in schedule.days for inst in day.events["office_work"]]


def test_labels_for_returns_task_labels_when_present():
    catalog = _catalog(_office_event(task_labels=LABELS))
    assert _labels_for("office_work", catalog) == LABELS


def test_labels_for_unknown_event_is_empty():
    catalog = _catalog(_office_event(task_labels=LABELS))
    assert _labels_for("missing", catalog) == []


def test_labels_for_event_without_calendar_variations_is_empty():
    catalog = _catalog(_office_event(task_labels=None))
    assert _labels_for("office_work", catalog) == []


def test_labels_for_event_with_empty_task_labels_is_empty():
    catalog = _catalog(_office_event(task_labels=[]))
    assert _labels_for("office_work", catalog) == []


def test_label_dealer_reshuffles_a_fresh_deck_each_pass():
    dealer = _LabelDealer(LABELS, seed=123)
    dealt = [dealer.deal() for _ in range(6)]
    # The deck holds one of each label, so every consecutive block of len(LABELS)
    # covers all of them once; never the same label drained back to back.
    assert sorted(dealt[:3]) == sorted(LABELS)
    assert sorted(dealt[3:]) == sorted(LABELS)


def test_each_person_uses_every_label_evenly():
    catalog = _catalog(_office_event(task_labels=LABELS))
    labelled = assign_task_labels(_schedule("p_0000", [1] * 12), catalog)
    counts = collections.Counter(_office_labels(labelled))
    # 12 episodes over 3 labels: a diverse, even mix - never all one label.
    assert counts == {"standup": 4, "doing research": 4, "external meeting": 4}


def test_each_block_of_k_episodes_covers_all_labels():
    catalog = _catalog(_office_event(task_labels=LABELS))
    office = _office_labels(assign_task_labels(_schedule("p_0000", [1] * 6), catalog))
    assert sorted(office[:3]) == sorted(LABELS)
    assert sorted(office[3:]) == sorted(LABELS)


def test_assign_leaves_events_without_labels_untouched():
    catalog = _catalog(_office_event(task_labels=LABELS))
    labelled = assign_task_labels(_schedule("p_0000", [1]), catalog)
    lunch = labelled.days[0].events["lunch"][0]
    assert lunch.label is None


def test_assign_event_absent_from_catalog_keeps_label_none():
    catalog = _catalog(_office_event(task_labels=None))
    labelled = assign_task_labels(_schedule("p_0000", [1, 1]), catalog)
    assert _office_labels(labelled) == [None, None]


def test_assign_is_deterministic():
    catalog = _catalog(_office_event(task_labels=LABELS))
    first = assign_task_labels(_schedule("p_0000", [1, 1, 1]), catalog)
    second = assign_task_labels(_schedule("p_0000", [1, 1, 1]), catalog)
    assert _office_labels(first) == _office_labels(second)


def test_distinct_persons_get_distinct_label_sequences():
    catalog = _catalog(_office_event(task_labels=LABELS))
    sequences = {
        pid: tuple(
            _office_labels(assign_task_labels(_schedule(pid, [1, 1, 1]), catalog))
        )
        for pid in ("p_0000", "p_0001", "p_0002", "p_0003", "p_0004")
    }
    assert len(set(sequences.values())) > 1


def test_multiple_episodes_same_day_get_distinct_labels():
    catalog = _catalog(_office_event(task_labels=LABELS))
    labelled = assign_task_labels(_schedule("p_0000", [3]), catalog)
    same_day = [inst.label for inst in labelled.days[0].events["office_work"]]
    assert sorted(same_day) == sorted(LABELS)


def test_assign_population_maps_over_every_schedule():
    catalog = _catalog(_office_event(task_labels=LABELS))
    schedules = [_schedule("p_0000", [1]), _schedule("p_0001", [1])]
    labelled = assign_population_task_labels(schedules, catalog)
    assert len(labelled) == 2
    assert all(_office_labels(s)[0] in LABELS for s in labelled)
