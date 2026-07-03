"""Assign random, diverse per-episode task labels to a person schedule.

A catalog event may define `calendar_variations.task_labels` (e.g. office_work with
[standup, doing research, external meeting]). This module adds one label per
episode to the schedule's `EventInstance` objects so the label is persisted in
the JSON and seen by every downstream consumer, not just recomputed at ICS time.

Labels are drawn from a per-person shuffled deck: a single person's episodes mix
all of the labels in random order, so no one person is stuck on one label, every
block of `len(labels)` episodes covers each label once, and the counts stay even.
Different persons get different sequences, and the draw is seeded from the person
so a run reproduces with the same seed.
"""

from __future__ import annotations

import hashlib
import random

from src.scripts.persona.domain.event import Catalog, EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule


def _seed_for(person_seed: int, person_id: str, event_name: str) -> int:
    """Deterministic per-(person, event) seed tied to the run seed."""
    digest = hashlib.sha256(
        f"{person_seed}|{person_id}|{event_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


class _LabelDealer:
    """Deals labels from a reshuffled deck so each person stays diverse."""

    def __init__(self, labels: list[str], seed: int) -> None:
        self._labels = list(labels)
        self._rng = random.Random(seed)
        self._deck: list[str] = []

    def deal(self) -> str:
        """Return the next label, reshuffling a fresh deck when one runs out."""
        if not self._deck:
            self._deck = list(self._labels)
            self._rng.shuffle(self._deck)
        return self._deck.pop()


def _labels_for(event_name: str, catalog: Catalog) -> list[str]:
    """Return the catalog task labels for an event, or an empty list."""
    if event_name not in catalog:
        return []
    export = catalog.get(event_name).calendar_variations
    if export is None or not export.task_labels:
        return []
    return list(export.task_labels)


def assign_task_labels(schedule: PersonSchedule, catalog: Catalog) -> PersonSchedule:
    """Return a copy of `schedule` whose episodes carry random, diverse labels.

    Events whose catalog entry defines `calendar_variations.task_labels` get one
    label per episode, dealt from the person's shuffled deck so the labels are
    randomly ordered yet evenly spread. Every other event is left untouched
    (its `label` stays `None`).
    """
    dealers: dict[str, _LabelDealer] = {}
    new_days: list[DaySchedule] = []
    for day in schedule.days:
        new_events: dict[str, list[EventInstance]] = {}
        for event_name, instances in day.events.items():
            labels = _labels_for(event_name, catalog)
            if not labels:
                new_events[event_name] = list(instances)
                continue
            dealer = dealers.get(event_name)
            if dealer is None:
                dealer = _LabelDealer(
                    labels,
                    _seed_for(schedule.person_seed, schedule.person_id, event_name),
                )
                dealers[event_name] = dealer
            new_events[event_name] = [
                EventInstance(
                    event_name=inst.event_name,
                    start=inst.start,
                    duration=inst.duration,
                    label=dealer.deal(),
                )
                for inst in instances
            ]
        new_days.append(
            DaySchedule(
                day_index=day.day_index,
                date=day.date,
                weekday=day.weekday,
                events=new_events,
                spillovers=list(day.spillovers),
            )
        )
    return PersonSchedule(
        person_id=schedule.person_id,
        persona_id=schedule.persona_id,
        person_seed=schedule.person_seed,
        days=new_days,
        contexts=list(schedule.contexts),
    )


def assign_population_task_labels(
    schedules: list[PersonSchedule], catalog: Catalog
) -> list[PersonSchedule]:
    """Apply `assign_task_labels` to every schedule in a population."""
    return [assign_task_labels(s, catalog) for s in schedules]
