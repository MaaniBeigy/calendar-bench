"""Output containers for the solver and export stages."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.event import EventInstance


@dataclass(frozen=True, slots=True)
class Spillover:
    """Portion of a day-d event that crosses midnight into day d+1."""

    event_name: str
    start: int
    duration: int
    orig_start: int
    orig_duration: int
    event_idx: int = 0


@dataclass(frozen=True)
class DaySchedule:
    """All events realized on a single day for a single person."""

    day_index: int
    date: _dt.date
    weekday: str
    events: dict[str, list[EventInstance]] = field(default_factory=dict)
    spillovers: list[Spillover] = field(default_factory=list)


@dataclass(frozen=True)
class PersonSchedule:
    """Full per-person schedule across the run horizon."""

    person_id: str
    persona_id: str
    person_seed: int
    days: list[DaySchedule] = field(default_factory=list)
    contexts: list[ContextEpisode] = field(default_factory=list)
