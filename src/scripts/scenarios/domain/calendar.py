"""Calendar domain types: pre-existing events and their augmented forms."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.task import ScheduledTask


@dataclass
class CalendarEvent:
    """Pre-existing calendar event x_j = (ℓ_j^x, τ_s,j^x, τ_e,j^x).

    The ground-truth concurrency and intensity fields are populated by
    calendar/loader.py from the EventDefinition in the persona pipeline's
    event_config.yaml (re-loaded via used_configs.json).

    Fields:
        label: event-type name (ℓ_j^x).
        start_minutes: start in minutes from midnight (τ_s,j^x).
        end_minutes: end in minutes from midnight (τ_e,j^x); equals
            start_minutes + duration from the persona JSON.
        date: calendar date of this event.
        is_concurrent: True when the event allows temporal overlap with
            other events (EventDefinition.is_concurrent).
        is_dividable: True when the event can be split within a day
            (EventDefinition.is_dividable).
        concurrent_with: explicit list of event labels this event may
            overlap with, regardless of is_concurrent
            (EventDefinition.concurrent_with).
        intensity: effort weight 1–5 (EventDefinition.intensity, default 1).
        display_label: per-episode display name from the persona JSON (e.g.
            an office_work episode shown as "standup"); `None` when the event
            carries no task label. Scoring and rule matching always use
            `label`; only display and the augmenter calendar view use this.
    """

    label: str
    start_minutes: int
    end_minutes: int
    date: datetime.date
    is_concurrent: bool = False
    is_dividable: bool = False
    concurrent_with: list[str] = field(default_factory=list)
    intensity: int = 1
    display_label: str | None = None

    @property
    def effective_label(self) -> str:
        """Per-episode display label when set, else the catalog label."""
        return self.display_label or self.label

    @property
    def oracle_label(self) -> str:
        """Label the evaluation judge scores: the actual label plus its parent.

        An office_work episode shown as "standup" scores as
        "standup (office_work)" so the judge sees the real activity and its
        catalog parent; events without a display label score as `label`.
        """
        if self.display_label and self.display_label != self.label:
            return f"{self.display_label} ({self.label})"
        return self.label


@dataclass
class CalendarTrace:
    """X^G; the current calendar of one person across the horizon.

    Events are expected to be sorted by (date, start_minutes), but the
    dataclass does not enforce this; sorting is the loader's responsibility.

    `contexts` carries the per-(person, day) context episodes loaded from
    `persons/<pid>.json` under the `contexts: [...]` key. Default-empty so
    fixtures and pre-context persona output keep working without changes.
    """

    person_id: str
    events: list[CalendarEvent] = field(default_factory=list)
    contexts: list[ContextEpisode] = field(default_factory=list)
    persona_id: str = ""


@dataclass
class AugmentedCalendar:
    """X^G_aug; the calendar after inserting or co-scheduling recommended tasks.

    Fields:
        person_id: identifier of the person whose calendar this is.
        base_events: all pre-existing CalendarEvents from the original calendar
            (the full original calendar is always preserved).
        scheduled_tasks: ALL ScheduledTask objects; both standalone tasks
            (is_standalone=True, concurrent_with=None) and concurrent tasks
            co-scheduled with an existing event (is_standalone=False,
            concurrent_with=<event_label>).
    """

    person_id: str
    base_events: list[CalendarEvent] = field(default_factory=list)
    scheduled_tasks: list[ScheduledTask] = field(default_factory=list)
