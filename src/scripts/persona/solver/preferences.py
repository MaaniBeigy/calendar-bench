"""Persona-stated preferences extracted from a `Person` for one day.

The persona's flat `stages` list carries every event the persona
engages with: meals, sleep, work, sports, hobbies, appointments, and
anything else the catalog defines. There is no hardcoded "lunch" or
"office_work" handling here; the day model just receives:

* `starts[event_name]` - persona-pinned start minute (HH:MM resolved).
* `durations[event_name]` - persona-pinned duration in minutes.
* `windows[event_name]` - persona-stated time-window token (e.g.
  `morning`) used to restrict `allowed_starts`.

A persona-stated value missing for some field (e.g. a stage with only
`time` set, no `duration_minutes`) just leaves that map without an
entry, and the day model falls back to the catalog spec.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from src.scripts.persona.config.schema import _NAMED_WINDOW_TOKENS
from src.scripts.persona.domain.persona import Person


def _hhmm_to_minutes(value: str) -> int | None:
    """Return minutes since midnight for an HH:MM string, or None otherwise."""
    if not isinstance(value, str) or ":" not in value:
        return None
    head, _, tail = value.partition(":")
    if not head.isdigit() or not tail.isdigit():
        return None
    return int(head) * 60 + int(tail)


def _maybe_window(time_str: str | None) -> str | None:
    """Return `time_str` if it is a window token, else None.

    A persona-stated `time` is either an HH:MM (handled by
    `_hhmm_to_minutes`), a named window (returned here), or None.
    """
    if not time_str or ":" in time_str:
        return None
    if time_str in _NAMED_WINDOW_TOKENS:
        return time_str
    return None


@dataclass(frozen=True, slots=True)
class DayPreferences:
    """Per-event preferred minute, duration, and window for a single day.

    `windows` carries persona-stated window tokens (`morning`,
    `afternoon`, ...). The day model uses them to restrict
    `allowed_starts` so a "walk in the morning" persona statement is
    actually enforced.
    """

    starts: dict[str, int]
    durations: dict[str, int]
    windows: dict[str, str]


def build_day_preferences(
    person: Person,
    *,
    weekday: str,
    date: _dt.date,
) -> DayPreferences:
    """Walk every persona-stage entry that fires on this calendar day.

    For each firing stage:
    * `time` resolved as HH:MM -> entry in `starts`.
    * `time` resolved as a window token -> entry in `windows`.
    * `duration_minutes` set -> entry in `durations`.

    A stage fires when its `date` equals `date`, or its weekly cadence
    `days` contains `weekday`. Stages that resolve to multiple matches
    (a stage with both `date` and `days`, both matching today) still
    contribute exactly one entry per map.
    """
    starts: dict[str, int] = {}
    durations: dict[str, int] = {}
    windows: dict[str, str] = {}

    for stage in person.stages:
        fires = (stage.date is not None and stage.date == date) or (
            weekday in stage.days
        )
        if not fires:
            continue

        if stage.duration_minutes is not None:
            durations[stage.name] = stage.duration_minutes

        if stage.time is not None:
            minutes = _hhmm_to_minutes(stage.time)
            if minutes is not None:
                starts[stage.name] = minutes
            else:
                window = _maybe_window(stage.time)
                if window is not None:
                    windows[stage.name] = window

    return DayPreferences(starts=starts, durations=durations, windows=windows)


__all__ = ["DayPreferences", "build_day_preferences"]
