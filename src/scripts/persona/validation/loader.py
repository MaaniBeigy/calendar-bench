"""Read per-person JSON files back into `PersonSchedule` for re-validation.

The CLI's `validate` subcommand uses this to check a previously generated
run without re-running the solver. The shape on disk is whatever
`export.json_writer` produced.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule, Spillover


def _events_from_dict(events_dict: dict) -> dict[str, list[EventInstance]]:
    out: dict[str, list[EventInstance]] = {}
    for name, events in events_dict.items():
        out[name] = [
            EventInstance(
                event_name=name,
                start=int(ev["start"]),
                duration=int(ev["duration"]),
                label=ev.get("label"),
            )
            for ev in events
        ]
    return out


def _spillover_from_dict(d: dict) -> Spillover:
    return Spillover(
        event_name=d["type"],
        start=int(d["start"]),
        duration=int(d["duration"]),
        orig_start=int(d["orig_start"]),
        orig_duration=int(d["orig_duration"]),
        event_idx=int(d.get("event_idx", 0)),
    )


def _day_from_dict(payload: dict) -> DaySchedule:
    return DaySchedule(
        day_index=int(payload["day_index"]),
        date=_dt.date.fromisoformat(payload["date"]),
        weekday=str(payload["weekday"]),
        events=_events_from_dict(payload.get("events", {})),
        spillovers=[_spillover_from_dict(s) for s in payload.get("spillovers", [])],
    )


def schedule_from_dict(payload: dict) -> PersonSchedule:
    """Inverse of `schedule_to_dict` in `export.json_writer`."""
    return PersonSchedule(
        person_id=str(payload["person_id"]),
        persona_id=str(payload["persona_id"]),
        person_seed=int(payload["person_seed"]),
        days=[_day_from_dict(d) for d in payload.get("days", [])],
    )


def load_schedule(path: Path | str) -> PersonSchedule:
    """Load one `<person_id>.json` from disk."""
    text = Path(path).read_text(encoding="utf-8")
    return schedule_from_dict(json.loads(text))


def load_schedules(persons_dir: Path | str) -> list[PersonSchedule]:
    """Load every `*.json` in `persons_dir`, sorted by person_id."""
    paths = sorted(Path(persons_dir).glob("*.json"))
    return [load_schedule(p) for p in paths]


__all__ = ["load_schedule", "load_schedules", "schedule_from_dict"]
