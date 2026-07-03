"""Serialize a `PersonSchedule` to one JSON file per person.

The on-disk shape mirrors the legacy multi-person json: a top-level object per
person, with a `days` list and `events` keyed by event name. The persona is
embedded so the file is self-describing, and per-day objects keep an explicit
weekday so consumers do not have to re-derive it from the date.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import (
    DaySchedule,
    EventInstance,
    PersonSchedule,
    Spillover,
)


def _event_to_dict(event: EventInstance) -> dict[str, int | str]:
    out: dict[str, int | str] = {"start": event.start, "duration": event.duration}
    if event.label is not None:
        out["label"] = event.label
    return out


def _spillover_to_dict(spill: Spillover) -> dict[str, int | str]:
    return {
        "type": spill.event_name,
        "start": spill.start,
        "duration": spill.duration,
        "orig_start": spill.orig_start,
        "orig_duration": spill.orig_duration,
        "event_idx": spill.event_idx,
    }


def _day_to_dict(day: DaySchedule) -> dict[str, object]:
    events_dict: dict[str, list[dict[str, int | str]]] = {}
    for event_name in sorted(day.events):
        events_dict[event_name] = [_event_to_dict(e) for e in day.events[event_name]]
    return {
        "day_index": day.day_index,
        "date": day.date.isoformat(),
        "weekday": day.weekday,
        "events": events_dict,
        "spillovers": [_spillover_to_dict(s) for s in day.spillovers],
    }


def _persona_payload(person: Person) -> dict[str, object]:
    """Render the persona snapshot embedded in each person json.

    The pydantic dump rounds-trips nicely; we route dates through `mode='json'`
    so YAML date objects become ISO strings.
    """
    return person.model_dump(mode="json", by_alias=True)


def _context_to_dict(episode: ContextEpisode) -> dict[str, object]:
    return {
        "name": episode.name,
        "category": episode.category,
        "date": episode.date.isoformat(),
        "start_minutes": episode.start_minutes,
        "end_minutes": episode.end_minutes,
        "ontology_uri": episode.ontology_uri,
        "dimension": episode.dimension,
        "polarity": episode.polarity,
        "instrument": episode.instrument,
        "theory_mappings": episode.theory_mappings,
    }


def schedule_to_dict(person: Person, schedule: PersonSchedule) -> dict[str, object]:
    """Build the in-memory dict that gets written as `person_<id>.json`."""
    return {
        "person_id": schedule.person_id,
        "persona_id": schedule.persona_id,
        "person_seed": schedule.person_seed,
        "persona": _persona_payload(person),
        "days": [_day_to_dict(d) for d in schedule.days],
        "contexts": [_context_to_dict(c) for c in schedule.contexts],
    }


def write_person_json(
    person: Person,
    schedule: PersonSchedule,
    out_dir: Path | str,
) -> Path:
    """Write `<out_dir>/person_<person_id>.json` and return its path."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    payload = schedule_to_dict(person, schedule)
    target = out_dir_path / f"{schedule.person_id}.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def write_population_json(
    persons: list[Person],
    schedules: list[PersonSchedule],
    out_dir: Path | str,
) -> list[Path]:
    """Write one json per person. Pairs are matched positionally."""
    if len(persons) != len(schedules):
        raise ValueError(
            f"persons ({len(persons)}) and schedules ({len(schedules)}) "
            "lengths must match"
        )
    return [write_person_json(p, s, out_dir) for p, s in zip(persons, schedules)]


__all__ = [
    "schedule_to_dict",
    "write_person_json",
    "write_population_json",
]
