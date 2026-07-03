"""Run-level index.json: one entry per generated person.

Each entry carries enough metadata for downstream consumers to know which
file to read, which persona it came from, the seed used, and a short health
flag (`unsat: true` when the day solver returned no events for the entire
horizon).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.scripts.persona.config.schema import EnvironmentConfig
from src.scripts.persona.domain.schedule import PersonSchedule


@dataclass(frozen=True, slots=True)
class PersonIndexEntry:
    """One person's row in `index.json`."""

    person_id: str
    persona_id: str
    person_seed: int
    json_path: str
    ics_path: str | None
    days: int
    unsat: bool


@dataclass(frozen=True)
class RunIndex:
    """The whole run-level index, ready to serialize."""

    run_id: str
    start_date: str
    weeks: int
    persons: list[PersonIndexEntry] = field(default_factory=list)


def _is_unsat(schedule: PersonSchedule) -> bool:
    """A person is `unsat` when not a single event was placed across the run."""
    if not schedule.days:
        return True
    for day in schedule.days:
        if any(day.events.values()):
            return False
    return True


def build_index(
    schedules: list[PersonSchedule],
    *,
    environment: EnvironmentConfig,
    json_paths: list[Path],
    ics_paths: list[Path] | None = None,
    run_id: str,
) -> RunIndex:
    """Combine schedules and on-disk paths into a `RunIndex`."""
    if len(schedules) != len(json_paths):
        raise ValueError(
            "schedules and json_paths must have matching lengths "
            f"({len(schedules)} vs {len(json_paths)})"
        )
    if ics_paths is not None and len(ics_paths) != len(schedules):
        raise ValueError(
            "ics_paths length does not match schedules "
            f"({len(ics_paths)} vs {len(schedules)})"
        )

    entries: list[PersonIndexEntry] = []
    for i, sched in enumerate(schedules):
        ics_str = ics_paths[i].as_posix() if ics_paths is not None else None
        entries.append(
            PersonIndexEntry(
                person_id=sched.person_id,
                persona_id=sched.persona_id,
                person_seed=sched.person_seed,
                json_path=json_paths[i].as_posix(),
                ics_path=ics_str,
                days=len(sched.days),
                unsat=_is_unsat(sched),
            )
        )
    return RunIndex(
        run_id=run_id,
        start_date=environment.horizon.start_date.isoformat(),
        weeks=environment.horizon.weeks,
        persons=entries,
    )


def index_to_dict(index: RunIndex) -> dict[str, object]:
    """Render a `RunIndex` as a plain dict suitable for `json.dumps`."""
    return {
        "run_id": index.run_id,
        "start_date": index.start_date,
        "weeks": index.weeks,
        "persons": [
            {
                "person_id": e.person_id,
                "persona_id": e.persona_id,
                "person_seed": e.person_seed,
                "json_path": e.json_path,
                "ics_path": e.ics_path,
                "days": e.days,
                "unsat": e.unsat,
            }
            for e in index.persons
        ],
    }


def write_index(index: RunIndex, out_dir: Path | str) -> Path:
    """Write `<out_dir>/index.json` and return the path."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    target = out_dir_path / "index.json"
    target.write_text(
        json.dumps(index_to_dict(index), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


__all__ = [
    "PersonIndexEntry",
    "RunIndex",
    "build_index",
    "index_to_dict",
    "write_index",
]
