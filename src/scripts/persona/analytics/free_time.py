"""Free-time analytics: how much of each day is left unscheduled.

A day's free fraction is `(1440 - scheduled_minutes) / 1440`. Scheduled
minutes count every emitted event, including spillovers' originating event
on the previous day, so an overnight sleep that consumes the night is
fully reflected on day d's number, not split across d and d+1.

Per-person stats roll those daily fractions up to mean / min / max. A
population-level rollup groups persons by one characteristic axis
(default `occupation_status`) so we can compare avg free time across
groups such as student / fulltime / parttime or has_kids true / false.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

DAY_MINUTES = 1440
DEFAULT_GROUP_BY = "occupation_status"
UNKNOWN_GROUP = "(unknown)"


def _format_group_value(value: str | bool | int | float | None) -> str:
    """Render a characteristic value as a stable group label."""
    if value is None:
        return UNKNOWN_GROUP
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


@dataclass(frozen=True, slots=True)
class PersonFreeTime:
    """Daily free-time fractions for one person plus the rollup."""

    person_id: str
    group_value: str
    daily_free_fractions: tuple[float, ...]
    avg_free: float
    min_free: float
    max_free: float


@dataclass(frozen=True)
class GroupFreeTime:
    """Population-level rollup for one characteristic-value group."""

    group_value: str
    person_count: int
    avg_free: float
    min_free: float
    max_free: float


@dataclass(frozen=True)
class FreeTimeReport:
    """Bundle of per-person and per-group rollups."""

    group_by: str = DEFAULT_GROUP_BY
    persons: list[PersonFreeTime] = field(default_factory=list)
    by_group: list[GroupFreeTime] = field(default_factory=list)


def free_minutes_in_day(day: DaySchedule) -> int:
    """Return free minutes on `day`. Each event's full duration is subtracted,
    so overnight events still count entirely on the day they begin.
    """
    scheduled = 0
    for events in day.events.values():
        for ev in events:
            scheduled += ev.duration
    return max(0, DAY_MINUTES - scheduled)


def free_fraction_in_day(day: DaySchedule) -> float:
    """Free minutes as a fraction of a day."""
    return free_minutes_in_day(day) / DAY_MINUTES


def person_free_time(schedule: PersonSchedule, group_value: str) -> PersonFreeTime:
    """Compute per-day fractions plus the avg / min / max rollup for one person."""
    fractions = tuple(free_fraction_in_day(d) for d in schedule.days)
    if not fractions:
        return PersonFreeTime(
            person_id=schedule.person_id,
            group_value=group_value,
            daily_free_fractions=(),
            avg_free=0.0,
            min_free=0.0,
            max_free=0.0,
        )
    return PersonFreeTime(
        person_id=schedule.person_id,
        group_value=group_value,
        daily_free_fractions=fractions,
        avg_free=sum(fractions) / len(fractions),
        min_free=min(fractions),
        max_free=max(fractions),
    )


def population_free_time(
    persons: list[Person],
    schedules: list[PersonSchedule],
    group_by: str = DEFAULT_GROUP_BY,
) -> list[PersonFreeTime]:
    """Compute per-person stats for the whole population, paired by index."""
    if len(persons) != len(schedules):
        raise ValueError(
            f"persons ({len(persons)}) and schedules ({len(schedules)}) "
            "lengths must match"
        )
    return [
        person_free_time(s, _format_group_value(p.characteristics.get(group_by)))
        for p, s in zip(persons, schedules)
    ]


def aggregate_by_group(
    stats: list[PersonFreeTime],
) -> list[GroupFreeTime]:
    """Group `stats` by their group value and roll up avg / min / max.

    Output is sorted by group value so the report is deterministic.
    """
    grouped: dict[str, list[PersonFreeTime]] = defaultdict(list)
    for s in stats:
        grouped[s.group_value].append(s)
    out: list[GroupFreeTime] = []
    for group_value in sorted(grouped):
        members = grouped[group_value]
        averages = [m.avg_free for m in members]
        out.append(
            GroupFreeTime(
                group_value=group_value,
                person_count=len(members),
                avg_free=sum(averages) / len(averages),
                min_free=min(m.min_free for m in members),
                max_free=max(m.max_free for m in members),
            )
        )
    return out


def build_free_time_report(
    persons: list[Person],
    schedules: list[PersonSchedule],
    group_by: str = DEFAULT_GROUP_BY,
) -> FreeTimeReport:
    """Top-level convenience that pairs per-person stats with the group rollup."""
    per_person = population_free_time(persons, schedules, group_by=group_by)
    return FreeTimeReport(
        group_by=group_by,
        persons=per_person,
        by_group=aggregate_by_group(per_person),
    )


def _fmt_pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def render_report(report: FreeTimeReport) -> str:
    """Render the report into the legacy text format, sorted for stability."""
    lines: list[str] = []
    lines.append(f"=== Free-time summary by {report.group_by} ===")
    if not report.by_group:
        lines.append("No persons in the population.")
    else:
        for grp in report.by_group:
            lines.append(
                f"- {grp.group_value}: persons={grp.person_count} | "
                f"avg={_fmt_pct(grp.avg_free)} | "
                f"min={_fmt_pct(grp.min_free)} | "
                f"max={_fmt_pct(grp.max_free)}"
            )
    lines.append("")
    lines.append("=== Free-time summary per person ===")
    if not report.persons:
        lines.append("No persons in the population.")
    else:
        for stats in sorted(report.persons, key=lambda s: s.person_id):
            lines.append(
                f"  {stats.person_id} ({stats.group_value}): "
                f"avg={_fmt_pct(stats.avg_free)} | "
                f"min={_fmt_pct(stats.min_free)} | "
                f"max={_fmt_pct(stats.max_free)}"
            )
    return "\n".join(lines) + "\n"


def write_report(report: FreeTimeReport, out_dir: Path | str) -> Path:
    """Write `<out_dir>/free_time_report.txt` and return the path."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    target = out_dir_path / "free_time_report.txt"
    target.write_text(render_report(report), encoding="utf-8")
    return target


__all__ = [
    "DAY_MINUTES",
    "DEFAULT_GROUP_BY",
    "FreeTimeReport",
    "GroupFreeTime",
    "PersonFreeTime",
    "UNKNOWN_GROUP",
    "aggregate_by_group",
    "build_free_time_report",
    "free_fraction_in_day",
    "free_minutes_in_day",
    "person_free_time",
    "population_free_time",
    "render_report",
    "write_report",
]
