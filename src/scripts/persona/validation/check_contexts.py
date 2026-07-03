"""Validate placed context episodes against the declared per-persona shape."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from src.scripts.persona.config.schema import ContextMember
from src.scripts.persona.context.resolver import resolve_for_person
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.schedule import PersonSchedule

DEFAULT_TOLERANCE = 1


@dataclass(frozen=True, slots=True)
class ContextViolation:
    """One mismatch between configured context shape and placed episodes."""

    person_id: str
    category: str
    member: str | None
    kind: str  # episode_count_mismatch | duration_band_violation | mutually_exclusive_overlap
    detail: str


def check_context_distributions(
    persons: Iterable[Person],
    schedules: dict[str, PersonSchedule],
    *,
    tolerance: int = DEFAULT_TOLERANCE,
) -> list[ContextViolation]:
    """Walk every person and surface configuration / placement mismatches."""
    out: list[ContextViolation] = []
    for person in persons:
        schedule = schedules.get(person.person_id)
        if schedule is None:
            continue
        if not person.contexts:
            continue
        resolved = resolve_for_person(person, person.contexts)
        out.extend(_check_one_person(person, schedule, resolved, tolerance))
    return out


def _check_one_person(
    person: Person,
    schedule: PersonSchedule,
    resolved,
    tolerance: int,
) -> list[ContextViolation]:
    out: list[ContextViolation] = []
    by_member: dict[tuple[str, str], list[ContextEpisode]] = defaultdict(list)
    for ep in schedule.contexts:
        by_member[(ep.category, ep.name)].append(ep)

    horizon_days = max(1, len(schedule.days))
    for cat_name, category in resolved.items():
        out.extend(
            _check_mutual_exclusion(
                person, schedule, cat_name, category.mutually_exclusive
            )
        )
        for member_name, member in category.members.items():
            placements = by_member.get((cat_name, member_name), [])
            out.extend(
                _check_episode_counts(
                    person,
                    cat_name,
                    member_name,
                    member,
                    placements,
                    horizon_days=horizon_days,
                    tolerance=tolerance,
                )
            )
            out.extend(
                _check_duration_bands(
                    person,
                    cat_name,
                    member_name,
                    member,
                    placements,
                )
            )
    return out


def _check_episode_counts(
    person: Person,
    category: str,
    member_name: str,
    member: ContextMember,
    placements: list[ContextEpisode],
    *,
    horizon_days: int,
    tolerance: int,
) -> list[ContextViolation]:
    band = member.total_event_episodes
    expected_min, expected_max = _expand_count_band_to_horizon(
        band.scale, band.min, band.max, horizon_days
    )
    realized = len(placements)
    if realized < expected_min - tolerance:
        return [
            ContextViolation(
                person_id=person.person_id,
                category=category,
                member=member_name,
                kind="episode_count_mismatch",
                detail=(
                    f"realized {realized} episodes, expected "
                    f">= {expected_min} (scale={band.scale})"
                ),
            )
        ]
    if realized > expected_max + tolerance:
        return [
            ContextViolation(
                person_id=person.person_id,
                category=category,
                member=member_name,
                kind="episode_count_mismatch",
                detail=(
                    f"realized {realized} episodes, expected "
                    f"<= {expected_max} (scale={band.scale})"
                ),
            )
        ]
    return []


def _expand_count_band_to_horizon(
    scale: str, lo: int, hi: int, horizon_days: int
) -> tuple[int, int]:
    """Translate a per-scale band to a horizon-total band."""
    if scale == "day":
        return lo * horizon_days, hi * horizon_days
    if scale == "week":
        weeks = max(1, horizon_days // 7)
        return lo * weeks, hi * weeks
    if scale == "month":
        months = max(1, horizon_days // 30)
        return lo * months, hi * months
    if scale == "season":
        seasons = max(1, horizon_days // 90)
        return lo * seasons, hi * seasons
    return lo, hi


def _check_duration_bands(
    person: Person,
    category: str,
    member_name: str,
    member: ContextMember,
    placements: list[ContextEpisode],
) -> list[ContextViolation]:
    band = member.per_event_duration
    factor = 60 if band.unit == "hours" else 1
    lo, hi = band.min * factor, band.max * factor
    out: list[ContextViolation] = []
    for ep in placements:
        if ep.duration < lo or ep.duration > hi:
            out.append(
                ContextViolation(
                    person_id=person.person_id,
                    category=category,
                    member=member_name,
                    kind="duration_band_violation",
                    detail=(
                        f"episode duration {ep.duration} min outside band "
                        f"[{lo}, {hi}] on {ep.date.isoformat()}"
                    ),
                )
            )
    return out


def _check_mutual_exclusion(
    person: Person,
    schedule: PersonSchedule,
    category: str,
    mutually_exclusive: bool,
) -> list[ContextViolation]:
    if not mutually_exclusive:
        return []
    in_category: dict = defaultdict(list)
    for ep in schedule.contexts:
        if ep.category == category:
            in_category[ep.date].append(ep)
    out: list[ContextViolation] = []
    for date, episodes in in_category.items():
        episodes.sort(key=lambda e: e.start_minutes)
        for i in range(len(episodes)):
            for j in range(i + 1, len(episodes)):
                a, b = episodes[i], episodes[j]
                if a.end_minutes <= b.start_minutes:
                    continue
                if b.end_minutes <= a.start_minutes:  # pragma: no cover - defensive
                    continue
                out.append(
                    ContextViolation(
                        person_id=person.person_id,
                        category=category,
                        member=None,
                        kind="mutually_exclusive_overlap",
                        detail=(
                            f"{a.name}[{a.start_minutes}-{a.end_minutes}] "
                            f"overlaps {b.name}[{b.start_minutes}-{b.end_minutes}] "
                            f"on {date.isoformat()}"
                        ),
                    )
                )
    return out


__all__ = [
    "ContextViolation",
    "DEFAULT_TOLERANCE",
    "check_context_distributions",
]
