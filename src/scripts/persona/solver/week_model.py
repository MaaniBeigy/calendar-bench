"""Per-weekday rollup of persona-stated cadence into a reusable template."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from src.scripts.persona.config.defaults import WEEKDAY_ORDER
from src.scripts.persona.domain.event import Catalog
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.persona.planner.allocator import allocate_day_counts

# Sentinel date used while building the template. Date-anchored stages
# (those with `stage.date` set) should not appear in a recurring weekly
# template; the horizon-level pipeline injects them per-day instead.
_SENTINEL_DATE = _dt.date(1970, 1, 1)


@dataclass(frozen=True)
class WeeklyTemplate:
    """Per-weekday event counts for a representative week."""

    weekday_to_counts: dict[str, dict[str, int]]

    def for_weekday(self, weekday: str) -> dict[str, int]:
        if weekday not in self.weekday_to_counts:
            raise KeyError(f"unknown weekday: {weekday!r}")
        return self.weekday_to_counts[weekday]

    def for_day_index(self, day_idx: int) -> dict[str, int]:
        return self.for_weekday(WEEKDAY_ORDER[day_idx % 7])


def build_weekly_template(
    person: Person,
    catalog: Catalog,
    *,
    window_map: WindowMap,
    total_days: int = 7,
) -> WeeklyTemplate:
    """Roll up persona-stated cadence into a 7-entry weekday lookup."""
    weekday_to_counts: dict[str, dict[str, int]] = {}
    for day_idx in range(7):
        weekday = WEEKDAY_ORDER[day_idx]
        counts = allocate_day_counts(
            person,
            catalog,
            day_idx=day_idx,
            total_days=total_days,
            date=_SENTINEL_DATE,
            window_map=window_map,
        )
        weekday_to_counts[weekday] = counts
    return WeeklyTemplate(weekday_to_counts=weekday_to_counts)


__all__ = ["WeeklyTemplate", "build_weekly_template"]
