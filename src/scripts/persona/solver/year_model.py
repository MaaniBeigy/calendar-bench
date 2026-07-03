"""Date-anchored events: persona-stated stages with a one-off `date`.

Yearly anchors come from any `PersonaEventStage` whose `date` field is
set. Each carries a date, an optional time (HH:MM or named window),
and an optional duration. The yearly pass resolves those into
`(day_index, start_minute, duration)` triples, which the day model
honours as occupied ranges. No z3 is involved at this level: a
fixed-time anchor is a deterministic obstacle, not a search variable.

Stages without a `time` or `duration_minutes` (only `date` set) are
not anchored as obstacles - they will fire through the regular day
solver instead.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from src.scripts.persona.config.schema import PersonaEventStage
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.domain.time_windows import WindowMap, parse_time_token

DAY_MINUTES = 1440


@dataclass(frozen=True, slots=True)
class YearlyAnchor:
    """One persona-stated, date-anchored event."""

    name: str
    day_index: int
    start: int
    duration: int

    @property
    def end(self) -> int:
        return self.start + self.duration


def _resolve_anchor_minutes(
    stage: PersonaEventStage, window_map: WindowMap
) -> int | None:
    """Map a stage's `time` to a concrete minute-of-day.

    Returns:
    * the HH:MM minute count for an exact time;
    * the window-start minute for a named window;
    * `None` if `stage.time` is None (the stage has no time pin and so
      cannot be anchored as a fixed obstacle).
    """
    if stage.time is None:
        return None
    anchor = parse_time_token(stage.time)
    if anchor.kind == "exact":
        return anchor.minutes
    win_start, _ = window_map.get(anchor.token)
    return win_start


def plan_yearly_anchors(
    person: Person,
    *,
    start_date: _dt.date,
    horizon_days: int,
    window_map: WindowMap,
) -> list[YearlyAnchor]:
    """Resolve persona stages with a one-off date that fall inside the horizon.

    A stage qualifies as an anchor when:
    * `stage.date` is set,
    * the date is inside `[start_date, start_date + horizon_days)`,
    * `stage.time` is set (so a concrete minute can be resolved),
    * `stage.duration_minutes` is set (so a concrete obstacle range can
      be reserved).

    Stages with `date` but missing `time` or `duration_minutes` flow
    through the regular day solver - they are not "yearly anchors".
    The output is sorted by `(day_index, start, name)` for deterministic
    downstream consumption.
    """
    if horizon_days < 0:
        raise ValueError(f"horizon_days must be >= 0, got {horizon_days}")
    if not person.stages:
        return []

    end_date = start_date + _dt.timedelta(days=horizon_days)
    anchors: list[YearlyAnchor] = []
    for stage in person.stages:
        if stage.date is None:
            continue
        if stage.date < start_date or stage.date >= end_date:
            continue
        if stage.duration_minutes is None:
            continue
        start = _resolve_anchor_minutes(stage, window_map)
        if start is None:
            continue
        duration = stage.duration_minutes
        day_index = (stage.date - start_date).days
        if start < 0 or start + duration > DAY_MINUTES:
            raise ValueError(
                f"stage {stage.name!r} on {stage.date} would extend past "
                f"midnight (start={start}, duration={duration})"
            )
        anchors.append(
            YearlyAnchor(
                name=stage.name,
                day_index=day_index,
                start=start,
                duration=duration,
            )
        )
    anchors.sort(key=lambda a: (a.day_index, a.start, a.name))
    return anchors


def anchors_for_day(anchors: list[YearlyAnchor], day_index: int) -> list[YearlyAnchor]:
    """Return the subset of `anchors` falling on `day_index`."""
    return [a for a in anchors if a.day_index == day_index]


def occupied_from_anchors(
    anchors: list[YearlyAnchor],
) -> list[tuple[int, int]]:
    """Translate anchors into `[start, end)` ranges for the day solver."""
    return [(a.start, a.end) for a in anchors if a.duration > 0]


__all__ = [
    "YearlyAnchor",
    "anchors_for_day",
    "occupied_from_anchors",
    "plan_yearly_anchors",
]
