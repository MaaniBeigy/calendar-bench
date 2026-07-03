"""Resolve a `TimeframeSpec` against a persona-run horizon.

A scenario's `timeframe:` block (week- or date-form) is validated for
self-consistency at schema load. Bounds against the actual horizon
(start_date, weeks) live here because the schema does not see
`environment.yaml`.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from src.scripts.scenarios.config.schema import TimeframeSpec


class TimeframeError(ValueError):
    """Raised when a TimeframeSpec does not fit inside the persona horizon."""


@dataclass(frozen=True)
class ResolvedWindow:
    """A scenario's sub-range of the persona horizon."""

    start_date: datetime.date
    weeks: int

    @property
    def days(self) -> int:
        return self.weeks * 7

    @property
    def end_date_inclusive(self) -> datetime.date:
        return self.start_date + datetime.timedelta(days=self.days - 1)


def resolve_timeframe(
    spec: TimeframeSpec | None,
    horizon_start_date: datetime.date,
    horizon_weeks: int,
    scenario_id: str = "",
) -> ResolvedWindow:
    """Resolve `spec` against the horizon; return the full horizon when spec is None."""
    if horizon_weeks < 1:
        raise TimeframeError(f"horizon_weeks must be >= 1; got {horizon_weeks}")
    if spec is None:
        return ResolvedWindow(horizon_start_date, horizon_weeks)

    where = f" (scenario={scenario_id!r})" if scenario_id else ""

    if spec.scale == "week":
        start_week = int(spec.start)
        end_week = int(spec.end)
        if end_week > horizon_weeks:
            raise TimeframeError(
                f"timeframe end (week {end_week}) is past horizon "
                f"({horizon_weeks} weeks){where}"
            )
        start_date = horizon_start_date + datetime.timedelta(days=(start_week - 1) * 7)
        return ResolvedWindow(start_date, end_week - start_week + 1)

    start = spec.start
    end = spec.end
    assert isinstance(start, datetime.date) and isinstance(end, datetime.date)

    horizon_end_inclusive = horizon_start_date + datetime.timedelta(
        days=horizon_weeks * 7 - 1
    )
    if start < horizon_start_date:
        raise TimeframeError(
            f"timeframe start {start.isoformat()} is before horizon start "
            f"{horizon_start_date.isoformat()}{where}"
        )
    if end > horizon_end_inclusive:
        raise TimeframeError(
            f"timeframe end {end.isoformat()} is past horizon end "
            f"{horizon_end_inclusive.isoformat()}{where}"
        )
    offset = (start - horizon_start_date).days
    if offset % 7 != 0:
        raise TimeframeError(
            f"timeframe start {start.isoformat()} is not week-aligned "
            f"with horizon start {horizon_start_date.isoformat()} "
            f"(offset {offset} days){where}"
        )
    span_days = (end - start).days + 1
    return ResolvedWindow(start, span_days // 7)
