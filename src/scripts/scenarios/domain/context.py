"""Scenarios-side context episode (loaded from persons/<pid>.json)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextEpisode:
    """One placed context episode for a single (person, day) interval."""

    name: str
    category: str
    date: datetime.date
    start_minutes: int
    end_minutes: int
    ontology_uri: str | None = None
    dimension: str | None = None
    polarity: str | None = None
    instrument: str | None = None
    theory_mappings: dict[str, Any] | None = None

    @property
    def duration(self) -> int:
        """Episode length in minutes; clamped at zero for inverted ranges."""
        return max(0, int(self.end_minutes - self.start_minutes))


__all__ = ["ContextEpisode"]
