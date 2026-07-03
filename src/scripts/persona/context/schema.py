"""Runtime types for placed context episodes."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

# Default mutually_exclusive flag per category when the persona YAML
# leaves it unset on the category block.
MUTUALLY_EXCLUSIVE_DEFAULTS: dict[str, bool] = {
    "mood_emotion": True,
    "energy_state": True,
    "physiological": False,
    "stress": False,
    "location": True,
    "social_context": True,
    "weather_environment": True,
    "behaviour_state": False,
    "capability_opportunity": False,
    "goal_intention": False,
    "trait_state": True,
}


@dataclass(frozen=True, slots=True)
class ContextEpisode:
    """One placed episode of a context member on a single day."""

    name: str
    category: str
    date: _dt.date
    start_minutes: int
    end_minutes: int
    ontology_uri: str | None = None
    dimension: str | None = None
    polarity: str | None = None
    instrument: str | None = None
    theory_mappings: dict | None = None

    @property
    def duration(self) -> int:
        return max(0, int(self.end_minutes - self.start_minutes))


__all__ = ["ContextEpisode", "MUTUALLY_EXCLUSIVE_DEFAULTS"]
