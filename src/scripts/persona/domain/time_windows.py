"""Time-window resolution: HH:MM and named windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.scripts.persona.config.schema import WindowRange

NAMED_WINDOWS: frozenset[str] = frozenset(
    {"early_morning", "morning", "afternoon", "evening", "night"}
)

TimeAnchorKind = Literal["exact", "window"]


@dataclass(frozen=True, slots=True)
class TimeAnchor:
    """A persona-stated time, normalized into a tagged form."""

    kind: TimeAnchorKind
    raw: str
    minutes: int | None = None  # set when kind == "exact"
    token: str | None = None  # set when kind == "window"


def parse_time_token(value: str) -> TimeAnchor:
    """Parse a YAML `time:` value into a TimeAnchor.

    Accepts HH:MM or one of the named windows. Anything else raises
    `ValueError`. The grammar is intentionally narrow and event-name
    agnostic - there are no `after_lunch`-style relative anchors built
    in.
    """
    raw = value.strip()
    if ":" in raw:
        hh, mm = raw.split(":", 1)
        return TimeAnchor(kind="exact", raw=raw, minutes=int(hh) * 60 + int(mm))
    if raw in NAMED_WINDOWS:
        return TimeAnchor(kind="window", raw=raw, token=raw)
    raise ValueError(f"unrecognized time token: {raw!r}")


@dataclass(frozen=True, slots=True)
class WindowMap:
    """Immutable mapping of named windows to [start, end) minute ranges."""

    ranges: dict[str, tuple[int, int]]

    @classmethod
    def from_config(cls, time_windows: dict[str, WindowRange]) -> WindowMap:
        return cls({name: (r.start, r.end) for name, r in time_windows.items()})

    def get(self, name: str) -> tuple[int, int]:
        if name not in self.ranges:
            raise KeyError(f"unknown time window: {name!r}")
        return self.ranges[name]

    def __contains__(self, name: object) -> bool:
        return name in self.ranges
