"""Resolves the intensity omega(e) in {1..4} for every eventuality the loss touches.

The intensity loss L_disp scores eventualities in a single 1..4 space.
For an eventuality e, omega(e) = max(cognitive-difficulty level, MET
quartile of mu(e)), where mu(e) is the physical demand (MET value of the
matched activity). Two legs contribute:

* cognitive-difficulty level; HealthTasks `LevelN` ancestor
  (Level1 .. Level4), plus the legacy `event_config.yaml` `intensity`
  field (1..5; `5` collapses to `4`).
* MET quartile of mu(e); quartile bucket against the precomputed
  HumanActivities cutoffs (`output/met_quartiles.json`).

`IntensityResolver.resolve(activity)` returns omega(e)

    max( level_bucket(activity), met_quartile_bucket(activity) )

with both legs defaulting to `0` when their source is unavailable.
A resolved value of `0` only happens for activities with no Level
ancestor and no MET match, both legs missing.

A JSONL cache maps the resolved label to its bucket so re-runs avoid the
MET-lookup round-trip for labels already seen.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.scripts.scenarios.domain.calendar import CalendarEvent
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.met_lookup import MetLookup, build_query_text
from src.scripts.scenarios.metrics.ontology_source import infer_source


@dataclass(frozen=True)
class IntensityRecord:
    """One resolution result, persisted to `intensity_cache.jsonl`.

    Fields:
        intensity: the resolved omega(e) in 1..4 (`0` only when both legs missing).
        level_bucket: cognitive-difficulty leg (`0` when not applicable).
        met_bucket: MET-quartile leg of mu(e) (`0` when no MET match).
        met: matched MET value mu(e) (`None` when no MET match).
        matched_uri: HumanActivities URI that won the MET match (`None` otherwise).
        query_text: query string handed to the embedder (informational).
    """

    intensity: int
    level_bucket: int
    met_bucket: int
    met: float | None
    matched_uri: str | None
    query_text: str


@dataclass(frozen=True)
class MetQuartiles:
    """MET-quartile cutoffs read from `output/met_quartiles.json`."""

    q1: float
    q2: float
    q3: float
    max_met: float


def load_quartiles_file(path: Path) -> MetQuartiles:
    """Load a quartile JSON written by `precompute_met_quartiles.py`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MetQuartiles(
        q1=float(data["q1"]),
        q2=float(data["q2"]),
        q3=float(data["q3"]),
        max_met=float(data["max_met"]),
    )


def met_quartile_bucket(met: float | None, quartiles: MetQuartiles) -> int:
    """Map a raw MET value mu(e) to the 1..4 quartile bucket.

    Returns `0` when `met` is `None` (signals "no MET match" so
    that `max(level, met_bucket)` falls back to the Level leg cleanly).

    Cut-points::

        mu <= q1            to 1
        q1 < mu <= q2       to 2
        q2 < mu <= q3       to 3
        mu > q3             to 4
    """
    if met is None:
        return 0
    if met <= quartiles.q1:
        return 1
    if met <= quartiles.q2:
        return 2
    if met <= quartiles.q3:
        return 3
    return 4


def level_bucket_for_health_task(task: RecommendedTask) -> int:
    """HealthTasks Level to the 1..4 cognitive-difficulty leg.

    Returns `task.difficulty_level` clamped to `[0, 4]`.  `0` is
    used as the "no Level ancestor" sentinel and lets `max(...)` fall
    back to the MET-quartile leg of mu(e).
    """
    n = int(task.difficulty_level or 0)
    if n <= 0:
        return 0
    return min(n, 4)


def level_bucket_for_event_intensity(event_intensity: int | None) -> int:
    """`event_config.yaml` 1..5 intensity to the 1..4 cognitive-difficulty leg.

    `5 to 4` (the legacy field maxed out at 5; the omega scale only
    has four).  Otherwise identity.  `None` or `0` to `0`.
    """
    if event_intensity is None:
        return 0
    n = int(event_intensity)
    if n <= 0:
        return 0
    return 4 if n >= 4 else n


def build_event_intensity_map(event_config: Any) -> dict[str, int]:
    """Flatten an `EventConfig` into `{event_name: intensity_1_to_5}`.

    Accepts a Pydantic `EventConfig` instance (with `categories` to
    `events` shape) and falls back to `getattr` / `dict` access so
    plain dicts also work; useful for unit tests.
    """
    out: dict[str, int] = {}
    categories = getattr(event_config, "categories", None)
    if categories is None and isinstance(event_config, dict):
        categories = event_config.get("categories")
    if not categories:
        return out
    for category in categories.values():
        events = getattr(category, "events", None)
        if events is None and isinstance(category, dict):
            events = category.get("events")
        if not events:
            continue
        for event_name, event_def in events.items():
            intensity = getattr(event_def, "intensity", None)
            if intensity is None and isinstance(event_def, dict):
                intensity = event_def.get("intensity")
            if intensity is not None:
                out[event_name] = int(intensity)
    return out


class IntensityResolver:
    """Resolves a `RecommendedTask` / `CalendarEvent` / `ScheduledTask` to omega(e) in 1..4.

    Args:
        quartiles: precomputed MET quartile cutoffs.
        met_lookup: `MetLookup` for the MET-quartile leg of mu(e).  When
            `None` only the Level leg is used (useful for tests).
        event_intensity_map: `{event_name: intensity_1_to_5}` loaded
            from `event_config.yaml`.  Empty by default.
        cache_path: path to `intensity_cache.jsonl`; loaded on init
            if present, appended on new resolutions.
    """

    def __init__(
        self,
        quartiles: MetQuartiles,
        *,
        met_lookup: MetLookup | None = None,
        event_intensity_map: dict[str, int] | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._quartiles = quartiles
        self._met_lookup = met_lookup
        self._event_intensity_map = dict(event_intensity_map or {})
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, IntensityRecord] = {}
        if self._cache_path is not None and self._cache_path.exists():
            self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, activity: Any) -> int:
        """Return the intensity omega(e) in 1..4 for *activity*.

        Accepts `RecommendedTask`, `ScheduledTask`, `CalendarEvent`,
        or any object exposing a `label`.  Returns `0` only when
        both legs are missing.

        Resolution is cached by label after the first call.
        """
        record = self.record(activity)
        return record.intensity

    def record(self, activity: Any) -> IntensityRecord:
        """Return the full :class:`IntensityRecord` for *activity*."""
        if isinstance(activity, ScheduledTask):
            return self._record_for_task(activity.task)
        if isinstance(activity, RecommendedTask):
            return self._record_for_task(activity)
        if isinstance(activity, CalendarEvent):
            return self._record_for_event(activity)
        # Fallback: treat as a labeled activity with no metadata.
        label = getattr(activity, "label", None)
        if label is None:
            return self._make_record("", level=0, met=None, uri=None, query_text="")
        return self._record_for_event(_LabelOnlyActivity(label=str(label)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_for_task(self, task: RecommendedTask) -> IntensityRecord:
        label = task.label
        cached = self._cache.get(label)
        if cached is not None:
            return cached

        level = level_bucket_for_health_task(task)
        met_value: float | None = None
        matched_uri: str | None = None
        branch_local = infer_source(task.ontology_uri) or _branch_from_uri(
            task.ontology_uri
        )
        # Average duration is a useful disambiguator for the embedder.
        duration: int | None = None
        if task.duration_min and task.duration_max:
            duration = int((task.duration_min + task.duration_max) // 2)
        query_text = build_query_text(
            display_name=task.effective_display_name,
            description=task.description,
            branch_local_name=branch_local or "",
            duration_minutes=duration,
        )
        if self._met_lookup is not None:
            # `met_for_task` checks the curated bridge first, then
            # falls back to the rich-text embedding query.
            match = self._met_lookup.met_for_task(
                task.ontology_uri or "", fallback_query=query_text
            )
            met_value = match.met
            matched_uri = match.uri
        return self._make_record(
            label,
            level=level,
            met=met_value,
            uri=matched_uri,
            query_text=query_text,
        )

    def _record_for_event(self, event: Any) -> IntensityRecord:
        label = getattr(event, "label", "")
        cached = self._cache.get(label)
        if cached is not None:
            return cached

        event_intensity = getattr(event, "intensity", None)
        if event_intensity is None:
            event_intensity = self._event_intensity_map.get(label)
        level = level_bucket_for_event_intensity(event_intensity)
        met_value: float | None = None
        matched_uri: str | None = None
        query_text = label
        if self._met_lookup is not None and query_text:
            match = self._met_lookup.met(query_text)
            met_value = match.met
            matched_uri = match.uri
        return self._make_record(
            label,
            level=level,
            met=met_value,
            uri=matched_uri,
            query_text=query_text,
        )

    def _make_record(
        self,
        label: str,
        *,
        level: int,
        met: float | None,
        uri: str | None,
        query_text: str,
    ) -> IntensityRecord:
        met_bucket = met_quartile_bucket(met, self._quartiles)
        intensity = max(level, met_bucket)
        record = IntensityRecord(
            intensity=intensity,
            level_bucket=level,
            met_bucket=met_bucket,
            met=met,
            matched_uri=uri,
            query_text=query_text,
        )
        if label:
            self._cache[label] = record
            self._save_to_cache(label, record)
        return record

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        try:
            raw = self._cache_path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                label = str(entry["label"])
                self._cache[label] = IntensityRecord(
                    intensity=int(entry["intensity"]),
                    level_bucket=int(entry["level_bucket"]),
                    met_bucket=int(entry["met_bucket"]),
                    met=(float(entry["met"]) if entry.get("met") is not None else None),
                    matched_uri=(
                        str(entry["matched_uri"])
                        if entry.get("matched_uri") is not None
                        else None
                    ),
                    query_text=str(entry.get("query_text", "")),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    def _save_to_cache(self, label: str, record: IntensityRecord) -> None:
        if self._cache_path is None:
            return
        line = json.dumps(
            {
                "label": label,
                "intensity": record.intensity,
                "level_bucket": record.level_bucket,
                "met_bucket": record.met_bucket,
                "met": record.met,
                "matched_uri": record.matched_uri,
                "query_text": record.query_text,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


@dataclass(frozen=True)
class _LabelOnlyActivity:
    """Internal helper: a bare-labeled activity for the fallback path."""

    label: str
    intensity: int | None = None


def _branch_from_uri(uri: str | None) -> str:
    """Extract a HealthTasks branch local-name hint from a URI, if recognizable.

    `https://w3id.org/calendar-bench/health/task/cook-meal` to `""`
    (the URI's local name is the task itself, not the branch; branch
    detection requires a session, which the resolver does not own here).
    Returning `""` lets `build_query_text` skip the field cleanly.
    """
    return ""
