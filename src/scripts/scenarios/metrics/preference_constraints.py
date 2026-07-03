"""Persona-scoped constraint bundle consumed by the L_pref scorers.

L_pref = |Omega|^-1 Sum_{j in Omega} delta_j averages a normalized gap
delta_j in [0,1] for each persona preference (per-occurrence or per-scale
duration, episode count, temporal pattern, or persona-stage alignment).
This bundle supplies the per-preference inputs those delta_j scorers
need: one PersonaConstraints is built per persona at scenario time,
carrying the resolved per-event EventDefinition, a facade over
get_event_constraints for per-day / per-scale duration and episode
bounds, stages_firing_on for time-of-day windows, and matched_events_for
that maps a recommended task y_k=(eta_k, d_low, d_high) to the events
whose label eta its placed eventuality e-hat_k can be scored against, via
PreferenceMapper.

Pure data: no Redis / Neo4j calls happen until a scorer requests
mapping resolution (and even those round-trip through the cache).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Iterable

from src.scripts.persona.config.defaults import WEEKDAY_ORDER
from src.scripts.persona.config.schema import (
    EventConfig,
    EventDefinition,
    EventOverride,
    Persona,
)
from src.scripts.persona.constraints.extract import (
    EventDayConstraints,
    get_event_constraints,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.preference_mapping import (
    MappedEvent,
    PreferenceMapper,
)


@dataclass(frozen=True, slots=True)
class ResolvedStage:
    """A persona `stage` materialized on a concrete date.

    Fields:
        name: catalog event label eta the stage names.
        window_start: legal earliest start in minutes from midnight.
        window_end: legal latest end in minutes from midnight.  When the
            stage carries an explicit `HH:MM` time, this is
            `start + duration`; when it uses a window-token
            (`morning` / `afternoon` / ...) the bounds are the
            daypart's range.
        duration_minutes: persona-stated duration when provided.
        human_activity_iri / health_task_iri: copied from the stage so
            scorers and report rows can quote the IRI without an extra
            lookup.
    """

    name: str
    window_start: int
    window_end: int
    duration_minutes: int | None = None
    human_activity_iri: str | None = None
    health_task_iri: str | None = None


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_event_def(
    catalog: EventDefinition,
    override: EventOverride | None,
) -> EventDefinition:
    """Overlay an `EventOverride` onto a catalog `EventDefinition`.

    Mirrors the catalog to persona-override to recommended-task resolution
    order used by the rest of the persona pipeline.  This realizes the
    total ontology lookup O(eta, a) for the event label eta against the
    catalog default.  Returns a new immutable `EventDefinition` instance
    (the schema is frozen, so we construct rather than mutate).
    """
    if override is None:
        return catalog
    fields = catalog.model_dump()
    if override.requires is not None:
        fields["requires"] = {
            k: v.model_dump(by_alias=True) for k, v in override.requires.items()
        }
    if override.weekdays is not None:
        fields["weekdays"] = list(override.weekdays)
    if override.intensity is not None:
        fields["intensity"] = override.intensity
    if override.is_concurrent is not None:
        fields["is_concurrent"] = override.is_concurrent
    if override.is_dividable is not None:
        fields["is_dividable"] = override.is_dividable
    if override.concurrent_with is not None:
        fields["concurrent_with"] = list(override.concurrent_with)
    if override.ontology_iri is not None:
        fields["ontology_iri"] = override.ontology_iri
    if override.human_activity_iri is not None:
        fields["human_activity_iri"] = override.human_activity_iri
    if override.health_task_iri is not None:
        fields["health_task_iri"] = override.health_task_iri
    if override.per_event_duration is not None:
        fields["per_event_duration"] = override.per_event_duration.model_dump()
    if override.total_event_duration is not None:
        fields["total_event_duration"] = override.total_event_duration.model_dump()
    if override.total_event_episodes is not None:
        fields["total_event_episodes"] = override.total_event_episodes.model_dump()
    if override.temporal_patterns is not None:
        fields["temporal_patterns"] = [
            p.model_dump() for p in override.temporal_patterns
        ]
    if override.calendar_variations is not None:
        fields["calendar_variations"] = override.calendar_variations.model_dump()
    return EventDefinition.model_validate(fields)


def _parse_hhmm(time_token: str) -> int | None:
    """Return minute-of-day for an `HH:MM` token; `None` for window names."""
    if not time_token or ":" not in time_token:
        return None
    try:
        h, m = time_token.split(":", 1)
        hours = int(h)
        minutes = int(m)
        if 0 <= hours <= 23 and 0 <= minutes <= 59:
            return hours * 60 + minutes
    except (TypeError, ValueError):
        return None
    return None


def _stage_fires_on(
    stage_days: list[str], stage_date: _dt.date | None, target: _dt.date
) -> bool:
    if stage_date is not None:
        return stage_date == target
    if not stage_days:
        return False
    weekday_short = WEEKDAY_ORDER[target.weekday()]
    return weekday_short in stage_days


# ---------------------------------------------------------------------------
# PersonaConstraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Inputs:
    """Frozen bundle of construction-time inputs."""

    persona: Persona
    event_config: EventConfig
    window_map: WindowMap
    horizon_days: int
    horizon_start_date: _dt.date | None
    experiment: str
    scenario: str
    mapper: PreferenceMapper | None
    resolved_events: dict[str, EventDefinition] = field(default_factory=dict)


class PersonaConstraints:
    """Resolution + lookup facade for one persona.

    Construct once per persona; pass to scorers as `constraints`.
    """

    def __init__(
        self,
        *,
        persona: Persona,
        event_config: EventConfig,
        window_map: WindowMap,
        horizon_days: int,
        horizon_start_date: _dt.date | None = None,
        mapper: PreferenceMapper | None = None,
        experiment: str = "default",
        scenario: str = "default",
    ) -> None:
        resolved: dict[str, EventDefinition] = {}
        for category in event_config.categories.values():
            for event_name, event_def in category.events.items():
                override = persona.event_overrides.get(event_name)
                resolved[event_name] = _resolve_event_def(event_def, override)
        self._in = _Inputs(
            persona=persona,
            event_config=event_config,
            window_map=window_map,
            horizon_days=horizon_days,
            horizon_start_date=horizon_start_date,
            experiment=experiment,
            scenario=scenario,
            mapper=mapper,
            resolved_events=resolved,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def persona(self) -> Persona:
        return self._in.persona

    @property
    def horizon_days(self) -> int:
        return self._in.horizon_days

    @property
    def horizon_start_date(self) -> _dt.date | None:
        return self._in.horizon_start_date

    @property
    def event_names(self) -> list[str]:
        return list(self._in.resolved_events)

    def resolved_event_def_for(self, label: str) -> EventDefinition | None:
        return self._in.resolved_events.get(label)

    def all_resolved_events(self) -> dict[str, EventDefinition]:
        return dict(self._in.resolved_events)

    # ------------------------------------------------------------------
    # Per-day constraints
    # ------------------------------------------------------------------

    def day_constraints_for(
        self,
        label: str,
        *,
        day_idx: int,
        day_of_week: int,
    ) -> EventDayConstraints | None:
        """Thin facade over :func:`get_event_constraints` with the right scale args."""
        ev = self._in.resolved_events.get(label)
        if ev is None:
            return None
        return get_event_constraints(
            ev,
            day_idx=day_idx,
            total_days=self._in.horizon_days,
            window_map=self._in.window_map,
            day_of_week=day_of_week,
            horizon_start_date=self._in.horizon_start_date,
        )

    # ------------------------------------------------------------------
    # Stage resolver
    # ------------------------------------------------------------------

    def stages_firing_on(self, date: _dt.date) -> list[ResolvedStage]:
        """Return every persona stage firing on the given date with its window.

        Each stage names a persona-stage preference whose normalized gap
        delta_j feeds L_pref: a placed eventuality e-hat_k that overlaps
        the routine is scored for semantic alignment against this window.
        """
        out: list[ResolvedStage] = []
        for stage in self._in.persona.stages:
            if not _stage_fires_on(list(stage.days), stage.date, date):
                continue
            window_start, window_end = self._resolve_stage_window(stage)
            out.append(
                ResolvedStage(
                    name=stage.name,
                    window_start=window_start,
                    window_end=window_end,
                    duration_minutes=stage.duration_minutes,
                    human_activity_iri=stage.human_activity_iri,
                    health_task_iri=stage.health_task_iri,
                )
            )
        return out

    def _resolve_stage_window(self, stage) -> tuple[int, int]:
        """Materialise (window_start, window_end) for one stage.

        * `HH:MM` time to `[time, time + duration_minutes)` (defaults to a
          1-minute window when no duration is set).
        * Window-token time (`morning` / `afternoon` / ...) to the
          full daypart range from `window_map`.
        * Time unset to the full day `[0, 1440)`.
        """
        time_token = stage.time
        duration = stage.duration_minutes or 0
        wm = self._in.window_map
        if time_token is None:
            return 0, 1440
        explicit_start = _parse_hhmm(time_token)
        if explicit_start is not None:
            end = explicit_start + max(duration, 1)
            return explicit_start, min(end, 1440)
        # Window-token path.
        if time_token in wm:
            start, end = wm.get(time_token)
            return start, end
        return 0, 1440

    # ------------------------------------------------------------------
    # Mapper bridge
    # ------------------------------------------------------------------

    def matched_events_for(self, task: RecommendedTask) -> list[MappedEvent]:
        """Resolve the events a recommended task y_k maps to via the mapper.

        Delegates to the :class:`PreferenceMapper` to find the catalog
        events whose label eta a placed eventuality e-hat_k=(eta_k, tau_s,
        tau_e) for this task can be scored against when computing delta_j.
        Returns an empty list when no mapper was wired in (a fully-offline
        constraint bundle, useful in tests).
        """
        if self._in.mapper is None:
            return []
        events: Iterable[EventDefinition] = self._in.resolved_events.values()
        return self._in.mapper.matched_events(
            task,
            events,
            experiment=self._in.experiment,
            scenario=self._in.scenario,
        )
