"""Parse a persona-pipeline run directory into scenarios domain types.

This is the only module in `src.scripts.scenarios` that imports from
`src.scripts.persona`.  All other scenarios modules work exclusively with
the `CalendarTrace` / `CalendarEvent` domain types.

Expected run-directory layout (written by the persona pipeline)::

    <run_dir>/
      used_configs.json     paths to the four persona YAML files
      index.json            one entry per person (person_id, json_path, …)
      persons/
        <person_id>.json    per-person schedule + embedded Person snapshot

`used_configs.json` format::

    {
      "environment": "path/to/environment.yaml",
      "persona_config": "path/to/persona_config.yaml",
      "event_config": "path/to/event_config.yaml",
      "rules": "path/to/temporal_relation_rules.yaml"
    }
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scripts.persona.config.loader import load_environment, load_event, load_rules
from src.scripts.persona.config.schema import (
    AllenPairRule,
    DailyWindow,
    EventOverride,
    TemporalRule,
    WindowRange,
)
from src.scripts.persona.domain.event import Catalog, apply_event_overrides
from src.scripts.scenarios.config.timeframe import ResolvedWindow
from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode


class CalendarLoaderError(ValueError):
    """Raised when a persona run directory cannot be read or parsed."""


# ---------------------------------------------------------------------------
# LoadedRun
# ---------------------------------------------------------------------------


@dataclass
class LoadedRun:
    """Everything the scenarios evaluation layer needs from a persona run.

    Fields:
        traces: one `CalendarTrace` per person, sorted by (date, start_minutes).
        allen_pair_rules: `AllenPairRule` entries from the run's
            `temporal_relation_rules.yaml`; consumed by `metrics/allen.py`.
        ltl_rules: `TemporalRule` entries from the same file; passed to
            `check_ltl_rules` for optional LTL re-validation.
        time_windows: named window definitions from `environment.yaml`;
            used by `_preference_deviation` in `metrics/loss.py`.
        daily_window: hard waking-hours bound applied to every augmenter so
            placements stay inside `[wake_minutes, sleep_minutes)`.
        horizon_days: `environment.horizon.weeks * 7`.
        horizon_start_date: `environment.horizon.start_date`; `None` only
            when the loader was driven by explicit paths in a test that
            does not require date-aware slicing.
    """

    traces: list[CalendarTrace] = field(default_factory=list)
    allen_pair_rules: list[AllenPairRule] = field(default_factory=list)
    ltl_rules: list[TemporalRule] = field(default_factory=list)
    time_windows: dict[str, WindowRange] = field(default_factory=dict)
    daily_window: DailyWindow = field(default_factory=DailyWindow)
    horizon_days: int = 0
    horizon_start_date: datetime.date | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise CalendarLoaderError(f"file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalendarLoaderError(f"could not read {path}: {exc}") from exc


def _read_used_configs(run_dir: Path) -> dict[str, str]:
    used_configs_path = run_dir / "used_configs.json"
    data = _read_json(used_configs_path)
    if not isinstance(data, dict):
        raise CalendarLoaderError(
            f"used_configs.json must be a JSON object; got {type(data).__name__}"
        )
    return data


def _make_calendar_event(
    label: str,
    start: int,
    duration: int,
    date: datetime.date,
    catalog: Catalog,
    display_label: str | None = None,
) -> CalendarEvent:
    """Build a CalendarEvent, enriching metadata from the effective catalog."""
    defn = catalog.events_by_name.get(label)
    return CalendarEvent(
        label=label,
        start_minutes=start,
        end_minutes=start + duration,
        date=date,
        is_concurrent=defn.is_concurrent if defn is not None else False,
        is_dividable=defn.is_dividable if defn is not None else False,
        concurrent_with=(list(defn.concurrent_with) if defn is not None else []),
        intensity=(
            defn.intensity if defn is not None and defn.intensity is not None else 1
        ),
        display_label=display_label,
    )


def _parse_context_episode(raw: dict[str, Any]) -> ContextEpisode:
    """Build a `ContextEpisode` from one entry of the per-person `contexts: [...]` list."""
    return ContextEpisode(
        name=str(raw["name"]),
        category=str(raw["category"]),
        date=datetime.date.fromisoformat(raw["date"]),
        start_minutes=int(raw["start_minutes"]),
        end_minutes=int(raw["end_minutes"]),
        ontology_uri=raw.get("ontology_uri"),
        dimension=raw.get("dimension"),
        polarity=raw.get("polarity"),
        instrument=raw.get("instrument"),
        theory_mappings=raw.get("theory_mappings"),
    )


def _parse_person_json(data: dict[str, Any], base_catalog: Catalog) -> CalendarTrace:
    """Convert one per-person JSON dict into a `CalendarTrace`.

    Steps:
    1. Extract `persona.event_overrides` and build an effective catalog
       (persona overrides win over base catalog values).
    2. Iterate `days[].events` and `days[].spillovers` to build
       `CalendarEvent` objects enriched from the effective catalog.
    3. Parse `contexts[]` if present (missing key falls back to empty list
       so pre-context persona JSON keeps loading).
    4. Sort events + contexts by `(date, start_minutes)` and return a `CalendarTrace`.
    """
    person_id: str = data["person_id"]
    persona_id: str = str(data.get("persona_id") or "")
    persona_data: dict[str, Any] = data.get("persona", {})
    raw_overrides: dict[str, Any] = persona_data.get("event_overrides", {})

    overrides: dict[str, EventOverride] = {}
    for name, raw_ov in raw_overrides.items():
        if isinstance(raw_ov, dict):
            try:
                overrides[name] = EventOverride.model_validate(raw_ov)
            except Exception:
                pass

    effective_catalog = apply_event_overrides(base_catalog, overrides)

    events: list[CalendarEvent] = []
    for day_data in data.get("days", []):
        date = datetime.date.fromisoformat(day_data["date"])

        for label, instances in day_data.get("events", {}).items():
            for inst in instances:
                events.append(
                    _make_calendar_event(
                        label,
                        inst["start"],
                        inst["duration"],
                        date,
                        effective_catalog,
                        display_label=inst.get("label"),
                    )
                )

        for spill in day_data.get("spillovers", []):
            events.append(
                _make_calendar_event(
                    spill["type"],
                    spill["start"],
                    spill["duration"],
                    date,
                    effective_catalog,
                )
            )

    events.sort(key=lambda e: (e.date, e.start_minutes))

    raw_contexts = data.get("contexts") or []
    contexts = [_parse_context_episode(c) for c in raw_contexts]
    contexts.sort(key=lambda c: (c.date, c.start_minutes, c.category, c.name))

    return CalendarTrace(
        person_id=person_id,
        events=events,
        contexts=contexts,
        persona_id=persona_id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_persona_run(
    run_dir: Path,
    *,
    event_config_path: Path | None = None,
    rules_path: Path | None = None,
    environment_path: Path | None = None,
) -> LoadedRun:
    """Parse a persona-pipeline run directory into a `LoadedRun`.

    If any of the keyword-only path arguments are omitted, the loader reads
    `<run_dir>/used_configs.json` to determine the paths.  Provide the
    paths explicitly in tests or when `used_configs.json` is absent.

    Args:
        run_dir: directory written by `persona generate`.
        event_config_path: path to the `event_config.yaml` used for the run.
        rules_path: path to the `temporal_relation_rules.yaml` used for the run.
        environment_path: path to the `environment.yaml` used for the run.

    Returns:
        A `LoadedRun` with one `CalendarTrace` per person.

    Raises:
        CalendarLoaderError: if any required file is missing or unparseable.
    """
    run_dir = Path(run_dir)

    if any(p is None for p in (event_config_path, rules_path, environment_path)):
        used = _read_used_configs(run_dir)
        if event_config_path is None:
            event_config_path = Path(used["event_config"])
        if rules_path is None:
            rules_path = Path(used["rules"])
        if environment_path is None:
            environment_path = Path(used["environment"])

    try:
        event_config = load_event(event_config_path)
        rules_config = load_rules(rules_path)
        env_config = load_environment(environment_path)
    except Exception as exc:
        raise CalendarLoaderError(
            f"failed to load persona pipeline configs: {exc}"
        ) from exc

    base_catalog = Catalog.from_event_config(event_config)
    time_windows = dict(env_config.time_windows)
    daily_window = env_config.daily_window
    horizon_days = env_config.horizon.weeks * 7

    index_data = _read_json(run_dir / "index.json")
    persons_entries: list[dict[str, Any]] = index_data.get("persons", [])

    traces: list[CalendarTrace] = []
    for entry in persons_entries:
        # The persona pipeline stores json_path relative to the container working
        # directory (e.g. "output/example_experiment/persons/p.json").  Try that first;
        # fall back to run_dir-relative for test fixtures that use short paths.
        json_path_str = entry["json_path"]
        direct = Path(json_path_str)
        json_path = direct if direct.exists() else run_dir / json_path_str
        person_data = _read_json(json_path)
        traces.append(_parse_person_json(person_data, base_catalog))

    return LoadedRun(
        traces=traces,
        allen_pair_rules=list(rules_config.allen_pair_rules),
        ltl_rules=list(rules_config.rules),
        time_windows=time_windows,
        daily_window=daily_window,
        horizon_days=horizon_days,
        horizon_start_date=env_config.horizon.start_date,
    )


def slice_run_for_window(run: LoadedRun, window: ResolvedWindow) -> LoadedRun:
    """Return a `LoadedRun` whose traces and horizon are restricted to `window`.

    Events and contexts dated outside the window are dropped from every
    trace. Rules, time windows, and the daily window are passed through
    unchanged. When the window already matches the run's horizon the
    traces are still rebuilt for consistency but contain the same data.
    """
    end_exclusive = window.start_date + datetime.timedelta(days=window.days)
    sliced: list[CalendarTrace] = []
    for trace in run.traces:
        sliced.append(
            CalendarTrace(
                person_id=trace.person_id,
                events=[
                    e
                    for e in trace.events
                    if window.start_date <= e.date < end_exclusive
                ],
                contexts=[
                    c
                    for c in trace.contexts
                    if window.start_date <= c.date < end_exclusive
                ],
                persona_id=trace.persona_id,
            )
        )
    return LoadedRun(
        traces=sliced,
        allen_pair_rules=list(run.allen_pair_rules),
        ltl_rules=list(run.ltl_rules),
        time_windows=dict(run.time_windows),
        daily_window=run.daily_window,
        horizon_days=window.days,
        horizon_start_date=window.start_date,
    )
