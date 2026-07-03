"""Per-persona chart rendering for generated schedules.

Four chart families, all grouped by `persona_id`:

1. Horizon line chart: one PNG per variable event with one line per
   persona that has that event. The line is the mean across the
   persona's persons; the shaded band is a 95% CI (normal
   approximation, `mean +/- 1.96 * std / sqrt(n)`). Two charts per
   variable event, one for episode counts and one for daily duration.
   Events whose catalog `temporal_patterns` are only `mode: fix` are
   skipped because the curve would be flat. The legacy overlay chart
   (one PNG per persona, every event) is still available via
   `plot_persona_horizon_lines`.
2. Calendar heatmap (calmap): daily aggregate (count or duration; sum
   or mean across persons) on a year calendar.
3. Single-day Gantt: one person, one date.
4. Weekly calendar view: one PNG per ISO week with base events and
   (optional) augmented tasks color-coded. Lives in
   `weekly_calendar.py` and runs through `render_all_charts`.

Data helpers (`compute_*`, `event_count_matrix`,
`daily_event_aggregate`) are matplotlib-free so tests can exercise
them without a backend. Matplotlib's `Agg` backend is forced at
import time so the chart helpers run in headless containers.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import calmap  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.scripts.persona.domain.schedule import PersonSchedule  # noqa: E402

if TYPE_CHECKING:  # avoid loading the schema module at chart-import time
    from src.scripts.persona.config.schema import (
        EventConfig,
        EventOverride,
        PersonaConfig,
    )

ALL_CHART_KINDS: tuple[str, ...] = (
    "lines",
    "heatmap",
    "gantt",
    "calendar",
    "context-lines",
    "context-heatmap",
)
DEFAULT_CHART_KINDS: tuple[str, ...] = ("lines", "heatmap", "gantt")

DAY_MINUTES = 1440
HOURS_PER_DAY = 24
SQUARE_FIGSIZE = (8, 8)
HEATMAP_FIGSIZE = (12, 8)
DPI = 300


# -------------------------------------------------------------------------------------
# ----------------------------- pure data computations --------------------------------
# -------------------------------------------------------------------------------------


def group_by_persona(
    schedules: list[PersonSchedule],
) -> dict[str, list[PersonSchedule]]:
    """Group schedules by `persona_id` in input order."""
    out: dict[str, list[PersonSchedule]] = defaultdict(list)
    for schedule in schedules:
        out[schedule.persona_id].append(schedule)
    return dict(out)


def variable_heatmap_events(
    event_config: "EventConfig",
    persona_config: "PersonaConfig",
) -> dict[str, set[str]]:
    """Return `{persona_id: {event_name, ...}}` of events worth heatmapping.

    An event is variable for a persona when either the catalog
    `temporal_patterns` has no `fix`-only entry, or the persona's
    override adds a `seasonality` / `trend` pattern. Events with only
    `fix` patterns produce flat heatmaps and are skipped. Overrides
    that replace a non-fix catalog pattern with `fix` are not modeled.
    """
    catalog_variable: set[str] = set()
    for category in event_config.categories.values():
        for event_name, event_def in category.events.items():
            patterns = event_def.temporal_patterns
            if not patterns or any(p.mode != "fix" for p in patterns):
                catalog_variable.add(event_name)

    by_persona: dict[str, set[str]] = {}
    for persona in persona_config.personas:
        events_for_persona: set[str] = set(catalog_variable)
        for event_name, override in persona.event_overrides.items():
            override_patterns = override.temporal_patterns
            if override_patterns is None:
                continue
            if any(p.mode in ("seasonality", "trend") for p in override_patterns):
                events_for_persona.add(event_name)
        by_persona[persona.id] = events_for_persona
    return by_persona


@dataclass(frozen=True)
class HorizonTrend:
    """Per-event-type mean / std across one persona's persons, indexed by day."""

    event_types: tuple[str, ...]
    num_days: int
    num_persons: int
    means: dict[str, np.ndarray] = field(default_factory=dict)
    stds: dict[str, np.ndarray] = field(default_factory=dict)


def _collect_event_types(schedules: list[PersonSchedule]) -> list[str]:
    """Sorted union of every event name observed across the schedules."""
    seen: set[str] = set()
    for schedule in schedules:
        for day in schedule.days:
            seen.update(day.events.keys())
    return sorted(seen)


def event_count_matrix(
    schedules: list[PersonSchedule], event_types: list[str] | None = None
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return `{event_type: matrix[day, person]}` plus the resolved event-type order.

    Each matrix entry is the number of episodes of that event type on that
    day for that person. Schedules of varying length are padded with zeros
    so all matrices share the same `(num_days, num_persons)` shape.
    """
    types = (
        list(event_types)
        if event_types is not None
        else _collect_event_types(schedules)
    )
    num_persons = len(schedules)
    num_days = max((len(s.days) for s in schedules), default=0)
    counts = {t: np.zeros((num_days, num_persons), dtype=float) for t in types}
    for p_idx, schedule in enumerate(schedules):
        for d_idx, day in enumerate(schedule.days):
            for t in types:
                counts[t][d_idx, p_idx] = len(day.events.get(t, []))
    return counts, types


def compute_horizon_trend(
    schedules: list[PersonSchedule],
    *,
    event_types: list[str] | None = None,
) -> HorizonTrend:
    """Mean and standard deviation of episode counts per event type per day."""
    counts, types = event_count_matrix(schedules, event_types=event_types)
    num_days = next(iter(counts.values())).shape[0] if counts else 0
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for t, matrix in counts.items():
        if matrix.shape[1] == 0:
            means[t] = np.zeros(matrix.shape[0])
            stds[t] = np.zeros(matrix.shape[0])
        else:
            means[t] = matrix.mean(axis=1)
            stds[t] = matrix.std(axis=1)
    return HorizonTrend(
        event_types=tuple(types),
        num_days=num_days,
        num_persons=len(schedules),
        means=means,
        stds=stds,
    )


def daily_event_aggregate(
    schedules: list[PersonSchedule],
    *,
    event_name: str | None = None,
    metric: str = "count",
    aggregate: str = "mean",
) -> pd.Series:
    """Daily aggregated metric, returned as a pandas Series indexed by date.

    Args:
        event_name: filter to one event type. `None` aggregates every event.
        metric: `"count"` (number of episodes) or `"duration"` (sum of
            episode minutes).
        aggregate: `"sum"` or `"mean"` across the persons in `schedules`.
            With one schedule both reduce to the same value.

    Empty population returns an empty Series.
    """
    if metric not in ("count", "duration"):
        raise ValueError(f"metric must be 'count' or 'duration', got {metric!r}")
    if aggregate not in ("sum", "mean"):
        raise ValueError(f"aggregate must be 'sum' or 'mean', got {aggregate!r}")

    by_date: dict[_dt.date, list[float]] = defaultdict(list)
    for schedule in schedules:
        for day in schedule.days:
            value = 0.0
            if event_name is not None:
                events = day.events.get(event_name, [])
                if metric == "count":
                    value = float(len(events))
                else:
                    value = float(sum(ev.duration for ev in events))
            else:
                for events in day.events.values():
                    if metric == "count":
                        value += len(events)
                    else:
                        value += sum(ev.duration for ev in events)
            by_date[day.date].append(value)

    if not by_date:
        return pd.Series(dtype=float)

    dates = sorted(by_date.keys())
    if aggregate == "sum":
        values = [sum(by_date[d]) for d in dates]
    else:
        values = [sum(by_date[d]) / len(by_date[d]) for d in dates]
    return pd.Series(values, index=pd.DatetimeIndex(dates))


# -------------------------------------------------------------------------------------
# --------------------------------- plot helpers --------------------------------------
# -------------------------------------------------------------------------------------


def _ensure_dir(out_dir: Path | str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _palette(n: int) -> list[str]:
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    return [mcolors.to_hex(cmap(i % cmap.N)) for i in range(n)]


def plot_persona_horizon_lines(
    persona_id: str,
    schedules: list[PersonSchedule],
    out_dir: Path | str,
) -> Path | None:
    """One PNG per persona: mean line + +/- 1 std band per event type.

    Returns the saved path, or `None` when there is nothing to plot
    (no schedules, no days, or no event types).
    """
    trend = compute_horizon_trend(schedules)
    if trend.num_days == 0 or not trend.event_types:
        return None
    target = _ensure_dir(out_dir) / f"lines_{persona_id}.png"
    days = np.arange(1, trend.num_days + 1)
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    for t in trend.event_types:
        mean = trend.means[t]
        std = trend.stds[t]
        ax.plot(days, mean, label=t, linewidth=1.5)
        ax.fill_between(days, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Day of horizon")
    ax.set_ylabel("Mean episodes per person per day")
    ax.set_title(
        f"Persona '{persona_id}' - episodes per day "
        f"(n={trend.num_persons}, mean +/- 1 std)"
    )
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(target, dpi=DPI)
    plt.close(fig)
    return target


def plot_person_day_gantt(
    schedule: PersonSchedule,
    date: _dt.date,
    out_dir: Path | str,
) -> Path | None:
    """Render a one-day Gantt for one person on one calendar date.

    Returns the saved path, or `None` if the person has no day matching
    `date` or that day has no scheduled events.
    """
    target_day = next((d for d in schedule.days if d.date == date), None)
    if target_day is None or not target_day.events:
        return None

    event_types = sorted(target_day.events.keys())
    color_map = dict(zip(event_types, _palette(len(event_types))))
    target = _ensure_dir(out_dir) / f"gantt_{schedule.person_id}_{date.isoformat()}.png"

    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    for row, event_type in enumerate(event_types):
        for ev_idx, ev in enumerate(target_day.events[event_type]):
            ax.barh(
                row,
                ev.duration,
                left=ev.start,
                color=color_map[event_type],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.text(
                ev.start + ev.duration / 2,
                row,
                str(ev_idx + 1),
                va="center",
                ha="center",
                color="white",
                fontsize=8,
            )
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_types)
    ax.set_xlim(0, DAY_MINUTES)
    xticks = [h * 60 for h in range(0, HOURS_PER_DAY + 1, 2)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, HOURS_PER_DAY + 1, 2)])
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Event type")
    ax.set_title(f"{schedule.person_id} - {date.isoformat()} ({target_day.weekday})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(target, dpi=DPI)
    plt.close(fig)
    return target


def heatmap_vmax_for_event(
    event_config: "EventConfig | None",
    event_name: str | None,
    metric: str,
    override: "EventOverride | None" = None,
) -> float | None:
    """Return the effective vmax for the heatmap colorbar.

    Reads `total_event_episodes.max` (metric=`count`) or
    `total_event_duration.max` in minutes (metric=`duration`). When an
    `override` is supplied and its matching field is set, the override
    wins (the persona's tighter bound is the one a reader cares about).
    Returns `None` when the event or config is absent.
    """
    return _heatmap_bound_for_event(
        event_config, event_name, metric, "max", override=override
    )


def heatmap_vmin_for_event(
    event_config: "EventConfig | None",
    event_name: str | None,
    metric: str,
    override: "EventOverride | None" = None,
) -> float | None:
    """Return the effective vmin for the heatmap colorbar.

    Reads `total_event_episodes.min` (metric=`count`) or
    `total_event_duration.min` in minutes (metric=`duration`).
    Honors an `override` when supplied.
    """
    return _heatmap_bound_for_event(
        event_config, event_name, metric, "min", override=override
    )


def _heatmap_bound_for_event(
    event_config: "EventConfig | None",
    event_name: str | None,
    metric: str,
    bound: str,
    *,
    override: "EventOverride | None" = None,
) -> float | None:
    """Resolve the `min` or `max` field for the event, honoring overrides.

    `metric` selects between `total_event_episodes` (count) and
    `total_event_duration` (duration); `bound` picks `min` or `max`.
    Per-persona overrides win when they set the matching field.
    """
    if event_config is None or event_name is None:
        return None
    for category in event_config.categories.values():
        event_def = category.events.get(event_name)
        if event_def is None:
            continue
        if metric == "count":
            field = (
                override.total_event_episodes
                if override is not None and override.total_event_episodes is not None
                else event_def.total_event_episodes
            )
            return float(getattr(field, bound))
        if metric == "duration":
            field = (
                override.total_event_duration
                if override is not None and override.total_event_duration is not None
                else event_def.total_event_duration
            )
            return float(_to_minutes(getattr(field, bound), field.unit))
        return None
    return None


def _to_minutes(value: int | float, unit: str) -> float:
    """Convert a `value` in `unit` (`minutes` / `hours`) to minutes."""
    if unit == "hours":
        return float(value) * 60.0
    return float(value)


def _valid_scale(vmin: float | None, vmax: float | None) -> bool:
    """True when both bounds are present and form a non-degenerate range."""
    return vmin is not None and vmax is not None and vmax > vmin


def plot_persona_heatmap(
    persona_id: str,
    schedules: list[PersonSchedule],
    out_dir: Path | str,
    *,
    event_name: str | None = None,
    metric: str = "count",
    aggregate: str = "mean",
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path | None:
    """Calendar heatmap of a daily aggregate, one PNG per persona.

    `event_name` filters to a single event type (default: every event).
    `metric` is `count` (episodes) or `duration` (minutes).
    `aggregate` is `mean` or `sum` across the persons of this persona.
    `vmin` and `vmax` anchor the colorbar range. When both are set and
    `vmax > vmin`, the gradient runs from `vmin` to `vmax` so cell
    color reflects the configured per-day band. When either is
    missing or the range is degenerate, the colorbar falls back to
    data-driven scaling.

    Returns the saved path, or `None` if the persona has no schedule
    days at all.
    """
    series = daily_event_aggregate(
        schedules,
        event_name=event_name,
        metric=metric,
        aggregate=aggregate,
    )
    if series.empty:
        return None

    label = event_name if event_name else "all events"
    fname_event = event_name if event_name else "all"
    target = _ensure_dir(out_dir) / (
        f"heatmap_{persona_id}_{fname_event}_{metric}_{aggregate}.png"
    )

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    use_config_scale = _valid_scale(vmin, vmax)
    yearplot_kwargs: dict = {
        "cmap": cmap,
        "daylabels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "ax": ax,
    }
    if use_config_scale:
        yearplot_kwargs["vmin"] = float(vmin)
        yearplot_kwargs["vmax"] = float(vmax)
    calmap.yearplot(series, **yearplot_kwargs)
    metric_label = "episodes" if metric == "count" else "minutes"
    fig.suptitle(
        f"Persona '{persona_id}' - {aggregate} of {metric_label} per day " f"({label})",
        fontsize=14,
        y=0.95,
    )

    avg = float(series.mean())
    total = float(series.sum())
    max_val = float(series.max())
    min_val = float(series.min())
    info_text = (
        f"Avg:   {avg:.2f}\n"
        f"Total: {total:.0f}\n"
        f"Max:   {max_val:.0f}\n"
        f"Min:   {min_val:.0f}"
    )
    if use_config_scale:
        info_text += f"\nScale: {float(vmin):.0f} to {float(vmax):.0f} (config)"
    ax.text(
        0.02,
        1.6,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )

    # calmap renders TWO `QuadMesh` collections: index 0 is the empty
    # grid background, index 1 is the actual data layer using the
    # caller-supplied colormap. The colorbar must be wired to the
    # second one or the gradient comes out blank.
    collections = ax.collections
    if len(collections) >= 2:  # pragma: no branch  (yearplot always emits 2)
        # Place the colorbar on a dedicated axes well below the calendar
        # so the month-label row stays visible. The default `fig.colorbar
        # (ax=ax)` route binds it tightly under the axis and tends to
        # overlap calmap's month tick labels.
        cbar_ax = fig.add_axes([0.10, 0.08, 0.80, 0.03])
        cbar = fig.colorbar(collections[1], cax=cbar_ax, orientation="horizontal")
        cbar_label = f"{metric_label} ({aggregate})"
        if use_config_scale:
            cbar_label += f"; {float(vmin):.0f} to {float(vmax):.0f} (config range)"
        cbar.set_label(cbar_label, fontsize=10)

    # Leave room at the top for the title and at the bottom for the
    # month-label row + the dedicated colorbar axes.
    plt.subplots_adjust(top=0.88, bottom=0.22, left=0.05, right=0.95)
    fig.savefig(target, dpi=DPI, facecolor="white")
    plt.close(fig)
    return target


# -------------------------------------------------------------------------------------
# --------------------------------- top-level entrypoint -------------------------------
# -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartArtifacts:
    """Paths produced by `render_all_charts`.

    `heatmaps` holds one aggregate heatmap per persona; populated when
    `heatmap_events_by_persona` is not supplied. `per_event_heatmaps`
    is keyed by `persona_id`, then `event_name`, then `metric`
    (`count` or `duration`); populated when `heatmap_events_by_persona`
    is supplied. A persona uses one mode or the other, never both.
    """

    lines: dict[str, Path] = field(default_factory=dict)
    per_event_lines: dict[str, dict[str, Path]] = field(default_factory=dict)
    heatmaps: dict[str, Path] = field(default_factory=dict)
    per_event_heatmaps: dict[str, dict[str, dict[str, Path]]] = field(
        default_factory=dict
    )
    gantts: list[Path] = field(default_factory=list)
    calendars: dict[str, list[Path]] = field(default_factory=dict)
    # Per-member calmap-style heatmaps keyed
    # `persona_id -> category -> member -> metric -> path`.
    per_member_context_heatmaps: dict[str, dict[str, dict[str, dict[str, Path]]]] = (
        field(default_factory=dict)
    )
    # Per-member cohort line plots keyed `category -> member -> metric -> path`.
    per_member_context_lines: dict[str, dict[str, dict[str, Path]]] = field(
        default_factory=dict
    )

    @property
    def all_paths(self) -> list[Path]:
        out: list[Path] = []
        out.extend(self.lines.values())
        for metric_paths in self.per_event_lines.values():
            out.extend(metric_paths.values())
        out.extend(self.heatmaps.values())
        for event_paths in self.per_event_heatmaps.values():
            for metric_paths in event_paths.values():
                out.extend(metric_paths.values())
        out.extend(self.gantts)
        for paths in self.calendars.values():
            out.extend(paths)
        for cat_map in self.per_member_context_heatmaps.values():
            for member_map in cat_map.values():
                for metric_map in member_map.values():
                    out.extend(metric_map.values())
        for member_map in self.per_member_context_lines.values():
            for metric_map in member_map.values():
                out.extend(metric_map.values())
        return out


def _resolve_kinds(kinds: Iterable[str] | None) -> set[str]:
    """Normalize the `kinds` argument to a set of known chart families.

    `None` is the legacy default (`DEFAULT_CHART_KINDS`, omits the
    weekly calendar view).  Unknown kinds raise `ValueError` so a CLI
    typo cannot silently produce an empty artifact set.
    """
    if kinds is None:
        return set(DEFAULT_CHART_KINDS)
    selected = set(kinds)
    unknown = selected - set(ALL_CHART_KINDS)
    if unknown:
        raise ValueError(
            f"unknown chart kinds: {sorted(unknown)}; "
            f"valid choices are {list(ALL_CHART_KINDS)}"
        )
    return selected


_PER_EVENT_HEATMAP_METRICS: tuple[str, ...] = ("count", "duration")
_PER_EVENT_LINE_METRICS: tuple[str, ...] = ("count", "duration")

# 95% normal-approximation z-score.
_CI95_Z = 1.959963984540054


def per_event_persona_daily(
    schedules: list[PersonSchedule],
    event_name: str,
    *,
    metric: str,
) -> np.ndarray:
    """Build a `(num_days, num_persons)` matrix of daily values for one event.

    `metric="count"` counts episodes per day; `metric="duration"` sums
    episode minutes per day. Shorter horizons are zero-padded.
    """
    if metric not in ("count", "duration"):
        raise ValueError(f"metric must be 'count' or 'duration', got {metric!r}")
    num_persons = len(schedules)
    num_days = max((len(s.days) for s in schedules), default=0)
    matrix = np.zeros((num_days, num_persons), dtype=float)
    for p_idx, schedule in enumerate(schedules):
        for d_idx, day in enumerate(schedule.days):
            events = day.events.get(event_name, [])
            if metric == "count":
                matrix[d_idx, p_idx] = float(len(events))
            else:
                matrix[d_idx, p_idx] = float(sum(ev.duration for ev in events))
    return matrix


def daily_mean_and_ci(
    values: np.ndarray, *, z: float = _CI95_Z
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-day mean and 95% CI bounds across the person axis.

    `values` is shaped `(num_days, num_persons)`. Returns `(mean,
    lower, upper)`. With `n <= 1` the CI collapses to the mean.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {values.shape}")
    num_days, num_persons = values.shape
    if num_days == 0:
        empty = np.zeros(0)
        return empty, empty.copy(), empty.copy()
    mean = values.mean(axis=1) if num_persons > 0 else np.zeros(num_days)
    if num_persons <= 1:
        return mean, mean.copy(), mean.copy()
    std = values.std(axis=1, ddof=1)
    half = z * std / np.sqrt(num_persons)
    return mean, mean - half, mean + half


def duration_unit_for_event(
    event_config: "EventConfig | None",
    event_name: str,
    override: "EventOverride | None" = None,
) -> str:
    """Resolve the display unit (`minutes` / `hours`) for an event duration.

    Override wins over catalog. Falls back to `"minutes"` when the
    event is unknown.
    """
    if override is not None and override.total_event_duration is not None:
        return override.total_event_duration.unit
    if event_config is None:
        return "minutes"
    for category in event_config.categories.values():
        event_def = category.events.get(event_name)
        if event_def is not None:
            return event_def.total_event_duration.unit
    return "minutes"


def _convert_minutes(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert a minutes-valued array to `unit` (`minutes` or `hours`)."""
    return values / 60.0 if unit == "hours" else values


def _metric_axis_label(metric: str, unit: str) -> str:
    if metric == "count":
        return "Episodes per day"
    return f"Total daily duration ({unit})"


def plot_per_event_persona_lines(
    event_name: str,
    metric: str,
    schedules_by_persona: dict[str, list[PersonSchedule]],
    out_dir: Path | str,
    *,
    duration_unit: str = "minutes",
) -> Path | None:
    """One PNG comparing every persona on one event and one metric.

    Each persona gets a mean line plus a 95% CI band built from the
    between-person variance. Returns `None` when no persona has any
    day or no episode of the event was logged.
    """
    if metric not in _PER_EVENT_LINE_METRICS:
        raise ValueError(f"metric must be 'count' or 'duration', got {metric!r}")
    series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    max_days = 0
    has_signal = False
    for persona_id in sorted(schedules_by_persona.keys()):
        scheds = schedules_by_persona[persona_id]
        if not scheds:
            continue
        matrix = per_event_persona_daily(scheds, event_name, metric=metric)
        if matrix.size == 0 or matrix.shape[0] == 0:
            continue
        mean, lo, hi = daily_mean_and_ci(matrix)
        max_days = max(max_days, mean.shape[0])
        if matrix.sum() > 0:
            has_signal = True
        if metric == "duration":
            mean = _convert_minutes(mean, duration_unit)
            lo = _convert_minutes(lo, duration_unit)
            hi = _convert_minutes(hi, duration_unit)
        series.append((persona_id, mean, lo, hi, matrix.shape[1]))
    if not series or max_days == 0 or not has_signal:
        return None

    target = _ensure_dir(out_dir) / f"lines_{event_name}_{metric}.png"
    colors = _palette(len(series))
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    for color, (persona_id, mean, lo, hi, n_persons) in zip(colors, series):
        days = np.arange(1, mean.shape[0] + 1)
        label = f"{persona_id} (n={n_persons})"
        ax.plot(days, mean, color=color, label=label, linewidth=1.6)
        ax.fill_between(days, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Day of horizon")
    ax.set_ylabel(_metric_axis_label(metric, duration_unit))
    metric_word = "episodes" if metric == "count" else "duration"
    ax.set_title(f"'{event_name}' daily {metric_word}; mean +/- 95% CI across persons")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(target, dpi=DPI)
    plt.close(fig)
    return target


def _persona_event_overrides(
    persona_config: "PersonaConfig | None", persona_id: str
) -> dict[str, "EventOverride"]:
    """Return `event_overrides` for `persona_id`, or `{}` when absent."""
    if persona_config is None:
        return {}
    for persona in persona_config.personas:
        if persona.id == persona_id:
            return dict(persona.event_overrides)
    return {}


def _members_for_persona(
    persona_id: str,
    persona_config: "PersonaConfig | None",
    persona_schedules: list[PersonSchedule],
) -> list[tuple[str, str]]:
    """Return sorted (category, member) pairs to chart for one persona."""
    if persona_config is not None:
        for persona in persona_config.personas:
            if persona.id != persona_id:
                continue
            pairs: list[tuple[str, str]] = []
            for cat_name, cat in persona.contexts.items():
                for member_name in cat.members.keys():
                    pairs.append((cat_name, member_name))
            return sorted(pairs)
    seen: set[tuple[str, str]] = set()
    for sched in persona_schedules:
        for ep in sched.contexts:
            seen.add((ep.category, ep.name))
    return sorted(seen)


def render_all_charts(
    schedules: list[PersonSchedule],
    *,
    out_dir: Path | str,
    kinds: Iterable[str] | None = None,
    heatmap_event: str | None = None,
    heatmap_metric: str = "count",
    heatmap_aggregate: str = "mean",
    heatmap_events_by_persona: dict[str, set[str]] | None = None,
    line_events_by_persona: dict[str, set[str]] | None = None,
    event_config: "EventConfig | None" = None,
    persona_config: "PersonaConfig | None" = None,
    augmented_by_person: dict[str, list] | None = None,
    weekly_calendar_dpi: int = 300,
    weekly_calendar_title_prefix: str | None = None,
    weekly_calendar_min_event_minutes: int | None = None,
    weekly_calendar_context_categories: Iterable[str] | None = None,
) -> ChartArtifacts:
    """Render the requested chart families per persona.

    Args:
        kinds: chart families to emit. `None` keeps the default
            (lines + heatmap + sample Gantt). Include `"calendar"` to
            also emit weekly calendar PNGs per ISO week.
        heatmap_events_by_persona: switches the heatmap branch to
            per-event mode (one count + one duration heatmap per
            variable event per persona). Use `variable_heatmap_events`
            to derive it. `None` keeps the legacy aggregate heatmap.
        line_events_by_persona: switches the lines branch to per-event
            mode (one chart per variable event, one line per persona).
            Same dict shape as the heatmap selector. `None` keeps the
            legacy per-persona overlay.
        event_config: anchors the heatmap colorbar at the catalog
            per-day `[min, max]`. `None` falls back to data-driven
            scaling.
        persona_config: per-persona overrides win over the catalog
            for both colorbar bounds and duration unit.
        augmented_by_person: per-person augmented scenarios tasks for
            the weekly calendar. Persons without a key get base-only.
    """
    selected_kinds = _resolve_kinds(kinds)
    grouped = group_by_persona(schedules)
    charts_dir = _ensure_dir(Path(out_dir) / "charts")

    lines: dict[str, Path] = {}
    per_event_lines: dict[str, dict[str, Path]] = {}
    heatmaps: dict[str, Path] = {}
    per_event_heatmaps: dict[str, dict[str, dict[str, Path]]] = {}
    gantts: list[Path] = []
    calendars: dict[str, list[Path]] = {}

    if "lines" in selected_kinds and line_events_by_persona is not None:
        # Per-event mode: invert persona-to-events into event-to-personas
        # and render one chart per (event, metric). Sorted for stable
        # filenames.
        persona_to_events = {
            pid: set(evs) for pid, evs in line_events_by_persona.items()
        }
        all_events: set[str] = set()
        for evs in persona_to_events.values():
            all_events.update(evs)
        for event_name in sorted(all_events):
            schedules_by_persona = {
                pid: scheds
                for pid, scheds in grouped.items()
                if event_name in persona_to_events.get(pid, set())
            }
            if not schedules_by_persona:
                continue
            override_unit: str | None = None
            for pid in sorted(schedules_by_persona.keys()):
                overrides = _persona_event_overrides(persona_config, pid)
                unit = duration_unit_for_event(
                    event_config, event_name, overrides.get(event_name)
                )
                # Mixed override units fall back to the catalog so the
                # y-axis label stays consistent.
                if override_unit is None:
                    override_unit = unit
                elif override_unit != unit:
                    override_unit = duration_unit_for_event(
                        event_config, event_name, None
                    )
                    break
            display_unit = override_unit or "minutes"
            for metric in _PER_EVENT_LINE_METRICS:
                path = plot_per_event_persona_lines(
                    event_name,
                    metric,
                    schedules_by_persona,
                    charts_dir,
                    duration_unit=display_unit,
                )
                if path is not None:
                    per_event_lines.setdefault(event_name, {})[metric] = path

    for persona_id, persona_schedules in grouped.items():
        if "lines" in selected_kinds and line_events_by_persona is None:
            line_path = plot_persona_horizon_lines(
                persona_id, persona_schedules, charts_dir
            )
            if line_path is not None:
                lines[persona_id] = line_path

        if "heatmap" in selected_kinds:
            overrides = _persona_event_overrides(persona_config, persona_id)
            if heatmap_events_by_persona is None:
                override = overrides.get(heatmap_event) if heatmap_event else None
                vmin = heatmap_vmin_for_event(
                    event_config, heatmap_event, heatmap_metric, override
                )
                vmax = heatmap_vmax_for_event(
                    event_config, heatmap_event, heatmap_metric, override
                )
                heat_path = plot_persona_heatmap(
                    persona_id,
                    persona_schedules,
                    charts_dir,
                    event_name=heatmap_event,
                    metric=heatmap_metric,
                    aggregate=heatmap_aggregate,
                    vmin=vmin,
                    vmax=vmax,
                )
                if heat_path is not None:
                    heatmaps[persona_id] = heat_path
            else:
                # Per-event mode: render BOTH a count and a duration
                # heatmap for each variable event. Sorted so filenames
                # and iteration order are deterministic regardless of
                # the caller's set ordering.
                event_names = sorted(heatmap_events_by_persona.get(persona_id, set()))
                for event_name in event_names:
                    override = overrides.get(event_name)
                    for metric in _PER_EVENT_HEATMAP_METRICS:
                        vmin = heatmap_vmin_for_event(
                            event_config, event_name, metric, override
                        )
                        vmax = heatmap_vmax_for_event(
                            event_config, event_name, metric, override
                        )
                        heat_path = plot_persona_heatmap(
                            persona_id,
                            persona_schedules,
                            charts_dir,
                            event_name=event_name,
                            metric=metric,
                            aggregate=heatmap_aggregate,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        if heat_path is not None:
                            per_event_heatmaps.setdefault(persona_id, {}).setdefault(
                                event_name, {}
                            )[metric] = heat_path

        if "gantt" in selected_kinds:
            # Sample Gantt: first person of the persona, first day with events.
            # When `sample_day` is found we know `plot_person_day_gantt` will
            # return a path (the date matches and the day's `events` is
            # non-empty), so no defensive `is not None` check is needed.
            sample = persona_schedules[0]
            sample_day = next((d for d in sample.days if d.events), None)
            if sample_day is not None:
                gantts.append(
                    plot_person_day_gantt(sample, sample_day.date, charts_dir)
                )

    if "calendar" in selected_kinds and schedules:
        # Imported lazily so the calendar-view dependency (and Pillow
        # font loading) only loads when this chart family is actually
        # requested.  Other callers still pay no import cost.
        from src.scripts.persona.analytics.weekly_calendar import (
            DEFAULT_MIN_VISIBLE_MINUTES,
            render_weekly_calendars,
        )

        calendars = render_weekly_calendars(
            schedules,
            augmented_by_person,
            charts_dir,
            dpi=weekly_calendar_dpi,
            title_prefix=weekly_calendar_title_prefix,
            min_visible_minutes=(
                weekly_calendar_min_event_minutes
                if weekly_calendar_min_event_minutes is not None
                else DEFAULT_MIN_VISIBLE_MINUTES
            ),
            context_categories=weekly_calendar_context_categories,
        )

    per_member_heatmaps_out: dict[str, dict[str, dict[str, dict[str, Path]]]] = {}
    per_member_lines_out: dict[str, dict[str, dict[str, Path]]] = {}
    context_kinds = {"context-lines", "context-heatmap"}
    if context_kinds & selected_kinds:
        from src.scripts.persona.analytics.context_charts import (
            context_vmax_for_member,
            context_vmin_for_member,
            plot_context_member_heatmap,
            plot_per_member_context_lines,
        )

        # Per-member calmap heatmaps: one chart per persona / category /
        # member / metric, written flat under charts/ alongside event heatmaps.
        if "context-heatmap" in selected_kinds:
            for persona_id, persona_schedules in grouped.items():
                members = _members_for_persona(
                    persona_id, persona_config, persona_schedules
                )
                for category, member in members:
                    for metric in _PER_EVENT_HEATMAP_METRICS:
                        vmin = context_vmin_for_member(
                            persona_config, persona_id, category, member, metric
                        )
                        vmax = context_vmax_for_member(
                            persona_config, persona_id, category, member, metric
                        )
                        p = plot_context_member_heatmap(
                            persona_id,
                            persona_schedules,
                            category=category,
                            member=member,
                            out_dir=charts_dir,
                            metric=metric,
                            aggregate=heatmap_aggregate,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        if p is not None:
                            per_member_heatmaps_out.setdefault(
                                persona_id, {}
                            ).setdefault(category, {}).setdefault(member, {})[
                                metric
                            ] = p

        # Per-member cohort lines: one chart per (category, member, metric)
        # with one line per persona-template (analogous to per-event lines).
        if "context-lines" in selected_kinds:
            all_members: set[tuple[str, str]] = set()
            for persona_id, persona_schedules in grouped.items():
                all_members.update(
                    _members_for_persona(persona_id, persona_config, persona_schedules)
                )
            for category, member in sorted(all_members):
                # Restrict each persona to schedules where the persona's
                # catalog actually declares this (category, member) so
                # a persona without that member is excluded from the chart.
                schedules_by_persona: dict[str, list[PersonSchedule]] = {}
                for persona_id, persona_schedules in grouped.items():
                    persona_members = set(
                        _members_for_persona(
                            persona_id, persona_config, persona_schedules
                        )
                    )
                    if (category, member) in persona_members:
                        schedules_by_persona[persona_id] = persona_schedules
                # Every member in `all_members` came from some persona's
                # own member set, so at least one always matches here.
                if not schedules_by_persona:  # pragma: no cover - guard only
                    continue
                for metric in _PER_EVENT_LINE_METRICS:
                    p = plot_per_member_context_lines(
                        category,
                        member,
                        metric,
                        schedules_by_persona,
                        charts_dir,
                    )
                    if p is not None:
                        per_member_lines_out.setdefault(category, {}).setdefault(
                            member, {}
                        )[metric] = p

    return ChartArtifacts(
        lines=lines,
        per_event_lines=per_event_lines,
        heatmaps=heatmaps,
        per_event_heatmaps=per_event_heatmaps,
        gantts=gantts,
        calendars=calendars,
        per_member_context_heatmaps=per_member_heatmaps_out,
        per_member_context_lines=per_member_lines_out,
    )


__all__ = [
    "ALL_CHART_KINDS",
    "ChartArtifacts",
    "DAY_MINUTES",
    "DEFAULT_CHART_KINDS",
    "DPI",
    "HEATMAP_FIGSIZE",
    "HOURS_PER_DAY",
    "HorizonTrend",
    "SQUARE_FIGSIZE",
    "compute_horizon_trend",
    "daily_event_aggregate",
    "daily_mean_and_ci",
    "duration_unit_for_event",
    "event_count_matrix",
    "group_by_persona",
    "heatmap_vmax_for_event",
    "heatmap_vmin_for_event",
    "per_event_persona_daily",
    "plot_per_event_persona_lines",
    "plot_person_day_gantt",
    "plot_persona_heatmap",
    "plot_persona_horizon_lines",
    "render_all_charts",
    "variable_heatmap_events",
]
