"""Per-persona context chart renderers keyed by category and member."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import matplotlib

matplotlib.use("Agg")
import calmap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if TYPE_CHECKING:
    from src.scripts.persona.config.schema import (
        ContextMember,
        Persona,
        PersonaConfig,
    )
    from src.scripts.persona.context.schema import ContextEpisode
    from src.scripts.persona.domain.schedule import PersonSchedule


# Match the event-heatmap defaults so context tiles look the same.
_HEATMAP_FIGSIZE = (12, 8)
_DPI = 300
_DEFAULT_CMAP = "YlOrRd"


# ---------------------------------------------------------------------------
# pure-data helpers
# ---------------------------------------------------------------------------


def contexts_by_day(
    episodes: Iterable["ContextEpisode"],
) -> dict[datetime.date, dict[str, dict[str, list["ContextEpisode"]]]]:
    """Group episodes by date, then category, then member name."""
    out: dict[datetime.date, dict[str, dict[str, list[ContextEpisode]]]] = {}
    for ep in episodes:
        day = out.setdefault(ep.date, {})
        cat = day.setdefault(ep.category, {})
        cat.setdefault(ep.name, []).append(ep)
    return out


def _members_for_category(
    episodes: Iterable["ContextEpisode"], category: str
) -> list[str]:
    """Return sorted distinct member names for one category."""
    seen: set[str] = set()
    for ep in episodes:
        if ep.category == category:
            seen.add(ep.name)
    return sorted(seen)


def daily_context_aggregate(
    schedules: Iterable["PersonSchedule"],
    *,
    category: str,
    member: str,
    metric: str,
    aggregate: str = "mean",
) -> pd.Series:
    """Return per-day count or duration for one context member, pooled across persons."""
    if metric not in ("count", "duration"):
        raise ValueError(f"metric must be 'count' or 'duration', got {metric!r}")
    if aggregate not in ("mean", "sum"):
        raise ValueError(f"aggregate must be 'mean' or 'sum', got {aggregate!r}")
    schedules = list(schedules)
    if not schedules:
        return pd.Series(dtype=float)
    all_dates: set[datetime.date] = set()
    for sched in schedules:
        for day in sched.days:
            all_dates.add(day.date)
    if not all_dates:
        return pd.Series(dtype=float)
    sorted_dates = sorted(all_dates)
    per_person: list[dict[datetime.date, float]] = []
    for sched in schedules:
        bucket: dict[datetime.date, float] = {d: 0.0 for d in sorted_dates}
        for ep in sched.contexts:
            if ep.category != category or ep.name != member:
                continue
            if ep.date not in bucket:
                continue
            if metric == "count":
                bucket[ep.date] += 1.0
            else:
                bucket[ep.date] += float(ep.duration)
        per_person.append(bucket)
    matrix = np.array(
        [[bucket[d] for d in sorted_dates] for bucket in per_person], dtype=float
    )
    if aggregate == "mean":
        values = matrix.mean(axis=0)
    else:
        values = matrix.sum(axis=0)
    return pd.Series(
        values, index=pd.DatetimeIndex([pd.Timestamp(d) for d in sorted_dates])
    )


def per_member_context_persona_daily(
    schedules: list["PersonSchedule"],
    *,
    category: str,
    member: str,
    metric: str,
) -> np.ndarray:
    """Return a (num_days, num_persons) matrix of daily values for one member."""
    if metric not in ("count", "duration"):
        raise ValueError(f"metric must be 'count' or 'duration', got {metric!r}")
    num_persons = len(schedules)
    num_days = max((len(s.days) for s in schedules), default=0)
    matrix = np.zeros((num_days, num_persons), dtype=float)
    for p_idx, sched in enumerate(schedules):
        if not sched.days:
            continue
        date_to_idx = {day.date: d_idx for d_idx, day in enumerate(sched.days)}
        for ep in sched.contexts:
            if ep.category != category or ep.name != member:
                continue
            d_idx = date_to_idx.get(ep.date)
            if d_idx is None:
                continue
            if metric == "count":
                matrix[d_idx, p_idx] += 1.0
            else:
                matrix[d_idx, p_idx] += float(ep.duration)
    return matrix


# ---------------------------------------------------------------------------
# persona-catalog vmin / vmax
# ---------------------------------------------------------------------------


def _to_minutes(value: int | float, unit: str) -> float:
    """Convert a duration value to minutes, accepting `minutes` or `hours`."""
    if unit == "hours":
        return float(value) * 60.0
    return float(value)


def _resolve_persona(
    persona_config: "PersonaConfig | None", persona_id: str
) -> "Persona | None":
    """Return the persona entry with the given id or None."""
    if persona_config is None:
        return None
    for persona in persona_config.personas:
        if persona.id == persona_id:
            return persona
    return None


def _resolve_member(
    persona_config: "PersonaConfig | None",
    persona_id: str,
    category: str,
    member: str,
) -> "ContextMember | None":
    """Return the ContextMember entry for the given persona, category, member, or None."""
    persona = _resolve_persona(persona_config, persona_id)
    if persona is None:
        return None
    cat = persona.contexts.get(category)
    if cat is None:
        return None
    return cat.members.get(member)


def context_vmin_for_member(
    persona_config: "PersonaConfig | None",
    persona_id: str,
    category: str,
    member: str,
    metric: str,
) -> float | None:
    """Return the catalog lower bound for the heatmap colorbar in member units."""
    member_def = _resolve_member(persona_config, persona_id, category, member)
    if member_def is None:
        return None
    if metric == "count":
        return float(member_def.total_event_episodes.min)
    if metric == "duration":
        td = member_def.total_event_duration
        return _to_minutes(td.min, td.unit)
    return None


def context_vmax_for_member(
    persona_config: "PersonaConfig | None",
    persona_id: str,
    category: str,
    member: str,
    metric: str,
) -> float | None:
    """Return the catalog upper bound for the heatmap colorbar in member units."""
    member_def = _resolve_member(persona_config, persona_id, category, member)
    if member_def is None:
        return None
    if metric == "count":
        return float(member_def.total_event_episodes.max)
    if metric == "duration":
        td = member_def.total_event_duration
        return _to_minutes(td.max, td.unit)
    return None


def _valid_scale(vmin: float | None, vmax: float | None) -> bool:
    """True when both bounds are present and form a non-degenerate range."""
    return vmin is not None and vmax is not None and vmax > vmin


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _ensure_dir(out_dir: Path | str) -> Path:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_context_member_heatmap(
    persona_id: str,
    schedules: list["PersonSchedule"],
    *,
    category: str,
    member: str,
    out_dir: Path | str,
    metric: str,
    aggregate: str = "mean",
    cmap: str = _DEFAULT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path | None:
    """Write a calmap year heatmap for one persona, category, member, and metric."""
    if metric not in ("count", "duration"):
        raise ValueError(f"metric must be 'count' or 'duration', got {metric!r}")
    series = daily_context_aggregate(
        schedules,
        category=category,
        member=member,
        metric=metric,
        aggregate=aggregate,
    )
    if series.empty:
        return None
    target = _ensure_dir(out_dir) / (
        f"heatmap_context_{persona_id}_{category}_{member}_{metric}_{aggregate}.png"
    )
    fig, ax = plt.subplots(figsize=_HEATMAP_FIGSIZE)
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
        f"Persona '{persona_id}' - {aggregate} of {metric_label} per day "
        f"(context {category}/{member})",
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
    collections = ax.collections
    if len(collections) >= 2:  # pragma: no branch
        cbar_ax = fig.add_axes([0.10, 0.08, 0.80, 0.03])
        cbar = fig.colorbar(collections[1], cax=cbar_ax, orientation="horizontal")
        cbar_label = f"{metric_label} ({aggregate})"
        if use_config_scale:
            cbar_label += f"; {float(vmin):.0f} to {float(vmax):.0f} (config range)"
        cbar.set_label(cbar_label, fontsize=10)
    plt.subplots_adjust(top=0.88, bottom=0.22, left=0.05, right=0.95)
    fig.savefig(target, dpi=_DPI, facecolor="white")
    plt.close(fig)
    return target


# 95% normal-approximation z-score.
_CI95_Z = 1.959963984540054


def _daily_mean_and_ci(
    values: np.ndarray, *, z: float = _CI95_Z
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-day mean and 95% CI bounds across the person axis."""
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


def _convert_minutes(values: np.ndarray, unit: str) -> np.ndarray:
    return values / 60.0 if unit == "hours" else values


def _metric_axis_label(metric: str, unit: str) -> str:
    if metric == "count":
        return "Episodes per day"
    return f"Total daily duration ({unit})"


def _palette(n: int) -> list[str]:
    """Return n matplotlib palette colors from tab10."""
    cmap = matplotlib.colormaps["tab10"]
    return [cmap(i % 10) for i in range(max(n, 1))]


def plot_per_member_context_lines(
    category: str,
    member: str,
    metric: str,
    schedules_by_persona: dict[str, list["PersonSchedule"]],
    out_dir: Path | str,
    *,
    duration_unit: str = "minutes",
) -> Path | None:
    """Write one PNG per category, member, metric with one mean+CI line per persona."""
    if metric not in ("count", "duration"):
        raise ValueError(f"metric must be 'count' or 'duration', got {metric!r}")
    series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    max_days = 0
    has_signal = False
    for persona_id in sorted(schedules_by_persona.keys()):
        scheds = schedules_by_persona[persona_id]
        if not scheds:
            continue
        matrix = per_member_context_persona_daily(
            scheds, category=category, member=member, metric=metric
        )
        if matrix.size == 0 or matrix.shape[0] == 0:
            continue
        mean, lo, hi = _daily_mean_and_ci(matrix)
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
    target = _ensure_dir(out_dir) / (f"lines_context_{category}_{member}_{metric}.png")
    colors = _palette(len(series))
    fig, ax = plt.subplots(figsize=(8, 8))
    for color, (persona_id, mean, lo, hi, n_persons) in zip(colors, series):
        days = np.arange(1, mean.shape[0] + 1)
        label = f"{persona_id} (n={n_persons})"
        ax.plot(days, mean, color=color, label=label, linewidth=1.6)
        ax.fill_between(days, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Day of horizon")
    ax.set_ylabel(_metric_axis_label(metric, duration_unit))
    metric_word = "episodes" if metric == "count" else "duration"
    ax.set_title(
        f"context {category}/{member} daily {metric_word}; "
        "mean +/- 95% CI across persons"
    )
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(target, dpi=_DPI)
    plt.close(fig)
    return target


__all__ = [
    "context_vmax_for_member",
    "context_vmin_for_member",
    "contexts_by_day",
    "daily_context_aggregate",
    "per_member_context_persona_daily",
    "plot_context_member_heatmap",
    "plot_per_member_context_lines",
]
