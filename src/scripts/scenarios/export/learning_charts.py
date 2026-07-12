"""Per-person learning charts for the augmenters, on the matplotlib Agg backend."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from statistics import stdev

from src.scripts.scenarios.export.report_writer import (
    aggregate_weekly_gain,
    per_person_trajectory,
    weekly_gain_distribution,
)

logger = logging.getLogger(__name__)

DPI = 300

_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    # "#009E73",  # bluish green
    "#BD3232",  # red
    # "#CC79A7",  # reddish purple
    "#00821A",  # green
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#A6028A",  # violet
    "#44AA99",  # teal
    # "#882255",  # wine
    "#031b8a",  # dark blue
    "#999999",  # gray
    "#F0E442",  # yellow
    "#000000",  # black
)
_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h")
_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")


def _style_at(idx: int) -> tuple[str, str, str]:
    """Return a distinct `(color, marker, linestyle)` for the `idx`-th series."""
    return (
        _PALETTE[idx % len(_PALETTE)],
        _MARKERS[idx % len(_MARKERS)],
        _LINESTYLES[idx % len(_LINESTYLES)],
    )


def _matplotlib():
    """Return the pyplot module on the Agg backend, or None when unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _series(rows: list[dict]) -> tuple[list[int], list[float]]:
    """Return (week_indices, gains) for one person, sorted and gain-complete."""
    pairs = []
    for row in rows:
        gain = row.get("weighted_gain")
        if gain is None:
            continue
        pairs.append((int(row.get("week_index", 0)), float(gain)))
    pairs.sort()
    return [w for w, _ in pairs], [g for _, g in pairs]


def _week_count(by_person: dict[str, list[dict]]) -> int:
    """Number of distinct week indices present across all persons."""
    weeks: set[int] = set()
    for rows in by_person.values():
        for row in rows:
            if row.get("weighted_gain") is not None:
                weeks.add(int(row.get("week_index", 0)))
    return len(weeks)


def _spaghetti(plt, by_person, distribution, out_path: Path, title: str) -> Path:
    """Thin per-person lines under the cohort median and p25-p75 band."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for rows in by_person.values():
        weeks, gains = _series(rows)
        if weeks:
            ax.plot(weeks, gains, color="0.75", linewidth=0.8, alpha=0.6)
    wk = [d["week_index"] for d in distribution]
    ax.plot(
        wk,
        [d["median_weighted_gain"] for d in distribution],
        color="C0",
        linewidth=2.5,
        label="median",
    )
    ax.fill_between(
        wk,
        [d["p25_weighted_gain"] for d in distribution],
        [d["p75_weighted_gain"] for d in distribution],
        color="C0",
        alpha=0.2,
        label="p25-p75",
    )
    ax.set_xlabel("week")
    ax.set_ylabel("weighted gain")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def _onset_aligned(
    plt, by_person, trajectory, distribution, out_path: Path, title: str
) -> Path | None:
    """Start-aligned mean next to the onset-aligned mean; None when no onset."""
    by_offset: dict[int, list[float]] = {}
    for pid, rows in by_person.items():
        onset = trajectory.get(pid, {}).get("onset_week")
        if onset is None:
            continue
        weeks, gains = _series(rows)
        for week, gain in zip(weeks, gains):
            by_offset.setdefault(week - onset, []).append(gain)
    if not by_offset:
        return None
    offsets = sorted(by_offset)
    onset_mean = [sum(by_offset[o]) / len(by_offset[o]) for o in offsets]
    base = distribution[0]["week_index"]
    start_x = [d["week_index"] - base for d in distribution]
    start_mean = [d["avg_weighted_gain"] for d in distribution]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(start_x, start_mean, color="0.5", linewidth=2.0, label="start-aligned mean")
    ax.plot(offsets, onset_mean, color="C1", linewidth=2.5, label="onset-aligned mean")
    ax.axvline(0, color="0.8", linewidth=1.0)
    ax.set_xlabel("weeks from start or onset")
    ax.set_ylabel("weighted gain")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def _rl_training_charts(
    plt, persons_dir: Path, out_dir: Path, scenario_id: str
) -> dict[str, Path]:
    """One reward + tasks-placed chart per `<pid>_rl_training.json`, if present."""
    paths: dict[str, Path] = {}
    for path in sorted(Path(persons_dir).glob("*_rl_training.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weeks = payload.get("weeks") or []
        if not weeks:
            continue
        pid = payload.get("person_id") or path.stem.replace("_rl_training", "")
        wk = [int(w.get("week_index", 0)) for w in weeks]
        reward = [float(w.get("episode_total_reward", 0.0)) for w in weeks]
        placed = [int(w.get("tasks_placed_this_week", 0)) for w in weeks]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wk, reward, color="C0", marker="o", label="episode reward")
        ax.set_xlabel("week")
        ax.set_ylabel("episode total reward", color="C0")
        twin = ax.twinx()
        twin.plot(wk, placed, color="C2", marker="s", label="tasks placed")
        twin.set_ylabel("tasks placed", color="C2")
        ax.set_title(f"{scenario_id} rl training: {pid}")
        out_path = out_dir / f"rl_training_{scenario_id}_{pid}.png"
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        paths[f"rl_training_{pid}"] = out_path
    return paths


def render_learning_charts(
    persons_dir: Path, out_dir: Path, *, scenario_id: str, method: str
) -> dict[str, Path]:
    """Render the per-run learning charts; return a name-to-path map (empty on skip)."""
    plt = _matplotlib()
    if plt is None:
        logger.warning("matplotlib unavailable; skipping learning charts")
        return {}
    by_person = aggregate_weekly_gain(persons_dir)["by_person"]
    if _week_count(by_person) < 2:
        logger.warning("fewer than 2 weeks of data; skipping learning charts")
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution = weekly_gain_distribution(by_person)
    trajectory = per_person_trajectory(by_person)
    tag = f"{scenario_id}_{method}"
    paths: dict[str, Path] = {
        "spaghetti": _spaghetti(
            plt,
            by_person,
            distribution,
            out_dir / f"learning_spaghetti_{tag}.png",
            f"{scenario_id} / {method}: per-person weekly gain",
        )
    }
    onset_path = _onset_aligned(
        plt,
        by_person,
        trajectory,
        distribution,
        out_dir / f"learning_onset_aligned_{tag}.png",
        f"{scenario_id} / {method}: onset-aligned mean",
    )
    if onset_path is not None:
        paths["onset_aligned"] = onset_path
    paths.update(_rl_training_charts(plt, persons_dir, out_dir, scenario_id))
    return paths


def _mean_ci_series(
    by_person: dict[str, list[dict]],
) -> tuple[list[int], list[float], list[float]]:
    """Per-week mean gain and 95% CI half-width (1.96 * SEM) across persons."""
    by_week: dict[int, list[float]] = {}
    for rows in by_person.values():
        for row in rows:
            gain = row.get("weighted_gain")
            if gain is None:
                continue
            by_week.setdefault(int(row.get("week_index", 0)), []).append(float(gain))
    weeks = sorted(by_week)
    means: list[float] = []
    cis: list[float] = []
    for week in weeks:
        values = by_week[week]
        n = len(values)
        means.append(sum(values) / n)
        cis.append(1.96 * stdev(values) / math.sqrt(n) if n > 1 else 0.0)
    return weeks, means, cis


def render_cross_augmenter_overlay(
    runs: list[tuple[str, str, Path]],
    out_dir: Path,
    *,
    filename: str = "cross_augmenter_weekly.png",
    labels: dict[str, str] | None = None,
) -> Path | None:
    """Overlay every run's average weekly gain with a 95% CI band, RL highlighted.

    `labels` maps a scenario id to a display name for the legend; ids
    absent from the map fall back to the id itself.
    """
    plt = _matplotlib()
    if plt is None:
        logger.warning("matplotlib unavailable; skipping cross-augmenter overlay")
        return None
    labels = labels or {}
    sid_counts = Counter(sid for sid, _, _ in runs)
    series: list[tuple[str, str, str, tuple[list[int], list[float], list[float]]]] = []
    for sid, method, persons_dir in runs:
        weeks, means, cis = _mean_ci_series(
            aggregate_weekly_gain(persons_dir)["by_person"]
        )
        if not weeks:
            continue
        display = labels.get(sid, sid)
        label = display if sid_counts[sid] == 1 else f"{display} / {method}"
        series.append((label, sid, method, (weeks, means, cis)))
    if not series:
        return None
    # Sort by label so styles depend only on the set of runs, not the order
    # they arrive in, and give each series a distinct palette color.
    series.sort(key=lambda item: item[0])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Square chart sized so its height is about 1.75x the (font-fixed)
    # legend height; shrinking the figure shrinks only the chart.
    fig, ax = plt.subplots(figsize=(6.05, 6.05))
    palette_idx = 0
    for label, sid, method, (weeks, means, cis) in series:
        if method == "human_coach":
            # The human-coach ceiling is pinned to a black line with a gray 95%
            # CI band, and stays out of the palette cycle so its presence never
            # shifts the colors the other methods carry in earlier runs.
            color, marker, linestyle, ci_color = "#000000", "o", "-", "#999999"
        else:
            color, marker, linestyle = _style_at(palette_idx)
            ci_color = color
            palette_idx += 1
        # RL is the learning baseline; draw it heavier and on top.
        emphasis = method == "rl"
        ax.plot(
            weeks,
            means,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=7,
            linewidth=2.6 if emphasis else 1.7,
            zorder=3 if emphasis else 2,
            label=label,
        )
        lower = [m - c for m, c in zip(means, cis)]
        upper = [m + c for m, c in zip(means, cis)]
        ax.fill_between(weeks, lower, upper, color=ci_color, alpha=0.15)
    ax.set_xlabel("weeks")
    ax.set_ylabel("Average Weighted Scheduling Gain with 95% CI")
    ax.set_ylim(0.0, 1.0)
    # Square plotting area regardless of the week / gain data ranges.
    ax.set_box_aspect(1)
    # Legend in the right margin so it never overlaps the lines.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Scheduler methods",
        fontsize="small",
        title_fontsize="small",
        framealpha=0.9,
    )
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
