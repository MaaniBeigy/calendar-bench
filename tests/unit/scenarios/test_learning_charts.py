"""Tests for the learning-chart rendering module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.scripts.scenarios.export import learning_charts
from src.scripts.scenarios.export.learning_charts import (
    render_cross_augmenter_overlay,
    render_learning_charts,
)


def _weekly(persons_dir: Path, pid: str, gains: list[float]) -> None:
    persons_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "week_index": i + 1,
            "week_start": f"2026-06-{1 + 7 * i:02d}",
            "weighted_gain": g,
        }
        for i, g in enumerate(gains)
    ]
    (persons_dir / f"{pid}_weekly_gain.json").write_text(
        json.dumps({"person_id": pid, "weeks": rows})
    )


def _rl_training(persons_dir: Path, pid: str, weeks: list[dict]) -> None:
    (persons_dir / f"{pid}_rl_training.json").write_text(
        json.dumps({"person_id": pid, "weeks": weeks})
    )


def test_matplotlib_returns_none_when_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert learning_charts._matplotlib() is None


def test_render_skips_without_matplotlib(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(learning_charts, "_matplotlib", lambda: None)
    _weekly(tmp_path / "persons", "p0", [0.3, 0.4, 0.6])
    out = render_learning_charts(
        tmp_path / "persons", tmp_path / "charts", scenario_id="s", method="rl"
    )
    assert out == {}


def test_render_skips_when_too_few_weeks(tmp_path: Path):
    _weekly(tmp_path / "persons", "p0", [0.3])
    out = render_learning_charts(
        tmp_path / "persons", tmp_path / "charts", scenario_id="s", method="rl"
    )
    assert out == {}


def test_render_writes_spaghetti_onset_and_rl_training(tmp_path: Path):
    persons = tmp_path / "persons"
    _weekly(persons, "p0", [0.3, 0.45, 0.6, 0.7])
    _weekly(persons, "p1", [0.4, 0.5, 0.62, 0.72])
    # a person with only a no-signal week exercises the empty-series skip
    (persons / "pnone_weekly_gain.json").write_text(
        json.dumps(
            {
                "person_id": "pnone",
                "weeks": [{"week_index": 1, "week_start": "x", "weighted_gain": None}],
            }
        )
    )
    _rl_training(
        persons,
        "p0",
        [
            {"week_index": 1, "episode_total_reward": 0.2, "tasks_placed_this_week": 1},
            {"week_index": 2, "episode_total_reward": 0.6, "tasks_placed_this_week": 5},
        ],
    )
    out = render_learning_charts(
        persons, tmp_path / "charts", scenario_id="s", method="rl"
    )
    assert out["spaghetti"].exists() and out["spaghetti"].stat().st_size > 0
    assert out["onset_aligned"].exists()
    assert out["rl_training_p0"].exists()


def test_render_skips_onset_when_no_learner(tmp_path: Path):
    persons = tmp_path / "persons"
    _weekly(persons, "p0", [0.5, 0.5, 0.5, 0.5])
    _weekly(persons, "p1", [0.6, 0.6, 0.6, 0.6])
    out = render_learning_charts(
        persons, tmp_path / "charts", scenario_id="s", method="greedy"
    )
    assert "spaghetti" in out
    assert "onset_aligned" not in out
    assert not any(k.startswith("rl_training") for k in out)


def test_rl_training_skips_empty_and_unparseable(tmp_path: Path):
    persons = tmp_path / "persons"
    _weekly(persons, "p0", [0.3, 0.45, 0.6, 0.7])
    _weekly(persons, "p1", [0.4, 0.5, 0.6, 0.7])
    _rl_training(persons, "p0", [])
    (persons / "p1_rl_training.json").write_text("{not json")
    out = render_learning_charts(
        persons, tmp_path / "charts", scenario_id="s", method="rl"
    )
    assert not any(k.startswith("rl_training") for k in out)


def test_cross_overlay_writes_one_global_png(tmp_path: Path):
    rl = tmp_path / "rl"
    greedy = tmp_path / "greedy"
    empty = tmp_path / "empty"
    empty.mkdir()
    # rl has several persons (95% CI band) including one with a missing week
    _weekly(rl, "p0", [0.3, 0.45, 0.6, 0.7])
    _weekly(rl, "p1", [0.35, 0.5, 0.62, 0.72])
    _weekly(rl, "p2", [0.3, None, 0.6, 0.7])
    _weekly(greedy, "p0", [0.5, 0.5, 0.5, 0.5])  # single person -> zero-width band
    out = render_cross_augmenter_overlay(
        [
            ("rl_baseline", "rl", rl),
            ("greedy_baseline", "greedy", greedy),
            ("ghost", "greedy", empty),  # no weekly data -> skipped
        ],
        tmp_path / "charts",
    )
    assert out is not None and out.name == "cross_augmenter_weekly.png"
    assert out.exists() and out.stat().st_size > 0


def test_cross_overlay_pins_yaxis_to_unit_range(tmp_path: Path, monkeypatch):
    plt = learning_charts._matplotlib()
    captured: dict = {}
    orig_subplots = plt.subplots

    def _spy(*a, **k):
        fig, ax = orig_subplots(*a, **k)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _spy)
    rl = tmp_path / "rl"
    _weekly(rl, "p0", [0.3, 0.45, 0.6, 0.7])
    render_cross_augmenter_overlay([("s1", "rl", rl)], tmp_path / "charts")
    assert captured["ax"].get_ylim() == (0.0, 1.0)


def test_cross_overlay_disambiguates_duplicate_scenarios(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _weekly(a, "p0", [0.3, 0.45, 0.6, 0.7])
    _weekly(b, "p0", [0.5, 0.5, 0.5, 0.5])
    out = render_cross_augmenter_overlay(
        [("s1", "rl", a), ("s1", "greedy", b)], tmp_path / "charts"
    )
    assert out is not None and out.exists()


def test_cross_overlay_none_without_matplotlib(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(learning_charts, "_matplotlib", lambda: None)
    out = render_cross_augmenter_overlay(
        [("s1", "rl", tmp_path / "rl")], tmp_path / "charts"
    )
    assert out is None


def test_cross_overlay_none_when_no_weekly_data(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = render_cross_augmenter_overlay([("s1", "rl", empty)], tmp_path / "charts")
    assert out is None


def test_cli_render_helpers_write_charts(tmp_path: Path):
    from src.scripts.scenarios import cli

    base = tmp_path / "scenarios" / "s1" / "rl"
    persons = base / "augmented" / "persons"
    _weekly(persons, "p0", [0.3, 0.45, 0.6, 0.7])
    cli._render_learning_charts_for_run(
        base / "evaluation", persons, scenario_id="s1", method="rl"
    )
    assert (base / "evaluation" / "charts").is_dir()

    greedy = tmp_path / "scenarios" / "s2" / "greedy" / "augmented" / "persons"
    _weekly(greedy, "p0", [0.5, 0.5, 0.5, 0.5])
    cli._render_cross_augmenter_charts(tmp_path, [("s1", "rl"), ("s2", "greedy")])
    assert (tmp_path / "charts" / "cross_augmenter_weekly.png").exists()


def _capture_axes(monkeypatch) -> list:
    """Collect the Axes object created by each overlay render."""
    plt = learning_charts._matplotlib()
    axes: list = []
    orig_subplots = plt.subplots

    def _spy(*a, **k):
        fig, ax = orig_subplots(*a, **k)
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _spy)
    return axes


def test_cross_overlay_uses_label_map_in_legend(tmp_path: Path, monkeypatch):
    axes = _capture_axes(monkeypatch)
    rl = tmp_path / "rl"
    _weekly(rl, "p0", [0.3, 0.45, 0.6, 0.7])
    render_cross_augmenter_overlay(
        [("dqn_rl_per_person", "rl", rl)],
        tmp_path / "charts",
        labels={"dqn_rl_per_person": "DQN RL per person"},
    )
    legend_texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert "DQN RL per person" in legend_texts
    assert "dqn_rl_per_person" not in legend_texts


def test_cross_overlay_styles_are_stable_across_input_order(
    tmp_path: Path, monkeypatch
):
    axes = _capture_axes(monkeypatch)
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    _weekly(a, "p0", [0.30, 0.45, 0.60, 0.70])
    _weekly(b, "p0", [0.50, 0.50, 0.50, 0.50])
    _weekly(c, "p0", [0.40, 0.50, 0.55, 0.60])
    runs = [("s_a", "greedy", a), ("s_b", "ptime", b), ("s_c", "llm_agent", c)]
    render_cross_augmenter_overlay(runs, tmp_path / "c1")
    render_cross_augmenter_overlay(list(reversed(runs)), tmp_path / "c2")

    def style_by_label(ax) -> dict:
        return {
            line.get_label(): (
                line.get_color(),
                line.get_marker(),
                line.get_linestyle(),
            )
            for line in ax.get_lines()
        }

    # Same set of runs in reversed order must produce identical per-series styles.
    assert style_by_label(axes[0]) == style_by_label(axes[1])


def test_cross_overlay_colors_are_distinct_within_palette(tmp_path: Path, monkeypatch):
    axes = _capture_axes(monkeypatch)
    runs = []
    for i in range(len(learning_charts._PALETTE)):
        d = tmp_path / f"s{i}"
        _weekly(d, "p0", [0.30, 0.45, 0.60, 0.70])
        runs.append((f"s_{i:02d}", "greedy", d))
    render_cross_augmenter_overlay(runs, tmp_path / "charts")
    colors = [line.get_color() for line in axes[0].get_lines()]
    # Up to the palette length every series takes a different color.
    assert len(colors) == len(learning_charts._PALETTE)
    assert len(set(colors)) == len(colors)


def test_cross_overlay_palette_is_colorblind_safe_set(tmp_path: Path, monkeypatch):
    palette = learning_charts._PALETTE
    assert len(set(palette)) == len(palette)
    assert all(c.startswith("#") and len(c) == 7 for c in palette)
