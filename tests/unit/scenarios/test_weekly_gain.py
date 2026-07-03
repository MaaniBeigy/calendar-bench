"""Tests for the per-week scheduling-gain plumbing."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from src.scripts.scenarios.export.benchmark_report import (
    _format_weekly_gain,
    build_benchmark_markdown,
)
from src.scripts.scenarios.export.json_writer import write_weekly_gain_sidecar
from src.scripts.scenarios.export.report_writer import (
    aggregate_weekly_gain,
    format_weekly_gain_report,
    write_evaluation_reports,
)

DATE = datetime.date(2026, 6, 1)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sidecar(pid: str, weeks):
    return {"person_id": pid, "weeks": weeks}


def _week(week_index: int, weighted_gain: float) -> dict:
    return {
        "week_index": week_index,
        "week_start": (DATE + datetime.timedelta(weeks=week_index - 1)).isoformat(),
        "weighted_gain": weighted_gain,
        "gains": {
            "recommended_task_coverage": weighted_gain,
            "user_preference_deviation": None,
        },
    }


# ---------------------------------------------------------------------------
# write_weekly_gain_sidecar
# ---------------------------------------------------------------------------


def test_write_weekly_gain_sidecar_writes_payload(tmp_path: Path):
    rows = [_week(1, 0.60), _week(2, 0.65)]
    out = write_weekly_gain_sidecar(
        tmp_path / "p1_weekly_gain.json", person_id="p1", weekly_records=rows
    )
    assert out is not None
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["person_id"] == "p1"
    assert len(payload["weeks"]) == 2


def test_write_rl_training_sidecar_writes_payload(tmp_path: Path):
    from src.scripts.scenarios.export.json_writer import write_rl_training_sidecar

    rows = [
        {
            "week_index": 1,
            "episode_steps": 12,
            "episode_valid_placements": 8,
            "episode_total_reward": 0.55,
            "episode_terminal_reward": 0.55,
            "epsilon_before_learn": 1.0,
            "tasks_placed_this_week": 8,
            "tasks_total": 20,
        }
    ]
    out = write_rl_training_sidecar(
        tmp_path / "p1_rl_training.json", person_id="p1", training_records=rows
    )
    assert out is not None
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["person_id"] == "p1"
    assert payload["weeks"][0]["episode_total_reward"] == 0.55


def test_write_rl_training_sidecar_empty_returns_none(tmp_path: Path):
    from src.scripts.scenarios.export.json_writer import write_rl_training_sidecar

    assert (
        write_rl_training_sidecar(
            tmp_path / "p1_rl_training.json", person_id="p1", training_records=[]
        )
        is None
    )


def test_write_weekly_gain_sidecar_empty_returns_none(tmp_path: Path):
    out = write_weekly_gain_sidecar(
        tmp_path / "p1.json", person_id="p1", weekly_records=[]
    )
    assert out is None


# ---------------------------------------------------------------------------
# aggregate_weekly_gain
# ---------------------------------------------------------------------------


def test_aggregate_weekly_gain_returns_empty_when_dir_missing(tmp_path: Path):
    agg = aggregate_weekly_gain(tmp_path / "missing")
    assert agg == {"weeks": [], "by_person": {}}


def test_aggregate_weekly_gain_averages_per_week(tmp_path: Path):
    _write(
        tmp_path / "p1_weekly_gain.json", _sidecar("p1", [_week(1, 0.5), _week(2, 0.6)])
    )
    _write(
        tmp_path / "p2_weekly_gain.json", _sidecar("p2", [_week(1, 0.7), _week(2, 0.8)])
    )
    agg = aggregate_weekly_gain(tmp_path)
    weeks = agg["weeks"]
    assert weeks[0]["week_index"] == 1
    assert weeks[0]["avg_weighted_gain"] == pytest.approx(0.60)
    assert weeks[0]["scored_persons"] == 2
    assert weeks[1]["avg_weighted_gain"] == pytest.approx(0.70)


def test_aggregate_weekly_gain_omits_non_renorm_key(tmp_path: Path):
    """The aggregate carries only the masked weighted gain, no non-renorm key."""
    _write(tmp_path / "p1_weekly_gain.json", _sidecar("p1", [_week(1, 0.5)]))
    agg = aggregate_weekly_gain(tmp_path)
    assert "avg_non_renormalized_gain" not in agg["weeks"][0]


def test_aggregate_weekly_gain_skips_unparseable(tmp_path: Path):
    (tmp_path / "p1_weekly_gain.json").write_text("not json", encoding="utf-8")
    _write(tmp_path / "p2_weekly_gain.json", _sidecar("p2", [_week(1, 0.5)]))
    agg = aggregate_weekly_gain(tmp_path)
    assert list(agg["by_person"].keys()) == ["p2"]


def test_aggregate_weekly_gain_drops_rows_without_gain(tmp_path: Path):
    bad = {"week_index": 1, "weighted_gain": None, "week_start": ""}
    good = _week(2, 0.5)
    _write(tmp_path / "p1_weekly_gain.json", _sidecar("p1", [bad, good]))
    agg = aggregate_weekly_gain(tmp_path)
    assert [w["week_index"] for w in agg["weeks"]] == [2]


def test_aggregate_weekly_gain_handles_empty_weeks(tmp_path: Path):
    _write(tmp_path / "p1_weekly_gain.json", _sidecar("p1", []))
    agg = aggregate_weekly_gain(tmp_path)
    assert agg == {"weeks": [], "by_person": {}}


# ---------------------------------------------------------------------------
# format_weekly_gain_report
# ---------------------------------------------------------------------------


def test_format_weekly_gain_report_empty():
    out = format_weekly_gain_report({"weeks": []})
    assert "no per-week data" in out


def test_format_weekly_gain_report_renders_rows():
    weeks = [
        {
            "week_index": 1,
            "week_start": "2026-06-01",
            "avg_weighted_gain": 0.61,
            "scored_persons": 4,
        }
    ]
    text = format_weekly_gain_report({"weeks": weeks})
    assert "Wk 1" in text
    assert "0.6100" in text
    assert "n=4" in text
    assert "non_renorm" not in text


# ---------------------------------------------------------------------------
# write_evaluation_reports weekly path
# ---------------------------------------------------------------------------


def test_write_evaluation_reports_emits_weekly_pair(tmp_path: Path):
    weekly = {
        "weeks": [
            {
                "week_index": 1,
                "week_start": "2026-06-01",
                "avg_weighted_gain": 0.5,
                "scored_persons": 1,
            }
        ]
    }
    written = write_evaluation_reports([], tmp_path, weekly_gain=weekly)
    assert "weekly_scheduling_gain" in written
    txt, js = written["weekly_scheduling_gain"]
    assert txt.exists() and js.exists()


def test_write_evaluation_reports_skips_weekly_when_empty(tmp_path: Path):
    written = write_evaluation_reports([], tmp_path, weekly_gain={"weeks": []})
    assert "weekly_scheduling_gain" not in written


# ---------------------------------------------------------------------------
# Benchmark report markdown
# ---------------------------------------------------------------------------


def _runs_with_weekly():
    def wk(idx, start, gain):
        return {
            "week_index": idx,
            "week_start": start,
            "avg_weighted_gain": gain,
            "scored_persons": 4,
        }

    return [
        (
            "scenario_a",
            "rl",
            {
                "weekly": {
                    "weeks": [
                        wk(1, "2026-06-01", 0.55),
                        wk(2, "2026-06-08", 0.70),
                    ]
                }
            },
        ),
        (
            "scenario_b",
            "greedy",
            {"weekly": {"weeks": [wk(1, "2026-06-01", 0.50)]}},
        ),
    ]


def test_format_weekly_gain_section_renders_table():
    lines = _format_weekly_gain(_runs_with_weekly())
    text = "\n".join(lines)
    assert "## Weekly scheduling gain" in text
    assert "Wk1" in text and "Wk2" in text
    assert "scenario_a" in text
    assert "0.5500" in text
    assert "+0.1500" in text  # delta last-first for scenario_a
    # No separate non-renormalized sub-table any more.
    assert "non-renormalized" not in text


def test_format_weekly_gain_section_returns_empty_when_absent():
    runs = [("scenario_x", "rl", {})]
    assert _format_weekly_gain(runs) == []


# ---------------------------------------------------------------------------
# Masked weighted gain semantics
# ---------------------------------------------------------------------------


def test_masked_weighted_charges_in_mask_none_legs():
    """A masked in-mask leg with no signal is charged full loss (gain 0)."""
    from src.scripts.scenarios.config.schema import LossWeights
    from src.scripts.scenarios.metrics.loss import LossComponents

    weights = LossWeights()
    comp = LossComponents(
        cov=0.0,
        cal=0.0,
        pref=None,
        disp=0.0,
        merge=None,
        spread=0.0,
        divide=None,
        context_fit=None,
    )
    only_solution_legs = comp.weighted(weights, {"cov", "cal"})
    with_pref = comp.weighted(weights, {"cov", "cal", "pref"})
    assert only_solution_legs == 0.0
    assert with_pref > only_solution_legs


def test_masked_weighted_zero_placement_charges_full_loss():
    """`zero_placement=True` charges every in-mask leg full loss (gain 0)."""
    from src.scripts.scenarios.config.schema import LossWeights
    from src.scripts.scenarios.metrics.loss import LossComponents

    weights = LossWeights()
    comp = LossComponents(
        cov=0.0,
        cal=0.0,
        pref=0.0,
        disp=0.0,
        merge=0.0,
        spread=0.0,
        divide=0.0,
        context_fit=0.0,
    )
    full = comp.weighted(weights, {"cov", "cal", "disp", "spread"}, zero_placement=True)
    assert full == pytest.approx(1.0)


def test_masked_view_marks_out_of_mask_none_and_in_mask_no_signal_full():
    """`masked_view` projects legs for the report: out->None, in-no-signal->1.0."""
    from src.scripts.scenarios.metrics.loss import LossComponents

    comp = LossComponents(
        cov=0.2,
        cal=0.1,
        pref=None,
        disp=0.3,
        merge=None,
        spread=0.0,
        divide=None,
        context_fit=None,
    )
    view = comp.masked_view({"cov", "cal", "pref"})
    assert view.cov == 0.2
    assert view.cal == 0.1
    assert view.pref == 1.0  # in mask, no signal -> full loss
    assert view.disp is None  # out of mask
    assert view.merge is None


# ---------------------------------------------------------------------------
# _slice_solution_for_week
# ---------------------------------------------------------------------------


def test_slice_solution_for_week_filters_dates():
    from src.scripts.scenarios.domain.calendar import (
        AugmentedCalendar,
        CalendarTrace,
    )
    from src.scripts.scenarios.domain.solution import SchedulingSolution
    from src.scripts.scenarios.metrics.loss import _slice_solution_for_week
    from tests.unit.scenarios.conftest import make_event, make_scheduled, make_task

    in_week = DATE
    out_week = DATE + datetime.timedelta(days=10)
    t1 = make_task(label="walk", duration_min=30, duration_max=30)
    t2 = make_task(label="run", duration_min=30, duration_max=30)
    st_in = make_scheduled(task=t1, date=in_week)
    st_out = make_scheduled(task=t2, date=out_week)
    ev_in = make_event(label="lunch", date=in_week)
    ev_out = make_event(label="dinner", date=out_week)
    solution = SchedulingSolution(
        person_id="p1",
        augmented_calendar=AugmentedCalendar(
            person_id="p1",
            base_events=[ev_in, ev_out],
            scheduled_tasks=[st_in, st_out],
        ),
        tasks=[t1, t2],
        scheduled=[st_in, st_out],
        unscheduled=[],
    )
    calendar = CalendarTrace(person_id="p1", events=[ev_in, ev_out])
    week_dates = [in_week + datetime.timedelta(days=i) for i in range(7)]
    sliced_sol, sliced_cal = _slice_solution_for_week(solution, calendar, week_dates)
    assert sliced_sol.scheduled == [st_in]
    assert sliced_sol.unscheduled == [t2]
    assert sliced_cal.events == [ev_in]


def test_build_benchmark_markdown_includes_weekly_section(tmp_path: Path):
    runs = [
        (
            "scenario_a",
            "rl",
            {
                "total": {
                    "average_total_gain": 0.6,
                    "scored_persons": 4,
                    "empty_plan_persons": 0,
                    "average_gains": {},
                },
                "weekly": {
                    "weeks": [
                        {
                            "week_index": 1,
                            "week_start": "2026-06-01",
                            "avg_weighted_gain": 0.55,
                            "scored_persons": 4,
                        }
                    ]
                },
            },
        )
    ]
    md = build_benchmark_markdown("experiment_x", runs)
    assert "## Weekly scheduling gain" in md
