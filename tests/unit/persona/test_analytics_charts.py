"""Unit tests for src.scripts.persona.analytics.charts."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scripts.persona.analytics.charts import (
    ChartArtifacts,
    HorizonTrend,
    _members_for_persona,
    compute_horizon_trend,
    daily_event_aggregate,
    daily_mean_and_ci,
    duration_unit_for_event,
    event_count_matrix,
    group_by_persona,
    heatmap_vmax_for_event,
    heatmap_vmin_for_event,
    per_event_persona_daily,
    plot_per_event_persona_lines,
    plot_person_day_gantt,
    plot_persona_heatmap,
    plot_persona_horizon_lines,
    render_all_charts,
    variable_heatmap_events,
)
from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    EventOverride,
    Persona,
    PersonaConfig,
    PersonaEventStage,
    TemporalPattern,
    TotalDuration,
)
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

_START = _dt.date(2026, 9, 7)  # Monday
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -------------------------------------------------------------------------------------
# ----------------------------------- fixtures ----------------------------------------
# -------------------------------------------------------------------------------------


def _day(
    day_index: int,
    events: dict[str, list[tuple[int, int]]],
    start_date: _dt.date = _START,
) -> DaySchedule:
    return DaySchedule(
        day_index=day_index,
        date=start_date + _dt.timedelta(days=day_index),
        weekday=_WEEKDAYS[day_index % 7],
        events={
            name: [
                EventInstance(event_name=name, start=s, duration=d) for s, d in pairs
            ]
            for name, pairs in events.items()
        },
        spillovers=[],
    )


def _schedule(
    person_id: str,
    day_events: list[dict[str, list[tuple[int, int]]]],
    *,
    persona_id: str = "alice",
    person_seed: int = 42,
    start_date: _dt.date = _START,
) -> PersonSchedule:
    return PersonSchedule(
        person_id=person_id,
        persona_id=persona_id,
        person_seed=person_seed,
        days=[_day(i, ev, start_date=start_date) for i, ev in enumerate(day_events)],
    )


# -------------------------------------------------------------------------------------
# --------------------------------- group_by_persona ----------------------------------
# -------------------------------------------------------------------------------------


def test_group_by_persona_groups_in_input_order():
    schedules = [
        _schedule("a0", [{}], persona_id="alice"),
        _schedule("b0", [{}], persona_id="bob"),
        _schedule("a1", [{}], persona_id="alice"),
    ]
    grouped = group_by_persona(schedules)
    assert list(grouped.keys()) == ["alice", "bob"]
    assert [s.person_id for s in grouped["alice"]] == ["a0", "a1"]
    assert [s.person_id for s in grouped["bob"]] == ["b0"]


def test_group_by_persona_returns_empty_dict_for_no_schedules():
    assert group_by_persona([]) == {}


# -------------------------------------------------------------------------------------
# ------------------------------ event_count_matrix -----------------------------------
# -------------------------------------------------------------------------------------


def test_event_count_matrix_shapes_and_values():
    schedules = [
        _schedule(
            "p0",
            [{"lunch": [(750, 45)]}, {"lunch": [(750, 45)], "running": [(420, 60)]}],
        ),
        _schedule("p1", [{"lunch": [(750, 45)]}, {}]),
    ]
    matrix, types = event_count_matrix(schedules)
    assert types == ["lunch", "running"]
    assert matrix["lunch"].shape == (2, 2)
    assert matrix["lunch"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert matrix["running"].tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_event_count_matrix_pads_uneven_schedules():
    schedules = [
        _schedule("p0", [{"lunch": [(750, 45)]}, {"lunch": [(750, 45)]}]),
        _schedule("p1", [{"lunch": [(750, 45)]}]),
    ]
    matrix, _ = event_count_matrix(schedules)
    assert matrix["lunch"].shape == (2, 2)
    assert matrix["lunch"][1, 1] == 0.0


def test_event_count_matrix_uses_provided_event_types():
    schedules = [_schedule("p0", [{"lunch": [(750, 45)]}])]
    matrix, types = event_count_matrix(schedules, event_types=["dinner", "lunch"])
    assert types == ["dinner", "lunch"]
    assert matrix["dinner"].sum() == 0
    assert matrix["lunch"][0, 0] == 1.0


def test_event_count_matrix_handles_empty_schedules():
    matrix, types = event_count_matrix([])
    assert types == []
    assert matrix == {}


# -------------------------------------------------------------------------------------
# ----------------------------- compute_horizon_trend ---------------------------------
# -------------------------------------------------------------------------------------


def test_compute_horizon_trend_returns_means_and_stds_per_day():
    schedules = [
        _schedule("p0", [{"lunch": [(750, 45)]}, {"lunch": [(750, 45), (800, 30)]}]),
        _schedule("p1", [{"lunch": [(750, 45)]}, {"lunch": [(750, 45)]}]),
    ]
    trend = compute_horizon_trend(schedules)
    assert isinstance(trend, HorizonTrend)
    assert trend.event_types == ("lunch",)
    assert trend.num_days == 2
    assert trend.num_persons == 2
    np.testing.assert_allclose(trend.means["lunch"], [1.0, 1.5])
    np.testing.assert_allclose(trend.stds["lunch"], [0.0, 0.5])


def test_compute_horizon_trend_with_empty_schedules_is_zero_shape():
    trend = compute_horizon_trend([])
    assert trend.event_types == ()
    assert trend.num_days == 0
    assert trend.num_persons == 0
    assert trend.means == {}
    assert trend.stds == {}


def test_compute_horizon_trend_with_pinned_event_types_and_no_persons():
    """Explicit `event_types` pins which events appear in the trend even
    on an empty population. The `shape[1] == 0` branch returns
    zero-filled arrays so callers can compare populations of different
    sizes without a NumPy "mean of empty slice" warning."""
    trend = compute_horizon_trend([], event_types=["lunch", "dinner"])
    assert trend.event_types == ("lunch", "dinner")
    assert trend.num_persons == 0
    assert trend.means["lunch"].shape == (0,)
    assert trend.stds["dinner"].shape == (0,)


def test_compute_horizon_trend_event_types_pin_zero_filled_when_event_missing():
    schedules = [_schedule("p0", [{"lunch": [(750, 45)]}])]
    trend = compute_horizon_trend(schedules, event_types=["dinner"])
    assert "dinner" in trend.means
    np.testing.assert_array_equal(trend.means["dinner"], np.zeros(1))
    np.testing.assert_array_equal(trend.stds["dinner"], np.zeros(1))


# -------------------------------------------------------------------------------------
# --------------------------- daily_event_aggregate -----------------------------------
# -------------------------------------------------------------------------------------


def test_daily_event_aggregate_counts_episodes_mean_across_persons():
    schedules = [
        _schedule("p0", [{"lunch": [(750, 45)]}, {"lunch": [(750, 45), (800, 30)]}]),
        _schedule("p1", [{"lunch": [(750, 45)]}, {"lunch": [(750, 45)]}]),
    ]
    series = daily_event_aggregate(schedules, event_name="lunch", aggregate="mean")
    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert series.iloc[0] == pytest.approx(1.0)
    assert series.iloc[1] == pytest.approx(1.5)


def test_daily_event_aggregate_counts_episodes_sum_across_persons():
    schedules = [
        _schedule("p0", [{"lunch": [(750, 45), (800, 30)]}]),
        _schedule("p1", [{"lunch": [(750, 45)]}]),
    ]
    series = daily_event_aggregate(schedules, event_name="lunch", aggregate="sum")
    assert series.iloc[0] == pytest.approx(3.0)


def test_daily_event_aggregate_duration_metric_sums_minutes():
    schedules = [
        _schedule("p0", [{"lunch": [(750, 45), (800, 30)]}]),
    ]
    series = daily_event_aggregate(
        schedules, event_name="lunch", metric="duration", aggregate="sum"
    )
    assert series.iloc[0] == pytest.approx(75.0)


def test_daily_event_aggregate_no_event_filter_aggregates_all_event_types():
    schedules = [
        _schedule(
            "p0",
            [{"lunch": [(750, 45)], "running": [(420, 30)]}],
        )
    ]
    series_count = daily_event_aggregate(schedules, metric="count", aggregate="sum")
    assert series_count.iloc[0] == pytest.approx(2.0)
    series_dur = daily_event_aggregate(schedules, metric="duration", aggregate="sum")
    assert series_dur.iloc[0] == pytest.approx(75.0)


def test_daily_event_aggregate_no_event_filter_mean_across_persons():
    schedules = [
        _schedule("p0", [{"lunch": [(750, 45)], "running": [(420, 30)]}]),
        _schedule("p1", [{"lunch": [(750, 45)]}]),
    ]
    series = daily_event_aggregate(schedules, metric="count", aggregate="mean")
    # p0: 2 episodes, p1: 1 episode to mean 1.5.
    assert series.iloc[0] == pytest.approx(1.5)


def test_daily_event_aggregate_no_event_filter_duration_mean():
    """The branch where `event_name=None` and `metric='duration'` still
    accumulates per-day duration before the across-persons mean."""
    schedules = [
        _schedule("p0", [{"lunch": [(750, 45)], "running": [(420, 30)]}]),
        _schedule("p1", [{"lunch": [(750, 60)]}]),
    ]
    series = daily_event_aggregate(schedules, metric="duration", aggregate="mean")
    # p0: 75 min total, p1: 60 min total to mean 67.5.
    assert series.iloc[0] == pytest.approx(67.5)


def test_daily_event_aggregate_returns_empty_for_no_schedules():
    series = daily_event_aggregate([])
    assert series.empty


def test_daily_event_aggregate_filtered_event_count_zero_when_missing():
    schedules = [_schedule("p0", [{"lunch": [(750, 45)]}])]
    series = daily_event_aggregate(schedules, event_name="dinner")
    assert series.iloc[0] == 0.0


def test_daily_event_aggregate_filtered_event_duration_zero_when_missing():
    schedules = [_schedule("p0", [{"lunch": [(750, 45)]}])]
    series = daily_event_aggregate(schedules, event_name="dinner", metric="duration")
    assert series.iloc[0] == 0.0


def test_daily_event_aggregate_rejects_invalid_metric():
    with pytest.raises(ValueError, match="metric"):
        daily_event_aggregate([], metric="hours")  # type: ignore[arg-type]


def test_daily_event_aggregate_rejects_invalid_aggregate():
    with pytest.raises(ValueError, match="aggregate"):
        daily_event_aggregate([], aggregate="median")  # type: ignore[arg-type]


# -------------------------------------------------------------------------------------
# --------------------------- plot_persona_horizon_lines ------------------------------
# -------------------------------------------------------------------------------------


def test_plot_persona_horizon_lines_writes_a_png(tmp_path: Path):
    schedules = [_schedule("p0", [{"lunch": [(750, 45)]}, {"lunch": [(750, 45)]}])]
    target = plot_persona_horizon_lines("alice", schedules, tmp_path)
    assert target is not None
    assert target.name == "lines_alice.png"
    assert target.exists()
    assert target.stat().st_size > 0


def test_plot_persona_horizon_lines_returns_none_for_empty_schedules(tmp_path: Path):
    assert plot_persona_horizon_lines("alice", [], tmp_path) is None


def test_plot_persona_horizon_lines_returns_none_when_no_event_types(tmp_path: Path):
    schedules = [_schedule("p0", [{}, {}])]
    assert plot_persona_horizon_lines("alice", schedules, tmp_path) is None


# -------------------------------------------------------------------------------------
# --------------------------- plot_person_day_gantt -----------------------------------
# -------------------------------------------------------------------------------------


def test_plot_person_day_gantt_writes_a_png_for_a_matching_date(tmp_path: Path):
    schedule = _schedule(
        "p0",
        [
            {"lunch": [(750, 45)], "running": [(420, 60)]},
            {"lunch": [(750, 45)]},
        ],
    )
    target = plot_person_day_gantt(schedule, _START + _dt.timedelta(days=1), tmp_path)
    assert target is not None
    assert target.name == f"gantt_p0_{(_START + _dt.timedelta(days=1)).isoformat()}.png"
    assert target.exists()


def test_plot_person_day_gantt_returns_none_when_date_is_outside_horizon(
    tmp_path: Path,
):
    schedule = _schedule("p0", [{"lunch": [(750, 45)]}])
    assert plot_person_day_gantt(schedule, _dt.date(2099, 1, 1), tmp_path) is None


def test_plot_person_day_gantt_returns_none_when_day_has_no_events(tmp_path: Path):
    schedule = _schedule("p0", [{}])
    assert plot_person_day_gantt(schedule, _START, tmp_path) is None


def test_plot_person_day_gantt_handles_multiple_episodes_of_same_event(
    tmp_path: Path,
):
    schedule = _schedule("p0", [{"smoking": [(960, 10), (1000, 10), (1040, 10)]}])
    target = plot_person_day_gantt(schedule, _START, tmp_path)
    assert target is not None
    assert target.exists()


# -------------------------------------------------------------------------------------
# --------------------------- plot_persona_heatmap ------------------------------------
# -------------------------------------------------------------------------------------


def _heatmap_schedules(num_days: int = 30) -> list[PersonSchedule]:
    """Build two persons with daily smoking that ramps from 1 to ~3 episodes."""
    persons: list[PersonSchedule] = []
    for pid in ("d0", "d1"):
        days_payload: list[dict[str, list[tuple[int, int]]]] = []
        for d in range(num_days):
            count = 1 + d // 10
            days_payload.append({"smoking": [(960 + i * 20, 10) for i in range(count)]})
        persons.append(_schedule(pid, days_payload, persona_id="d_smoker"))
    return persons


def test_plot_persona_heatmap_writes_a_png(tmp_path: Path):
    schedules = _heatmap_schedules()
    target = plot_persona_heatmap("d_smoker", schedules, tmp_path, event_name="smoking")
    assert target is not None
    assert target.name == "heatmap_d_smoker_smoking_count_mean.png"
    assert target.exists()


def test_plot_persona_heatmap_supports_duration_sum(tmp_path: Path):
    schedules = _heatmap_schedules()
    target = plot_persona_heatmap(
        "d_smoker",
        schedules,
        tmp_path,
        event_name="smoking",
        metric="duration",
        aggregate="sum",
    )
    assert target is not None
    assert target.name == "heatmap_d_smoker_smoking_duration_sum.png"


def test_plot_persona_heatmap_falls_back_to_all_events(tmp_path: Path):
    """No `event_name` -> filename uses 'all' and the chart aggregates
    every event type."""
    schedules = _heatmap_schedules()
    target = plot_persona_heatmap("d_smoker", schedules, tmp_path)
    assert target is not None
    assert target.name == "heatmap_d_smoker_all_count_mean.png"


def test_plot_persona_heatmap_returns_none_for_empty_population(tmp_path: Path):
    assert plot_persona_heatmap("d_smoker", [], tmp_path) is None


# -------------------------------------------------------------------------------------
# ------------------------------- render_all_charts -----------------------------------
# -------------------------------------------------------------------------------------


def test_render_all_charts_writes_lines_heatmap_and_gantt_per_persona(tmp_path: Path):
    schedules = [
        *_heatmap_schedules(),
        _schedule("a0", [{"lunch": [(750, 45)]} for _ in range(7)], persona_id="alice"),
    ]
    artifacts = render_all_charts(schedules, out_dir=tmp_path)
    assert isinstance(artifacts, ChartArtifacts)
    assert set(artifacts.lines.keys()) == {"d_smoker", "alice"}
    assert set(artifacts.heatmaps.keys()) == {"d_smoker", "alice"}
    # One sample Gantt per persona (first person, first day with events).
    assert len(artifacts.gantts) == 2
    for path in artifacts.all_paths:
        assert path.exists()


def test_render_all_charts_with_empty_population_returns_empty_artifacts(
    tmp_path: Path,
):
    artifacts = render_all_charts([], out_dir=tmp_path)
    assert artifacts.lines == {}
    assert artifacts.heatmaps == {}
    assert artifacts.gantts == []
    assert artifacts.all_paths == []


def test_render_all_charts_skips_lines_and_gantt_for_persona_with_no_events(
    tmp_path: Path,
):
    """A persona whose only person has every day empty produces no
    line chart and no Gantt (no event types to plot, no day with
    events to anchor the Gantt). The heatmap is still rendered: every
    day is on the calendar, just with a value of zero, which is a
    meaningful "nothing happened" view."""
    schedules = [_schedule("p0", [{}, {}], persona_id="ghost")]
    artifacts = render_all_charts(schedules, out_dir=tmp_path)
    assert artifacts.lines == {}
    assert artifacts.gantts == []
    assert set(artifacts.heatmaps.keys()) == {"ghost"}


def test_render_all_charts_skips_heatmap_when_persona_has_zero_day_schedule(
    tmp_path: Path,
):
    """A persona whose schedules have an empty `days` list returns no
    heatmap (the daily aggregate is empty), no line chart, and no
    Gantt. The persona stays in the grouping but contributes zero
    artifacts."""
    schedule = PersonSchedule(
        person_id="p0", persona_id="zero_days", person_seed=0, days=[]
    )
    artifacts = render_all_charts([schedule], out_dir=tmp_path)
    assert artifacts.lines == {}
    assert artifacts.heatmaps == {}
    assert artifacts.gantts == []


def test_render_all_charts_threads_heatmap_options_through(tmp_path: Path):
    """`heatmap_event` / `heatmap_metric` / `heatmap_aggregate` must
    reach `plot_persona_heatmap` so the filename reflects the
    requested view."""
    schedules = _heatmap_schedules()
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        heatmap_event="smoking",
        heatmap_metric="duration",
        heatmap_aggregate="sum",
    )
    [heat] = artifacts.heatmaps.values()
    assert heat.name == "heatmap_d_smoker_smoking_duration_sum.png"


# -------------------------------------------------------------------------------------
# ----------------------------- kinds + calendar overlay ------------------------------
# -------------------------------------------------------------------------------------


def test_render_all_charts_kinds_filter_skips_unrequested_families(tmp_path: Path):
    schedules = _heatmap_schedules()
    artifacts = render_all_charts(schedules, out_dir=tmp_path, kinds=["heatmap"])
    assert artifacts.lines == {}
    assert artifacts.gantts == []
    assert set(artifacts.heatmaps.keys()) == {"d_smoker"}


def test_render_all_charts_unknown_kind_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown chart kinds"):
        render_all_charts([], out_dir=tmp_path, kinds=["bogus"])


def test_render_all_charts_writes_per_member_context_heatmaps_and_lines(tmp_path: Path):
    """Context kinds render per-member heatmaps and cohort lines from episodes."""
    from src.scripts.persona.context.schema import ContextEpisode

    days = [_day(i, {}, start_date=_START) for i in range(7)]
    episodes = [
        ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=_START + _dt.timedelta(days=i),
            start_minutes=480,
            end_minutes=540,
        )
        for i in range(7)
    ]
    schedule = PersonSchedule(
        person_id="p0",
        persona_id="alice",
        person_seed=1,
        days=days,
        contexts=episodes,
    )
    artifacts = render_all_charts(
        [schedule], out_dir=tmp_path, kinds=["context-heatmap", "context-lines"]
    )
    heatmaps = artifacts.per_member_context_heatmaps
    assert "happy" in heatmaps["alice"]["mood_emotion"]
    assert "happy" in artifacts.per_member_context_lines["mood_emotion"]
    for path in artifacts.all_paths:
        assert path.exists()


def test_render_all_charts_context_skips_declared_member_with_no_data(tmp_path: Path):
    """A config-declared member with no day data returns no heatmap or line."""
    from types import SimpleNamespace

    schedule = PersonSchedule(
        person_id="p0", persona_id="alice", person_seed=1, days=[]
    )
    member_stub = SimpleNamespace(
        total_event_episodes=SimpleNamespace(min=0, max=2),
        total_event_duration=SimpleNamespace(min=0, max=60, unit="minutes"),
    )
    cat = SimpleNamespace(members={"happy": member_stub})
    persona_config = SimpleNamespace(
        personas=[SimpleNamespace(id="alice", contexts={"mood_emotion": cat})]
    )
    artifacts = render_all_charts(
        [schedule],
        out_dir=tmp_path,
        kinds=["context-heatmap", "context-lines"],
        persona_config=persona_config,
    )
    # Both plots return None for the empty member, so nothing is recorded.
    assert artifacts.per_member_context_heatmaps == {}
    assert artifacts.per_member_context_lines == {}


def test_render_all_charts_calendar_kind_writes_weekly_pngs(tmp_path: Path):
    """Selecting the `calendar` kind should call the weekly_calendar
    module and return populated `calendars` entries keyed by person."""
    schedules = [
        _schedule("p0", [{"lunch": [(720, 45)]} for _ in range(7)], persona_id="alice")
    ]
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["calendar"],
        weekly_calendar_dpi=120,
    )
    assert "p0" in artifacts.calendars
    assert artifacts.calendars["p0"]
    for path in artifacts.calendars["p0"]:
        assert path.exists()
    # `all_paths` must also surface the calendar PNGs.
    cal_paths = set(artifacts.calendars["p0"])
    assert cal_paths.issubset(set(artifacts.all_paths))


def test_render_all_charts_calendar_kind_skipped_when_no_schedules(tmp_path: Path):
    artifacts = render_all_charts([], out_dir=tmp_path, kinds=["calendar"])
    assert artifacts.calendars == {}


def test_render_all_charts_calendar_kind_overlays_augmented_tasks(tmp_path: Path):
    """When `augmented_by_person` is provided, the calendar render
    receives the augmented overlay for matching persons."""
    schedules = [_schedule("p0", [{"lunch": [(720, 45)]}], persona_id="alice")]
    # Mock the underlying renderer to capture its inputs.
    captured: dict = {}
    from src.scripts.persona.analytics import weekly_calendar as wc

    real_render = wc.render_weekly_calendars

    def spy(scheds, aug_by_person, out_dir, **kw):  # type: ignore[no-untyped-def]
        captured["aug"] = aug_by_person
        captured["dpi"] = kw.get("dpi")
        captured["title_prefix"] = kw.get("title_prefix")
        return real_render(scheds, aug_by_person, out_dir, **kw)

    import unittest.mock

    with unittest.mock.patch.object(wc, "render_weekly_calendars", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["calendar"],
            augmented_by_person={"p0": []},
            weekly_calendar_dpi=200,
            weekly_calendar_title_prefix="my-prefix",
        )
    assert captured["aug"] == {"p0": []}
    assert captured["dpi"] == 200
    assert captured["title_prefix"] == "my-prefix"


# -------------------------------------------------------------------------------------
# --------------------------- variable_heatmap_events --------------------------------
# -------------------------------------------------------------------------------------


def _event_def(
    name: str,
    *,
    category: str = "test",
    temporal_patterns: list[TemporalPattern] | None = None,
) -> EventDefinition:
    """Minimal `EventDefinition` for the variability tests.

    The numeric fields are required by the schema but irrelevant to the
    `variable_heatmap_events` selection rule, so they are pinned to a
    safe default.
    """
    return EventDefinition(
        name=name,
        category=category,
        per_event_duration=DurationRange(min=10, max=10, unit="minutes"),
        total_event_duration=TotalDuration(scale="day", min=10, max=10, unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=1, max=1),
        temporal_patterns=temporal_patterns or [],
    )


def _persona(
    persona_id: str,
    *,
    overrides: dict[str, EventOverride] | None = None,
) -> Persona:
    """Minimal `Persona` fixture; one stage just to satisfy the schema."""
    return Persona(
        id=persona_id,
        instances=1,
        occupation_status="parttime",
        stages=[PersonaEventStage(name="placeholder", days=["Mon"])],
        event_overrides=overrides or {},
    )


def _event_config(events: dict[str, EventDefinition]) -> EventConfig:
    return EventConfig(
        categories={"test": Category(name="test", events=events)},
    )


def test_variable_heatmap_events_excludes_event_with_only_fix_pattern():
    """An event whose catalog `temporal_patterns` is exclusively `mode: fix`
    is dropped because its daily count would be a flat bar."""
    cfg = _event_config(
        {"sleep": _event_def("sleep", temporal_patterns=[TemporalPattern(mode="fix")])}
    )
    persona_cfg = PersonaConfig(personas=[_persona("alice")])
    out = variable_heatmap_events(cfg, persona_cfg)
    assert out == {"alice": set()}


def test_variable_heatmap_events_includes_event_with_no_temporal_patterns():
    """No `temporal_patterns` at all means no window pin; the persona
    is free to schedule it day-to-day, so the heatmap is meaningful."""
    cfg = _event_config({"office_work": _event_def("office_work")})
    persona_cfg = PersonaConfig(personas=[_persona("alice")])
    assert variable_heatmap_events(cfg, persona_cfg) == {"alice": {"office_work"}}


def test_variable_heatmap_events_includes_event_with_seasonality_in_catalog():
    cfg = _event_config(
        {
            "walking": _event_def(
                "walking",
                temporal_patterns=[TemporalPattern(mode="seasonality")],
            )
        }
    )
    persona_cfg = PersonaConfig(personas=[_persona("alice")])
    assert variable_heatmap_events(cfg, persona_cfg) == {"alice": {"walking"}}


def test_variable_heatmap_events_includes_event_with_trend_in_catalog():
    cfg = _event_config(
        {
            "walking": _event_def(
                "walking",
                temporal_patterns=[TemporalPattern(mode="trend")],
            )
        }
    )
    persona_cfg = PersonaConfig(personas=[_persona("alice")])
    assert variable_heatmap_events(cfg, persona_cfg) == {"alice": {"walking"}}


def test_variable_heatmap_events_includes_event_when_persona_override_adds_seasonality():
    """Catalog pins the event to a fixed window, but the persona injects
    a `mode: seasonality` override; the heatmap will then show that
    persona's day-to-day variation, so include it for that persona."""
    cfg = _event_config(
        {
            "walking": _event_def(
                "walking", temporal_patterns=[TemporalPattern(mode="fix")]
            )
        }
    )
    persona_cfg = PersonaConfig(
        personas=[
            _persona(
                "alice",
                overrides={
                    "walking": EventOverride(
                        temporal_patterns=[TemporalPattern(mode="seasonality")]
                    )
                },
            ),
            _persona("bob"),
        ]
    )
    out = variable_heatmap_events(cfg, persona_cfg)
    assert out["alice"] == {"walking"}
    # `bob` has no override and the catalog pin excludes the event.
    assert out["bob"] == set()


def test_variable_heatmap_events_includes_event_when_persona_override_adds_trend():
    cfg = _event_config(
        {
            "walking": _event_def(
                "walking", temporal_patterns=[TemporalPattern(mode="fix")]
            )
        }
    )
    persona_cfg = PersonaConfig(
        personas=[
            _persona(
                "alice",
                overrides={
                    "walking": EventOverride(
                        temporal_patterns=[TemporalPattern(mode="trend")]
                    )
                },
            )
        ]
    )
    assert variable_heatmap_events(cfg, persona_cfg) == {"alice": {"walking"}}


def test_variable_heatmap_events_override_with_none_temporal_patterns_is_skipped():
    """An override whose `temporal_patterns` field stays `None` (default)
    does not promote the event; only explicit seasonality / trend
    overrides add events back into the variable set."""
    cfg = _event_config(
        {"sleep": _event_def("sleep", temporal_patterns=[TemporalPattern(mode="fix")])}
    )
    persona_cfg = PersonaConfig(
        personas=[
            _persona(
                "alice",
                overrides={"sleep": EventOverride()},
            )
        ]
    )
    assert variable_heatmap_events(cfg, persona_cfg) == {"alice": set()}


def test_variable_heatmap_events_persona_override_with_only_fix_does_not_promote():
    """A persona override that only restates `mode: fix` does not promote the event."""
    cfg = _event_config(
        {"sleep": _event_def("sleep", temporal_patterns=[TemporalPattern(mode="fix")])}
    )
    persona_cfg = PersonaConfig(
        personas=[
            _persona(
                "alice",
                overrides={
                    "sleep": EventOverride(
                        temporal_patterns=[TemporalPattern(mode="fix")]
                    )
                },
            )
        ]
    )
    assert variable_heatmap_events(cfg, persona_cfg) == {"alice": set()}


def test_variable_heatmap_events_mixed_catalog_returns_only_variable_events():
    """Realistic mix: fixed meals/sleep + variable sports + override-promoted
    walking. Only the variable events come out for each persona."""
    cfg = _event_config(
        {
            "sleep": _event_def(
                "sleep", temporal_patterns=[TemporalPattern(mode="fix")]
            ),
            "lunch": _event_def(
                "lunch", temporal_patterns=[TemporalPattern(mode="fix")]
            ),
            "office_work": _event_def("office_work"),
            "walking": _event_def(
                "walking", temporal_patterns=[TemporalPattern(mode="fix")]
            ),
            "yoga": _event_def("yoga"),
        }
    )
    persona_cfg = PersonaConfig(
        personas=[
            _persona(
                "alice",
                overrides={
                    "walking": EventOverride(
                        temporal_patterns=[
                            TemporalPattern(mode="seasonality"),
                            TemporalPattern(mode="trend"),
                        ]
                    )
                },
            ),
            _persona("bob"),
        ]
    )
    out = variable_heatmap_events(cfg, persona_cfg)
    assert out["alice"] == {"office_work", "walking", "yoga"}
    # `bob` lacks the walking override, so walking falls back to the
    # catalog `mode: fix` exclusion.
    assert out["bob"] == {"office_work", "yoga"}


def test_variable_heatmap_events_returns_one_entry_per_persona():
    """Even personas without any variable event keep an entry (with an
    empty set), so callers can iterate `out.items()` deterministically."""
    cfg = _event_config(
        {"sleep": _event_def("sleep", temporal_patterns=[TemporalPattern(mode="fix")])}
    )
    persona_cfg = PersonaConfig(
        personas=[_persona("alice"), _persona("bob"), _persona("carol")]
    )
    out = variable_heatmap_events(cfg, persona_cfg)
    assert set(out.keys()) == {"alice", "bob", "carol"}
    assert all(v == set() for v in out.values())


# -------------------------------------------------------------------------------------
# ------------------- render_all_charts: heatmap_events_by_persona --------------------
# -------------------------------------------------------------------------------------


def test_render_all_charts_per_event_heatmaps_one_png_per_variable_event(
    tmp_path: Path,
):
    """When `heatmap_events_by_persona` is supplied, the heatmap branch
    emits one PNG per (persona, event) and the legacy aggregate field
    stays empty."""
    schedules = _heatmap_schedules()
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona={"d_smoker": {"smoking"}},
    )
    assert artifacts.heatmaps == {}
    assert set(artifacts.per_event_heatmaps.keys()) == {"d_smoker"}
    assert set(artifacts.per_event_heatmaps["d_smoker"].keys()) == {"smoking"}
    metric_paths = artifacts.per_event_heatmaps["d_smoker"]["smoking"]
    assert set(metric_paths.keys()) == {"count", "duration"}
    assert metric_paths["count"].name == "heatmap_d_smoker_smoking_count_mean.png"
    assert metric_paths["duration"].name == "heatmap_d_smoker_smoking_duration_mean.png"
    for path in metric_paths.values():
        assert path.exists()


def test_render_all_charts_per_event_heatmaps_skips_persona_with_empty_set(
    tmp_path: Path,
):
    """A persona present in the dict with an empty set produces no
    heatmap (its events are all catalog-fixed and would be flat)."""
    schedules = _heatmap_schedules()
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona={"d_smoker": set()},
    )
    assert artifacts.heatmaps == {}
    assert artifacts.per_event_heatmaps == {}


def test_render_all_charts_per_event_heatmaps_skips_persona_missing_from_dict(
    tmp_path: Path,
):
    """A persona absent from the dict is silently skipped; the dict is
    the source of truth for which personas get heatmaps in the new mode."""
    schedules = [
        *_heatmap_schedules(),
        _schedule("a0", [{"lunch": [(750, 45)]} for _ in range(7)], persona_id="alice"),
    ]
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona={"d_smoker": {"smoking"}},
    )
    assert set(artifacts.per_event_heatmaps.keys()) == {"d_smoker"}
    assert "alice" not in artifacts.per_event_heatmaps


def test_render_all_charts_per_event_heatmaps_renders_multiple_events(
    tmp_path: Path,
):
    """Each variable event for one persona produces BOTH a count and a
    duration PNG."""
    schedules = [
        _schedule(
            "p0",
            [{"walking": [(420, 30)], "yoga": [(600, 60)]} for _ in range(7)],
            persona_id="alice",
        )
    ]
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona={"alice": {"walking", "yoga"}},
    )
    assert set(artifacts.per_event_heatmaps["alice"].keys()) == {"walking", "yoga"}
    for event_name, metric_paths in artifacts.per_event_heatmaps["alice"].items():
        assert set(metric_paths.keys()) == {"count", "duration"}
        for metric, path in metric_paths.items():
            assert path.name == f"heatmap_alice_{event_name}_{metric}_mean.png"
            assert path.exists()


def test_render_all_charts_per_event_heatmaps_aggregate_threads_through(
    tmp_path: Path,
):
    """`heatmap_aggregate` flows through to per-event PNGs so a
    `sum` aggregate names the file accordingly. The per-event mode
    always emits both metrics regardless of `heatmap_metric`."""
    schedules = _heatmap_schedules()
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_aggregate="sum",
        heatmap_events_by_persona={"d_smoker": {"smoking"}},
    )
    metric_paths = artifacts.per_event_heatmaps["d_smoker"]["smoking"]
    assert metric_paths["count"].name == "heatmap_d_smoker_smoking_count_sum.png"
    assert metric_paths["duration"].name == "heatmap_d_smoker_smoking_duration_sum.png"


def test_render_all_charts_per_event_heatmaps_paths_in_all_paths(tmp_path: Path):
    """`ChartArtifacts.all_paths` must surface every per-event heatmap
    PNG (count + duration) so downstream consumers (CLI listings,
    manifest writers) see them."""
    schedules = _heatmap_schedules()
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona={"d_smoker": {"smoking"}},
    )
    paths = set(artifacts.all_paths)
    for metric_path in artifacts.per_event_heatmaps["d_smoker"]["smoking"].values():
        assert metric_path in paths


def test_render_all_charts_per_event_skips_path_when_plot_returns_none(
    tmp_path: Path,
):
    """When `plot_persona_heatmap` returns `None` for one metric (e.g.
    the schedule has zero days), the loop does not record that metric
    in `per_event_heatmaps`."""
    schedules = [
        PersonSchedule(
            person_id="p0",
            persona_id="alice",
            person_seed=0,
            days=[],
        )
    ]
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona={"alice": {"walking"}},
    )
    # Zero days means daily_event_aggregate is empty for every metric,
    # so the per-event dict for alice never gets created.
    assert artifacts.per_event_heatmaps == {}


def test_render_all_charts_legacy_heatmap_field_unchanged_when_param_absent(
    tmp_path: Path,
):
    """Backwards compat: omitting `heatmap_events_by_persona` keeps the
    legacy single-aggregate-per-persona behaviour and leaves
    `per_event_heatmaps` empty. Existing callers see no shape change."""
    schedules = _heatmap_schedules()
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
    )
    assert set(artifacts.heatmaps.keys()) == {"d_smoker"}
    assert artifacts.per_event_heatmaps == {}


# -------------------------------------------------------------------------------------
# --------------------------- heatmap_vmax_for_event ----------------------------------
# -------------------------------------------------------------------------------------


def _vmax_event_def(
    name: str,
    *,
    episodes_max: int,
    duration_max: int,
    duration_unit: str = "minutes",
) -> EventDefinition:
    """Catalog `EventDefinition` for the vmax-resolution tests."""
    duration_min = 1 if duration_unit == "hours" else 10
    return EventDefinition(
        name=name,
        category="test",
        per_event_duration=DurationRange(
            min=duration_min, max=duration_max, unit=duration_unit
        ),
        total_event_duration=TotalDuration(
            scale="day", min=duration_min, max=duration_max, unit=duration_unit
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=episodes_max),
        temporal_patterns=[],
    )


def test_heatmap_vmax_for_event_count_reads_episodes_max():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=3, duration_max=90)}
    )
    assert heatmap_vmax_for_event(cfg, "walking", "count") == 3.0


def test_heatmap_vmax_for_event_duration_reads_minutes():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=90)}
    )
    assert heatmap_vmax_for_event(cfg, "walking", "duration") == 90.0


def test_heatmap_vmax_for_event_duration_hours_converted_to_minutes():
    cfg = _event_config(
        {
            "office_work": _vmax_event_def(
                "office_work", episodes_max=4, duration_max=4, duration_unit="hours"
            )
        }
    )
    assert heatmap_vmax_for_event(cfg, "office_work", "duration") == 240.0


def test_heatmap_vmax_for_event_returns_none_when_event_config_missing():
    assert heatmap_vmax_for_event(None, "walking", "count") is None


def test_heatmap_vmax_for_event_returns_none_when_event_name_missing():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=60)}
    )
    assert heatmap_vmax_for_event(cfg, None, "count") is None


def test_heatmap_vmax_for_event_returns_none_for_unknown_event():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=60)}
    )
    assert heatmap_vmax_for_event(cfg, "gym", "count") is None


def test_heatmap_vmax_for_event_returns_none_for_unknown_metric():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=60)}
    )
    assert heatmap_vmax_for_event(cfg, "walking", "intensity") is None


# -------------------------------------------------------------------------------------
# --------------------------- heatmap_vmin_for_event ----------------------------------
# -------------------------------------------------------------------------------------


def test_heatmap_vmin_for_event_count_reads_episodes_min():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=3, duration_max=90)}
    )
    # `_vmax_event_def` pins min to 0 for episode counts.
    assert heatmap_vmin_for_event(cfg, "walking", "count") == 0.0


def test_heatmap_vmin_for_event_duration_reads_minutes():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=90)}
    )
    # `_vmax_event_def` pins min to 10 minutes when unit=minutes.
    assert heatmap_vmin_for_event(cfg, "walking", "duration") == 10.0


def test_heatmap_vmin_for_event_duration_hours_converted_to_minutes():
    cfg = _event_config(
        {
            "office_work": _vmax_event_def(
                "office_work", episodes_max=4, duration_max=4, duration_unit="hours"
            )
        }
    )
    # min=1 hour -> 60 minutes.
    assert heatmap_vmin_for_event(cfg, "office_work", "duration") == 60.0


def test_heatmap_vmin_for_event_returns_none_when_event_config_missing():
    assert heatmap_vmin_for_event(None, "walking", "count") is None


def test_heatmap_vmin_for_event_returns_none_when_event_name_missing():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=60)}
    )
    assert heatmap_vmin_for_event(cfg, None, "count") is None


def test_heatmap_vmin_for_event_returns_none_for_unknown_event():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=60)}
    )
    assert heatmap_vmin_for_event(cfg, "gym", "count") is None


def test_heatmap_vmin_for_event_returns_none_for_unknown_metric():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=1, duration_max=60)}
    )
    assert heatmap_vmin_for_event(cfg, "walking", "intensity") is None


# -------------------------------------------------------------------------------------
# --------------------- plot_persona_heatmap with vmin / vmax -------------------------
# -------------------------------------------------------------------------------------


def test_plot_persona_heatmap_accepts_vmin_vmax_and_writes_png(tmp_path: Path):
    """`vmin` + `vmax` overrides produce the PNG and anchor the colorbar
    scale at the supplied range."""
    schedules = _heatmap_schedules()
    target = plot_persona_heatmap(
        "d_smoker",
        schedules,
        tmp_path,
        event_name="smoking",
        metric="count",
        vmin=1.0,
        vmax=12.0,
    )
    assert target is not None
    assert target.exists()


def test_plot_persona_heatmap_invalid_range_falls_back(tmp_path: Path):
    """A vmin >= vmax range is degenerate; the colorbar falls back to
    data-driven scaling rather than crashing matplotlib."""
    schedules = _heatmap_schedules()
    target = plot_persona_heatmap(
        "d_smoker",
        schedules,
        tmp_path,
        event_name="smoking",
        vmin=5.0,
        vmax=5.0,
    )
    assert target is not None


def test_plot_persona_heatmap_forwards_vmin_and_vmax_to_calmap(tmp_path: Path):
    """When both `vmin` and `vmax` are set and form a valid range,
    they reach calmap.yearplot via the keyword arguments."""
    schedules = _heatmap_schedules()
    captured: dict = {}

    from src.scripts.persona.analytics import charts as charts_mod

    real_yearplot = charts_mod.calmap.yearplot

    def spy(series, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return real_yearplot(series, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod.calmap, "yearplot", spy):
        plot_persona_heatmap(
            "d_smoker",
            schedules,
            tmp_path,
            event_name="smoking",
            vmin=2.0,
            vmax=15.0,
        )
    assert captured.get("vmin") == 2.0
    assert captured.get("vmax") == 15.0


def test_plot_persona_heatmap_degenerate_range_falls_back_to_data_scale(tmp_path: Path):
    """When vmin == vmax (e.g. sleep min=max=1), do NOT pass vmin/vmax
    to calmap; fall back to data-driven scaling."""
    schedules = _heatmap_schedules()
    captured: dict = {}

    from src.scripts.persona.analytics import charts as charts_mod

    real_yearplot = charts_mod.calmap.yearplot

    def spy(series, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return real_yearplot(series, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod.calmap, "yearplot", spy):
        plot_persona_heatmap(
            "d_smoker",
            schedules,
            tmp_path,
            event_name="smoking",
            vmin=1.0,
            vmax=1.0,
        )
    assert "vmin" not in captured
    assert "vmax" not in captured


def test_plot_persona_heatmap_only_vmax_set_falls_back(tmp_path: Path):
    """vmax alone (no vmin) does not anchor the scale; the colorbar
    stays data-driven."""
    schedules = _heatmap_schedules()
    captured: dict = {}

    from src.scripts.persona.analytics import charts as charts_mod

    real_yearplot = charts_mod.calmap.yearplot

    def spy(series, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return real_yearplot(series, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod.calmap, "yearplot", spy):
        plot_persona_heatmap(
            "d_smoker",
            schedules,
            tmp_path,
            event_name="smoking",
            vmax=15.0,
        )
    assert "vmin" not in captured
    assert "vmax" not in captured


def test_plot_persona_heatmap_without_vmax_omits_calmap_scale_kwargs(tmp_path: Path):
    schedules = _heatmap_schedules()
    captured: dict = {}

    from src.scripts.persona.analytics import charts as charts_mod

    real_yearplot = charts_mod.calmap.yearplot

    def spy(series, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return real_yearplot(series, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod.calmap, "yearplot", spy):
        plot_persona_heatmap(
            "d_smoker",
            schedules,
            tmp_path,
            event_name="smoking",
        )
    assert "vmin" not in captured
    assert "vmax" not in captured


# -------------------------------------------------------------------------------------
# ----------------- render_all_charts with event_config (vmax wired) -----------------
# -------------------------------------------------------------------------------------


def test_render_all_charts_per_event_uses_config_vmin_and_vmax(tmp_path: Path):
    """When `event_config` is supplied, each per-event heatmap call
    receives both `vmin` and `vmax` derived from the matching event's
    catalog min and max."""
    schedules = [
        _schedule(
            "p0",
            [{"walking": [(420, 30)]} for _ in range(7)],
            persona_id="alice",
        )
    ]
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=3, duration_max=240)}
    )
    captured: list[tuple[str, float | None, float | None]] = []

    from src.scripts.persona.analytics import charts as charts_mod

    real_plot = charts_mod.plot_persona_heatmap

    def spy(persona_id, schedules, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(
            (
                kwargs.get("metric", "count"),
                kwargs.get("vmin"),
                kwargs.get("vmax"),
            )
        )
        return real_plot(persona_id, schedules, out_dir, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod, "plot_persona_heatmap", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["heatmap"],
            heatmap_events_by_persona={"alice": {"walking"}},
            event_config=cfg,
        )
    # vmax for count comes from total_event_episodes.max (3)
    # vmax for duration comes from total_event_duration.max in min (240)
    # vmin for count comes from total_event_episodes.min (0)
    # vmin for duration comes from total_event_duration.min in min (10)
    assert ("count", 0.0, 3.0) in captured
    assert ("duration", 10.0, 240.0) in captured


def test_render_all_charts_per_event_without_event_config_passes_none_bounds(
    tmp_path: Path,
):
    """No `event_config` means no anchored bounds; the plot falls back
    to data-driven scaling for both vmin and vmax."""
    schedules = _heatmap_schedules()
    captured: list[tuple[float | None, float | None]] = []

    from src.scripts.persona.analytics import charts as charts_mod

    real_plot = charts_mod.plot_persona_heatmap

    def spy(persona_id, schedules, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((kwargs.get("vmin"), kwargs.get("vmax")))
        return real_plot(persona_id, schedules, out_dir, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod, "plot_persona_heatmap", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["heatmap"],
            heatmap_events_by_persona={"d_smoker": {"smoking"}},
        )
    assert captured == [(None, None), (None, None)]


def test_render_all_charts_per_event_persona_override_wins_over_catalog(
    tmp_path: Path,
):
    """When the persona override tightens the per-day bounds, the
    heatmap colorbar must reflect the override, not the loose catalog
    default."""
    schedules = [
        _schedule(
            "p0",
            [{"meeting": [(420, 30)]} for _ in range(7)],
            persona_id="alice",
        )
    ]
    cfg = _event_config(
        # Loose catalog: 0-8 episodes, 10-240 min.
        {"meeting": _vmax_event_def("meeting", episodes_max=8, duration_max=240)}
    )
    persona_cfg = PersonaConfig(
        personas=[
            Persona(
                id="alice",
                instances=1,
                occupation_status="parttime",
                stages=[PersonaEventStage(name="meeting", days=["Mon"])],
                event_overrides={
                    "meeting": EventOverride(
                        total_event_episodes=EpisodeRange(scale="day", min=1, max=4),
                        total_event_duration=TotalDuration(
                            scale="day", min=15, max=120, unit="minutes"
                        ),
                    )
                },
            )
        ]
    )
    captured: list[tuple[str, float | None, float | None]] = []

    from src.scripts.persona.analytics import charts as charts_mod

    real_plot = charts_mod.plot_persona_heatmap

    def spy(persona_id, schedules, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(
            (
                kwargs.get("metric", "count"),
                kwargs.get("vmin"),
                kwargs.get("vmax"),
            )
        )
        return real_plot(persona_id, schedules, out_dir, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod, "plot_persona_heatmap", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["heatmap"],
            heatmap_events_by_persona={"alice": {"meeting"}},
            event_config=cfg,
            persona_config=persona_cfg,
        )
    # Override wins: episodes 1-4, duration 15-120.
    assert ("count", 1.0, 4.0) in captured
    assert ("duration", 15.0, 120.0) in captured


def test_render_all_charts_persona_config_missing_persona_falls_back_to_catalog(
    tmp_path: Path,
):
    """A persona_config that lists a different persona id leaves the
    bounds at catalog defaults (the lookup returns no overrides)."""
    schedules = [
        _schedule(
            "p0",
            [{"meeting": [(420, 30)]} for _ in range(7)],
            persona_id="alice",
        )
    ]
    cfg = _event_config(
        {"meeting": _vmax_event_def("meeting", episodes_max=8, duration_max=240)}
    )
    # persona_config has no entry for 'alice'; the lookup walks every
    # persona and returns the empty dict on no match.
    persona_cfg = PersonaConfig(
        personas=[
            Persona(
                id="bob",
                instances=1,
                occupation_status="parttime",
                stages=[PersonaEventStage(name="meeting", days=["Mon"])],
            )
        ]
    )
    captured: list[tuple[str, float | None, float | None]] = []

    from src.scripts.persona.analytics import charts as charts_mod

    real_plot = charts_mod.plot_persona_heatmap

    def spy(persona_id, schedules, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(
            (
                kwargs.get("metric", "count"),
                kwargs.get("vmin"),
                kwargs.get("vmax"),
            )
        )
        return real_plot(persona_id, schedules, out_dir, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod, "plot_persona_heatmap", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["heatmap"],
            heatmap_events_by_persona={"alice": {"meeting"}},
            event_config=cfg,
            persona_config=persona_cfg,
        )
    # Catalog bounds: episodes 0-8, duration 10-240.
    assert ("count", 0.0, 8.0) in captured
    assert ("duration", 10.0, 240.0) in captured


def test_render_all_charts_legacy_heatmap_uses_config_bounds_when_event_pinned(
    tmp_path: Path,
):
    """Legacy aggregate mode also picks up the config vmin + vmax when
    `heatmap_event` is set."""
    schedules = _heatmap_schedules()
    cfg = _event_config(
        {"smoking": _vmax_event_def("smoking", episodes_max=10, duration_max=100)}
    )
    captured: list[tuple[float | None, float | None]] = []

    from src.scripts.persona.analytics import charts as charts_mod

    real_plot = charts_mod.plot_persona_heatmap

    def spy(persona_id, schedules, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((kwargs.get("vmin"), kwargs.get("vmax")))
        return real_plot(persona_id, schedules, out_dir, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod, "plot_persona_heatmap", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["heatmap"],
            heatmap_event="smoking",
            event_config=cfg,
        )
    # vmin = total_event_episodes.min = 0
    # vmax = total_event_episodes.max = 10
    assert captured == [(0.0, 10.0)]


# -------------------------------------------------------------------------------------
# ---------------------------- per_event_persona_daily --------------------------------
# -------------------------------------------------------------------------------------


def test_per_event_persona_daily_count_shape_and_values():
    schedules = [
        _schedule(
            "p0",
            [{"walking": [(420, 30)]}, {"walking": [(420, 30), (600, 20)]}],
        ),
        _schedule("p1", [{"walking": [(420, 30)]}, {}]),
    ]
    matrix = per_event_persona_daily(schedules, "walking", metric="count")
    assert matrix.shape == (2, 2)
    assert matrix.tolist() == [[1.0, 1.0], [2.0, 0.0]]


def test_per_event_persona_daily_duration_sums_minutes():
    schedules = [
        _schedule(
            "p0",
            [{"walking": [(420, 30), (600, 15)]}],
        ),
        _schedule("p1", [{"walking": [(420, 45)]}]),
    ]
    matrix = per_event_persona_daily(schedules, "walking", metric="duration")
    assert matrix.shape == (1, 2)
    assert matrix.tolist() == [[45.0, 45.0]]


def test_per_event_persona_daily_pads_uneven_horizons():
    schedules = [
        _schedule("p0", [{"walking": [(420, 30)]}, {"walking": [(420, 30)]}]),
        _schedule("p1", [{"walking": [(420, 30)]}]),
    ]
    matrix = per_event_persona_daily(schedules, "walking", metric="count")
    assert matrix.shape == (2, 2)
    assert matrix[1, 1] == 0.0


def test_per_event_persona_daily_missing_event_returns_zeros():
    schedules = [_schedule("p0", [{"lunch": [(750, 45)]}])]
    matrix = per_event_persona_daily(schedules, "walking", metric="count")
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0


def test_per_event_persona_daily_empty_population_is_zero_shape():
    matrix = per_event_persona_daily([], "walking", metric="count")
    assert matrix.shape == (0, 0)


def test_per_event_persona_daily_rejects_invalid_metric():
    with pytest.raises(ValueError, match="metric"):
        per_event_persona_daily([], "walking", metric="hours")  # type: ignore[arg-type]


# -------------------------------------------------------------------------------------
# ------------------------------ daily_mean_and_ci ------------------------------------
# -------------------------------------------------------------------------------------


def test_daily_mean_and_ci_computes_mean_and_bounds():
    """Mean and 95% CI half-width = 1.96 * std / sqrt(n) (ddof=1)."""
    values = np.array(
        [
            [1.0, 3.0, 5.0],
            [2.0, 2.0, 2.0],
        ]
    )
    mean, lo, hi = daily_mean_and_ci(values)
    np.testing.assert_allclose(mean, [3.0, 2.0])
    # Day 0: std=2.0, half = 1.96 * 2 / sqrt(3)
    half0 = 1.959963984540054 * 2.0 / np.sqrt(3)
    np.testing.assert_allclose(lo[0], 3.0 - half0)
    np.testing.assert_allclose(hi[0], 3.0 + half0)
    # Day 1: zero variance, CI collapses to the mean.
    np.testing.assert_allclose(lo[1], 2.0)
    np.testing.assert_allclose(hi[1], 2.0)


def test_daily_mean_and_ci_single_person_collapses_ci():
    """With one person the CI collapses to the mean."""
    values = np.array([[5.0], [3.0]])
    mean, lo, hi = daily_mean_and_ci(values)
    np.testing.assert_allclose(mean, [5.0, 3.0])
    np.testing.assert_allclose(lo, mean)
    np.testing.assert_allclose(hi, mean)


def test_daily_mean_and_ci_empty_horizon_returns_empty_arrays():
    mean, lo, hi = daily_mean_and_ci(np.zeros((0, 3)))
    assert mean.shape == (0,)
    assert lo.shape == (0,)
    assert hi.shape == (0,)


def test_daily_mean_and_ci_zero_persons_returns_zero_mean():
    """An empty person axis falls back to zeros instead of NaN."""
    mean, lo, hi = daily_mean_and_ci(np.zeros((3, 0)))
    np.testing.assert_array_equal(mean, np.zeros(3))
    np.testing.assert_array_equal(lo, np.zeros(3))
    np.testing.assert_array_equal(hi, np.zeros(3))


def test_daily_mean_and_ci_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        daily_mean_and_ci(np.zeros(5))


# -------------------------------------------------------------------------------------
# --------------------------- duration_unit_for_event ---------------------------------
# -------------------------------------------------------------------------------------


def test_duration_unit_for_event_returns_catalog_unit():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=5, duration_max=90)}
    )
    assert duration_unit_for_event(cfg, "walking") == "minutes"


def test_duration_unit_for_event_returns_hours_when_catalog_uses_hours():
    cfg = _event_config(
        {
            "sleep": _vmax_event_def(
                "sleep", episodes_max=1, duration_max=8, duration_unit="hours"
            )
        }
    )
    assert duration_unit_for_event(cfg, "sleep") == "hours"


def test_duration_unit_for_event_override_wins_over_catalog():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=5, duration_max=90)}
    )
    override = EventOverride(
        total_event_duration=TotalDuration(scale="day", min=1, max=4, unit="hours"),
    )
    assert duration_unit_for_event(cfg, "walking", override) == "hours"


def test_duration_unit_for_event_falls_back_to_minutes_when_no_config():
    assert duration_unit_for_event(None, "walking") == "minutes"


def test_duration_unit_for_event_falls_back_to_minutes_when_event_missing():
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=5, duration_max=90)}
    )
    assert duration_unit_for_event(cfg, "running") == "minutes"


def test_duration_unit_for_event_override_without_duration_field_falls_back():
    """An override with `total_event_duration=None` falls through to the catalog."""
    cfg = _event_config(
        {
            "sleep": _vmax_event_def(
                "sleep", episodes_max=1, duration_max=8, duration_unit="hours"
            )
        }
    )
    override = EventOverride()
    assert duration_unit_for_event(cfg, "sleep", override) == "hours"


# -------------------------------------------------------------------------------------
# ------------------------- plot_per_event_persona_lines ------------------------------
# -------------------------------------------------------------------------------------


def _lines_schedules(persona_id: str, person_ids: list[str]) -> list[PersonSchedule]:
    """Two-week walking schedule with mild between-person variation."""
    out: list[PersonSchedule] = []
    for offset, pid in enumerate(person_ids):
        days = []
        for d in range(14):
            duration = 30 + offset * 5 + (d % 3) * 2
            days.append({"walking": [(420, duration)]})
        out.append(_schedule(pid, days, persona_id=persona_id))
    return out


def test_plot_per_event_persona_lines_writes_count_png(tmp_path: Path):
    by_persona = {"g_fulltime": _lines_schedules("g_fulltime", ["p0", "p1", "p2"])}
    target = plot_per_event_persona_lines("walking", "count", by_persona, tmp_path)
    assert target is not None
    assert target.name == "lines_walking_count.png"
    assert target.exists()
    assert target.stat().st_size > 0


def test_plot_per_event_persona_lines_writes_duration_png(tmp_path: Path):
    by_persona = {"g_fulltime": _lines_schedules("g_fulltime", ["p0", "p1", "p2"])}
    target = plot_per_event_persona_lines(
        "walking", "duration", by_persona, tmp_path, duration_unit="hours"
    )
    assert target is not None
    assert target.name == "lines_walking_duration.png"


def test_plot_per_event_persona_lines_compares_multiple_personas(tmp_path: Path):
    by_persona = {
        "alice": _lines_schedules("alice", ["a0", "a1"]),
        "bob": _lines_schedules("bob", ["b0", "b1", "b2"]),
    }
    target = plot_per_event_persona_lines("walking", "count", by_persona, tmp_path)
    assert target is not None
    assert target.exists()


def test_plot_per_event_persona_lines_returns_none_when_event_never_logged(
    tmp_path: Path,
):
    """All-zero data across personas returns `None` instead of a flat chart."""
    by_persona = {
        "alice": [_schedule("a0", [{"lunch": [(720, 30)]} for _ in range(7)])]
    }
    target = plot_per_event_persona_lines("walking", "count", by_persona, tmp_path)
    assert target is None


def test_plot_per_event_persona_lines_returns_none_for_empty_input(tmp_path: Path):
    assert plot_per_event_persona_lines("walking", "count", {}, tmp_path) is None


def test_plot_per_event_persona_lines_skips_persona_with_no_schedules(
    tmp_path: Path,
):
    """A persona with an empty schedule list is skipped while others render."""
    by_persona = {
        "ghost": [],
        "alice": _lines_schedules("alice", ["p0", "p1"]),
    }
    target = plot_per_event_persona_lines("walking", "count", by_persona, tmp_path)
    assert target is not None


def test_plot_per_event_persona_lines_returns_none_when_only_empty_schedules(
    tmp_path: Path,
):
    """Every persona is empty so the helper returns `None`."""
    by_persona = {"ghost": []}
    assert (
        plot_per_event_persona_lines("walking", "count", by_persona, tmp_path) is None
    )


def test_plot_per_event_persona_lines_returns_none_when_all_days_empty(
    tmp_path: Path,
):
    """Schedules with no days at all produce no chart."""
    by_persona = {
        "ghost": [
            PersonSchedule(person_id="p0", persona_id="ghost", person_seed=0, days=[])
        ]
    }
    assert (
        plot_per_event_persona_lines("walking", "count", by_persona, tmp_path) is None
    )


def test_plot_per_event_persona_lines_rejects_invalid_metric(tmp_path: Path):
    by_persona = {"alice": _lines_schedules("alice", ["a0"])}
    with pytest.raises(ValueError, match="metric"):
        plot_per_event_persona_lines(
            "walking", "hours", by_persona, tmp_path  # type: ignore[arg-type]
        )


# -------------------------------------------------------------------------------------
# ----------------- render_all_charts: line_events_by_persona -------------------------
# -------------------------------------------------------------------------------------


def test_render_all_charts_per_event_lines_emits_count_and_duration_per_event(
    tmp_path: Path,
):
    """Per-event mode emits two PNGs per variable event and leaves `lines` empty."""
    schedules = _lines_schedules("g_fulltime", ["p0", "p1", "p2"])
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["lines"],
        line_events_by_persona={"g_fulltime": {"walking"}},
    )
    assert artifacts.lines == {}
    assert set(artifacts.per_event_lines.keys()) == {"walking"}
    assert set(artifacts.per_event_lines["walking"].keys()) == {"count", "duration"}
    for path in artifacts.per_event_lines["walking"].values():
        assert path.exists()


def test_render_all_charts_per_event_lines_compares_personas_on_same_chart(
    tmp_path: Path,
):
    """Two personas with the same event share one chart per metric."""
    schedules = [
        *_lines_schedules("alice", ["a0", "a1"]),
        *_lines_schedules("bob", ["b0", "b1", "b2"]),
    ]
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["lines"],
        line_events_by_persona={"alice": {"walking"}, "bob": {"walking"}},
    )
    assert set(artifacts.per_event_lines["walking"].keys()) == {"count", "duration"}
    # Exactly one PNG per metric (not one per persona).
    assert len(artifacts.per_event_lines["walking"]) == 2


def test_render_all_charts_per_event_lines_skips_event_with_no_persona(
    tmp_path: Path,
):
    """An event whose persona is absent from the schedules is silently skipped."""
    schedules = _lines_schedules("alice", ["a0", "a1"])
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["lines"],
        line_events_by_persona={"alice": set(), "ghost": {"walking"}},
    )
    assert artifacts.per_event_lines == {}


def test_render_all_charts_per_event_lines_paths_in_all_paths(tmp_path: Path):
    schedules = _lines_schedules("g_fulltime", ["p0", "p1"])
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["lines"],
        line_events_by_persona={"g_fulltime": {"walking"}},
    )
    paths = set(artifacts.all_paths)
    for path in artifacts.per_event_lines["walking"].values():
        assert path in paths


def test_render_all_charts_per_event_lines_uses_persona_override_unit(tmp_path: Path):
    """A persona override of `hours` flows through to the plot helper."""
    schedules = _lines_schedules("g_fulltime", ["p0", "p1"])
    cfg = _event_config(
        {"walking": _vmax_event_def("walking", episodes_max=5, duration_max=400)}
    )
    persona_cfg = PersonaConfig(
        personas=[
            _persona(
                "g_fulltime",
                overrides={
                    "walking": EventOverride(
                        total_event_duration=TotalDuration(
                            scale="day", min=1, max=8, unit="hours"
                        ),
                    )
                },
            )
        ]
    )
    captured: list[str] = []
    from src.scripts.persona.analytics import charts as charts_mod

    real_plot = charts_mod.plot_per_event_persona_lines

    def spy(event_name, metric, schedules_by_persona, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("duration_unit", "minutes"))
        return real_plot(event_name, metric, schedules_by_persona, out_dir, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod, "plot_per_event_persona_lines", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["lines"],
            line_events_by_persona={"g_fulltime": {"walking"}},
            event_config=cfg,
            persona_config=persona_cfg,
        )
    # One call per metric, both with unit='hours'.
    assert captured == ["hours", "hours"]


def test_render_all_charts_per_event_lines_falls_back_to_catalog_unit_on_conflict(
    tmp_path: Path,
):
    """Mixed override units fall back to the catalog unit."""
    schedules = [
        *_lines_schedules("alice", ["a0", "a1"]),
        *_lines_schedules("bob", ["b0", "b1"]),
    ]
    cfg = _event_config(
        {
            "walking": _vmax_event_def(
                "walking", episodes_max=5, duration_max=90, duration_unit="minutes"
            )
        }
    )
    persona_cfg = PersonaConfig(
        personas=[
            _persona(
                "alice",
                overrides={
                    "walking": EventOverride(
                        total_event_duration=TotalDuration(
                            scale="day", min=1, max=4, unit="hours"
                        )
                    )
                },
            ),
            _persona(
                "bob",
                overrides={
                    "walking": EventOverride(
                        total_event_duration=TotalDuration(
                            scale="day", min=15, max=240, unit="minutes"
                        )
                    )
                },
            ),
        ]
    )
    captured: list[str] = []
    from src.scripts.persona.analytics import charts as charts_mod

    real_plot = charts_mod.plot_per_event_persona_lines

    def spy(event_name, metric, schedules_by_persona, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("duration_unit", "minutes"))
        return real_plot(event_name, metric, schedules_by_persona, out_dir, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(charts_mod, "plot_per_event_persona_lines", spy):
        render_all_charts(
            schedules,
            out_dir=tmp_path,
            kinds=["lines"],
            line_events_by_persona={"alice": {"walking"}, "bob": {"walking"}},
            event_config=cfg,
            persona_config=persona_cfg,
        )
    # Catalog says minutes; conflict pushes the resolver back to that.
    assert captured == ["minutes", "minutes"]


def test_render_all_charts_legacy_lines_field_unchanged_when_param_absent(
    tmp_path: Path,
):
    """Without the selector, the legacy `lines` field is populated."""
    schedules = _lines_schedules("g_fulltime", ["p0", "p1"])
    artifacts = render_all_charts(schedules, out_dir=tmp_path, kinds=["lines"])
    assert set(artifacts.lines.keys()) == {"g_fulltime"}
    assert artifacts.per_event_lines == {}


def test_render_all_charts_per_event_lines_skips_metric_when_plot_returns_none(
    tmp_path: Path,
):
    """When the plot helper returns `None`, no entry is recorded."""
    schedules = [
        _schedule("p0", [{"lunch": [(720, 30)]} for _ in range(7)], persona_id="alice")
    ]
    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["lines"],
        line_events_by_persona={"alice": {"walking"}},
    )
    assert artifacts.per_event_lines == {}


# ---------------------------------------------------------------------------
# ChartArtifacts.all_paths surfaces per-member context heatmaps and lines
# ---------------------------------------------------------------------------


def test_chart_artifacts_all_paths_includes_per_member_context_heatmaps(tmp_path: Path):
    """`all_paths` walks the four-level per_member_context_heatmaps nested dict."""
    p1 = tmp_path / "h_count.png"
    p2 = tmp_path / "h_duration.png"
    p1.touch()
    p2.touch()
    art = ChartArtifacts(
        per_member_context_heatmaps={
            "alice": {"mood_emotion": {"happy": {"count": p1, "duration": p2}}}
        }
    )
    paths = set(art.all_paths)
    assert {p1, p2}.issubset(paths)


def test_chart_artifacts_all_paths_includes_per_member_context_lines(tmp_path: Path):
    """`all_paths` walks the three-level per_member_context_lines nested dict."""
    p1 = tmp_path / "l_count.png"
    p2 = tmp_path / "l_duration.png"
    p1.touch()
    p2.touch()
    art = ChartArtifacts(
        per_member_context_lines={
            "mood_emotion": {"happy": {"count": p1, "duration": p2}}
        }
    )
    paths = set(art.all_paths)
    assert {p1, p2}.issubset(paths)


# ---------------------------------------------------------------------------
# _members_for_persona resolution
# ---------------------------------------------------------------------------


class TestMembersForPersona:
    def test_reads_pairs_from_persona_catalog(self):
        """A persona declaring context categories returns sorted (category, member) pairs."""
        from types import SimpleNamespace

        cat = SimpleNamespace(members={"happy": object(), "sad": object()})
        persona = SimpleNamespace(id="alice", contexts={"mood_emotion": cat})
        cfg = SimpleNamespace(personas=[persona])
        out = _members_for_persona("alice", cfg, [])
        assert out == [("mood_emotion", "happy"), ("mood_emotion", "sad")]

    def test_unknown_persona_in_catalog_returns_empty(self):
        """An unknown persona id with a non-None catalog still returns an empty list."""
        from types import SimpleNamespace

        cfg = SimpleNamespace(personas=[SimpleNamespace(id="other", contexts={})])
        assert _members_for_persona("missing", cfg, []) == []

    def test_falls_back_to_schedule_contexts_when_catalog_absent(self):
        """No persona_config falls back to scanning the persona's contexts list."""
        import datetime as _dt

        from src.scripts.persona.context.schema import ContextEpisode
        from src.scripts.persona.domain.schedule import PersonSchedule

        ep1 = ContextEpisode(
            name="happy",
            category="mood_emotion",
            date=_dt.date(2026, 5, 4),
            start_minutes=480,
            end_minutes=540,
        )
        ep2 = ContextEpisode(
            name="tired",
            category="energy_state",
            date=_dt.date(2026, 5, 4),
            start_minutes=600,
            end_minutes=660,
        )
        sched = PersonSchedule(
            person_id="p1",
            persona_id="alice",
            person_seed=1,
            days=[],
            contexts=[ep1, ep2],
        )
        assert _members_for_persona("alice", None, [sched]) == [
            ("energy_state", "tired"),
            ("mood_emotion", "happy"),
        ]


# ---------------------------------------------------------------------------
# render_all_charts: context-heatmap and context-lines families
# ---------------------------------------------------------------------------


def _context_schedule(person_id: str, persona_id: str, name: str = "happy"):
    """Build a 14-day schedule with one mood_emotion episode per day."""
    import datetime as _dt

    from src.scripts.persona.context.schema import ContextEpisode
    from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

    start = _dt.date(2026, 5, 4)
    days = []
    episodes = []
    for i in range(14):
        d = start + _dt.timedelta(days=i)
        days.append(
            DaySchedule(
                day_index=i,
                date=d,
                weekday=d.strftime("%a"),
                events={},
                spillovers=[],
            )
        )
        episodes.append(
            ContextEpisode(
                name=name,
                category="mood_emotion",
                date=d,
                start_minutes=480,
                end_minutes=540,
            )
        )
    return PersonSchedule(
        person_id=person_id,
        persona_id=persona_id,
        person_seed=1,
        days=days,
        contexts=episodes,
    )


def test_render_all_charts_context_heatmap_writes_one_per_member_metric(
    tmp_path: Path,
):
    """Selecting `context-heatmap` writes one PNG per (persona, category, member, metric)."""
    sched = _context_schedule("p1", "alice")
    art = render_all_charts(
        [sched],
        out_dir=tmp_path,
        kinds=["context-heatmap"],
    )
    by_member = art.per_member_context_heatmaps.get("alice", {}).get("mood_emotion", {})
    happy = by_member.get("happy", {})
    assert "count" in happy and "duration" in happy
    for path in happy.values():
        assert path.exists()


def test_render_all_charts_context_lines_writes_one_per_member_metric(tmp_path: Path):
    """Selecting `context-lines` writes one PNG per (category, member, metric) across personas."""
    sched_a = _context_schedule("p1", "alice")
    sched_b = _context_schedule("p2", "bob")
    art = render_all_charts(
        [sched_a, sched_b],
        out_dir=tmp_path,
        kinds=["context-lines"],
    )
    by_member = art.per_member_context_lines.get("mood_emotion", {})
    happy = by_member.get("happy", {})
    assert "count" in happy and "duration" in happy
    for path in happy.values():
        assert path.exists()


def test_render_all_charts_context_lines_skips_personas_without_member(tmp_path: Path):
    """`context-lines` only charts personas whose catalog declares the (category, member) pair."""
    from types import SimpleNamespace

    sched_a = _context_schedule("p1", "alice", name="happy")
    sched_b = _context_schedule("p2", "bob", name="sad")
    # Stub persona_config so alice declares only `happy` and bob declares only `sad`.
    cat_alice = SimpleNamespace(members={"happy": object()})
    cat_bob = SimpleNamespace(members={"sad": object()})
    cfg = SimpleNamespace(
        personas=[
            SimpleNamespace(id="alice", contexts={"mood_emotion": cat_alice}),
            SimpleNamespace(id="bob", contexts={"mood_emotion": cat_bob}),
        ]
    )
    art = render_all_charts(
        [sched_a, sched_b],
        out_dir=tmp_path,
        kinds=["context-lines"],
        persona_config=cfg,
    )
    by_cat = art.per_member_context_lines.get("mood_emotion", {})
    assert "happy" in by_cat
    assert "sad" in by_cat
