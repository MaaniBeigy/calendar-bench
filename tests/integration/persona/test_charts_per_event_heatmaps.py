"""Integration tests for the per-event heatmap pipeline.

Drives `render_all_charts` end-to-end with a small synthetic schedule
plus a matching `EventConfig`, then asserts that both the count and
the duration PNG land on disk per variable event and that the colour
scale reflects the catalog's per-day max.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from src.scripts.persona.analytics.charts import (
    heatmap_vmax_for_event,
    heatmap_vmin_for_event,
    render_all_charts,
    variable_heatmap_events,
)
from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    Persona,
    PersonaConfig,
    PersonaEventStage,
    TemporalPattern,
    TotalDuration,
)
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

_START = _dt.date(2026, 5, 18)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _event_def(
    name: str,
    *,
    episodes_max: int,
    duration_max: int,
    temporal_patterns: list[TemporalPattern] | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category="test",
        per_event_duration=DurationRange(min=10, max=duration_max, unit="minutes"),
        total_event_duration=TotalDuration(
            scale="day", min=10, max=duration_max, unit="minutes"
        ),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=episodes_max),
        temporal_patterns=temporal_patterns or [],
    )


def _event_config() -> EventConfig:
    return EventConfig(
        categories={
            "test": Category(
                name="test",
                events={
                    "walking": _event_def(
                        "walking",
                        episodes_max=2,
                        duration_max=240,
                        temporal_patterns=[TemporalPattern(mode="trend")],
                    ),
                    "office_work": _event_def(
                        "office_work",
                        episodes_max=4,
                        duration_max=240,
                        temporal_patterns=[TemporalPattern(mode="seasonality")],
                    ),
                },
            )
        }
    )


def _persona_config() -> PersonaConfig:
    return PersonaConfig(
        personas=[
            Persona(
                id="g_fulltime",
                instances=1,
                occupation_status="fulltime",
                stages=[PersonaEventStage(name="walking", days=["Mon"])],
            )
        ]
    )


def _schedule(person_id: str) -> PersonSchedule:
    days: list[DaySchedule] = []
    for i in range(14):
        events = {
            "walking": [EventInstance(event_name="walking", start=420, duration=30)],
            "office_work": [
                EventInstance(event_name="office_work", start=540, duration=60),
                EventInstance(event_name="office_work", start=720, duration=60),
            ],
        }
        days.append(
            DaySchedule(
                day_index=i,
                date=_START + _dt.timedelta(days=i),
                weekday=_WEEKDAYS[i % 7],
                events=events,
                spillovers=[],
            )
        )
    return PersonSchedule(
        person_id=person_id,
        persona_id="g_fulltime",
        person_seed=0,
        days=days,
    )


def test_render_all_charts_emits_both_metrics_per_variable_event(tmp_path: Path):
    schedules = [_schedule("g_fulltime_0000")]
    event_cfg = _event_config()
    persona_cfg = _persona_config()
    heatmap_events = variable_heatmap_events(event_cfg, persona_cfg)

    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona=heatmap_events,
        event_config=event_cfg,
    )

    fulltime = artifacts.per_event_heatmaps["g_fulltime"]
    assert set(fulltime.keys()) == {"walking", "office_work"}
    for event_name, metric_paths in fulltime.items():
        assert set(metric_paths.keys()) == {"count", "duration"}
        for metric, path in metric_paths.items():
            assert path.exists()
            assert path.name == (f"heatmap_g_fulltime_{event_name}_{metric}_mean.png")


def test_render_all_charts_vmin_vmax_reflect_catalog_range(tmp_path: Path):
    """The vmin / vmax resolved per event must match the catalog's
    per-day range (episode count for `count`, minutes for `duration`)."""
    event_cfg = _event_config()
    assert heatmap_vmin_for_event(event_cfg, "walking", "count") == 0.0
    assert heatmap_vmax_for_event(event_cfg, "walking", "count") == 2.0
    assert heatmap_vmin_for_event(event_cfg, "walking", "duration") == 10.0
    assert heatmap_vmax_for_event(event_cfg, "walking", "duration") == 240.0
    assert heatmap_vmin_for_event(event_cfg, "office_work", "count") == 0.0
    assert heatmap_vmax_for_event(event_cfg, "office_work", "count") == 4.0
    assert heatmap_vmin_for_event(event_cfg, "office_work", "duration") == 10.0
    assert heatmap_vmax_for_event(event_cfg, "office_work", "duration") == 240.0


def test_render_all_charts_per_event_lines_end_to_end(tmp_path: Path):
    """`line_events_by_persona` writes one PNG per `(variable_event, metric)`."""
    schedules = [_schedule("g_fulltime_0000"), _schedule("g_fulltime_0001")]
    event_cfg = _event_config()
    persona_cfg = _persona_config()
    variable_events = variable_heatmap_events(event_cfg, persona_cfg)

    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["lines"],
        line_events_by_persona=variable_events,
        event_config=event_cfg,
        persona_config=persona_cfg,
    )

    assert set(artifacts.per_event_lines.keys()) == {"walking", "office_work"}
    for event_name, metric_paths in artifacts.per_event_lines.items():
        assert set(metric_paths.keys()) == {"count", "duration"}
        for metric, path in metric_paths.items():
            assert path.exists()
            assert path.name == f"lines_{event_name}_{metric}.png"


def test_render_all_charts_all_paths_includes_every_metric(tmp_path: Path):
    schedules = [_schedule("g_fulltime_0000")]
    event_cfg = _event_config()
    persona_cfg = _persona_config()

    artifacts = render_all_charts(
        schedules,
        out_dir=tmp_path,
        kinds=["heatmap"],
        heatmap_events_by_persona=variable_heatmap_events(event_cfg, persona_cfg),
        event_config=event_cfg,
    )
    all_paths = set(artifacts.all_paths)
    # 2 events x 2 metrics = 4 heatmap PNGs
    expected_count = 0
    for event_paths in artifacts.per_event_heatmaps.values():
        for metric_paths in event_paths.values():
            expected_count += len(metric_paths)
            for path in metric_paths.values():
                assert path in all_paths
    assert expected_count == 4
