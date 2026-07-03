"""End-to-end integration tests for the L_pref v2 pipeline.

Wires :class:`PreferenceCache` + :class:`PreferenceMapper` +
:class:`PersonaConstraints` + :func:`compute_l_pref_v2` together with
fakeredis and verifies that augmenter placements at varying alignment
levels produce ordered loss values.

These tests guard the integration boundary between the cache, mapper,
constraints, and aggregator; if any one component breaks its contract
the ordering assertions below will fail.
"""

from __future__ import annotations

import datetime as _dt

import fakeredis

from src.scripts.persona.config.schema import (
    Category,
    DurationRange,
    EpisodeRange,
    EventConfig,
    EventDefinition,
    EventOverride,
    Persona,
    PersonaEventStage,
    TemporalPattern,
    TotalDuration,
    WindowRange,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.domain.calendar import AugmentedCalendar
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.export.report_writer import (
    format_preference_breakdown_report,
)
from src.scripts.scenarios.metrics.loss import compute_l_pref_v2
from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
from src.scripts.scenarios.metrics.preference_constraints import PersonaConstraints
from src.scripts.scenarios.metrics.preference_mapping import PreferenceMapper

HEALTH_PREFIX = "https://w3id.org/calendar-bench/health/task/"


def _build_scenario(
    *,
    placements: list[tuple[int, int, _dt.date]],
):
    """Build a tiny synthetic scenario.

    Returns `(solution, persona_constraints, window_map)`.
    """
    cache = PreferenceCache.from_client(fakeredis.FakeRedis(decode_responses=True))
    mapper = PreferenceMapper(cache=cache)
    window_map = WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        }
    )
    # Catalog event with rich preferences: per-event duration 30-75 min,
    # day total 30-75, weekend seasonality boost via temporal patterns,
    # and a HealthTasks literal IRI for the literal-tier match.
    walking = EventDefinition(
        name="walking",
        category="sports",
        per_event_duration=DurationRange(min=30, max=75, unit="minutes"),
        total_event_duration=TotalDuration(min=30, max=75, scale="day", unit="minutes"),
        total_event_episodes=EpisodeRange(scale="day", min=0, max=1),
        health_task_iri=HEALTH_PREFIX + "schedule-a-30-minute-walk",
        temporal_patterns=[
            TemporalPattern(
                mode="fix", details={"within": "morning"}
            ),  # walking should be in the morning
        ],
    )
    ev_cfg = EventConfig(
        categories={"sports": Category(name="sports", events={"walking": walking})}
    )
    persona = Persona(
        id="b_parttime_morning",
        occupation_status="parttime",
        stages=[
            PersonaEventStage(
                name="walking", time="morning", duration_minutes=45, days=["Mon"]
            )
        ],
    )
    pc = PersonaConstraints(
        persona=persona,
        event_config=ev_cfg,
        window_map=window_map,
        horizon_days=28,
        horizon_start_date=_dt.date(2026, 6, 1),
        mapper=mapper,
        experiment="exp_b",
        scenario="health_l1",
    )
    task = RecommendedTask(
        label="schedule-a-30-minute-walk",
        duration_min=30,
        duration_max=30,
        ontology_uri=HEALTH_PREFIX + "schedule-a-30-minute-walk",
    )
    schedules = [
        ScheduledTask(
            task=task,
            start_minutes=start,
            end_minutes=end,
            is_standalone=True,
            concurrent_with=None,
            date=date,
        )
        for (start, end, date) in placements
    ]
    solution = SchedulingSolution(
        person_id="b_parttime_morning_001",
        augmented_calendar=AugmentedCalendar(person_id="b_parttime_morning_001"),
        tasks=[task],
        scheduled=schedules,
        unscheduled=[],
    )
    return solution, pc, window_map


class TestEndToEndOrdering:
    def test_perfect_alignment_beats_partial_beats_violation(self):
        """An augmenter that places the walk at 07:00 for 30 minutes on
        Monday morning must score better than one that places the same
        walk at 22:00 (night, ignoring the morning fix-pattern)."""
        # 1) Perfect alignment: Monday morning, 30 min in [30, 75]
        good, pc_good, wm_good = _build_scenario(
            placements=[(420, 450, _dt.date(2026, 6, 1))]  # Mon 07:00-07:30
        )
        # 2) Wrong time-of-day (afternoon): per_event ok, but fix-pattern
        # violated
        afternoon, pc_after, wm_after = _build_scenario(
            placements=[(800, 830, _dt.date(2026, 6, 1))]  # Mon afternoon
        )
        # 3) Wrong time AND wrong duration (5 min night walk)
        bad, pc_bad, wm_bad = _build_scenario(
            placements=[(1320, 1325, _dt.date(2026, 6, 1))]  # Mon 22:00 5 min
        )
        good_loss, _, _ = compute_l_pref_v2(
            good, persona_constraints=pc_good, window_map=wm_good
        )
        afternoon_loss, _, _ = compute_l_pref_v2(
            afternoon, persona_constraints=pc_after, window_map=wm_after
        )
        bad_loss, _, _ = compute_l_pref_v2(
            bad, persona_constraints=pc_bad, window_map=wm_bad
        )
        assert good_loss is not None
        assert afternoon_loss is not None
        assert bad_loss is not None
        # Monotonic ordering: less alignment to higher loss.
        assert good_loss < afternoon_loss < bad_loss

    def test_warm_up_then_subsequent_calls_use_cache(self):
        """Warming the cache before evaluation must not change results."""
        solution, pc, wm = _build_scenario(
            placements=[(420, 450, _dt.date(2026, 6, 1))]
        )
        # First pass: cache empty to mapper computes
        loss1, _, _ = compute_l_pref_v2(solution, persona_constraints=pc, window_map=wm)
        # Second pass: cache populated to mapper reads back
        loss2, _, _ = compute_l_pref_v2(solution, persona_constraints=pc, window_map=wm)
        assert loss1 == loss2

    def test_pattern_rows_surface_for_violation(self):
        solution, pc, wm = _build_scenario(
            placements=[(800, 830, _dt.date(2026, 6, 1))]  # afternoon
        )
        loss, rows, _ = compute_l_pref_v2(
            solution, persona_constraints=pc, window_map=wm
        )
        assert loss is not None
        assert any(r.mode == "fix" for r in rows)


class TestReporterIntegration:
    def test_preference_breakdown_renders_into_report(self):
        """The per-leg L_pref roll-up renders into its own dedicated
        report file (`preference_breakdown.{txt,json}`); split out
        of the legacy combined `scheduling_loss.txt` 2026-05-14."""
        breakdown = {
            "per_occurrence_duration": {"applicable_tasks": 5, "mean_loss": 0.10},
            "temporal_pattern_fix": {
                "applicable_tasks": 5,
                "mean_loss": 0.40,
                "mape": 0.55,
            },
            "persona_stage_semantic": {
                "applicable_tasks": 0,
                "mean_loss": None,
                "mape": None,
            },
        }
        rendered = format_preference_breakdown_report(breakdown)
        assert "L_pref Preference Breakdown" in rendered
        assert "per_occurrence_duration" in rendered
        assert "loss=0.1000" in rendered
        assert "applicable_tasks=5" in rendered
        # MAPE rendering for trend / seasonality rows
        assert "mape=55.0%" in rendered
        # No-signal row uses `n/a`
        assert "loss=n/a" in rendered
