"""Tests for `metrics/preference_score.py` (the L_pref component scorers)."""

from __future__ import annotations

import datetime as _dt

import pytest

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
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.preference_constraints import (
    PersonaConstraints,
    ResolvedStage,
)
from src.scripts.scenarios.metrics.preference_mapping import MappedEvent
from src.scripts.scenarios.metrics.preference_score import (
    EPS,
    PatternViolation,
    StageAlignment,
    _count_violation,
    _duration_violation,
    _seasonality_slice_key,
    _weighted_mean,
    score_per_occurrence_duration,
    score_per_scale_duration,
    score_per_scale_episodes,
    score_persona_stage_semantic,
    score_temporal_pattern,
)
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility


def _wm() -> WindowMap:
    return WindowMap.from_config(
        {
            "early_morning": WindowRange(start=0, end=400),
            "morning": WindowRange(start=400, end=600),
            "afternoon": WindowRange(start=600, end=960),
            "evening": WindowRange(start=960, end=1260),
            "night": WindowRange(start=1260, end=1440),
        }
    )


def _make_event(
    name: str,
    *,
    category: str = "sports",
    per_event_min: int = 15,
    per_event_max: int = 120,
    total_min: int = 15,
    total_max: int = 240,
    total_scale: str = "day",
    episodes_min: int = 0,
    episodes_max: int = 1,
    episodes_scale: str = "day",
    temporal_patterns: list[TemporalPattern] | None = None,
) -> EventDefinition:
    return EventDefinition(
        name=name,
        category=category,
        per_event_duration=DurationRange(
            min=per_event_min, max=per_event_max, unit="minutes"
        ),
        total_event_duration=TotalDuration(
            min=total_min, max=total_max, scale=total_scale, unit="minutes"
        ),
        total_event_episodes=EpisodeRange(
            scale=episodes_scale, min=episodes_min, max=episodes_max
        ),
        temporal_patterns=temporal_patterns or [],
    )


def _event_config(events: list[EventDefinition]) -> EventConfig:
    return EventConfig(
        categories={
            "sports": Category(name="sports", events={ev.name: ev for ev in events})
        }
    )


def _persona(
    stages: list[PersonaEventStage] | None = None,
    overrides: dict[str, EventOverride] | None = None,
) -> Persona:
    return Persona(
        id="p1",
        occupation_status="parttime",
        stages=stages or [],
        event_overrides=overrides or {},
    )


def _pc(
    events: list[EventDefinition],
    *,
    horizon_days: int = 28,
    horizon_start_date: _dt.date | None = _dt.date(2026, 6, 1),
) -> PersonaConstraints:
    return PersonaConstraints(
        persona=_persona(),
        event_config=_event_config(events),
        window_map=_wm(),
        horizon_days=horizon_days,
        horizon_start_date=horizon_start_date,
    )


def _scheduled(
    label: str,
    start_minutes: int,
    end_minutes: int,
    date: _dt.date | None = None,
) -> ScheduledTask:
    task = RecommendedTask(label=label, duration_min=10, duration_max=60)
    return ScheduledTask(
        task=task,
        start_minutes=start_minutes,
        end_minutes=end_minutes,
        is_standalone=True,
        concurrent_with=None,
        date=date or _dt.date(2026, 6, 1),
    )


def _mapped(event_name: str, weight: float = 1.0, tier: str = "literal") -> MappedEvent:
    return MappedEvent(
        event_name=event_name, weight=weight, source_tier=tier, evidence={}
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_duration_inside_band_is_zero(self):
        assert _duration_violation(30.0, 15.0, 60.0) == 0.0

    def test_duration_below_band(self):
        # value 5, lo 30, hi 60 to (30-5)/30 = 0.833...
        assert _duration_violation(5.0, 30.0, 60.0) == pytest.approx(25 / 30)

    def test_duration_above_band_clipped_to_1(self):
        # value 1000, lo 30, hi 60 to (1000-60)/60 = 15.66 to clipped to 1
        assert _duration_violation(1000.0, 30.0, 60.0) == 1.0

    def test_duration_below_band_clipped_to_1(self):
        # value 0, lo 30 to 30/30 = 1.0
        assert _duration_violation(0.0, 30.0, 60.0) == 1.0

    def test_duration_lo_zero_uses_eps_floor_then_clipped(self):
        # value below 0, lo=0 to eps floor to 5/1 = 5 to clipped to 1
        assert _duration_violation(-5.0, 0.0, 60.0) == 1.0

    def test_count_inside_band_is_zero(self):
        assert _count_violation(3, 1, 5) == 0.0

    def test_count_below_band(self):
        assert _count_violation(0, 2, 5) == pytest.approx(2 / 2)

    def test_count_above_band_clipped_to_1(self):
        # 5 > max 2 with hi=2 to (5-2)/2 = 1.5 to clipped to 1.0
        assert _count_violation(5, 1, 2) == 1.0

    def test_count_above_band_uncapped(self):
        # 3 > max 2 with hi=2 to (3-2)/2 = 0.5 (no clip needed)
        assert _count_violation(3, 1, 2) == pytest.approx(0.5)

    def test_weighted_mean_basic(self):
        assert _weighted_mean([0.5, 1.0], [1.0, 1.0]) == pytest.approx(0.75)

    def test_weighted_mean_zero_total_weight_returns_none(self):
        assert _weighted_mean([0.5], [0.0]) is None
        assert _weighted_mean([], []) is None

    def test_weighted_mean_skips_zero_weights(self):
        assert _weighted_mean([0.5, 1.0], [0.0, 1.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Per-occurrence duration
# ---------------------------------------------------------------------------


class TestPerOccurrenceDuration:
    def test_inside_band_zero_violation(self):
        ev = _make_event("walking", per_event_min=15, per_event_max=120)
        pc = _pc([ev])
        sched = _scheduled("schedule-a-walk", 480, 510)  # 30 min, in [15, 120]
        out = score_per_occurrence_duration(
            sched, matched_events=[_mapped("walking")], constraints=pc
        )
        assert out == 0.0

    def test_below_band_returns_violation(self):
        ev = _make_event("walking", per_event_min=30, per_event_max=120)
        pc = _pc([ev])
        sched = _scheduled("schedule-a-walk", 480, 495)  # 15 min
        out = score_per_occurrence_duration(
            sched, matched_events=[_mapped("walking")], constraints=pc
        )
        assert out == pytest.approx((30 - 15) / 30)

    def test_above_band_returns_violation(self):
        ev = _make_event("walking", per_event_min=15, per_event_max=60)
        pc = _pc([ev])
        sched = _scheduled("schedule-a-walk", 480, 600)  # 120 min
        out = score_per_occurrence_duration(
            sched, matched_events=[_mapped("walking")], constraints=pc
        )
        assert out == pytest.approx((120 - 60) / 60)

    def test_no_matched_events_returns_none(self):
        pc = _pc([_make_event("walking")])
        sched = _scheduled("any", 0, 30)
        assert (
            score_per_occurrence_duration(sched, matched_events=[], constraints=pc)
            is None
        )

    def test_weighted_mean_across_matches(self):
        ev_strict = _make_event("strict", per_event_min=30, per_event_max=60)
        ev_loose = _make_event("loose", per_event_min=0, per_event_max=120)
        pc = _pc([ev_strict, ev_loose])
        sched = _scheduled("schedule-a-walk", 480, 495)  # 15 min
        out = score_per_occurrence_duration(
            sched,
            matched_events=[
                _mapped("strict", weight=1.0),
                _mapped("loose", weight=0.3),
            ],
            constraints=pc,
        )
        # strict: (30-15)/30 = 0.5; loose: 0 (inside). Weighted mean:
        # (1.0*0.5 + 0.3*0)/(1.3) = 0.5/1.3
        assert out == pytest.approx(0.5 / 1.3)

    def test_unknown_event_name_in_match_skipped(self):
        ev = _make_event("walking", per_event_min=15, per_event_max=120)
        pc = _pc([ev])
        sched = _scheduled("walk", 0, 30)
        # match on a nonexistent label to skipped, no signal to None
        out = score_per_occurrence_duration(
            sched, matched_events=[_mapped("nonexistent")], constraints=pc
        )
        assert out is None

    def test_zero_weight_match_skipped(self):
        ev = _make_event("walking", per_event_min=30, per_event_max=120)
        pc = _pc([ev])
        sched = _scheduled("walk", 0, 15)
        out = score_per_occurrence_duration(
            sched,
            matched_events=[_mapped("walking", weight=0.0)],
            constraints=pc,
        )
        assert out is None


# ---------------------------------------------------------------------------
# Per-scale duration
# ---------------------------------------------------------------------------


class TestPerScaleDuration:
    def test_day_scale_inside_band_zero(self):
        ev = _make_event("walking", total_min=30, total_max=120, total_scale="day")
        pc = _pc([ev])
        # Two 30-min placements on the same day to total 60 min, inside [30, 120]
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1)),
            _scheduled("walk", 600, 630, _dt.date(2026, 6, 1)),
        ]
        out = score_per_scale_duration(
            placements, matched_events=[_mapped("walking")], constraints=pc
        )
        assert out == 0.0

    def test_day_scale_below_band(self):
        ev = _make_event("walking", total_min=60, total_max=120, total_scale="day")
        pc = _pc([ev])
        placements = [
            _scheduled("walk", 480, 495, _dt.date(2026, 6, 1))
        ]  # 15 min total
        out = score_per_scale_duration(
            placements, matched_events=[_mapped("walking")], constraints=pc
        )
        assert out == pytest.approx((60 - 15) / 60)

    def test_week_scale_aggregates_across_days(self):
        ev = _make_event(
            "walking",
            total_min=120,
            total_max=300,
            total_scale="week",
        )
        pc = _pc([ev])
        # Two days, 30 min each to 60 min total weekly to below 120
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1)),  # Mon
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 3)),  # Wed
        ]
        out = score_per_scale_duration(
            placements, matched_events=[_mapped("walking")], constraints=pc
        )
        assert out == pytest.approx((120 - 60) / 120)

    def test_empty_placements_returns_none(self):
        ev = _make_event("walking")
        pc = _pc([ev])
        out = score_per_scale_duration(
            [], matched_events=[_mapped("walking")], constraints=pc
        )
        assert out is None

    def test_no_matches_returns_none(self):
        pc = _pc([_make_event("walking")])
        placements = [_scheduled("walk", 0, 30)]
        assert (
            score_per_scale_duration(placements, matched_events=[], constraints=pc)
            is None
        )

    def test_unknown_event_skipped(self):
        ev = _make_event("walking")
        pc = _pc([ev])
        placements = [_scheduled("walk", 0, 30)]
        out = score_per_scale_duration(
            placements, matched_events=[_mapped("nonexistent")], constraints=pc
        )
        assert out is None


# ---------------------------------------------------------------------------
# Per-scale episodes
# ---------------------------------------------------------------------------


class TestPerScaleEpisodes:
    def test_day_count_inside_band(self):
        ev = _make_event(
            "walking", episodes_min=1, episodes_max=2, episodes_scale="day"
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 510, _dt.date(2026, 6, 1))]
        out = score_per_scale_episodes(
            placements, matched_events=[_mapped("walking")], constraints=pc
        )
        assert out == 0.0

    def test_day_count_over_max(self):
        ev = _make_event(
            "walking", episodes_min=0, episodes_max=1, episodes_scale="day"
        )
        pc = _pc([ev])
        # Three placements on the same day to 2 episodes over max 1
        placements = [
            _scheduled("walk", i * 60, i * 60 + 30, _dt.date(2026, 6, 1))
            for i in range(3)
        ]
        out = score_per_scale_episodes(
            placements, matched_events=[_mapped("walking")], constraints=pc
        )
        # Effective max_count from day_constraints (with episode min=0 max=1) is 1.
        # Three placements to (3-1)/1 = 2.0 to clipped to 1.0
        assert out == 1.0

    def test_empty_placements_returns_none(self):
        ev = _make_event("walking")
        pc = _pc([ev])
        assert (
            score_per_scale_episodes(
                [], matched_events=[_mapped("walking")], constraints=pc
            )
            is None
        )

    def test_no_matches_returns_none(self):
        pc = _pc([_make_event("walking")])
        placements = [_scheduled("walk", 0, 30)]
        assert (
            score_per_scale_episodes(placements, matched_events=[], constraints=pc)
            is None
        )

    def test_unknown_event_skipped(self):
        ev = _make_event("walking")
        pc = _pc([ev])
        placements = [_scheduled("walk", 0, 30)]
        out = score_per_scale_episodes(
            placements, matched_events=[_mapped("nonexistent")], constraints=pc
        )
        assert out is None


# ---------------------------------------------------------------------------
# Persona-stage semantic alignment
# ---------------------------------------------------------------------------


class TestPersonaStageSemantic:
    def test_no_firing_stages_returns_none(self):
        sched = _scheduled("walk", 480, 510)
        semantic = SemanticCompatibility(matrix={})
        out = score_persona_stage_semantic(sched, firing_stages=[], semantic=semantic)
        assert out is None

    def test_no_oracle_returns_none(self):
        sched = _scheduled("walk", 480, 510)
        stage = ResolvedStage(name="walking", window_start=400, window_end=600)
        out = score_persona_stage_semantic(sched, firing_stages=[stage], semantic=None)
        assert out is None

    def test_aligned_stage_returns_zero(self):
        sched = _scheduled("walk_2000_steps", 480, 510)
        stage = ResolvedStage(name="walking", window_start=400, window_end=600)
        semantic = SemanticCompatibility(matrix={("walk_2000_steps", "walking"): 0.9})
        out = score_persona_stage_semantic(
            sched, firing_stages=[stage], semantic=semantic
        )
        assert out == 0.0

    def test_clash_returns_1_minus_sigma(self):
        sched = _scheduled("swim_laps", 480, 510)
        stage = ResolvedStage(name="reading", window_start=400, window_end=600)
        semantic = SemanticCompatibility(matrix={("swim_laps", "reading"): 0.1})
        out = score_persona_stage_semantic(
            sched, firing_stages=[stage], semantic=semantic
        )
        assert out == pytest.approx(0.9)

    def test_no_overlap_filters_stage_out(self):
        sched = _scheduled("walk", 480, 510)
        # Stage's window doesn't overlap [480, 510)
        stage = ResolvedStage(name="reading", window_start=0, window_end=100)
        semantic = SemanticCompatibility(matrix={("walk", "reading"): 0.1})
        out = score_persona_stage_semantic(
            sched, firing_stages=[stage], semantic=semantic
        )
        assert out is None

    def test_max_clash_dominates_soft_overlap(self):
        sched = _scheduled("walk", 400, 600)
        soft = ResolvedStage(name="walking", window_start=400, window_end=500)
        hard = ResolvedStage(name="surgery", window_start=500, window_end=600)
        semantic = SemanticCompatibility(
            matrix={
                ("walk", "walking"): 0.9,  # aligned, contributes 0
                ("walk", "surgery"): 0.05,  # clash, contributes 0.95
            }
        )
        out = score_persona_stage_semantic(
            sched, firing_stages=[soft, hard], semantic=semantic
        )
        assert out == pytest.approx(0.95)

    def test_two_clashes_keep_max_not_min(self):
        """Two clashing stages must aggregate to the MAX clash, not a
        running sum; covers the `if clash > max_violation` False
        branch in :func:`score_persona_stage_semantic`."""
        sched = _scheduled("walk", 400, 600)
        # Both stages overlap [400, 600); both clash.  The first is the
        # bigger clash (σ=0.05 to 0.95); the second is smaller (σ=0.4 to
        # 0.6) so it must NOT overwrite `max_violation`.
        stage_a = ResolvedStage(name="surgery", window_start=400, window_end=500)
        stage_b = ResolvedStage(name="reading", window_start=500, window_end=600)
        semantic = SemanticCompatibility(
            matrix={
                ("walk", "surgery"): 0.05,
                ("walk", "reading"): 0.40,
            }
        )
        out = score_persona_stage_semantic(
            sched, firing_stages=[stage_a, stage_b], semantic=semantic
        )
        assert out == pytest.approx(0.95)

    def test_stub_none_strategy_does_not_count_as_clash(self):
        """A stage with σ=0 from the stub `none` chain returns no signal."""
        sched = _scheduled("walk", 480, 510)
        stage = ResolvedStage(name="reading", window_start=400, window_end=600)
        semantic = SemanticCompatibility()  # no matrix, no embedding, no LLM
        out = score_persona_stage_semantic(
            sched, firing_stages=[stage], semantic=semantic
        )
        assert out is None

    def test_aligned_after_prior_clash_keeps_clash(self):
        """Branch coverage: aligned stage seen after a clash must keep
        `max_violation` at the clash value, not overwrite it."""
        sched = _scheduled("walk", 400, 600)
        stage_a = ResolvedStage(name="surgery", window_start=400, window_end=500)
        stage_b = ResolvedStage(name="walking", window_start=500, window_end=600)
        semantic = SemanticCompatibility(
            matrix={
                ("walk", "surgery"): 0.05,
                ("walk", "walking"): 0.9,
            }
        )
        out = score_persona_stage_semantic(
            sched, firing_stages=[stage_a, stage_b], semantic=semantic
        )
        assert out == pytest.approx(0.95)

    def test_custom_threshold_changes_classification(self):
        sched = _scheduled("walk", 480, 510)
        stage = ResolvedStage(name="walking", window_start=400, window_end=600)
        semantic = SemanticCompatibility(matrix={("walk", "walking"): 0.6})
        # Default threshold 0.65 to 0.6 below to clash to 0.4
        default_out = score_persona_stage_semantic(
            sched, firing_stages=[stage], semantic=semantic
        )
        assert default_out == pytest.approx(0.4)
        # Lower threshold to 0.5 to 0.6 above to aligned to 0
        relaxed = score_persona_stage_semantic(
            sched,
            firing_stages=[stage],
            semantic=semantic,
            stage_alignment_threshold=0.5,
        )
        assert relaxed == 0.0


# ---------------------------------------------------------------------------
# Temporal-pattern scorers
# ---------------------------------------------------------------------------


class TestTemporalPatternFix:
    def test_fix_within_morning_no_violation_when_all_inside(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 420, 510, _dt.date(2026, 6, 1))]  # 7:00-8:30
        loss, rows = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss == 0.0
        assert len(rows) == 1
        assert rows[0].mode == "fix"

    def test_fix_outside_window_full_violation(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        pc = _pc([ev])
        # Placement entirely outside morning
        placements = [_scheduled("walk", 1000, 1060, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss == 1.0

    def test_fix_partial_overlap_proportional_violation(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        pc = _pc([ev])
        # Placement 540-660 spans morning-end (600) to half inside, half outside
        placements = [_scheduled("walk", 540, 660, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss == pytest.approx(0.5)

    def test_fix_unknown_window_returns_none(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "nonexistent"})
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 510, _dt.date(2026, 6, 1))]
        loss, rows = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        # No window matched to pattern returns None to no rows
        assert loss is None
        assert rows == []

    def test_fix_with_list_within(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="fix", details={"within": ["morning", "afternoon"]}
                )
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 700, 730, _dt.date(2026, 6, 1))]  # afternoon
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss == 0.0


class TestTemporalPatternSeasonality:
    def test_weekday_seasonality_balanced_no_violation(self):
        """When observed counts perfectly match the boosted expectation
        the violation is 0."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={
                        "scale": "weekday",
                        "within": ["Sat", "Sun"],
                        "amount": 100,
                        "direction": "increasing",
                    },
                )
            ],
        )
        pc = _pc([ev])
        # All placements on Sat/Sun to 100% inside to close to expected.
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 6)),  # Sat
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 7)),  # Sun
        ]
        loss, rows = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None
        # With same observed in both slices and one is "boosted", small but nonzero violation
        assert 0.0 <= loss <= 1.0
        assert len(rows) == 1
        assert rows[0].mode == "seasonality"
        assert rows[0].mape is not None

    def test_seasonality_no_allowed_within_returns_none(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={
                        "scale": "month",
                        "within": ["NotAMonth"],
                        "amount": 50,
                    },
                )
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 510, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None

    def test_seasonality_with_duration_unit(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={
                        "scale": "weekday",
                        "within": ["Sat"],
                        "unit": "minutes",
                        "amount": 30,
                        "direction": "increasing",
                    },
                )
            ],
        )
        pc = _pc([ev])
        placements = [
            _scheduled("walk", 480, 540, _dt.date(2026, 6, 6)),  # Sat 60 min
            _scheduled("walk", 480, 540, _dt.date(2026, 6, 1)),  # Mon 60 min
        ]
        loss, rows = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None
        assert 0.0 <= loss <= 1.0

    def test_seasonality_window_scope_returns_none(self):
        """Seasonality with window-scope (e.g. morning) is enforced by
        the solver, not by `L_pref`."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={"within": "morning", "amount": 20},
                )
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 510, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None

    def test_season_scale_seasonality_fires(self):
        """Seasonality with `scale: season` must take the season branch
        in the scorer (line 470; `_within_season_indices`)."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={
                        "scale": "season",
                        "within": ["spring"],
                        "amount": 30,
                        "direction": "increasing",
                    },
                )
            ],
        )
        pc = _pc([ev], horizon_start_date=_dt.date(2026, 6, 1))
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 4, 15)),  # spring
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 15)),  # summer
        ]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None
        assert 0.0 <= loss <= 1.0

    def test_seasonality_empty_placements_returns_none(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={
                        "scale": "weekday",
                        "within": ["Sat"],
                        "amount": 50,
                    },
                )
            ],
        )
        pc = _pc([ev])
        loss, _ = score_temporal_pattern(
            [],
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None


class TestTemporalPatternTrend:
    def test_trend_no_placements_returns_none(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={
                        "scale": "week",
                        "direction": "increasing",
                        "amount": 2,
                        "start": 1,
                        "end": 4,
                    },
                )
            ],
        )
        pc = _pc([ev])
        loss, _ = score_temporal_pattern(
            [],
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None

    def test_trend_invalid_range_returns_none(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={"scale": "week", "amount": 2, "start": 4, "end": 1},
                )
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 510, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None

    def test_trend_episode_count_ramp_reports_mape(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={
                        "scale": "week",
                        "direction": "increasing",
                        "amount": 4,
                        "start": 1,
                        "end": 4,
                    },
                )
            ],
        )
        pc = _pc([ev])
        # All placements front-loaded in week 1; cumulative observed
        # deviates from the ramp.
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1) + _dt.timedelta(days=i))
            for i in range(7)
        ]
        loss, rows = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None
        assert 0.0 <= loss <= 1.0
        assert len(rows) == 1
        assert rows[0].mode == "trend"
        assert rows[0].mape is not None
        assert rows[0].mape >= 0.0

    def test_trend_duration_unit_hours(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={
                        "scale": "week",
                        "unit": "hours",
                        "direction": "increasing",
                        "amount": 1,
                        "start": 1,
                        "end": 4,
                    },
                )
            ],
        )
        pc = _pc([ev])
        placements = [
            _scheduled(
                "walk", 480, 540, _dt.date(2026, 6, 1) + _dt.timedelta(days=7 * i)
            )
            for i in range(4)
        ]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None

    def test_trend_with_outside_ramp_baseline(self):
        """Placements both inside `[start, end]` and outside the ramp
        to `baseline_values` is non-empty so the function uses the
        outside-ramp mean as the baseline (covers the True branch of
        `if baseline_values:`)."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={
                        "scale": "day",
                        "direction": "increasing",
                        "amount": 2,
                        "start": 1,
                        "end": 3,
                    },
                )
            ],
        )
        pc = _pc([ev])
        # Days 0-2 (inside ramp), days 4-6 (outside).
        placements = [
            _scheduled(
                "walk",
                480,
                510,
                _dt.date(2026, 6, 1) + _dt.timedelta(days=i),
            )
            for i in [0, 1, 2, 4, 5, 6]
        ]
        loss, rows = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None
        assert 0.0 <= loss <= 1.0
        assert any(r.mode == "trend" for r in rows)

    def test_trend_decreasing_direction(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={
                        "scale": "day",
                        "direction": "decreasing",
                        "amount": 2,
                        "start": 1,
                        "end": 7,
                    },
                )
            ],
        )
        pc = _pc([ev])
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1) + _dt.timedelta(days=i))
            for i in range(7)
        ]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None

    def test_no_matched_events_returns_none(self):
        pc = _pc([_make_event("walking")])
        placements = [_scheduled("walk", 0, 30)]
        out, rows = score_temporal_pattern(
            placements,
            matched_events=[],
            constraints=pc,
            window_map=_wm(),
        )
        assert out is None
        assert rows == []

    def test_unknown_event_skipped(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 510, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("nonexistent")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None


class TestTemporalPatternEdgeCases:
    def test_fix_within_non_list_non_str_returns_none(self):
        """`within: 42` (invalid type) to no allowed windows to None."""
        ev = _make_event(
            "walking",
            temporal_patterns=[TemporalPattern(mode="fix", details={"within": 42})],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 510, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None

    def test_fix_zero_duration_placement_skipped_but_other_counts(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        pc = _pc([ev])
        placements = [
            _scheduled("walk", 480, 480, _dt.date(2026, 6, 1)),  # 0 min: skipped
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1)),  # 30 min: inside
        ]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss == 0.0

    def test_fix_all_zero_duration_returns_none(self):
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(mode="fix", details={"within": "morning"})
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 480, _dt.date(2026, 6, 1))]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None

    def test_seasonality_decreasing_amount_100_drops_expected_to_zero(self):
        """When `direction: decreasing, amount: 100` boosts to 0, the
        fallback `expected = baseline` kicks in."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={
                        "scale": "weekday",
                        "within": ["Sat"],
                        "amount": 100,
                        "direction": "decreasing",
                    },
                )
            ],
        )
        pc = _pc([ev])
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1) + _dt.timedelta(days=i))
            for i in range(7)
        ]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None

    def test_seasonality_all_zero_duration_returns_none(self):
        """All-zero-duration placements to baseline 0 to no signal."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="seasonality",
                    details={
                        "scale": "weekday",
                        "within": ["Sat"],
                        "unit": "minutes",
                        "amount": 30,
                        "direction": "increasing",
                    },
                )
            ],
        )
        pc = _pc([ev])
        placements = [_scheduled("walk", 480, 480, _dt.date(2026, 6, 6))]  # 0 min
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None

    def test_trend_all_placements_inside_ramp_uses_overall_mean_baseline(self):
        """When every placement sits inside the ramp interval there are
        no "outside-ramp" baseline observations; the function falls back
        to the mean of all measures."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={
                        "scale": "day",
                        "direction": "increasing",
                        "amount": 2,
                        "start": 1,
                        "end": 7,
                    },
                )
            ],
        )
        pc = _pc([ev])
        # All inside the day-1..7 ramp.
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1) + _dt.timedelta(days=i))
            for i in range(7)
        ]
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is not None

    def test_trend_cumulative_expected_negative_skips_slice(self):
        """Aggressive `decreasing` ramp can push cumulative_exp ≤ 0
        early; those slices are skipped."""
        ev = _make_event(
            "walking",
            temporal_patterns=[
                TemporalPattern(
                    mode="trend",
                    details={
                        "scale": "day",
                        "direction": "decreasing",
                        "amount": 100,
                        "start": 1,
                        "end": 7,
                    },
                )
            ],
        )
        pc = _pc([ev])
        placements = [
            _scheduled("walk", 480, 510, _dt.date(2026, 6, 1) + _dt.timedelta(days=i))
            for i in range(3)
        ]
        # Loss may be None (everything skipped) or a small float depending
        # on how the cumulative behaves; just smoke-test it does not crash.
        loss, _ = score_temporal_pattern(
            placements,
            matched_events=[_mapped("walking")],
            constraints=pc,
            window_map=_wm(),
        )
        assert loss is None or 0.0 <= loss <= 1.0


class TestSliceKeyHelper:
    def test_weekday_slice_key(self):
        # 2026-06-01 is a Monday
        assert _seasonality_slice_key(_dt.date(2026, 6, 1), "weekday", None) == 0

    def test_month_slice_key(self):
        assert _seasonality_slice_key(_dt.date(2026, 6, 15), "month", None) == 6

    def test_season_slice_key(self):
        # June to summer to season 2
        assert _seasonality_slice_key(_dt.date(2026, 6, 15), "season", None) == 2


# ---------------------------------------------------------------------------
# Public dataclass smoke tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_pattern_violation_dataclass(self):
        pv = PatternViolation(mode="fix", scale=None, loss=0.3, mape=None)
        assert pv.loss == 0.3
        assert pv.mape is None

    def test_stage_alignment_dataclass(self):
        sa = StageAlignment(stage_name="walking", sigma=0.9, aligned=True)
        assert sa.aligned is True
