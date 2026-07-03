"""Tests for the per-persona context chart renderers."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scripts.persona.analytics.charts import ALL_CHART_KINDS
from src.scripts.persona.analytics.context_charts import (
    _daily_mean_and_ci,
    _members_for_category,
    _palette,
    _resolve_member,
    _resolve_persona,
    _to_minutes,
    _valid_scale,
    context_vmax_for_member,
    context_vmin_for_member,
    contexts_by_day,
    daily_context_aggregate,
    per_member_context_persona_daily,
    plot_context_member_heatmap,
    plot_per_member_context_lines,
)
from src.scripts.persona.context.schema import ContextEpisode
from src.scripts.persona.domain.event import EventInstance
from src.scripts.persona.domain.schedule import DaySchedule, PersonSchedule

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MONDAY = datetime.date(2026, 5, 4)


def _episode(
    name: str,
    category: str,
    date: datetime.date,
    start: int = 480,
    end: int = 540,
) -> ContextEpisode:
    return ContextEpisode(
        name=name,
        category=category,
        date=date,
        start_minutes=start,
        end_minutes=end,
    )


def _day(idx: int, date: datetime.date) -> DaySchedule:
    return DaySchedule(
        day_index=idx,
        date=date,
        weekday=date.strftime("%a"),
        events={"lunch": [EventInstance(event_name="lunch", start=720, duration=30)]},
        spillovers=[],
    )


def _schedule(
    person_id: str,
    contexts: list[ContextEpisode],
    *,
    persona_id: str = "alice",
    horizon_days: int = 7,
) -> PersonSchedule:
    return PersonSchedule(
        person_id=person_id,
        persona_id=persona_id,
        person_seed=7,
        days=[
            _day(i, _MONDAY + datetime.timedelta(days=i)) for i in range(horizon_days)
        ],
        contexts=contexts,
    )


# ---------------------------------------------------------------------------
# top-level wiring
# ---------------------------------------------------------------------------


def test_context_kinds_registered() -> None:
    for kind in ("context-lines", "context-heatmap"):
        assert kind in ALL_CHART_KINDS


def test_context_calendar_kind_dropped() -> None:
    assert "context-calendar" not in ALL_CHART_KINDS


# ---------------------------------------------------------------------------
# pure-data helpers
# ---------------------------------------------------------------------------


class TestContextsByDay:
    def test_groups_per_date_category_member(self) -> None:
        d = _MONDAY
        e1 = _episode("happy", "mood_emotion", d, 480, 540)
        e2 = _episode("happy", "mood_emotion", d, 700, 760)
        e3 = _episode("home", "location", d, 0, 1440)
        grouped = contexts_by_day([e1, e2, e3])
        assert grouped[d]["mood_emotion"]["happy"] == [e1, e2]
        assert grouped[d]["location"]["home"] == [e3]


class TestMembersForCategory:
    def test_returns_sorted_unique_names(self) -> None:
        eps = [
            _episode("zebra", "mood_emotion", _MONDAY),
            _episode("apple", "mood_emotion", _MONDAY),
            _episode("apple", "mood_emotion", _MONDAY),
            _episode("home", "location", _MONDAY),
        ]
        assert _members_for_category(eps, "mood_emotion") == ["apple", "zebra"]

    def test_unknown_category_returns_empty(self) -> None:
        eps = [_episode("happy", "mood_emotion", _MONDAY)]
        assert _members_for_category(eps, "no_such") == []


class TestDailyContextAggregate:
    def test_empty_schedules_returns_empty_series(self) -> None:
        out = daily_context_aggregate(
            [], category="mood_emotion", member="happy", metric="count"
        )
        assert out.empty

    def test_empty_days_returns_empty_series(self) -> None:
        sched = PersonSchedule(person_id="p", persona_id="a", person_seed=1, days=[])
        out = daily_context_aggregate(
            [sched], category="mood_emotion", member="happy", metric="count"
        )
        assert out.empty

    def test_count_aggregate_mean_across_persons(self) -> None:
        p1 = _schedule(
            "p1",
            [
                _episode("happy", "mood_emotion", _MONDAY, 480, 540),
                _episode("happy", "mood_emotion", _MONDAY, 700, 760),
            ],
        )
        p2 = _schedule("p2", [_episode("happy", "mood_emotion", _MONDAY, 480, 540)])
        out = daily_context_aggregate(
            [p1, p2], category="mood_emotion", member="happy", metric="count"
        )
        # 2 + 1 episodes / 2 persons = 1.5 on Monday; 0 elsewhere.
        assert out.loc[pd.Timestamp(_MONDAY)] == pytest.approx(1.5)
        assert out.sum() == pytest.approx(1.5)

    def test_duration_metric_sums_minutes(self) -> None:
        p1 = _schedule(
            "p1",
            [_episode("happy", "mood_emotion", _MONDAY, start=480, end=540)],
        )
        out = daily_context_aggregate(
            [p1], category="mood_emotion", member="happy", metric="duration"
        )
        assert out.loc[pd.Timestamp(_MONDAY)] == pytest.approx(60.0)

    def test_sum_aggregate_pools_persons(self) -> None:
        p1 = _schedule("p1", [_episode("happy", "mood_emotion", _MONDAY, 480, 540)])
        p2 = _schedule(
            "p2",
            [
                _episode("happy", "mood_emotion", _MONDAY, 480, 540),
                _episode("happy", "mood_emotion", _MONDAY, 540, 600),
            ],
        )
        out = daily_context_aggregate(
            [p1, p2],
            category="mood_emotion",
            member="happy",
            metric="count",
            aggregate="sum",
        )
        assert out.loc[pd.Timestamp(_MONDAY)] == pytest.approx(3.0)

    def test_other_categories_ignored(self) -> None:
        sched = _schedule(
            "p1",
            [
                _episode("happy", "mood_emotion", _MONDAY),
                _episode("home", "location", _MONDAY),
            ],
        )
        out = daily_context_aggregate(
            [sched], category="mood_emotion", member="happy", metric="count"
        )
        assert out.loc[pd.Timestamp(_MONDAY)] == pytest.approx(1.0)

    def test_episodes_outside_horizon_ignored(self) -> None:
        sched = _schedule(
            "p1",
            [_episode("happy", "mood_emotion", datetime.date(2050, 1, 1))],
        )
        out = daily_context_aggregate(
            [sched], category="mood_emotion", member="happy", metric="count"
        )
        assert out.sum() == pytest.approx(0.0)

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric must be"):
            daily_context_aggregate([], category="x", member="y", metric="bogus")

    def test_invalid_aggregate_raises(self) -> None:
        with pytest.raises(ValueError, match="aggregate must be"):
            daily_context_aggregate(
                [], category="x", member="y", metric="count", aggregate="bogus"
            )


class TestPerMemberContextPersonaDaily:
    def test_empty_returns_zero_shaped_matrix(self) -> None:
        out = per_member_context_persona_daily(
            [], category="m", member="h", metric="count"
        )
        assert out.shape == (0, 0)

    def test_empty_days_within_schedule(self) -> None:
        sched = PersonSchedule(person_id="p", persona_id="a", person_seed=1, days=[])
        out = per_member_context_persona_daily(
            [sched], category="m", member="h", metric="count"
        )
        assert out.shape == (0, 1)

    def test_counts_only_matching_episodes(self) -> None:
        p1 = _schedule(
            "p1",
            [
                _episode("happy", "mood_emotion", _MONDAY),
                _episode("sad", "mood_emotion", _MONDAY),
            ],
            horizon_days=3,
        )
        out = per_member_context_persona_daily(
            [p1], category="mood_emotion", member="happy", metric="count"
        )
        assert out.shape == (3, 1)
        assert out[0, 0] == pytest.approx(1.0)
        assert out[1:, 0].sum() == pytest.approx(0.0)

    def test_duration_sums_minutes(self) -> None:
        p1 = _schedule(
            "p1",
            [_episode("happy", "mood_emotion", _MONDAY, start=480, end=540)],
            horizon_days=2,
        )
        out = per_member_context_persona_daily(
            [p1], category="mood_emotion", member="happy", metric="duration"
        )
        assert out[0, 0] == pytest.approx(60.0)

    def test_episode_outside_schedule_dates_dropped(self) -> None:
        p1 = _schedule(
            "p1",
            [_episode("happy", "mood_emotion", datetime.date(2050, 1, 1))],
        )
        out = per_member_context_persona_daily(
            [p1], category="mood_emotion", member="happy", metric="count"
        )
        assert out.sum() == pytest.approx(0.0)

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric must be"):
            per_member_context_persona_daily(
                [], category="m", member="h", metric="weird"
            )


# ---------------------------------------------------------------------------
# vmin / vmax helpers
# ---------------------------------------------------------------------------


class _StubMember:
    def __init__(
        self,
        ep_min: float,
        ep_max: float,
        dur_min: float,
        dur_max: float,
        unit: str = "minutes",
    ):
        from types import SimpleNamespace

        self.total_event_episodes = SimpleNamespace(min=ep_min, max=ep_max)
        self.total_event_duration = SimpleNamespace(min=dur_min, max=dur_max, unit=unit)


class _StubCategory:
    def __init__(self, members: dict):
        self.members = members


class _StubPersona:
    def __init__(self, id_: str, contexts: dict):
        self.id = id_
        self.contexts = contexts


class _StubPersonaConfig:
    def __init__(self, personas: list):
        self.personas = personas


def _stub_config_with_member(
    persona_id: str, category: str, member: str, **kw
) -> _StubPersonaConfig:
    return _StubPersonaConfig(
        personas=[
            _StubPersona(
                persona_id,
                {category: _StubCategory({member: _StubMember(**kw)})},
            )
        ]
    )


class TestVminVmaxForMember:
    def test_count_metric_reads_episode_bounds(self) -> None:
        cfg = _stub_config_with_member(
            "alice", "mood_emotion", "happy", ep_min=1, ep_max=4, dur_min=0, dur_max=0
        )
        assert (
            context_vmin_for_member(cfg, "alice", "mood_emotion", "happy", "count")
            == 1.0
        )
        assert (
            context_vmax_for_member(cfg, "alice", "mood_emotion", "happy", "count")
            == 4.0
        )

    def test_duration_metric_converts_hours(self) -> None:
        cfg = _stub_config_with_member(
            "alice",
            "energy_state",
            "energetic",
            ep_min=0,
            ep_max=0,
            dur_min=1,
            dur_max=6,
            unit="hours",
        )
        assert (
            context_vmin_for_member(
                cfg, "alice", "energy_state", "energetic", "duration"
            )
            == 60.0
        )
        assert (
            context_vmax_for_member(
                cfg, "alice", "energy_state", "energetic", "duration"
            )
            == 360.0
        )

    def test_duration_metric_minutes_unit_passthrough(self) -> None:
        cfg = _stub_config_with_member(
            "alice",
            "stress",
            "anxious",
            ep_min=0,
            ep_max=0,
            dur_min=10,
            dur_max=90,
            unit="minutes",
        )
        assert (
            context_vmin_for_member(cfg, "alice", "stress", "anxious", "duration")
            == 10.0
        )

    def test_unknown_persona_returns_none(self) -> None:
        cfg = _stub_config_with_member(
            "alice", "m", "h", ep_min=0, ep_max=1, dur_min=0, dur_max=0
        )
        assert context_vmin_for_member(cfg, "bob", "m", "h", "count") is None
        assert context_vmax_for_member(cfg, "bob", "m", "h", "count") is None

    def test_unknown_category_returns_none(self) -> None:
        cfg = _stub_config_with_member(
            "alice", "m", "h", ep_min=0, ep_max=1, dur_min=0, dur_max=0
        )
        assert context_vmin_for_member(cfg, "alice", "no_cat", "h", "count") is None

    def test_unknown_member_returns_none(self) -> None:
        cfg = _stub_config_with_member(
            "alice", "m", "h", ep_min=0, ep_max=1, dur_min=0, dur_max=0
        )
        assert context_vmin_for_member(cfg, "alice", "m", "no_member", "count") is None

    def test_none_persona_config_returns_none(self) -> None:
        assert context_vmin_for_member(None, "a", "c", "m", "count") is None
        assert context_vmax_for_member(None, "a", "c", "m", "count") is None

    def test_unknown_metric_returns_none(self) -> None:
        cfg = _stub_config_with_member(
            "alice", "m", "h", ep_min=0, ep_max=1, dur_min=0, dur_max=0
        )
        assert context_vmin_for_member(cfg, "alice", "m", "h", "bogus") is None
        assert context_vmax_for_member(cfg, "alice", "m", "h", "bogus") is None

    def test_resolve_persona_returns_none_when_missing(self) -> None:
        assert _resolve_persona(None, "alice") is None
        cfg = _stub_config_with_member(
            "alice", "m", "h", ep_min=0, ep_max=1, dur_min=0, dur_max=0
        )
        assert _resolve_persona(cfg, "alice").id == "alice"

    def test_resolve_member_returns_none_when_persona_missing(self) -> None:
        assert _resolve_member(None, "alice", "m", "h") is None


# ---------------------------------------------------------------------------
# small private helpers
# ---------------------------------------------------------------------------


class TestSmallHelpers:
    def test_to_minutes_passthrough(self) -> None:
        assert _to_minutes(15, "minutes") == 15.0

    def test_to_minutes_hours_to_minutes(self) -> None:
        assert _to_minutes(2, "hours") == 120.0

    def test_valid_scale_true(self) -> None:
        assert _valid_scale(0.0, 10.0) is True

    def test_valid_scale_false_when_missing(self) -> None:
        assert _valid_scale(None, 10.0) is False
        assert _valid_scale(0.0, None) is False

    def test_valid_scale_false_when_degenerate(self) -> None:
        assert _valid_scale(5.0, 5.0) is False
        assert _valid_scale(5.0, 1.0) is False

    def test_palette_returns_at_least_one(self) -> None:
        assert len(_palette(0)) == 1
        assert len(_palette(3)) == 3

    def test_daily_mean_and_ci_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            _daily_mean_and_ci(np.array([1.0, 2.0]))

    def test_daily_mean_and_ci_empty_days(self) -> None:
        mean, lo, hi = _daily_mean_and_ci(np.zeros((0, 3)))
        assert mean.shape == lo.shape == hi.shape == (0,)

    def test_daily_mean_and_ci_zero_persons(self) -> None:
        mean, lo, hi = _daily_mean_and_ci(np.zeros((3, 0)))
        assert np.array_equal(mean, np.zeros(3))
        assert np.array_equal(lo, np.zeros(3))
        assert np.array_equal(hi, np.zeros(3))

    def test_daily_mean_and_ci_single_person_collapses(self) -> None:
        mean, lo, hi = _daily_mean_and_ci(np.array([[1.0], [2.0]]))
        assert np.array_equal(mean, lo)
        assert np.array_equal(mean, hi)

    def test_daily_mean_and_ci_band_widens_with_variance(self) -> None:
        values = np.array([[1.0, 1.0], [0.0, 2.0]])
        mean, lo, hi = _daily_mean_and_ci(values)
        # Row 0: zero variance so the band collapses to the mean.
        assert mean[0] == pytest.approx(1.0)
        assert lo[0] == pytest.approx(1.0)
        assert hi[0] == pytest.approx(1.0)
        # Row 1: non-zero variance so the band straddles the mean.
        assert mean[1] == pytest.approx(1.0)
        assert lo[1] < mean[1] < hi[1]


# ---------------------------------------------------------------------------
# plot_context_member_heatmap
# ---------------------------------------------------------------------------


class TestPlotContextMemberHeatmap:
    def test_returns_none_when_no_schedules(self, tmp_path: Path) -> None:
        out = plot_context_member_heatmap(
            "alice",
            [],
            category="mood_emotion",
            member="happy",
            out_dir=tmp_path,
            metric="count",
        )
        assert out is None

    def test_writes_count_png(self, tmp_path: Path) -> None:
        sched = _schedule("p1", [_episode("happy", "mood_emotion", _MONDAY)])
        out = plot_context_member_heatmap(
            "alice",
            [sched],
            category="mood_emotion",
            member="happy",
            out_dir=tmp_path,
            metric="count",
        )
        assert out is not None and out.exists()
        assert out.name == "heatmap_context_alice_mood_emotion_happy_count_mean.png"

    def test_writes_duration_png_with_config_scale(self, tmp_path: Path) -> None:
        sched = _schedule(
            "p1",
            [_episode("happy", "mood_emotion", _MONDAY, start=480, end=540)],
        )
        out = plot_context_member_heatmap(
            "alice",
            [sched],
            category="mood_emotion",
            member="happy",
            out_dir=tmp_path,
            metric="duration",
            vmin=0.0,
            vmax=120.0,
        )
        assert out is not None and out.exists()
        assert out.name == "heatmap_context_alice_mood_emotion_happy_duration_mean.png"

    def test_invalid_metric_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="metric must be"):
            plot_context_member_heatmap(
                "alice",
                [],
                category="m",
                member="h",
                out_dir=tmp_path,
                metric="weird",
            )


# ---------------------------------------------------------------------------
# plot_per_member_context_lines
# ---------------------------------------------------------------------------


class TestPlotPerMemberContextLines:
    def test_returns_none_when_no_persona(self, tmp_path: Path) -> None:
        out = plot_per_member_context_lines(
            "mood_emotion", "happy", "count", {}, tmp_path
        )
        assert out is None

    def test_returns_none_when_all_zero_signal(self, tmp_path: Path) -> None:
        # Schedule has no episodes of the requested member so the matrix is all-zero.
        sched = _schedule("p1", [])
        out = plot_per_member_context_lines(
            "mood_emotion",
            "happy",
            "count",
            {"alice": [sched]},
            tmp_path,
        )
        assert out is None

    def test_skips_persona_with_empty_schedule_list(self, tmp_path: Path) -> None:
        sched = _schedule("p1", [_episode("happy", "mood_emotion", _MONDAY)])
        out = plot_per_member_context_lines(
            "mood_emotion",
            "happy",
            "count",
            {"alice": [sched], "bob": []},
            tmp_path,
        )
        assert out is not None and out.exists()

    def test_skips_persona_with_zero_horizon_days(self, tmp_path: Path) -> None:
        """A schedule whose days list is empty produces a zero-day matrix and is skipped."""
        with_signal = _schedule("p1", [_episode("happy", "mood_emotion", _MONDAY)])
        empty_days = PersonSchedule(
            person_id="p2",
            persona_id="bob",
            person_seed=1,
            days=[],
        )
        out = plot_per_member_context_lines(
            "mood_emotion",
            "happy",
            "count",
            {"alice": [with_signal], "bob": [empty_days]},
            tmp_path,
        )
        # bob is skipped (no days); alice provides the signal so the PNG is written.
        assert out is not None and out.exists()

    def test_writes_count_png(self, tmp_path: Path) -> None:
        sched_a = _schedule(
            "p1",
            [_episode("happy", "mood_emotion", _MONDAY)],
            persona_id="alice",
        )
        sched_b = _schedule(
            "p2",
            [
                _episode("happy", "mood_emotion", _MONDAY),
                _episode("happy", "mood_emotion", _MONDAY, start=600, end=660),
            ],
            persona_id="bob",
        )
        out = plot_per_member_context_lines(
            "mood_emotion",
            "happy",
            "count",
            {"alice": [sched_a], "bob": [sched_b]},
            tmp_path,
        )
        assert out is not None and out.exists()
        assert out.name == "lines_context_mood_emotion_happy_count.png"

    def test_writes_duration_png_with_unit(self, tmp_path: Path) -> None:
        sched = _schedule(
            "p1",
            [_episode("happy", "mood_emotion", _MONDAY, start=480, end=540)],
        )
        out = plot_per_member_context_lines(
            "mood_emotion",
            "happy",
            "duration",
            {"alice": [sched]},
            tmp_path,
            duration_unit="hours",
        )
        assert out is not None and out.exists()
        assert out.name == "lines_context_mood_emotion_happy_duration.png"

    def test_invalid_metric_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="metric must be"):
            plot_per_member_context_lines("m", "h", "bogus", {"a": []}, tmp_path)
