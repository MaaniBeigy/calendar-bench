"""Tests for the PTIME SVM-rank preference learning path."""

from __future__ import annotations

import datetime
import random

import pytest

from src.scripts.persona.config.schema import DailyWindow, WindowRange
from src.scripts.scenarios.augmentation.ptime import (
    PTimeAugmenter,
    _event_to_pseudo_task,
    _feature_vector_at,
    _pair_keys,
    _shifted_negative_starts,
    blend_importance,
    blend_interaction,
    extract_pairwise_examples,
    fit_learned_coefficients,
)
from src.scripts.scenarios.config.schema import (
    _PTIME_CRITERIA,
    AugmentationConfig,
    PTimeConfig,
    PTimeLearningConfig,
)
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.metrics.allen import _build_rule_index
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)


def _wide_window() -> DailyWindow:
    return DailyWindow(wake_minutes=0, sleep_minutes=1440)


# ---------------------------------------------------------------------------
# _event_to_pseudo_task
# ---------------------------------------------------------------------------


def test_event_to_pseudo_task_preserves_duration_and_intensity():
    ev = make_event(start_minutes=420, end_minutes=480, intensity=3)
    t = _event_to_pseudo_task(ev)
    assert t.duration_min == 60 and t.duration_max == 60
    assert t.intensity == 3
    assert t.label == ev.label


def test_event_to_pseudo_task_handles_zero_span():
    # Defensive: degenerate events still produce a valid task.
    ev = make_event(start_minutes=420, end_minutes=420)
    t = _event_to_pseudo_task(ev)
    assert t.duration_min == 1


# ---------------------------------------------------------------------------
# _pair_keys / _feature_vector_at
# ---------------------------------------------------------------------------


def test_pair_keys_are_sorted_and_complete():
    keys = _pair_keys()
    n = len(_PTIME_CRITERIA)
    assert len(keys) == n * (n - 1) // 2
    assert all(k == "|".join(sorted(k.split("|"))) for k in keys)


def test_feature_vector_has_singletons_then_pairwise_mins():
    ev = make_event(start_minutes=420, end_minutes=450)
    pseudo = _event_to_pseudo_task(ev)
    rule_index = _build_rule_index([])
    z = _feature_vector_at(
        pseudo,
        420,
        450,
        DATE,
        events_by_date={},
        time_windows={},
        rule_index=rule_index,
        duration_preference="minimum",
    )
    n = len(_PTIME_CRITERIA)
    assert len(z) == n + len(_pair_keys())
    assert all(0.0 <= v <= 1.0 for v in z)
    # The pairwise block holds the min of the two singleton utilities.
    singles = z[:n]
    expected = [min(singles[i], singles[j]) for i in range(n) for j in range(i + 1, n)]
    assert z[n:] == expected


# ---------------------------------------------------------------------------
# _shifted_negative_starts
# ---------------------------------------------------------------------------


def test_shifted_negatives_clip_to_window_and_dedupe():
    ev = make_event(start_minutes=400, end_minutes=430)
    rng = random.Random(7)
    shifted = _shifted_negative_starts(
        ev, window_start=0, window_end=1440, shift_minutes=60, n_samples=3, rng=rng
    )
    assert len(shifted) <= 3
    assert ev.start_minutes not in shifted
    for s in shifted:
        assert 0 <= s <= 1440 - 30


def test_shifted_negatives_inner_guards_with_tiny_shift_window():
    # `shift_minutes=1` forces frequent `delta=0` skips and duplicate clips;
    # the loop still terminates and returns unique non-zero shifts only.
    ev = make_event(start_minutes=600, end_minutes=630)
    rng = random.Random(0)
    shifted = _shifted_negative_starts(
        ev, window_start=0, window_end=1440, shift_minutes=1, n_samples=5, rng=rng
    )
    # The +/-1-minute neighbourhood has at most two unique non-zero shifts.
    assert all(s != ev.start_minutes for s in shifted)
    assert len(set(shifted)) == len(shifted)


def test_shifted_negatives_empty_when_event_fills_window():
    # Daily window can only fit the event itself; no alternatives exist.
    ev = make_event(start_minutes=0, end_minutes=60)
    rng = random.Random(0)
    shifted = _shifted_negative_starts(
        ev, window_start=0, window_end=60, shift_minutes=120, n_samples=3, rng=rng
    )
    assert shifted == []


# ---------------------------------------------------------------------------
# extract_pairwise_examples
# ---------------------------------------------------------------------------


def test_extract_pairwise_examples_returns_pairs_when_features_differ():
    # Event sits inside the configured `time_window`; shifted negatives
    # fall outside so the `time` criterion differs.
    tw = {"morning": WindowRange(start=420, end=480)}
    ev = make_event(start_minutes=420, end_minutes=450)
    cal = CalendarTrace(person_id="p1", events=[ev])
    rule_index = _build_rule_index([])
    rng = random.Random(1)
    pairs = extract_pairwise_examples(
        cal,
        events_by_date={DATE: [ev]},
        time_windows=tw,
        rule_index=rule_index,
        window_start=0,
        window_end=1440,
        duration_preference="minimum",
        negative_samples_per_event=3,
        negative_shift_minutes=120,
        rng=rng,
    )
    assert pairs, "expected at least one (positive, negative) pair"
    for pos, neg in pairs:
        # Positive lies inside the window (time-utility = 1.0); negatives
        # shift outside, so their time-utility is at most the positive's.
        assert pos[0] >= neg[0]


def test_extract_pairwise_examples_empty_when_no_events():
    cal = CalendarTrace(person_id="p1", events=[])
    rule_index = _build_rule_index([])
    pairs = extract_pairwise_examples(
        cal,
        events_by_date={},
        time_windows={},
        rule_index=rule_index,
        window_start=0,
        window_end=1440,
        duration_preference="minimum",
        negative_samples_per_event=3,
        negative_shift_minutes=60,
        rng=random.Random(0),
    )
    assert pairs == []


# ---------------------------------------------------------------------------
# fit_learned_coefficients
# ---------------------------------------------------------------------------


def test_fit_returns_none_when_no_pairs():
    assert fit_learned_coefficients([], regularization=1.0, seed=0) is None


def test_fit_learns_dominant_importance():
    # Positive scores 1.0 on the first singleton, negative on the second;
    # SVM should mass importance weight on the first.
    dim = len(_PTIME_CRITERIA) + len(_pair_keys())
    pos = [0.0] * dim
    pos[0] = 1.0
    neg = [0.0] * dim
    neg[1] = 1.0
    learned = fit_learned_coefficients([(pos, neg)], regularization=1.0, seed=0)
    assert learned is not None
    importance, _interaction = learned
    first, second = _PTIME_CRITERIA[0], _PTIME_CRITERIA[1]
    assert importance[first] > importance[second]
    assert importance[first] == pytest.approx(1.0)
    assert sum(importance.values()) == pytest.approx(1.0)


def test_fit_clips_negative_importances_to_zero():
    dim = len(_PTIME_CRITERIA) + len(_pair_keys())
    pos = [0.0] * dim
    pos[0] = 1.0
    neg = [0.0] * dim
    neg[1] = 1.0
    importance, _ = fit_learned_coefficients([(pos, neg)], regularization=1.0, seed=0)
    assert all(v >= 0.0 for v in importance.values())


def test_fit_learns_pairwise_interaction_rescaled():
    # Positive favours the first pairwise feature, negative the second; the
    # learned interactions land in [-1, 1] with the first the larger.
    n = len(_PTIME_CRITERIA)
    keys = _pair_keys()
    dim = n + len(keys)
    pos = [0.0] * dim
    pos[n] = 1.0
    neg = [0.0] * dim
    neg[n + 1] = 1.0
    _imp, interaction = fit_learned_coefficients(
        [(pos, neg)], regularization=1.0, seed=0
    )
    assert all(-1.0 <= v <= 1.0 for v in interaction.values())
    assert interaction[keys[0]] > interaction[keys[1]]


def test_fit_degenerate_pairs_return_zero_coefficients():
    dim = len(_PTIME_CRITERIA) + len(_pair_keys())
    z = [0.5] * dim
    learned = fit_learned_coefficients([(z, z), (z, z)], regularization=1.0, seed=0)
    importance, interaction = learned
    assert importance == {c: 0.0 for c in _PTIME_CRITERIA}
    assert interaction == {k: 0.0 for k in _pair_keys()}


# ---------------------------------------------------------------------------
# blend_importance
# ---------------------------------------------------------------------------


def test_blend_returns_elicited_when_learned_is_none():
    elicited = {c: 1 / len(_PTIME_CRITERIA) for c in _PTIME_CRITERIA}
    assert blend_importance(elicited, None, 0.6) == elicited


def test_blend_interpolates_with_alpha():
    elicited = {c: 0.0 for c in _PTIME_CRITERIA}
    elicited["time"] = 1.0
    learned = {c: 0.0 for c in _PTIME_CRITERIA}
    learned["duration"] = 1.0
    out = blend_importance(elicited, learned, alpha=0.7)
    assert out["time"] == pytest.approx(0.7)
    assert out["duration"] == pytest.approx(0.3)


def test_blend_alpha_zero_is_pure_learned():
    elicited = {c: 1.0 for c in _PTIME_CRITERIA}
    learned = {c: 0.5 for c in _PTIME_CRITERIA}
    learned["time"] = 0.9
    out = blend_importance(elicited, learned, alpha=0.0)
    assert out == learned


def test_blend_alpha_one_is_pure_elicited():
    elicited = {c: 0.4 for c in _PTIME_CRITERIA}
    learned = {c: 0.0 for c in _PTIME_CRITERIA}
    learned["time"] = 1.0
    out = blend_importance(elicited, learned, alpha=1.0)
    assert out == elicited


# ---------------------------------------------------------------------------
# blend_interaction
# ---------------------------------------------------------------------------


def test_blend_interaction_returns_elicited_when_learned_is_none():
    elicited = {k: 0.1 for k in _pair_keys()}
    assert blend_interaction(elicited, None, 0.6) == elicited


def test_blend_interaction_interpolates_with_alpha():
    keys = _pair_keys()
    elicited = {k: 0.0 for k in keys}
    elicited[keys[0]] = 1.0
    learned = {k: 0.0 for k in keys}
    learned[keys[1]] = -1.0
    out = blend_interaction(elicited, learned, alpha=0.7)
    assert out[keys[0]] == pytest.approx(0.7)
    assert out[keys[1]] == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# End-to-end PTimeAugmenter with learning enabled
# ---------------------------------------------------------------------------


def test_augment_with_learning_enabled_still_places_task():
    # Calendar contains only morning events; SVM should weight `time`.
    morning_events = [
        make_event(label="breakfast", start_minutes=420, end_minutes=480, date=DATE),
        make_event(
            label="standup",
            start_minutes=420,
            end_minutes=450,
            date=DATE + datetime.timedelta(days=1),
        ),
    ]
    aug = PTimeAugmenter(
        time_windows={"morning": WindowRange(start=420, end=540)},
        daily_window=_wide_window(),
        seed=0,
    )
    trace = CalendarTrace(person_id="p1", events=morning_events)
    task = make_task(label="walk", duration_min=30, duration_max=30)
    cfg = AugmentationConfig(
        method="ptime",
        repeat_per_week=False,
        ptime=PTimeConfig(
            importance={c: 0.0 for c in _PTIME_CRITERIA} | {"time": 1.0},
            learning=PTimeLearningConfig(enabled=True, alpha=0.5),
        ),
    )
    sol = aug.augment(trace, [task], cfg)
    assert len(sol.scheduled) == 1
    # Time criterion (elicited and learned) lands the task in the morning band.
    st = sol.scheduled[0]
    assert 420 <= st.start_minutes < 540


def test_augment_with_learning_no_events_falls_back_to_elicited():
    # Empty calendar returns no pairs; blend keeps elicited unchanged.
    aug = PTimeAugmenter(daily_window=_wide_window(), seed=0)
    trace = CalendarTrace(person_id="p1", events=[])
    task = make_task(label="walk", duration_min=30, duration_max=30)
    cfg = AugmentationConfig(
        method="ptime",
        repeat_per_week=False,
        ptime=PTimeConfig(
            learning=PTimeLearningConfig(enabled=True, alpha=0.6),
        ),
    )
    sol = aug.augment(trace, [task], cfg, horizon=(DATE, 7))
    assert len(sol.scheduled) == 1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_learning_config_defaults():
    lc = PTimeLearningConfig()
    assert lc.enabled is False
    assert lc.alpha == 0.6
    assert lc.negative_samples_per_event == 2
    assert lc.negative_shift_minutes == 120
    assert lc.regularization == 1.0


def test_learning_config_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        PTimeLearningConfig(alpha=1.5)
    with pytest.raises(ValueError):
        PTimeLearningConfig(alpha=-0.1)


def test_learning_config_rejects_non_positive_regularization():
    with pytest.raises(ValueError):
        PTimeLearningConfig(regularization=0.0)
