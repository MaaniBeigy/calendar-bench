"""Unit tests for PTIME candidate enumeration and best-pick logic."""

from __future__ import annotations

import datetime
import random

import pytest

from src.scripts.persona.config.schema import WindowRange
from src.scripts.scenarios.augmentation.ptime import (
    Candidate,
    _candidate_durations,
    _candidate_starts_for_gap,
    _pick_best,
    enumerate_candidates,
    merge_importance,
    merge_interaction,
)
from src.scripts.scenarios.metrics.allen import _build_rule_index
from tests.unit.scenarios.conftest import make_task

DATE = datetime.date(2026, 5, 4)


def test_candidate_durations_span_the_band():
    t = make_task(duration_min=30, duration_max=60)
    assert _candidate_durations(t) == [30, 45, 60]


def test_candidate_durations_collapse_when_band_is_a_point():
    t = make_task(duration_min=30, duration_max=30)
    assert _candidate_durations(t) == [30]


def test_candidate_starts_earliest_only_returns_one():
    starts = _candidate_starts_for_gap(
        (420, 600), duration=30, time_windows={}, strategy="earliest_only"
    )
    assert starts == [420]


def test_candidate_starts_windows_adds_window_aligned():
    tw = {"a": WindowRange(start=480, end=540), "b": WindowRange(start=550, end=600)}
    starts = _candidate_starts_for_gap(
        (420, 600), duration=30, time_windows=tw, strategy="windows"
    )
    assert starts == [420, 480, 550]


def test_candidate_starts_windows_and_center_adds_center():
    starts = _candidate_starts_for_gap(
        (420, 600), duration=30, time_windows={}, strategy="windows_and_center"
    )
    # gap 420..600, duration 30; center = (420 + 570) // 2 = 495.
    assert 495 in starts
    assert starts[0] == 420


def test_candidate_starts_windows_and_center_skips_duplicate_center():
    # gap [0, 30], duration 30; center = 0 = gap_start (already present).
    starts = _candidate_starts_for_gap(
        (0, 30), duration=30, time_windows={}, strategy="windows_and_center"
    )
    assert starts == [0]


def test_candidate_starts_skips_windows_without_room():
    tw = {"too_late": WindowRange(start=590, end=600)}
    starts = _candidate_starts_for_gap(
        (420, 600), duration=30, time_windows=tw, strategy="windows"
    )
    # 590 + 30 = 620 > gap_end 600; window-aligned start dropped.
    assert starts == [420]


def test_candidate_starts_empty_when_duration_exceeds_gap():
    starts = _candidate_starts_for_gap(
        (420, 440), duration=30, time_windows={}, strategy="windows"
    )
    assert starts == []


def test_enumerate_candidates_skips_reused_gaps_unless_allowed():
    task = make_task(duration_min=30, duration_max=60)
    # No events => one base gap covering the full day.
    used = {(DATE, 0, 1440)}
    rule_index = _build_rule_index([])
    importance = merge_importance({})
    interaction = merge_interaction({})

    no_reuse = enumerate_candidates(
        task,
        [DATE],
        events_by_date={},
        placed_by_date={},
        used_base_gaps=used,
        time_windows={},
        rule_index=rule_index,
        window_start=0,
        window_end=1440,
        importance=importance,
        interaction=interaction,
        duration_preference="minimum",
        strategy="windows",
        allow_reuse=False,
    )
    assert no_reuse == []

    with_reuse = enumerate_candidates(
        task,
        [DATE],
        events_by_date={},
        placed_by_date={},
        used_base_gaps=used,
        time_windows={},
        rule_index=rule_index,
        window_start=0,
        window_end=1440,
        importance=importance,
        interaction=interaction,
        duration_preference="minimum",
        strategy="windows",
        allow_reuse=True,
    )
    assert len(with_reuse) >= 1
    assert all(c.base_gap == (0, 1440) for c in with_reuse)


def test_enumerate_candidates_skips_starts_that_overflow_gap():
    # Reuse-anchor at 450 plus duration 30 overflows the `[420, 470)` gap;
    # the candidate is skipped.
    task = make_task(duration_min=30, duration_max=30)
    rule_index = _build_rule_index([])
    placed = {DATE: [(420, 450)]}
    used = {(DATE, 420, 470)}  # mark base as reused to enter the allow_reuse path

    # Build a `[420, 470)` base gap by wrapping events around it.
    from tests.unit.scenarios.conftest import make_event

    pre = make_event(label="pre", start_minutes=0, end_minutes=420, date=DATE)
    post = make_event(label="post", start_minutes=470, end_minutes=1440, date=DATE)
    ev_by_date = {DATE: [pre, post]}

    cands = enumerate_candidates(
        task,
        [DATE],
        events_by_date=ev_by_date,
        placed_by_date=placed,
        used_base_gaps=used,
        time_windows={},
        rule_index=rule_index,
        window_start=0,
        window_end=1440,
        importance=merge_importance({}),
        interaction=merge_interaction({}),
        duration_preference="minimum",
        strategy="windows",
        allow_reuse=True,
    )
    # The only fit is at 420 which overlaps the placed task; no candidates.
    assert cands == []


def test_enumerate_candidates_excludes_intervals_overlapping_placed_tasks():
    task = make_task(duration_min=30, duration_max=30)
    rule_index = _build_rule_index([])
    placed = {DATE: [(420, 480)]}  # blocks 420..480.
    cands = enumerate_candidates(
        task,
        [DATE],
        events_by_date={},
        placed_by_date=placed,
        used_base_gaps=set(),
        time_windows={},
        rule_index=rule_index,
        window_start=0,
        window_end=1440,
        importance=merge_importance({}),
        interaction=merge_interaction({}),
        duration_preference="minimum",
        strategy="windows",
        allow_reuse=False,
    )
    # No candidate may overlap `[420, 480)`.
    assert not any(c.start < 480 and c.end > 420 for c in cands)


def test_pick_best_returns_none_on_empty():
    assert _pick_best([], random.Random(0)) is None


def test_pick_best_sorts_ties_by_day_and_start():
    c1 = Candidate(420, 450, DATE, (420, 480), score=0.9, utilities={})
    c2 = Candidate(450, 480, DATE, (420, 480), score=0.9, utilities={})
    chosen = _pick_best([c2, c1], random.Random(0))
    # Same score and day; earliest start wins.
    assert chosen is c1


def test_pick_best_picks_higher_score():
    c1 = Candidate(420, 450, DATE, (420, 480), score=0.4, utilities={})
    c2 = Candidate(450, 480, DATE, (420, 480), score=0.9, utilities={})
    assert _pick_best([c1, c2], random.Random(0)) is c2


def test_pick_best_breaks_full_tie_by_rng():
    # Two candidates sharing `(day, start)` so the rng-tie path fires.
    c1 = Candidate(420, 450, DATE, (0, 1), score=1.0, utilities={})
    c2 = Candidate(420, 450, DATE, (2, 3), score=1.0, utilities={})
    pick = _pick_best([c1, c2], random.Random(0))
    assert pick in (c1, c2)
