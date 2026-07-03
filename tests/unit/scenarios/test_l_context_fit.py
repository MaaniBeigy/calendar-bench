"""L_context_fit: IRI/member-level matching with a generated-category denominator."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from src.scripts.scenarios.config.schema import LossWeights
from src.scripts.scenarios.domain.calendar import AugmentedCalendar, CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.context_fit import (
    HB_TASK_PREFIX,
    collect_verdicts,
    compute_l_context_fit,
    load_context_categories_by_iri,
    load_context_links_by_uri,
    per_task_fit,
)

# Context member IRIs (the join key between tasks and episodes).
HAPPY = "http://purl.obolibrary.org/obo/MFOEM_000042"
CALM = "http://purl.obolibrary.org/obo/MFOEM_000107"
ANXIOUS = "http://purl.obolibrary.org/obo/MFOEM_000028"
ALONE = "https://w3id.org/calendar-bench/context#social/alone"
WITH_FAMILY = "http://humanbehaviourchange.org/ontology/BCIO_006002"
WEATHER = "http://example.org/weather"

TEA = HB_TASK_PREFIX + "tea-time"
WALK = HB_TASK_PREFIX + "walk-5000-steps"
BLOCK = HB_TASK_PREFIX + "block-distractions"
D = datetime.date(2026, 5, 4)


def _desired(label: str, uri: str | None) -> RecommendedTask:
    return RecommendedTask(
        label=label,
        duration_min=10,
        duration_max=20,
        intensity=2,
        ontology_uri=uri,
    )


def _scheduled(
    task: RecommendedTask, *, start: int, end: int, date: datetime.date
) -> ScheduledTask:
    return ScheduledTask(
        task=task,
        start_minutes=start,
        end_minutes=end,
        is_standalone=True,
        concurrent_with=None,
        date=date,
    )


def _context(
    name: str,
    category: str,
    *,
    start: int,
    end: int,
    date: datetime.date,
    ontology_uri: str | None = None,
) -> ContextEpisode:
    return ContextEpisode(
        name=name,
        category=category,
        date=date,
        start_minutes=start,
        end_minutes=end,
        ontology_uri=ontology_uri,
    )


def _solution(
    scheduled: list[ScheduledTask], person_id: str = "p1"
) -> SchedulingSolution:
    aug = AugmentedCalendar(
        person_id=person_id, base_events=[], scheduled_tasks=scheduled
    )
    return SchedulingSolution(
        person_id=person_id,
        augmented_calendar=aug,
        tasks=[s.task for s in scheduled],
        scheduled=scheduled,
        unscheduled=[],
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_load_context_categories_by_iri_maps_each_member(tmp_path: Path) -> None:
    payload = {
        "context": {
            "mood_emotion": {"happy": {"iri": HAPPY}, "blank": {"label": "no iri"}},
            "social_context": {"alone": {"iri": ALONE}, "with_family": {"iri": WITH_FAMILY}},
        }
    }
    path = tmp_path / "context_iris.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = load_context_categories_by_iri(path)
    assert out == {
        HAPPY: "mood_emotion",
        ALONE: "social_context",
        WITH_FAMILY: "social_context",
    }


def test_load_context_links_by_uri_groups_iris_by_category(tmp_path: Path) -> None:
    ttl = f"""
@prefix hb:    <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk: <https://w3id.org/calendar-bench/health/task/> .
hb-tk:block-distractions hb:contextLink <{ALONE}> ;
    hb:contextLink <{HAPPY}> ;
    hb:contextLink <https://unmapped.example/X> .
"""
    path = tmp_path / "HealthTasks_test.ttl"
    path.write_text(ttl, encoding="utf-8")
    cats = {ALONE: "social_context", HAPPY: "mood_emotion"}
    out = load_context_links_by_uri(path, cats)
    assert out == {
        BLOCK: {
            "social_context": frozenset({ALONE}),
            "mood_emotion": frozenset({HAPPY}),
        }
    }


def test_load_context_links_drops_task_with_only_unmapped_links(tmp_path: Path) -> None:
    ttl = """
@prefix hb:    <https://w3id.org/calendar-bench/health/> .
@prefix hb-tk: <https://w3id.org/calendar-bench/health/task/> .
hb-tk:empty-task hb:contextLink <https://unmapped.example/Y> .
"""
    path = tmp_path / "HealthTasks_test.ttl"
    path.write_text(ttl, encoding="utf-8")
    assert load_context_links_by_uri(path, {}) == {}


# ---------------------------------------------------------------------------
# per_task_fit: IRI/member-level matching, generated-category denominator
# ---------------------------------------------------------------------------


def test_per_task_fit_returns_none_when_no_recommendation() -> None:
    task = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    ep = _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY)
    assert per_task_fit(task, [ep], context_links_by_uri={}) is None


def test_per_task_fit_returns_none_when_no_category_generated() -> None:
    """All recommended categories absent from the trace: the task is not scorable."""
    task = _scheduled(_desired("block", BLOCK), start=540, end=560, date=D)
    contexts = [
        _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY),
    ]
    links = {BLOCK: {"social_context": frozenset({ALONE})}}
    assert per_task_fit(task, contexts, links) is None


def test_per_task_fit_excludes_category_not_generated_from_denominator() -> None:
    """A recommended category the persona never generates does not shrink the score."""
    task = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    contexts = [
        _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY),
    ]
    links = {
        TEA: {
            "mood_emotion": frozenset({HAPPY}),
            "weather_environment": frozenset({WEATHER}),
        }
    }
    verdict = per_task_fit(task, contexts, links)
    assert verdict.fit == pytest.approx(1.0)
    assert verdict.recommended_categories == ("mood_emotion",)


def test_per_task_fit_full_overlap_returns_fit_one() -> None:
    task = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    contexts = [
        _context("happy", "mood_emotion", start=480, end=600, date=D, ontology_uri=HAPPY),
        _context("alone", "social_context", start=0, end=720, date=D, ontology_uri=ALONE),
    ]
    links = {TEA: {"mood_emotion": frozenset({HAPPY}), "social_context": frozenset({ALONE})}}
    verdict = per_task_fit(task, contexts, links)
    assert verdict is not None
    assert verdict.fit == pytest.approx(1.0)
    assert verdict.overlapped_categories == ("mood_emotion", "social_context")


def test_per_task_fit_member_mismatch_scores_zero_alone_vs_with_family() -> None:
    """Regression: linking social_context:alone earns nothing during with_family,
    even though the persona does generate alone elsewhere in the trace."""
    task = _scheduled(_desired("block", BLOCK), start=540, end=560, date=D)
    links = {BLOCK: {"social_context": frozenset({ALONE})}}
    # alone is generated (so social_context is realizable), but the placement
    # overlaps a with_family episode instead of the alone one.
    miss = [
        _context("with_family", "social_context", start=540, end=600, date=D, ontology_uri=WITH_FAMILY),
        _context("alone", "social_context", start=0, end=100, date=D, ontology_uri=ALONE),
    ]
    assert per_task_fit(task, miss, links).fit == pytest.approx(0.0)
    hit = [_context("alone", "social_context", start=540, end=600, date=D, ontology_uri=ALONE)]
    assert per_task_fit(task, hit, links).fit == pytest.approx(1.0)


def test_per_task_fit_no_overlap_returns_zero_wrong_day() -> None:
    task = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    # happy is generated (realizable) but only on another day, so never overlaps.
    other_day = _context(
        "happy", "mood_emotion", start=540, end=600,
        date=datetime.date(2026, 5, 5), ontology_uri=HAPPY,
    )
    links = {TEA: {"mood_emotion": frozenset({HAPPY})}}
    assert per_task_fit(task, [other_day], links).fit == pytest.approx(0.0)


def test_per_task_fit_partial_realizable_category_not_overlapped() -> None:
    """A generated category the placement misses still counts in the denominator."""
    task = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    contexts = [
        _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY),
        _context("alone", "social_context", start=0, end=100, date=D, ontology_uri=ALONE),
    ]
    links = {TEA: {"mood_emotion": frozenset({HAPPY}), "social_context": frozenset({ALONE})}}
    verdict = per_task_fit(task, contexts, links)
    assert verdict.fit == pytest.approx(0.5)
    assert verdict.recommended_categories == ("mood_emotion", "social_context")
    assert verdict.overlapped_categories == ("mood_emotion",)


# ---------------------------------------------------------------------------
# Observed-budget scoping
# ---------------------------------------------------------------------------


def test_per_task_fit_observed_scope_returns_none_when_intersection_empty() -> None:
    task = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    ep = _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY)
    links = {TEA: {"mood_emotion": frozenset({HAPPY})}}
    assert (
        per_task_fit(task, [ep], links, observed_categories=frozenset({"behaviour_state"}))
        is None
    )


def test_per_task_fit_observed_scope_collapses_denominator() -> None:
    task = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    contexts = [
        _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY),
        _context("alone", "social_context", start=0, end=100, date=D, ontology_uri=ALONE),
        _context("weather", "weather_environment", start=0, end=100, date=D, ontology_uri=WEATHER),
    ]
    links = {
        TEA: {
            "mood_emotion": frozenset({HAPPY}),
            "social_context": frozenset({ALONE}),
            "weather_environment": frozenset({WEATHER}),
        }
    }
    full = per_task_fit(task, contexts, links)
    assert full.fit == pytest.approx(1 / 3)
    assert full.fit_full == pytest.approx(1 / 3)
    scoped = per_task_fit(
        task, contexts, links, observed_categories=frozenset({"mood_emotion"})
    )
    assert scoped.fit == pytest.approx(1.0)
    assert scoped.fit_full == pytest.approx(1 / 3)
    assert scoped.scored_categories == ("mood_emotion",)
    assert scoped.observed_categories == ("mood_emotion",)


# ---------------------------------------------------------------------------
# compute_l_context_fit and collect_verdicts
# ---------------------------------------------------------------------------


def test_compute_l_context_fit_returns_none_when_no_task_has_recommendation() -> None:
    task = _scheduled(_desired("walk", None), start=540, end=600, date=D)
    sol = _solution([task])
    assert compute_l_context_fit(sol, CalendarTrace(person_id="p1"), {}) is None


def test_compute_l_context_fit_returns_one_minus_mean_fit() -> None:
    """Two tasks: one hit (fit=1.0), one miss (fit=0.0); mean=0.5; loss=0.5."""
    t1 = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    t2 = _scheduled(
        _desired("walk", WALK), start=540, end=560, date=datetime.date(2026, 5, 5)
    )
    cal = CalendarTrace(
        person_id="p1",
        contexts=[
            _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY),
        ],
    )
    links = {
        TEA: {"mood_emotion": frozenset({HAPPY})},
        WALK: {"mood_emotion": frozenset({HAPPY})},
    }
    assert compute_l_context_fit(_solution([t1, t2]), cal, links) == pytest.approx(0.5)


def test_collect_verdicts_emits_one_per_recommended_task() -> None:
    t1 = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    t2 = _scheduled(_desired("walk", WALK), start=540, end=560, date=D)
    t3 = _scheduled(_desired("noname", None), start=540, end=560, date=D)
    cal = CalendarTrace(
        person_id="p1",
        contexts=[
            _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY),
        ],
    )
    links = {TEA: {"mood_emotion": frozenset({HAPPY})}}
    verdicts = collect_verdicts(_solution([t1, t2, t3]), cal, links)
    assert [v.task_label for v in verdicts] == ["tea"]


def test_collect_verdicts_skips_tasks_with_empty_scoped_intersection() -> None:
    t_in = _scheduled(_desired("tea", TEA), start=540, end=560, date=D)
    t_out = _scheduled(_desired("walk", WALK), start=540, end=600, date=D)
    cal = CalendarTrace(
        person_id="p1",
        contexts=[
            _context("happy", "mood_emotion", start=540, end=600, date=D, ontology_uri=HAPPY),
            _context("weather", "weather_environment", start=0, end=100, date=D, ontology_uri=WEATHER),
        ],
    )
    links = {
        TEA: {"mood_emotion": frozenset({HAPPY})},
        WALK: {"weather_environment": frozenset({WEATHER})},
    }
    verdicts = collect_verdicts(
        _solution([t_in, t_out]), cal, links,
        observed_categories=frozenset({"mood_emotion"}),
    )
    assert [v.task_label for v in verdicts] == ["tea"]


# ---------------------------------------------------------------------------
# LossWeights sum-to-one invariant (unchanged)
# ---------------------------------------------------------------------------


def test_loss_weights_with_lambda_context_fit_zero_keeps_sum_to_one() -> None:
    assert LossWeights().lambda_context_fit == 0.0


def test_loss_weights_with_lambda_context_fit_violates_sum_to_one_when_unbalanced() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="sum to 1.0"):
        LossWeights(lambda_context_fit=0.10)


def test_loss_weights_with_lambda_context_fit_balanced_passes() -> None:
    weights = LossWeights(
        lambda_cov=0.18,
        lambda_cal=0.18,
        lambda_pref=0.13,
        lambda_disp=0.18,
        lambda_merge=0.13,
        lambda_spread=0.10,
        lambda_divide=0.00,
        lambda_context_fit=0.10,
    )
    assert weights.lambda_context_fit == 0.10
