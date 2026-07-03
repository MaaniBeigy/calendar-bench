"""Tests for the per-person input-applicability mask and masked scoring."""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from src.scripts.scenarios.config.schema import LossWeights
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.metrics.loss import (
    LossComponents,
    _merge_opportunity_exists,
    compute_applicability_mask,
)
from tests.unit.scenarios.conftest import make_event, make_task

HB_TASK = "https://w3id.org/calendar-bench/health/task/walk"
SLEEP_IRI = "http://purl.obolibrary.org/obo/SLEEP_0001"
NUTRITION_IRI = "http://purl.obolibrary.org/obo/NUTR_0001"
_D = datetime.date(2026, 5, 4)


def _ctx(category: str, iri: str) -> ContextEpisode:
    return ContextEpisode(
        name=category, category=category, date=_D,
        start_minutes=0, end_minutes=60, ontology_uri=iri,
    )


def _cal_with(*episodes: ContextEpisode) -> CalendarTrace:
    return CalendarTrace(person_id="p1", events=[], contexts=list(episodes))


class _Semantic:
    """Minimal SemanticCompatibility stand-in returning a fixed score."""

    def __init__(self, score: float = 0.9) -> None:
        self._score = score

    def score(self, a: str, b: str) -> float:
        return self._score


class _Constraints:
    """Minimal PersonaConstraints stand-in for the pref predicate."""

    def __init__(self, weight: float) -> None:
        self._weight = weight

    def matched_events_for(self, task):
        return [SimpleNamespace(weight=self._weight)]


def _weights(**overrides) -> SimpleNamespace:
    """Return a weights stand-in (avoids the sum-to-1.0 schema constraint)."""
    base = {
        "lambda_cov": 0.18,
        "lambda_cal": 0.17,
        "lambda_pref": 0.15,
        "lambda_disp": 0.1,
        "lambda_merge": 0.1,
        "lambda_spread": 0.1,
        "lambda_divide": 0.1,
        "lambda_context_fit": 0.1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# _merge_opportunity_exists
# ---------------------------------------------------------------------------


def test_merge_opportunity_false_when_no_concurrent_events():
    task = make_task(label="walk", is_concurrent=True)
    cal = CalendarTrace(person_id="p1", events=[make_event(label="lunch")])
    assert not _merge_opportunity_exists(
        [task], cal, _Semantic(0.9), merge_threshold=0.65
    )


def test_merge_opportunity_false_when_task_not_concurrent():
    task = make_task(label="walk", is_concurrent=False)
    cal = CalendarTrace(
        person_id="p1", events=[make_event(label="lunch", is_concurrent=True)]
    )
    assert not _merge_opportunity_exists(
        [task], cal, _Semantic(0.9), merge_threshold=0.65
    )


def test_merge_opportunity_false_when_below_threshold():
    task = make_task(label="walk", is_concurrent=True)
    cal = CalendarTrace(
        person_id="p1", events=[make_event(label="lunch", is_concurrent=True)]
    )
    assert not _merge_opportunity_exists(
        [task], cal, _Semantic(0.4), merge_threshold=0.65
    )


def test_merge_opportunity_true_when_concurrent_pair_clears_threshold():
    task = make_task(label="walk", is_concurrent=True)
    cal = CalendarTrace(
        person_id="p1", events=[make_event(label="lunch", is_concurrent=True)]
    )
    assert _merge_opportunity_exists([task], cal, _Semantic(0.9), merge_threshold=0.65)


def test_merge_opportunity_scores_actual_label_plus_parent():
    """The mask proxy scores the episode's actual label plus its parent."""
    seen: list[tuple[str, str]] = []

    class _Recorder:
        def score(self, a: str, b: str) -> float:
            seen.append((a, b))
            return 0.9

    task = make_task(label="walk", is_concurrent=True)
    cal = CalendarTrace(
        person_id="p1",
        events=[
            make_event(label="office_work", is_concurrent=True, display_label="standup")
        ],
    )
    assert _merge_opportunity_exists([task], cal, _Recorder(), merge_threshold=0.65)
    assert seen == [("walk", "standup (office_work)")]


# ---------------------------------------------------------------------------
# compute_applicability_mask
# ---------------------------------------------------------------------------


def test_mask_empty_when_no_tasks():
    cal = CalendarTrace(person_id="p1", events=[])
    mask = compute_applicability_mask([], cal, LossWeights())
    assert mask == frozenset()


def test_mask_solution_only_legs_when_tasks_exist():
    task = make_task(label="walk")
    cal = CalendarTrace(person_id="p1", events=[])
    mask = compute_applicability_mask([task], cal, LossWeights())
    assert {"cov", "cal", "disp", "spread"} <= mask
    # No pref/merge/divide/context inputs supplied, so they stay out.
    assert "pref" not in mask
    assert "merge" not in mask
    assert "divide" not in mask
    assert "context_fit" not in mask


def test_mask_excludes_zero_weight_leg():
    task = make_task(label="walk")
    cal = CalendarTrace(person_id="p1", events=[])
    mask = compute_applicability_mask([task], cal, _weights(lambda_spread=0.0))
    assert "spread" not in mask
    assert "cov" in mask


def test_mask_includes_pref_when_persona_matches():
    task = make_task(label="walk")
    cal = CalendarTrace(person_id="p1", events=[])
    mask = compute_applicability_mask(
        [task], cal, LossWeights(), persona_constraints=_Constraints(weight=0.8)
    )
    assert "pref" in mask


def test_mask_excludes_pref_when_no_weighted_match():
    task = make_task(label="walk")
    cal = CalendarTrace(person_id="p1", events=[])
    mask = compute_applicability_mask(
        [task], cal, LossWeights(), persona_constraints=_Constraints(weight=0.0)
    )
    assert "pref" not in mask


def test_mask_includes_merge_when_opportunity_exists():
    task = make_task(label="walk", is_concurrent=True)
    cal = CalendarTrace(
        person_id="p1", events=[make_event(label="lunch", is_concurrent=True)]
    )
    mask = compute_applicability_mask(
        [task], cal, LossWeights(), semantic=_Semantic(0.9), merge_threshold=0.65
    )
    assert "merge" in mask


def test_mask_excludes_merge_when_no_semantic():
    """No semantic oracle wired: merge cannot be assessed, so it is out of mask."""
    task = make_task(label="walk", is_concurrent=True)
    cal = CalendarTrace(
        person_id="p1", events=[make_event(label="lunch", is_concurrent=True)]
    )
    mask = compute_applicability_mask([task], cal, LossWeights(), semantic=None)
    assert "merge" not in mask


def test_mask_includes_divide_when_dividable_task_present():
    task = make_task(label="walk", is_dividable=True)
    cal = CalendarTrace(person_id="p1", events=[])
    mask = compute_applicability_mask([task], cal, LossWeights())
    assert "divide" in mask


def test_mask_includes_context_fit_when_links_intersect_observed():
    task = make_task(label="walk", ontology_uri=HB_TASK)
    cal = _cal_with(_ctx("sleep", SLEEP_IRI))
    # Default LossWeights has lambda_context_fit=0; use the stand-in weights
    # so the leg carries a positive weight and can enter the mask.
    mask = compute_applicability_mask(
        [task],
        cal,
        _weights(),
        context_links_by_uri={HB_TASK: {"sleep": frozenset({SLEEP_IRI})}},
        observed_categories=frozenset({"sleep"}),
    )
    assert "context_fit" in mask


def test_mask_excludes_context_fit_when_no_observed_overlap():
    task = make_task(label="walk", ontology_uri=HB_TASK)
    cal = _cal_with(_ctx("sleep", SLEEP_IRI))
    mask = compute_applicability_mask(
        [task],
        cal,
        LossWeights(),
        context_links_by_uri={HB_TASK: {"sleep": frozenset({SLEEP_IRI})}},
        observed_categories=frozenset({"weather"}),
    )
    assert "context_fit" not in mask


def test_mask_includes_context_fit_when_no_observation_budget():
    task = make_task(label="walk", ontology_uri=HB_TASK)
    cal = _cal_with(_ctx("sleep", SLEEP_IRI))
    mask = compute_applicability_mask(
        [task],
        cal,
        _weights(),
        context_links_by_uri={HB_TASK: {"sleep": frozenset({SLEEP_IRI})}},
        observed_categories=None,
    )
    assert "context_fit" in mask


def test_mask_excludes_context_fit_when_linked_category_not_generated():
    """The persona never enters the linked context, so the leg stays out of the mask."""
    task = make_task(label="walk", ontology_uri=HB_TASK)
    cal = _cal_with(_ctx("nutrition", NUTRITION_IRI))
    mask = compute_applicability_mask(
        [task],
        cal,
        _weights(),
        context_links_by_uri={HB_TASK: {"sleep": frozenset({SLEEP_IRI})}},
        observed_categories=frozenset({"sleep"}),
    )
    assert "context_fit" not in mask


def test_mask_excludes_context_fit_when_task_uri_absent_from_links():
    """A non-empty link map that omits the task URI skips the task, no leg."""
    task = make_task(label="walk", ontology_uri=HB_TASK)
    cal = _cal_with(_ctx("sleep", SLEEP_IRI))
    mask = compute_applicability_mask(
        [task],
        cal,
        _weights(),
        context_links_by_uri={"https://other/task": {"sleep": frozenset({SLEEP_IRI})}},
        observed_categories=frozenset({"sleep"}),
    )
    assert "context_fit" not in mask


def test_mask_excludes_context_fit_when_links_miss_observed_categories():
    task = make_task(label="walk", ontology_uri=HB_TASK)
    cal = _cal_with(_ctx("sleep", SLEEP_IRI), _ctx("nutrition", NUTRITION_IRI))
    mask = compute_applicability_mask(
        [task],
        cal,
        _weights(),
        context_links_by_uri={
            HB_TASK: {
                "sleep": frozenset({SLEEP_IRI}),
                "nutrition": frozenset({NUTRITION_IRI}),
            }
        },
        observed_categories=frozenset({"weather", "location"}),
    )
    assert "context_fit" not in mask


# ---------------------------------------------------------------------------
# masked_view / weighted edge cases
# ---------------------------------------------------------------------------


def test_weighted_no_mask_skips_zero_weight_legs():
    comp = LossComponents(
        cov=0.5, cal=0.5, pref=0.5, disp=0.5, merge=0.5, spread=0.5, divide=0.5
    )
    weights = _weights(lambda_divide=0.0)
    # With no mask, only positive-weight legs contribute; divide drops out.
    assert comp.weighted(weights) == pytest.approx(0.5)


def test_weighted_returns_zero_when_no_leg_in_scope():
    comp = LossComponents(cov=0.5, cal=0.5, pref=None, disp=0.5, merge=None)
    assert comp.weighted(LossWeights(), frozenset()) == 0.0


def test_masked_view_zero_placement_marks_all_in_mask_full():
    comp = LossComponents(cov=0.1, cal=0.1, pref=0.1, disp=0.1, merge=0.1, spread=0.1)
    view = comp.masked_view({"cov", "cal"}, zero_placement=True)
    assert view.cov == 1.0
    assert view.cal == 1.0
    assert view.pref is None


# ---------------------------------------------------------------------------
# RLAugmenter.set_constraint_factories
# ---------------------------------------------------------------------------


def test_rl_set_constraint_factories_installs_factories():
    """`set_constraint_factories` wires the per-person pref/window factories."""
    from src.scripts.scenarios.augmentation.rl.augmenter import RLAugmenter

    aug = RLAugmenter()
    sentinel_constraints = object()
    sentinel_window = object()
    aug.set_constraint_factories(
        lambda trace: sentinel_constraints, lambda trace: sentinel_window
    )
    assert aug._persona_constraints_for(None) is sentinel_constraints
    assert aug._window_map_for(None) is sentinel_window
