"""Tests for per-person deterministic RL seeding."""

from __future__ import annotations

import datetime

from src.scripts.scenarios.augmentation.rl.augmenter import RLAugmenter, _person_seed
from src.scripts.scenarios.config.schema import AugmentationConfig, RLConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from tests.unit.scenarios.conftest import make_event, make_task

DATE = datetime.date(2026, 5, 4)


def _cfg(**rl):
    base = dict(policy="random")
    base.update(rl)
    return AugmentationConfig(method="rl", rl=RLConfig(**base))


def test_person_seed_stable_for_same_person():
    assert _person_seed(20260601, "p1") == _person_seed(20260601, "p1")


def test_person_seed_differs_across_persons():
    assert _person_seed(20260601, "p1") != _person_seed(20260601, "p2")


def test_person_seed_independent_of_base_offset_order():
    # The same person resolves to the same seed no matter the base seed value,
    # shifted only by the base, so two bases differ by exactly their delta.
    a = _person_seed(0, "p1")
    b = _person_seed(5, "p1")
    assert (b - a) % (2**31) == 5


def test_person_seed_within_numpy_range():
    big = _person_seed(2**31 - 1, "i_fulltime_0007")
    assert 0 <= big < 2**31


def test_augment_sets_per_person_rng():
    aug = RLAugmenter(seed=20260601)
    task = make_task(label="walk", duration_min=30, duration_max=30)
    ev = make_event(label="lunch", start_minutes=720, end_minutes=780, date=DATE)
    trace = CalendarTrace(person_id="p1", events=[ev])
    aug.augment(trace, [task], _cfg(), horizon=(DATE, 7))
    assert aug._rng is not None


def test_mask_fallback_uses_person_rng_when_unset():
    # With no augment() call the rng guard still returns a valid action.
    aug = RLAugmenter(seed=0)

    class _Env:
        class env:
            @staticmethod
            def valid_action_mask():
                return [False, True, False]

    out = aug._mask_to_valid_action(_Env(), 0)
    assert out == 1
