"""Test the CLI factory dispatch for the `ptime` method."""

from __future__ import annotations

from dataclasses import dataclass

from src.scripts.scenarios.augmentation.ptime import PTimeAugmenter
from src.scripts.scenarios.cli import _build_augmenter


@dataclass
class _FakeEnv:
    seed: int = 42


@dataclass
class _FakeRun:
    time_windows: dict
    daily_window: object
    allen_pair_rules: list
    environment: _FakeEnv


def test_build_augmenter_returns_ptime_augmenter():
    run = _FakeRun(
        time_windows={},
        daily_window=None,
        allen_pair_rules=[],
        environment=_FakeEnv(seed=7),
    )
    aug = _build_augmenter("ptime", run=run)
    assert isinstance(aug, PTimeAugmenter)
    # Seed should propagate from the persona environment.
    assert aug._seed == 7


def test_build_augmenter_ptime_without_run_falls_back():
    aug = _build_augmenter("ptime")
    assert isinstance(aug, PTimeAugmenter)
    assert aug._seed == 0  # default when no env seed


def test_build_augmenter_unknown_method_returns_none():
    assert _build_augmenter("nope") is None
