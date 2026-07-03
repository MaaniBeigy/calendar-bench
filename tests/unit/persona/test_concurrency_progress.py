"""Unit tests for src.scripts.persona.concurrency.progress."""

from __future__ import annotations

import sys

from src.scripts.persona.concurrency import progress as progress_module
from src.scripts.persona.concurrency.progress import (
    default_show_progress,
    wrap_progress,
)


def test_wrap_progress_show_false_returns_original_iterable():
    items = [1, 2, 3]
    out = list(wrap_progress(items, total=3, desc="x", show=False))
    assert out == items


def test_wrap_progress_show_true_yields_same_items_when_tqdm_present(monkeypatch):
    """tqdm is installed in the docker image; we intercept the call so the
    test does not actually print a bar."""
    captured: dict = {}

    def fake_tqdm(iterable, total, desc, unit):
        captured["total"] = total
        captured["desc"] = desc
        captured["unit"] = unit
        return iter(list(iterable))

    monkeypatch.setattr(progress_module, "_load_tqdm", lambda: fake_tqdm)
    items = [10, 20, 30]
    out = list(wrap_progress(items, total=3, desc="solve", show=True))
    assert out == items
    assert captured["total"] == 3
    assert captured["desc"] == "solve"
    assert captured["unit"] == "person"


def test_wrap_progress_show_true_falls_back_when_tqdm_missing(monkeypatch):
    """If tqdm is unavailable, the wrapper still yields items without raising."""
    monkeypatch.setattr(progress_module, "_load_tqdm", lambda: None)
    items = [1, 2]
    out = list(wrap_progress(items, total=2, desc="x", show=True))
    assert out == items


def test_default_show_progress_follows_stderr_tty(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert default_show_progress() is True
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    assert default_show_progress() is False


def test_wrap_progress_iterates_lazily_when_disabled():
    """Show=False should not materialize the iterable; pulling one item is enough."""
    pulled: list[int] = []

    def gen():
        for i in range(5):
            pulled.append(i)
            yield i

    iterator = wrap_progress(gen(), total=5, desc="x", show=False)
    assert next(iterator) == 0
    # Only one item pulled so far.
    assert pulled == [0]
