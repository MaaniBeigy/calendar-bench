"""tqdm-backed progress bar with a graceful fallback.

The pool wraps its results iterator through `wrap_progress` so a long run
(several hundred persons across multiple weeks) prints a per-person tick.
We import tqdm at call time, not at module top-level, so a missing tqdm
on the host (or a CI environment that wants no output) does not crash
imports of the persona pipeline.

`default_show_progress()` returns True when stderr looks like a real
terminal, False otherwise (tests, CI, redirected stderr). CLI flags can
still force on or off.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def default_show_progress() -> bool:
    """Best-guess default: show a bar only when running in a real terminal."""
    return sys.stderr.isatty()


def _load_tqdm():  # pragma: no cover  (trivial import wrapper)
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm


def wrap_progress(
    iterable: Iterable[T],
    *,
    total: int,
    desc: str,
    show: bool,
) -> Iterator[T]:
    """Yield items from `iterable`, optionally with a tqdm bar.

    `show=False` returns the iterable as-is; `show=True` wraps it through
    tqdm. If tqdm is not installed we fall back to no-op iteration so the
    pipeline still works on a minimal install.
    """
    if not show:
        yield from iterable
        return
    tqdm = _load_tqdm()
    if tqdm is None:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, unit="person")


__all__ = ["default_show_progress", "wrap_progress"]
