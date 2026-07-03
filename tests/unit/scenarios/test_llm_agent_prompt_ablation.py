"""Prompt-ablation kwargs forwarded by `_build_oneshot_prompt`."""

from __future__ import annotations

import datetime
from typing import Any

import pytest

from src.scripts.persona.config.schema import DailyWindow
from src.scripts.scenarios.augmentation.llm_agent import LLMAugmenter
from src.scripts.scenarios.config.schema import AugmentationConfig, LLMAgentConfig
from tests.unit.scenarios.conftest import make_task

DATE = datetime.date(2026, 5, 4)


def _oneshot_config() -> AugmentationConfig:
    return AugmentationConfig(
        method="llm_agent",
        allow_merge=True,
        merge_threshold=0.70,
        llm_agent=LLMAgentConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_retries=1,
            prompt_template="augment_oneshot",
        ),
    )


def _capture_render_call(monkeypatch) -> dict[str, Any]:
    """Patch `render` in the llm_agent module and record its kwargs."""
    captured: dict[str, Any] = {}

    def fake_render(template_name: str, **kwargs: Any) -> str:
        captured["template_name"] = template_name
        captured["kwargs"] = kwargs
        return "PROMPT-OK"

    from src.scripts.scenarios.augmentation import llm_agent as la

    monkeypatch.setattr(la, "render", fake_render)
    return captured


def test_build_oneshot_prompt_injects_ablate_block_when_active(monkeypatch):
    """A non-empty `prompt_ablate` adds `_ablate` to the render kwargs."""
    captured = _capture_render_call(monkeypatch)
    ablate = frozenset({"calendar_summary"})
    augmenter = LLMAugmenter(
        llm_pipeline=None,
        prompt_ablate=ablate,
        daily_window=DailyWindow(),
    )
    task = make_task("yoga")
    augmenter._build_oneshot_prompt(  # type: ignore[attr-defined]
        tasks=[task],
        week_dates=[DATE],
        events_by_date={},
        scheduled_so_far=None,
        config=_oneshot_config(),
    )
    assert captured["template_name"] == "augment_oneshot"
    assert captured["kwargs"]["_ablate"] == ablate
    assert "_placebos" not in captured["kwargs"]


def test_build_oneshot_prompt_injects_placebos_when_supplied(monkeypatch):
    """Both `prompt_ablate` and `prompt_placebos` flow to the render kwargs."""
    captured = _capture_render_call(monkeypatch)
    ablate = frozenset({"calendar_summary"})
    placebos = {"calendar_summary": "placebo text"}
    augmenter = LLMAugmenter(
        llm_pipeline=None,
        prompt_ablate=ablate,
        prompt_placebos=placebos,
        daily_window=DailyWindow(),
    )
    task = make_task("yoga")
    augmenter._build_oneshot_prompt(  # type: ignore[attr-defined]
        tasks=[task],
        week_dates=[DATE],
        events_by_date={},
        scheduled_so_far=None,
        config=_oneshot_config(),
    )
    assert captured["kwargs"]["_ablate"] == ablate
    assert captured["kwargs"]["_placebos"] == placebos


def test_build_oneshot_prompt_skips_ablate_kwargs_without_active_ablation(monkeypatch):
    """No `prompt_ablate` set adds neither `_ablate` nor `_placebos`."""
    captured = _capture_render_call(monkeypatch)
    augmenter = LLMAugmenter(llm_pipeline=None, daily_window=DailyWindow())
    augmenter._build_oneshot_prompt(  # type: ignore[attr-defined]
        tasks=[make_task("yoga")],
        week_dates=[DATE],
        events_by_date={},
        scheduled_so_far=None,
        config=_oneshot_config(),
    )
    assert "_ablate" not in captured["kwargs"]
    assert "_placebos" not in captured["kwargs"]
