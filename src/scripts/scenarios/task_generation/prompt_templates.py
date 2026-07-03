"""Central registry and renderer for every named LLM prompt in the benchmark.

The prompt content itself lives in separate per-prompt modules so that
researchers and developers can iterate on prompt wording without
touching the renderer:

  * `task_generation/prompts/` ; health_improvement,
    health_improvement_with_blacklist, senior_lifestyle, productivity,
    persona_paraphrase, _shared.VERBATIM_CONTENT_RULES.
  * `augmentation/prompts/`    ; augment_agent, augment_oneshot.
  * `metrics/prompts/`         ; semantic_merge_score.

This file only wires the per-prompt `TEMPLATE` constants into the
`render()` / `list_templates()` API and provides the minimal
`{{ variable }}` / `{{ list | join("sep") }}` substitution engine.

Templates use a minimal `{{ variable }}` / `{{ list | join("sep") }}`
syntax rendered by a lightweight built-in engine; no Jinja2 dependency.

Built-in templates
------------------
health_improvement     Generate physical-activity / nutrition tasks for a
                       working-age cohort profile.
senior_lifestyle       Generate nutrition / mental-wellbeing tasks for a
                       senior leisure cohort.
productivity           Generate work-life balance and focus tasks.
augment_agent          System prompt for the LLM-agent augmenter loop.
semantic_merge_score   Prompt requesting a 0–1 compatibility score between
                       two calendar activity labels.

Usage::

    from src.scripts.scenarios.task_generation.prompt_templates import render

    prompt = render(
        "health_improvement",
        num_tasks=5,
        domains=["PhysicalActivityTask", "NutritionTask"],
        difficulty=["Level2", "Level3"],
        profile_block="  - occupation_status: fulltime\\n  - age: 43",
        existing_event_labels=["sleep", "office_work", "lunch"],
        scenario_description="Increase daily physical activity.",
    )
"""

from __future__ import annotations

import re
from typing import Any

from src.scripts.scenarios.augmentation.prompts.augment_agent import (
    TEMPLATE as _AUGMENT_AGENT,
)
from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
    render as _render_augment_oneshot,
)
from src.scripts.scenarios.metrics.prompts.semantic_merge_score import (
    TEMPLATE as _SEMANTIC_MERGE_SCORE,
)
from src.scripts.scenarios.task_generation.prompts.health_improvement import (
    TEMPLATE as _HEALTH_IMPROVEMENT,
)
from src.scripts.scenarios.task_generation.prompts.health_improvement_with_blacklist import (
    TEMPLATE as _HEALTH_IMPROVEMENT_WITH_BLACKLIST,
)
from src.scripts.scenarios.task_generation.prompts.persona_paraphrase import (
    TEMPLATE as _PERSONA_PARAPHRASE,
)
from src.scripts.scenarios.task_generation.prompts.productivity import (
    TEMPLATE as _PRODUCTIVITY,
)
from src.scripts.scenarios.task_generation.prompts.senior_lifestyle import (
    TEMPLATE as _SENIOR_LIFESTYLE,
)

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

# `augment_oneshot` is Jinja2-backed (see
# `src/scripts/scenarios/augmentation/prompts/augment_oneshot.py`)
# so it does NOT live in `_TEMPLATES`; `render()` dispatches to its
# own renderer when the caller asks for it by name. Every other
# template stays on the lightweight `{{ var }}` engine below.
_TEMPLATES: dict[str, str] = {
    "health_improvement": _HEALTH_IMPROVEMENT,
    "health_improvement_with_blacklist": _HEALTH_IMPROVEMENT_WITH_BLACKLIST,
    "persona_paraphrase": _PERSONA_PARAPHRASE,
    "senior_lifestyle": _SENIOR_LIFESTYLE,
    "productivity": _PRODUCTIVITY,
    "augment_agent": _AUGMENT_AGENT,
    "semantic_merge_score": _SEMANTIC_MERGE_SCORE,
}

# Name of the Jinja2-backed prompt routed through its own renderer.
_AUGMENT_ONESHOT_NAME = "augment_oneshot"

# ---------------------------------------------------------------------------
# Minimal {{ }} renderer
# ---------------------------------------------------------------------------

_EXPR_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")
_JOIN_RE = re.compile(r"""^(\w+)\s*\|\s*join\(["']([^"']*)["']\)""")


def _render_expr(expr: str, variables: dict[str, Any]) -> str:
    """Evaluate one `{{ … }}` expression against the variable dict."""
    join_match = _JOIN_RE.match(expr.strip())
    if join_match:
        var_name, sep = join_match.group(1), join_match.group(2)
        val = variables.get(var_name, [])
        items = val if isinstance(val, (list, tuple)) else [val]
        return sep.join(str(v) for v in items)
    return str(variables.get(expr.strip(), ""))


def _render(template: str, variables: dict[str, Any]) -> str:
    """Substitute all `{{ … }}` placeholders in *template*."""
    return _EXPR_RE.sub(lambda m: _render_expr(m.group(1), variables), template)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render(template_name: str, **kwargs: Any) -> str:
    """Render a named template with the supplied keyword arguments.

    For `augment_oneshot`, two underscore-prefixed kwargs route to
    its Jinja2 renderer and are stripped before substitution:
    `_ablate` (iterable of block names to drop) and `_placebos`
    (per-block replacement text).
    """
    if template_name == _AUGMENT_ONESHOT_NAME:
        ablate = kwargs.pop("_ablate", ())
        placebos = kwargs.pop("_placebos", None)
        return _render_augment_oneshot(kwargs, ablate=ablate, placebos=placebos)
    if template_name not in _TEMPLATES:
        raise ValueError(
            f"unknown template {template_name!r}; available: "
            f"{sorted(list_templates())}"
        )
    return _render(_TEMPLATES[template_name], kwargs)


def list_templates() -> list[str]:
    """Return every registered template name in sorted order."""
    return sorted([*_TEMPLATES.keys(), _AUGMENT_ONESHOT_NAME])
