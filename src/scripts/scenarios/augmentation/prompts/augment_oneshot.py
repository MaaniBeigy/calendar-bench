"""Renderer for the `augment_oneshot` LLM prompt template.

The prose lives in `augment_oneshot.j2`. Named `{% block %}` markers
allow individual sections to be dropped or replaced at render time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .augment_oneshot_placebos import DEFAULT_PLACEBOS

# Bit order shared with the Plackett-Burman design matrices and the
# benchmark-report tables. `split_dividable_tasks` is the 12th block;
# PB-12 supports only 11 factors so its 12th column is held at +1
# (always kept). PB-24 tests the 12th and 13th blocks via the folded
# half. `context_block` is the 13th block.
ABLATABLE_BLOCKS: tuple[str, ...] = (
    "default_path",
    "fit_examples",
    "rule_of_thumb_1",
    "rule_of_thumb_2",
    "rule_of_thumb_3",
    "worked_example_standalone",
    "worked_example_concurrent",
    "worked_example_waking",
    "hard_constraints",
    "do_not_drop",
    "self_verify",
    "split_dividable_tasks",
    "context_block",
)

_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_NAME = "augment_oneshot.j2"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
    autoescape=False,
)


def _build_child_source(
    ablate: Iterable[str],
    placebos: Mapping[str, str] | None,
) -> str:
    """Build a child template that overrides the named blocks."""
    placebos = placebos or {}
    unknown = set(ablate) - set(ABLATABLE_BLOCKS)
    if unknown:
        raise ValueError(
            f"unknown ablation target(s): {sorted(unknown)}; "
            f"valid blocks: {ABLATABLE_BLOCKS}"
        )
    overrides = []
    for name in ablate:
        body = placebos.get(name, "")
        overrides.append(
            f"{{% block {name} %}}{{% raw %}}{body}{{% endraw %}}{{% endblock %}}"
        )
    return f'{{% extends "{_TEMPLATE_NAME}" %}}\n' + "\n".join(overrides)


def render(
    context: Mapping[str, object],
    ablate: Iterable[str] = (),
    placebos: Mapping[str, str] | None = None,
) -> str:
    """Render the prompt; optionally drop or replace named sections.

    Args:
        context: template variables (`week_start`, `week_end`,
            `wake_start`, `sleep_start`, `calendar_summary`,
            `num_tasks`, `tasks_list`).
        ablate: block names to drop. Unknown names raise `ValueError`.
        placebos: per-block replacement text. Wins over deletion when
            both are set.

    Returns:
        The rendered prompt with no Jinja artifacts.
    """
    ablate_list = list(ablate)
    if not ablate_list:
        template = _env.get_template(_TEMPLATE_NAME)
    else:
        template = _env.from_string(_build_child_source(ablate_list, placebos))
    return template.render(**dict(context))


def get_template_source() -> str:
    """Return the raw `.j2` source."""
    return (_TEMPLATE_DIR / _TEMPLATE_NAME).read_text(encoding="utf-8")


def default_placebo_for(block: str) -> str:
    """Return the length-matched stand-in for `block`, or empty string."""
    return DEFAULT_PLACEBOS.get(block, "")
