"""Parse raw LLM text into a list of RecommendedTask objects.

The parser handles three common LLM output formats:

1. Plain JSON array:
       [{"label": "yoga", "duration_min": 20, ...}]

2. Markdown-fenced JSON (with or without `json` language tag):
       ```json
       [{"label": "yoga", ...}]
       ```

3. JSON array embedded in surrounding prose:
       "Here are the suggested tasks:\n\n[{...}]\n\nLet me know..."

Field defaults (applied when the LLM omits a field):

    duration_min           15 minutes
    duration_max           60 minutes
    intensity              2
    is_dividable           False
    is_concurrent          False
    ontology_uri           None

Legacy keys (`preferred_epoch`, `admissible_epochs`,
`preferred_start_minutes`, `source_ontology`) are silently dropped at
parse time so older task JSONs load without intervention.

Scenario-level `task_overrides` (dict mapping label to field to value)
are applied on top of the LLM values, allowing the scenario YAML to fine-tune
individual tasks without changing the LLM prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.scripts.scenarios.domain.task import RecommendedTask

# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

# Markdown code-fence: ``json ... `` or `` ... ``
_FENCE_RE = re.compile(r"``(?:json)?[ \t]*([\s\S]*?)``", re.IGNORECASE)

# Greedy match for a JSON array: first '[' to last ']'
_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

# ---------------------------------------------------------------------------
# Field defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "duration_min": 15,
    "duration_max": 60,
    "intensity": 2,
    "is_dividable": False,
    "is_concurrent": False,
    "ontology_uri": None,
    "display_name": "",
    "description": "",
    "difficulty_level": 0,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> list[dict[str, Any]]:
    """Extract and parse a JSON array from raw LLM text.

    Priority:
    1. Content inside a markdown code fence.
    2. The first `[` … last `]` substring.
    3. The entire text (already a JSON array).

    Raises:
        ValueError: if the extracted text is not valid JSON or not a list.
    """
    text = raw.strip()

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        array_match = _ARRAY_RE.search(text)
        if array_match:
            text = array_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse JSON from LLM response: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"expected a JSON array in LLM response; got {type(data).__name__}"
        )
    return data


def _coerce_str_or_none(value: Any) -> str | None:
    """Return the value as str, or None if it is falsy."""
    if not value:
        return None
    return str(value)


def _task_from_dict(
    raw: dict[str, Any],
    task_overrides: dict[str, Any] | None,
) -> RecommendedTask:
    """Construct a RecommendedTask from a raw dict, applying defaults and overrides.

    Priority for each field: task_overrides > LLM value > default.
    """
    label = (
        str(raw.get("label", "unknown_task")).strip().replace(" ", "_")
        or "unknown_task"
    )
    per_label_overrides: dict[str, Any] = (task_overrides or {}).get(label, {})

    def _get(field: str) -> Any:
        if field in per_label_overrides:
            return per_label_overrides[field]
        if field in raw:
            return raw[field]
        return _DEFAULTS[field]

    return RecommendedTask(
        label=label,
        duration_min=int(_get("duration_min")),
        duration_max=int(_get("duration_max")),
        intensity=int(_get("intensity")),
        is_dividable=bool(_get("is_dividable")),
        is_concurrent=bool(_get("is_concurrent")),
        ontology_uri=_coerce_str_or_none(_get("ontology_uri")),
        display_name=str(_get("display_name")),
        description=str(_get("description")),
        difficulty_level=int(_get("difficulty_level") or 0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_paraphrases(raw_text: str) -> list[dict[str, Any]]:
    """Parse the persona-paraphrase LLM response into a list of dicts.

    Expected payload (one entry per task, in input order)::

        [
          { "label": "...", "personalized_description": "..." },
          ...
        ]

    Tolerant to surrounding markdown fences and free-form text; same
    extraction priority as :func:`extract_raw_task_dicts`.  Non-dict
    items are filtered out.  Returns `[]` for empty input.

    Raises:
        ValueError: if the JSON is malformed or not a list.
    """
    if not raw_text.strip():
        return []
    return [item for item in _extract_json(raw_text) if isinstance(item, dict)]


def extract_raw_task_dicts(raw_text: str) -> list[dict[str, Any]]:
    """Extract raw task dicts from LLM text without constructing RecommendedTask objects.

    Useful when the caller wants to enrich the dicts with additional data
    (e.g. ontology properties) before converting to `RecommendedTask`.

    Args:
        raw_text: Raw LLM output, possibly with markdown fences.

    Returns:
        List of dict items from the JSON array.  Non-dict items are filtered
        out.  Returns `[]` for empty or whitespace-only input.

    Raises:
        ValueError: same conditions as :func:`parse_task_list`.
    """
    if not raw_text.strip():
        return []
    return [item for item in _extract_json(raw_text) if isinstance(item, dict)]


def parse_task_list(
    raw_text: str,
    task_overrides: dict[str, Any] | None = None,
) -> list[RecommendedTask]:
    """Parse a raw LLM response into a list of `RecommendedTask` objects.

    Args:
        raw_text: Raw LLM output, possibly with markdown fences.
        task_overrides: `{label: {field: value}}` applied on top of LLM
            values.  Labels are matched after whitespace normalization.

    Returns:
        List of `RecommendedTask` instances.  Non-dict items in the JSON array
        are silently skipped.  Returns `[]` for empty or whitespace-only
        input.

    Raises:
        ValueError: if the text is non-empty but contains no parseable JSON
            array, or if the extracted JSON is not a list.
    """
    if not raw_text.strip():
        return []

    items = _extract_json(raw_text)
    return [
        _task_from_dict(item, task_overrides)
        for item in items
        if isinstance(item, dict)
    ]
