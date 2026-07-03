"""LLM prompt: `productivity` — task-gen for work-life-balance cohorts."""

from __future__ import annotations

from src.scripts.scenarios.task_generation.prompts._shared import VERBATIM_CONTENT_RULES

TEMPLATE = (
    """\
You are a personal productivity coach. Recommend exactly {{ num_tasks }} tasks
from the {{ domains | join(" and ") }} domains at difficulty {{ difficulty | join(" or ") }}
to help the following person achieve a better work-life balance.

Person profile:
{{ profile_block }}
  - Current schedule: {{ existing_event_labels | join(", ") }}

Goal: {{ scenario_description }}

Return a JSON array where each item has: label, display_name, description,
ontology_uri, duration_min, duration_max, intensity (1–5),
is_dividable, is_concurrent.

"""
    + VERBATIM_CONTENT_RULES
    + """
Return ONLY the JSON array."""
)
