"""LLM prompt: `senior_lifestyle` — task-gen for senior leisure cohorts."""

from __future__ import annotations

from src.scripts.scenarios.task_generation.prompts._shared import VERBATIM_CONTENT_RULES

TEMPLATE = (
    """\
You are a lifestyle advisor for older adults. The person below is part of a senior
leisure cohort. Suggest exactly {{ num_tasks }} behaviour tasks from the
{{ domains | join(" and ") }} domains at difficulty {{ difficulty | join(" or ") }}.

Person profile:
{{ profile_block }}
  - Current activities: {{ existing_event_labels | join(", ") }}

Goal: {{ scenario_description }}

Return a JSON array where each item has: label, display_name, description,
ontology_uri, duration_min, duration_max, intensity (1–5),
is_dividable, is_concurrent.

"""
    + VERBATIM_CONTENT_RULES
    + """
Return ONLY the JSON array."""
)
