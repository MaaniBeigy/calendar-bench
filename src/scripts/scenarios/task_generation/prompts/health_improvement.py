"""LLM prompt: `health_improvement` — default task-gen for working-age cohorts."""

from __future__ import annotations

from src.scripts.scenarios.task_generation.prompts._shared import VERBATIM_CONTENT_RULES

TEMPLATE = (
    """\
You are a health coaching assistant. Given a person's profile and current calendar,
suggest exactly {{ num_tasks }} new health behavior tasks drawn from the
{{ domains | join(" and ") }} domains at difficulty {{ difficulty | join(" or ") }}.

Person profile:
{{ profile_block }}
  - Existing scheduled activities: {{ existing_event_labels | join(", ") }}

Scenario goal: {{ scenario_description }}

For EACH task return a JSON object with these fields:
  label                   (string)  short snake_case activity name grounded in the ontology
  display_name            (string)  human-readable title — see CONTENT RULES below
  description             (string)  one-sentence description — see CONTENT RULES below
  ontology_uri            (string)  specific INSTANCE URI — see CONTENT RULES below
  duration_min            (int)     minimum realistic duration in minutes
  duration_max            (int)     maximum realistic duration in minutes
  intensity               (int)     1 (very light) … 5 (very demanding)
  is_dividable            (bool)    true when the session can be split across two time slots
  is_concurrent           (bool)    true when this task can overlap another activity

"""
    + VERBATIM_CONTENT_RULES
    + """
Return ONLY a valid JSON array — no markdown fences, no explanatory text."""
)
