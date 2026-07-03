"""LLM prompt: `health_improvement_with_blacklist` — retry path that excludes
already-accepted and previously-rejected URIs so the generator does not loop
on the same proposals.
"""

from __future__ import annotations

from src.scripts.scenarios.task_generation.prompts._shared import VERBATIM_CONTENT_RULES

TEMPLATE = (
    """\
You are a health coaching assistant. Given a person's profile and current calendar,
suggest exactly {{ needed }} NEW health behavior tasks drawn from the
{{ domains | join(" and ") }} domains at difficulty {{ difficulty | join(" or ") }}.

Person profile:
{{ profile_block }}
  - Existing scheduled activities: {{ existing_event_labels | join(", ") }}

Scenario goal: {{ scenario_description }}

You have already proposed the tasks below in earlier attempts.  Pick
fresh, distinct tasks from the GraphRAG retrieval context that are NOT
in either list.  Do NOT propose minor variations of any of these URIs
either (e.g. ``prepare-a-simple-snack`` is just as forbidden as
``prepare-a-healthy-snack``).

Already accepted (by display_name):
{{ already_accepted_block }}

Already accepted ontology_uri (do NOT repeat any of these):
{{ already_accepted_uri_block }}

Previously REJECTED ontology_uri — these were fabricated or duplicate in
earlier attempts; the validator dropped them.  Do NOT propose any of
these or any close variant of them; pick a different URI from the
retrieval context instead:
{{ rejected_uri_block }}

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
Return ONLY a valid JSON array of length {{ needed }} — no markdown fences,
no explanatory text."""
)
