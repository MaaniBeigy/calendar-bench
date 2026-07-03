"""LLM prompt: `persona_paraphrase` — per-persona rewording of canonical
ontology task descriptions, with strict no-scope-creep / emoji-parity rules.
"""

from __future__ import annotations

TEMPLATE = """\
You are personalizing {{ num_tasks }} short health-task descriptions for one
person.  The persona profile and the canonical ontology entries are below.

Hard constraints (a paraphrase that violates ANY of these will be discarded
and the canonical description will be used instead):
  - Keep the same activity (no scope creep, no new sub-tasks).
  - Keep the same factual claims (no new facts, no new numbers).
  - EMOJIS in your paraphrased description must MATCH the canonical
    description EXACTLY — same emojis, same count, same characters.
      • Do NOT add an emoji the canonical description doesn't have.
      • Do NOT remove an emoji the canonical description does have.
      • Do NOT swap one emoji for a similar-looking one.
      • Do NOT change the number of times an emoji appears.
    The display_name's emoji is fixed by the ontology and is shown
    separately — never repeat it inside the description.
  - Keep the paraphrase LENGTH within ±{{ length_delta_pct }}% of the
    canonical description length.  Short, slight rewordings only.
  - Adjust tone to suit the persona without naming persona internals
    (do not say "as a fulltime student"; just adjust the voice).

Persona: {{ profile_summary }}

Tasks:
{{ tasks_block }}

Return ONLY a JSON array of length {{ num_tasks }}, in the SAME ORDER as
the input tasks:
[
  { "label": "<copy of label>", "personalized_description": "<your rewrite>" },
  ...
]"""
