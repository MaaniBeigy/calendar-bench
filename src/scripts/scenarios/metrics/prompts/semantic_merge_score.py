"""LLM prompt: `semantic_merge_score`, requests a 0-1 compatibility score
between a task (A) and a calendar event (B) used by the LLM-backed
`SemanticCompatibility` scorer in `metrics/llm_judge.py`.
"""

from __future__ import annotations

# Bump when TEMPLATE changes so the judge cache re-scores instead of
# returning values produced by an older scale.
SEMANTIC_MERGE_SCORE_VERSION = "v2"

TEMPLATE = """\
A personal calendar already contains a scheduled event (B). A scheduling
system wants to attach a new task (A) to that event, meaning the person would
perform the task DURING that slot, at the same time as B.

Task          A: {{ label_a }}
Calendar event B: {{ label_b }}

Score whether a person already doing event B can genuinely do task A AT THE
SAME TIME, without stopping or leaving B. Ask: can and should someone who is
in B also do A right then?

Scale:
  0.0: cannot co-occur; doing A means stopping or leaving B (e.g. "yoga" during "driving")
  0.3: only loosely related, or A interrupts B or needs the same hands and attention (e.g. "step outside for fresh air" during "lunch", "ask someone for support" during "a meeting")
  0.5: possible but a stretch; A is a separate activity that sits near B, not truly during it
  0.7: genuine concurrent fit; A naturally happens during B without disrupting it (e.g. "mindful eating" during "lunch", "a few squats" during "desk work", "a breathing exercise" during "a calm break")
  0.9: strong fit; A clearly belongs inside B
  1.0: A and B are the SAME activity, differing only in intensity, count, or focus (e.g. "walk 5,000 steps" during "walking", "write three gratitudes" during "journaling")

Important: if A and B are the SAME TYPE of activity, that is a perfect fit
(1.0); the person is already in that context.

Return ONLY valid JSON, no markdown, no explanation:
{"score": <float 0.0 to 1.0>, "reason": "<one sentence>"}"""
