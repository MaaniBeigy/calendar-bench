"""Default length-matched placebo strings for `augment_oneshot` blocks.

A placebo replaces a block at render time so the prompt length stays
roughly constant when a block is ablated. Holding length constant
rules out "the prompt is shorter" as a confounder when interpreting
the per-block effect estimates.
"""

from __future__ import annotations

# Each entry is a one-line generic instruction matched roughly to the
# length of the block it stands in for.
DEFAULT_PLACEBOS: dict[str, str] = {
    "default_path": (
        "Place each task into a free gap unless a host event is a "
        "genuine concurrent fit."
    ),
    "fit_examples": (
        "Use judgement on whether the task and a candidate host event "
        "can happen at the same time."
    ),
    "rule_of_thumb_1": (
        "Treat words like during, while, with in a task title as hints "
        "rather than rules."
    ),
    "rule_of_thumb_2": (
        "Preparation tasks belong in the free gap that precedes their "
        "host event, not concurrent with it."
    ),
    "rule_of_thumb_3": (
        "When a task names or refines the same activity as a calendar "
        "event, schedule it concurrent with that event."
    ),
    "worked_example_standalone": (
        "Pick the first free gap wide enough to satisfy the task's " "duration_min."
    ),
    "worked_example_concurrent": (
        "When concurrent, the task interval must sit strictly inside "
        "the host interval."
    ),
    "worked_example_waking": (
        "Both task.start and task.end must lie inside the waking " "window."
    ),
    "hard_constraints": (
        "Respect duration band, waking window, intra-day non-overlap, "
        "and at most one intensity-4-or-higher task per date."
    ),
    "do_not_drop": (
        "Avoid setting scheduled to false; almost every persona has "
        "enough free time to place the task somewhere."
    ),
    "self_verify": (
        "Re-check waking window, duration band, and host containment "
        "before writing the JSON array."
    ),
    "split_dividable_tasks": (
        "When `dividable_ok` is true on a task, prefer to split it "
        "into two or more sibling sessions via the `pieces` array."
    ),
    "context_block": (
        "Some weeks include a CONTEXT summary listing the persona's "
        "context; such as mood, energy, location or weather at relevant times."
    ),
}
