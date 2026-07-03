"""Stage 2 + Stage 3 of the grounded task-generation pipeline.

Stage 2 (`paraphrase_for_persona`)
    Takes the verified Stage-1 task list and a persona profile,
    issues ONE batched LLM call to rewrite each task's description
    in the persona's voice, and returns the paraphrased descriptions
    aligned with the input task order.  No GraphRAG retrieval; the
    canonical strings are already in the prompt.

Stage 3 (`apply_gates`)
    Runs the three gates from :mod:`gates` on every (canonical,
    paraphrase) pair and returns the decisions in input order.  The
    decisions carry both texts plus every measured value, ready for
    the telemetry sidecar.

Both functions are pure: every collaborator (the LLM, the embedder,
the renderer, the parser, the debug-log writer) is injected so tests
can swap them for in-memory fakes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from src.scripts.scenarios.task_generation.gates import GateDecision, gate_paraphrase
from src.scripts.scenarios.task_generation.parser import extract_paraphrases
from src.scripts.scenarios.task_generation.prompt_templates import render

log = logging.getLogger(__name__)


@dataclass
class ParaphraseStats:
    """Counters surfaced into the telemetry sidecar for Stage 2."""

    attempts: int = 1
    parse_failed: bool = False
    wall_time_seconds: float = 0.0


@dataclass
class ParaphraseResult:
    """Return type of :func:`paraphrase_for_persona`."""

    paraphrases: list[str] = field(default_factory=list)
    stats: ParaphraseStats = field(default_factory=ParaphraseStats)
    raw_prompt: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Stage 2; persona paraphrase
# ---------------------------------------------------------------------------

LLMInvoker = Callable[[str], str]
"""Take a prompt string and return the LLM's raw response string."""


def paraphrase_for_persona(
    *,
    invoke: LLMInvoker,
    profile_summary: str,
    verified_tasks: list[dict[str, Any]],
    length_delta_pct: float,
) -> ParaphraseResult:
    """Issue one batched LLM call and return the per-task paraphrases.

    Args:
        invoke: any callable that takes a prompt and returns the raw
            LLM response text (the production wiring usually wraps an
            `LLM.invoke(prompt).content` round-trip).
        profile_summary: human-readable persona profile string used in
            the paraphrase prompt's persona block.
        verified_tasks: the Stage-1 output; each dict must carry at
            least `label`, `display_name` and `description`.
        length_delta_pct: written into the prompt so the LLM aims for
            paraphrases within the same budget the gates enforce
            (e.g. 0.10 to "±10 %").

    Returns:
        `ParaphraseResult` with one paraphrase per task in the same
        order.  When the LLM response cannot be parsed or the count
        does not match, every paraphrase is the canonical description
        (so the gates downstream all pass length+emoji and the
        scenario keeps moving) and `stats.parse_failed` is `True`.
    """
    n = len(verified_tasks)
    tasks_block = json.dumps(
        [
            {
                "label": t.get("label"),
                "display_name": t.get("display_name", ""),
                "ontology_description": t.get("description", ""),
            }
            for t in verified_tasks
        ],
        ensure_ascii=False,
        indent=2,
    )
    prompt = render(
        "persona_paraphrase",
        num_tasks=n,
        length_delta_pct=int(round(length_delta_pct * 100)),
        profile_summary=profile_summary,
        tasks_block=tasks_block,
    )

    t0 = time.monotonic()
    raw_response = invoke(prompt) or ""
    wall = time.monotonic() - t0
    stats = ParaphraseStats(attempts=1, wall_time_seconds=wall)

    # Per-task fallback: canonical description.  Used when the parse
    # fails or when an entry is missing / malformed.
    canonical_fallback = [t.get("description", "") for t in verified_tasks]

    try:
        parsed = extract_paraphrases(raw_response)
    except ValueError as exc:
        log.warning("paraphrase_for_persona: response not parseable (%s)", exc)
        stats.parse_failed = True
        return ParaphraseResult(
            paraphrases=list(canonical_fallback),
            stats=stats,
            raw_prompt=prompt,
            raw_response=raw_response,
        )

    # Build a label to personalized_description map from the parsed
    # response, ignoring entries that don't carry both fields.  Then
    # reassemble in input task order so a missing entry falls back to
    # canonical without disturbing alignment.
    by_label: dict[str, str] = {}
    for entry in parsed:
        label = str(entry.get("label") or "").strip()
        text = entry.get("personalized_description")
        if label and isinstance(text, str) and text.strip():
            by_label[label] = text

    paraphrases: list[str] = []
    for task, fallback in zip(verified_tasks, canonical_fallback):
        label = str(task.get("label") or "").strip()
        paraphrases.append(by_label.get(label, fallback))

    if len(by_label) < n:
        log.info(
            "paraphrase_for_persona: matched %d/%d tasks by label; "
            "missing entries fell back to canonical.",
            len(by_label),
            n,
        )

    return ParaphraseResult(
        paraphrases=paraphrases,
        stats=stats,
        raw_prompt=prompt,
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Stage 3; apply the three gates
# ---------------------------------------------------------------------------


def apply_gates(
    *,
    verified_tasks: list[dict[str, Any]],
    paraphrases: Iterable[str],
    embedder: Any,
    length_delta_pct: float,
    similarity_threshold: float,
) -> list[GateDecision]:
    """Run :func:`gates.gate_paraphrase` for every (task, paraphrase) pair.

    The result list is aligned with the input `verified_tasks` order
    and is ready to be folded into both the per-persona telemetry and
    the final `RecommendedTask` description string.

    Args:
        verified_tasks: Stage-1 output (same dicts the paraphraser saw).
        paraphrases: Stage-2 output (one entry per task, same order).
        embedder: any object with `embed_query(text) -> list[float]`.
        length_delta_pct: max |Δlen| / len ratio for gate A.
        similarity_threshold: min cosine for gate C.
    """
    paraphrases = list(paraphrases)
    decisions: list[GateDecision] = []
    for task, paraphrase in zip(verified_tasks, paraphrases):
        canonical_desc = task.get("description", "") or ""
        display_name = task.get("display_name", "") or ""
        canonical_text = (
            f"{display_name}. {canonical_desc}" if display_name else canonical_desc
        )
        decision = gate_paraphrase(
            canonical_description=canonical_desc,
            canonical_text=canonical_text,
            paraphrase=paraphrase,
            embedder=embedder,
            length_delta_pct=length_delta_pct,
            similarity_threshold=similarity_threshold,
        )
        decisions.append(decision)
        if decision.failed_gate is not None:
            log.info(
                "paraphrase rejected (gate=%s, length_delta=%.3f, sim=%.3f) "
                "for label=%s; using canonical description.",
                decision.failed_gate,
                decision.length_delta,
                decision.similarity,
                task.get("label"),
            )
    return decisions
