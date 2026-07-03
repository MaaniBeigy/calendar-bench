"""Per-stage debug-log writer for the task-generation pipeline.

Mirrors the augmenter's debug-log convention so a single
`output/llm_debug/<pid>/...` tree captures every LLM call the
pipeline issues:

* augmenter writes `<pid>/<week-start>__attempt<N>.txt`;
* task generator writes `<pid>/<stage>__attempt<N>.txt` where
  *stage* is `task_gen` (Stage 1 fetch) or `paraphrase`
  (Stage 2 personalization).

Activated by the `LLM_DEBUG_DIR` environment variable; falls back
to `LLM_AGENT_DEBUG_DIR` for compatibility with existing setups.
A missing / unwritable directory short-circuits silently; debug
logging is best-effort and must never block a generation run.

The file format is identical to the augmenter's:

    === person ===
    <pid>

    === summary ===
    <one-line summary>

    === prompt ===
    <prompt body>

    === raw response ===
    <llm response>
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

# Stage names recognized by `write_debug_log`.  Anything else is
# accepted but a typo will silently land in a different file.
STAGE_TASK_GEN: str = "task_gen"
STAGE_PARAPHRASE: str = "paraphrase"


def _resolve_debug_dir() -> str | None:
    """Pick the active debug directory.

    Honours `LLM_DEBUG_DIR` first (the new pipeline-wide flag) and
    falls back to `LLM_AGENT_DEBUG_DIR` so existing user shells
    keep working.  Returns `None` when neither is set.
    """
    return os.environ.get("LLM_DEBUG_DIR") or os.environ.get("LLM_AGENT_DEBUG_DIR")


def write_debug_log(
    person_id: str,
    stage: str,
    attempt: int,
    *,
    prompt: str,
    raw_response: str,
    summary_lines: Iterable[str] = (),
) -> Path | None:
    """Write a single debug-log file; return its path or `None`.

    Args:
        person_id: persona identifier; used as the directory name so
            concurrent per-persona runs do not overwrite each other.
        stage: short tag for the call site (`task_gen`,
            `paraphrase`, …).  Becomes part of the file name.
        attempt: 0-based retry index.  Each retry writes its own file
            so the full audit trail is preserved.
        prompt: rendered prompt sent to the LLM.
        raw_response: LLM response (already a string; callers must
            decode bytes themselves).
        summary_lines: optional iterable of one-line summaries that
            land in the `=== summary ===` block; typically the
            counts of accepted / dropped / fabricated / duplicate
            tasks for that attempt.

    Returns:
        The written `Path` on success, `None` when the env var is
        unset or the write failed (best-effort).
    """
    debug_dir = _resolve_debug_dir()
    if not debug_dir:
        return None
    try:
        out_dir = Path(debug_dir) / person_id
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{stage}__attempt{attempt}.txt"
        summary = "\n".join(summary_lines) if summary_lines else "(no summary)"
        fname.write_text(
            f"=== person ===\n{person_id}\n\n"
            f"=== summary ===\n{summary}\n\n"
            f"=== prompt ===\n{prompt}\n\n"
            f"=== raw response ===\n{raw_response}\n",
            encoding="utf-8",
        )
        return fname
    except OSError:
        return None


def write_proposed_tasks_log(
    person_id: str,
    *,
    scenario_id: str,
    accepted: list[dict[str, Any]],
    proposals: Iterable[dict[str, Any]],
) -> Path | None:
    """Aggregate every URI the LLM proposed across all fetch attempts
    into one human-readable JSON file per persona.

    The file lands at
    `<LLM_DEBUG_DIR>/<person_id>/tasks_proposed.json` and carries:

    * `accepted`; the tasks that survived the validator and reached
      the augmenter (label, display_name, ontology_uri).
    * `proposed`; every URI the LLM emitted across all attempts,
      labeled with the validator's verdict (`ok` / `duplicate` /
      `not_instance` / `wrong_branch` / `wrong_level` /
      `missing_uri`).  Lets the user see at a glance whether the
      LLM is fabricating, picking off-domain URIs, or just churning.

    Returns the written path on success, `None` when the env var
    is unset or the write fails (best-effort, never blocks the run).
    """
    debug_dir = _resolve_debug_dir()
    if not debug_dir:
        return None
    try:
        out_dir = Path(debug_dir) / person_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tasks_proposed.json"
        # Compact verdict-rollup so the user sees the headline at the
        # top of the file without parsing the per-task list.
        verdict_counts: dict[str, int] = {}
        proposals_list = list(proposals)
        for entry in proposals_list:
            v = str(entry.get("verdict", "?"))
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        payload = {
            "person_id": person_id,
            "scenario_id": scenario_id,
            "accepted_count": len(accepted),
            "proposed_count": len(proposals_list),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "accepted": [
                {
                    "label": t.get("label"),
                    "display_name": t.get("display_name"),
                    "ontology_uri": t.get("ontology_uri"),
                }
                for t in accepted
            ],
            "proposed": proposals_list,
        }
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path
    except OSError:
        return None
