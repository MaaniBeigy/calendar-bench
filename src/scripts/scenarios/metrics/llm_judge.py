"""LLM-as-Judge oracle for concurrent scheduling quality evaluation.

This oracle is ONLY instantiated in `_cmd_evaluate`. Augmentors are never
given access to it; augmenters perform purely structural validation, and
the oracle is the independent, ontology-agnostic source of truth for
`L_concurrent` (semantic quality of concurrent placements).

The oracle scores σ(task, event) ∈ [0, 1] answering:
  "Is the existing calendar event the right context or opportunity for
   this task?"

Two LLM invocation modes are supported (whichever is provided wins):

  • `client`; OpenAI-compatible object with `chat.completions.create`.
  • `invoke_fn`; a plain `Callable[[str], str]` that returns the raw
    response text.  Use this to plug in a `neo4j_graphrag.llm.LLMInterface`
    (`lambda p: llm.invoke(p).content`) or any other backend.

All scores are persisted to a JSONL cache keyed by the frozenset of the
two labels. Once a pair is scored it is never re-sent to the LLM. A
`cache_version` gates loading: records from a different version are
ignored so a changed scoring prompt re-scores instead of reusing old values.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.scripts.scenarios.task_generation.prompt_templates import render

if TYPE_CHECKING:
    from src.scripts.scenarios.domain.calendar import CalendarTrace
    from src.scripts.scenarios.domain.task import RecommendedTask


class LLMJudgeOracle:
    """Solver-independent LLM-as-Judge oracle for concurrent scheduling quality.

    Implements the same `.score(a, b)` interface as `SemanticCompatibility`
    so it can be passed directly to `compute_l_concurrent`.

    Args:
        client: OpenAI-compatible client with a `chat.completions.create`
                method.  Required when *invoke_fn* is not supplied.
        model: model name (default `"gpt-4o-mini"` for cost efficiency).
                Only used by the OpenAI-compatible path.
        cache_path: path to a JSONL file; loaded on init, appended on new scores.
        temperature: sampling temperature; 0.0 for deterministic scoring.
        invoke_fn: alternative single-call adapter; `Callable[[str], str]`
                that returns the raw LLM response text.  When supplied the
                oracle bypasses the OpenAI shape entirely.  Useful for plugging
                in `LLMInterface`-style clients.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "gpt-4o-mini",
        cache_path: Path | None = None,
        temperature: float = 0.0,
        invoke_fn: Callable[[str], str] | None = None,
        cache_version: str = "",
    ) -> None:
        if client is None and invoke_fn is None:
            raise ValueError(
                "LLMJudgeOracle requires either `client` (OpenAI-shaped) or "
                "`invoke_fn` (Callable[[str], str])."
            )
        self._client = client
        self._invoke_fn = invoke_fn
        self._model = model
        self._temperature = temperature
        self._version = cache_version
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[frozenset, float] = {}
        if self._cache_path is not None and self._cache_path.exists():
            self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        task_label: str,
        event_label: str,
        *,
        task_display: str = "",
        task_description: str = "",
    ) -> float:
        """Return σ ∈ [0, 1]: how well *event_label* serves as context for *task_label*.

        Calls the LLM on first access; subsequent calls for the same pair
        return the cached value without additional LLM calls.
        """
        cache_key: frozenset = frozenset({task_label, event_label})
        if cache_key in self._cache:
            return self._cache[cache_key]

        text_a = self._build_task_text(task_label, task_display, task_description)
        text_b = event_label.replace("_", " ")
        prompt = render("semantic_merge_score", label_a=text_a, label_b=text_b)
        sc = self._call_llm(prompt)
        self._save(cache_key, task_label, event_label, sc)
        return sc

    def score_all_pairs(
        self,
        tasks: list[RecommendedTask],
        calendar: CalendarTrace,
    ) -> dict[tuple[str, str], float]:
        """Score every unique (task_label, event_label) pair in one pass.

        Useful for pre-warming the cache before the per-person evaluation loop
        so subsequent calls are all cache hits.

        Returns:
            `{(task_label, event_label): score}` for all unique pairs.
        """
        unique_tasks: dict[str, RecommendedTask] = {}
        for t in tasks:
            unique_tasks.setdefault(t.label, t)
        # Score the actual activity plus its parent (e.g. "standup (office_work)")
        # so the judge distinguishes office_work episodes by their real label.
        unique_event_labels = {e.oracle_label for e in calendar.events}

        results: dict[tuple[str, str], float] = {}
        for t_label, task in unique_tasks.items():
            for e_label in sorted(unique_event_labels):
                sc = self.score(
                    t_label,
                    e_label,
                    task_display=task.display_name,
                    task_description=task.description,
                )
                results[(t_label, e_label)] = sc
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_task_text(label: str, display: str, description: str) -> str:
        """Build the richest available text for the task side of the prompt."""
        if display and description:
            return f"{display}: {description}"
        if display:
            return display
        return label.replace("_", " ")

    def _call_llm(self, prompt: str) -> float:
        if self._invoke_fn is not None:
            raw = str(self._invoke_fn(prompt)).strip()
        else:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
        return self._parse_score(raw)

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Parse a float score from LLM JSON output; returns 0.0 on failure."""
        try:
            data = json.loads(raw)
            return max(0.0, min(1.0, float(data["score"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        return 0.0

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        try:
            raw = self._cache_path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if self._version and entry.get("prompt_version", "") != self._version:
                    continue
                key: frozenset = frozenset({entry["task_label"], entry["event_label"]})
                self._cache[key] = float(entry["score"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

    def _save(
        self,
        cache_key: frozenset,
        task_label: str,
        event_label: str,
        score: float,
    ) -> None:
        self._cache[cache_key] = score
        if self._cache_path is None:
            return
        record = json.dumps(
            {
                "task_label": task_label,
                "event_label": event_label,
                "score": score,
                "model": self._model,
                "prompt_version": self._version,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as fh:
            fh.write(record + "\n")
            fh.flush()
