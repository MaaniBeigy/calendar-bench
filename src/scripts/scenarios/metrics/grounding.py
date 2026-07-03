"""Ontology-grounding metric for the task-generation phase.

A *grounded* task is one whose `ontology_uri` survived the
generator's URI-validation pass; i.e. the URI resolved to a node in
the knowledge graph.  Tasks whose URI was fabricated by the LLM are
stripped to `ontology_uri=None` upstream, so they appear here as
ungrounded.

The metric is reported alongside the scheduling-loss components so
the benchmark publishes its own honesty score: a high
:func:`compute_grounding`'s `ratio` means the task generator is
producing tasks that genuinely cite the ontology, while a low ratio
flags an LLM hallucination problem that no augmenter can fix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable


def compute_grounding(
    tasks_dir: Path,
    person_ids: Iterable[str],
    *,
    expected_total: int | None = None,
    uri_validator: Callable[[str], bool] | None = None,
) -> dict:
    """Read `<tasks_dir>/<pid>_tasks.json` for each person and return
    grounding statistics.

    Two grounding tiers are reported:

    * **grounded**; count of tasks whose `ontology_uri` field is
      truthy.  This is a presence-only check; it trusts the generator
      to have produced valid URIs.
    * **verified**; count of tasks whose URI ALSO passes
      *uri_validator* (typically a partial of
      :func:`ontology_bridge.uri_resolves` bound to a live Neo4j
      session).  This is the safety-net check that catches URIs that
      no longer resolve in the live ontology; e.g. after an ontology
      rename or a generator regression.

    When *uri_validator* is None the verification step is skipped and
    the returned dict carries `verified=None` everywhere, so the
    consumer can render "verification not run" instead of "0/N
    verified".

    Args:
        tasks_dir: directory containing per-person task JSON files
            (the output of the `generate-tasks` phase).
        person_ids: identifiers whose task files should be inspected.
            Persons whose file does not exist or fails to parse are
            skipped silently; the metric is best-effort.
        expected_total: optional per-persona task-count target
            (typically `cfg.task_generation.num_tasks`).  When set,
            each per-person row carries an `expected` field plus a
            `short_fetched` boolean so the report can flag personas
            whose Stage-1 fetch loop ran out of retries.
        uri_validator: optional `(uri) -> bool` callable used to
            re-check every URI against the live ontology at eval time.
            When supplied, the returned dict carries the verified
            counts in addition to the presence-only counts.

    Returns:
        Dict with the shape::

            {
              "per_person": {
                "<pid>": {
                    "grounded":      int,
                    "verified":      int|None,
                    "total":         int,
                    "ratio":         float,
                    "verified_ratio": float|None,
                    "expected":      int|None,
                    "short_fetched": bool,
                    "unverified_uris": list[str],
                },
                ...
              },
              "grounded":                  int,
              "verified":                  int|None,
              "total":                     int,
              "ratio":                     float,
              "verified_ratio":            float|None,
              "expected_per_person":        int|None,
              "persons_short_fetched":      int,
              "persons_with_unverified":    int|None,
            }
    """
    tasks_dir = Path(tasks_dir)
    per_person: dict[str, dict] = {}
    grounded_total = 0
    verified_total = 0
    total_total = 0
    short_count = 0
    unverified_persons = 0
    for pid in person_ids:
        fp = tasks_dir / f"{pid}_tasks.json"
        if not fp.exists():
            continue
        try:
            tasks = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(tasks, dict):
            # Per-week wrapper: flatten to the union of every week.
            tasks = [t for week in (tasks.get("weeks") or []) for t in week]
        if not isinstance(tasks, list):
            continue
        grounded = 0
        verified = 0
        unverified_uris: list[str] = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            uri = t.get("ontology_uri")
            if not uri:
                continue
            grounded += 1
            if uri_validator is not None:
                if uri_validator(str(uri)):
                    verified += 1
                else:
                    unverified_uris.append(str(uri))
        total = len(tasks)
        is_short = expected_total is not None and total < expected_total
        if is_short:
            short_count += 1
        if uri_validator is not None and unverified_uris:
            unverified_persons += 1
        per_person[pid] = {
            "grounded": grounded,
            "verified": verified if uri_validator is not None else None,
            "total": total,
            "ratio": grounded / total if total else 0.0,
            "verified_ratio": (
                verified / total if (uri_validator is not None and total) else None
            ),
            "expected": expected_total,
            "short_fetched": is_short,
            "unverified_uris": unverified_uris,
        }
        grounded_total += grounded
        verified_total += verified
        total_total += total
    return {
        "per_person": per_person,
        "grounded": grounded_total,
        "verified": verified_total if uri_validator is not None else None,
        "total": total_total,
        "ratio": grounded_total / total_total if total_total else 0.0,
        "verified_ratio": (
            verified_total / total_total
            if (uri_validator is not None and total_total)
            else None
        ),
        "expected_per_person": expected_total,
        "persons_short_fetched": short_count,
        "persons_with_unverified": (
            unverified_persons if uri_validator is not None else None
        ),
    }
