"""Stage 1 of the grounded task-generation pipeline; fetch + verify + retry.

Loops the GraphRAG `search` call up to `max_retries` times, validating
every returned URI against the live ontology and rejecting duplicates,
until the persona has `num_tasks` verified unique tasks (or the retry
budget is exhausted).

The function is intentionally pure: every collaborator (the GraphRAG
pipeline, the Neo4j session, the prompt renderer, the debug-log writer,
and the telemetry sink) is injected.  Tests exercise the loop with
in-memory fakes; production wires the real implementations.

Each retry rebuilds the prompt with the *do-not-repeat* URIs already
accepted, so the LLM never sees a clean slate after attempt 1.

Outputs the verified task dicts in acceptance order plus a per-attempt
summary that the telemetry layer persists for cost / quality auditing.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)


@dataclass
class FetchAttemptStats:
    """Per-attempt counters surfaced into the telemetry sidecar."""

    attempt: int
    proposed: int = 0
    accepted: int = 0
    fabricated_dropped: int = 0
    duplicate_dropped: int = 0
    wrong_branch_dropped: int = 0
    wrong_level_dropped: int = 0
    # Number of URIs accepted but with branch-resolution returning
    # None; i.e. accepted under the validator's graceful-degradation
    # policy.  A consistently non-zero value signals the live Neo4j
    # graph isn't exposing rdf:type to SUBCLASSOF chains as expected.
    branch_unresolved_passes: int = 0
    parse_failed: bool = False
    wall_time_seconds: float = 0.0
    # `proposals` records every URI the LLM emitted in this attempt
    # plus the verdict the validator returned for it.  Aggregated
    # across attempts into the per-persona `tasks_proposed.json`.
    # Each entry: `{"uri": str, "verdict": str, "label": str|None,
    # "display_name": str|None}`.
    proposals: list[dict] = field(default_factory=list)


@dataclass
class FetchResult:
    """Return type of :func:`fetch_grounded_unique`."""

    accepted: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[FetchAttemptStats] = field(default_factory=list)
    short_fetched: bool = False  # True iff len(accepted) < num_tasks
    # Accepted count per resolved branch URI; empty when the resolver
    # does not report branches (the legacy bool / 2-tuple contract).
    branch_counts: dict[str, int] = field(default_factory=dict)


# Type aliases for the injected callables; small enough to inline at call
# sites but documented for future maintainers.
PromptRenderer = Callable[[int, list[str], list[str], list[str]], str]
"""Build the fetch prompt from
`(needed, already_accepted_display_names, already_accepted_uris,
rejected_uris)`.

`rejected_uris` covers every URI that was proposed in an earlier
attempt but failed verification (fabricated or duplicate).  Carrying
them forward stops the LLM from looping on the same fabrications
across retries; an earlier regression where 4
straight retries proposed only `take-a-short-walk` /
`prepare-a-healthy-snack`."""

ResponseParser = Callable[[str], list[dict[str, Any]]]
"""Parse the LLM's raw answer into a list of raw task dicts."""

UriResolver = Callable[[str], "tuple[bool, str, str | None] | tuple[bool, str] | bool"]
"""Validate a candidate URI.  Returns a plain `bool` (legacy), a
`(ok, reason)` tuple, or a `(ok, reason, branch)` tuple where `branch`
is the resolved domain URI used for the per-domain floor.  Reason
strings recognized by the loop:

  * `"ok"`; accept;
  * `"not_instance"`; node missing or class-only;
  * `"wrong_branch"`; off-domain;
  * `"wrong_level"`; off-difficulty.

Plain `bool` returns are treated as `"ok"` / `"not_instance"`."""

DebugWriter = Callable[[int, str, str, Iterable[str]], None]
"""Persist (attempt, prompt, raw_response, summary_lines) to disk."""


def fetch_grounded_unique(
    *,
    pipeline: Any,
    num_tasks: int,
    max_retries: int,
    render_prompt: PromptRenderer,
    parse_response: ResponseParser,
    uri_resolves: UriResolver,
    debug_writer: DebugWriter | None = None,
    top_k: int | None = None,
    domain_floor: dict[str, int] | None = None,
    exclude_uris: set[str] | None = None,
) -> FetchResult:
    """Run the fetch-and-verify loop and return `num_tasks` verified
    unique tasks (or fewer if the retry budget runs out).

    Args:
        pipeline: any object exposing `search(query_text=str)`;
            its `response.answer` is the LLM's raw output.
        num_tasks: target number of grounded unique tasks.
        max_retries: hard cap on the number of LLM calls (≥ 1).
        render_prompt: builds the fetch prompt from
            `(needed, accepted_display_names, accepted_uris)`.
        parse_response: parses the LLM's raw answer into a list of
            raw task dicts (each carrying at least `ontology_uri`).
            May raise `ValueError`; the loop treats that as a
            parse-failed attempt and continues.
        uri_resolves: function that returns `True` iff a URI
            resolves to a node in the ontology graph (typically a
            partial-application of :func:`ontology_bridge.uri_resolves`
            with the open Neo4j session bound).
        debug_writer: optional callable invoked once per attempt with
            `(attempt, prompt, raw_response, summary_lines)`.  Used
            by the production CLI to write per-persona debug logs.

    Returns:
        `FetchResult` carrying the accepted dicts, per-attempt stats,
        and a `short_fetched` flag.

    The returned dicts are SHALLOW COPIES of the LLM-supplied dicts -
    the loop does not mutate the caller's data.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be ≥ 1; got {max_retries}")
    if num_tasks < 1:
        raise ValueError(f"num_tasks must be ≥ 1; got {num_tasks}")

    accepted: list[dict[str, Any]] = []
    # `exclude_uris` (prior weeks' accepted URIs) seed the blacklist so a
    # cross-week-distinct request never re-proposes an earlier week's task.
    seen_uris: set[str] = set(exclude_uris or set())
    rejected_uris: set[str] = set()  # proposed-but-rejected URIs (fabricated)
    branch_counts: Counter[str] = Counter()
    floor = domain_floor or {}
    attempts: list[FetchAttemptStats] = []

    for attempt in range(1, max_retries + 1):
        # The acceptance loop below breaks the moment
        # `len(accepted) == num_tasks`, so `needed` is always ≥ 1 here.
        needed = num_tasks - len(accepted)

        stats = FetchAttemptStats(attempt=attempt)
        accepted_names = [
            str(d.get("display_name") or d.get("label") or "") for d in accepted
        ]
        prompt = render_prompt(
            needed,
            accepted_names,
            sorted(seen_uris),
            sorted(rejected_uris),
        )

        t0 = time.monotonic()
        # Forward `top_k` to the GraphRAG retriever when supplied; the
        # neo4j_graphrag library accepts it via `retriever_config`.
        # Without this the retriever defaults to 5 hits, which is too
        # narrow for fetch; the LLM ends up picking the same handful
        # of URIs across 30 personas.
        if top_k is not None:
            response = pipeline.search(
                query_text=prompt,
                retriever_config={"top_k": int(top_k)},
            )
        else:
            response = pipeline.search(query_text=prompt)
        raw_text = getattr(response, "answer", "") or ""
        stats.wall_time_seconds = time.monotonic() - t0

        try:
            raw_dicts = parse_response(raw_text)
        except ValueError as exc:
            log.debug("attempt %d parse failed: %s", attempt, exc)
            stats.parse_failed = True
            raw_dicts = []

        stats.proposed = len(raw_dicts)
        # Classify every proposal once into an ordered outcome list; the
        # acceptable ones (verdict deferred) are selected with a
        # per-domain floor preference, then recorded in proposal order so
        # the audit log keeps the order the model emitted.
        outcomes: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        attempt_seen: set[str] = set()
        for raw in raw_dicts:
            label = (raw.get("label") or "") if isinstance(raw, dict) else ""
            display_name = (
                (raw.get("display_name") or "") if isinstance(raw, dict) else ""
            )
            uri = (
                (raw.get("ontology_uri") or "").strip() if isinstance(raw, dict) else ""
            )
            if not uri:
                stats.fabricated_dropped += 1
                outcomes.append(
                    _reject_outcome(uri, "missing_uri", label, display_name)
                )
                continue
            if uri in seen_uris or uri in attempt_seen:
                stats.duplicate_dropped += 1
                rejected_uris.add(uri)  # carry forward; LLM is repeating itself
                outcomes.append(_reject_outcome(uri, "duplicate", label, display_name))
                continue
            attempt_seen.add(uri)
            ok, reason, branch = _normalise_verdict(uri_resolves(uri))
            if not ok:
                if reason == "wrong_branch":
                    stats.wrong_branch_dropped += 1
                elif reason == "wrong_level":
                    stats.wrong_level_dropped += 1
                else:
                    stats.fabricated_dropped += 1
                rejected_uris.add(uri)  # the next prompt's blacklist
                outcomes.append(_reject_outcome(uri, reason, label, display_name))
                continue
            cand = {
                "raw": raw,
                "uri": uri,
                "branch": branch,
                "reason": reason,
                "label": label,
                "display_name": display_name,
            }
            candidates.append(cand)
            outcomes.append({"verdict": None, "cand": cand})

        # Select up to `needed` candidates, greedily preferring a branch
        # still below its floor each pick so a small domain is not
        # crowded out by an abundant one. Without a floor this picks in
        # proposal order (the first `needed`).
        needed = num_tasks - len(accepted)
        running = dict(branch_counts)
        pool = list(candidates)
        selected: set[int] = set()
        while len(selected) < needed and pool:
            below = [
                c
                for c in pool
                if running.get(c["branch"], 0) < floor.get(c["branch"], 0)
            ]
            pick = below[0] if below else pool[0]
            pool.remove(pick)
            selected.add(id(pick))
            running[pick["branch"]] = running.get(pick["branch"], 0) + 1

        for outcome in outcomes:
            if outcome["verdict"] is not None:
                _record_proposal(
                    stats,
                    outcome["uri"],
                    outcome["verdict"],
                    outcome["label"],
                    outcome["display_name"],
                )
                continue
            cand = outcome["cand"]
            if id(cand) not in selected:
                continue  # valid but over the per-attempt cap; left for next attempt
            accepted.append(dict(cand["raw"]))  # shallow copy; never mutate caller
            seen_uris.add(cand["uri"])
            stats.accepted += 1
            if cand["branch"] is not None:
                branch_counts[cand["branch"]] += 1
            # Audit verdict: strict accept to `accepted`; graceful
            # branch-unresolved accept to `accepted_branch_unresolved`
            # so the user can distinguish them in tasks_proposed.json.
            if cand["reason"] == "ok_branch_unresolved":
                stats.branch_unresolved_passes += 1
                _record_proposal(
                    stats,
                    cand["uri"],
                    "accepted_branch_unresolved",
                    cand["label"],
                    cand["display_name"],
                )
            else:
                _record_proposal(
                    stats, cand["uri"], "accepted", cand["label"], cand["display_name"]
                )

        attempts.append(stats)

        if debug_writer is not None:
            debug_writer(
                attempt,
                prompt,
                raw_text,
                _summary_lines(stats, target=num_tasks, accumulated=len(accepted)),
            )

        if len(accepted) == num_tasks:
            break

    short = len(accepted) < num_tasks
    if short:
        log.error(
            "fetch_grounded_unique: only %d/%d grounded unique tasks after %d "
            "attempts (max_retries=%d); augmenter will see fewer tasks than "
            "configured.",
            len(accepted),
            num_tasks,
            len(attempts),
            max_retries,
        )

    return FetchResult(
        accepted=accepted,
        attempts=attempts,
        short_fetched=short,
        branch_counts=dict(branch_counts),
    )


def _reject_outcome(
    uri: str, verdict: str, label: str, display_name: str
) -> dict[str, Any]:
    """Build an ordered outcome record for a rejected proposal."""
    return {
        "verdict": verdict,
        "uri": uri,
        "label": label,
        "display_name": display_name,
    }


def _record_proposal(
    stats: FetchAttemptStats,
    uri: str,
    verdict: str,
    label: str,
    display_name: str,
) -> None:
    """Append one proposal verdict to the attempt's audit log."""
    stats.proposals.append(
        {
            "uri": uri,
            "verdict": verdict,
            "label": label or None,
            "display_name": display_name or None,
        }
    )


def _normalise_verdict(verdict_raw: Any) -> tuple[bool, str, str | None]:
    """Coerce a resolver return into `(ok, reason, branch)`."""
    if isinstance(verdict_raw, tuple):
        if len(verdict_raw) == 3:
            return bool(verdict_raw[0]), verdict_raw[1], verdict_raw[2]
        return bool(verdict_raw[0]), verdict_raw[1], None
    ok = bool(verdict_raw)
    return ok, "ok" if ok else "not_instance", None


def _summary_lines(
    stats: FetchAttemptStats, *, target: int, accumulated: int
) -> list[str]:
    """Build the `=== summary ===` block for the per-attempt debug log."""
    return [
        f"attempt={stats.attempt}",
        f"proposed={stats.proposed}",
        f"accepted={stats.accepted}",
        f"branch_unresolved_passes={stats.branch_unresolved_passes}",
        f"fabricated_dropped={stats.fabricated_dropped}",
        f"duplicate_dropped={stats.duplicate_dropped}",
        f"wrong_branch_dropped={stats.wrong_branch_dropped}",
        f"wrong_level_dropped={stats.wrong_level_dropped}",
        f"parse_failed={stats.parse_failed}",
        f"wall_time_seconds={stats.wall_time_seconds:.3f}",
        f"accumulated={accumulated}/{target}",
    ]
