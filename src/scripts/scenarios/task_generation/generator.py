"""Top-level task generator: scenario config + Person to list[RecommendedTask].

The `TaskGenerator` orchestrates the full task-generation pipeline:

1. **Prompt**; renders a named template with the person's profile, existing
   calendar labels, and scenario filters.
2. **Search**; calls the GraphRAG pipeline's `search()` method to ground
   the task candidates in the ontology knowledge graph.
3. **Parse**; extracts a JSON array of raw task dicts from the LLM answer.
4. **Enrich**; for each dict with an `ontology_uri`, queries Neo4j for
   `isDividable`, `isConcurrent`, `estimatedDurationMinutes`, and the
   difficulty level.  Ontology values fill in fields the LLM did not
   provide (LLM value always wins over ontology value).
5. **Construct**; converts enriched dicts to `RecommendedTask` objects, applying
   scenario-level `task_overrides` on top.

For `method: manual`, steps 2–4 are skipped and tasks are built directly
from `config.manual_tasks`.  For `method: template`, the same code path
as `graphrag` is used (the prompt template is already specified in config).

The GraphRAG pipeline is duck-typed: any object with a
`search(query_text: str, **kwargs) -> response` method works, where
`response.answer` is the raw LLM string.
"""

from __future__ import annotations

import logging
import math
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

from src.scripts.persona.domain.persona import Person
from src.scripts.scenarios.config.schema import TaskGenerationConfig
from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.telemetry import (
    TaskGenTelemetry,
    TaskTelemetryRow,
    estimate_tokens,
    write_taskgen_sidecar,
)
from src.scripts.scenarios.task_generation.debug_log import (
    STAGE_PARAPHRASE,
    STAGE_TASK_GEN,
    write_debug_log,
    write_proposed_tasks_log,
)
from src.scripts.scenarios.task_generation.fetch_loop import fetch_grounded_unique
from src.scripts.scenarios.task_generation.ontology_bridge import (
    BRANCH_URIS_BY_LOCAL_NAME,
    enrich_task_from_ontology,
    uri_resolves,
    validate_task_uri_groups,
)
from src.scripts.scenarios.task_generation.paraphrase import (
    apply_gates,
    paraphrase_for_persona,
)
from src.scripts.scenarios.task_generation.parser import (
    _task_from_dict,
    extract_raw_task_dicts,
)
from src.scripts.scenarios.task_generation.profile import (
    profile_block,
    profile_summary,
)
from src.scripts.scenarios.task_generation.prompt_templates import render

log = logging.getLogger(__name__)

# Minimum share of a week's generated tasks each named domain should take,
# so one abundant domain cannot fill the whole plan.
_MIN_DOMAIN_SHARE = 0.25


def _strip_uri_citation(description: str, uri: str) -> str:
    """Remove a trailing `[<uri>]` citation from *description*.

    The task-generation prompt instructs the LLM to append the
    ontology URI in square brackets at the end of the description.
    When that URI turns out to be fabricated, the citation must be
    removed too; otherwise the JSON / ICS output keeps a fake
    reference that looks authoritative.

    Returns the description unchanged when *description* or *uri*
    is empty / falsy, or when no matching trailing citation is found.
    """
    if not description or not uri:
        return description
    pattern = r"\s*\[" + re.escape(uri) + r"\]\s*$"
    return re.sub(pattern, "", description).rstrip()


class TaskGenerator:
    """Orchestrates task generation for a single person.

    Args:
        config: `TaskGenerationConfig` from the scenario YAML.
        graphrag_pipeline: optional GraphRAG pipeline; `None` causes
            non-manual methods to return `[]`.
        neo4j_session: optional open Neo4j session for ontology enrichment;
            `None` skips the enrichment step.  The session is reused for
            every `generate()` call; supply this when the caller is
            single-threaded.
        neo4j_driver: optional Neo4j driver.  When provided (and no
            `neo4j_session` is given) the generator opens a fresh
            session per `_enrich_dicts` call; Neo4j sessions are not
            thread-safe, but drivers are, so this lets the same
            `TaskGenerator` instance be safely reused across worker
            threads in a thread-pool.
    """

    def __init__(
        self,
        config: TaskGenerationConfig,
        graphrag_pipeline: Any = None,
        neo4j_session: Any = None,
        neo4j_driver: Any = None,
        *,
        # Grounded-path collaborators; optional, only the
        # `graphrag_grounded` method needs them.
        paraphrase_llm: Any = None,
        embedder: Any = None,
        scenario_id: str = "",
        tasks_dir: Path | str | None = None,
        provider: str = "openai",
        fetch_model: str = "",
        paraphrase_model: str = "",
        embedder_model: str = "",
    ) -> None:
        self._config = config
        self._pipeline = graphrag_pipeline
        self._session = neo4j_session
        self._driver = neo4j_driver
        self._paraphrase_llm = paraphrase_llm
        self._embedder = embedder
        self._scenario_id = scenario_id
        self._tasks_dir: Path | None = (
            Path(tasks_dir) if tasks_dir is not None else None
        )
        self._provider = provider
        self._fetch_model = fetch_model
        self._paraphrase_model = paraphrase_model
        self._embedder_model = embedder_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        person: Person,
        calendar: CalendarTrace,
        scenario_description: str = "",
        exclude_uris: set[str] | None = None,
    ) -> list[RecommendedTask]:
        """Generate desired-behavior tasks for one person.

        Args:
            person: runtime `Person` from the persona pipeline (provides
                the characteristics profile and existing event stages).
            calendar: the person's current `CalendarTrace` (existing events
                supply the list of already-scheduled activity labels).
            scenario_description: human-readable scenario goal injected into
                the prompt template.
            exclude_uris: ontology URIs that must not be reused, so an
                earlier week's grounded tasks are not repeated.

        Returns:
            List of `RecommendedTask` objects.  Returns `[]` when the method
            is `graphrag`/`template` but no pipeline is configured, or
            when the LLM response is empty / unparseable.
        """
        if self._config.method == "manual":
            return self._from_manual()

        if self._config.method == "graphrag_grounded":
            return self._generate_grounded(
                person, calendar, scenario_description, exclude_uris=exclude_uris
            )

        # "graphrag" and "template" share the same pipeline path.
        if self._pipeline is None:
            log.warning("TaskGenerator: no graphrag_pipeline configured; returning []")
            return []

        prompt = self._build_prompt(person, calendar, scenario_description)
        response = self._pipeline.search(query_text=prompt)
        raw_text: str = getattr(response, "answer", "") or ""

        # Per-attempt debug log; best-effort, only writes when
        # `LLM_DEBUG_DIR` (or the legacy `LLM_AGENT_DEBUG_DIR`)
        # is set.  No retry loop on the legacy path, so attempt=0.
        write_debug_log(
            person.person_id,
            STAGE_TASK_GEN,
            attempt=0,
            prompt=prompt,
            raw_response=raw_text,
            summary_lines=[f"method={self._config.method}"],
        )

        try:
            raw_dicts = extract_raw_task_dicts(raw_text)
        except ValueError as exc:
            log.warning("Could not parse task list from LLM response: %s", exc)
            return []

        raw_dicts = self._enrich_dicts(raw_dicts)
        return [_task_from_dict(d, self._config.task_overrides) for d in raw_dicts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _from_manual(self) -> list[RecommendedTask]:
        """Build `RecommendedTask` list directly from `config.manual_tasks`.

        Items in `config.manual_tasks` are guaranteed to be dicts by the
        `TaskGenerationConfig` Pydantic schema, so no isinstance check is needed.
        """
        return [
            _task_from_dict(d, self._config.task_overrides)
            for d in self._config.manual_tasks
        ]

    def _build_prompt(
        self,
        person: Person,
        calendar: CalendarTrace,
        scenario_description: str,
    ) -> str:
        """Render the configured prompt template."""
        existing_labels = sorted({e.label for e in calendar.events})
        groups = self._config.filter_groups()
        domains = sorted({d for group in groups for d in group.domains})
        difficulty = sorted({lvl for group in groups for lvl in group.difficulty})
        return render(
            self._config.prompt_template,
            num_tasks=self._config.num_tasks,
            domains=domains,
            difficulty=difficulty,
            occupation_status=person.occupation_status,
            profile_block=profile_block(person, self._config.profile_characteristics),
            existing_event_labels=existing_labels,
            scenario_description=scenario_description,
        )

    @contextmanager
    def _session_ctx(self) -> Iterator[Any]:
        """Yield a Neo4j session for the duration of an enrichment pass.

        Returns the explicitly-injected session when one is set; otherwise
        opens a fresh session from the injected driver and closes it on
        exit.  When neither is configured, yields `None` and the caller
        must short-circuit.
        """
        if self._session is not None:
            with nullcontext(self._session) as s:
                yield s
        elif self._driver is not None:
            with self._driver.session() as s:
                yield s
        else:
            yield None

    def _enrich_dicts(self, raw_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enrich raw task dicts with ontology properties from Neo4j.

        For each dict that has an `ontology_uri`, fetches
        `isDividable`, `isConcurrent`, `estimatedDurationMinutes`,
        `displayName` / `rdfs:label`, `rdfs:comment`, and
        `difficulty_level` from the graph.

        Merge rules:
        * **Ontology-wins fields**; `display_name` and `description`.
          When the ontology supplies a non-empty value, it OVERWRITES any
          LLM-supplied value (the ontology is the canonical source for
          these strings, including any authored emojis).
        * **Fill-only fields**; `is_dividable`, `is_concurrent`.
          Ontology fills in only when the LLM did not provide the field.
        * **Duration mapping**; `estimatedDurationMinutes` is mapped to
          both `duration_min` and `duration_max` only when neither is
          in the dict.

        URI validation
        ~~~~~~~~~~~~~~
        When an `ontology_uri` is supplied but does not resolve to any
        node in the graph (the LLM fabricated a plausible-looking URI),
        the dict is NOT enriched with ontology fields.  Instead the
        bridge logs a debug message and the URI is treated as
        fabricated: `ontology_uri` is set to `None` and any trailing
        `[<uri>]` citation in `description` is stripped so the
        downstream JSON / ICS output does not carry a fraudulent
        citation.  Tasks with no `ontology_uri` and tasks whose URI
        resolves are unaffected.
        """
        # Fields where the ontology beats whatever the LLM said. Authored
        # text (display name, description) flows through verbatim; the
        # boolean ontology facts (`hb:isConcurrent`, `hb:isDividable`)
        # also win, since the LLM has no way to know them and tends to
        # default both to `false`.
        ONTOLOGY_WINS = {"display_name", "description", "is_concurrent", "is_dividable"}

        with self._session_ctx() as session:
            if session is None:
                return raw_dicts

            enriched: list[dict[str, Any]] = []
            for raw in raw_dicts:
                uri = raw.get("ontology_uri")
                if uri:
                    raw = dict(raw)  # shallow copy; never mutate the caller's dict
                    if not uri_resolves(str(uri), session):
                        # DEBUG (not WARNING); this fires per fabricated URI,
                        # so a 30-persona run would print 30+ lines on top of
                        # the tqdm bar.  The aggregate is published once at
                        # evaluate-time as the `ontology_grounding` ratio
                        # (see `metrics/grounding.py`).
                        log.debug(
                            "Task %r carries fabricated ontology_uri %s; "
                            "stripping URI + citation.",
                            raw.get("label"),
                            uri,
                        )
                        raw["ontology_uri"] = None
                        if "description" in raw:
                            raw["description"] = _strip_uri_citation(
                                raw["description"], str(uri)
                            )
                        enriched.append(raw)
                        continue

                    props = enrich_task_from_ontology(str(uri), session)
                    for key, value in props.items():
                        if key == "duration_minutes":
                            dm = int(value)
                            if "duration_min" not in raw:
                                raw["duration_min"] = dm
                            if "duration_max" not in raw:
                                raw["duration_max"] = dm
                        elif key in ONTOLOGY_WINS:
                            raw[key] = value
                        elif key not in raw:
                            raw[key] = value
                enriched.append(raw)
            return enriched

    # ------------------------------------------------------------------
    # Grounded path (Stage 1 + Stage 2 + Stage 3)
    # ------------------------------------------------------------------

    def _profile_summary(self, person: Person) -> str:
        """One-line persona profile blob for the paraphrase prompt."""
        return profile_summary(person, self._config.profile_characteristics)

    def _render_fetch_prompt(
        self,
        person: Person,
        calendar: CalendarTrace,
        scenario_description: str,
        *,
        needed: int,
        accepted_names: list[str],
        accepted_uris: list[str],
        rejected_uris: list[str],
    ) -> str:
        """Render `health_improvement_with_blacklist` for one attempt.

        Names + URIs from earlier accepted tasks AND URIs that were
        rejected (fabricated / duplicate) in earlier attempts are
        formatted into bullet blocks so the LLM sees them inline and
        does not loop on the same proposals.
        """
        existing_labels = sorted({e.label for e in calendar.events})
        # The prompt hints the union of every group's domains and levels;
        # the per-URI gate then enforces each group's exact pairing.
        groups = self._config.filter_groups()
        domains = sorted({d for group in groups for d in group.domains})
        difficulty = sorted({lvl for group in groups for lvl in group.difficulty})

        def _block(items: list[str], formatter=lambda x: f"  - {x}") -> str:
            return "\n".join(formatter(i) for i in items) if items else "  (none yet)"

        return render(
            "health_improvement_with_blacklist",
            needed=needed,
            domains=domains,
            difficulty=difficulty,
            occupation_status=person.occupation_status,
            profile_block=profile_block(person, self._config.profile_characteristics),
            existing_event_labels=existing_labels,
            scenario_description=scenario_description,
            already_accepted_block=_block(
                accepted_names, formatter=lambda n: f"  - {n!r}"
            ),
            already_accepted_uri_block=_block(accepted_uris),
            rejected_uri_block=_block(rejected_uris),
        )

    def _generate_grounded(
        self,
        person: Person,
        calendar: CalendarTrace,
        scenario_description: str,
        exclude_uris: set[str] | None = None,
    ) -> list[RecommendedTask]:
        """Three-stage grounded generation.  Returns N (or fewer) tasks.

        Stage 1; fetch + verify + retry until `num_tasks` grounded
                  unique tasks are accepted (or `max_fetch_retries`
                  exhausted).
        Stage 2; single batched LLM paraphrase, persona-aware.
        Stage 3; three-gate acceptance; failures fall back to
                  canonical descriptions.

        Telemetry is collected at every layer and persisted to
        `<tasks_dir>/_telemetry/<pid>.json` when `tasks_dir` was
        injected at construction.
        """
        if self._pipeline is None:
            log.warning(
                "TaskGenerator (grounded): no graphrag_pipeline configured; "
                "returning []"
            )
            return []

        telemetry = TaskGenTelemetry(
            person_id=person.person_id,
            scenario_id=self._scenario_id,
            method=self._config.method,
            provider=self._provider,
            fetch_model=self._fetch_model,
            paraphrase_model=self._paraphrase_model,
            embedder_model=self._embedder_model,
        )

        # ---------------- Stage 1; fetch + verify + retry -----------
        with self._session_ctx() as session:
            if session is None:
                log.warning(
                    "TaskGenerator (grounded): no Neo4j session; cannot "
                    "verify URIs.  Returning []."
                )
                return []

            # Resolve each filter group's domain local names to ontology
            # branch URIs.  An unknown local name (typo, new branch the
            # bridge doesn't know yet) is silently dropped; an empty
            # branch set means no domain check for that group rather than
            # rejecting everything.
            groups: list[tuple[set[str], list[str]]] = []
            present_branches: set[str] = set()
            for group in self._config.filter_groups():
                allowed_branches = {
                    BRANCH_URIS_BY_LOCAL_NAME[name]
                    for name in group.domains
                    if name in BRANCH_URIS_BY_LOCAL_NAME
                }
                groups.append((allowed_branches, list(group.difficulty)))
                present_branches |= allowed_branches

            def _resolves(uri: str) -> tuple[bool, str, str | None]:
                return validate_task_uri_groups(uri, session, groups=groups)

            # Reserve at least `_MIN_DOMAIN_SHARE` of the week's tasks for
            # each named domain so one abundant branch cannot fill it.
            floor_count = (
                math.ceil(_MIN_DOMAIN_SHARE * self._config.num_tasks)
                if present_branches
                else 0
            )
            domain_floor = (
                {branch: floor_count for branch in present_branches}
                if floor_count
                else None
            )

            # Per-attempt prompt + response so we can estimate tokens
            # (the GraphRAG pipeline.search response doesn't expose
            # usage_metadata at this layer).  Index-aligned with
            # `fetch_result.attempts`.
            attempt_payloads: list[tuple[str, str]] = []

            def _writer(
                attempt: int, prompt: str, raw_response: str, summary_lines
            ) -> None:
                attempt_payloads.append((prompt, raw_response))
                write_debug_log(
                    person.person_id,
                    STAGE_TASK_GEN,
                    attempt,
                    prompt=prompt,
                    raw_response=raw_response,
                    summary_lines=summary_lines,
                )

            def _render(
                needed: int,
                accepted_names: list[str],
                accepted_uris: list[str],
                rejected_uris: list[str],
            ) -> str:
                return self._render_fetch_prompt(
                    person,
                    calendar,
                    scenario_description,
                    needed=needed,
                    accepted_names=accepted_names,
                    accepted_uris=accepted_uris,
                    rejected_uris=rejected_uris,
                )

            fetch_result = fetch_grounded_unique(
                pipeline=self._pipeline,
                num_tasks=self._config.num_tasks,
                max_retries=self._config.max_fetch_retries,
                render_prompt=_render,
                parse_response=extract_raw_task_dicts,
                uri_resolves=_resolves,
                debug_writer=_writer,
                top_k=self._config.fetch_top_k,
                domain_floor=domain_floor,
                exclude_uris=exclude_uris,
            )

            for idx, attempt_stats in enumerate(fetch_result.attempts):
                # `pipeline.search()` doesn't surface usage_metadata
                # here; estimate input/output tokens with the same
                # `cl100k_base` tokenizer the OpenAI API uses (with a
                # char/4 fallback when tiktoken is unavailable).
                prompt_text, response_text = (
                    attempt_payloads[idx] if idx < len(attempt_payloads) else ("", "")
                )
                telemetry.record_llm_call(
                    stage="fetch",
                    input_tokens=estimate_tokens(prompt_text),
                    output_tokens=estimate_tokens(response_text),
                    wall_time_seconds=attempt_stats.wall_time_seconds,
                )

            telemetry.record_fetch_summary(
                num_tasks_requested=self._config.num_tasks,
                num_tasks_accepted=len(fetch_result.accepted),
                per_attempt_fabricated=[
                    a.fabricated_dropped for a in fetch_result.attempts
                ],
                per_attempt_duplicate=[
                    a.duplicate_dropped for a in fetch_result.attempts
                ],
                per_attempt_parse_failed=[
                    a.parse_failed for a in fetch_result.attempts
                ],
            )

            # Per-persona audit log: every URI the LLM proposed across
            # all attempts (with verdict) so the user can grep one file
            # to see whether the LLM is fabricating, picking off-domain
            # URIs, or just churning.  Best-effort; silently skipped
            # when no LLM_DEBUG_DIR is set.
            all_proposals: list[dict[str, Any]] = []
            for a in fetch_result.attempts:
                for entry in a.proposals:
                    all_proposals.append({"attempt": a.attempt, **entry})
            write_proposed_tasks_log(
                person.person_id,
                scenario_id=self._scenario_id,
                accepted=fetch_result.accepted,
                proposals=all_proposals,
            )

            verified = fetch_result.accepted
            if not verified:
                self._persist_telemetry(telemetry)
                return []

            # Enrich the verified dicts with ontology props (display_name,
            # description, duration, etc.) using the existing helper -
            # this is the "ontology wins" merge already covered by tests.
            enriched = self._enrich_dicts_inner(verified, session)

        # Map fetch_attempt per accepted task (1-based).  Walk attempts
        # in order, attribute each accepted task to the attempt where
        # it was first counted.
        per_task_fetch_attempt: list[int] = []
        for a in fetch_result.attempts:
            per_task_fetch_attempt.extend([a.attempt] * a.accepted)

        # ---------------- Stage 2 + Stage 3 (optional) ---------------
        decisions = None
        paraphrases: list[str] | None = None
        if self._config.paraphrase and self._paraphrase_llm is not None:

            def _invoke(prompt: str) -> str:
                # `langchain`-style: invoke returns an AIMessage with
                # `.content`.  Fall back to `str(response)` if the
                # injected LLM returns a plain string (test fakes do).
                response = self._paraphrase_llm.invoke(prompt)
                content = getattr(response, "content", response)
                # Track tokens when the model exposes `usage_metadata`;
                # otherwise estimate from the prompt + response strings
                # so the telemetry sidecar always has a real number.
                usage = getattr(response, "usage_metadata", None) or {}
                in_t = int(usage.get("input_tokens") or 0) or estimate_tokens(prompt)
                response_text = str(content) if content is not None else ""
                out_t = int(usage.get("output_tokens") or 0) or estimate_tokens(
                    response_text
                )
                _invoke._captured_usage = (in_t, out_t)  # type: ignore[attr-defined]
                return response_text

            _invoke._captured_usage = (0, 0)  # type: ignore[attr-defined]

            t0 = time.monotonic()
            paraphrase_result = paraphrase_for_persona(
                invoke=_invoke,
                profile_summary=self._profile_summary(person),
                verified_tasks=enriched,
                length_delta_pct=(self._config.paraphrase_max_length_delta_pct),
            )
            wall = time.monotonic() - t0
            in_t, out_t = _invoke._captured_usage  # type: ignore[attr-defined]
            telemetry.record_llm_call(
                stage="paraphrase",
                input_tokens=in_t,
                output_tokens=out_t,
                wall_time_seconds=wall,
            )

            write_debug_log(
                person.person_id,
                STAGE_PARAPHRASE,
                attempt=0,
                prompt=paraphrase_result.raw_prompt,
                raw_response=paraphrase_result.raw_response,
                summary_lines=[
                    f"parse_failed={paraphrase_result.stats.parse_failed}",
                    f"wall_time_seconds={wall:.3f}",
                ],
            )

            paraphrases = paraphrase_result.paraphrases

            if self._embedder is not None:
                t1 = time.monotonic()
                decisions = apply_gates(
                    verified_tasks=enriched,
                    paraphrases=paraphrases,
                    embedder=self._embedder,
                    length_delta_pct=(self._config.paraphrase_max_length_delta_pct),
                    similarity_threshold=(self._config.paraphrase_similarity_threshold),
                )
                gate_wall = time.monotonic() - t1
                # Two embedder calls per task (canonical + paraphrase).
                telemetry.record_embedder_call(
                    input_tokens=0,  # most embedders report n/a here
                    wall_time_seconds=gate_wall,
                )
            else:
                log.info(
                    "TaskGenerator (grounded): no embedder; skipping Stage 3 "
                    "and using paraphrases as-is."
                )

        # ---------------- Materialize RecommendedTasks -------------------
        tasks: list[RecommendedTask] = []
        for idx, raw in enumerate(enriched):
            row_paraphrase_used = "skipped"
            row_length_delta: float | None = None
            row_sim: float | None = None
            row_missing: dict[str, int] = {}
            row_extra: dict[str, int] = {}
            row_failed: str | None = None
            row_personalized = ""

            if decisions is not None:
                d = decisions[idx]
                # Mutate description on the dict before _task_from_dict
                # so the paraphrase / canonical choice flows through.
                raw = {**raw, "description": d.text}
                row_paraphrase_used = d.used
                row_length_delta = d.length_delta
                row_sim = d.similarity
                row_missing = dict(d.missing_emojis)
                row_extra = dict(d.extra_emojis)
                row_failed = d.failed_gate
                row_personalized = paraphrases[idx] if paraphrases is not None else ""
            elif paraphrases is not None:
                # Stage 2 ran but Stage 3 did not.
                raw = {**raw, "description": paraphrases[idx]}
                row_paraphrase_used = "paraphrase"
                row_personalized = paraphrases[idx]

            task = _task_from_dict(raw, self._config.task_overrides)
            tasks.append(task)

            telemetry.record_task(
                TaskTelemetryRow(
                    label=task.label,
                    ontology_uri=task.ontology_uri,
                    fetch_attempt=(
                        per_task_fetch_attempt[idx]
                        if idx < len(per_task_fetch_attempt)
                        else 0
                    ),
                    paraphrase_used=row_paraphrase_used,
                    length_delta=row_length_delta,
                    similarity=row_sim,
                    missing_emojis=row_missing,
                    extra_emojis=row_extra,
                    failed_gate=row_failed,
                    canonical_description=enriched[idx].get("description", ""),
                    personalized_description=row_personalized,
                )
            )

        self._persist_telemetry(telemetry)
        return tasks

    def _enrich_dicts_inner(
        self, raw_dicts: list[dict[str, Any]], session: Any
    ) -> list[dict[str, Any]]:
        """Enrichment loop split out so the grounded path can run it
        with an already-open session (avoids re-entering `_session_ctx`).

        Behaviour matches :meth:`_enrich_dicts`; the only difference is
        that the session is taken as an explicit argument.
        """
        ONTOLOGY_WINS = {"display_name", "description", "is_concurrent", "is_dividable"}
        enriched: list[dict[str, Any]] = []
        for raw in raw_dicts:
            uri = raw.get("ontology_uri")
            if uri:
                raw = dict(raw)
                # In the grounded path, every URI has already passed
                # `uri_resolves` in Stage 1, so we skip the
                # fabrication branch and go straight to the enrichment.
                props = enrich_task_from_ontology(str(uri), session)
                for key, value in props.items():
                    if key == "duration_minutes":
                        dm = int(value)
                        if "duration_min" not in raw:
                            raw["duration_min"] = dm
                        if "duration_max" not in raw:
                            raw["duration_max"] = dm
                    elif key in ONTOLOGY_WINS:
                        raw[key] = value
                    elif key not in raw:
                        raw[key] = value
                # Defensive strip: even though the new prompt instructs
                # the LLM NOT to append `[<uri>]` to the description,
                # any trailing citation that slipped through is removed
                # here so the paraphrase stage sees clean canonical text
                # and the downstream JSON / ICS output stays uncluttered.
                desc = raw.get("description")
                if isinstance(desc, str):
                    raw["description"] = _strip_uri_citation(desc, str(uri))
            enriched.append(raw)
        return enriched

    def _persist_telemetry(self, telemetry: TaskGenTelemetry) -> None:
        """Write the per-persona telemetry sidecar when `tasks_dir` is
        injected.  Best-effort: any I/O error is logged at DEBUG and
        swallowed so a generation run is never blocked by telemetry."""
        if self._tasks_dir is None:
            return
        try:
            write_taskgen_sidecar(telemetry, self._tasks_dir)
        except OSError as exc:
            log.debug(
                "Could not persist telemetry sidecar for %s: %s",
                telemetry.person_id,
                exc,
            )
