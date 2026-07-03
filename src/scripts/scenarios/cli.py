"""CLI entry point for the scenarios pipeline.

Usage::

    python -m src.scripts.scenarios.cli generate-tasks --scenario YAML [--out-dir DIR] [--seed N]
    python -m src.scripts.scenarios.cli augment        --scenario YAML [--out-dir DIR] [--method METHOD]
    python -m src.scripts.scenarios.cli evaluate       --scenario YAML [--run-dir DIR]
    python -m src.scripts.scenarios.cli run            --scenario YAML [--out-dir DIR] [--seed N]

Exit codes: 0 success, 1 evaluation violations, 2 config / usage error.

End-to-end example (after generating base persona calendars)::

    python -m src.scripts.scenarios.cli run \\
      --scenario src/experiments/persona/example_experiment/scenarios.yaml \\
      --out-dir  ./output/example_experiment \\
      --seed     20260503

Diagnostics: set LOG_LEVEL=DEBUG (or pass --log-level DEBUG) to see per-step
timing.  The most common stall point is `build_graphrag()` waiting for Neo4j
or the LLM API; look for the "connecting …" / "pipeline ready" log lines.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.scripts.scenarios.config.loader import (
    ScenarioConfigError,
    load_config,
    load_scenario,
)
from src.scripts.scenarios.config.schema import (
    AugmentationConfig,
    AugmentationMethodConfig,
    CalendarSourceConfig,
    ExperimentScenariosConfig,
    LossWeights,
    ScenarioConfig,
    ScenarioDefinition,
    ScenarioOutputConfig,
)
from src.scripts.scenarios.task_generation.profile import profile_summary

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2

log = logging.getLogger("scenarios.cli")


# Third-party loggers that emit one INFO line per HTTP / Bolt round-trip.
# Demoted to WARNING so a long pipeline does not drown its own status lines.
# A user-requested DEBUG level lets them through unchanged.
_NOISY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "openai",
    "openai._base_client",
    "urllib3",
    "neo4j",
    "neo4j.notifications",
    "neo4j.io",
    "neo4j.pool",
)


def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if numeric > logging.DEBUG:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Private pipeline helpers
# ---------------------------------------------------------------------------


def _try_load_persona_run(cfg: ScenarioConfig, run_dir_override: Path | None = None):
    """Try to load a persona run.  Returns `(LoadedRun, run_dir)` on success,
    `(None, run_dir)` when the directory is missing, or `(error_str, run_dir)`
    on a load error.
    """
    from src.scripts.scenarios.calendar.loader import (
        CalendarLoaderError,
        load_persona_run,
    )

    run_dir_str = getattr(cfg.calendar, "run_dir", None)
    run_dir: Path | None = run_dir_override or (
        Path(run_dir_str) if run_dir_str else None
    )

    if run_dir is None or not run_dir.exists():
        return None, run_dir

    log.debug("Loading persona run from %s …", run_dir)
    t0 = time.monotonic()
    try:
        run = load_persona_run(
            run_dir=run_dir,
            event_config_path=(
                Path(cfg.calendar.persona_events)
                if cfg.calendar.persona_events
                else None
            ),
            rules_path=(
                Path(cfg.calendar.persona_rules) if cfg.calendar.persona_rules else None
            ),
            environment_path=(
                Path(cfg.calendar.persona_environment)
                if cfg.calendar.persona_environment
                else None
            ),
        )
        log.debug(
            "Persona run loaded: %d persons, %d days horizon (%.1fs)",
            len(run.traces),
            run.horizon_days,
            time.monotonic() - t0,
        )
        return run, run_dir
    except CalendarLoaderError as exc:
        return str(exc), run_dir


def _read_timeframe_resolved(path: Path) -> dict | None:
    """Return the `resolved` block from a `timeframe.json` sidecar, or None."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("resolved")


def _write_timeframe_resolved(path: Path, cfg: ScenarioConfig, window) -> None:
    """Persist the resolved `timeframe` window next to the augmented dir."""
    spec = getattr(cfg, "timeframe", None)
    payload = {
        "scenario_id": cfg.id or "",
        "resolved": {
            "start_date": window.start_date.isoformat(),
            "weeks": window.weeks,
            "days": window.days,
            "end_date_inclusive": window.end_date_inclusive.isoformat(),
        },
        "declared": (
            None
            if spec is None
            else {
                "scale": spec.scale,
                "start": (
                    spec.start.isoformat()
                    if hasattr(spec.start, "isoformat")
                    else spec.start
                ),
                "end": (
                    spec.end.isoformat() if hasattr(spec.end, "isoformat") else spec.end
                ),
            }
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scope_run_to_timeframe(run, cfg: ScenarioConfig):
    """Resolve `cfg.timeframe` against `run`; return `(scoped_run, window_or_none)`.

    Returns the input run unchanged with `window=None` when no timeframe
    is declared. When declared, slices the run to that window and returns
    the `ResolvedWindow` so callers can build a horizon hint.
    """
    from src.scripts.scenarios.calendar.loader import slice_run_for_window
    from src.scripts.scenarios.config.timeframe import (
        TimeframeError,
        resolve_timeframe,
    )

    spec = getattr(cfg, "timeframe", None)
    if spec is None:
        return run, None
    horizon_start = getattr(run, "horizon_start_date", None)
    if horizon_start is None:
        raise TimeframeError(
            "scenario declares `timeframe` but persona run has no "
            "horizon_start_date; rerun persona generate or load with an "
            "explicit environment.yaml"
        )
    window = resolve_timeframe(
        spec,
        horizon_start_date=horizon_start,
        horizon_weeks=run.horizon_days // 7,
        scenario_id=cfg.id or "",
    )
    return slice_run_for_window(run, window), window


def _read_person_json(run_dir: Path, json_path_rel: str) -> dict:
    """Read one per-person JSON from the persona run directory.

    The persona pipeline stores `json_path` relative to the container working
    directory (e.g. `"output/example_experiment/persons/p.json"`).  Try the path
    as-is first; fall back to run_dir-relative for test fixtures that store
    short paths like `"persons/p.json"`.
    """
    direct = Path(json_path_rel)
    p = direct if direct.exists() else run_dir / json_path_rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _person_from_json(person_data: dict):
    """Reconstruct a `Person` from the embedded persona snapshot in a person JSON."""
    from src.scripts.persona.domain.persona import Person

    try:
        return Person.model_validate(person_data.get("persona", {}))
    except Exception:
        return None


def _load_person_json_lookup(run_dir: Path) -> dict[str, str]:
    """Return `{person_id: json_path}` from `run_dir/index.json`.

    Empty dict when the index is missing or unparseable; callers treat
    that as "no L_pref v2 wiring available for this run".
    """
    idx_path = run_dir / "index.json"
    if not idx_path.exists():
        return {}
    try:
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        entries = data.get("persons", [])
        return {e["person_id"]: e["json_path"] for e in entries if "person_id" in e}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def _build_persona_constraints_for_trace(
    *,
    trace,
    run_dir,
    person_json_lookup: dict[str, str],
    event_config,
    window_map,
    horizon_days: int,
    horizon_start_date,
    mapper,
    experiment: str,
    scenario: str,
):
    """Materialise a `PersonaConstraints` for one persona, or `None`.

    Returns `None` (which makes the loss dispatcher fall back to the
    legacy `compute_l_pref`) when any of the required pieces are
    missing; most commonly when the per-person JSON cannot be loaded,
    or when `event_config` / `window_map` were not wired up at the
    scenario level.  We never crash the augment loop over an L_pref
    wiring problem; the legacy path keeps the run going.
    """
    if event_config is None or window_map is None:
        return None
    json_path_rel = person_json_lookup.get(trace.person_id)
    if json_path_rel is None:
        return None
    try:
        from src.scripts.persona.config.schema import Persona
        from src.scripts.scenarios.metrics.preference_constraints import (
            PersonaConstraints,
        )

        person_data = _read_person_json(Path(run_dir), json_path_rel)
        if not person_data:
            return None
        persona_data = person_data.get("persona") or {}
        if not persona_data:
            return None
        synthetic_persona = Persona(
            id=persona_data.get("persona_id")
            or persona_data.get("person_id")
            or trace.person_id,
            occupation_status=persona_data.get("occupation_status"),
            stages=persona_data.get("stages", []),
            event_overrides=persona_data.get("event_overrides", {}),
        )
        return PersonaConstraints(
            persona=synthetic_persona,
            event_config=event_config,
            window_map=window_map,
            horizon_days=horizon_days,
            horizon_start_date=horizon_start_date,
            mapper=mapper,
            experiment=experiment,
            scenario=scenario,
        )
    except Exception:
        return None


def _llm_settings_with_override(model_cfg):
    """Return an `LLMSettings` instance with `model_cfg` baked in, or `None`.

    `None` (the default) signals "use whatever `LLMSettings.from_env()`
    gives us" so the historical env-driven path keeps working unchanged.

    When `model_cfg` is set, the returned settings inherit every API key
    / base URL from the env defaults (the YAML never carries secrets)
    and override only the provider + the per-provider model field. Each
    branch is symmetric: the active provider's `*_model` is replaced
    with `model_cfg.model`; the other provider fields stay at their env
    defaults so a misconfigured fall-back doesn't crash with an empty
    string.
    """
    if model_cfg is None:
        return None
    from src.graphrag.config import LLMSettings

    base = LLMSettings.from_env()
    provider = model_cfg.provider or base.provider
    return LLMSettings(
        provider=provider,
        openai_api_key=base.openai_api_key,
        openai_model=(model_cfg.model if provider == "openai" else base.openai_model),
        anthropic_api_key=base.anthropic_api_key,
        anthropic_model=(
            model_cfg.model if provider == "anthropic" else base.anthropic_model
        ),
        openrouter_api_key=base.openrouter_api_key,
        openrouter_base_url=base.openrouter_base_url,
        openrouter_model=(
            model_cfg.model if provider == "openrouter" else base.openrouter_model
        ),
    )


def _augment_llm_identity(method: str, cfg) -> tuple[str, str]:
    """Resolve the provider/model the llm_agent augmenter actually calls.

    Mirrors the `_build_augmenter` override rule: only an llm_agent
    block whose provider/model the YAML set explicitly re-targets the
    client; otherwise env defaults apply and the model stays empty so
    the report renders `env-default`. The returned pair feeds the
    augment telemetry sidecar, whose pricing lookup is keyed on
    provider + model.
    """
    if method != "llm_agent":
        return "openai", ""
    la = getattr(cfg.augmentation, "llm_agent", None)
    if la is None or not ({"provider", "model"} & la.model_fields_set):
        return "openai", ""
    return (la.provider or "openai"), (la.model or "")


def _build_task_generator(cfg: ScenarioConfig, tasks_dir: Path | None = None):
    """Build a `TaskGenerator`; attempts GraphRAG pipeline when method
    is `graphrag` or `graphrag_grounded`.

    For the grounded path the function additionally constructs a
    paraphrase-only LLM (separate from the GraphRAG one) and an
    Embedder for the Stage 3 similarity gate.  Failures to build any
    of those collaborators are non-fatal; the generator falls back to
    the cheapest path it can run (e.g. canonical descriptions when
    the embedder is missing).

    Args:
        cfg: parsed scenario config.
        tasks_dir: directory where per-persona task JSONs are written;
            also doubles as the parent for the
            `_telemetry/<pid>.json` sidecars.  When `None` no
            sidecar is written.
    """
    from src.scripts.scenarios.task_generation.generator import TaskGenerator

    method = cfg.task_generation.method
    is_graphrag = method in {"graphrag", "graphrag_grounded"}

    pipeline = None
    driver = None
    paraphrase_llm = None
    embedder = None
    provider = "openai"
    fetch_model = ""
    paraphrase_model = ""
    embedder_model = ""

    # Resolve any per-stage LLM override declared on the scenario YAML.
    # `task_generator_model` re-targets BOTH the GraphRAG fetch LLM and
    # the paraphrase LLM (paraphrase belongs to the task-gen stage), so
    # the same override flows into every collaborator built below.
    task_gen_settings = _llm_settings_with_override(
        getattr(cfg, "task_generator_model", None)
    )

    if is_graphrag:
        log.debug(
            "Connecting to GraphRAG pipeline (Neo4j + LLM); "
            "this may take a few seconds …"
        )
        t0 = time.monotonic()
        try:
            from src.graphrag.config import LLMSettings, Neo4jSettings
            from src.graphrag.neo4j_client import make_driver
            from src.graphrag.ontology_metadata import by_key as _ontology_by_key
            from src.graphrag.ontology_metadata import uri_prefixes_for_keys
            from src.graphrag.pipeline import build_graphrag
            from src.scripts.scenarios.task_generation.ontology_bridge import (
                BRANCH_URIS_BY_LOCAL_NAME,
            )

            # Honour the scenario YAML's `task_generation.ontologies`
            # list: GraphRAG retrieval is constrained to URIs from the
            # ontologies the scenario actually asked for.  Without this
            # gate the shared vector index; which carries every
            # imported ontology; surfaces off-domain candidates that
            # the validator must reject, exhausting fetch retries with
            # zero accepted tasks (an earlier empty-context regression
            # observed once HumanActivities was added to the index).
            #
            # An ontology key with no declared `uri_namespaces`
            # contributes nothing to the gate; we warn so the scenario
            # author notices that listing it had no effect, then fall
            # back to unfiltered retrieval if NOTHING contributed (i.e.
            # legacy "background-only" configs keep working unchanged).
            requested = list(cfg.task_generation.ontologies)
            for name in requested:
                spec = _ontology_by_key(name)
                if spec is None:
                    log.warning(
                        "Scenario YAML lists unknown ontology %r "
                        "(no entry in ontology_metadata.ONTOLOGIES); "
                        "the GraphRAG retrieval gate will skip it.",
                        name,
                    )
                elif not spec.uri_namespaces:
                    log.debug(
                        "Ontology %r has no uri_namespaces declared; "
                        "it contributes background context but does "
                        "not gate retrieval for this scenario.",
                        name,
                    )
            uri_prefixes = uri_prefixes_for_keys(requested) or None

            # Push the scenario's branch + level filters and the
            # validator's instance-only check into the retrieval Cypher
            # itself, BUT only for the strict `graphrag_grounded`
            # path.  The legacy `graphrag` method emits unvalidated
            # output and historically tolerated background context, so
            # narrowing its retriever would change behavior for any
            # comparison run still using it.
            # When the YAML pinned a per-stage model, build the GraphRAG
            # pipeline with that LLM rather than the env default.
            override_llm = None
            if task_gen_settings is not None:  # pragma: no cover
                from src.graphrag.llm import make_llm as _make_llm

                try:
                    override_llm = _make_llm(task_gen_settings)
                except Exception as exc:  # pragma: no cover - cred misconf
                    log.warning(
                        "task_generator_model override unavailable (%s); "
                        "falling back to env-default LLM for GraphRAG.",
                        exc,
                    )

            if method == "graphrag_grounded":
                # The retriever pre-filter spans the union of every filter
                # group; the per-URI gate then enforces each group's exact
                # domain-difficulty pairing.
                groups = cfg.task_generation.filter_groups()
                allowed_branches = sorted(
                    {
                        BRANCH_URIS_BY_LOCAL_NAME[d]
                        for group in groups
                        for d in group.domains
                        if d in BRANCH_URIS_BY_LOCAL_NAME
                    }
                )
                allowed_levels = sorted(
                    {level for group in groups for level in group.difficulty}
                )
                pipeline = build_graphrag(
                    uri_prefixes=uri_prefixes,
                    allowed_branches=allowed_branches or None,
                    allowed_levels=allowed_levels or None,
                    instance_only=True,
                    llm=override_llm,
                )
            else:
                pipeline = build_graphrag(
                    uri_prefixes=uri_prefixes,
                    llm=override_llm,
                )
            try:
                driver = make_driver(Neo4jSettings.from_env())
                driver.verify_connectivity()
            except Exception as exc:
                log.warning(
                    "Neo4j driver unavailable for ontology enrichment: %s  "
                    "- task generation will skip dcterms title/description "
                    "propagation.",
                    exc,
                )
                driver = None

            # Provider + model labels for the telemetry sidecar. Honour any
            # per-stage override so the sidecar reflects the model that
            # actually ran (otherwise a benchmark sweep would label every
            # run with the env-default model; useless for A/B reporting).
            try:
                llm_settings = task_gen_settings or LLMSettings.from_env()
                provider = llm_settings.provider
                if provider == "openai":
                    fetch_model = paraphrase_model = llm_settings.openai_model
                elif provider == "anthropic":
                    fetch_model = paraphrase_model = llm_settings.anthropic_model
                else:
                    fetch_model = paraphrase_model = llm_settings.openrouter_model
            except Exception:  # pragma: no cover - depends on .env
                pass

            log.debug("GraphRAG pipeline ready (%.1fs).", time.monotonic() - t0)
        except Exception as exc:
            log.warning(
                "GraphRAG pipeline unavailable (%.1fs): %s  "
                "- task generation will return empty lists.",
                time.monotonic() - t0,
                exc,
            )

    # Paraphrase LLM + embedder are only needed for the grounded path.
    if method == "graphrag_grounded":
        if cfg.task_generation.paraphrase:
            try:
                from src.graphrag.llm import make_usage_llm

                # Same `task_generator_model` override applies to the
                # paraphrase LLM; it is part of the task-gen stage. The
                # usage-tracking client surfaces `usage_metadata`, which
                # the generator already prefers over token estimates.
                paraphrase_llm = make_usage_llm(task_gen_settings)
            except Exception as exc:
                log.warning(
                    "Paraphrase LLM unavailable (%s); Stage 2 will be "
                    "skipped and canonical descriptions used as-is.",
                    exc,
                )
            try:
                from src.graphrag.embeddings import EMBED_MODEL, make_embedder

                embedder = make_embedder()
                embedder_model = EMBED_MODEL
            except Exception as exc:
                log.warning(
                    "Embedder unavailable (%s); Stage 3 similarity gate "
                    "will be skipped (paraphrases ship as-is).",
                    exc,
                )

    scenario_id = cfg.id or ""
    return TaskGenerator(
        config=cfg.task_generation,
        graphrag_pipeline=pipeline,
        neo4j_driver=driver,
        paraphrase_llm=paraphrase_llm,
        embedder=embedder,
        scenario_id=scenario_id,
        tasks_dir=tasks_dir,
        provider=provider,
        fetch_model=fetch_model,
        paraphrase_model=paraphrase_model,
        embedder_model=embedder_model,
    )


def _write_augmented_timeline(
    *,
    trace,
    scheduled,
    out_path: Path,
) -> Path:
    """Emit one TSV row per (event, context, augmented_task) for the trace + scheduled set."""
    from src.scripts.persona.export.timeline_writer import (
        augmented_row,
        context_row,
        event_row,
        write_timeline_tsv,
    )

    catalog = None
    if trace.contexts:
        try:
            from src.scripts.persona.context.catalog import load_catalog

            catalog = load_catalog()
        except Exception:
            catalog = None

    rows: list[dict] = []
    person_id = trace.person_id
    persona_id = getattr(trace, "persona_id", "") or ""

    for ev in trace.events:
        rows.append(
            event_row(
                person_id,
                persona_id,
                label=ev.label,
                date=ev.date,
                start_minutes=ev.start_minutes,
                end_minutes=ev.end_minutes,
                is_concurrent=ev.is_concurrent,
                is_dividable=ev.is_dividable,
                intensity=ev.intensity,
            )
        )
    for ep in trace.contexts:
        rows.append(context_row(person_id, persona_id, episode=ep, catalog=catalog))
    for task in scheduled:
        rows.append(augmented_row(person_id, persona_id, task=task))

    return write_timeline_tsv(rows, out_path).path


def _task_to_dict(t) -> dict:
    """Serialise one `RecommendedTask` to its JSON form."""
    return {
        "label": t.label,
        "display_name": t.effective_display_name,
        "description": t.effective_description,
        "duration_min": t.duration_min,
        "duration_max": t.duration_max,
        "intensity": t.intensity,
        "is_dividable": t.is_dividable,
        "is_concurrent": t.is_concurrent,
        "ontology_uri": t.ontology_uri,
        "difficulty_level": t.difficulty_level,
    }


def _dict_to_task(d: dict):
    """Deserialise one task dict into a `RecommendedTask`."""
    from src.scripts.scenarios.domain.task import RecommendedTask

    return RecommendedTask(
        label=d["label"],
        display_name=d.get("display_name", ""),
        description=d.get("description", ""),
        duration_min=d.get("duration_min", 15),
        duration_max=d.get("duration_max", 60),
        intensity=d.get("intensity", 2),
        is_dividable=d.get("is_dividable", False),
        is_concurrent=d.get("is_concurrent", False),
        ontology_uri=d.get("ontology_uri"),
        difficulty_level=int(d.get("difficulty_level", 0) or 0),
    )


def _weeks_payload(data) -> list[list[dict]]:
    """Return the per-week task-dict lists from a parsed task file.

    A `{"weeks": [...]}` wrapper returns its weeks; a bare top-level
    array (the legacy single-week format) returns one week.
    """
    if isinstance(data, dict):
        return list(data.get("weeks") or [])
    return [list(data)]


def _write_tasks(tasks: list, path: Path) -> None:
    """Serialise a flat list of `RecommendedTask` objects to a JSON file."""
    data = [_task_to_dict(t) for t in tasks]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_tasks(path: Path) -> list:
    """Deserialise a task JSON file into a flat list of `RecommendedTask`.

    Tolerates both the legacy bare array and the per-week wrapper, which
    it flattens into the union of every week.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_dict_to_task(d) for week in _weeks_payload(data) for d in week]


def _write_weekly_tasks(weekly_tasks: list, path: Path) -> None:
    """Serialise per-week `RecommendedTask` batches to a `{"weeks": [...]}` file."""
    weeks = [[_task_to_dict(t) for t in week] for week in weekly_tasks]
    path.write_text(json.dumps({"weeks": weeks}, indent=2), encoding="utf-8")


def _read_weekly_tasks(path: Path) -> list:
    """Deserialise a task JSON file into per-week `RecommendedTask` lists.

    A bare array (legacy single-week format) loads as one week.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [[_dict_to_task(d) for d in week] for week in _weeks_payload(data)]


def _resolve_weekly_task_configs(configs: list, horizon_weeks: int) -> list:
    """Resolve task-generation configs to the per-week list to generate.

    A single block with no `week` returns a length-1 list (broadcast to
    every week). A per-week list returns one config per week `1..N`, an
    explicit `week` winning over the `week=None` default. Raises when a
    week has no config or an explicit `week` exceeds the horizon.
    """
    if len(configs) == 1 and configs[0].week is None:
        return [configs[0]]
    default = next((c for c in configs if c.week is None), None)
    by_week = {c.week: c for c in configs if c.week is not None}
    over = [w for w in by_week if w > horizon_weeks]
    if over:
        raise ValueError(
            f"task_generation week {max(over)} exceeds the "
            f"{horizon_weeks}-week horizon"
        )
    resolved = []
    for week in range(1, horizon_weeks + 1):
        chosen = by_week.get(week, default)
        if chosen is None:
            raise ValueError(
                f"task_generation has no config for week {week} and no default "
                "(an entry without `week`)"
            )
        resolved.append(chosen)
    return resolved


#: Fallback MET quartiles for the `L_disp` leg when
#: `output/met_quartiles.json` is absent.
_DEFAULT_MET_QUARTILES_VALUES = {"q1": 1.8, "q2": 3.0, "q3": 6.0, "max_met": 16.8}


def _load_met_quartiles_or_default():
    """Load `output/met_quartiles.json` if present; else use safe defaults."""
    from src.scripts.scenarios.metrics.intensity_resolver import (
        MetQuartiles,
        load_quartiles_file,
    )

    candidates = [
        Path("./output/met_quartiles.json"),
        Path("output/met_quartiles.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return load_quartiles_file(p)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Could not parse %s: %s; using defaults.", p, exc)
                break
    return MetQuartiles(**_DEFAULT_MET_QUARTILES_VALUES)


def _build_augmenter(
    method: str,
    run=None,
    llm_recorder=None,
    prompt_ablate: frozenset[str] | None = None,
    prompt_placebos: dict[str, str] | None = None,
    cfg: "ScenarioConfig | None" = None,
):
    """Instantiate the appropriate augmenter.  Returns `None` for unknown methods.

    *llm_recorder*; optional :class:`LLMCallRecorder` shared with the
    CLI augment loop so per-person token counts + wall-time can be
    captured into an :class:`AugmentTelemetry` sidecar.  Ignored for
    non-LLM methods.

    *prompt_ablate* / *prompt_placebos*; optional prompt-component
    ablation parameters forwarded to :class:`LLMAugmenter`. Ignored
    for non-LLM methods; ignored at render time when the configured
    template is not `augment_oneshot`.

    *cfg*; optional `ScenarioConfig`; required for the RL augmenter so
    the online reward matches the benchmark loss exactly. Ignored for
    other methods.
    """
    time_windows = getattr(run, "time_windows", {}) if run is not None else {}
    daily_window = getattr(run, "daily_window", None) if run is not None else None
    allen_rules = list(getattr(run, "allen_pair_rules", [])) if run is not None else []
    if method == "greedy":
        from src.scripts.scenarios.augmentation.greedy import GreedyAugmenter

        # Greedy is a pure FCFS dumb baseline; no SemanticCompatibility,
        # no intensity / MET awareness.  The RNG seed is derived from the
        # persona-pipeline environment seed so runs are reproducible.
        env_seed = (
            getattr(getattr(run, "environment", None), "seed", None)
            if run is not None
            else None
        )
        return GreedyAugmenter(
            time_windows=time_windows,
            daily_window=daily_window,
            seed=env_seed,
        )
    if method == "ptime":
        from src.scripts.scenarios.augmentation.ptime import PTimeAugmenter

        env_seed = (
            getattr(getattr(run, "environment", None), "seed", None)
            if run is not None
            else None
        )
        return PTimeAugmenter(
            time_windows=time_windows,
            daily_window=daily_window,
            allen_rules=allen_rules,
            seed=env_seed,
        )
    if method == "rl":
        return _build_rl_augmenter(
            run=run,
            cfg=cfg,
            time_windows=time_windows,
            daily_window=daily_window,
            allen_rules=allen_rules,
        )
    if method == "llm_agent":
        from src.scripts.scenarios.augmentation.llm_agent import (
            DirectLLMPipeline,
            LLMAugmenter,
        )

        # Separation of concerns: the augmenter places already-
        # grounded tasks into time slots; it does NOT need ontology
        # retrieval or URI citations.  Routing through `build_graphrag`
        # would (a) issue a wasted embedding API call per week, (b)
        # round-trip Neo4j for a vector search whose context the
        # placement prompt does not use, and (c) inflate the prompt
        # with 500-2000 tokens of ontology context blocks the LLM
        # ignores.  Ontology grounding has already happened at task-
        # generation time; the augmenter calls the LLM directly.
        pipeline = None
        try:
            from src.graphrag.llm import make_usage_llm

            # Honour the per-scenario `llm_agent.provider/model` pin. The
            # schema default-constructs an llm_agent block on every method
            # config, so only a block whose provider/model the YAML set
            # explicitly counts as an override; otherwise env defaults
            # apply as before.
            la = getattr(cfg.augmentation, "llm_agent", None) if cfg else None
            settings = None
            if la is not None and ({"provider", "model"} & la.model_fields_set):
                settings = _llm_settings_with_override(la)
            pipeline = DirectLLMPipeline(
                make_usage_llm(settings), recorder=llm_recorder
            )
        except Exception as exc:
            log.warning("LLM augmenter pipeline construction failed: %s", exc)
        return LLMAugmenter(
            llm_pipeline=pipeline,
            allen_rules=allen_rules,
            daily_window=daily_window,
            prompt_ablate=prompt_ablate,
            prompt_placebos=prompt_placebos,
        )
    return None


_WEEKLY_GAIN_FIELDS: tuple[tuple[str, str], ...] = (
    ("cov", "recommended_task_coverage"),
    ("cal", "task_event_and_task_task_temporal_relations"),
    ("pref", "user_preference_deviation"),
    ("disp", "intensive_task_dispersion"),
    ("merge", "semantic_coscheduling_merge"),
    ("spread", "recommended_task_spread"),
    ("divide", "dividable_task_split_reward"),
    ("context_fit", "user_context_recommendation_fit"),
)


def _weekly_components_to_gains(components) -> dict[str, float | None]:
    """Return the long-name `1 - loss` gain map for one week's components."""
    out: dict[str, float | None] = {}
    for attr, long_name in _WEEKLY_GAIN_FIELDS:
        value = getattr(components, attr)
        out[long_name] = None if value is None else 1.0 - float(value)
    return out


def _resolve_weekly_chunks_for_solution(solution, *, horizon_start_date=None):
    """Return consecutive calendar-week chunks anchored at the horizon start.

    Falls back to the first placement date; returns `[]` when nothing is scheduled.
    """
    from src.scripts.scenarios.augmentation.greedy import _get_weekly_chunks

    dates = sorted({st.date for st in solution.scheduled})
    if not dates:
        return []
    anchor = horizon_start_date if horizon_start_date is not None else dates[0]
    if anchor > dates[0]:
        anchor = dates[0]
    span_days = (dates[-1] - anchor).days + 1
    full = [anchor + datetime.timedelta(days=i) for i in range(span_days)]
    return _get_weekly_chunks(full)


def _evaluate_weekly_gain(
    *,
    solution,
    calendar,
    loss_fn,
    persona_run,
    persona_constraints,
    window_map,
    observed_categories,
    mask,
) -> list[dict]:
    """Score each ISO week of a reconstructed solution under the eval mask.

    Returns one record per week with the masked `weighted_gain` and the
    per-leg gain map, using the authoritative `compute_weekly` (judge
    oracle + persona constraints + the same per-person mask the totals
    use). Returns `[]` when nothing was scheduled.
    """
    time_windows = getattr(persona_run, "time_windows", None)
    weeks = _resolve_weekly_chunks_for_solution(
        solution,
        horizon_start_date=getattr(persona_run, "horizon_start_date", None),
    )
    records: list[dict] = []
    for week_idx, week_dates in enumerate(weeks):
        week_loss, week_comp = loss_fn.compute_weekly(
            solution,
            calendar,
            list(week_dates),
            time_windows=time_windows,
            persona_constraints=persona_constraints,
            window_map=window_map,
            observed_categories=observed_categories,
            mask=mask,
        )
        records.append(
            {
                "week_index": week_idx + 1,
                "week_start": week_dates[0].isoformat(),
                "weighted_gain": 1.0 - float(week_loss),
                "gains": _weekly_components_to_gains(week_comp),
            }
        )
    return records


def _write_evaluate_leg_sidecars(
    *,
    solution,
    trace,
    persons_dir: Path,
    person_id: str,
    components,
    persona_constraints,
    window_map,
    semantic,
    context_links,
    observed_categories,
) -> None:
    """Write the per-leg breakdown sidecars at evaluate time.

    Producer for `pref_violations`, `divide_verdicts`, and `context_fit`
    sidecars, all from the authoritative recompute. Each write is wrapped
    so a sidecar failure never aborts the evaluate loop.
    """
    if persona_constraints is not None and window_map is not None:
        try:
            from src.scripts.scenarios.export.json_writer import (
                write_preference_violations_sidecar,
            )
            from src.scripts.scenarios.metrics.loss import compute_l_pref_v2

            _agg, pattern_rows, leg_stats = compute_l_pref_v2(
                solution,
                persona_constraints=persona_constraints,
                window_map=window_map,
                semantic=semantic,
            )
            write_preference_violations_sidecar(
                persons_dir / f"{person_id}_pref_violations.jsonl",
                person_id=person_id,
                leg_stats=leg_stats,
                pattern_rows=pattern_rows,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "pref sidecar write failed for %s: %s (continuing)", person_id, exc
            )

    try:
        from src.scripts.scenarios.export.json_writer import (
            write_divide_verdicts_sidecar,
        )

        write_divide_verdicts_sidecar(
            persons_dir / f"{person_id}_divide_verdicts.jsonl",
            person_id=person_id,
            records=components.divide_verdicts,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "divide sidecar write failed for %s: %s (continuing)", person_id, exc
        )

    if context_links and trace is not None:
        try:
            from src.scripts.scenarios.export.json_writer import (
                write_context_fit_sidecar,
            )
            from src.scripts.scenarios.metrics.context_fit import collect_verdicts

            verdicts = collect_verdicts(
                solution,
                trace,
                context_links,
                observed_categories=observed_categories,
            )
            write_context_fit_sidecar(
                persons_dir / f"{person_id}_context_fit.jsonl",
                person_id=person_id,
                verdicts=verdicts,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "context_fit sidecar write failed for %s: %s (continuing)",
                person_id,
                exc,
            )


def _write_rl_training_sidecar_for_solution(
    *,
    solution,
    trace,
    persons_dir: Path,
) -> Path | None:
    """Write `<pid>_rl_training.json` when the augmenter attached telemetry.

    Returns `None` for non-RL augmenters (the attribute is absent) or
    empty trajectories. Wrapped in try/except so a sidecar failure never
    blocks the augment loop.
    """
    from src.scripts.scenarios.export.json_writer import write_rl_training_sidecar

    try:
        raw = getattr(solution, "rl_training_records", None)
        rows: list[dict] = list(raw) if isinstance(raw, list) else []
        if not rows:
            return None
        return write_rl_training_sidecar(
            persons_dir / f"{trace.person_id}_rl_training.json",
            person_id=trace.person_id,
            training_records=rows,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.debug(
            "rl_training sidecar skipped for %s: %s",
            getattr(trace, "person_id", "?"),
            exc,
        )
        return None


def _build_rl_augmenter(
    *,
    run=None,
    cfg=None,
    time_windows=None,
    daily_window=None,
    allen_rules=None,
    met_embedder=None,
):
    """Build the RL augmenter with the full per-scenario loss stack.

    Reuses the same evaluate-time builders (matcher, resolver, ruleset,
    semantic oracle, context links) so the DQN training reward matches
    the benchmark gain. Falls back to defaults when `cfg` is `None`.
    """
    from src.scripts.scenarios.augmentation.rl.augmenter import RLAugmenter
    from src.scripts.scenarios.config.schema import LossWeights
    from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
    from src.scripts.scenarios.metrics.intensity_resolver import IntensityResolver

    eval_cfg = getattr(cfg, "evaluation", None) if cfg is not None else None
    buffer_minutes = getattr(eval_cfg, "buffer_minutes", 30) if eval_cfg else 30
    merge_threshold = getattr(eval_cfg, "merge_threshold", 0.65) if eval_cfg else 0.65
    half_life_days = getattr(eval_cfg, "disp_half_life_days", 2.0) if eval_cfg else 2.0
    divide_tolerance_pct = (
        getattr(eval_cfg, "divide_duration_tolerance_pct", 0.15) if eval_cfg else 0.15
    )
    quartiles = _load_met_quartiles_or_default()
    met_lookup = _build_met_lookup_with_bridge()
    resolver = IntensityResolver(quartiles, met_lookup=met_lookup)
    matcher = SelectorMatcher(resolver=resolver)
    ruleset = RuleSet(list(allen_rules or []))
    semantic_oracle = _build_semantic_oracle(
        Path(cfg.output.dir) if cfg is not None else Path("."),
        evaluator_model=getattr(cfg, "evaluator_model", None) if cfg else None,
    )
    cfg_ttl_path = Path("src/assets/ontologies/HealthTasks_2026.05.19.ttl")
    cfg_ctx_iris_path = Path("src/assets/ontologies/context_iris.json")
    if cfg_ttl_path.exists() and cfg_ctx_iris_path.exists():
        from src.scripts.scenarios.metrics.context_fit import (
            load_context_categories_by_iri,
            load_context_links_by_uri,
        )

        context_links_by_uri = load_context_links_by_uri(
            cfg_ttl_path, load_context_categories_by_iri(cfg_ctx_iris_path)
        )
    else:
        context_links_by_uri = {}
    loss_weights = getattr(cfg, "loss", None) if cfg is not None else LossWeights()
    if loss_weights is None:
        loss_weights = LossWeights()
    env_seed = (
        getattr(getattr(run, "environment", None), "seed", None)
        if run is not None
        else None
    )
    return RLAugmenter(
        time_windows=time_windows,
        daily_window=daily_window,
        allen_rules=allen_rules,
        semantic=semantic_oracle,
        ruleset=ruleset,
        matcher=matcher,
        resolver=resolver,
        loss_weights=loss_weights,
        merge_threshold=merge_threshold,
        buffer_minutes=buffer_minutes,
        half_life_days=half_life_days,
        max_met=quartiles.max_met,
        divide_tolerance_pct=divide_tolerance_pct,
        context_links_by_uri=context_links_by_uri,
        seed=int(env_seed) if env_seed is not None else 0,
    )


def _wire_rl_constraints(augmenter, *, run, run_dir, cfg) -> None:
    """Wire per-person preference constraints into the RL augmenter.

    RL scores each week slice internally as its training reward, so it
    needs the same L_pref machinery the evaluate stack builds: a
    `PreferenceMapper`, the catalog `EventConfig`, and a per-person
    `WindowMap`. Builds them once and installs `(persona_constraints_for,
    window_map_for)` factories on the augmenter. A wiring failure leaves
    the L_pref leg out of the RL reward (logged), never crashing augment.
    """
    from src.scripts.persona.domain.time_windows import WindowMap
    from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
    from src.scripts.scenarios.metrics.preference_mapping import PreferenceMapper

    try:
        from src.scripts.persona.config.loader import load_event as _load_event

        if not cfg.calendar.persona_events:
            return
        event_config = _load_event(Path(cfg.calendar.persona_events))
        window_map_v2 = WindowMap.from_config(run.time_windows)
        pref_driver = None
        try:
            from src.graphrag.config import Neo4jSettings
            from src.graphrag.neo4j_client import make_driver

            pref_driver = make_driver(Neo4jSettings.from_env())
            pref_driver.verify_connectivity()
        except Exception as exc:
            log.info(
                "RL L_pref: Neo4j ancestor walk unavailable (%s); "
                "tier 2 disabled, semantic tier only.",
                exc,
            )
            pref_driver = None
        semantic_oracle = _build_semantic_oracle(
            Path(cfg.output.dir),
            evaluator_model=getattr(cfg, "evaluator_model", None),
        )
        pref_mapper = PreferenceMapper(
            cache=PreferenceCache.from_env(),
            neo4j_driver=pref_driver,
            semantic=semantic_oracle,
        )
        person_json_lookup = _load_person_json_lookup(run_dir)
        horizon_start_date = getattr(run, "horizon_start_date", None)

        def _constraints_for(trace):
            return _build_persona_constraints_for_trace(
                trace=trace,
                run_dir=run_dir,
                person_json_lookup=person_json_lookup,
                event_config=event_config,
                window_map=window_map_v2,
                horizon_days=run.horizon_days,
                horizon_start_date=horizon_start_date,
                mapper=pref_mapper,
                experiment=getattr(cfg, "experiment", "default"),
                scenario=cfg.id,
            )

        def _window_map_for(_trace):
            return window_map_v2

        augmenter.set_constraint_factories(_constraints_for, _window_map_for)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("RL L_pref wiring failed (reward omits pref leg): %s", exc)


def _resolve_augment_pool(args, method: str) -> tuple[int, str]:
    """Resolve the augment worker count + executor, clamping RL to serial.

    `rl` gives threads no speedup and risks a CUDA-context OOM, and the
    process executor is not supported for augment, so RL is forced to one
    worker. Returns `(workers, executor_kind)`.
    """
    workers = int(getattr(args, "workers", 1) or 1)
    executor_kind = getattr(args, "executor", "thread") or "thread"
    if method == "rl" and executor_kind == "thread" and workers > 1:
        log.warning(
            "rl augment under the thread executor gives no speedup and can "
            "OOM the GPU; clamping --workers %d to 1.",
            workers,
        )
        workers = 1
    return max(1, workers), executor_kind


def _build_worker_augmenter(
    method: str,
    run,
    *,
    run_dir,
    cfg,
    prompt_ablate,
    prompt_placebos,
):
    """Build a fresh augmenter + LLM-call recorder for one pool worker.

    Each worker gets its own augmenter so the per-person LLM-call recorder
    and any per-call augmenter state stay thread-isolated. Construction is
    cheap on the augment path (no embedder / scorer loads after the
    scoring move to evaluate). Returns `(augmenter, recorder)`.
    """
    recorder = None
    if method == "llm_agent":
        from src.scripts.scenarios.augmentation.llm_agent import LLMCallRecorder

        recorder = LLMCallRecorder()
    aug = _build_augmenter(
        method,
        run,
        llm_recorder=recorder,
        prompt_ablate=frozenset(prompt_ablate),
        prompt_placebos=dict(prompt_placebos),
        cfg=cfg,
    )
    if method == "rl":  # pragma: no cover - rl is clamped to the serial path
        _wire_rl_constraints(aug, run=run, run_dir=run_dir, cfg=cfg)
    return aug, recorder


def _run_augment_pool(
    items,
    fn,
    *,
    workers: int,
    executor_kind: str,
    build_worker,
    total: int,
    show: bool,
) -> int:
    """Run `fn(idx, trace, augmenter, recorder)` over `items` in a pool.

    Threads share the parent's read-only state and are the right model for
    the blocking-LLM `llm_agent` path; each task builds its own augmenter
    via `build_worker` so the LLM-call recorder is thread-isolated. The
    `process` executor is supported only for `rl` (handled before this call);
    for the other methods it falls back to threads with a warning. Returns the
    number of personas written.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.scripts.persona.concurrency.progress import wrap_progress

    if executor_kind == "process":
        log.warning(
            "augment --executor process is supported only for rl; running %d "
            "thread workers instead for this method.",
            workers,
        )

    def _task(item) -> bool:
        idx, trace = item
        person_augmenter, person_recorder = build_worker()
        return bool(fn(idx, trace, person_augmenter, person_recorder))

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, item) for item in items]
        for fut in wrap_progress(
            as_completed(futures), total=total, desc="Augmenting", show=show
        ):
            if fut.result():
                written += 1
    return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* and dispatch to the appropriate sub-command handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging bootstrap; honour --log-level (with --quiet override) before
    # the first output line.
    level = (
        "WARNING"
        if getattr(args, "quiet", False)
        else getattr(args, "log_level", "INFO")
    )
    _setup_logging(level)

    try:
        return args.func(args)
    except Exception as exc:  # pragma: no branch
        log.exception("Unhandled error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE


# ---------------------------------------------------------------------------
# Multi-scenario adapters
# ---------------------------------------------------------------------------


_PERSONA_YAML_NAMES: dict[str, str] = {
    "persona_environment": "environment.yaml",
    "persona_config": "persona_config.yaml",
    "persona_events": "event_config.yaml",
    "persona_rules": "temporal_relation_rules.yaml",
}


def _detect_persona_paths(scenario_file: Path | None) -> dict[str, str | None]:
    """Return CalendarSourceConfig keyword arguments for persona YAML paths.

    When *scenario_file* is provided the function looks for the four standard
    persona YAML files in the same directory (they are co-located by convention).
    Missing files are silently set to `None` so the caller can fall back to
    `used_configs.json` for those paths.
    """
    if scenario_file is None:
        return {k: None for k in _PERSONA_YAML_NAMES}
    exp_dir = Path(scenario_file).resolve().parent
    return {
        field: str(exp_dir / name) if (exp_dir / name).exists() else None
        for field, name in _PERSONA_YAML_NAMES.items()
    }


def _method_cfg_to_scenario_config(
    exp: ExperimentScenariosConfig,
    scenario: ScenarioDefinition,
    method_cfg: AugmentationMethodConfig,
    scenario_file: Path | None = None,
) -> ScenarioConfig:
    """Construct a legacy `ScenarioConfig` from a multi-scenario method entry.

    This lets all existing single-scenario CLI helpers work unchanged when
    called from the multi-scenario dispatch loop.

    *scenario_file* is the path to the `scenarios.yaml` on disk; when
    provided the four standard persona YAML files are auto-detected as siblings
    so `load_persona_run` can read configs directly rather than falling back
    to the `used_configs.json` path-lookup (which stores model data, not paths).
    """
    method_out = (
        method_cfg.output.dir
        if method_cfg.output is not None
        else exp.scenario_method_dir(scenario.id, method_cfg.method)
    )
    write_ics = method_cfg.output.write_ics if method_cfg.output else True
    write_json = method_cfg.output.write_json if method_cfg.output else True
    write_report = method_cfg.output.write_report if method_cfg.output else True

    persona_paths = _detect_persona_paths(scenario_file)
    cal = CalendarSourceConfig(
        run_dir=exp.effective_run_dir(),
        **persona_paths,
    )
    aug = AugmentationConfig(
        method=method_cfg.method,
        allow_merge=method_cfg.allow_merge,
        merge_threshold=method_cfg.merge_threshold,
        repeat_per_week=method_cfg.repeat_per_week,
        greedy=method_cfg.greedy,
        llm_agent=method_cfg.llm_agent,
        rl=method_cfg.rl,
        ptime=method_cfg.ptime,
        observation=method_cfg.observation,
    )
    return ScenarioConfig(
        id=f"{scenario.id}/{method_cfg.method}",
        description=scenario.description,
        calendar=cal,
        task_generation=scenario.task_generation,
        # Per-stage LLM overrides flow through the synthesized
        # single-scenario config so existing CLI helpers see them
        # without any additional plumbing.
        task_generator_model=scenario.task_generator_model,
        augmentation=aug,
        loss=method_cfg.loss,
        evaluator_model=scenario.evaluator_model,
        output=ScenarioOutputConfig(
            dir=method_out,
            write_ics=write_ics,
            write_json=write_json,
            write_report=write_report,
        ),
        timeframe=scenario.timeframe,
    )


def _normalize_scenario_id_filter(
    raw: str | list[str] | None,
) -> list[str] | None:
    """Coerce the `--scenario-id` arg into a `list[str] | None`.

    Argparse with `nargs='+'` always produces a list; legacy callers
    that pass a single string still work. Each token may itself be a
    comma-separated list (`--scenario-id a,b c` selects all three), so
    both the space form and the `persona.cli`-style comma form work.
    `None` (no flag) means "all scenarios". An empty list is treated
    like `None` so a misconfigured upstream caller never silently runs
    zero scenarios.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    out = [tok.strip() for item in raw for tok in str(item).split(",") if tok.strip()]
    return out or None


def _validate_multi_filters(
    exp: ExperimentScenariosConfig,
    scenario_id: str | list[str] | None,
    method_filter: str | None = None,
) -> int:
    """Verify `--scenario-id` / `--method` match real entries in *exp*.

    Without this guard the multi-scenario runners silently iterate an
    empty filtered set and return `EXIT_OK` having done zero work -
    indistinguishable in CI logs from a legitimate run.  Mistyping
    `--scenario-id senior_leisure_l23` when the YAML defines
    `senior_leisure_l123` would otherwise produce no error and no
    output (encountered 2026-05-14).

    `scenario_id` accepts either a single id (legacy) or a list (new
    `--scenario-id A B C` form). Every supplied id must match one of
    the YAML's scenarios; a single typo aborts the whole batch so
    nothing runs partially.
    """
    selected = _normalize_scenario_id_filter(scenario_id)
    if selected is not None:
        valid_ids = [s.id for s in exp.scenarios]
        unknown = [sid for sid in selected if sid not in valid_ids]
        if unknown:
            log.error(
                "--scenario-id %s not found. Valid scenario IDs: %s",
                ", ".join(repr(u) for u in unknown),
                ", ".join(sorted(valid_ids)) or "<none>",
            )
            return EXIT_USAGE
    if method_filter is not None:
        valid_methods: set[str] = set()
        selected_set = set(selected) if selected else None
        for scenario in exp.scenarios:
            if selected_set is not None and scenario.id not in selected_set:
                continue
            for m in scenario.augmentation:
                valid_methods.add(m.method)
        if method_filter not in valid_methods:
            log.error(
                "--method %r not found among augmentation methods for "
                "selected scenario(s). Valid methods: %s",
                method_filter,
                ", ".join(sorted(valid_methods)) or "<none>",
            )
            return EXIT_USAGE
    return EXIT_OK


def _multi_scenario_tasks_dir(exp: ExperimentScenariosConfig, scenario_id: str) -> Path:
    """Per-person task JSON directory for *scenario_id*.

    Single source of truth shared by `_multi_generate_tasks` (which
    writes per-person task JSONs here) and `_multi_augment` (which
    reads from here). `_cmd_generate_tasks_with_cfg` appends the
    `tasks/` subdir below its `out_dir` argument, so the generator's
    `out_dir` is one level above this helper's return value. Any
    drift between the two paths silently no-ops the augmenter; every
    person is "skipped" because its task JSON is searched for at the
    wrong location.
    """
    return Path(exp.task_generation_dir(scenario_id)) / "tasks"


# ---------------------------------------------------------------------------
# Prompt-component ablation fan-out
# ---------------------------------------------------------------------------


def _ablation_variants_for_method(method_cfg):
    """Return the variant list for one `(scenario, method)` pair.

    `[]` means "single baseline run, no ablation".
    """
    from src.scripts.scenarios.augmentation.prompts.ablation_designs import build_design

    if method_cfg.method != "llm_agent":
        return []
    ablation = getattr(method_cfg.llm_agent, "prompt_ablation", None)
    if ablation is None or ablation.design == "single":
        return []
    custom: list[tuple[str, list[str]]] | None = None
    if ablation.design == "custom":
        custom = [(v.id, list(v.ablate)) for v in ablation.variants]
    return build_design(ablation.design, fold=ablation.fold, custom=custom)


def _resolve_placebos(method_cfg, ablated: frozenset[str]) -> dict[str, str]:
    """Resolve placebo bodies for the ablated blocks of one variant.

    Per-block overrides win, then `DEFAULT_PLACEBOS` when
    `use_default_placebos` is `True`, then the empty string.
    """
    from src.scripts.scenarios.augmentation.prompts.augment_oneshot_placebos import (
        DEFAULT_PLACEBOS,
    )

    ablation = getattr(method_cfg.llm_agent, "prompt_ablation", None)
    if ablation is None:
        return {}
    out: dict[str, str] = {}
    for block in ablated:
        if block in ablation.placebos:
            out[block] = ablation.placebos[block]
        elif ablation.use_default_placebos:
            out[block] = DEFAULT_PLACEBOS.get(block, "")
    return out


def _ablation_variant_dir(method_dir: Path, variant_id: str) -> Path:
    """Per-variant subdirectory under one `(scenario, method)` run."""
    return Path(method_dir) / "ablation" / variant_id


def _write_ablation_index(method_dir: Path, variants) -> None:
    """Persist the design summary as `ablation/variants.json`."""
    payload = {
        "variants": [
            {"variant_id": v.variant_id, "ablated": sorted(v.ablated)} for v in variants
        ]
    }
    ablation_root = Path(method_dir) / "ablation"
    ablation_root.mkdir(parents=True, exist_ok=True)
    (ablation_root / "variants.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _multi_generate_tasks(
    exp: ExperimentScenariosConfig,
    args: argparse.Namespace,
    scenario_id: str | list[str] | None,
) -> int:
    """Run generate-tasks for every scenario (or a filtered subset)."""
    rc = _validate_multi_filters(exp, scenario_id)
    if rc != EXIT_OK:
        return rc
    selected = _normalize_scenario_id_filter(scenario_id)
    selected_set = set(selected) if selected else None
    scenario_file = Path(args.scenario) if getattr(args, "scenario", None) else None
    for scenario in exp.scenarios:
        if selected_set is not None and scenario.id not in selected_set:
            continue
        # Tasks live under the new `task_generation/<scenario_id>/` umbrella
        # so every task-gen artifact (per-person JSON, telemetry sidecars,
        # summary reports) groups together at the experiment level.
        tg_dir = exp.task_generation_dir(scenario.id)
        cfg = _method_cfg_to_scenario_config(
            exp, scenario, scenario.augmentation[0], scenario_file=scenario_file
        )
        sub_args = argparse.Namespace(
            scenario=args.scenario,
            out_dir=Path(tg_dir),
            seed=getattr(args, "seed", None),
            workers=getattr(args, "workers", 5),
            log_level=getattr(args, "log_level", "INFO"),
        )
        rc = _cmd_generate_tasks_with_cfg(cfg, sub_args)
        if rc != EXIT_OK:
            return rc
    return EXIT_OK


def _multi_augment(
    exp: ExperimentScenariosConfig,
    args: argparse.Namespace,
    scenario_id: str | list[str] | None,
    method_filter: str | None,
) -> int:
    """Run augment for every scenario x method (or filtered subset).

    When a method declares `prompt_ablation` other than `single`, the
    pair fans out into one augment pass per variant under
    `scenarios/<sid>/<method>/ablation/<variant_id>/`. The
    `--no-ablation` flag forces a single baseline run.
    """
    rc = _validate_multi_filters(exp, scenario_id, method_filter)
    if rc != EXIT_OK:
        return rc
    selected = _normalize_scenario_id_filter(scenario_id)
    selected_set = set(selected) if selected else None
    scenario_file = Path(args.scenario) if getattr(args, "scenario", None) else None
    no_ablation = bool(getattr(args, "no_ablation", False))
    variant_filter = _variant_filter_set(getattr(args, "ablation_variant", None))
    for scenario in exp.scenarios:
        if selected_set is not None and scenario.id not in selected_set:
            continue
        tasks_dir = _multi_scenario_tasks_dir(exp, scenario.id)
        for method_cfg in scenario.augmentation:
            if method_filter and method_cfg.method != method_filter:
                continue
            cfg = _method_cfg_to_scenario_config(
                exp, scenario, method_cfg, scenario_file=scenario_file
            )
            variants = [] if no_ablation else _ablation_variants_for_method(method_cfg)
            if not variants:
                sub_args = _augment_sub_args(args, cfg, tasks_dir, method_cfg.method)
                rc = _cmd_augment_with_cfg(cfg, sub_args)
                if rc != EXIT_OK:
                    return rc
                continue
            method_dir = Path(cfg.output.dir)
            _write_ablation_index(method_dir, variants)
            for variant in variants:
                if (
                    variant_filter is not None
                    and variant.variant_id not in variant_filter
                ):
                    continue
                variant_dir = _ablation_variant_dir(method_dir, variant.variant_id)
                sub_args = _augment_sub_args(
                    args,
                    cfg,
                    tasks_dir,
                    method_cfg.method,
                    out_dir=variant_dir,
                    prompt_ablate=variant.ablated,
                    prompt_placebos=_resolve_placebos(method_cfg, variant.ablated),
                )
                rc = _cmd_augment_with_cfg(cfg, sub_args)
                if rc != EXIT_OK:
                    return rc
    return EXIT_OK


def _variant_filter_set(raw) -> set[str] | None:
    """Coerce a `--ablation-variant` CLI argument into a set or `None`."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return {raw}
    items = list(raw)
    return set(items) if items else None


def _augment_sub_args(
    args: argparse.Namespace,
    cfg: ScenarioConfig,
    tasks_dir: Path,
    method: str,
    *,
    out_dir: Path | None = None,
    prompt_ablate: frozenset[str] | None = None,
    prompt_placebos: dict[str, str] | None = None,
) -> argparse.Namespace:
    """Build the sub-Namespace passed to `_cmd_augment_with_cfg`."""
    return argparse.Namespace(
        scenario=args.scenario,
        out_dir=out_dir,
        tasks_dir=tasks_dir,
        method=method,
        log_level=getattr(args, "log_level", "INFO"),
        charts=getattr(args, "charts", None),
        calendar_dpi=getattr(args, "calendar_dpi", 300),
        calendar_min_event_minutes=getattr(args, "calendar_min_event_minutes", 45),
        prompt_ablate=prompt_ablate,
        prompt_placebos=prompt_placebos,
        # `run` carries augment workers on `--augment-workers` (its
        # `--workers` drives generate-tasks); `augment` carries them on
        # `--workers`. Prefer the run-specific flag when present.
        workers=(
            getattr(args, "augment_workers", None)
            if getattr(args, "augment_workers", None) is not None
            else getattr(args, "workers", 1)
        ),
        executor=getattr(args, "executor", "thread"),
    )


def _multi_evaluate(
    exp: ExperimentScenariosConfig,
    args: argparse.Namespace,
    scenario_id: str | list[str] | None,
    method_filter: str | None,
) -> int:
    """Run evaluate for every scenario x method (or filtered subset).

    Symmetric to `_multi_augment`: when the method declares a
    non-`single` ablation, the evaluator walks the per-variant
    directories and scores each one.
    """
    rc = _validate_multi_filters(exp, scenario_id, method_filter)
    if rc != EXIT_OK:
        return rc
    selected = _normalize_scenario_id_filter(scenario_id)
    selected_set = set(selected) if selected else None
    scenario_file = Path(args.scenario) if getattr(args, "scenario", None) else None
    evaluated_pairs: list[tuple[str, str]] = []
    ablation_runs: list[tuple[str, str, str]] = []
    no_ablation = bool(getattr(args, "no_ablation", False))
    variant_filter = _variant_filter_set(getattr(args, "ablation_variant", None))
    for scenario in exp.scenarios:
        if selected_set is not None and scenario.id not in selected_set:
            continue
        tasks_dir = _multi_scenario_tasks_dir(exp, scenario.id)
        for method_cfg in scenario.augmentation:
            if method_filter and method_cfg.method != method_filter:
                continue
            cfg = _method_cfg_to_scenario_config(
                exp, scenario, method_cfg, scenario_file=scenario_file
            )
            variants = [] if no_ablation else _ablation_variants_for_method(method_cfg)
            if not variants:
                sub_args = _evaluate_sub_args(args, tasks_dir)
                rc = _cmd_evaluate_with_cfg(cfg, sub_args)
                if rc != EXIT_OK:
                    return rc
                evaluated_pairs.append((scenario.id, method_cfg.method))
                continue
            method_dir = Path(cfg.output.dir)
            for variant in variants:
                if (
                    variant_filter is not None
                    and variant.variant_id not in variant_filter
                ):
                    continue
                variant_dir = _ablation_variant_dir(method_dir, variant.variant_id)
                sub_args = _evaluate_sub_args(args, tasks_dir, run_dir=variant_dir)
                rc = _cmd_evaluate_with_cfg(cfg, sub_args)
                if rc != EXIT_OK:
                    return rc
                ablation_runs.append(
                    (scenario.id, method_cfg.method, variant.variant_id)
                )
            evaluated_pairs.append((scenario.id, method_cfg.method))

    _write_benchmark_report_if_possible(
        exp,
        evaluated_pairs,
        ablation_runs,
        learning_charts=bool(getattr(args, "learning_charts", False)),
    )
    return EXIT_OK


def _evaluate_sub_args(
    args: argparse.Namespace,
    tasks_dir: Path,
    *,
    run_dir: Path | None = None,
) -> argparse.Namespace:
    """Build the sub-Namespace passed to `_cmd_evaluate_with_cfg`."""
    return argparse.Namespace(
        scenario=args.scenario,
        run_dir=run_dir,
        tasks_dir=tasks_dir,
        log_level=getattr(args, "log_level", "INFO"),
        learning_charts=getattr(args, "learning_charts", False),
    )


def _render_learning_charts_for_run(
    eval_dir: Path, persons_dir: Path, *, scenario_id: str, method: str
) -> None:
    """Render one run's learning charts; failures are logged, never fatal."""
    from src.scripts.scenarios.export.learning_charts import render_learning_charts

    try:
        render_learning_charts(
            persons_dir, eval_dir / "charts", scenario_id=scenario_id, method=method
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("learning charts skipped for %s/%s: %s", scenario_id, method, exc)


def _render_cross_augmenter_charts(
    experiment_dir: Path,
    pairs: list[tuple[str, str]],
    labels: dict[str, str] | None = None,
) -> None:
    """Render one global cross-augmenter weekly overlay over every run; never fatal."""
    from src.scripts.scenarios.export.learning_charts import (
        render_cross_augmenter_overlay,
    )

    runs = [
        (
            sid,
            method,
            experiment_dir / "scenarios" / sid / method / "augmented" / "persons",
        )
        for sid, method in pairs
    ]
    try:
        render_cross_augmenter_overlay(runs, experiment_dir / "charts", labels=labels)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("cross-augmenter chart skipped: %s", exc)


def _write_benchmark_report_if_possible(
    exp: ExperimentScenariosConfig,
    evaluated_pairs: list[tuple[str, str]],
    ablation_runs: list[tuple[str, str, str]] | None = None,
    *,
    learning_charts: bool = False,
) -> None:
    """Auto-generate the aggregated Markdown benchmark report.

    Failures are non-fatal. Both `evaluated_pairs` and `ablation_runs`
    are merged with every matching artifact on disk so a partial-run
    `evaluate` still produces a complete report.
    """
    experiment_dir = Path(exp.effective_run_dir())
    augmented_pairs = _augment_pairs_with_disk_runs(
        experiment_dir, exp, evaluated_pairs
    )
    augmented_ablation_runs = _augment_ablation_runs_with_disk(
        experiment_dir, exp, ablation_runs or []
    )
    if not augmented_pairs and not augmented_ablation_runs:
        return
    try:
        from src.scripts.persona.cli import _load_experiment_name_from_run_dir
        from src.scripts.scenarios.export.benchmark_report import write_benchmark_report

        experiment_name = _load_experiment_name_from_run_dir(experiment_dir)
        target = write_benchmark_report(
            experiment_dir,
            exp.experiment_id,
            scenarios_cfg=exp,
            experiment_name=experiment_name,
            scenario_method_pairs=augmented_pairs,
            ablation_runs=augmented_ablation_runs,
        )
        log.info("[evaluate] benchmark report written to %s", target)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("benchmark report skipped: %s", exc)
    if learning_charts and augmented_pairs:
        labels = {s.id: s.label for s in exp.scenarios if getattr(s, "label", "")}
        _render_cross_augmenter_charts(experiment_dir, augmented_pairs, labels)


def _augment_pairs_with_disk_runs(
    experiment_dir: Path,
    exp: ExperimentScenariosConfig,
    seed_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Merge `seed_pairs` with pairs that have a top-level evaluation
    sidecar on disk.

    Pairs whose only data lives under `ablation/<variant_id>/` are
    excluded; those belong to the ablation section.
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sid, method in seed_pairs:
        eval_path = (
            experiment_dir
            / "scenarios"
            / sid
            / method
            / "evaluation"
            / "total_scheduling_gain.json"
        )
        if eval_path.is_file() and (sid, method) not in seen:
            found.append((sid, method))
            seen.add((sid, method))
    for scenario in exp.scenarios:
        for method_cfg in scenario.augmentation:
            key = (scenario.id, method_cfg.method)
            if key in seen:
                continue
            eval_path = (
                experiment_dir
                / "scenarios"
                / scenario.id
                / method_cfg.method
                / "evaluation"
                / "total_scheduling_gain.json"
            )
            if eval_path.is_file():
                found.append(key)
                seen.add(key)
    return found


def _augment_ablation_runs_with_disk(
    experiment_dir: Path,
    exp: ExperimentScenariosConfig,
    seed_runs: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Merge `seed_runs` with every variant directory found on disk."""
    out: list[tuple[str, str, str]] = list(seed_runs)
    seen: set[tuple[str, str, str]] = set(out)
    for scenario in exp.scenarios:
        for method_cfg in scenario.augmentation:
            ablation_root = (
                experiment_dir
                / "scenarios"
                / scenario.id
                / method_cfg.method
                / "ablation"
            )
            if not ablation_root.is_dir():
                continue
            for entry in sorted(ablation_root.iterdir()):
                if not entry.is_dir():
                    continue
                key = (scenario.id, method_cfg.method, entry.name)
                if key in seen:
                    continue
                if (entry / "evaluation" / "total_scheduling_gain.json").is_file():
                    out.append(key)
                    seen.add(key)
    return out


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_ablation_flags(parser: argparse.ArgumentParser) -> None:
    """Attach `--ablation-variant` and `--no-ablation` to a sub-parser."""
    parser.add_argument(
        "--ablation-variant",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Restrict prompt-component ablation to one or more variant "
            "bitstring IDs (e.g. `00000000000 10000000000`). Ignored when "
            "the scenario YAML does not declare `prompt_ablation`."
        ),
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help=(
            "Force a single baseline augment even when the scenario YAML "
            "declares `prompt_ablation`. Useful for quick iteration on a "
            "non-ablation run without editing the YAML."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scenarios",
        description="Scenario-based calendar augmentation pipeline.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help=(
            "Shortcut for --log-level WARNING. Hides per-step INFO chatter "
            "(generator/augmenter progress lines, third-party HTTP traces) "
            "and keeps only warnings / errors."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    _MULTI_FILTER_HELP = (
        "Multi-scenario YAML only: restrict to one or more scenarios. Pass "
        "space-separated IDs to run a subset (e.g. `--scenario-id A B C`). "
        "Ignored for legacy single-scenario YAMLs."
    )

    # ── generate-tasks ────────────────────────────────────────────────────
    gen = sub.add_parser(
        "generate-tasks",
        help="Generate recommended tasks for each person in a persona run.",
    )
    gen.add_argument("--scenario", required=True, type=Path, metavar="YAML")
    gen.add_argument("--out-dir", type=Path, default=None, metavar="DIR")
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument(
        "--scenario-id",
        nargs="+",
        default=None,
        metavar="ID",
        help=_MULTI_FILTER_HELP,
    )
    gen.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Parallel GraphRAG workers for task generation (default: 5). "
            "Each worker makes independent OpenAI+Neo4j calls; "
            "5 workers reduces 30-person runtime from ~8 min to ~2 min."
        ),
    )
    gen.set_defaults(func=_cmd_generate_tasks)

    # ── augment ───────────────────────────────────────────────────────────
    aug = sub.add_parser(
        "augment",
        help="Augment each person's calendar with the generated tasks.",
    )
    aug.add_argument("--scenario", required=True, type=Path, metavar="YAML")
    aug.add_argument("--tasks-dir", type=Path, default=None, metavar="DIR")
    aug.add_argument("--out-dir", type=Path, default=None, metavar="DIR")
    aug.add_argument(
        "--scenario-id",
        nargs="+",
        default=None,
        metavar="ID",
        help=_MULTI_FILTER_HELP,
    )
    aug.add_argument(
        "--method",
        choices=["greedy", "llm_agent", "rl", "ptime"],
        default=None,
        help="Override / filter the augmentation method.",
    )
    aug.add_argument(
        "--charts",
        nargs="*",
        default=None,
        metavar="KIND",
        help=(
            "Render weekly augmented calendars (and optionally other persona "
            "chart kinds) under <out-dir>/charts/<person_id>/. Bare flag = "
            "calendar only; pass kinds (lines, heatmap, gantt, calendar) to "
            "select a subset."
        ),
    )
    aug.add_argument(
        "--calendar-dpi",
        type=int,
        default=300,
        metavar="DPI",
        help="DPI metadata for the weekly calendar PNG exports (default: 300).",
    )
    aug.add_argument(
        "--calendar-min-event-minutes",
        type=int,
        default=45,
        metavar="N",
        help=(
            "Visually pad short events up to this many minutes so titles "
            "stay legible. The actual clock window is preserved in the "
            "event's notes. Default: 45."
        ),
    )
    aug.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel workers for the per-person augment loop (default: 1, "
            "serial). For the I/O-bound `llm_agent` path use 6-8 to cut a "
            "day-long run to hours. `rl` is forced to 1 under the thread "
            "executor (threads give it no speedup and risk a CUDA-context "
            "OOM); use `--executor process` with `rl.device=cpu` to fan it out."
        ),
    )
    aug.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help=(
            "Worker pool kind for `--workers > 1` (default: thread). Threads "
            "share the parent's state and are right for the blocking-LLM "
            "`llm_agent` path; processes are needed only for CPU-bound RL."
        ),
    )
    _add_ablation_flags(aug)
    aug.set_defaults(func=_cmd_augment)

    # ── evaluate ──────────────────────────────────────────────────────────
    ev = sub.add_parser(
        "evaluate",
        help="Evaluate augmentation quality (scheduling loss) for an existing run.",
    )
    ev.add_argument("--scenario", required=True, type=Path, metavar="YAML")
    ev.add_argument("--run-dir", type=Path, default=None, metavar="DIR")
    ev.add_argument(
        "--scenario-id",
        nargs="+",
        default=None,
        metavar="ID",
        help=_MULTI_FILTER_HELP,
    )
    ev.add_argument(
        "--method",
        choices=["greedy", "llm_agent", "rl", "ptime"],
        default=None,
        help="Restrict evaluation to one augmentation method (multi-scenario only).",
    )
    ev.add_argument(
        "--learning-charts",
        action="store_true",
        help="Render per-person learning charts as PNGs (needs matplotlib).",
    )
    _add_ablation_flags(ev)
    ev.set_defaults(func=_cmd_evaluate)

    # ── run ───────────────────────────────────────────────────────────────
    run = sub.add_parser(
        "run",
        help="End-to-end: generate-tasks to augment to evaluate in one shot.",
    )
    run.add_argument("--scenario", required=True, type=Path, metavar="YAML")
    run.add_argument("--out-dir", type=Path, default=None, metavar="DIR")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument(
        "--scenario-id",
        nargs="+",
        default=None,
        metavar="ID",
        help=_MULTI_FILTER_HELP,
    )
    run.add_argument(
        "--method",
        choices=["greedy", "llm_agent", "rl", "ptime"],
        default=None,
        help="Run only this augmentation method (multi-scenario only).",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="N",
        help="Parallel workers for generate-tasks step (default: 5).",
    )
    run.add_argument(
        "--charts",
        nargs="*",
        default=None,
        metavar="KIND",
        help=(
            "Render weekly augmented calendars after augment. Bare flag = "
            "calendar only; pass kinds (lines, heatmap, gantt, calendar) to "
            "select a subset."
        ),
    )
    run.add_argument(
        "--calendar-dpi",
        type=int,
        default=300,
        metavar="DPI",
        help="DPI metadata for the weekly calendar PNG exports (default: 300).",
    )
    run.add_argument(
        "--calendar-min-event-minutes",
        type=int,
        default=45,
        metavar="N",
        help=(
            "Visually pad short events up to this many minutes so titles "
            "stay legible. Default: 45."
        ),
    )
    _add_ablation_flags(run)
    run.set_defaults(func=_cmd_run)

    # ── report ────────────────────────────────────────────────────────────
    rep = sub.add_parser(
        "report",
        help=(
            "Aggregate the per-scenario evaluation sidecars into a single "
            "Markdown benchmark report under the experiment dir."
        ),
    )
    rep.add_argument("--scenario", required=True, type=Path, metavar="YAML")
    rep.add_argument(
        "--scenario-id",
        nargs="+",
        default=None,
        metavar="ID",
        help=_MULTI_FILTER_HELP,
    )
    rep.add_argument(
        "--method",
        choices=["greedy", "llm_agent", "rl", "ptime"],
        default=None,
        help="Restrict the report to one augmentation method.",
    )
    rep.add_argument(
        "--baseline",
        default=None,
        metavar="ID[/METHOD]",
        help=(
            "Pairwise baseline override, e.g. `--baseline oneshot_gpt_4o_mini_l234` "
            "or `--baseline oneshot_gpt_4o_mini_l234/llm_agent`. "
            "Defaults to the first scenario × method run discovered."
        ),
    )
    rep.add_argument(
        "--learning-charts",
        action="store_true",
        help="Render per-person learning charts as PNGs (needs matplotlib).",
    )
    rep.set_defaults(func=_cmd_report)

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_generate_tasks_with_cfg(cfg: ScenarioConfig, args: argparse.Namespace) -> int:
    """Core generate-tasks logic given an already-loaded `ScenarioConfig`."""
    log.info(
        "Scenario loaded: id=%s  method=%s",
        cfg.id,
        cfg.weekly_task_generation()[0].method,
    )

    out_dir: Path = getattr(args, "out_dir", None) or Path(cfg.output.dir)
    tasks_dir = out_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    run, run_dir = _try_load_persona_run(cfg)

    if isinstance(run, str):
        log.error("Persona run error: %s", run)
        return EXIT_USAGE

    if run is None:
        seed_msg = (
            f" (seed={args.seed})" if getattr(args, "seed", None) is not None else ""
        )
        log.warning("No persona run found at %s; no tasks written.", run_dir)
        print(
            f"[generate-tasks] scenario={cfg.id} to {tasks_dir}{seed_msg} (0 persons)"
        )
        return EXIT_OK

    run, _window = _scope_run_to_timeframe(run, cfg)

    weekly_raw = cfg.weekly_task_generation()
    if len(weekly_raw) == 1 and weekly_raw[0].week is None:
        weekly_cfgs = weekly_raw  # uniform; broadcast to every week
    else:
        try:
            weekly_cfgs = _resolve_weekly_task_configs(
                weekly_raw, max(1, int(run.horizon_days) // 7)
            )
        except ValueError as exc:
            log.error("task_generation error: %s", exc)
            return EXIT_USAGE

    log.debug("Building task generator(s) …")
    generators = [
        _build_task_generator(
            cfg.model_copy(update={"task_generation": wc}), tasks_dir=tasks_dir
        )
        for wc in weekly_cfgs
    ]

    # Build person_id to raw JSON map from the run index
    index_path = run_dir / "index.json"
    persons_map: dict[str, dict] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index.get("persons", []):
            persons_map[entry["person_id"]] = _read_person_json(
                run_dir, entry["json_path"]
            )

    n_total = len(run.traces)
    n_workers = max(1, getattr(args, "workers", 5))
    log.debug(
        "Generating tasks for %d persons with %d parallel worker(s) …",
        n_total,
        n_workers,
    )

    def _generate_one(
        idx_trace: tuple[int, object],
    ) -> tuple[str, list | None]:
        """Generate tasks for one person; returns (person_id, tasks|None)."""
        idx, trace = idx_trace
        person = _person_from_json(persons_map.get(trace.person_id, {}))
        if person is None:
            log.warning(
                "  [%d/%d] %s; persona snapshot missing, skipping.",
                idx,
                n_total,
                trace.person_id,
            )
            return trace.person_id, None

        log.debug(
            "  [%d/%d] Generating tasks for %s (profile: %s) …",
            idx,
            n_total,
            trace.person_id,
            profile_summary(person),
        )
        t0 = time.monotonic()
        accumulated: set[str] = set()
        weekly_tasks: list[list] = []
        for gen, week_cfg in zip(generators, weekly_cfgs):
            week_tasks = gen.generate(
                person=person,
                calendar=trace,
                scenario_description=cfg.description,
                exclude_uris=set(accumulated) if week_cfg.cross_week_distinct else None,
            )
            weekly_tasks.append(week_tasks)
            if week_cfg.cross_week_distinct:
                accumulated.update(t.ontology_uri for t in week_tasks if t.ontology_uri)
        log.debug(
            "  [%d/%d] %s to %d tasks over %d week(s)  (%.1fs)",
            idx,
            n_total,
            trace.person_id,
            sum(len(w) for w in weekly_tasks),
            len(weekly_tasks),
            time.monotonic() - t0,
        )
        return trace.person_id, weekly_tasks

    from src.scripts.persona.concurrency.progress import (
        default_show_progress,
        wrap_progress,
    )

    n = 0
    empty: list[str] = []
    t_start = time.monotonic()
    indexed = list(enumerate(run.traces, 1))
    show_bar = default_show_progress()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_generate_one, item): item for item in indexed}
        for future in wrap_progress(
            as_completed(futures),
            total=n_total,
            desc="Generating tasks",
            show=show_bar,
        ):
            person_id, weekly_tasks = future.result()
            if weekly_tasks is not None:
                _write_weekly_tasks(weekly_tasks, tasks_dir / f"{person_id}_tasks.json")
                n += 1
                if not any(weekly_tasks):
                    empty.append(person_id)

    seed_msg = f" (seed={args.seed})" if getattr(args, "seed", None) is not None else ""
    elapsed = time.monotonic() - t_start
    log.info(
        "generate-tasks complete: %d/%d persons in %.1fs  (%.1f s/person avg)",
        n,
        n_total,
        elapsed,
        elapsed / max(n, 1),
    )
    # Mark empty-task persons in a machine-readable summary so a transient
    # graphrag fetch failure (0 tasks for a person) is visible downstream
    # instead of silently surfacing as `n/a` in the report. The console
    # warning is easy to miss in a long run; this sidecar is not.
    (tasks_dir / "_generation_summary.json").write_text(
        json.dumps(
            {
                "scenario_id": cfg.id or "",
                "total_persons": n_total,
                "written_persons": n - len(empty),
                "empty_persons": empty,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if empty:
        log.warning(
            "%d/%d persons produced ZERO tasks (see %s): %s",
            len(empty),
            n_total,
            tasks_dir / "_generation_summary.json",
            ", ".join(empty[:10]) + (" ..." if len(empty) > 10 else ""),
        )
    empty_msg = f"  ({len(empty)} EMPTY, see _generation_summary.json)" if empty else ""
    print(
        f"[generate-tasks] scenario={cfg.id} to {n} persons, "
        f"{tasks_dir}{seed_msg}{empty_msg}"
    )
    return EXIT_OK


def _cmd_generate_tasks(args: argparse.Namespace) -> int:
    """Dispatch generate-tasks to single-scenario or multi-scenario handler."""
    try:
        raw = load_config(args.scenario)
    except ScenarioConfigError as exc:
        log.error("Config error: %s", exc)
        return EXIT_USAGE

    if isinstance(raw, ExperimentScenariosConfig):
        return _multi_generate_tasks(raw, args, getattr(args, "scenario_id", None))
    return _cmd_generate_tasks_with_cfg(raw, args)


def _augment_person_to_disk(
    trace,
    person_augmenter,
    *,
    idx: int,
    n_total: int,
    method: str,
    cfg: ScenarioConfig,
    tasks_dir: Path,
    persons_dir: Path,
    ics_dir: Path,
    aug_dir: Path,
    horizon_hint,
    recorder=None,
) -> bool:
    """Augment one persona and write its artifacts; True when written.

    Each persona writes its own per-person files (distinct paths), so this runs
    safely under a thread or process worker. `recorder` is the per-task LLM call
    recorder (or None for non-llm_agent).
    """
    from src.scripts.scenarios.export.ics_writer import write_augmented_ics
    from src.scripts.scenarios.export.json_writer import write_solution_json

    task_path = Path(tasks_dir) / f"{trace.person_id}_tasks.json"
    if not task_path.exists():
        log.debug(
            "  [%d/%d] %s; no task file, skipping.", idx, n_total, trace.person_id
        )
        return False

    weekly_tasks = _read_weekly_tasks(task_path)
    tasks = [t for week in weekly_tasks for t in week]
    log.debug(
        "  [%d/%d] Augmenting %s  (%d tasks over %d week(s)) …",
        idx,
        n_total,
        trace.person_id,
        len(tasks),
        len(weekly_tasks),
    )
    if recorder is not None:  # pragma: no cover
        recorder.reset()
    t0 = time.monotonic()
    solution = person_augmenter.augment(
        trace, tasks, cfg.augmentation, horizon=horizon_hint, weekly_tasks=weekly_tasks
    )
    elapsed_aug = time.monotonic() - t0

    placed = len(solution.scheduled)
    total_slots = len(solution.tasks)  # sum of the per-week task counts
    log.debug(
        "  [%d/%d] %s to %d/%d scheduled  (%.1fs)",
        idx,
        n_total,
        trace.person_id,
        placed,
        total_slots,
        elapsed_aug,
    )

    # Augment-time telemetry sidecar; wall-time always; LLM tokens +
    # cost only when the recorder captured anything (llm_agent).  The
    # model name is read off the per-method config so cost estimates
    # use the right pricing entry.
    from src.scripts.scenarios.metrics.telemetry import (
        AugmentTelemetry,
        write_augment_sidecar,
    )

    augment_provider, augment_model = _augment_llm_identity(method, cfg)
    aug_telem = AugmentTelemetry(
        person_id=trace.person_id,
        scenario_id=cfg.id or "",
        method=method,
        provider=augment_provider,
        model=augment_model,
        wall_time_seconds=elapsed_aug,
        n_tasks_attempted=total_slots,
        n_tasks_placed=placed,
        n_tasks_dropped=len(solution.unscheduled),
    )
    if recorder is not None:  # pragma: no cover
        for in_tok, out_tok, w, reasoning_tok in recorder.calls:
            aug_telem.record_llm_call(
                input_tokens=in_tok,
                output_tokens=out_tok,
                wall_time_seconds=w,
                reasoning_tokens=reasoning_tok,
            )
    write_augment_sidecar(aug_telem, aug_dir)

    if cfg.output.write_json:
        write_solution_json(solution, persons_dir / f"{trace.person_id}.json")
    if cfg.output.write_ics:
        write_augmented_ics(
            solution, ics_dir / f"{trace.person_id}.ics", calendar=trace
        )
    _write_augmented_timeline(
        trace=trace,
        scheduled=solution.scheduled,
        out_path=persons_dir / f"{trace.person_id}_timeline.tsv",
    )

    # RL is the only learning augmenter; its per-week training trajectory is a
    # train-time diagnostic (not a reporting leg) so it stays attached here.
    # Every scoring leg moves to evaluate.
    _write_rl_training_sidecar_for_solution(
        solution=solution,
        trace=trace,
        persons_dir=persons_dir,
    )
    log.debug(
        "  [%d/%d] %s placed %d/%d", idx, n_total, trace.person_id, placed, total_slots
    )
    return True


def _run_rl_augment_pool(
    *,
    run,
    run_dir,
    cfg: ScenarioConfig,
    requested_workers: int,
    tasks_dir: Path,
    persons_dir: Path,
    ics_dir: Path,
    aug_dir: Path,
    horizon_hint,
) -> int:
    """Fan the per-person RL augment across a spawn process pool; return written.

    Caps the worker count to fit the host (cores + RAM), retries failed persons,
    and forces each worker's embedder to CPU so they never stack Qwen on the GPU.
    """
    from src.scripts.scenarios.augmentation.rl.parallel import (
        RLWorkerPayload,
        resolve_rl_pool_workers,
        run_rl_process_pool,
    )

    rl_cfg = cfg.augmentation.rl
    workers, reason = resolve_rl_pool_workers(
        requested_workers, per_worker_gb=rl_cfg.parallel_per_worker_gb
    )
    log.info(
        "rl augment process pool: requested %d workers, running %d (bound by %s)",
        requested_workers,
        workers,
        reason,
    )
    n_total = len(run.traces)
    payloads = [
        RLWorkerPayload(
            person_index=idx,
            n_total=n_total,
            run=run,
            run_dir=str(run_dir),
            cfg=cfg,
            tasks_dir=str(tasks_dir),
            persons_dir=str(persons_dir),
            ics_dir=str(ics_dir),
            aug_dir=str(aug_dir),
            horizon_hint=horizon_hint,
            worker_threads=rl_cfg.parallel_worker_threads,
        )
        for idx in range(1, n_total + 1)
    ]
    report = run_rl_process_pool(
        payloads, workers=workers, max_retries=rl_cfg.parallel_max_retries
    )
    if report.failed:
        log.warning(
            "rl augment: %d/%d persons failed after retries: %s",
            len(report.failed),
            n_total,
            ", ".join(sorted(report.failed)),
        )
    retried = sum(1 for c in report.attempts.values() if c > 1)
    if retried:
        log.info("rl augment: %d persons needed a retry", retried)
    return report.written


def _cmd_augment_with_cfg(cfg: ScenarioConfig, args: argparse.Namespace) -> int:
    """Core augment logic given an already-loaded `ScenarioConfig`."""
    out_dir: Path = getattr(args, "out_dir", None) or Path(cfg.output.dir)
    tasks_dir: Path = getattr(args, "tasks_dir", None) or (out_dir / "tasks")
    aug_dir = out_dir / "augmented"
    persons_dir = aug_dir / "persons"
    ics_dir = aug_dir / "ics_per_person"
    persons_dir.mkdir(parents=True, exist_ok=True)
    ics_dir.mkdir(parents=True, exist_ok=True)

    method = getattr(args, "method", None) or cfg.augmentation.method
    log.info("Scenario loaded: id=%s  augmenter=%s", cfg.id, method)

    run, run_dir = _try_load_persona_run(cfg)
    if isinstance(run, str):
        log.error("Persona run error: %s", run)
        return EXIT_USAGE
    if run is None:
        log.error("No persona run found at %s", run_dir)
        return EXIT_USAGE

    run, window = _scope_run_to_timeframe(run, cfg)
    horizon_hint = (window.start_date, window.days) if window is not None else None
    if window is not None:
        _write_timeframe_resolved(aug_dir / "timeframe.json", cfg, window)

    n_total = len(run.traces)
    t_start = time.monotonic()

    # RL fans out across a spawn process pool when asked. Route here BEFORE
    # building any augmenter in the parent: the per-person worker rebuilds its
    # own, and a parent-side embedder load would touch CUDA and break spawn.
    requested_workers = int(getattr(args, "workers", 1) or 1)
    executor_kind = getattr(args, "executor", "thread") or "thread"
    if method == "rl" and executor_kind == "process" and requested_workers > 1:
        n = _run_rl_augment_pool(
            run=run,
            run_dir=run_dir,
            cfg=cfg,
            requested_workers=requested_workers,
            tasks_dir=tasks_dir,
            persons_dir=persons_dir,
            ics_dir=ics_dir,
            aug_dir=aug_dir,
            horizon_hint=horizon_hint,
        )
        elapsed = time.monotonic() - t_start
        log.info("augment complete: %d/%d persons in %.1fs", n, n_total, elapsed)
        print(f"[augment] scenario={cfg.id} method={method} to {n} persons, {aug_dir}")
        charts_value = getattr(args, "charts", None)
        if charts_value is not None:
            rc = _render_augment_charts(cfg, args, aug_dir, persons_dir, run)
            if rc != EXIT_OK:
                return rc
        return EXIT_OK

    log.debug("Building %s augmenter …", method)
    # An LLM-call recorder is only meaningful for `llm_agent`; for greedy
    # it stays `None` and the augment sidecar carries zeros under
    # `llm_calls` so the by_stage report still has uniform shape.
    llm_recorder = None
    if method == "llm_agent":
        from src.scripts.scenarios.augmentation.llm_agent import LLMCallRecorder

        llm_recorder = LLMCallRecorder()
    # Prompt-component ablation parameters arrive on the args
    # namespace when the multi-scenario dispatch loop is fanning a
    # single (scenario, method) pair out across variants; baseline
    # runs leave them at the empty defaults so `_build_augmenter`
    # constructs the unchanged augmenter.
    prompt_ablate = getattr(args, "prompt_ablate", None) or frozenset()
    prompt_placebos = getattr(args, "prompt_placebos", None) or {}
    augmenter = _build_augmenter(
        method,
        run,
        llm_recorder=llm_recorder,
        prompt_ablate=frozenset(prompt_ablate),
        prompt_placebos=dict(prompt_placebos),
        cfg=cfg,
    )
    if augmenter is None:
        log.error("Unknown augmentation method: %r", method)
        return EXIT_USAGE

    # Augment produces placements only; every reporting-path leg is
    # scored at evaluate time (the single source of truth). The RL
    # augmenter is the one exception: it scores week slices internally
    # as its training reward, so when the method is `rl` we wire the
    # same per-person preference constraints the evaluate stack uses
    # into the augmenter via `set_constraint_factories`.
    if method == "rl":
        _wire_rl_constraints(augmenter, run=run, run_dir=run_dir, cfg=cfg)

    from src.scripts.persona.concurrency.progress import (
        default_show_progress,
        wrap_progress,
    )

    show_bar = default_show_progress()

    def _augment_one(idx, trace, person_augmenter, person_recorder) -> bool:
        return _augment_person_to_disk(
            trace,
            person_augmenter,
            idx=idx,
            n_total=n_total,
            method=method,
            cfg=cfg,
            tasks_dir=tasks_dir,
            persons_dir=persons_dir,
            ics_dir=ics_dir,
            aug_dir=aug_dir,
            horizon_hint=horizon_hint,
            recorder=person_recorder,
        )

    workers, executor_kind = _resolve_augment_pool(args, method)
    items = list(enumerate(run.traces, 1))

    if workers == 1:
        iterator = wrap_progress(items, total=n_total, desc="Augmenting", show=show_bar)
        n = sum(
            1
            for idx, trace in iterator
            if _augment_one(idx, trace, augmenter, llm_recorder)
        )
    else:
        n = _run_augment_pool(
            items,
            _augment_one,
            workers=workers,
            executor_kind=executor_kind,
            build_worker=lambda: _build_worker_augmenter(
                method,
                run,
                run_dir=run_dir,
                cfg=cfg,
                prompt_ablate=prompt_ablate,
                prompt_placebos=prompt_placebos,
            ),
            total=n_total,
            show=show_bar,
        )

    elapsed = time.monotonic() - t_start
    log.info("augment complete: %d/%d persons in %.1fs", n, n_total, elapsed)
    print(f"[augment] scenario={cfg.id} method={method} to {n} persons, {aug_dir}")

    charts_value = getattr(args, "charts", None)
    if charts_value is not None:
        rc = _render_augment_charts(cfg, args, aug_dir, persons_dir, run)
        if rc != EXIT_OK:
            return rc
    return EXIT_OK


def _render_augment_charts(
    cfg: ScenarioConfig,
    args: argparse.Namespace,
    aug_dir: Path,
    persons_dir: Path,
    run,
) -> int:
    """Render weekly calendars (and optional other kinds) for an augment run.

    Reads the augmented per-person JSONs just written and overlays them
    on the base persona schedules from `run.traces`. Outputs land
    under `aug_dir / "charts" / <person_id> / week_NN.png`.
    """
    from src.scripts.persona.analytics.charts import (
        ALL_CHART_KINDS,
        render_all_charts,
        variable_heatmap_events,
    )
    from src.scripts.persona.analytics.weekly_calendar import (
        load_augmented_tasks_by_person,
    )
    from src.scripts.persona.cli import _load_event_persona_from_run_dir
    from src.scripts.persona.validation.loader import load_schedules

    persona_run_dir = Path(cfg.calendar.run_dir) if cfg.calendar.run_dir else None
    if persona_run_dir is None or not (persona_run_dir / "persons").is_dir():
        log.warning(
            "charts skipped: persona run directory %s/persons is missing.",
            persona_run_dir,
        )
        return EXIT_OK

    schedules = load_schedules(persona_run_dir / "persons")
    if not schedules:
        log.warning("charts skipped: no persona JSONs under %s", persona_run_dir)
        return EXIT_OK

    augmented = load_augmented_tasks_by_person(persons_dir)
    selected = set(args.charts) if args.charts else {"calendar"}
    if not selected.issubset(set(ALL_CHART_KINDS)):
        unknown = sorted(selected - set(ALL_CHART_KINDS))
        log.error("charts: unknown kinds %s", unknown)
        return EXIT_USAGE

    event_cfg, persona_cfg = _load_event_persona_from_run_dir(persona_run_dir)
    heatmap_events_by_persona = (
        variable_heatmap_events(event_cfg, persona_cfg)
        if event_cfg is not None and persona_cfg is not None
        else None
    )

    method_label = getattr(args, "method", None) or cfg.augmentation.method
    title_prefix = f"{cfg.id or method_label}"
    try:
        render_all_charts(
            schedules,
            out_dir=aug_dir,
            kinds=selected,
            heatmap_events_by_persona=heatmap_events_by_persona,
            augmented_by_person=augmented,
            weekly_calendar_dpi=getattr(args, "calendar_dpi", 300),
            weekly_calendar_title_prefix=title_prefix,
            weekly_calendar_min_event_minutes=getattr(
                args, "calendar_min_event_minutes", None
            ),
        )
    except ValueError as exc:
        log.error("charts: %s", exc)
        return EXIT_USAGE
    print(f"[augment] charts written under {aug_dir / 'charts'}")
    return EXIT_OK


def _cmd_augment(args: argparse.Namespace) -> int:
    """Dispatch augment to single-scenario or multi-scenario handler."""
    try:
        raw = load_config(args.scenario)
    except ScenarioConfigError as exc:
        log.error("Config error: %s", exc)
        return EXIT_USAGE

    if isinstance(raw, ExperimentScenariosConfig):
        return _multi_augment(
            raw,
            args,
            getattr(args, "scenario_id", None),
            getattr(args, "method", None),
        )
    return _cmd_augment_with_cfg(raw, args)


class _LazyCpuEmbedder:
    """Defer the local Qwen load until the first `embed_query`, pinned to CPU.

    A warm MET / bridge cache answers without ever calling `embed_query`, so a
    parallel RL worker that hits the cache loads no model at all. The first real
    miss builds one CPU embedder, cached for the rest of the process.
    """

    def __init__(self) -> None:
        self._embedder = None

    def embed_query(self, text: str):
        if self._embedder is None:
            from src.graphrag.embeddings import make_embedder

            self._embedder = make_embedder(device="cpu")
        return self._embedder.embed_query(text)


def _build_met_lookup_with_bridge(embedder=None):
    """Build a `MetLookup` seeded with the matched_activity MET bridge.

    Returns None when neither the bridge artefacts nor a Neo4j driver
    are reachable; callers pass None into `IntensityResolver` which then
    skips the MET-bucket leg cleanly. `embedder` overrides the default eager
    embedder (used by parallel RL workers to pass a lazy CPU embedder).
    """
    from src.scripts.scenarios.metrics.met_lookup import (
        MetLookup,
        load_matched_activity_index,
    )

    ontologies_dir = Path("src/assets/ontologies")
    jsonl_path = ontologies_dir / "matched_activity.jsonl"
    ha_ttl = ontologies_dir / "HumanActivities_2026.05.03.ttl"
    bridge = None
    if jsonl_path.exists() and ha_ttl.exists():
        try:
            bridge = load_matched_activity_index(jsonl_path, ha_ttl)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "MET bridge load failed (%s); falling back to embedding lookup.",
                exc,
            )

    driver = None
    try:
        from src.graphrag.config import Neo4jSettings
        from src.graphrag.embeddings import make_embedder
        from src.graphrag.neo4j_client import make_driver

        driver = make_driver(Neo4jSettings.from_env())
        driver.verify_connectivity()
        if embedder is None:
            embedder = make_embedder()
    except Exception as exc:
        log.info(
            "MET lookup: Neo4j embedding pipeline unavailable (%s); "
            "only the bridge path will resolve MET.",
            exc,
        )
        driver = None
        embedder = None

    if bridge is None and (driver is None or embedder is None):
        return None
    return MetLookup(driver=driver, embedder=embedder, bridge=bridge)


def _build_activity_bridge(cfg):
    """Build the L_merge cross-ontology pre-filter; returns None when unwired.

    Needs a reachable Neo4j driver plus the event-label to
    `human_activity_iri` map from the persona event catalog. The bridge
    degrades to None when either is missing; `compute_l_concurrent`
    treats None as "no pre-filter" and runs the full σ oracle path.
    """
    from src.scripts.scenarios.metrics.preference_mapping import ActivityFamilyBridge

    driver = None
    try:
        from src.graphrag.config import Neo4jSettings
        from src.graphrag.neo4j_client import make_driver

        driver = make_driver(Neo4jSettings.from_env())
        driver.verify_connectivity()
    except Exception as exc:
        log.info(
            "ActivityFamilyBridge: Neo4j unavailable (%s); L_merge pre-filter "
            "disabled, full σ oracle path will run.",
            exc,
        )
        return None

    ha_by_label: dict[str, str] = {}
    try:
        from src.scripts.persona.config.loader import load_event as _load_event

        if cfg.calendar.persona_events:
            event_config = _load_event(Path(cfg.calendar.persona_events))
            for category in event_config.categories.values():
                for ev_name, ev in category.events.items():
                    if ev.human_activity_iri:
                        ha_by_label[ev_name] = ev.human_activity_iri
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "ActivityFamilyBridge: could not load event catalog (%s); "
            "pre-filter has no event IRIs to bridge against.",
            exc,
        )

    if not ha_by_label:
        return None

    eval_cfg = getattr(cfg, "evaluation", None)
    max_hops = getattr(eval_cfg, "cross_ancestor_max_hops", 2) if eval_cfg else 2
    return ActivityFamilyBridge(
        driver=driver,
        event_ha_iri_by_label=ha_by_label,
        max_hops=int(max_hops),
    )


def _build_judge_oracle(cache_path: Path, evaluator_model=None):
    """Construct an `LLMJudgeOracle` from whatever LLM backend is available.

    Returns `None` (with a warning) when no LLM client can be built; the
    caller skips the oracle-based recompute and persisted `L_concurrent`
    values are reused.

    The oracle runs in evaluation only; never at augment time.

    `evaluator_model` (an :class:`LLMModelConfig` from the scenario YAML,
    or `None`) overrides the env-default LLM. This is what enables a
    benchmark sweep to score every persona with two different judge
    models and compare their σ outputs without touching `.env`.
    """
    from src.scripts.scenarios.metrics.llm_judge import LLMJudgeOracle
    from src.scripts.scenarios.metrics.prompts.semantic_merge_score import (
        SEMANTIC_MERGE_SCORE_VERSION,
    )

    try:
        from src.graphrag.llm import make_usage_llm

        llm = make_usage_llm(_llm_settings_with_override(evaluator_model))
    except Exception as exc:
        log.warning(
            "LLMJudgeOracle unavailable (%s); L_concurrent will be read from "
            "the augment-time persisted values without recomputation.",
            exc,
        )
        return None

    from src.scripts.scenarios.augmentation.llm_agent import LLMCallRecorder
    from src.scripts.scenarios.metrics.telemetry import _Timer, estimate_tokens

    recorder = LLMCallRecorder()

    def _invoke(prompt: str) -> str:
        with _Timer() as t:
            response = llm.invoke(prompt)
        content = getattr(response, "content", "") or ""
        usage = getattr(response, "usage_metadata", None)
        if not isinstance(usage, dict):
            usage = {}
        recorder.record(
            input_tokens=int(usage.get("input_tokens") or 0) or estimate_tokens(prompt),
            output_tokens=int(usage.get("output_tokens") or 0)
            or estimate_tokens(content),
            wall_time_seconds=t.elapsed,
            reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        )
        return content

    # Surface the resolved model name on the oracle so telemetry sidecars
    # and cache records reflect the model that actually ran. The oracle
    # already accepts a `model` kwarg used for the cache JSONL header.
    judge_model_name = (
        evaluator_model.model if evaluator_model is not None else "gpt-4o-mini"
    )
    oracle = LLMJudgeOracle(
        invoke_fn=_invoke,
        cache_path=cache_path,
        model=judge_model_name,
        cache_version=SEMANTIC_MERGE_SCORE_VERSION,
    )
    # Per-call token/wall sink; the evaluate loop resets it before each
    # person and drains it into that person's EvaluateTelemetry, the
    # same pattern the augment loop uses with DirectLLMPipeline.
    oracle.call_recorder = recorder
    return oracle


def _build_semantic_oracle(out_dir: Path, evaluator_model=None):
    """Build the augment-time σ oracle: Neo4j embedding cosine only.

    Chain: Neo4j `EmbeddedConcept` cosine ⟶ `none` stub when Neo4j is
    unreachable or the labels miss the embedding index. The stub keeps
    augment-time fast: `score_persona_stage_semantic` treats it as
    no-signal (returns `None`) and `compute_l_concurrent` returns
    `None` for that person; the evaluate stage then recomputes
    `L_concurrent` against the LLM-judge oracle which DOES call the
    LLM, but only once per scenario.

    `evaluator_model` is accepted for symmetry with the judge-oracle
    builder; it is currently unused here because the LLM scorer is
    not wired in (each augment-time σ pair would be an LLM call and
    blow up the run-time on a cold cache).
    """
    del evaluator_model  # reserved for a future opt-in LLM fallback
    from src.scripts.scenarios.metrics.semantic import (
        SemanticCompatibility,
        make_neo4j_embedding_fn,
    )

    cache_path = out_dir / "augmented" / "semantic_cache.jsonl"

    embedding_fn = None
    try:
        from src.graphrag.config import Neo4jSettings
        from src.graphrag.neo4j_client import make_driver

        driver = make_driver(Neo4jSettings.from_env())
        driver.verify_connectivity()
        embedding_fn = make_neo4j_embedding_fn(driver)
    except Exception as exc:
        log.info(
            "Neo4j embedding oracle unavailable (%s); augment-time σ "
            "will fall back to the `none` stub.",
            exc,
        )

    return SemanticCompatibility(
        embedding_fn=embedding_fn,
        cache_path=cache_path,
    )


def _recommended_task_from_dict(s: dict):
    """Rebuild a `RecommendedTask` from a persisted scheduled/unscheduled entry.

    `is_concurrent` and `is_dividable` MUST round-trip faithfully; without
    them every reconstructed task falls back to the `RecommendedTask`
    default (False) and gets excluded from L_merge's Φ.
    """
    from src.scripts.scenarios.domain.task import RecommendedTask

    return RecommendedTask(
        label=s["label"],
        display_name=s.get("display_name", ""),
        description=s.get("description", ""),
        duration_min=(
            int(s.get("duration_min", 0)) if s.get("duration_min") is not None else 0
        ),
        duration_max=(
            int(s.get("duration_max", 0)) if s.get("duration_max") is not None else 0
        ),
        intensity=int(s.get("intensity", 1)),
        is_concurrent=bool(s.get("is_concurrent", False)),
        is_dividable=bool(s.get("is_dividable", False)),
        ontology_uri=s.get("ontology_uri"),
    )


def _reconstruct_solution_from_json(person_data: dict):
    """Rebuild a `SchedulingSolution` from a persisted augmented JSON.

    Reconstructs the full `scheduled` and `unscheduled` lists, and the
    `tasks` set as their union (matching how the augmenters populate
    `solution.tasks` with per-week instances), so the authoritative
    evaluate-time scoring sees a correct `L_cov` denominator and the same
    applicability inputs the augmenter saw.
    """
    from src.scripts.scenarios.domain.calendar import AugmentedCalendar
    from src.scripts.scenarios.domain.solution import SchedulingSolution
    from src.scripts.scenarios.domain.task import ScheduledTask

    person_id = person_data["person_id"]
    scheduled: list[ScheduledTask] = []
    for s in person_data.get("scheduled", []):
        task = _recommended_task_from_dict(s)
        scheduled.append(
            ScheduledTask(
                task=task,
                start_minutes=int(s["start_minutes"]),
                end_minutes=int(s["end_minutes"]),
                is_standalone=bool(s.get("is_standalone", True)),
                concurrent_with=s.get("concurrent_with"),
                date=datetime.date.fromisoformat(s["date"]),
                parent_task_label=s.get("parent_task_label"),
            )
        )
    unscheduled = [
        _recommended_task_from_dict(s) for s in person_data.get("unscheduled", [])
    ]
    tasks = [st.task for st in scheduled] + list(unscheduled)

    return SchedulingSolution(
        person_id=person_id,
        augmented_calendar=AugmentedCalendar(
            person_id=person_id, scheduled_tasks=list(scheduled)
        ),
        tasks=tasks,
        scheduled=scheduled,
        unscheduled=unscheduled,
    )


def _cmd_evaluate_with_cfg(cfg: ScenarioConfig, args: argparse.Namespace) -> int:
    """Core evaluate logic given an already-loaded `ScenarioConfig`.

    Evaluate is the single source of truth: it reconstructs each
    augmented solution from `<pid>.json` and scores every leg from
    scratch with the authoritative stack (judge oracle for `merge`,
    persona constraints for `pref`). A per-person input-applicability
    mask decides which legs count; an in-mask leg with no signal, or a
    placement-free solution, is charged full loss. Nothing is read from
    augment-time persisted scores.
    """
    run_dir: Path = getattr(args, "run_dir", None) or Path(cfg.output.dir)
    persons_dir = run_dir / "augmented" / "persons"
    eval_dir = run_dir / "evaluation"

    if not persons_dir.exists():
        log.error("No augmented persons at %s", persons_dir)
        return EXIT_USAGE

    # Drive off the solution files; exclude the `_`-suffixed JSON sidecars
    # (`_loss`, `_weekly_gain`, `_rl_training`) so only the per-person
    # `<pid>.json` solutions remain.
    solution_files = sorted(
        p
        for p in persons_dir.glob("*.json")
        if not p.stem.endswith(("_loss", "_weekly_gain", "_rl_training"))
    )
    if not solution_files:
        log.error("No solution records found in %s", persons_dir)
        return EXIT_USAGE

    log.debug("Reading %d solutions from %s …", len(solution_files), persons_dir)

    eval_dir.mkdir(parents=True, exist_ok=True)
    oracle = _build_judge_oracle(
        eval_dir / "oracle_cache.jsonl",
        evaluator_model=getattr(cfg, "evaluator_model", None),
    )

    run_obj, run_dir_for_log = _try_load_persona_run(cfg)
    persona_run = None
    if isinstance(run_obj, str):
        log.warning(
            "Evaluate cannot load persona run at %s (%s); pref/merge/context "
            "legs will be out of mask for every person.",
            run_dir_for_log,
            run_obj,
        )
    elif run_obj is None:
        log.warning(
            "Evaluate cannot find persona run dir %r; pref/merge/context "
            "legs will be out of mask for every person.",
            str(run_dir_for_log) if run_dir_for_log else None,
        )
    else:
        persona_run, _ = _scope_run_to_timeframe(run_obj, cfg)

    from src.scripts.persona.domain.time_windows import WindowMap
    from src.scripts.scenarios.export.report_writer import write_evaluation_reports
    from src.scripts.scenarios.metrics.allen import RuleSet, SelectorMatcher
    from src.scripts.scenarios.metrics.intensity_resolver import IntensityResolver
    from src.scripts.scenarios.metrics.loss import (
        SchedulingLoss,
        compute_applicability_mask,
    )
    from src.scripts.scenarios.metrics.semantic import SemanticCompatibility
    from src.scripts.scenarios.metrics.telemetry import (
        EvaluateTelemetry,
        _Timer,
        write_evaluate_sidecar,
    )

    traces_by_id: dict[str, Any] = {}
    if persona_run is not None:
        traces_by_id = {t.person_id: t for t in persona_run.traces}

    # Build the authoritative scoring stack. `merge` uses the LLM-judge
    # oracle (or `None` when unavailable, in which case the merge leg is
    # left out of the mask). The bridge bypasses the judge when a
    # (task, host) pair shares a HumanActivities family.
    quartiles = _load_met_quartiles_or_default()
    met_lookup = _build_met_lookup_with_bridge()
    resolver = IntensityResolver(quartiles, met_lookup=met_lookup)
    matcher = SelectorMatcher(resolver=resolver)
    ruleset = RuleSet(list(getattr(persona_run, "allen_pair_rules", []) or []))
    eval_cfg = getattr(cfg, "evaluation", None)
    buffer_minutes = getattr(eval_cfg, "buffer_minutes", 30) if eval_cfg else 30
    merge_threshold = (
        getattr(eval_cfg, "merge_threshold", cfg.augmentation.merge_threshold)
        if eval_cfg
        else cfg.augmentation.merge_threshold
    )
    half_life_days = getattr(eval_cfg, "disp_half_life_days", 2.0) if eval_cfg else 2.0
    divide_tolerance_pct = (
        getattr(eval_cfg, "divide_duration_tolerance_pct", 0.15) if eval_cfg else 0.15
    )
    activity_bridge = _build_activity_bridge(cfg)

    eval_ttl_path = Path("src/assets/ontologies/HealthTasks_2026.05.19.ttl")
    eval_ctx_iris_path = Path("src/assets/ontologies/context_iris.json")
    if eval_ttl_path.exists() and eval_ctx_iris_path.exists():
        from src.scripts.scenarios.metrics.context_fit import (
            load_context_categories_by_iri,
            load_context_links_by_uri,
        )

        eval_context_links = load_context_links_by_uri(
            eval_ttl_path, load_context_categories_by_iri(eval_ctx_iris_path)
        )
    else:
        eval_context_links = {}

    # `merge` is scored by the judge oracle when present; otherwise a stub
    # SemanticCompatibility (constant 0.0) makes `compute_l_concurrent`
    # return None and the merge leg drops out of the mask.
    merge_semantic = oracle if oracle is not None else SemanticCompatibility()
    # L_pref uses the embedding oracle (`score_with_strategy`); L_merge uses
    # the judge's `score` via `merge_semantic`.
    pref_semantic = _build_semantic_oracle(
        run_dir, evaluator_model=getattr(cfg, "evaluator_model", None)
    )
    loss_fn = SchedulingLoss(
        weights=cfg.loss,
        semantic=pref_semantic,
        ruleset=ruleset,
        matcher=matcher,
        resolver=resolver,
        merge_threshold=merge_threshold,
        buffer_minutes=buffer_minutes,
        half_life_days=half_life_days,
        max_met=quartiles.max_met,
        divide_tolerance_pct=divide_tolerance_pct,
        context_links_by_uri=eval_context_links,
        merge_bridge=activity_bridge,
        merge_semantic=merge_semantic,
    )

    # Augmenter's observation budget scopes the L_context_fit denominator
    # to categories the augmenter could see; blind methods leave it None.
    eval_observation_block = getattr(cfg.augmentation, "observation", None)
    eval_observed_categories: frozenset[str] | None = (
        frozenset(eval_observation_block.contexts)
        if eval_observation_block is not None and eval_observation_block.contexts
        else None
    )

    # L_pref machinery, built once: catalog event config, window map,
    # preference mapper, and the per-person JSON lookup. A wiring failure
    # leaves pref out of the mask (logged), never crashing evaluate.
    pref_event_config = None
    pref_window_map: WindowMap | None = None
    pref_mapper = None
    pref_person_lookup: dict[str, str] = {}
    pref_horizon_start = getattr(persona_run, "horizon_start_date", None)
    pref_horizon_days = getattr(persona_run, "horizon_days", 0)
    if persona_run is not None:
        try:
            from src.scripts.persona.config.loader import load_event as _load_event
            from src.scripts.scenarios.metrics.preference_cache import PreferenceCache
            from src.scripts.scenarios.metrics.preference_mapping import (
                PreferenceMapper,
            )

            if cfg.calendar.persona_events:
                pref_event_config = _load_event(Path(cfg.calendar.persona_events))
                pref_window_map = WindowMap.from_config(persona_run.time_windows)
                pref_driver = None
                try:
                    from src.graphrag.config import Neo4jSettings
                    from src.graphrag.neo4j_client import make_driver

                    pref_driver = make_driver(Neo4jSettings.from_env())
                    pref_driver.verify_connectivity()
                except Exception as exc:
                    log.info(
                        "Evaluate L_pref: Neo4j ancestor walk unavailable (%s); "
                        "tier 2 disabled, semantic tier only.",
                        exc,
                    )
                    pref_driver = None
                pref_mapper = PreferenceMapper(
                    cache=PreferenceCache.from_env(),
                    neo4j_driver=pref_driver,
                    semantic=pref_semantic,
                )
                pref_person_lookup = _load_person_json_lookup(
                    run_dir_for_log or run_dir
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Evaluate L_pref wiring failed (pref out of mask): %s", exc)
            pref_event_config = None
            pref_window_map = None
            pref_mapper = None
            pref_person_lookup = {}

    from src.scripts.persona.concurrency.progress import (
        default_show_progress,
        wrap_progress,
    )

    show_bar = default_show_progress()
    results = []
    eval_telems: list[EvaluateTelemetry] = []
    weekly_records_by_person: dict[str, list[dict]] = {}
    eval_method = persons_dir.parent.parent.name or ""
    for sf in wrap_progress(
        solution_files, total=len(solution_files), desc="Evaluating", show=show_bar
    ):
        try:
            sol_data = json.loads(sf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Skipping unreadable solution %s: %s", sf.name, exc)
            continue
        person_id = sol_data["person_id"]
        trace = traces_by_id.get(person_id)
        _evaluator_model = getattr(cfg, "evaluator_model", None)
        _judge_model = getattr(oracle, "_model", "") if oracle is not None else ""
        eval_telem = EvaluateTelemetry(
            person_id=person_id,
            scenario_id=cfg.id or "",
            method=eval_method,
            provider=(
                (_evaluator_model.provider or "openai")
                if _evaluator_model is not None
                else "openai"
            ),
            judge_model=_judge_model if isinstance(_judge_model, str) else "",
        )
        # Duck-type the recorder so injected oracle test doubles without
        # a real recorder (or with mock auto-attributes) are ignored.
        judge_recorder = getattr(oracle, "call_recorder", None)
        if not isinstance(getattr(judge_recorder, "calls", None), list):
            judge_recorder = None
        if judge_recorder is not None:
            judge_recorder.reset()

        with _Timer() as person_timer:
            solution = _reconstruct_solution_from_json(sol_data)
            persona_constraints = (
                _build_persona_constraints_for_trace(
                    trace=trace,
                    run_dir=run_dir_for_log or run_dir,
                    person_json_lookup=pref_person_lookup,
                    event_config=pref_event_config,
                    window_map=pref_window_map,
                    horizon_days=pref_horizon_days,
                    horizon_start_date=pref_horizon_start,
                    mapper=pref_mapper,
                    experiment=getattr(cfg, "experiment", "default"),
                    scenario=cfg.id,
                )
                if trace is not None
                else None
            )
            # The trace carries the base calendar; without it (persona run
            # missing) cal/disp/merge score against an empty calendar
            # (vacuous), and pref/merge/context fall out of the mask.
            if trace is not None:
                score_calendar = trace
            else:
                from src.scripts.scenarios.domain.calendar import CalendarTrace

                score_calendar = CalendarTrace(
                    person_id=person_id, events=[], contexts=[]
                )
            try:
                mask = compute_applicability_mask(
                    solution.tasks,
                    score_calendar,
                    cfg.loss,
                    persona_constraints=persona_constraints,
                    semantic=merge_semantic,
                    context_links_by_uri=eval_context_links,
                    observed_categories=eval_observed_categories,
                    merge_threshold=merge_threshold,
                )
                with _Timer() as sem_timer:
                    new_total, new_comp = loss_fn.compute(
                        solution,
                        score_calendar,
                        pref_horizon_days or 7,
                        getattr(persona_run, "time_windows", None),
                        persona_constraints=persona_constraints,
                        window_map=pref_window_map,
                        observed_categories=eval_observed_categories,
                        mask=mask,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(
                    "Scoring failed for %s: %s; skipping this person.",
                    person_id,
                    exc,
                )
                continue
            if oracle is not None and "merge" in mask:
                eval_telem.record_semantic_lookup(wall_time_seconds=sem_timer.elapsed)

            # Authoritative per-week gain over the reconstructed solution,
            # under the same oracle + mask, sliced per ISO week.
            weekly_records_by_person[person_id] = _evaluate_weekly_gain(
                solution=solution,
                calendar=score_calendar,
                loss_fn=loss_fn,
                persona_run=persona_run,
                persona_constraints=persona_constraints,
                window_map=pref_window_map,
                observed_categories=eval_observed_categories,
                mask=mask,
            )
            _write_evaluate_leg_sidecars(
                solution=solution,
                trace=trace,
                persons_dir=persons_dir,
                person_id=person_id,
                components=new_comp,
                persona_constraints=persona_constraints,
                window_map=pref_window_map,
                semantic=pref_semantic,
                context_links=eval_context_links,
                observed_categories=eval_observed_categories,
            )
        eval_telem.wall_time_seconds = person_timer.elapsed
        if judge_recorder is not None:
            for in_tok, out_tok, wall, reasoning_tok in judge_recorder.calls:
                eval_telem.record_judge_call(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    wall_time_seconds=wall,
                    reasoning_tokens=reasoning_tok,
                )
        eval_telems.append(eval_telem)
        results.append((person_id, new_total, new_comp))

    # Ontology-grounding metric; share of generated tasks whose
    # `ontology_uri` survived the bridge's resolve check.  Tasks
    # with a fabricated URI (stripped to `None` during generation)
    # count as ungrounded.  `tasks/` lives next to the augmented
    # output for both single-scenario (`run_dir/tasks`) and
    # multi-scenario (`run_dir/../tasks`) layouts; we try both.
    from src.scripts.scenarios.metrics.grounding import compute_grounding
    from src.scripts.scenarios.metrics.telemetry import aggregate_full_pipeline
    from src.scripts.scenarios.task_generation.ontology_bridge import (
        uri_resolves as _uri_resolves,
    )

    person_ids = [pid for pid, _, _ in results]
    # Candidate task dirs, in priority order:
    #
    #   1. an explicit `args.tasks_dir` from the multi-scenario runner
    #      (`_multi_scenario_tasks_dir` is the source of truth for both
    #      augment and evaluate paths in the new tree-style layout).
    #   2. legacy single-scenario `<run_dir>/tasks/`.
    #   3. legacy multi-scenario `<run_dir>/../tasks/` (when augment
    #      output lives under `<scenario_id>/<method>/`).
    #
    # When NONE is a real directory, `grounding` stays `None` and the
    # ontology_grounding.json sidecar is never written; which means
    # the benchmark report's `## Ontology grounding` section renders
    # all-`n/a` cells. Adding entry (1) fixes the 2026-05-18 regression
    # where the new `task_generation/<id>/tasks/` umbrella was unreachable
    # from any of the legacy run-dir-relative candidates.
    explicit_tasks_dir = getattr(args, "tasks_dir", None)
    candidate_tasks_dirs: list[Path] = []
    if explicit_tasks_dir is not None:
        candidate_tasks_dirs.append(Path(explicit_tasks_dir))
    candidate_tasks_dirs += [run_dir / "tasks", run_dir.parent / "tasks"]
    grounding: dict | None = None
    matched_tasks_dir: Path | None = None
    # Across the horizon a person's task file holds the sum of every
    # week's count, so the grounding expectation sums the per-week totals.
    weekly_cfgs = cfg.weekly_task_generation()
    if len(weekly_cfgs) == 1:
        expected_total = weekly_cfgs[0].num_tasks
    else:
        expected_total = sum(c.num_tasks for c in weekly_cfgs)

    # Per-person URI-validation counter: a single shared counter keyed
    # by person_id won't work because `compute_grounding` doesn't
    # tell the validator which person owns the URI it's checking.
    # We keep a global tally and divide it equally across persons at
    # write time; coarser than per-person but enough to surface the
    # cohort-level URI-validation cost.
    uri_validation_total_calls = 0
    uri_validation_total_wall = 0.0

    # Build a uri_validator that re-checks every URI against the live
    # ontology; best-effort.  When Neo4j is unreachable we skip the
    # verification pass and the report renders "n/a" for that column.
    validator = None
    eval_driver = None
    eval_session_ctx = None
    try:
        from src.graphrag.config import Neo4jSettings
        from src.graphrag.neo4j_client import make_driver, session_scope

        eval_driver = make_driver(Neo4jSettings.from_env())
        eval_driver.verify_connectivity()
        eval_session_ctx = session_scope(eval_driver)
    except Exception as exc:
        log.debug(
            "evaluate: Neo4j unreachable (%s); skipping URI re-validation.",
            exc,
        )
        eval_driver = None
        eval_session_ctx = None

    try:
        eval_session = (
            eval_session_ctx.__enter__() if eval_session_ctx is not None else None
        )
        if eval_session is not None:

            def validator(uri: str) -> bool:
                nonlocal uri_validation_total_calls, uri_validation_total_wall
                with _Timer() as uri_timer:
                    result = _uri_resolves(uri, eval_session)
                uri_validation_total_calls += 1
                uri_validation_total_wall += uri_timer.elapsed
                return result

        for candidate in candidate_tasks_dirs:
            if candidate.is_dir():
                grounding = compute_grounding(
                    candidate,
                    person_ids,
                    expected_total=expected_total,
                    uri_validator=validator,
                )
                matched_tasks_dir = candidate
                break
    finally:
        if eval_session_ctx is not None:
            eval_session_ctx.__exit__(None, None, None)
        if eval_driver is not None:
            eval_driver.close()

    # Distribute the URI-validation totals across per-person evaluate
    # sidecars before writing them.  Equal split; see comment above.
    if eval_telems and uri_validation_total_calls:
        n = len(eval_telems)
        per_person_calls = uri_validation_total_calls // n
        per_person_wall = uri_validation_total_wall / n
        for telem in eval_telems:
            telem.uri_validation.n_calls = per_person_calls
            telem.uri_validation.wall_time_seconds = round(per_person_wall, 4)
    for telem in eval_telems:
        write_evaluate_sidecar(telem, eval_dir)

    # Roll up task-gen + augment + evaluate sidecars into the unified
    # telemetry summary the report writer consumes.  The augmented dir
    # lives one level above `persons/`; the evaluate dir is the
    # currently-being-written `eval_dir`.  When the task-gen sidecar
    # path can't be resolved (no grounded path was used), the
    # aggregator falls back to an empty task-generation block.
    augment_dir_for_aggregate = persons_dir.parent
    telemetry_summary = aggregate_full_pipeline(
        tasks_dir=matched_tasks_dir or (run_dir / "tasks"),
        augmented_dir=augment_dir_for_aggregate,
        evaluation_dir=eval_dir,
        scenario_id=cfg.id or "",
        method=eval_method,
    )

    # Aggregate per-persona `preference_violations.jsonl` sidecars
    # into the per-leg `preference_breakdown` dict the
    # reporter renders in its `L_pref preference breakdown` block.
    try:
        from src.scripts.scenarios.metrics.preference_score import (
            read_preference_violations,
        )

        sidecar_paths = list(persons_dir.glob("*_pref_violations.jsonl"))
        preference_breakdown = read_preference_violations(sidecar_paths)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Could not read preference_violations sidecars: %s", exc)
        preference_breakdown = None

    # Cohort divide_breakdown dict for the L_divide report block.
    try:
        from src.scripts.scenarios.metrics.divide_breakdown import read_divide_verdicts

        divide_paths = list(persons_dir.glob("*_divide_verdicts.jsonl"))
        divide_breakdown = read_divide_verdicts(divide_paths)
        if not divide_breakdown.get("applicable_buckets"):
            divide_breakdown = None
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Could not read divide_verdicts sidecars: %s", exc)
        divide_breakdown = None

    # Cohort context_fit_breakdown dict for the L_context_fit report block.
    try:
        from src.scripts.scenarios.metrics.context_fit_breakdown import (
            read_context_fit_verdicts,
        )

        ctx_paths = list(persons_dir.glob("*_context_fit.jsonl"))
        context_fit_breakdown = read_context_fit_verdicts(ctx_paths)
        if not context_fit_breakdown.get("applicable_tasks"):
            context_fit_breakdown = None
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Could not read context_fit sidecars: %s", exc)
        context_fit_breakdown = None

    # Write the authoritative per-person weekly-gain sidecars from the
    # evaluate-time recompute, then aggregate them for the report.
    try:
        from src.scripts.scenarios.export.json_writer import write_weekly_gain_sidecar

        for pid, weekly_records in weekly_records_by_person.items():
            write_weekly_gain_sidecar(
                persons_dir / f"{pid}_weekly_gain.json",
                person_id=pid,
                weekly_records=weekly_records,
            )
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Could not write weekly gain sidecars: %s", exc)

    try:
        from src.scripts.scenarios.export.report_writer import aggregate_weekly_gain

        weekly_gain_summary = aggregate_weekly_gain(persons_dir)
        if not weekly_gain_summary.get("weeks"):
            weekly_gain_summary = None
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Could not aggregate weekly gain sidecars: %s", exc)
        weekly_gain_summary = None

    write_evaluation_reports(
        results,
        eval_dir,
        grounding=grounding,
        telemetry=telemetry_summary,
        preference_breakdown=preference_breakdown or None,
        divide_breakdown=divide_breakdown,
        context_fit_breakdown=context_fit_breakdown,
        window=_read_timeframe_resolved(run_dir / "augmented" / "timeframe.json"),
        weekly_gain=weekly_gain_summary,
    )
    if getattr(args, "learning_charts", False):
        _render_learning_charts_for_run(
            eval_dir,
            persons_dir,
            scenario_id=(cfg.id or "").split("/")[0],
            method=cfg.augmentation.method,
        )
    # Exclude personas whose generated task list was empty so a 0/N
    # fetch failure does not appear as a perfect 1.0 gain on the
    # console (matches the new `results_to_dict` exclusion policy).
    empty_pids = {
        pid
        for pid, info in ((grounding or {}).get("per_person") or {}).items()
        if not (info or {}).get("total")
    }
    scored_results = [r for r in results if r[0] not in empty_pids]
    if scored_results:
        avg_gain: float | None = 1.0 - sum(r[1] for r in scored_results) / len(
            scored_results
        )
    else:
        avg_gain = None
    avg_gain_s = "n/a" if avg_gain is None else f"{avg_gain:.4f}"
    short_suffix = f"  (excluded {len(empty_pids)} empty plans)" if empty_pids else ""
    if grounding is not None and grounding["total"]:
        log.info(
            "evaluate complete: %d persons, avg_gain=%s, "
            "ontology_grounding=%d/%d (%.1f%%)%s",
            len(results),
            avg_gain_s,
            grounding["grounded"],
            grounding["total"],
            grounding["ratio"] * 100,
            short_suffix,
        )
    else:
        log.info(
            "evaluate complete: %d persons, avg_gain=%s%s",
            len(results),
            avg_gain_s,
            short_suffix,
        )
    print(
        f"[evaluate] scenario={cfg.id} to {len(results)} persons, "
        f"avg_gain={avg_gain_s}, {eval_dir}"
    )
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Dispatch evaluate to single-scenario or multi-scenario handler."""
    try:
        raw = load_config(args.scenario)
    except ScenarioConfigError as exc:
        log.error("Config error: %s", exc)
        return EXIT_USAGE

    if isinstance(raw, ExperimentScenariosConfig):
        return _multi_evaluate(
            raw,
            args,
            getattr(args, "scenario_id", None),
            getattr(args, "method", None),
        )
    return _cmd_evaluate_with_cfg(raw, args)


def _cmd_report(args: argparse.Namespace) -> int:
    """Aggregate per-scenario evaluation sidecars into a Markdown report.

    Pure read-from-disk command; never re-evaluates.
    """
    from src.scripts.scenarios.export.benchmark_report import write_benchmark_report

    try:
        raw = load_config(args.scenario)
    except ScenarioConfigError as exc:
        log.error("Config error: %s", exc)
        return EXIT_USAGE

    if not isinstance(raw, ExperimentScenariosConfig):
        log.error(
            "report: --scenario must point at a multi-scenario YAML "
            "(top-level `scenarios:` list); legacy single-scenario configs "
            "are not aggregated."
        )
        return EXIT_USAGE

    scenario_id = getattr(args, "scenario_id", None)
    method_filter = getattr(args, "method", None)
    rc = _validate_multi_filters(raw, scenario_id, method_filter)
    if rc != EXIT_OK:
        return rc

    selected = _normalize_scenario_id_filter(scenario_id)
    selected_set = set(selected) if selected else None
    seed_pairs: list[tuple[str, str]] = []
    for scenario in raw.scenarios:
        if selected_set is not None and scenario.id not in selected_set:
            continue
        for method_cfg in scenario.augmentation:
            if method_filter and method_cfg.method != method_filter:
                continue
            seed_pairs.append((scenario.id, method_cfg.method))

    from src.scripts.persona.cli import _load_experiment_name_from_run_dir

    experiment_dir = Path(raw.effective_run_dir())
    # Walk disk for both top-level evaluations and per-variant
    # ablation evaluations so the report covers every artifact present
    # regardless of which `--scenario-id` was passed to the last evaluate.
    pairs = _augment_pairs_with_disk_runs(experiment_dir, raw, seed_pairs)
    ablation_runs = _augment_ablation_runs_with_disk(experiment_dir, raw, [])

    baseline = _parse_baseline_arg(getattr(args, "baseline", None), pairs)
    experiment_name = _load_experiment_name_from_run_dir(experiment_dir)
    target = write_benchmark_report(
        experiment_dir,
        raw.experiment_id,
        scenarios_cfg=raw,
        experiment_name=experiment_name,
        scenario_method_pairs=pairs or None,
        baseline=baseline,
        ablation_runs=ablation_runs or None,
    )
    if getattr(args, "learning_charts", False):
        for sid, method in pairs:
            base = experiment_dir / "scenarios" / sid / method
            _render_learning_charts_for_run(
                base / "evaluation",
                base / "augmented" / "persons",
                scenario_id=sid,
                method=method,
            )
        labels = {s.id: s.label for s in raw.scenarios if getattr(s, "label", "")}
        _render_cross_augmenter_charts(experiment_dir, pairs, labels)
    print(f"[report] wrote benchmark report to {target}")
    return EXIT_OK


def _parse_baseline_arg(
    raw: str | None,
    pairs: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Parse `--baseline ID[/METHOD]` into a `(scenario_id, method)` tuple.

    `None` means "let `write_benchmark_report` pick the first run".
    When the user passes a bare scenario id without a method, we resolve
    to the first method discovered for that id so the Δ-table has a real
    target; otherwise the first overall pair wins.
    """
    if not raw:
        return None
    if "/" in raw:
        sid, method = raw.split("/", 1)
        return (sid.strip(), method.strip())
    sid = raw.strip()
    for s_id, method in pairs:
        if s_id == sid:
            return (s_id, method)
    return (sid, "")


def _cmd_run(args: argparse.Namespace) -> int:
    """End-to-end: generate-tasks to augment to evaluate (single or multi-scenario)."""
    try:
        raw = load_config(args.scenario)
    except ScenarioConfigError as exc:
        log.error("Config error: %s", exc)
        return EXIT_USAGE

    if isinstance(raw, ExperimentScenariosConfig):
        scenario_id = getattr(args, "scenario_id", None)
        method_filter = getattr(args, "method", None)
        rc = _multi_generate_tasks(raw, args, scenario_id)
        if rc != EXIT_OK:
            return rc
        rc = _multi_augment(raw, args, scenario_id, method_filter)
        if rc != EXIT_OK:
            return rc
        return _multi_evaluate(raw, args, scenario_id, method_filter)

    # Legacy single-scenario path.
    gen_args = argparse.Namespace(
        scenario=args.scenario,
        out_dir=getattr(args, "out_dir", None),
        seed=getattr(args, "seed", None),
        workers=getattr(args, "workers", 5),
        log_level=getattr(args, "log_level", "INFO"),
    )
    rc = _cmd_generate_tasks_with_cfg(raw, gen_args)
    if rc != EXIT_OK:
        return rc

    aug_args = argparse.Namespace(
        scenario=args.scenario,
        out_dir=getattr(args, "out_dir", None),
        tasks_dir=None,
        method=None,
        log_level=getattr(args, "log_level", "INFO"),
        charts=getattr(args, "charts", None),
        calendar_dpi=getattr(args, "calendar_dpi", 300),
        calendar_min_event_minutes=getattr(args, "calendar_min_event_minutes", 45),
    )
    rc = _cmd_augment_with_cfg(raw, aug_args)
    if rc != EXIT_OK:
        return rc

    ev_args = argparse.Namespace(
        scenario=args.scenario,
        run_dir=getattr(args, "out_dir", None),
        log_level=getattr(args, "log_level", "INFO"),
    )
    return _cmd_evaluate_with_cfg(raw, ev_args)


if __name__ == "__main__":
    sys.exit(main())
