"""Token / latency / count telemetry for the full scenarios pipeline.

There are three per-person sidecar families, one per pipeline stage:

  * `output/<exp>/<scenario>/tasks/_telemetry/<person_id>.json`
   ; task generation: GraphRAG fetch + paraphrase + gate.
  * `output/<exp>/<scenario>/<method>/augmented/_telemetry/<person_id>.json`
   ; augmentation: wall-time always, LLM tokens / cost when the
    augmenter is `llm_agent`.
  * `output/<exp>/<scenario>/<method>/evaluation/_telemetry/<person_id>.json`
   ; evaluation: wall-time always, semantic-oracle embedding lookups,
    Neo4j URI re-validations, and LLM-judge tokens / cost when the
    judge oracle is wired.

A per-stage aggregator rolls those sidecars up into a single `by_stage`
+ `by_persona` block, written to `telemetry.{txt,json}` by the
report writer.

Naming convention; `person` vs `persona`:

  * `person` is an individual sampled from a persona (cohort
    template) and identified by `person_id` like
    `b_parttime_morning_0001`.  Per-person stats average over the
    individual sidecar files.
  * `persona` is the cohort template defined in
    `persona_config.yaml`.  Per-persona stats group persons by their
    `persona_id` (e.g. `b_parttime_morning`) and report cohort-
    level totals plus the within-cohort per-person average.

Token costs are estimated from a per-model pricing table loaded from
`config/llm_pricing.json` and keyed on `provider` + `model` (USD per 1M
tokens).  Unknown models return `estimated_usd = None`; never crash.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens), loaded from a JSON table
# ---------------------------------------------------------------------------

# Single source of truth for per-model prices.  The file path is the
# only pricing detail hardcoded here; the numbers live in the JSON and
# are refreshed with `python -m src.scripts.scenarios.metrics.pricing_update`.
PRICING_FILE = Path(__file__).resolve().parent.parent / "config" / "llm_pricing.json"

_PRICING: dict[str, dict[str, dict[str, float]]] | None = None


def load_pricing(*, force: bool = False) -> dict[str, dict[str, dict[str, float]]]:
    """Return the per-model price table read once from `PRICING_FILE`.

    An unreadable or malformed file logs a warning and returns an empty
    table, so every cost downstream renders `n/a` rather than crashing.
    Pass `force=True` to re-read after the file is rewritten.
    """
    global _PRICING
    if _PRICING is None or force:
        try:
            data = json.loads(PRICING_FILE.read_text(encoding="utf-8"))
            _PRICING = data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "pricing file %s unreadable (%s); costs render n/a.",
                PRICING_FILE,
                exc,
            )
            _PRICING = {}
    return _PRICING


def estimate_usd(
    *, provider: str, model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """Return the USD cost estimate for one LLM round-trip, or `None`.

    The lookup is exact-match against `PRICING_FILE`; an unknown
    provider, unknown model, or malformed entry returns `None` so the
    consumer renders `n/a` instead of a misleading number.
    """
    table = load_pricing().get(provider, {}).get(model)
    if not isinstance(table, dict):
        return None
    price_in = table.get("in")
    price_out = table.get("out")
    if not isinstance(price_in, (int, float)) or not isinstance(
        price_out, (int, float)
    ):
        return None
    return (input_tokens / 1_000_000) * price_in + (
        output_tokens / 1_000_000
    ) * price_out


_TIKTOKEN_ENCODER: object | None = None


def estimate_tokens(text: str) -> int:
    """Best-effort token count for *text*.

    Uses `tiktoken` (the OpenAI tokenizer; its counts correlate well
    with most other vendors' counts) when available.  Falls back to
    `len(text) // 4` (the standard rough-estimate ratio) when
    `tiktoken` is missing or its encoder cannot load, so the
    telemetry sidecar always has a non-zero number even on minimal
    installs and even when the underlying LLM client doesn't expose
    `usage_metadata` (an earlier regression).
    """
    if not text:
        return 0
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            import tiktoken

            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover; exercised only when tiktoken absent
            _TIKTOKEN_ENCODER = False
    if _TIKTOKEN_ENCODER:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:  # pragma: no cover; defensive only
            pass
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# person_id to persona_id
# ---------------------------------------------------------------------------

# By convention person_ids end with `_<NNNN>` where NNNN is a 4-digit
# instance index (e.g. `b_parttime_morning_0001`).  When the suffix
# matches we strip it to recover the persona_id; otherwise we fall
# back to the full person_id so the aggregator still groups predictably.
_PERSON_ID_INSTANCE_RE = re.compile(r"^(.*)_\d{4}$")


def derive_persona_id(person_id: str) -> str:
    """Recover the persona_id (cohort template name) from a person_id.

    The persona pipeline writes person_ids of the form
    `<persona_id>_<NNNN>` (4-digit instance index).  When the suffix
    is missing we return the input unchanged so callers always have a
    grouping key; there is no exception path for "unknown" personas.
    """
    if not person_id:
        return person_id
    match = _PERSON_ID_INSTANCE_RE.match(person_id)
    return match.group(1) if match else person_id


# ---------------------------------------------------------------------------
# Per-call records (task generation)
# ---------------------------------------------------------------------------


@dataclass
class LLMCallStats:
    """Token + latency aggregate for one logical LLM stage (fetch /
    paraphrase / augment / judge).  Multiple physical calls (retries)
    accumulate here."""

    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Provider-reported hidden reasoning share, already included in
    # `output_tokens` (and therefore in the cost); tracked separately
    # so reports can show how much of the spend is reasoning.
    reasoning_tokens: int = 0
    wall_time_seconds: float = 0.0


@dataclass
class EmbedderStats:
    """Aggregate for embedder calls (Stage 3 gate C)."""

    calls: int = 0
    input_tokens: int = 0  # best-effort; many embedders don't report
    wall_time_seconds: float = 0.0


@dataclass
class FetchSummary:
    """Roll-up of per-attempt fetch stats; mirrors `FetchResult`."""

    num_tasks_requested: int = 0
    num_tasks_accepted: int = 0
    fabricated_dropped_per_attempt: list[int] = field(default_factory=list)
    duplicate_dropped_per_attempt: list[int] = field(default_factory=list)
    parse_failed_per_attempt: list[bool] = field(default_factory=list)


@dataclass
class TaskTelemetryRow:
    """Per-task audit row; one entry under `"tasks"` in the sidecar."""

    label: str
    ontology_uri: str | None
    fetch_attempt: int  # 1-based
    paraphrase_used: str  # "paraphrase" | "canonical" | "skipped"
    length_delta: float | None
    similarity: float | None
    missing_emojis: dict[str, int]
    extra_emojis: dict[str, int]
    failed_gate: str | None
    canonical_description: str
    personalized_description: str


# ---------------------------------------------------------------------------
# Per-person collector (task generation)
# ---------------------------------------------------------------------------


@dataclass
class TaskGenTelemetry:
    """Single-person collector for the task-generation stage.

    `person_id` is the individual instance key (e.g.
    `b_parttime_morning_0001`); `persona_id` is the cohort template
    (e.g. `b_parttime_morning`).  The two are stored separately so
    the aggregator can group by persona without parsing person_ids.
    """

    person_id: str
    scenario_id: str
    method: str
    persona_id: str = ""
    provider: str = "openai"
    fetch_model: str = ""
    paraphrase_model: str = ""
    embedder_model: str = ""

    fetch_stats: LLMCallStats = field(default_factory=LLMCallStats)
    paraphrase_stats: LLMCallStats = field(default_factory=LLMCallStats)
    embedder_stats: EmbedderStats = field(default_factory=EmbedderStats)
    fetch_summary: FetchSummary = field(default_factory=FetchSummary)
    tasks: list[TaskTelemetryRow] = field(default_factory=list)

    wall_time_by_stage: dict[str, float] = field(
        default_factory=lambda: {"fetch": 0.0, "paraphrase": 0.0, "gate": 0.0}
    )

    def __post_init__(self) -> None:
        if not self.persona_id:
            self.persona_id = derive_persona_id(self.person_id)

    # ----------------- recording API ---------------------------------

    def record_llm_call(
        self,
        *,
        stage: str,
        input_tokens: int,
        output_tokens: int,
        wall_time_seconds: float,
    ) -> None:
        """Add one LLM round-trip to the named stage's aggregate."""
        bucket = self.fetch_stats if stage == "fetch" else self.paraphrase_stats
        bucket.attempts += 1
        bucket.input_tokens += int(input_tokens)
        bucket.output_tokens += int(output_tokens)
        bucket.total_tokens += int(input_tokens) + int(output_tokens)
        bucket.wall_time_seconds += float(wall_time_seconds)
        self.wall_time_by_stage[stage] = self.wall_time_by_stage.get(
            stage, 0.0
        ) + float(wall_time_seconds)

    def record_embedder_call(
        self, *, input_tokens: int, wall_time_seconds: float
    ) -> None:
        self.embedder_stats.calls += 1
        self.embedder_stats.input_tokens += int(input_tokens)
        self.embedder_stats.wall_time_seconds += float(wall_time_seconds)
        self.wall_time_by_stage["gate"] = self.wall_time_by_stage.get(
            "gate", 0.0
        ) + float(wall_time_seconds)

    def record_fetch_summary(
        self,
        *,
        num_tasks_requested: int,
        num_tasks_accepted: int,
        per_attempt_fabricated: Iterable[int],
        per_attempt_duplicate: Iterable[int],
        per_attempt_parse_failed: Iterable[bool],
    ) -> None:
        self.fetch_summary = FetchSummary(
            num_tasks_requested=int(num_tasks_requested),
            num_tasks_accepted=int(num_tasks_accepted),
            fabricated_dropped_per_attempt=list(per_attempt_fabricated),
            duplicate_dropped_per_attempt=list(per_attempt_duplicate),
            parse_failed_per_attempt=list(per_attempt_parse_failed),
        )

    def record_task(self, row: TaskTelemetryRow) -> None:
        self.tasks.append(row)

    # ----------------- serialization ---------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Render the per-person JSON sidecar payload."""
        total_wall = sum(self.wall_time_by_stage.values())
        return {
            "person_id": self.person_id,
            "persona_id": self.persona_id,
            "scenario_id": self.scenario_id,
            "method": self.method,
            "provider": self.provider,
            "models": {
                "fetch": self.fetch_model,
                "paraphrase": self.paraphrase_model,
                "embedder": self.embedder_model,
            },
            "wall_time_seconds_total": round(total_wall, 4),
            "wall_time_seconds_by_stage": {
                k: round(v, 4) for k, v in self.wall_time_by_stage.items()
            },
            "llm_calls": {
                "fetch": _llm_to_dict(
                    self.fetch_stats, self.provider, self.fetch_model
                ),
                "paraphrase": _llm_to_dict(
                    self.paraphrase_stats, self.provider, self.paraphrase_model
                ),
            },
            "embeddings": {
                "calls": self.embedder_stats.calls,
                "input_tokens": self.embedder_stats.input_tokens,
                "wall_time_seconds": round(self.embedder_stats.wall_time_seconds, 4),
                "estimated_usd": estimate_usd(
                    provider=self.provider,
                    model=self.embedder_model,
                    input_tokens=self.embedder_stats.input_tokens,
                    output_tokens=0,
                ),
            },
            "fetch_summary": {
                "num_tasks_requested": self.fetch_summary.num_tasks_requested,
                "num_tasks_accepted": self.fetch_summary.num_tasks_accepted,
                "fabricated_dropped_per_attempt": list(
                    self.fetch_summary.fabricated_dropped_per_attempt
                ),
                "duplicate_dropped_per_attempt": list(
                    self.fetch_summary.duplicate_dropped_per_attempt
                ),
                "parse_failed_per_attempt": list(
                    self.fetch_summary.parse_failed_per_attempt
                ),
            },
            "tasks": [
                {
                    "label": t.label,
                    "ontology_uri": t.ontology_uri,
                    "fetch_attempt": t.fetch_attempt,
                    "paraphrase": {
                        "used": t.paraphrase_used,
                        "length_delta": t.length_delta,
                        "similarity": t.similarity,
                        "missing_emojis": dict(t.missing_emojis),
                        "extra_emojis": dict(t.extra_emojis),
                        "failed_gate": t.failed_gate,
                        "canonical_description": t.canonical_description,
                        "personalized_description": t.personalized_description,
                    },
                }
                for t in self.tasks
            ],
        }


def _llm_to_dict(stats: LLMCallStats, provider: str, model: str) -> dict[str, Any]:
    # Reasoning tokens are billed at the output rate and are already a
    # subset of `output_tokens`, so `estimated_usd` covers them; the
    # separate reasoning estimate breaks out that share for reporting.
    reasoning_usd = None
    if stats.reasoning_tokens:
        reasoning_usd = estimate_usd(
            provider=provider,
            model=model,
            input_tokens=0,
            output_tokens=stats.reasoning_tokens,
        )
    return {
        "attempts": stats.attempts,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "total_tokens": stats.total_tokens,
        "reasoning_tokens": stats.reasoning_tokens,
        "wall_time_seconds": round(stats.wall_time_seconds, 4),
        "estimated_usd": estimate_usd(
            provider=provider,
            model=model,
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
        ),
        "estimated_reasoning_usd": reasoning_usd,
    }


# ---------------------------------------------------------------------------
# Per-person collector (augmentation)
# ---------------------------------------------------------------------------


@dataclass
class AugmentTelemetry:
    """Single-person collector for the augmentation stage.

    For `llm_agent` the `llm_calls` block carries token + cost
    aggregates; for `greedy` the same block stays at zeros (only
    wall-time is meaningful for a local solver, but keeping the shape
    uniform across methods means the aggregator and report renderer
    don't need a special case).
    """

    person_id: str
    scenario_id: str
    method: str  # "greedy" | "llm_agent"
    persona_id: str = ""
    provider: str = "openai"
    model: str = ""

    wall_time_seconds: float = 0.0
    n_tasks_attempted: int = 0
    n_tasks_placed: int = 0
    n_tasks_dropped: int = 0
    llm_calls: LLMCallStats = field(default_factory=LLMCallStats)

    def __post_init__(self) -> None:
        if not self.persona_id:
            self.persona_id = derive_persona_id(self.person_id)

    def record_llm_call(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        wall_time_seconds: float,
        reasoning_tokens: int = 0,
    ) -> None:
        """Add one augmenter LLM round-trip to the aggregate."""
        self.llm_calls.attempts += 1
        self.llm_calls.input_tokens += int(input_tokens)
        self.llm_calls.output_tokens += int(output_tokens)
        self.llm_calls.total_tokens += int(input_tokens) + int(output_tokens)
        self.llm_calls.reasoning_tokens += int(reasoning_tokens)
        self.llm_calls.wall_time_seconds += float(wall_time_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "persona_id": self.persona_id,
            "scenario_id": self.scenario_id,
            "method": self.method,
            "provider": self.provider,
            "model": self.model,
            "wall_time_seconds": round(self.wall_time_seconds, 4),
            "n_tasks_attempted": self.n_tasks_attempted,
            "n_tasks_placed": self.n_tasks_placed,
            "n_tasks_dropped": self.n_tasks_dropped,
            "llm_calls": _llm_to_dict(self.llm_calls, self.provider, self.model),
        }


# ---------------------------------------------------------------------------
# Per-person collector (evaluation)
# ---------------------------------------------------------------------------


@dataclass
class OracleStats:
    """Counter + timer for an evaluator-side oracle (semantic / URI / judge)."""

    n_calls: int = 0
    wall_time_seconds: float = 0.0


@dataclass
class EvaluateTelemetry:
    """Single-person collector for the evaluation stage.

    Captures wall-time of the per-person loss compute plus the cost of
    every external oracle the evaluator consults: the semantic-
    compatibility embedding lookups (Neo4j vector search), the URI
    re-validation pass (Neo4j Cypher exists-check), and; when wired -
    the LLM-judge oracle's token spend.
    """

    person_id: str
    scenario_id: str
    method: str
    persona_id: str = ""
    provider: str = "openai"
    judge_model: str = ""

    wall_time_seconds: float = 0.0
    semantic_oracle: OracleStats = field(default_factory=OracleStats)
    uri_validation: OracleStats = field(default_factory=OracleStats)
    judge_calls: LLMCallStats = field(default_factory=LLMCallStats)

    def __post_init__(self) -> None:
        if not self.persona_id:
            self.persona_id = derive_persona_id(self.person_id)

    def record_semantic_lookup(self, *, wall_time_seconds: float) -> None:
        self.semantic_oracle.n_calls += 1
        self.semantic_oracle.wall_time_seconds += float(wall_time_seconds)

    def record_uri_validation(self, *, wall_time_seconds: float) -> None:
        self.uri_validation.n_calls += 1
        self.uri_validation.wall_time_seconds += float(wall_time_seconds)

    def record_judge_call(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        wall_time_seconds: float,
        reasoning_tokens: int = 0,
    ) -> None:
        self.judge_calls.attempts += 1
        self.judge_calls.input_tokens += int(input_tokens)
        self.judge_calls.output_tokens += int(output_tokens)
        self.judge_calls.total_tokens += int(input_tokens) + int(output_tokens)
        self.judge_calls.reasoning_tokens += int(reasoning_tokens)
        self.judge_calls.wall_time_seconds += float(wall_time_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "persona_id": self.persona_id,
            "scenario_id": self.scenario_id,
            "method": self.method,
            "provider": self.provider,
            "judge_model": self.judge_model,
            "wall_time_seconds": round(self.wall_time_seconds, 4),
            "semantic_oracle": {
                "n_calls": self.semantic_oracle.n_calls,
                "wall_time_seconds": round(self.semantic_oracle.wall_time_seconds, 4),
            },
            "uri_validation": {
                "n_calls": self.uri_validation.n_calls,
                "wall_time_seconds": round(self.uri_validation.wall_time_seconds, 4),
            },
            "judge_calls": _llm_to_dict(
                self.judge_calls, self.provider, self.judge_model
            ),
        }


# ---------------------------------------------------------------------------
# Sidecar writers
# ---------------------------------------------------------------------------

_SUMMARY_NAME = "_summary.json"


def _write_sidecar(payload: dict[str, Any], dir_path: Path, person_id: str) -> Path:
    """Persist *payload* to `<dir_path>/_telemetry/<person_id>.json`."""
    out_dir = Path(dir_path) / "_telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{person_id}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def write_taskgen_sidecar(telemetry: TaskGenTelemetry, tasks_dir: Path) -> Path:
    """Persist the per-person task-generation telemetry JSON.

    The file lands at `<tasks_dir>/_telemetry/<person_id>.json`.
    Parent directories are created if missing.
    """
    return _write_sidecar(telemetry.to_dict(), tasks_dir, telemetry.person_id)


def write_augment_sidecar(telemetry: AugmentTelemetry, augmented_dir: Path) -> Path:
    """Persist the per-person augmentation telemetry JSON.

    The file lands at `<augmented_dir>/_telemetry/<person_id>.json`
    where `augmented_dir` is `<scenario>/<method>/augmented/`.
    """
    return _write_sidecar(telemetry.to_dict(), augmented_dir, telemetry.person_id)


def write_evaluate_sidecar(telemetry: EvaluateTelemetry, evaluation_dir: Path) -> Path:
    """Persist the per-person evaluation telemetry JSON.

    The file lands at `<evaluation_dir>/_telemetry/<person_id>.json`
    where `evaluation_dir` is `<scenario>/<method>/evaluation/`.
    """
    return _write_sidecar(telemetry.to_dict(), evaluation_dir, telemetry.person_id)


def _read_sidecar_payloads(stage_dir: Path) -> list[dict[str, Any]]:
    """Read every `<stage_dir>/_telemetry/<person>.json` payload."""
    tel_dir = Path(stage_dir) / "_telemetry"
    if not tel_dir.is_dir():
        return []
    payloads: list[dict[str, Any]] = []
    for fp in sorted(p for p in tel_dir.glob("*.json") if p.name != _SUMMARY_NAME):
        try:
            payloads.append(json.loads(fp.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


# ---------------------------------------------------------------------------
# Stage aggregators
# ---------------------------------------------------------------------------


def _sum_taskgen_payload(
    payload: dict[str, Any],
) -> tuple[float, int, float | None, dict[str, Any]]:
    """Return `(wall, tokens, usd_or_None, fetch_meta)` for one payload."""
    wall = float(payload.get("wall_time_seconds_total") or 0.0)
    tokens = 0
    usd_val = 0.0
    usd_known = False
    for bucket in ("fetch", "paraphrase"):
        block = (payload.get("llm_calls") or {}).get(bucket) or {}
        tokens += int(block.get("total_tokens") or 0)
        usd = block.get("estimated_usd")
        if usd is not None:
            usd_val += float(usd)
            usd_known = True
    emb = payload.get("embeddings") or {}
    emb_usd = emb.get("estimated_usd")
    if emb_usd is not None:
        usd_val += float(emb_usd)
        usd_known = True
    fs = payload.get("fetch_summary") or {}
    fetch_meta = {
        "attempts": len(fs.get("fabricated_dropped_per_attempt") or []),
        "short_fetched": int(fs.get("num_tasks_accepted") or 0)
        < int(fs.get("num_tasks_requested") or 0),
        "tasks": payload.get("tasks") or [],
    }
    return wall, tokens, (usd_val if usd_known else None), fetch_meta


def _sum_augment_payload(
    payload: dict[str, Any],
) -> tuple[float, int, float | None, int, int, int]:
    """Return `(wall, tokens, usd_or_None, attempted, placed, dropped,
    reasoning_tokens, reasoning_usd_or_None)`."""
    wall = float(payload.get("wall_time_seconds") or 0.0)
    block = payload.get("llm_calls") or {}
    tokens = int(block.get("total_tokens") or 0)
    usd = block.get("estimated_usd")
    attempted = int(payload.get("n_tasks_attempted") or 0)
    placed = int(payload.get("n_tasks_placed") or 0)
    dropped = int(payload.get("n_tasks_dropped") or 0)
    reasoning = int(block.get("reasoning_tokens") or 0)
    reasoning_usd = block.get("estimated_reasoning_usd")
    return (
        wall,
        tokens,
        (None if usd is None else float(usd)),
        attempted,
        placed,
        dropped,
        reasoning,
        (None if reasoning_usd is None else float(reasoning_usd)),
    )


def _sum_evaluate_payload(
    payload: dict[str, Any],
) -> tuple[float, int, float | None, int, float, int, float]:
    """Return `(wall, judge_tokens, judge_usd_or_None, sem_calls, sem_wall,
    uri_calls, uri_wall)`."""
    wall = float(payload.get("wall_time_seconds") or 0.0)
    judge = payload.get("judge_calls") or {}
    j_tokens = int(judge.get("total_tokens") or 0)
    j_usd = judge.get("estimated_usd")
    sem = payload.get("semantic_oracle") or {}
    uri = payload.get("uri_validation") or {}
    return (
        wall,
        j_tokens,
        (None if j_usd is None else float(j_usd)),
        int(sem.get("n_calls") or 0),
        float(sem.get("wall_time_seconds") or 0.0),
        int(uri.get("n_calls") or 0),
        float(uri.get("wall_time_seconds") or 0.0),
    )


def _empty_stage(stage: str) -> dict[str, Any]:
    base = {
        "wall_time_seconds_total": 0.0,
        "wall_time_seconds_avg_per_person": 0.0,
        "wall_time_seconds_avg_per_persona": 0.0,
        "tokens_total": 0,
        "tokens_avg_per_person": 0,
        "estimated_usd_total": None,
    }
    if stage == "augmentation":
        base.update(
            {
                "n_tasks_placed": 0,
                "n_tasks_dropped": 0,
                "n_tasks_attempted": 0,
                "reasoning_tokens_total": 0,
                "estimated_reasoning_usd_total": None,
            }
        )
    elif stage == "evaluation":
        base.update(
            {
                "embedding_lookups_total": 0,
                "embedding_wall_time_seconds_total": 0.0,
                "uri_validations_total": 0,
                "uri_validation_wall_time_seconds_total": 0.0,
            }
        )
    return base


def _per_person_avg(total: float, n_persons: int) -> float:
    return round(total / n_persons, 4) if n_persons else 0.0


def _per_persona_avg(total: float, n_personas: int) -> float:
    return round(total / n_personas, 4) if n_personas else 0.0


def _stage_aggregate(payloads: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    """Aggregate a list of per-person payloads for one stage."""
    if not payloads:
        return _empty_stage(stage)
    n = len(payloads)
    n_personas = len({p.get("persona_id") for p in payloads if p.get("persona_id")})
    wall_total = 0.0
    tokens_total = 0
    usd_total = 0.0
    usd_known = False
    extra: dict[str, float | int] = {}
    for p in payloads:
        if stage == "task_generation":
            w, t, u, _meta = _sum_taskgen_payload(p)
        elif stage == "augmentation":
            (
                w,
                t,
                u,
                attempted,
                placed,
                dropped,
                reasoning,
                reasoning_usd,
            ) = _sum_augment_payload(p)
            extra["n_tasks_attempted"] = (
                int(extra.get("n_tasks_attempted") or 0) + attempted
            )
            extra["n_tasks_placed"] = int(extra.get("n_tasks_placed") or 0) + placed
            extra["n_tasks_dropped"] = int(extra.get("n_tasks_dropped") or 0) + dropped
            extra["reasoning_tokens_total"] = (
                int(extra.get("reasoning_tokens_total") or 0) + reasoning
            )
            if reasoning_usd is not None:
                extra["estimated_reasoning_usd_total"] = round(
                    float(extra.get("estimated_reasoning_usd_total") or 0.0)
                    + reasoning_usd,
                    6,
                )
        else:  # evaluation
            w, t, u, sem_n, sem_w, uri_n, uri_w = _sum_evaluate_payload(p)
            extra["embedding_lookups_total"] = (
                int(extra.get("embedding_lookups_total") or 0) + sem_n
            )
            extra["embedding_wall_time_seconds_total"] = round(
                float(extra.get("embedding_wall_time_seconds_total") or 0.0) + sem_w,
                4,
            )
            extra["uri_validations_total"] = (
                int(extra.get("uri_validations_total") or 0) + uri_n
            )
            extra["uri_validation_wall_time_seconds_total"] = round(
                float(extra.get("uri_validation_wall_time_seconds_total") or 0.0)
                + uri_w,
                4,
            )
        wall_total += w
        tokens_total += t
        if u is not None:
            usd_total += u
            usd_known = True

    if stage == "augmentation":
        extra.setdefault("estimated_reasoning_usd_total", None)

    out: dict[str, Any] = {
        "wall_time_seconds_total": round(wall_total, 4),
        "wall_time_seconds_avg_per_person": _per_person_avg(wall_total, n),
        "wall_time_seconds_avg_per_persona": _per_persona_avg(wall_total, n_personas),
        "tokens_total": tokens_total,
        "tokens_avg_per_person": tokens_total // n if n else 0,
        "estimated_usd_total": round(usd_total, 6) if usd_known else None,
    }
    out.update(extra)
    return out


def aggregate_taskgen_sidecars(tasks_dir: Path) -> list[dict[str, Any]]:
    """Read every per-person sidecar under `<tasks_dir>/_telemetry/`."""
    return _read_sidecar_payloads(tasks_dir)


def aggregate_augment_sidecars(augmented_dir: Path) -> list[dict[str, Any]]:
    """Read every per-person sidecar under `<augmented_dir>/_telemetry/`."""
    return _read_sidecar_payloads(augmented_dir)


def aggregate_evaluate_sidecars(evaluation_dir: Path) -> list[dict[str, Any]]:
    """Read every per-person sidecar under `<evaluation_dir>/_telemetry/`."""
    return _read_sidecar_payloads(evaluation_dir)


# ---------------------------------------------------------------------------
# By-persona breakdown
# ---------------------------------------------------------------------------


def _by_persona_breakdown(
    taskgen: list[dict[str, Any]],
    augment: list[dict[str, Any]],
    evaluate: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group every stage's per-person spend by `persona_id`.

    Returns a nested dict keyed by persona_id; each entry carries cohort
    totals (wall-time, tokens, cost) plus a `n_persons` count and the
    within-cohort per-person averages.
    """
    by_persona: dict[str, dict[str, Any]] = {}
    by_persona_persons: dict[str, set[str]] = {}

    def _bump(
        persona: str, person: str, *, wall: float, tokens: int, usd: float | None
    ) -> None:
        if not persona:
            return
        bucket = by_persona.setdefault(
            persona,
            {
                "n_persons": 0,
                "wall_time_seconds_total": 0.0,
                "tokens_total": 0,
                "estimated_usd_total": 0.0,
                "_usd_known": False,
            },
        )
        bucket["wall_time_seconds_total"] = round(
            float(bucket["wall_time_seconds_total"]) + wall, 4
        )
        bucket["tokens_total"] = int(bucket["tokens_total"]) + tokens
        if usd is not None:
            bucket["estimated_usd_total"] = round(
                float(bucket["estimated_usd_total"]) + usd, 6
            )
            bucket["_usd_known"] = True
        seen = by_persona_persons.setdefault(persona, set())
        if person and person not in seen:
            seen.add(person)
            bucket["n_persons"] = len(seen)

    for p in taskgen:
        w, t, u, _meta = _sum_taskgen_payload(p)
        _bump(
            p.get("persona_id") or "", p.get("person_id") or "", wall=w, tokens=t, usd=u
        )
    for p in augment:
        w, t, u, *_ = _sum_augment_payload(p)
        _bump(
            p.get("persona_id") or "", p.get("person_id") or "", wall=w, tokens=t, usd=u
        )
    for p in evaluate:
        w, t, u, *_ = _sum_evaluate_payload(p)
        _bump(
            p.get("persona_id") or "", p.get("person_id") or "", wall=w, tokens=t, usd=u
        )

    for persona, bucket in by_persona.items():
        n = int(bucket["n_persons"]) or 1
        bucket["wall_time_seconds_avg_per_person"] = round(
            float(bucket["wall_time_seconds_total"]) / n, 4
        )
        bucket["tokens_avg_per_person"] = int(bucket["tokens_total"]) // n
        if bucket.pop("_usd_known"):
            bucket["estimated_usd_avg_per_person"] = round(
                float(bucket["estimated_usd_total"]) / n, 6
            )
        else:
            bucket["estimated_usd_total"] = None
            bucket["estimated_usd_avg_per_person"] = None
    return by_persona


# ---------------------------------------------------------------------------
# Top-level full-pipeline aggregator
# ---------------------------------------------------------------------------


def aggregate_full_pipeline(
    *,
    tasks_dir: Path,
    augmented_dir: Path | None,
    evaluation_dir: Path | None,
    scenario_id: str,
    method: str,
) -> dict[str, Any]:
    """Roll up task-gen + augment + evaluate sidecars into one summary.

    Returns the dict the report writer renders into `telemetry.txt` /
    `telemetry.json`.  Per-stage aggregates live under
    `by_stage.{task_generation, augmentation, evaluation}` and
    cohort-level aggregates under `by_persona.{persona_id: {...}}`.

    Top-level fields collapse the three stages into headline totals
    (wall-time / tokens / USD) plus paraphrase-gate counters carried
    over from the task-generation sidecars (the only stage that has
    them).  When a stage has no sidecars on disk, its block is the
    empty-stage skeleton so downstream consumers always see the same
    keys.
    """
    taskgen = aggregate_taskgen_sidecars(Path(tasks_dir)) if tasks_dir else []
    augment = aggregate_augment_sidecars(Path(augmented_dir)) if augmented_dir else []
    evaluate = (
        aggregate_evaluate_sidecars(Path(evaluation_dir)) if evaluation_dir else []
    )

    n_persons = len(
        {
            p.get("person_id")
            for p in (taskgen + augment + evaluate)
            if p.get("person_id")
        }
    )
    n_personas = len(
        {
            p.get("persona_id")
            for p in (taskgen + augment + evaluate)
            if p.get("persona_id")
        }
    )

    stages = {
        "task_generation": _stage_aggregate(taskgen, "task_generation"),
        "augmentation": _stage_aggregate(augment, "augmentation"),
        "evaluation": _stage_aggregate(evaluate, "evaluation"),
    }

    wall_total = sum(float(stages[s]["wall_time_seconds_total"]) for s in stages)
    tokens_total = sum(int(stages[s]["tokens_total"]) for s in stages)
    usd_known = any(stages[s]["estimated_usd_total"] is not None for s in stages)
    usd_total = sum(
        float(stages[s]["estimated_usd_total"] or 0.0)
        for s in stages
        if stages[s]["estimated_usd_total"] is not None
    )

    # Per-task token average uses the count of generated tasks recorded
    # in the task-generation sidecars (the single source of truth for
    # "how many tasks are we evaluating against").
    total_tasks = sum(len(p.get("tasks") or []) for p in taskgen)

    # Paraphrase gates + fetch-attempt stats live exclusively in the
    # task-generation sidecars; carry them at the top level so the
    # report renderer keeps the existing block shape.
    fetch_attempts: list[int] = []
    persons_short_fetched = 0
    accepted_paraphrases = 0
    paraphrase_total_tasks = 0
    gate_failures = {"length": 0, "emoji": 0, "similarity": 0}
    for p in taskgen:
        _w, _t, _u, meta = _sum_taskgen_payload(p)
        if meta["attempts"]:
            fetch_attempts.append(int(meta["attempts"]))
        if meta["short_fetched"]:
            persons_short_fetched += 1
        for task in meta["tasks"]:
            paraphrase_total_tasks += 1
            paraphrase = (
                (task.get("paraphrase") or {}) if isinstance(task, dict) else {}
            )
            if paraphrase.get("used") == "paraphrase":
                accepted_paraphrases += 1
            failed = paraphrase.get("failed_gate")
            if failed in gate_failures:
                gate_failures[failed] += 1

    return {
        "scenario_id": scenario_id,
        "method": method,
        "n_persons": n_persons,
        "n_personas": n_personas,
        "wall_time_seconds_total": round(wall_total, 4),
        "wall_time_seconds_avg_per_person": _per_person_avg(wall_total, n_persons),
        "wall_time_seconds_avg_per_persona": _per_persona_avg(wall_total, n_personas),
        "tokens_total": tokens_total,
        "tokens_avg_per_person": tokens_total // n_persons if n_persons else 0,
        "tokens_avg_per_persona": tokens_total // n_personas if n_personas else 0,
        "tokens_avg_per_task": tokens_total // total_tasks if total_tasks else 0,
        "estimated_usd_total": round(usd_total, 6) if usd_known else None,
        "estimated_usd_avg_per_person": (
            round(usd_total / n_persons, 6) if (usd_known and n_persons) else None
        ),
        "estimated_usd_avg_per_persona": (
            round(usd_total / n_personas, 6) if (usd_known and n_personas) else None
        ),
        "by_stage": stages,
        "by_persona": _by_persona_breakdown(taskgen, augment, evaluate),
        "fetch_attempts_avg": (
            round(sum(fetch_attempts) / len(fetch_attempts), 3)
            if fetch_attempts
            else 0.0
        ),
        "fetch_attempts_max": max(fetch_attempts) if fetch_attempts else 0,
        "persons_short_fetched": persons_short_fetched,
        "paraphrase_acceptance_rate": (
            round(accepted_paraphrases / paraphrase_total_tasks, 4)
            if paraphrase_total_tasks
            else 0.0
        ),
        "gate_failures": gate_failures,
    }


def write_summary_sidecar(summary: dict[str, Any], stage_dir: Path) -> Path:
    """Persist *summary* to `<stage_dir>/_telemetry/_summary.json`."""
    out_dir = Path(stage_dir) / "_telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _SUMMARY_NAME
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


# ---------------------------------------------------------------------------
# Convenience timer
# ---------------------------------------------------------------------------


class _Timer:
    """`with _Timer() as t: ...` then read `t.elapsed` (seconds).

    Trivial helper to keep the call-site code readable when wrapping
    LLM / embedder calls.
    """

    elapsed: float

    def __enter__(self) -> "_Timer":
        self._t0 = time.monotonic()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D401
        self.elapsed = time.monotonic() - self._t0
