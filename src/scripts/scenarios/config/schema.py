"""Pydantic v2 models for scenario YAML configuration files.

Two formats are supported:

**Legacy (single scenario)** ; top-level `scenario:` key::

    scenario:
      id: my_scenario
      augmentation:
        method: greedy
      output:
        dir: ./output/my_scenario

**Multi-scenario (new)** ; top-level `scenarios:` list::

    experiment_id: example_experiment
    run_dir: ./output/example_experiment

    scenarios:
      - id: nutrition_l1
        augmentation:
          - method: greedy
            loss: {lambda_cov: 0.30, ...}
          - method: llm_agent
            loss: {lambda_cov: 0.25, ...}
      - id: senior_mobility
        ...

Output directories are auto-derived under a tree-style layout that
keeps task-generation and per-method augmenter artifacts in separate
sibling subtrees of the experiment root:

* tasks       to `{output_base}/{experiment_id}/task_generation/{scenario_id}/`
* augmenter   to `{output_base}/{experiment_id}/scenarios/{scenario_id}/{method}/`
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# calendar source
# ---------------------------------------------------------------------------


class CalendarSourceConfig(_Frozen):
    """Points to an already-generated persona-pipeline run directory.

    `run_dir` is the primary handle; the four persona YAML paths are
    optional and used when `run_dir` is absent (triggering a fresh
    persona run) or when the loader cannot find `used_configs.json`.
    """

    run_dir: str | None = None
    persona_environment: str | None = None
    persona_config: str | None = None
    persona_events: str | None = None
    persona_rules: str | None = None


# ---------------------------------------------------------------------------
# per-stage LLM model selection
# ---------------------------------------------------------------------------


class LLMModelConfig(_Frozen):
    """Per-stage LLM provider/model override.

    A scenario YAML can plant one of these blocks at the scenario level
    (`task_generator_model:`, `evaluator_model:`) to A/B different models
    for the SAME persona/scenario without touching `.env`. Useful for
    benchmark sweeps such as "compare gpt-4o-mini vs gpt-4.1-mini for
    LLM-judge σ scoring on identical inputs".

    Fields:

    * `provider`; `openai` / `anthropic` / `openrouter`. When omitted
      the env-default `LLMSettings.from_env().provider` wins, so existing
      runs that only specify `model:` keep using whichever provider
      `.env` is configured for.
    * `model`; provider-specific model id (e.g. `gpt-4o-mini`,
      `claude-sonnet-4-6`). Required.
    * `max_retries`; generic retry budget consumed by callers that
      honour it (LLM-agent oneshot caller; future judge / fetch loops).
      Defaults to 1.
    """

    provider: Literal["anthropic", "openai", "openrouter"] | None = None
    model: str = Field(min_length=1)
    max_retries: int = Field(default=1, ge=1)


# ---------------------------------------------------------------------------
# task generation
# ---------------------------------------------------------------------------


class TaskFilters(_Frozen):
    """Sub-class and difficulty filters applied to the ontology query."""

    domains: list[str] = Field(default_factory=list)
    difficulty: list[str] = Field(default_factory=list)


class TaskFilterGroup(_Frozen):
    """One paired domain and difficulty group for task generation.

    A task matches the group when its branch is in `domains` and its level
    is in `difficulty`. Multiple groups combine by union, so a task is kept
    when it matches any group.
    """

    domains: list[str] = Field(default_factory=list)
    difficulty: list[str] = Field(default_factory=list)


class TaskGenerationConfig(_Frozen):
    """Controls how desired-behavior tasks are generated for each person.

    `num_tasks` is the count generated for the week this config covers.
    `week` is the 1-based horizon week it applies to, or `None` to cover
    every week no other entry claims. `filters` pairs domains with
    difficulties: either a single `{domains, difficulty}` block or a list
    of `{domains, difficulty}` groups combined by union. When
    `cross_week_distinct` is True the week's tasks stay disjoint from the
    tasks generated for earlier weeks.
    """

    method: Literal["graphrag", "graphrag_grounded", "manual", "template"] = "graphrag"
    seed: int = Field(default=20260503, ge=0)
    num_tasks: int = Field(default=5, ge=1)
    week: int | None = Field(default=None, ge=1)
    cross_week_distinct: bool = True
    ontologies: list[str] = Field(
        default_factory=lambda: ["HealthTasks", "HumanActivities"]
    )
    prompt_template: str = "health_improvement"
    filters: TaskFilters | list[TaskFilterGroup] = Field(default_factory=TaskFilters)
    manual_tasks: list[dict[str, Any]] = Field(default_factory=list)
    task_overrides: dict[str, Any] = Field(default_factory=dict)
    # Characteristic axes exposed to the prompt templates as the person
    # profile.  `None` exposes every axis the person carries; a list
    # restricts (and orders) the profile to the named axes.  Axes a
    # person does not carry are skipped silently.
    profile_characteristics: list[str] | None = None

    # ── Grounded-path knobs (read only when method == "graphrag_grounded") ──
    #
    # max_fetch_retries
    #     Hard cap on how many times Stage 1 re-issues the GraphRAG
    #     `search` call.  Each retry passes the already-accepted URIs
    #     as a do-not-repeat blacklist.  ≤ this number of LLM calls per
    #     persona for fetch.
    # paraphrase
    #     When True, run Stage 2 (LLM paraphrase) + Stage 3 (three
    #     gates) on the verified Stage-1 list.  When False, ship the
    #     canonical descriptions verbatim; cheapest grounded baseline.
    # paraphrase_similarity_threshold
    #     Gate C minimum cosine.  A paraphrase scoring below this falls
    #     back to the canonical description.
    # paraphrase_max_length_delta_pct
    #     Gate A maximum |Δlen| / len ratio (0.10 to ±10 %).  A
    #     paraphrase outside that band falls back to canonical.
    max_fetch_retries: int = Field(default=5, ge=1)
    paraphrase: bool = True
    paraphrase_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    paraphrase_max_length_delta_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    # GraphRAG retriever `top_k` for the Stage-1 fetch.  Bumping this
    # widens the LLM's candidate pool; too small a value (the
    # neo4j_graphrag default of 5) caused a regression
    # where 30 personas converged on the same 4 URIs.
    fetch_top_k: int = Field(default=20, ge=1)

    def filter_groups(self) -> list[TaskFilterGroup]:
        """Return the filters as a list of groups, normalizing the legacy form."""
        if isinstance(self.filters, list):
            return list(self.filters)
        return [
            TaskFilterGroup(
                domains=self.filters.domains, difficulty=self.filters.difficulty
            )
        ]


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------


class GreedyConfig(_Frozen):
    """Settings for the deterministic gap-filling greedy augmenter."""

    strategy: Literal["earliest_fit", "latest_fit", "preference_first"] = (
        "preference_first"
    )
    retry_on_miss: bool = True
    max_backtrack: int = Field(default=3, ge=0)


class AblationVariantSpec(_Frozen):
    """One variant for the `custom` ablation design.

    `id` becomes the on-disk directory name. `ablate` lists the block
    names from `augment_oneshot.ABLATABLE_BLOCKS` to drop.
    """

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    ablate: list[str] = Field(default_factory=list)


class PromptAblationConfig(_Frozen):
    """Per-method prompt-component ablation declaration.

    Lives under `augmentation[*].llm_agent.prompt_ablation`. With
    `design: single` (or when absent) the augmenter runs once. Any
    other design expands the `(scenario, method)` pair into one
    augment + evaluate pass per variant.

    Fields:
        design: one of `ablation_designs.DESIGN_NAMES`.
        fold: forwarded to `plackett_burman_12` to promote it to
            PB-24. Ignored elsewhere.
        placebos: per-block replacement text. Keys must be valid
            block names.
        use_default_placebos: when `true`, blocks not named in
            `placebos` fall back to the default catalog. When
            `false`, those blocks are deleted instead.
        variants: required when `design == "custom"`.
        seed: reserved for future replicate handling.
    """

    design: Literal[
        "single",
        "leave_one_out",
        "plackett_burman_12",
        "plackett_burman_16",
        "plackett_burman_24",
        "full_factorial",
        "custom",
    ] = "single"
    fold: bool = True
    use_default_placebos: bool = True
    placebos: dict[str, str] = Field(default_factory=dict)
    variants: list[AblationVariantSpec] = Field(default_factory=list)
    seed: int = Field(default=20260518, ge=0)

    @model_validator(mode="after")
    def _check_variants_match_design(self) -> PromptAblationConfig:
        if self.design == "custom":
            if not self.variants:
                raise ValueError(
                    "design 'custom' requires at least one entry in `variants`"
                )
            ids = [v.id for v in self.variants]
            if len(ids) != len(set(ids)):
                raise ValueError("custom variant ids must be unique")
        else:
            if self.variants:
                raise ValueError(
                    f"design {self.design!r} must not declare `variants`; "
                    f"use design: custom for user-supplied variants"
                )
        return self

    @model_validator(mode="after")
    def _check_placebo_keys_known(self) -> PromptAblationConfig:
        # Late import keeps the schema module free of a Jinja2 dep at
        # import time.
        from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
            ABLATABLE_BLOCKS,
        )

        unknown = sorted(set(self.placebos) - set(ABLATABLE_BLOCKS))
        if unknown:
            raise ValueError(
                f"placebos reference unknown block(s) {unknown}; "
                f"valid blocks: {ABLATABLE_BLOCKS}"
            )
        for v in self.variants:
            unknown_v = sorted(set(v.ablate) - set(ABLATABLE_BLOCKS))
            if unknown_v:
                raise ValueError(
                    f"custom variant {v.id!r} references unknown block(s) "
                    f"{unknown_v}; valid blocks: {ABLATABLE_BLOCKS}"
                )
        return self


class LLMAgentConfig(_Frozen):
    """Settings for the one-shot LLM-agent augmenter.

    The LLM agent receives all recommended tasks and the full week
    calendar in a single prompt and returns a complete placement plan
    for that week as a JSON array, one entry per task.

    `max_retries` controls how many times the whole one-shot call is
    retried on malformed JSON. `prompt_template` names the template
    in `prompt_templates` used to build the one-shot prompt.
    `prompt_ablation` is optional; see `PromptAblationConfig`.
    """

    provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_retries: int = Field(default=1, ge=1)
    prompt_template: str = "augment_oneshot"
    prompt_ablation: PromptAblationConfig | None = None


class RLConfig(_Frozen):
    """Settings for the RL-based augmenter.

    Fields:
        policy: `dqn`, `ppo`, or `random`.
        checkpoint: optional file for warm-start and end-of-run save.
        train_steps_per_week: gradient steps to run after each week's episode.
        learning_rate: optimizer learning rate.
        gamma: reward discount factor.
        buffer_size: replay buffer capacity.
        batch_size: minibatch size for each gradient step.
        epsilon_start: exploration rate at week 1.
        epsilon_end: exploration rate after the decay window.
        epsilon_decay_weeks: weeks taken to linearly anneal epsilon.
        seed: RNG seed for the env, agent, and exploration noise.
        parallel_per_worker_gb: host RAM budgeted per parallel RL worker.
        parallel_max_retries: retry attempts per person that fails in the pool.
        parallel_worker_threads: torch CPU threads per parallel RL worker.
        train_steps: legacy pre-online budget; kept for back-compat.
        eval_episodes: legacy eval episode count; kept for back-compat.
    """

    policy: Literal["ppo", "dqn", "random"] = "dqn"
    checkpoint: str | None = None
    train_steps_per_week: int = Field(default=4000, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    gamma: float = Field(default=0.95, ge=0.0, le=1.0)
    buffer_size: int = Field(default=50_000, ge=100)
    batch_size: int = Field(default=64, ge=1)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay_weeks: int = Field(default=4, ge=1)
    seed: int = Field(default=0, ge=0)
    # Width (in minutes) of each time-bucket cell in the env's action
    # space. Smaller = finer scheduling resolution but bigger Q-network
    # output. 15 keeps historical behavior; 60 cuts action space 4x to
    # make Q-learning tractable with limited training budgets.
    bucket_minutes: int = Field(default=15, ge=1, le=1440)
    # Torch device for the DQN network. `cpu` is recommended when the
    # GPU is already loaded by the semantic embedder or persona pipeline;
    # the small Q-network trains fast enough on CPU.
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # Per-step potential-based shaping reward magnitude. Each valid
    # placement adds `per_step_shaping / n_tasks` to env.step's
    # reward; a fully-placed episode sums to `per_step_shaping` from
    # shaping alone. Helps DQN credit-assign across 30-60 step
    # episodes (where gamma^N decay otherwise drowns the terminal
    # weekly-gain signal). Default 0 keeps historical behavior.
    per_step_shaping: float = Field(default=0.0, ge=0.0, le=10.0)
    # When True, the env's `AutoTaskActionWrapper` hides the task_idx
    # dimension from the agent and always places the next-unscheduled
    # task. Shrinks the action space by a factor of `n_tasks`, giving
    # the Q-network ~20x more effective samples per output. Recommended
    # whenever task ordering is irrelevant to the gain metric.
    auto_pick_task: bool = False
    # When True, pin the DQN exploration rate per week on a linear
    # `epsilon_start` to `epsilon_end` curriculum over `epsilon_decay_weeks`,
    # overriding SB3's per-call anneal. Default keeps SB3's schedule.
    cross_week_epsilon: bool = False
    # When > 0, give up on a task after this many failed placement attempts so
    # the episode advances instead of burning `max_steps` on one unplaceable
    # task. 0 disables the guard.
    max_task_attempts: int = Field(default=0, ge=0)
    # Host RAM budgeted per RL worker when `augment --executor process` fans the
    # per-person DQN runs out. The pool caps workers at usable_RAM / this value.
    parallel_per_worker_gb: float = Field(default=1.5, gt=0.0)
    # Retry attempts per person that fails in the parallel pool, run after the
    # round finishes. 0 disables retries (one attempt per person).
    parallel_max_retries: int = Field(default=1, ge=0)
    # Torch CPU threads each parallel RL worker uses; 1 stops N workers from
    # oversubscribing the cores.
    parallel_worker_threads: int = Field(default=1, ge=1)
    train_steps: int = Field(default=50000, ge=1)
    eval_episodes: int = Field(default=100, ge=1)


_PTIME_CRITERIA: tuple[str, ...] = (
    "time",
    "duration",
    "overlap",
    "stability",
)


class PTimeLearningConfig(_Frozen):
    """SVM-rank refinement of PTIME importance weights.

    When `enabled` is True, an `sklearn.svm.LinearSVC` is fitted per
    person on pairwise `(preferred, dispreferred)` slot examples drawn
    from the calendar. The fitted `B` vector blends with the elicited
    `A` as `alpha * A + (1 - alpha) * B`. `alpha=1.0` keeps elicited
    only; `alpha=0.0` uses learned only.

    Fields:
        enabled: toggle for the SVM-rank pass.
        alpha: blend factor in `[0, 1]`.
        negative_samples_per_event: shifted alternatives drawn per event.
        negative_shift_minutes: maximum shift used to build negatives.
        regularization: sklearn `LinearSVC` `C`.
        seed: RNG seed for negative-sample shifts.
    """

    enabled: bool = False
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    negative_samples_per_event: int = Field(default=2, ge=1)
    negative_shift_minutes: int = Field(default=120, ge=10)
    regularization: float = Field(default=1.0, gt=0.0)
    seed: int = Field(default=0, ge=0)


class PTimeConfig(_Frozen):
    """Settings for the PTIME Choquet-based augmenter.

    PTIME scores candidate placements with a 2-order Choquet integral
    over four scheduling-time criteria: `time` (fit inside
    `environment.time_windows`), `duration` (fit inside the task's
    `[duration_min, duration_max]`), `overlap` (Allen-rule
    admissibility against base events), and `stability` (constant 1.0
    since base events are immutable).

    Fields:
        candidate_strategy: how many anchors to score per base gap.
            `earliest_only` keeps the gap start, `windows` adds one
            anchor per `time_windows` band that fits, `windows_and_center`
            also adds the gap center.
        duration_preference: how `utility_duration` rewards slot length
            inside `[duration_min, duration_max]` (`minimum` / `maximum`
            / `midpoint`).
        importance: Choquet `a_i` in `[0, 1]`; missing keys default to
            `1/n` so the elicited model is uninformative by default.
        interaction: Choquet `a_ij` in `[-1, 1]`; keys are alphabetically
            sorted pairs like `"duration|time"`.
        solver: `mcs` for branch-and-bound, `greedy` for top-1 decode.
        mcs_time_budget_seconds: per-week MCS wall-time cap.
        learning: optional SVM-rank refinement of `importance`.
    """

    candidate_strategy: Literal["earliest_only", "windows", "windows_and_center"] = (
        "windows"
    )
    duration_preference: Literal["minimum", "maximum", "midpoint"] = "minimum"
    importance: dict[str, float] = Field(default_factory=dict)
    interaction: dict[str, float] = Field(default_factory=dict)
    solver: Literal["mcs", "greedy"] = "mcs"
    mcs_time_budget_seconds: float = Field(default=10.0, gt=0.0)
    learning: PTimeLearningConfig = Field(default_factory=PTimeLearningConfig)

    @model_validator(mode="after")
    def _check_keys(self) -> PTimeConfig:
        for k, v in self.importance.items():
            if k not in _PTIME_CRITERIA:
                raise ValueError(
                    f"ptime.importance has unknown criterion {k!r}; "
                    f"known: {list(_PTIME_CRITERIA)}"
                )
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"ptime.importance[{k!r}]={v!r} must be in [0, 1]")
        allowed_pairs = {
            "|".join(sorted((a, b)))
            for i, a in enumerate(_PTIME_CRITERIA)
            for b in _PTIME_CRITERIA[i + 1 :]
        }
        for k, v in self.interaction.items():
            if k not in allowed_pairs:
                raise ValueError(
                    f"ptime.interaction has unknown pair {k!r}; "
                    f"keys must be 'a|b' with a,b in {list(_PTIME_CRITERIA)} "
                    f"and a < b alphabetically"
                )
            if not -1.0 <= float(v) <= 1.0:
                raise ValueError(f"ptime.interaction[{k!r}]={v!r} must be in [-1, 1]")
        return self


# CalendarEvent fields the augmenter can opt into via `observation.host_flags`.
# Base `{type, label, start, end}` always renders without opt-in.
_KNOWN_HOST_FLAGS: frozenset[str] = frozenset(
    {
        "is_concurrent",
        "is_dividable",
        "concurrent_with",
        "intensity",
    }
)

# Recommended-task fields renderable in the augmenter's tasks-list block.
# `context_recommends` is the intersection of the task's
# `context_links` with `observation.contexts`; rendered only when both
# the task has matching recommendations and the flag is enabled.
_KNOWN_TASK_FLAGS: frozenset[str] = frozenset(
    {
        "concurrent_ok",
        "dividable_ok",
        "duration_min",
        "duration_max",
        "intensity",
        "context_recommends",
    }
)

_DEFAULT_TASK_FLAGS: tuple[str, ...] = (
    "concurrent_ok",
    "dividable_ok",
    "duration_min",
    "duration_max",
    "intensity",
    "context_recommends",
)


def _load_context_categories() -> frozenset[str]:
    """Return the Context category slugs from Neo4j, lazy-cached per process.

    Tests can pre-populate `_load_context_categories._cache` to bypass the
    Neo4j round-trip.
    """
    cached = getattr(_load_context_categories, "_cache", None)
    if cached is not None:
        return cached

    from src.graphrag.config import Neo4jSettings
    from src.graphrag.neo4j_client import make_driver, session_scope
    from src.scripts.scenarios.task_generation.ontology_bridge import (
        fetch_context_categories,
    )

    settings = Neo4jSettings.from_env()
    driver = make_driver(settings)
    try:
        with session_scope(driver, settings.database) as session:
            cats = fetch_context_categories(session)
    finally:
        driver.close()
    _load_context_categories._cache = cats  # type: ignore[attr-defined]
    return cats


class ObservationConfig(_Frozen):
    """Per-augmenter info-leakage gate applied at augment time."""

    contexts: list[str] = Field(default_factory=list)
    context_detail: Literal["summary", "full"] = "summary"
    host_flags: list[str] = Field(default_factory=list)
    task_flags: list[str] = Field(default_factory=lambda: list(_DEFAULT_TASK_FLAGS))

    @model_validator(mode="after")
    def _check_known_blocks(self) -> ObservationConfig:
        """Reject unknown context categories or flag names."""
        unknown_task = [f for f in self.task_flags if f not in _KNOWN_TASK_FLAGS]
        if unknown_task:
            raise ValueError(
                f"observation.task_flags has unknown fields {unknown_task!r}; "
                f"known: {sorted(_KNOWN_TASK_FLAGS)}"
            )
        unknown_host = [f for f in self.host_flags if f not in _KNOWN_HOST_FLAGS]
        if unknown_host:
            raise ValueError(
                f"observation.host_flags has unknown fields {unknown_host!r}; "
                f"known: {sorted(_KNOWN_HOST_FLAGS)}"
            )
        if self.contexts:
            known_cats = _load_context_categories()
            if known_cats:
                unknown_ctx = [c for c in self.contexts if c not in known_cats]
                if unknown_ctx:
                    raise ValueError(
                        f"observation.contexts has unknown categories "
                        f"{unknown_ctx!r}; known: {sorted(known_cats)}"
                    )
        return self


class AugmentationConfig(_Frozen):
    """Selects the augmentation method and its settings.

    `repeat_per_week` controls the planning horizon for all augmenters.
    When `True` (default) every augmenter receives a **fresh copy** of the
    task list at the start of each calendar week, so `num_tasks` tasks are
    targeted every week.  This gives all solvers the same decision horizon and
    makes comparisons fair.  When `False` unplaced tasks carry forward to
    the next week (one-shot scheduling).

    `max_iterations` in `llm_agent` is the per-task LLM call budget when
    `repeat_per_week=True` (each task gets up to `max_iterations` attempts
    per week) or the total budget shared across all tasks when `False`.
    """

    method: Literal["greedy", "llm_agent", "rl", "ptime", "human_coach", "gamebus_coach"] = "greedy"
    allow_merge: bool = True
    merge_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    repeat_per_week: bool = True
    greedy: GreedyConfig = Field(default_factory=GreedyConfig)
    llm_agent: LLMAgentConfig = Field(default_factory=LLMAgentConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    ptime: PTimeConfig = Field(default_factory=PTimeConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)


# ---------------------------------------------------------------------------
# scheduling-loss weights
# ---------------------------------------------------------------------------


class LossWeights(_Frozen):
    """Seven component weights for the scheduling loss `L(S)`.

    Components: `L_cov` (unscheduled fraction), `L_cal` (rule + buffer
    violations), `L_pref` (preferred-start deviation), `L_disp` (median
    MET / intensity closeness with half-life lag), `L_merge`
    (concurrency opportunity cost), `L_spread` (date-spread penalty),
    `L_divide` (reward for splitting dividable tasks).

    The weights must sum to `1.0` (within `1e-6`).

    `admissible_relation_overrides` and `semantic_matrix` are legacy
    fields kept for backwards compatibility.
    """

    lambda_cov: float = Field(default=0.18, ge=0.0, le=1.0)
    lambda_cal: float = Field(default=0.18, ge=0.0, le=1.0)
    lambda_pref: float = Field(default=0.13, ge=0.0, le=1.0)
    lambda_disp: float = Field(default=0.18, ge=0.0, le=1.0)
    lambda_merge: float = Field(default=0.13, ge=0.0, le=1.0)
    lambda_spread: float = Field(default=0.10, ge=0.0, le=1.0)
    lambda_divide: float = Field(default=0.10, ge=0.0, le=1.0)
    # Context-fit leg. Default 0.0 keeps the historical seven-component
    # sum-to-one invariant for scenarios that predate context-awareness.
    lambda_context_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    admissible_relation_overrides: dict[str, list[str]] = Field(default_factory=dict)
    semantic_matrix: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sum_to_one(self) -> LossWeights:
        total = (
            self.lambda_cov
            + self.lambda_cal
            + self.lambda_pref
            + self.lambda_disp
            + self.lambda_merge
            + self.lambda_spread
            + self.lambda_divide
            + self.lambda_context_fit
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"loss weights must sum to 1.0 (within 1e-6); got {total:.8f}"
            )
        return self


# ---------------------------------------------------------------------------
# evaluation tunables
# ---------------------------------------------------------------------------


class EvaluationConfig(_Frozen):
    """Numeric tunables for the loss components at evaluate time.

    Lives under `scenarios.yaml` `evaluation:`. Fields:

    * `buffer_minutes`: global default for `L_cal`'s buffer-aware
      partial credit on `p` / `P` violations. Per-rule `buffer:`
      overrides win.
    * `merge_threshold`: minimum σ for a `(task, event)` pair to enter
      `L_merge`'s Φ.
    * `disp_half_life_days`: half-life `H` of the exponential time-lag
      in `L_disp`: `lag(δ) = (1/2)^(δ / H)`.
    * `divide_duration_tolerance_pct`: multiplicative band around the
      original duration `D` for `L_divide`. A bucket with `n` dividable
      instances in a week is `divided_valid` when `sum(pieces)` is
      inside `[n * D * (1 - tau), n * D * (1 + tau)]`.
    """

    buffer_minutes: int = Field(default=30, ge=0)
    merge_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    disp_half_life_days: float = Field(default=2.0, gt=0.0)
    divide_duration_tolerance_pct: float = Field(default=0.15, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


class ScenarioOutputConfig(_Frozen):
    """Controls what the pipeline writes to disk."""

    dir: str
    write_ics: bool = True
    write_json: bool = True
    write_report: bool = True


# ---------------------------------------------------------------------------
# top-level; legacy single-scenario format
# ---------------------------------------------------------------------------


class ScenarioConfig(_Frozen):
    """Inner content of a legacy scenario YAML (under the `scenario:` key).

    `task_generator_model` / `evaluator_model` (both optional) override
    the env-default LLM for that stage only. Leaving them `None` keeps
    the historical behavior of resolving the model from `.env`.
    """

    id: str = Field(min_length=1)
    description: str = ""
    calendar: CalendarSourceConfig = Field(default_factory=CalendarSourceConfig)
    task_generation: TaskGenerationConfig | list[TaskGenerationConfig] = Field(
        default_factory=TaskGenerationConfig
    )
    task_generator_model: LLMModelConfig | None = None
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    loss: LossWeights = Field(default_factory=LossWeights)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    evaluator_model: LLMModelConfig | None = None
    output: ScenarioOutputConfig
    timeframe: "TimeframeSpec | None" = None

    def weekly_task_generation(self) -> list[TaskGenerationConfig]:
        """Return the task-generation configs as a list, wrapping a single block."""
        if isinstance(self.task_generation, list):
            return list(self.task_generation)
        return [self.task_generation]


# ---------------------------------------------------------------------------
# Multi-scenario format
# ---------------------------------------------------------------------------


class AugmentationMethodConfig(_Frozen):
    """One augmentation strategy within a scenario, with its own loss and output.

    `output` is optional; when absent the CLI auto-derives the directory as
    `{output_base}/{experiment_id}/{scenario_id}/{method}/`.
    """

    method: Literal["greedy", "llm_agent", "rl", "ptime", "human_coach", "gamebus_coach"] = "greedy"
    allow_merge: bool = True
    merge_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    repeat_per_week: bool = True
    greedy: GreedyConfig = Field(default_factory=GreedyConfig)
    llm_agent: LLMAgentConfig = Field(default_factory=LLMAgentConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    ptime: PTimeConfig = Field(default_factory=PTimeConfig)
    loss: LossWeights = Field(default_factory=LossWeights)
    output: ScenarioOutputConfig | None = None
    observation: ObservationConfig = Field(default_factory=ObservationConfig)


class TimeframeSpec(_Frozen):
    """Optional per-scenario sub-range of the persona horizon.

    Two forms, picked by `scale`:
    * `scale: week` -- `start` and `end` are 1-based week indices, inclusive.
    * `scale: dates` -- `start` and `end` are ISO dates, inclusive. The
      window length (end - start + 1) must be a multiple of 7 days.

    Bounds against the actual horizon (start_date, weeks count) are
    checked at runtime by `resolve_timeframe`, not at schema load.
    """

    scale: Literal["week", "dates"]
    start: int | _dt.date
    end: int | _dt.date

    @model_validator(mode="after")
    def _check(self) -> TimeframeSpec:
        if self.scale == "week":
            if not isinstance(self.start, int) or isinstance(self.start, bool):
                raise ValueError("timeframe.start must be an integer when scale=week")
            if not isinstance(self.end, int) or isinstance(self.end, bool):
                raise ValueError("timeframe.end must be an integer when scale=week")
            if self.start < 1:
                raise ValueError("timeframe.start must be >= 1 when scale=week")
            if self.end < self.start:
                raise ValueError("timeframe.end must be >= timeframe.start")
        else:
            if not isinstance(self.start, _dt.date) or isinstance(
                self.start, _dt.datetime
            ):
                raise ValueError("timeframe.start must be an ISO date when scale=dates")
            if not isinstance(self.end, _dt.date) or isinstance(self.end, _dt.datetime):
                raise ValueError("timeframe.end must be an ISO date when scale=dates")
            if self.end < self.start:
                raise ValueError("timeframe.end must be >= timeframe.start")
            span_days = (self.end - self.start).days + 1
            if span_days % 7 != 0:
                raise ValueError(
                    f"timeframe span must be a multiple of 7 days; got {span_days}"
                )
        return self


class ScenarioDefinition(_Frozen):
    """One scenario within an :class:`ExperimentScenariosConfig`.

    `augmentation` is a list so multiple solvers can be defined in one file
    and evaluated against the same task set.

    `evaluation` carries the numeric evaluation tunables (buffer_minutes,
    merge_threshold, disp_half_life_days); each scenario can override the
    defaults independently.

    `timeframe` optionally pins the scenario to a sub-range of the
    persona horizon. Omit for full-horizon (default).

    `label` is an optional human-readable display name for charts and
    the benchmark report; when empty the report falls back to its
    auto-derived compact label and the chart legend to the scenario id.
    """

    id: str = Field(min_length=1)
    description: str = ""
    label: str = ""
    timeframe: TimeframeSpec | None = None
    task_generation: TaskGenerationConfig | list[TaskGenerationConfig] = Field(
        default_factory=TaskGenerationConfig
    )
    task_generator_model: LLMModelConfig | None = None
    augmentation: list[AugmentationMethodConfig] = Field(
        default_factory=lambda: [AugmentationMethodConfig()]
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    evaluator_model: LLMModelConfig | None = None

    def weekly_task_generation(self) -> list[TaskGenerationConfig]:
        """Return the task-generation configs as a list, wrapping a single block."""
        if isinstance(self.task_generation, list):
            return list(self.task_generation)
        return [self.task_generation]

    @model_validator(mode="after")
    def _check_task_generation(self) -> ScenarioDefinition:
        if not isinstance(self.task_generation, list):
            if self.task_generation.week is not None:
                raise ValueError(
                    "a single task_generation block must not set `week`; use a "
                    "list to tie configs to weeks"
                )
            return self
        weeks = [entry.week for entry in self.task_generation]
        if sum(1 for w in weeks if w is None) > 1:
            raise ValueError(
                "at most one task_generation entry may omit `week` (the default)"
            )
        explicit = [w for w in weeks if w is not None]
        if len(explicit) != len(set(explicit)):
            raise ValueError("task_generation entries must not repeat a `week`")
        for method_cfg in self.augmentation:
            if not method_cfg.repeat_per_week:
                raise ValueError(
                    "a per-week task_generation list requires repeat_per_week=True on "
                    f"every augmentation method; {method_cfg.method!r} has it False"
                )
        return self


class ExperimentScenariosConfig(_Frozen):
    """Multi-scenario config co-located with the persona-pipeline YAML files.

    Fields:
        experiment_id: identifier that matches the persona-pipeline run.
        run_dir: path to the persona-pipeline output directory.
            When empty the CLI defaults to `{output_base}/{experiment_id}`.
        output_base: root directory for all augmentation and evaluation output.
        scenarios: list of scenario definitions, each with ≥ 1 augmentation method.
    """

    experiment_id: str = Field(min_length=1)
    run_dir: str = ""
    output_base: str = "./output"
    scenarios: list[ScenarioDefinition] = Field(min_length=1)

    def effective_run_dir(self) -> str:
        """Return the resolved persona-pipeline output directory."""
        return self.run_dir or f"{self.output_base}/{self.experiment_id}"

    def task_generation_dir(self, scenario_id: str) -> str:
        """Per-scenario task-generation output, sibling of `scenarios/`.

        Layout:
            {output_base}/{experiment_id}/task_generation/{scenario_id}/
                ├── _telemetry/<person_id>.json
                ├── <person_id>_tasks.json
                └── summary_report.txt   (when written)

        Tasks are scenario-scoped (different filters / seeds / models)
        but stage-grouped, so all task-generation artifacts for an
        experiment live under a single `task_generation/` umbrella next
        to the persona `persons/` and the augmenter `scenarios/` dirs.
        """
        return f"{self.output_base}/{self.experiment_id}/task_generation/{scenario_id}"

    def scenario_method_dir(self, scenario_id: str, method: str) -> str:
        """Per-scenario × per-method augment + evaluate output dir.

        Layout:
            {output_base}/{experiment_id}/scenarios/{scenario_id}/{method}/
                ├── augmented/persons/<person_id>.json
                ├── augmented/_telemetry/...
                ├── augmented/charts/<person_id>/week_NN.png
                ├── ics_per_person/<person_id>.ics
                └── evaluation/<reports + telemetry>

        Grouping every per-method run under `scenarios/` makes the
        experiment dir self-documenting: persona artifacts at the root,
        task-gen under `task_generation/`, augmenter outputs under
        `scenarios/`.
        """
        return (
            f"{self.output_base}/{self.experiment_id}/scenarios/"
            f"{scenario_id}/{method}"
        )
