"""Eight-component scheduling loss `L(S)` in `[0, 1]`.

`L(S) = (Σ_{i∈A} λ_i L_i) / (Σ_{i∈A} λ_i)` with scheduling gain
`G(S) = 1 − L(S)`, where the applicable set `A = {i : λ_i > 0 and
Ω_i ≠ ∅}` holds the positive-weight components whose own set `Ω_i`
carries signal. Each `L_i ∈ [0, 1]` is higher-is-worse and is weighted
by the matching field on `LossWeights`. A placed task is the realized
eventuality `ê_k = (η_k, τ_s, τ_e)`; `S = {ê_k}` is the set of
placements, `Y_-` the unplaced tasks, and `ρ(e, e')` the idle gap
between two eventualities.

* `L_cov`: `|Y_-| / |Y|`, the fraction of recommended tasks left
  unscheduled.
* `L_cal`: `|Ω|^-1 Σ v(e, e')` over same-day `(task × event) ∪
  (task × task)` pairs; `v` is the Allen-rule violation, `0` when the
  relation is admissible, `1 − ρ / β` on a near miss inside buffer `β`,
  and `1` otherwise.
* `L_pref`: `|Ω|^-1 Σ δ_j`, the mean normalized deviation `δ_j` from one
  persona preference (per-occurrence or per-scale duration, episode
  count, temporal pattern, or persona-stage semantic alignment).
* `L_disp`: `median[ max((ω_e + ω_e') / 2ω_max, (μ_e + μ_e') / 2μ_max)
  × 2^(−ρ / h) ]` over `{(e, e') : e ∈ S, e' ∈ S ∪ X}`; intensity `ω`
  and physical demand `μ` of two demanding eventualities close in time.
* `L_merge`: `1 − Σ σ_act(ê_k) / Σ σ_best(ê_k)`, the concurrency
  opportunity cost of the chosen host versus the best mergeable one.
* `L_spread`: `1 − |π_D(S)| / min(|S|, |T_D|)`, the day-spread penalty
  over the day scale `D`.
* `L_divide`: `1 − |Ω|^-1 Σ_w |Y^val_w| / |Y^div_w|` over weeks holding a
  dividable task, with `Y^val_w` the validly split ones.
* `L_context_fit`: `1 − |Ω|^-1 Σ |Z^obs_k ∩ Z^rec_k| / |Z^rec_k|`, the
  share of `ê_k`'s recommended context categories realized during it.
"""

from __future__ import annotations

import datetime
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # avoid circular import at runtime
    from src.scripts.persona.domain.time_windows import WindowMap
    from src.scripts.scenarios.metrics.preference_constraints import PersonaConstraints
    from src.scripts.scenarios.metrics.preference_score import PatternViolation

from src.scripts.persona.config.schema import WindowRange
from src.scripts.scenarios.config.schema import LossWeights
from src.scripts.scenarios.domain.calendar import CalendarEvent, CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import RecommendedTask, ScheduledTask
from src.scripts.scenarios.metrics.allen import (
    R_ALL,
    R_BUFFER,
    R_MERGE,
    R_SEP,
    AllenRelation,
    RuleSet,
    SelectorMatcher,
    build_admissible_pair,
    compute_allen_relation,
    compute_buffer_violation,
)
from src.scripts.scenarios.metrics.intensity_resolver import IntensityResolver
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

# ---------------------------------------------------------------------------
# LossComponents
# ---------------------------------------------------------------------------


_COMPONENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("cov", "lambda_cov"),
    ("cal", "lambda_cal"),
    ("pref", "lambda_pref"),
    ("disp", "lambda_disp"),
    ("merge", "lambda_merge"),
    ("spread", "lambda_spread"),
    ("divide", "lambda_divide"),
    ("context_fit", "lambda_context_fit"),
)


@dataclass(frozen=True)
class LossComponents:
    """Per-component breakdown of the scheduling loss.

    Each field is a float in [0, 1] when the component produced a
    signal, or `None` when it produced none (no dividable tasks, an
    empty merge opportunity set, no preference legs that fired).

    `weighted` scores against an input-applicability mask: only legs in
    the mask contribute, and an in-mask leg whose value is `None` is
    charged the maximum loss `1.0` so a do-nothing augmenter cannot
    collect free credit for a dimension that applied. See
    `compute_applicability_mask` for how the mask is built from the
    persona inputs.

    `divide_verdicts` carries the per-bucket telemetry produced
    alongside the `divide` scalar. Empty tuple when no dividable
    signal exists.
    """

    cov: float
    cal: float
    pref: float | None
    disp: float
    merge: float | None
    spread: float = 0.0
    divide: float | None = None
    divide_verdicts: tuple["DivideVerdictRecord", ...] = ()
    context_fit: float | None = None

    def weighted(
        self,
        w: LossWeights,
        mask: "frozenset[str] | set[str] | None" = None,
        *,
        zero_placement: bool = False,
    ) -> float:
        """Return the masked weighted loss in `[0, 1]`.

        Only legs in `mask` contribute; an in-mask leg with a `None`
        value is charged full loss `1.0` (the augmenter produced no
        signal for a dimension that applied). When `mask` is `None`
        every positive-weight leg is in scope. `zero_placement` charges
        every in-mask leg full loss (the augmenter placed nothing while
        tasks applied), overriding vacuous-perfect `cal`/`disp`/`spread`.
        Returns `0.0` when no leg is in scope.
        """
        num = 0.0
        den = 0.0
        for attr, lam_attr in _COMPONENT_FIELDS:
            lam = float(getattr(w, lam_attr, 0.0))
            if mask is None:
                # No mask: the bare arithmetic primitive scores only legs
                # that produced a signal (a no-signal leg is dropped, not
                # charged). The reporting path never takes this branch; it
                # always passes a per-person mask, which is where the
                # applicable-but-unaddressed charging lives.
                if lam <= 0.0:
                    continue
                value = getattr(self, attr)
                if value is None:
                    continue
                loss_i = float(value)
            else:
                if attr not in mask:
                    continue
                if zero_placement:
                    loss_i = 1.0
                else:
                    value = getattr(self, attr)
                    loss_i = 1.0 if value is None else float(value)
            num += lam * loss_i
            den += lam
        if den == 0.0:
            return 0.0
        return num / den

    def gain(
        self,
        w: LossWeights,
        mask: "frozenset[str] | set[str] | None" = None,
        *,
        zero_placement: bool = False,
    ) -> float:
        """Return the masked scheduling gain `G(S) = 1 - L(S)` in `[0, 1]`."""
        return 1.0 - self.weighted(w, mask, zero_placement=zero_placement)

    def masked_view(
        self,
        mask: "frozenset[str] | set[str]",
        *,
        zero_placement: bool = False,
    ) -> "LossComponents":
        """Project leg values to their report-display form under `mask`.

        Out-of-mask legs become `None` (rendered `n/a`, excluded from
        per-component averages); in-mask legs keep their value, or become
        `1.0` (gain 0) when they produced no signal or placement was
        empty. The returned object is for reporting only; `weighted`
        stays the source of truth for the scalar loss.
        """
        fields: dict[str, float | None] = {}
        for attr, _ in _COMPONENT_FIELDS:
            if attr not in mask:
                fields[attr] = None
            elif zero_placement:
                fields[attr] = 1.0
            else:
                value = getattr(self, attr)
                fields[attr] = 1.0 if value is None else float(value)
        return LossComponents(
            cov=fields["cov"],
            cal=fields["cal"],
            pref=fields["pref"],
            disp=fields["disp"],
            merge=fields["merge"],
            spread=fields["spread"] if fields["spread"] is not None else 0.0,
            divide=fields["divide"],
            divide_verdicts=self.divide_verdicts,
            context_fit=fields["context_fit"],
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_by_date(
    items: Iterable,
) -> dict[datetime.date, list]:
    result: dict[datetime.date, list] = defaultdict(list)
    for item in items:
        result[item.date].append(item)
    return dict(result)


def _temporal_gap(ti: ScheduledTask, tj: ScheduledTask) -> float:
    """Gap in minutes between two realized events (0 when they overlap).

    Events on different days are flattened onto a shared timeline: the
    earlier-date event's times are shifted by `days_diff × 1440`.
    """
    day_diff = (ti.date - tj.date).days
    si = ti.start_minutes + day_diff * 1440
    ei = ti.end_minutes + day_diff * 1440
    sj = tj.start_minutes
    ej = tj.end_minutes
    return float(max(0, max(si - ej, sj - ei)))


def _temporal_gap_event_task(task: ScheduledTask, event: CalendarEvent) -> float:
    """Same as :func:`_temporal_gap` but for one task and one event."""
    day_diff = (task.date - event.date).days
    si = task.start_minutes + day_diff * 1440
    ei = task.end_minutes + day_diff * 1440
    sj = event.start_minutes
    ej = event.end_minutes
    return float(max(0, max(si - ej, sj - ei)))


def _temporal_gap_task_context(task: ScheduledTask, episode) -> float:  # noqa: ANN001
    """Gap between one scheduled task and one context episode (same shape as `_temporal_gap`)."""
    day_diff = (task.date - episode.date).days
    si = task.start_minutes + day_diff * 1440
    ei = task.end_minutes + day_diff * 1440
    sj = episode.start_minutes
    ej = episode.end_minutes
    return float(max(0, max(si - ej, sj - ei)))


def _rules_reference_context(rules) -> bool:  # noqa: ANN001
    """True iff any rule names `kind: context` on either endpoint."""
    for rule in rules:
        a = getattr(rule, "event_a", None)
        b = getattr(rule, "event_b", None)
        if (a is not None and getattr(a, "kind", None) == "context") or (
            b is not None and getattr(b, "kind", None) == "context"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# L_cov
# ---------------------------------------------------------------------------


def compute_l_cov(solution: SchedulingSolution) -> float:
    """L_cov = |Y_-| / |Y|, the unplaced fraction.  Zero when |Y| = 0."""
    n = len(solution.tasks)
    if n == 0:
        return 0.0
    return len(solution.unscheduled) / n


# ---------------------------------------------------------------------------
# L_pref
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegStats:
    """Per-leg roll-up for the `L_pref` sidecar / reporter.

    Fields:
        name: leg identifier (`per_occurrence_duration` /
            `per_scale_duration` / `per_scale_episodes` /
            `persona_stage_semantic` / `temporal_pattern_*`).
        mean_loss: mean of non-`None` violations contributing to this
            leg, or `None` when the leg never fired.
        applicable_count: number of (task × placement) signals the leg
            evaluated; distinct from "zero violations" so the report
            can show "leg fired ten times, all aligned".
        mape: uncapped mean-absolute-percentage-error, only set on
            temporal-pattern legs that report it.
    """

    name: str
    mean_loss: float | None
    applicable_count: int
    mape: float | None = None


def _leg_stats_from_buckets(
    name: str, values: list[float], mape_values: list[float] | None = None
) -> LegStats:
    if not values:
        return LegStats(name=name, mean_loss=None, applicable_count=0, mape=None)
    mean = sum(values) / len(values)
    mape = None
    if mape_values:
        mape = sum(mape_values) / len(mape_values)
    return LegStats(name=name, mean_loss=mean, applicable_count=len(values), mape=mape)


def compute_l_pref_v2(
    solution: SchedulingSolution,
    *,
    persona_constraints: "PersonaConstraints",
    window_map: "WindowMap",
    semantic: SemanticCompatibility | None = None,
    stage_alignment_threshold: float = 0.65,
) -> tuple[float | None, list["PatternViolation"], list[LegStats]]:
    """`L_pref = |Ω|^-1 Σ_j δ_j`, the mean normalized preference gap.

    Aggregator: walks each recommended task, computes up-to-five per-leg
    deviations `δ_j ∈ [0, 1]` against the curated YAML preferences, and
    returns the mean over every signal-carrying leg `j ∈ Ω`. `None` is
    returned when no leg has signal; same convention as
    :func:`compute_l_divide`.

    Legs (each `δ_j ∈ [0, 1]`, weighted internally over matched events):

    1. Per-occurrence duration band; *per scheduled placement*.
    2. Per-scale duration band; *per recommended task* (sum across that
       task's placements).
    3. Per-scale episode count; *per recommended task*.
    4. Persona-stage semantic alignment; *per scheduled placement*.
    5. Temporal-pattern alignment; *per recommended task* (fix / seasonality
       / trend collapsed into the leg's overall violation).

    Returns `(aggregate, pattern_rows, leg_stats)` where `leg_stats`
    is the per-leg breakdown consumed by the sidecar writer + the
    report's `preference_breakdown` block.  The legacy two-tuple
    callers still work because the third element is additive (Python
    tuples grow gracefully via index access).
    """
    # Late imports to avoid a hard dependency on the constraint-driven
    # scoring modules for the legacy `compute_l_pref` path.
    from src.scripts.scenarios.metrics.preference_score import (
        score_per_occurrence_duration,
        score_per_scale_duration,
        score_per_scale_episodes,
        score_persona_stage_semantic,
        score_temporal_pattern,
    )

    if not solution.tasks:
        return None, [], []

    pattern_rows: list = []
    legs: list[float] = []
    per_occ_values: list[float] = []
    per_scale_dur_values: list[float] = []
    per_scale_ep_values: list[float] = []
    stage_values: list[float] = []
    pattern_by_mode: dict[str, list[float]] = defaultdict(list)
    pattern_mape_by_mode: dict[str, list[float]] = defaultdict(list)

    # Group scheduled placements by recommended-task label so the
    # per-scale / temporal-pattern legs see all occurrences of one
    # task family at once (matters for `is_dividable` tasks).
    placements_by_label: dict[str, list[ScheduledTask]] = defaultdict(list)
    for placement in solution.scheduled:
        placements_by_label[placement.task.label].append(placement)

    for recommended_task in solution.tasks:
        matched = persona_constraints.matched_events_for(recommended_task)
        placements = placements_by_label.get(recommended_task.label, [])

        # ------------------------------------------------------------------
        # Per-placement legs (per-occurrence duration, persona-stage).
        # ------------------------------------------------------------------
        for placement in placements:
            firing_stages = persona_constraints.stages_firing_on(placement.date)
            per_occ = score_per_occurrence_duration(
                placement, matched_events=matched, constraints=persona_constraints
            )
            if per_occ is not None:
                legs.append(per_occ)
                per_occ_values.append(per_occ)
            stage_leg = score_persona_stage_semantic(
                placement,
                firing_stages=firing_stages,
                semantic=semantic,
                stage_alignment_threshold=stage_alignment_threshold,
            )
            if stage_leg is not None:
                legs.append(stage_leg)
                stage_values.append(stage_leg)

        # ------------------------------------------------------------------
        # Per-task aggregate legs (per-scale duration, episodes, temporal).
        # ------------------------------------------------------------------
        if not placements:
            continue
        scale_duration = score_per_scale_duration(
            placements, matched_events=matched, constraints=persona_constraints
        )
        if scale_duration is not None:
            legs.append(scale_duration)
            per_scale_dur_values.append(scale_duration)
        scale_episodes = score_per_scale_episodes(
            placements, matched_events=matched, constraints=persona_constraints
        )
        if scale_episodes is not None:
            legs.append(scale_episodes)
            per_scale_ep_values.append(scale_episodes)
        temporal, rows = score_temporal_pattern(
            placements,
            matched_events=matched,
            constraints=persona_constraints,
            window_map=window_map,
        )
        if temporal is not None:
            legs.append(temporal)
        pattern_rows.extend(rows)
        for row in rows:
            pattern_by_mode[f"temporal_pattern_{row.mode}"].append(row.loss)
            if row.mape is not None:
                pattern_mape_by_mode[f"temporal_pattern_{row.mode}"].append(row.mape)

    leg_stats: list[LegStats] = [
        _leg_stats_from_buckets("per_occurrence_duration", per_occ_values),
        _leg_stats_from_buckets("per_scale_duration", per_scale_dur_values),
        _leg_stats_from_buckets("per_scale_episodes", per_scale_ep_values),
        _leg_stats_from_buckets("persona_stage_semantic", stage_values),
    ]
    # Temporal-pattern rows split by mode; stable order for the report.
    for mode in ("fix", "seasonality", "trend"):
        name = f"temporal_pattern_{mode}"
        leg_stats.append(
            _leg_stats_from_buckets(
                name,
                pattern_by_mode.get(name, []),
                pattern_mape_by_mode.get(name, []),
            )
        )

    if not legs:
        return None, pattern_rows, leg_stats
    return sum(legs) / len(legs), pattern_rows, leg_stats


# ---------------------------------------------------------------------------
# L_spread
# ---------------------------------------------------------------------------


def compute_l_spread(
    solution: SchedulingSolution,
    horizon_days: int,
) -> float:
    """L_spread = 1 − |π_D(S)| / min(|S|, |T_D|), the day-spread penalty."""
    if not solution.scheduled or horizon_days <= 0:
        return 0.0
    unique_dates = {st.date for st in solution.scheduled}
    denom = min(len(solution.scheduled), horizon_days)
    return 1.0 - len(unique_dates) / denom


# ---------------------------------------------------------------------------
# L_cal; merged (task×event) ∪ (task×task) with buffer-aware partial credit
# ---------------------------------------------------------------------------


def _eligible_pairs(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
    *,
    include_contexts: bool = False,
) -> list[tuple[Any, Any]]:
    """Return the eligible-pair set restricted to same-day pairs.

    Always includes `(task x event) U (task x task)`. When
    `include_contexts` is True, also adds `(task x context)` pairs for
    callers whose Allen rules reference `kind: context` selectors.
    `event x event` and `event x context` pairs are excluded.
    """
    pairs: list[tuple[Any, Any]] = []
    events_by_date = _group_by_date(calendar.events)
    tasks_by_date = _group_by_date(solution.scheduled)
    contexts_by_date: dict = {}
    if include_contexts:
        for ep in calendar.contexts:
            contexts_by_date.setdefault(ep.date, []).append(ep)
    for date, day_tasks in tasks_by_date.items():
        for task in day_tasks:
            for ev in events_by_date.get(date, []):
                pairs.append((task, ev))
        for i, ti in enumerate(day_tasks):
            for tj in day_tasks[i + 1 :]:
                pairs.append((ti, tj))
        if include_contexts:
            for task in day_tasks:
                for ep in contexts_by_date.get(date, []):
                    pairs.append((task, ep))
    return pairs


def compute_l_cal(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
    ruleset: RuleSet,
    matcher: SelectorMatcher,
    *,
    buffer_minutes: int = 30,
) -> float:
    """`L_cal = |Ω|^-1 Σ_(e,e') v(e, e')` over same-day pairs.

    `Ω` is the merged `(task × event) ∪ (task × task)` same-day pair
    set. For each pair `build_admissible_pair` gives the admissible
    Allen relation set `R_ee'` (the relations every applicable rule
    allows), and `compute_buffer_violation` returns `v`: `0` when
    `rel(e, e') ∈ R_ee'`, `1 − ρ / β` on a near miss with gap `ρ < β`,
    and `1` otherwise. Returns `0.0` for an empty pair set.
    """
    include_contexts = _rules_reference_context(ruleset.rules)
    pairs = _eligible_pairs(solution, calendar, include_contexts=include_contexts)
    if not pairs:
        return 0.0
    total = 0.0
    for a, b in pairs:
        admissible = build_admissible_pair(a, b, ruleset, matcher)
        rel = compute_allen_relation(
            a.start_minutes, a.end_minutes, b.start_minutes, b.end_minutes
        )
        if isinstance(b, CalendarEvent):
            gap = _temporal_gap_event_task(a, b)
        elif isinstance(b, ContextEpisode):
            gap = _temporal_gap_task_context(a, b)
        else:
            gap = _temporal_gap(a, b)
        rule_buffer = ruleset.buffer_for(a, b, matcher, default=buffer_minutes)
        total += compute_buffer_violation(rel, gap, rule_buffer, admissible)
    return total / len(pairs)


# ---------------------------------------------------------------------------
# L_disp; median half-life lag × max(categorical, continuous)
# ---------------------------------------------------------------------------


_CATEGORICAL_DENOM = 8.0  # 2 × max_intensity = 2 × 4 = 8


def _half_life_lag(gap_minutes: float, half_life_days: float) -> float:
    """Return `(1/2)^(δ_days / H)` ∈ (0, 1]; never zero."""
    if half_life_days <= 0:
        return 0.0
    delta_days = gap_minutes / 1440.0
    return float(2.0 ** (-delta_days / float(half_life_days)))


def _disp_pairs(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
) -> list[tuple[Any, Any]]:
    """Pair set for `L_disp`: `task × event ∪ task × task` across the
    whole horizon.

    Includes cross-day pairs (the half-life lag dampens distant
    pairs). `event × event` pairs are excluded.
    """
    pairs: list[tuple[Any, Any]] = []
    scheduled = list(solution.scheduled)
    events = list(calendar.events)
    for task in scheduled:
        for ev in events:
            pairs.append((task, ev))
    for i, ti in enumerate(scheduled):
        for tj in scheduled[i + 1 :]:
            pairs.append((ti, tj))
    return pairs


def compute_l_disp(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
    resolver: IntensityResolver,
    *,
    half_life_days: float,
    max_met: float,
) -> float:
    """`L_disp = median_(e,e') [ max(categorical, continuous) × 2^(−ρ/h) ]`.

    * `categorical = (ω_e + ω_e') / 2ω_max` with `ω ∈ {1..4}` the
      cognitive-or-MET-quartile intensity; `ω_max = 4`, so the
      denominator is `8`.
    * `continuous  = (μ_e + μ_e') / 2μ_max` with `μ` the MET value.
    * `per_pair    = max(categorical, continuous) × 2^(−ρ/h)`, the
      gap `ρ` discounted by half-life `h`.
    * Aggregate by median over `Ω = {(e, e') : e ∈ S, e' ∈ S ∪ X}`
      across the whole horizon.

    Returns `0.0` when the pair set is empty. `max_met = 0` disables
    the continuous leg.
    """
    pairs = _disp_pairs(solution, calendar)
    if not pairs:
        return 0.0
    cont_denom = 2.0 * float(max_met) if max_met > 0 else None
    per_pair_scores: list[float] = []
    for a, b in pairs:
        rec_a = resolver.record(a)
        rec_b = resolver.record(b)
        cat = (rec_a.intensity + rec_b.intensity) / _CATEGORICAL_DENOM
        if cont_denom and rec_a.met is not None and rec_b.met is not None:
            cont = (rec_a.met + rec_b.met) / cont_denom
        else:
            cont = 0.0
        if isinstance(b, CalendarEvent):
            gap = _temporal_gap_event_task(a, b)
        else:
            gap = _temporal_gap(a, b)
        lag = _half_life_lag(gap, half_life_days)
        per_pair = max(cat, cont) * lag
        per_pair_scores.append(min(1.0, max(0.0, per_pair)))
    # `per_pair_scores` is guaranteed non-empty here; the empty
    # case is handled earlier by the `if not pairs` guard.
    return float(statistics.median(per_pair_scores))


# ---------------------------------------------------------------------------
# L_merge; inclusion ∧ ¬exclusion concurrency opportunity
# ---------------------------------------------------------------------------


def _build_event_concurrent_lookup(
    event_intensity_map: dict[str, int] | None,
    calendar: CalendarTrace,
) -> dict[str, bool]:
    """Build `{event_label: is_concurrent_field}` from the calendar trace.

    Uses each `CalendarEvent.is_concurrent` field.  When the same
    label appears in multiple events with conflicting values, the
    last-wins (events tend to be uniform within one calendar trace).
    """
    out: dict[str, bool] = {}
    for ev in calendar.events:
        out[ev.label] = bool(ev.is_concurrent)
    return out


def is_excluded_pair(
    task: ScheduledTask,
    event: CalendarEvent,
    *,
    event_concurrent_map: dict[str, bool],
    ruleset: RuleSet,
    matcher: SelectorMatcher,
) -> bool:
    """Return True when `(task, event)` must not be considered for merge.

    Three exclusion sources, each independent (logical OR):

    * `event_config[event].is_concurrent == False`: catalog forbids
      overlap on the event side. Falls back to the realized event's
      `is_concurrent` attribute when the label is unmapped.
    * `RecommendedTask.is_concurrent == False`: the task carries the
      HealthTasks `hb:isConcurrent=false` flag.
    * Selector Allen rule with admissible set
      `⊂ R_SEP ∪ R_BUFFER`: forbids every overlap variant.
    """
    # (i) base-event side: prefer the catalog flag, fall back to the
    #     realized event's attribute when the label is unmapped.
    event_concurrent = event_concurrent_map.get(event.label, event.is_concurrent)
    if event_concurrent is False:
        return True
    # (ii) task side: the HealthTasks ontology said no.
    if not task.task.is_concurrent:
        return True
    # (iii) Allen rule that admits only separation/buffer variants.
    admissible = build_admissible_pair(task, event, ruleset, matcher)
    if admissible and admissible.issubset(R_SEP | R_BUFFER):
        return True
    return False


def _bridged_score(
    bridge: Any | None,
    task: RecommendedTask,
    host: CalendarEvent,
    semantic: SemanticCompatibility,
) -> float:
    """Return 1.0 when the bridge confirms a shared family, else `semantic.score`.

    The bridge looks up the host's catalog `label` (e.g. office_work), while the
    judge scores the host's `oracle_label` (e.g. "standup (office_work)") so the
    semantic oracle sees the actual activity alongside its parent.
    """
    if bridge is not None and task.ontology_uri:
        if bridge.shares_family(task.ontology_uri, host.label):
            return 1.0
    return float(semantic.score(task.label, host.oracle_label))


def compute_l_concurrent(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
    semantic: SemanticCompatibility,
    *,
    ruleset: RuleSet,
    matcher: SelectorMatcher,
    merge_threshold: float = 0.65,
    bridge: Any | None = None,
) -> float | None:
    """`L_merge = 1 − Σ σ_act / Σ σ_best`, the concurrency opportunity cost.

    **Weekly scope.**  Augmenter decisions are weekly (one one-shot LLM
    call per week, repeated per-week placements), so the opportunity
    set Φ is scoped per ISO week; a task scheduled in week W is
    paired only with concurrent-friendly events that exist in week W.

    **Event-centered filtering.**  Concurrent-friendly events are
    pre-filtered to those with `is_concurrent=True` in the catalog
    (exclusion source (i)).  The remaining per-pair exclusions -
    task-side `RecommendedTask.is_concurrent=False` (ii) and selector
    Allen rule with admissible ⊂ `R_SEP ∪ R_BUFFER` (iii); are
    still checked.  For each scheduled task we find its **best
    achievable host** (highest σ among non-excluded events in the
    same week); if that σ ≥ `merge_threshold` the task enters Φ.
    This fixes the best-host bug where a task whose σ-best partner was
    an exclusive event got dropped from Φ entirely even though a
    second-best concurrent-friendly host was available.

    **Scoring.**  For each task `ê_k` in Φ:

      * `σ_best(ê_k)`; semantic compatibility with the best mergeable
        host.
      * `σ_act(ê_k)`; compatibility with the host the augmenter
        actually co-scheduled it with, if any, looked up by
        `(date, label)` on the concurrent-friendly event list.
        Augmenter placements against exclusive hosts contribute 0 to
        `σ_act`; an invalid merge is not credited.

    `L_merge = 1 − Σ σ_act(ê_k) / Σ σ_best(ê_k)`.

    Returns `None` when Φ is empty (no opportunity to assess) -
    mirrors :func:`compute_l_divide`'s "no signal" convention so the
    report renders `n/a` rather than laundering a stub σ oracle
    into a perfect `G_merge = 1.0`.
    """
    if not solution.scheduled:
        return None

    # Drop non-concurrent events upfront: shrinks the inner loop from
    # O(scheduled × all_events) to O(scheduled × concurrent_in_week).
    concurrent_events = [ev for ev in calendar.events if ev.is_concurrent]
    if not concurrent_events:
        return None

    events_by_week: dict[tuple[int, int], list[CalendarEvent]] = defaultdict(list)
    for ev in concurrent_events:
        events_by_week[_iso_year_week(ev.date)].append(ev)

    tasks_by_week: dict[tuple[int, int], list[ScheduledTask]] = defaultdict(list)
    for st in solution.scheduled:
        tasks_by_week[_iso_year_week(st.date)].append(st)

    event_concurrent_map = _build_event_concurrent_lookup(None, calendar)
    sigma_best_sum = 0.0
    sigma_actual_sum = 0.0

    for week, events_in_week in events_by_week.items():
        tasks_in_week = tasks_by_week.get(week, [])
        if not tasks_in_week:
            continue
        for st in tasks_in_week:
            # Best achievable host: σ-best among non-excluded events the
            # task can actually fit inside. `is_excluded_pair` covers the
            # task-side and Allen-rule exclusions; event-side is already
            # filtered above.
            best_event: CalendarEvent | None = None
            best_score = -1.0
            for ev in events_in_week:
                if is_excluded_pair(
                    st,
                    ev,
                    event_concurrent_map=event_concurrent_map,
                    ruleset=ruleset,
                    matcher=matcher,
                ):
                    continue
                # A host shorter than the task cannot contain it, so it is
                # not a placeable concurrent opportunity.
                if st.task.duration_min > ev.end_minutes - ev.start_minutes:
                    continue
                sc = _bridged_score(bridge, st.task, ev, semantic)
                if sc > best_score:
                    best_score = sc
                    best_event = ev
            if best_event is None or best_score < merge_threshold:
                continue
            sigma_best_sum += best_score
            # σ_actual: credit the augmenter only when its declared
            # `concurrent_with` resolves to a concurrent-friendly
            # event on the SAME date.  Augmenter placements against
            # exclusive hosts (event not in `events_in_week`) get
            # 0; invalid merges are not credited.
            if st.concurrent_with is not None:
                for ev in events_in_week:
                    if ev.date == st.date and ev.label == st.concurrent_with:
                        sigma_actual_sum += _bridged_score(
                            bridge, st.task, ev, semantic
                        )
                        break

    if sigma_best_sum == 0.0:
        return None
    return 1.0 - sigma_actual_sum / sigma_best_sum


# ---------------------------------------------------------------------------
# L_divide; reward for honoring isDividable
# ---------------------------------------------------------------------------


def _refers_to(scheduled: ScheduledTask, desired: RecommendedTask) -> bool:
    """Return True iff `scheduled` is one piece of `desired`."""
    if scheduled.parent_task_label is not None:
        return scheduled.parent_task_label == desired.label
    return scheduled.task.label == desired.label


def _iso_year_week(date: datetime.date) -> tuple[int, int]:
    """Return `(iso_year, iso_week)`; used to bucket scheduled tasks per week."""
    iso = date.isocalendar()
    return (iso.year, iso.week)


DIVIDE_VERDICT_VALID = "divided_valid"
DIVIDE_VERDICT_NOT_DIVIDED = "not_divided"
DIVIDE_VERDICT_PIECES_TOO_LONG = "divided_invalid_pieces_too_long"
DIVIDE_VERDICT_SUM_TOO_LOW = "divided_invalid_sum_too_low"
DIVIDE_VERDICT_SUM_TOO_HIGH = "divided_invalid_sum_too_high"

DIVIDE_VERDICTS: tuple[str, ...] = (
    DIVIDE_VERDICT_VALID,
    DIVIDE_VERDICT_NOT_DIVIDED,
    DIVIDE_VERDICT_PIECES_TOO_LONG,
    DIVIDE_VERDICT_SUM_TOO_LOW,
    DIVIDE_VERDICT_SUM_TOO_HIGH,
)


@dataclass(frozen=True, slots=True)
class DivideVerdictRecord:
    """One per-week per-label divide verdict with its inputs.

    Carries the bucket key, original duration, observed pieces,
    tolerance band, and the assigned verdict.
    """

    iso_year: int
    iso_week: int
    task_label: str
    instances_in_week: int
    original_duration_minutes: int
    piece_durations: tuple[int, ...]
    piece_dates: tuple[str, ...]
    piece_starts: tuple[int, ...]
    piece_ends: tuple[int, ...]
    sum_minutes: int
    tolerance_band: tuple[float, float]
    verdict: str


def _piece_duration(st: ScheduledTask) -> int:
    """Realised duration of one scheduled piece in minutes."""
    return max(0, int(st.end_minutes - st.start_minutes))


def _divide_verdict(
    pieces: list[ScheduledTask],
    *,
    n_instances: int,
    original_duration: int,
    tolerance_pct: float,
) -> tuple[str, tuple[float, float]]:
    """Classify one `(week, label)` bucket and return its tolerance band.

    Order: `not_divided` (`<2` pieces); `divided_invalid_pieces_too_long`
    (any piece `>= D`); `divided_invalid_sum_too_low` or
    `..._sum_too_high` (sum outside `[n*D*(1-tau), n*D*(1+tau)]`);
    otherwise `divided_valid`.
    """
    expected = float(n_instances) * float(original_duration)
    lo = expected * (1.0 - tolerance_pct)
    hi = expected * (1.0 + tolerance_pct)
    if len(pieces) < 2:
        return DIVIDE_VERDICT_NOT_DIVIDED, (lo, hi)
    if any(_piece_duration(p) >= original_duration for p in pieces):
        return DIVIDE_VERDICT_PIECES_TOO_LONG, (lo, hi)
    total = sum(_piece_duration(p) for p in pieces)
    if total < lo:
        return DIVIDE_VERDICT_SUM_TOO_LOW, (lo, hi)
    if total > hi:
        return DIVIDE_VERDICT_SUM_TOO_HIGH, (lo, hi)
    return DIVIDE_VERDICT_VALID, (lo, hi)


def compute_l_divide(
    solution: SchedulingSolution,
    *,
    tolerance_pct: float = 0.15,
) -> tuple[float | None, list[DivideVerdictRecord]]:
    """`L_div = 1 − |Ω|^-1 Σ_w |Y^val_w| / |Y^div_w|`, the mean per-week
    invalid-split fraction over weeks holding a dividable task.

    Returns `(scalar, records)`. `scalar` is `None` when no
    `(week, dividable label)` bucket carries any signal. `records` is
    one `DivideVerdictRecord` per observed bucket.

    Per week `w`, `|Y^val_w| / |Y^div_w|` is the count of `divided_valid`
    buckets over the dividable-task bucket count; the loss is `1` minus
    the mean of that fraction over the weeks in `Ω`.
    """
    if not solution.tasks:
        return None, []

    dividable_tasks: list[RecommendedTask] = [
        t for t in solution.tasks if t.is_dividable
    ]
    if not dividable_tasks:
        return None, []

    # One representative task per dividable label.
    dividable_by_label: dict[str, RecommendedTask] = {}
    for d in dividable_tasks:
        dividable_by_label.setdefault(d.label, d)

    # Bucket scheduled pieces by `(week, label)`.
    pieces_by_bucket: dict[tuple[tuple[int, int], str], list[ScheduledTask]] = (
        defaultdict(list)
    )
    weeks_with_pieces_by_label: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for st in solution.scheduled:
        for label, d in dividable_by_label.items():
            if _refers_to(st, d):
                week = _iso_year_week(st.date)
                pieces_by_bucket[(week, label)].append(st)
                weeks_with_pieces_by_label[label].add(week)
                break

    # Anchor unscheduled-only dividable labels to the earliest scheduled
    # week so they show up as `not_divided` buckets.
    if solution.unscheduled:
        scheduled_dates = sorted({st.date for st in solution.scheduled})
        if scheduled_dates:
            anchor_week = _iso_year_week(scheduled_dates[0])
            for t in solution.unscheduled:
                if t.is_dividable and t.label not in weeks_with_pieces_by_label:
                    pieces_by_bucket.setdefault((anchor_week, t.label), [])
                    weeks_with_pieces_by_label[t.label].add(anchor_week)
                    dividable_by_label.setdefault(t.label, t)

    if not pieces_by_bucket:
        return None, []

    records: list[DivideVerdictRecord] = []
    per_week_valid: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for (week, label), pieces in pieces_by_bucket.items():
        d = dividable_by_label[label]
        # Each label is recommended once per week, so a bucket is one
        # instance whose pieces should sum to about its `duration_max`.
        n = 1
        verdict, band = _divide_verdict(
            pieces,
            n_instances=n,
            original_duration=int(d.duration_max),
            tolerance_pct=tolerance_pct,
        )
        durations = tuple(_piece_duration(p) for p in pieces)
        records.append(
            DivideVerdictRecord(
                iso_year=week[0],
                iso_week=week[1],
                task_label=label,
                instances_in_week=n,
                original_duration_minutes=int(d.duration_max),
                piece_durations=durations,
                piece_dates=tuple(p.date.isoformat() for p in pieces),
                piece_starts=tuple(int(p.start_minutes) for p in pieces),
                piece_ends=tuple(int(p.end_minutes) for p in pieces),
                sum_minutes=int(sum(durations)),
                tolerance_band=(float(band[0]), float(band[1])),
                verdict=verdict,
            )
        )
        per_week_valid[week].append(verdict == DIVIDE_VERDICT_VALID)

    week_fractions = [
        sum(1 for v in verdicts if v) / len(verdicts)
        for verdicts in per_week_valid.values()
    ]
    mean_fraction = sum(week_fractions) / len(week_fractions)
    return 1.0 - mean_fraction, records


# ---------------------------------------------------------------------------
# Input-applicability mask
# ---------------------------------------------------------------------------


def _merge_opportunity_exists(
    tasks: list[RecommendedTask],
    calendar: CalendarTrace,
    semantic: SemanticCompatibility,
    *,
    merge_threshold: float,
) -> bool:
    """Return True iff some recommended task could co-schedule with an event.

    Cheap input-side check for the `merge` mask leg: at least one
    concurrent-friendly event and at least one (concurrent-capable
    recommended task, that event) pair clears `merge_threshold` under the
    embedding proxy. The reported value still comes from the judge oracle.
    """
    concurrent_events = [ev for ev in calendar.events if ev.is_concurrent]
    if not concurrent_events:
        return False
    for task in tasks:
        if not task.is_concurrent:
            continue
        for ev in concurrent_events:
            if float(semantic.score(task.label, ev.oracle_label)) >= merge_threshold:
                return True
    return False


def compute_applicability_mask(
    tasks: list[RecommendedTask],
    calendar: CalendarTrace,
    weights: LossWeights,
    *,
    persona_constraints: "PersonaConstraints | None" = None,
    semantic: SemanticCompatibility | None = None,
    context_links_by_uri: dict[str, dict[str, frozenset[str]]] | None = None,
    observed_categories: frozenset[str] | None = None,
    merge_threshold: float = 0.65,
) -> frozenset[str]:
    """Return the in-mask legs for one person from INPUTS only.

    A leg is in the mask when its scenario weight is positive and the
    inputs carry a signal the augmenter could act on. Placement is not
    consulted, so the mask is the same whether the augmenter placed
    everything or nothing (the zero-placement penalty is applied
    separately by the caller).
    """
    mask: set[str] = set()
    if not tasks:
        return frozenset(mask)

    def _has_weight(attr: str) -> bool:
        for leg_attr, lam_attr in _COMPONENT_FIELDS:
            if leg_attr == attr:
                return float(getattr(weights, lam_attr)) > 0.0
        return False  # pragma: no cover - every caller passes a declared leg

    # cov / cal / disp / spread: applicable whenever there are tasks.
    for leg in ("cov", "cal", "disp", "spread"):
        if _has_weight(leg):
            mask.add(leg)

    # pref: persona has at least one recommended task that maps to a
    # curated preference event.
    if _has_weight("pref") and persona_constraints is not None:
        for task in tasks:
            matched = persona_constraints.matched_events_for(task)
            if any(m.weight > 0.0 for m in matched):
                mask.add("pref")
                break

    # merge: a concurrent-friendly opportunity clears the sigma proxy.
    if (
        _has_weight("merge")
        and semantic is not None
        and _merge_opportunity_exists(
            tasks, calendar, semantic, merge_threshold=merge_threshold
        )
    ):
        mask.add("merge")

    # divide: at least one dividable task in the set.
    if _has_weight("divide") and any(t.is_dividable for t in tasks):
        mask.add("divide")

    # context_fit: a task must link a context member the persona actually
    # generates (then intersected with the observed categories); a linked
    # context the person never enters never enters the mask.
    if _has_weight("context_fit") and context_links_by_uri:
        trace_iris = {ep.ontology_uri for ep in calendar.contexts if ep.ontology_uri}
        for task in tasks:
            recommended = context_links_by_uri.get(task.ontology_uri or "")
            if not recommended:
                continue
            realizable = {cat for cat, iris in recommended.items() if iris & trace_iris}
            if observed_categories is not None:
                realizable &= observed_categories
            if realizable:
                mask.add("context_fit")
                break

    return frozenset(mask)


# ---------------------------------------------------------------------------
# SchedulingLoss; assembler
# ---------------------------------------------------------------------------


class SchedulingLoss:
    """Assembles and weights the eight scheduling-loss components.

    Args:
        weights: eight λ values (must sum to 1.0).
        semantic: `SemanticCompatibility` for `L_merge` scoring.
        ruleset: `RuleSet` of selector-driven Allen rules.
        matcher: `SelectorMatcher` (typically backed by an
            `IntensityResolver`).
        resolver: `IntensityResolver` for `L_disp` per-pair
            intensity / MET lookups.
        merge_threshold: minimum σ for a (task, event) pair to enter
            `L_merge`'s Φ.
        buffer_minutes: global default for `L_cal`'s p/P partial
            credit ramp.
        half_life_days: `H` for `L_disp`'s exponential lag.
        max_met: ontology-wide MET maximum (for `L_disp`'s continuous
            normalizer).  `0.0` disables the continuous leg.
    """

    def __init__(
        self,
        weights: LossWeights,
        semantic: SemanticCompatibility,
        ruleset: RuleSet,
        matcher: SelectorMatcher,
        resolver: IntensityResolver,
        *,
        merge_threshold: float = 0.65,
        buffer_minutes: int = 30,
        half_life_days: float = 2.0,
        max_met: float = 16.8,
        divide_tolerance_pct: float = 0.15,
        context_links_by_uri: dict[str, dict[str, frozenset[str]]] | None = None,
        merge_bridge: Any | None = None,
        merge_semantic: SemanticCompatibility | None = None,
    ) -> None:
        self._weights = weights
        self._semantic = semantic
        # `semantic` scores L_pref (needs `score_with_strategy`) and is the
        # default merge scorer. `merge_semantic` lets evaluate route L_merge
        # to the LLM-judge oracle while L_pref keeps the embedding oracle.
        self._merge_semantic = (
            merge_semantic if merge_semantic is not None else semantic
        )
        self._ruleset = ruleset
        self._matcher = matcher
        self._resolver = resolver
        self._merge_threshold = merge_threshold
        self._buffer_minutes = buffer_minutes
        self._half_life_days = half_life_days
        self._max_met = max_met
        self._divide_tolerance_pct = divide_tolerance_pct
        self._context_links_by_uri = context_links_by_uri or {}
        self._merge_bridge = merge_bridge

    def compute(
        self,
        solution: SchedulingSolution,
        calendar: CalendarTrace,
        horizon_epochs: int,
        time_windows: dict[str, WindowRange] | None = None,
        *,
        persona_constraints: "PersonaConstraints | None" = None,
        window_map: "WindowMap | None" = None,
        observed_categories: frozenset[str] | None = None,
        mask: "frozenset[str] | set[str] | None" = None,
    ) -> tuple[float, LossComponents]:
        """Compute L(S) and its per-component breakdown via the v2 L_pref path.

        `persona_constraints` and `window_map` are required for the L_pref
        leg; when either is None the leg produces no signal.

        When `mask` is supplied the returned loss is the masked weighted
        loss (only in-mask legs count; an in-mask leg with no signal is
        charged full loss), the returned components are the report-display
        `masked_view`, and the zero-placement rule charges every in-mask
        leg full loss when the solution scheduled nothing while tasks
        applied. When `mask` is None the raw components and the
        all-positive-weight weighted loss are returned.
        """
        pref_value: float | None
        if persona_constraints is not None and window_map is not None:
            pref_value, _pattern_rows, _leg_stats = compute_l_pref_v2(
                solution,
                persona_constraints=persona_constraints,
                window_map=window_map,
                semantic=self._semantic,
            )
        else:
            pref_value = None
        divide_value, divide_records = compute_l_divide(
            solution, tolerance_pct=self._divide_tolerance_pct
        )
        if self._context_links_by_uri:
            from src.scripts.scenarios.metrics.context_fit import compute_l_context_fit

            context_fit_value = compute_l_context_fit(
                solution,
                calendar,
                self._context_links_by_uri,
                observed_categories=observed_categories,
            )
        else:
            context_fit_value = None
        components = LossComponents(
            cov=compute_l_cov(solution),
            cal=compute_l_cal(
                solution,
                calendar,
                self._ruleset,
                self._matcher,
                buffer_minutes=self._buffer_minutes,
            ),
            pref=pref_value,
            disp=compute_l_disp(
                solution,
                calendar,
                self._resolver,
                half_life_days=self._half_life_days,
                max_met=self._max_met,
            ),
            merge=compute_l_concurrent(
                solution,
                calendar,
                self._merge_semantic,
                ruleset=self._ruleset,
                matcher=self._matcher,
                merge_threshold=self._merge_threshold,
                bridge=self._merge_bridge,
            ),
            spread=compute_l_spread(solution, horizon_epochs),
            divide=divide_value,
            divide_verdicts=tuple(divide_records),
            context_fit=context_fit_value,
        )
        if mask is None:
            return components.weighted(self._weights), components
        zero_placement = not solution.scheduled and bool(solution.tasks)
        loss = components.weighted(self._weights, mask, zero_placement=zero_placement)
        return loss, components.masked_view(mask, zero_placement=zero_placement)

    def compute_weekly(
        self,
        solution: SchedulingSolution,
        calendar: CalendarTrace,
        week_dates: list[datetime.date],
        time_windows: dict[str, WindowRange] | None = None,
        *,
        persona_constraints: "PersonaConstraints | None" = None,
        window_map: "WindowMap | None" = None,
        observed_categories: frozenset[str] | None = None,
        mask: "frozenset[str] | set[str] | None" = None,
        week_tasks: list | None = None,
    ) -> tuple[float, LossComponents]:
        """Slice the solution + calendar to `week_dates` and score that week.

        Args:
            solution: full cohort-wide solution.
            calendar: full base calendar trace.
            week_dates: ordered list of 7 dates defining the week.
            time_windows: optional window dictionary forwarded to `compute`.
            persona_constraints: per-person preference constraints.
            window_map: per-person window map.
            observed_categories: context category opt-ins for `L_context_fit`.
            week_tasks: tasks recommended for this ISO week; when supplied they
                become the coverage / divide denominator so the week is scored
                against its own batch, not the whole-horizon task list.

        Returns:
            `(weighted_loss, components)` for the week-restricted slice.
        """
        sliced_solution, sliced_calendar = _slice_solution_for_week(
            solution, calendar, week_dates, week_tasks
        )
        return self.compute(
            sliced_solution,
            sliced_calendar,
            horizon_epochs=max(1, len(week_dates)),
            time_windows=time_windows,
            persona_constraints=persona_constraints,
            window_map=window_map,
            observed_categories=observed_categories,
            mask=mask,
        )


def _slice_solution_for_week(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
    week_dates: list[datetime.date],
    week_tasks: list | None = None,
) -> tuple[SchedulingSolution, CalendarTrace]:
    """Return week-restricted copies of `solution` and `calendar`.

    The new solution keeps tasks whose `date` is inside `week_dates`;
    `unscheduled` is recomputed as `tasks - scheduled` so `L_cov` is
    correct for the week. `week_tasks`, when supplied, is that week's
    recommended batch and becomes the task universe, so coverage and
    splitting are scored against the week's own tasks rather than the
    whole-horizon list (otherwise the denominator would be every week's
    tasks and per-week coverage would collapse to about `1 / num_weeks`).
    """
    from src.scripts.scenarios.domain.calendar import AugmentedCalendar

    window = set(week_dates)
    sched = [st for st in solution.scheduled if st.date in window]
    tasks = list(week_tasks) if week_tasks is not None else list(solution.tasks)
    scheduled_labels = {st.task.label for st in sched}
    unscheduled = [t for t in tasks if t.label not in scheduled_labels]
    aug_cal = AugmentedCalendar(
        person_id=solution.person_id,
        base_events=[ev for ev in calendar.events if ev.date in window],
        scheduled_tasks=sched,
    )
    sliced_solution = SchedulingSolution(
        person_id=solution.person_id,
        augmented_calendar=aug_cal,
        tasks=tasks,
        scheduled=sched,
        unscheduled=unscheduled,
    )
    sliced_calendar = CalendarTrace(
        person_id=calendar.person_id,
        events=[ev for ev in calendar.events if ev.date in window],
        contexts=[c for c in getattr(calendar, "contexts", []) if c.date in window],
    )
    return sliced_solution, sliced_calendar
