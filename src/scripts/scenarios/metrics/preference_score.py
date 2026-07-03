"""Per-leg scorers for the preference loss `L_pref = |Ω|^-1 Σ_{j∈Ω} δ_j`.

Each scorer computes one normalized gap `δ_j ∈ [0, 1]` from a single
persona preference for a placed task `ê_k`, where `0` is "perfectly
aligned" and `1` is "fully violated". Scorers with no signal return
`None`; the aggregator mean-averages over the non-`None` legs that form
the preference set `Ω`.

Legs (one function each), matching the `δ_j` preference families:

* `score_per_occurrence_duration`; per-occurrence duration leg.  Is one
  occurrence's duration inside `per_event_duration[min, max]` for the
  matched event?
* `score_per_scale_duration`; per-scale duration leg.  Does the sum
  of durations within one scale slice (day / week / month / season)
  satisfy `total_event_duration[min, max]`?
* `score_per_scale_episodes`; per-scale episode-count leg.  Does the
  count of occurrences within one scale slice satisfy
  `total_event_episodes[min, max]`?
* `score_persona_stage_semantic`; semantic alignment with the routine
  activity the placement overlaps.  Does `ê_k` overlap a firing persona
  stage, and is `ê_k` semantically compatible (σ) with that stage?
* `score_temporal_pattern`; temporal-pattern alignment (fix /
  seasonality / trend) per the quantitative grammar.

The functions consume the typed bundles produced by
:class:`PersonaConstraints` and the
:class:`PreferenceMapper`-resolved :class:`MappedEvent` list.
They are pure, side-effect-free, and easily mocked.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable

from src.scripts.persona.config.schema import EventDefinition, TemporalPattern
from src.scripts.persona.constraints.extract import (
    _amount_in_minutes,
    _calendar_month_to_season,
    _day_to_calendar_month,
    _resolve_amount_unit,
    _scale_indices,
    _within_month_indices,
    _within_season_indices,
    _within_weekday_indices,
)
from src.scripts.persona.domain.time_windows import WindowMap
from src.scripts.scenarios.domain.task import ScheduledTask
from src.scripts.scenarios.metrics.preference_constraints import (
    PersonaConstraints,
    ResolvedStage,
)
from src.scripts.scenarios.metrics.preference_mapping import MappedEvent
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

EPS = 1.0  # floor for "max(expected, ε)" in violation scoring


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _duration_violation(value: float, lo: float, hi: float) -> float:
    """Return the normalized gap `δ ∈ [0, 1]` for `value` against `[lo, hi]`.

    Inside-band gives 0.  Below band gives `(lo - value) / max(lo, EPS)`
    clipped to 1.  Above band gives `(value - hi) / max(hi, EPS)` clipped
    to 1.
    """
    if value < lo:
        denom = max(lo, EPS)
        return min(1.0, (lo - value) / denom)
    if value > hi:
        denom = max(hi, EPS)
        return min(1.0, (value - hi) / denom)
    return 0.0


def _count_violation(observed: float, lo: int, hi: int) -> float:
    """Normalized gap `δ ∈ [0, 1]` for an integer count against `[lo, hi]`."""
    if lo <= observed <= hi:
        return 0.0
    if observed < lo:
        return min(1.0, (lo - observed) / max(lo, EPS))
    return min(1.0, (observed - hi) / max(hi, EPS))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    """`Σ w·v / Σ w`; `None` when no positive weight is present."""
    total_w = sum(w for w in weights if w > 0)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights) if w > 0) / total_w


def _overlap_minutes(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


# ---------------------------------------------------------------------------
# Per-occurrence + per-scale duration
# ---------------------------------------------------------------------------


def score_per_occurrence_duration(
    scheduled: ScheduledTask,
    *,
    matched_events: Sequence[MappedEvent],
    constraints: PersonaConstraints,
) -> float | None:
    """Weighted-mean `δ` over each matched event's per-occurrence band.

    `None` is returned when no matched event resolves to a defined
    duration band (e.g. when the task carries no IRI and the mapper
    returns no matches).
    """
    if not matched_events:
        return None
    duration = max(0, scheduled.end_minutes - scheduled.start_minutes)
    day_idx, day_of_week = _date_to_day_idx_dow(scheduled.date, constraints)
    legs: list[float] = []
    weights: list[float] = []
    for mapped in matched_events:
        dc = constraints.day_constraints_for(
            mapped.event_name, day_idx=day_idx, day_of_week=day_of_week
        )
        if dc is None or mapped.weight <= 0:
            continue
        violation = _duration_violation(
            float(duration), float(dc.per_event_min), float(dc.per_event_max)
        )
        legs.append(violation)
        weights.append(mapped.weight)
    return _weighted_mean(legs, weights)


def score_per_scale_duration(
    placements: Sequence[ScheduledTask],
    *,
    matched_events: Sequence[MappedEvent],
    constraints: PersonaConstraints,
) -> float | None:
    """Per-scale duration gap `δ` across the persona's full placement list.

    For each matched event we look up the resolved
    `total_event_duration.scale` (day / week / month / season), bucket
    every placement by that scale, sum durations per slice, and compare
    to the catalog's `[total_min, total_max]` band.  The slice-level gaps
    are averaged into one per-event leg, then weighted-mean'd across
    matched events.

    `None` when no matched event resolves to a defined band.
    """
    if not matched_events or not placements:
        return None
    legs: list[float] = []
    weights: list[float] = []
    for mapped in matched_events:
        ev = constraints.resolved_event_def_for(mapped.event_name)
        if ev is None or mapped.weight <= 0:
            continue
        slice_totals = _bucket_durations(
            placements, scale=ev.total_event_duration.scale, constraints=constraints
        )
        if not slice_totals:  # pragma: no cover - defensive (placements is non-empty)
            continue
        per_slice_violations: list[float] = []
        for slice_key, total_minutes in slice_totals.items():
            day_idx, day_of_week = slice_key
            dc = constraints.day_constraints_for(
                mapped.event_name, day_idx=day_idx, day_of_week=day_of_week
            )
            if dc is None:  # pragma: no cover - defensive (mapped event resolves)
                continue
            per_slice_violations.append(
                _duration_violation(
                    float(total_minutes), float(dc.total_min), float(dc.total_max)
                )
            )
        if not per_slice_violations:  # pragma: no cover - defensive
            continue
        legs.append(sum(per_slice_violations) / len(per_slice_violations))
        weights.append(mapped.weight)
    return _weighted_mean(legs, weights)


# ---------------------------------------------------------------------------
# Per-scale episode count
# ---------------------------------------------------------------------------


def score_per_scale_episodes(
    placements: Sequence[ScheduledTask],
    *,
    matched_events: Sequence[MappedEvent],
    constraints: PersonaConstraints,
) -> float | None:
    """Episode-count gap `δ` across the persona's placements.

    For each matched event we bucket placement counts by the event's
    `total_event_episodes.scale` and compare to the
    `[min, max]` band.  Gaps averaged per-event, then weighted-mean'd
    across matched events.
    """
    if not matched_events or not placements:
        return None
    legs: list[float] = []
    weights: list[float] = []
    for mapped in matched_events:
        ev = constraints.resolved_event_def_for(mapped.event_name)
        if ev is None or mapped.weight <= 0:
            continue
        slice_counts = _bucket_counts(
            placements, scale=ev.total_event_episodes.scale, constraints=constraints
        )
        if not slice_counts:  # pragma: no cover - defensive (placements is non-empty)
            continue
        per_slice_violations: list[float] = []
        for slice_key, count in slice_counts.items():
            day_idx, day_of_week = slice_key
            dc = constraints.day_constraints_for(
                mapped.event_name, day_idx=day_idx, day_of_week=day_of_week
            )
            if dc is None:  # pragma: no cover - defensive (mapped event resolves)
                continue
            per_slice_violations.append(
                _count_violation(float(count), dc.min_count, dc.max_count)
            )
        if not per_slice_violations:  # pragma: no cover - defensive
            continue
        legs.append(sum(per_slice_violations) / len(per_slice_violations))
        weights.append(mapped.weight)
    return _weighted_mean(legs, weights)


# ---------------------------------------------------------------------------
# Persona-stage semantic alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageAlignment:
    """One `(stage, σ, classification)` row for sidecar / report use."""

    stage_name: str
    sigma: float
    aligned: bool


def score_persona_stage_semantic(
    scheduled: ScheduledTask,
    *,
    firing_stages: Sequence[ResolvedStage],
    semantic: SemanticCompatibility | None,
    stage_alignment_threshold: float = 0.65,
) -> float | None:
    """Semantic-alignment gap `δ` between `ê_k` and the routine it overlaps.

    Per overlapping stage we compute σ(ê_k.label, stage.name), the
    semantic compatibility of the placement with the routine activity it
    overlaps. A stage with σ resolved from the `none` stub (no embedding,
    no LLM scorer) carries no real signal and is skipped, so the leg
    returns `None` instead of saturating at `1 - 0`. A stage with
    σ ≥ threshold is aligned (contributes 0). A stage with a real σ below
    threshold contributes `1 - σ`. The leg returns the maximum gap across
    signal-carrying overlapping stages, or `None` when no overlapping
    stage carries signal.
    """
    if not firing_stages or semantic is None:
        return None
    overlapping = [
        stage
        for stage in firing_stages
        if _overlap_minutes(
            scheduled.start_minutes,
            scheduled.end_minutes,
            stage.window_start,
            stage.window_end,
        )
        > 0
    ]
    if not overlapping:
        return None
    max_violation: float | None = None
    for stage in overlapping:
        sigma, strategy = semantic.score_with_strategy(scheduled.task.label, stage.name)
        if strategy == "none":
            continue  # no oracle resolved this pair; skip rather than penalize
        if sigma >= stage_alignment_threshold:
            if max_violation is None:
                max_violation = 0.0
            continue
        clash = max(0.0, 1.0 - sigma)
        if max_violation is None or clash > max_violation:
            max_violation = clash
    return max_violation


# ---------------------------------------------------------------------------
# Temporal-pattern scorers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatternViolation:
    """Per-pattern report row: clipped loss + uncapped MAPE for sidecar."""

    mode: str  # "fix" | "seasonality" | "trend"
    scale: str | None
    loss: float
    mape: float | None  # uncapped Mean-Absolute-Percentage-Error, or None


def score_temporal_pattern(
    placements: Sequence[ScheduledTask],
    *,
    matched_events: Sequence[MappedEvent],
    constraints: PersonaConstraints,
    window_map: WindowMap,
) -> tuple[float | None, list[PatternViolation]]:
    """Aggregate temporal-pattern gap `δ` across every matched event.

    Each pattern declared on a matched event contributes one gap; we
    weighted-mean across (matched-event × pattern).  Patterns the matched
    event does not declare contribute nothing.

    Also returns the raw `PatternViolation` rows so the reporter can
    surface the MAPE alongside the clipped score.
    """
    rows: list[PatternViolation] = []
    if not matched_events:
        return None, rows
    weighted_legs: list[float] = []
    weights: list[float] = []
    for mapped in matched_events:
        ev = constraints.resolved_event_def_for(mapped.event_name)
        if ev is None or mapped.weight <= 0:
            continue
        for pattern in ev.temporal_patterns:
            row = _score_one_pattern(
                pattern,
                placements=placements,
                event=ev,
                constraints=constraints,
                window_map=window_map,
            )
            if row is None:
                continue
            rows.append(row)
            weighted_legs.append(row.loss)
            weights.append(mapped.weight)
    if not weighted_legs:
        return None, rows
    aggregate = _weighted_mean(weighted_legs, weights)
    return aggregate, rows


def _score_one_pattern(
    pattern: TemporalPattern,
    *,
    placements: Sequence[ScheduledTask],
    event: EventDefinition,
    constraints: PersonaConstraints,
    window_map: WindowMap,
) -> PatternViolation | None:
    """Dispatch one pattern to its scorer."""
    mode = pattern.mode
    details = pattern.details
    scale = details.get("scale")
    if mode == "fix":
        loss = _score_fix_within(placements, pattern=pattern, window_map=window_map)
        if loss is None:
            return None
        return PatternViolation(mode="fix", scale=None, loss=loss, mape=None)
    if mode == "seasonality":
        loss, mape = _score_seasonality(
            placements,
            pattern=pattern,
            event=event,
            constraints=constraints,
        )
        if loss is None:
            return None
        return PatternViolation(mode="seasonality", scale=scale, loss=loss, mape=mape)
    if mode == "trend":
        loss, mape = _score_trend(
            placements,
            pattern=pattern,
            event=event,
            constraints=constraints,
        )
        if loss is None:
            return None
        return PatternViolation(mode="trend", scale=scale, loss=loss, mape=mape)
    return None  # pragma: no cover - schema blocks other modes


def _score_fix_within(
    placements: Sequence[ScheduledTask],
    *,
    pattern: TemporalPattern,
    window_map: WindowMap,
) -> float | None:
    """`mode: fix`; fraction of placement duration outside the declared windows."""
    within = pattern.details.get("within")
    if isinstance(within, str):
        windows = [within] if within in window_map else []
    elif isinstance(within, list):
        windows = [w for w in within if w in window_map]
    else:
        windows = []
    if not windows or not placements:
        return None
    ranges = [window_map.get(w) for w in windows]
    total_minutes = 0
    outside_minutes = 0
    for placement in placements:
        duration = max(0, placement.end_minutes - placement.start_minutes)
        if duration == 0:
            continue
        total_minutes += duration
        inside = sum(
            _overlap_minutes(placement.start_minutes, placement.end_minutes, start, end)
            for start, end in ranges
        )
        outside_minutes += max(0, duration - inside)
    if total_minutes == 0:
        return None
    return outside_minutes / total_minutes


def _score_seasonality(
    placements: Sequence[ScheduledTask],
    *,
    pattern: TemporalPattern,
    event: EventDefinition,
    constraints: PersonaConstraints,
) -> tuple[float | None, float | None]:
    """`mode: seasonality`; observed-vs-expected ratio per scale slice."""
    details = pattern.details
    amt_unit = _resolve_amount_unit(details, "seasonality")
    is_percent = amt_unit == "percent"
    is_duration = amt_unit in ("minutes", "hours") or (
        is_percent and details.get("target") == "duration"
    )
    scale = details.get("scale")
    within = details.get("within")
    amount = float(details.get("amount", 0))
    direction = details.get("direction", "increasing")
    if not placements:
        return None, None
    if scale not in ("weekday", "month", "season"):
        # Window-fraction seasonality is enforced by the solver; the
        # metric does not re-score that here.
        return None, None
    horizon_start = constraints.horizon_start_date
    # Group placements by the scale's slicing.
    slice_keys: dict[int, list[ScheduledTask]] = {}
    for placement in placements:
        key = _seasonality_slice_key(placement.date, scale, horizon_start)
        slice_keys.setdefault(key, []).append(placement)
    if not slice_keys:  # pragma: no cover - defensive (placements non-empty)
        return None, None
    # Build the membership predicate for "this slice is inside `within`".
    allowed: set[int]
    if scale == "weekday":
        allowed = set(_within_weekday_indices(within))
    elif scale == "month":
        allowed = set(_within_month_indices(within))
    else:
        allowed = set(_within_season_indices(within))
    if not allowed:
        return None, None
    sign = 1.0 if direction == "increasing" else -1.0
    # Compute "observed" measure per slice + "expected" baseline.
    measures: dict[int, float] = {}
    for key, sli in slice_keys.items():
        if is_duration:
            measures[key] = sum(max(0, p.end_minutes - p.start_minutes) for p in sli)
        else:
            measures[key] = float(len(sli))
    baseline = sum(measures.values()) / len(measures)
    if baseline == 0:  # pragma: no cover - defensive (count never 0 here)
        return None, None
    if is_percent:
        expected_inside = baseline * (1.0 + sign * amount / 100.0)
    else:
        amount_abs = _amount_in_minutes(amount, amt_unit) if is_duration else amount
        expected_inside = baseline + sign * amount_abs
    expected_outside = baseline
    per_slice_violations: list[float] = []
    abs_pcts: list[float] = []
    for key, observed in measures.items():
        expected = expected_inside if key in allowed else expected_outside
        if expected <= 0:
            expected = baseline
        diff = abs(observed - expected)
        per_slice_violations.append(min(1.0, diff / max(expected, EPS)))
        abs_pcts.append(diff / max(expected, EPS))
    loss = sum(per_slice_violations) / len(per_slice_violations)
    mape = sum(abs_pcts) / len(abs_pcts)
    return loss, mape


def _score_trend(
    placements: Sequence[ScheduledTask],
    *,
    pattern: TemporalPattern,
    event: EventDefinition,
    constraints: PersonaConstraints,
) -> tuple[float | None, float | None]:
    """`mode: trend`; cumulative-expected ramp comparison.

    An augmenter that front-loads in slice 1 and stops cannot game the
    metric: we compare cumulative observed against cumulative expected
    at each slice index.
    """
    details = pattern.details
    amt_unit = _resolve_amount_unit(details, "trend")
    is_percent = amt_unit == "percent"
    is_duration = amt_unit in ("minutes", "hours") or (
        is_percent and details.get("target") == "duration"
    )
    scale = details.get("scale", "day")
    amount = (
        _amount_in_minutes(float(details.get("amount", 0)), amt_unit)
        if is_duration and not is_percent
        else float(details.get("amount", 0))
    )
    direction = details.get("direction", "increasing")
    horizon_start = constraints.horizon_start_date
    horizon_days = constraints.horizon_days
    scale_total = _scale_indices(
        scale, day_idx=0, total_days=horizon_days, horizon_start_date=horizon_start
    )[1]
    start_idx = int(details.get("start", 1))
    end_idx = int(details.get("end", scale_total))
    if end_idx <= start_idx or not placements:
        return None, None
    # Group placements by scale slice.
    measures: dict[int, float] = {}
    for placement in placements:
        scale_idx, _ = _scale_indices(
            scale,
            day_idx=_date_to_day_idx(placement.date, constraints),
            total_days=horizon_days,
            horizon_start_date=horizon_start,
        )
        if is_duration:
            measures[scale_idx] = measures.get(scale_idx, 0.0) + max(
                0, placement.end_minutes - placement.start_minutes
            )
        else:
            measures[scale_idx] = measures.get(scale_idx, 0.0) + 1
    if not measures:  # pragma: no cover - defensive (placements non-empty)
        return None, None
    # Compute baseline = average measure across slices outside the ramp.
    inside_indices = list(range(start_idx - 1, end_idx))
    outside_indices = [i for i in measures if i not in inside_indices]
    baseline_values = [measures[i] for i in outside_indices if measures.get(i, 0) > 0]
    if baseline_values:
        baseline = sum(baseline_values) / len(baseline_values)
    else:
        # No "outside ramp" observations; use the mean of all measures.
        baseline = sum(measures.values()) / len(measures)
    # Cumulative observed vs cumulative expected.
    cumulative_obs = 0.0
    cumulative_exp = 0.0
    per_slice_violations: list[float] = []
    abs_pcts: list[float] = []
    sign = 1.0 if direction == "increasing" else -1.0
    span = end_idx - start_idx
    for i in range(start_idx - 1, end_idx):
        progress = ((i - (start_idx - 1)) / span) if span > 0 else 0.0
        if is_percent:
            expected_at_i = baseline * (1.0 + sign * amount / 100.0 * progress)
        else:
            expected_at_i = baseline + sign * amount * progress
        cumulative_obs += measures.get(i, 0.0)
        cumulative_exp += expected_at_i
        if cumulative_exp <= 0:
            continue
        diff = abs(cumulative_obs - cumulative_exp)
        per_slice_violations.append(min(1.0, diff / max(cumulative_exp, EPS)))
        abs_pcts.append(diff / max(cumulative_exp, EPS))
    if not per_slice_violations:  # pragma: no cover - defensive (rare degenerate case)
        return None, None
    loss = sum(per_slice_violations) / len(per_slice_violations)
    mape = sum(abs_pcts) / len(abs_pcts)
    return loss, mape


def _seasonality_slice_key(
    date: _dt.date, scale: str | None, horizon_start: _dt.date | None
) -> int:
    """Map `date` to the slice index used by seasonality grouping."""
    if scale == "weekday":
        return date.weekday()
    if scale == "month":
        return date.month  # 1..12
    if scale == "season":
        return _calendar_month_to_season(date.month)
    return 0  # pragma: no cover - guarded by caller


# ---------------------------------------------------------------------------
# Shared bucketing helpers
# ---------------------------------------------------------------------------


def _date_to_day_idx(date: _dt.date, constraints: PersonaConstraints) -> int:
    """Convert calendar date to 0-indexed day-of-horizon."""
    base = constraints.horizon_start_date or date
    delta = (date - base).days
    return max(0, delta)


def _date_to_day_idx_dow(
    date: _dt.date, constraints: PersonaConstraints
) -> tuple[int, int]:
    return _date_to_day_idx(date, constraints), date.weekday()


def _bucket_durations(
    placements: Iterable[ScheduledTask],
    *,
    scale: str,
    constraints: PersonaConstraints,
) -> dict[tuple[int, int], int]:
    """Sum placement durations per (day_idx, day_of_week) bucket.

    The `scale` field is honored at lookup time via
    `day_constraints_for` so we only need one representative
    `(day_idx, day_of_week)` per slice; we pick the first one.
    """
    horizon_start = constraints.horizon_start_date
    horizon_days = constraints.horizon_days
    # Map slice_idx to (day_idx_repr, day_of_week_repr, total_minutes)
    buckets: dict[int, tuple[int, int, int]] = {}
    for placement in placements:
        day_idx = _date_to_day_idx(placement.date, constraints)
        day_of_week = placement.date.weekday()
        scale_idx, _ = _scale_indices(
            scale,
            day_idx=day_idx,
            total_days=horizon_days,
            horizon_start_date=horizon_start,
        )
        duration = max(0, placement.end_minutes - placement.start_minutes)
        prev = buckets.get(scale_idx)
        if prev is None:
            buckets[scale_idx] = (day_idx, day_of_week, duration)
        else:
            buckets[scale_idx] = (prev[0], prev[1], prev[2] + duration)
    return {(d, dow): total for (d, dow, total) in buckets.values()}


def _bucket_counts(
    placements: Iterable[ScheduledTask],
    *,
    scale: str,
    constraints: PersonaConstraints,
) -> dict[tuple[int, int], int]:
    """Count placements per scale-slice."""
    horizon_start = constraints.horizon_start_date
    horizon_days = constraints.horizon_days
    buckets: dict[int, tuple[int, int, int]] = {}
    for placement in placements:
        day_idx = _date_to_day_idx(placement.date, constraints)
        day_of_week = placement.date.weekday()
        scale_idx, _ = _scale_indices(
            scale,
            day_idx=day_idx,
            total_days=horizon_days,
            horizon_start_date=horizon_start,
        )
        prev = buckets.get(scale_idx)
        if prev is None:
            buckets[scale_idx] = (day_idx, day_of_week, 1)
        else:
            buckets[scale_idx] = (prev[0], prev[1], prev[2] + 1)
    return {(d, dow): cnt for (d, dow, cnt) in buckets.values()}


# ---------------------------------------------------------------------------
# Sidecar aggregation (sidecar + reporter)
# ---------------------------------------------------------------------------


def read_preference_violations(sidecar_paths: Iterable[Any]) -> dict[str, dict]:
    """Aggregate per-persona `preference_violations.jsonl` sidecars.

    Returns the `preference_breakdown` dict consumed by
    :func:`scenarios.export.report_writer.format_loss_report`.
    Missing files are silently skipped; malformed rows are dropped so
    one bad sidecar cannot poison the whole report.
    """
    import json as _json
    from pathlib import Path as _Path

    by_leg: dict[str, dict[str, Any]] = {}

    def _accumulate(
        leg_name: str,
        *,
        applicable: int,
        total_loss: float,
        mape_total: float | None,
        mape_n: int,
    ) -> None:
        cur = by_leg.setdefault(
            leg_name,
            {"applicable_tasks": 0, "_loss_sum": 0.0, "_mape_sum": 0.0, "_mape_n": 0},
        )
        cur["applicable_tasks"] += applicable
        cur["_loss_sum"] += total_loss
        if mape_total is not None:
            cur["_mape_sum"] += mape_total
            cur["_mape_n"] += mape_n

    for path in sidecar_paths:
        p = _Path(path)
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - defensive
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("kind") != "leg":
                continue
            name = row.get("name")
            count = int(row.get("applicable_count") or 0)
            mean_loss = row.get("mean_loss")
            mape = row.get("mape")
            if not isinstance(name, str):
                continue
            if count == 0 or mean_loss is None:
                # No signal; register the leg so the report shows it
                # exists, but don't push a numeric contribution.
                by_leg.setdefault(
                    name,
                    {
                        "applicable_tasks": 0,
                        "_loss_sum": 0.0,
                        "_mape_sum": 0.0,
                        "_mape_n": 0,
                    },
                )
                continue
            _accumulate(
                name,
                applicable=count,
                total_loss=float(mean_loss) * count,
                mape_total=(float(mape) * count) if mape is not None else None,
                mape_n=count if mape is not None else 0,
            )

    out: dict[str, dict] = {}
    for name, agg in by_leg.items():
        applicable = agg["applicable_tasks"]
        mean_loss = (agg["_loss_sum"] / applicable) if applicable else None
        mape = (agg["_mape_sum"] / agg["_mape_n"]) if agg["_mape_n"] else None
        row: dict[str, Any] = {
            "applicable_tasks": applicable,
            "mean_loss": mean_loss,
        }
        if mape is not None:
            row["mape"] = mape
        out[name] = row
    return out


# Keep a couple of imports alive for downstream consumers that want
# month/season helpers without importing from `extract.py`.
__all__ = [
    "PatternViolation",
    "StageAlignment",
    "score_per_occurrence_duration",
    "score_per_scale_duration",
    "score_per_scale_episodes",
    "score_persona_stage_semantic",
    "score_temporal_pattern",
    "read_preference_violations",
    "_calendar_month_to_season",
    "_day_to_calendar_month",
]
