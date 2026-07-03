"""Write scheduling-gain evaluation reports as five isolated file pairs.

Each section of the evaluation lives in its own `.txt` + `.json` pair
so a downstream consumer can fetch exactly the slice they need without
parsing a single combined report:

  * `person_instance_scheduling_gain_report.{txt,json}`; per-person
    rows (total gain + per-component gain breakdown).
  * `total_scheduling_gain.{txt,json}`               ; cohort
    aggregates (`average_total_gain` + `average_gains` +
    `scored_persons` + `empty_plan_persons`).
  * `ontology_grounding.{txt,json}`                  ; share of
    generated tasks whose `ontology_uri` resolves in the knowledge
    graph.  Written only when grounding is supplied.
  * `telemetry.{txt,json}`                           ; cost +
    wall-time + paraphrase-gate counters.  Written only when telemetry
    is supplied.
  * `preference_breakdown.{txt,json}`                ; per-leg
    `L_pref` roll-up.  Written only when the breakdown dict is
    supplied.

Reports surface per-component **gains** rather than losses; every
component is normalized to `[0, 1]` as a loss internally
(:class:`LossComponents` in :mod:`metrics.loss`), and the writer
converts to `gain = 1 − loss` with the long form key name before the
value reaches a human or a JSON consumer.

The full-name mapping below is the single source of truth for the
user-facing terminology.  Math code and docstrings inside
`metrics/loss.py` keep the short abbreviations (`cov`, `cal`, …)
since they appear in equations; this layer renames them at the report
boundary so nobody has to remember what `L_cov` stood for when
reading a report.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.scripts.scenarios.metrics.loss import LossComponents

# Short abbreviation to user-facing long name.  Keep this list in the
# same order as the fields on :class:`LossComponents` so the per-person
# report rows render in a predictable sequence.
#
#   cov          recommended_task_coverage
#   cal          task_event_and_task_task_temporal_relations
#   pref         user_preference_deviation
#   disp         intensive_task_dispersion
#   merge        semantic_coscheduling_merge
#   spread       recommended_task_spread
#   divide       dividable_task_split_reward
#   context_fit  user_context_recommendation_fit
_COMPONENT_FULL_NAMES: tuple[tuple[str, str], ...] = (
    ("cov", "recommended_task_coverage"),
    ("cal", "task_event_and_task_task_temporal_relations"),
    ("pref", "user_preference_deviation"),
    ("disp", "intensive_task_dispersion"),
    ("merge", "semantic_coscheduling_merge"),
    ("spread", "recommended_task_spread"),
    ("divide", "dividable_task_split_reward"),
    ("context_fit", "user_context_recommendation_fit"),
)

# Public filenames; the index that downstream tooling can rely on.
PERSON_REPORT_BASENAME = "person_instance_scheduling_gain_report"
TOTAL_REPORT_BASENAME = "total_scheduling_gain"
WEEKLY_GAIN_REPORT_BASENAME = "weekly_scheduling_gain"
ONTOLOGY_REPORT_BASENAME = "ontology_grounding"
TELEMETRY_REPORT_BASENAME = "telemetry"
PREFERENCE_REPORT_BASENAME = "preference_breakdown"
DIVIDE_REPORT_BASENAME = "divide_breakdown"
CONTEXT_FIT_REPORT_BASENAME = "context_fit_breakdown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _components_to_gains(comp: LossComponents) -> dict[str, float | None]:
    """Return per-component **gains** keyed by the user-facing long name.

    Each component value on :class:`LossComponents` lives in `[0, 1]`
    as a loss; the gain is the complement.  `None` losses (currently
    `divide` when no dividable tasks exist, `merge` when Φ is
    empty) surface as `None` gains so the report can render them as
    `n/a` instead of giving the augmenter free credit for a dimension
    it had no chance to act on.
    """
    out: dict[str, float | None] = {}
    for abbr, full in _COMPONENT_FULL_NAMES:
        value = getattr(comp, abbr)
        out[full] = None if value is None else 1.0 - value
    return out


def _empty_plan_persons(grounding: dict | None) -> set[str]:
    """Return the set of person_ids whose generated task list was empty.

    With zero tasks every loss component aggregates to zero, so the
    naive `gain = 1 − loss` formula reports a perfect `1.0` for
    what is actually a fully-failed run.  The grounding sidecar's
    per-person `total` field is the canonical "how many tasks did
    this person have to schedule" signal; we use it to mark those
    persons' gain as `null` instead of pretending they aced the
    benchmark.
    """
    if not grounding:
        return set()
    per_person = grounding.get("per_person") or {}
    return {pid for pid, info in per_person.items() if not (info or {}).get("total")}


def _gain_str(value: float | None) -> str:
    """Format a gain value for the plain-text report (`"n/a"` when None)."""
    return "n/a" if value is None else f"{value:.4f}"


def _name_col_width() -> int:
    """Width chosen so value columns line up across every `G_<name>` row."""
    return max(len(name) for _, name in _COMPONENT_FULL_NAMES)


# ---------------------------------------------------------------------------
# Section 1; per-person gain breakdown
# ---------------------------------------------------------------------------


def _format_window_line(window: dict | None) -> str | None:
    """Return `Window: 2026-06-01 .. 2026-06-14 (2 weeks)` or None."""
    if not window:
        return None
    start = window.get("start_date")
    end = window.get("end_date_inclusive")
    weeks = window.get("weeks")
    if not (start and end and weeks):
        return None
    return f"Window: {start} .. {end} ({weeks} weeks)"


def format_person_gains_report(
    results: list[tuple[str, float, LossComponents]],
    *,
    grounding: dict | None = None,
    window: dict | None = None,
) -> str:
    """Format the per-person section as plain text."""
    empty_pids = _empty_plan_persons(grounding)
    width = _name_col_width()
    lines = ["=== Person Instance Scheduling Gain Report ===", ""]
    window_line = _format_window_line(window)
    if window_line is not None:
        lines = [lines[0], window_line, ""]
    for person_id, total, comp in results:
        gain_value: float | None = None if person_id in empty_pids else 1.0 - total
        per_component = _components_to_gains(comp)
        lines.append(f"Person : {person_id}")
        lines.append(f"  Gain : {_gain_str(gain_value)}")
        for _abbr, full_name in _COMPONENT_FULL_NAMES:
            lines.append(
                f"  G_{full_name:<{width}} : " f"{_gain_str(per_component[full_name])}"
            )
        lines.append("")
    return "\n".join(lines)


def person_gains_to_dict(
    results: list[tuple[str, float, LossComponents]],
    *,
    grounding: dict | None = None,
    window: dict | None = None,
) -> dict:
    """Convert the per-person section to a JSON-serialisable dict."""
    empty_pids = _empty_plan_persons(grounding)
    persons = [
        {
            "person_id": pid,
            "total_gain": None if pid in empty_pids else 1.0 - total,
            "gains": _components_to_gains(comp),
        }
        for pid, total, comp in results
    ]
    payload: dict = {"persons": persons}
    if window:
        payload["window"] = window
    return payload


# ---------------------------------------------------------------------------
# Section 2; cohort total + per-component averages
# ---------------------------------------------------------------------------


def _cohort_aggregates(
    results: list[tuple[str, float, LossComponents]],
    empty_pids: set[str],
) -> tuple[float | None, dict[str, float | None] | None, int, int]:
    """Return `(avg_total_gain, avg_per_component_gain, scored, empty)`.

    Per-component averaging skips persons whose component is `None`
    so the divide / merge cells are not inflated by personas with no
    signal.
    """
    scored = [r for r in results if r[0] not in empty_pids]
    if not scored:
        return None, None, 0, len(empty_pids)
    avg_loss = sum(r[1] for r in scored) / len(scored)
    avg_per_component_gain: dict[str, float | None] = {}
    for abbr, full in _COMPONENT_FULL_NAMES:
        sig_values = [
            1.0 - getattr(comp, abbr)
            for _, _, comp in scored
            if getattr(comp, abbr) is not None
        ]
        avg_per_component_gain[full] = (
            sum(sig_values) / len(sig_values) if sig_values else None
        )
    return 1.0 - avg_loss, avg_per_component_gain, len(scored), len(empty_pids)


def format_total_gain_report(
    results: list[tuple[str, float, LossComponents]],
    *,
    grounding: dict | None = None,
    window: dict | None = None,
) -> str:
    """Format the cohort-aggregate section as plain text."""
    empty_pids = _empty_plan_persons(grounding)
    avg_total, avg_per_component, scored, empty = _cohort_aggregates(
        results, empty_pids
    )
    width = _name_col_width()
    lines = ["=== Total Scheduling Gain ===", ""]
    window_line = _format_window_line(window)
    if window_line is not None:
        lines = [lines[0], window_line, ""]
    if avg_total is not None and avg_per_component is not None:
        lines += [
            f"Average total gain : {avg_total:.4f}  (over {scored} scored persons)",
            "",
            "── Average per-component gain ──",
        ]
        for _abbr, full_name in _COMPONENT_FULL_NAMES:
            lines.append(
                f"  G_{full_name:<{width}} : "
                f"{_gain_str(avg_per_component[full_name])}"
            )
    elif results:
        lines.append("Average total gain : n/a  (every persona had 0 tasks)")
    else:
        lines.append("Average total gain : n/a  (no persons evaluated)")
    if empty:
        lines.append("")
        lines.append(
            f"Empty plans excluded from average : {empty} "
            "(0 tasks generated for these persons; gain reported as n/a)"
        )
    return "\n".join(lines)


def total_gain_to_dict(
    results: list[tuple[str, float, LossComponents]],
    *,
    grounding: dict | None = None,
    window: dict | None = None,
) -> dict:
    """Convert the cohort-aggregate section to a JSON-serialisable dict."""
    empty_pids = _empty_plan_persons(grounding)
    avg_total, avg_per_component, scored, empty = _cohort_aggregates(
        results, empty_pids
    )
    payload: dict = {
        "average_total_gain": avg_total,
        "average_gains": avg_per_component,
        "scored_persons": scored,
        "empty_plan_persons": empty,
    }
    if window:
        payload["window"] = window
    return payload


# ---------------------------------------------------------------------------
# Section 3; ontology grounding
# ---------------------------------------------------------------------------


def format_ontology_grounding_report(grounding: dict) -> str:
    """Format the ontology-grounding section as plain text."""
    lines = ["=== Ontology Grounding ===", ""]
    if not grounding.get("total"):
        lines.append("Ontology grounding : n/a  (no generated tasks)")
        return "\n".join(lines)
    lines.append(
        f"Ontology grounding : "
        f"{grounding['grounded']}/{grounding['total']} = "
        f"{grounding['ratio'] * 100:.1f}%"
    )
    verified = grounding.get("verified")
    verified_ratio = grounding.get("verified_ratio")
    if verified is not None and verified_ratio is not None:
        lines.append(
            f"Verified in ontology : "
            f"{verified}/{grounding['total']} = "
            f"{verified_ratio * 100:.1f}%"
        )
        unverified = grounding.get("persons_with_unverified") or 0
        if unverified:
            lines.append(
                f"Persons with unverified URIs : {unverified} "
                "(URI no longer resolves to an instance node)"
            )
    if grounding.get("persons_short_fetched"):
        lines.append(
            f"Persons short-fetched : "
            f"{grounding['persons_short_fetched']} "
            f"(< {grounding.get('expected_per_person')} grounded tasks)"
        )
    return "\n".join(lines)


def ontology_grounding_to_dict(grounding: dict) -> dict:
    """Return the ontology-grounding dict (passes through unchanged)."""
    return dict(grounding)


# ---------------------------------------------------------------------------
# Section 4; telemetry (cost, latency, paraphrase gates)
# ---------------------------------------------------------------------------


def _usd_str(value: float | int | None) -> str:
    """Format a USD cost value (`"n/a"` when None or non-numeric)."""
    return f"${value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _format_stage_block(name: str, stage: dict) -> list[str]:
    """One `── <Stage> ──` block (4-line totals + stage-specific extras)."""
    lines = [
        f"  ── {name} ──",
        f"    wall_time_total       : {stage['wall_time_seconds_total']:.1f}s",
        f"    tokens_total          : {stage['tokens_total']}",
        f"    estimated_cost        : {_usd_str(stage.get('estimated_usd_total'))}",
    ]
    if "n_tasks_placed" in stage:
        lines += [
            f"    tasks_attempted       : {stage['n_tasks_attempted']}",
            f"    tasks_placed          : {stage['n_tasks_placed']}",
            f"    tasks_dropped         : {stage['n_tasks_dropped']}",
        ]
    if "embedding_lookups_total" in stage:
        lines += [
            f"    embedding_lookups     : {stage['embedding_lookups_total']}",
            f"    uri_validations       : {stage['uri_validations_total']}",
        ]
    return lines


def _format_persona_block(persona_id: str, bucket: dict) -> str:
    """One per-persona row inside the `── Per-persona ──` block."""
    n = bucket.get("n_persons", 0)
    wall = bucket.get("wall_time_seconds_total", 0.0)
    avg_per_person = bucket.get("wall_time_seconds_avg_per_person", 0.0)
    tokens = bucket.get("tokens_total", 0)
    usd = bucket.get("estimated_usd_total")
    return (
        f"    {persona_id:<32} : "
        f"{n} persons, {wall:.1f}s total, "
        f"{avg_per_person:.2f}s/person, "
        f"{tokens} tokens, {_usd_str(usd)}"
    )


def format_telemetry_report(telemetry: dict) -> str:
    """Format the telemetry section as plain text; by_stage + by_persona."""
    lines = ["=== Telemetry ===", ""]
    n_persons = telemetry.get("n_persons")
    if not n_persons:
        lines.append("Telemetry : n/a  (no persons reported)")
        return "\n".join(lines)
    usd_s = _usd_str(telemetry.get("estimated_usd_total"))
    lines += [
        "── Cost & Latency ──",
        f"  persons               : {telemetry['n_persons']}",
        f"  personas              : {telemetry.get('n_personas', 0)}",
        f"  wall_time_total       : {telemetry['wall_time_seconds_total']:.1f}s",
        (
            f"  wall_time_per_person  : "
            f"{telemetry['wall_time_seconds_avg_per_person']:.2f}s"
        ),
        (
            f"  wall_time_per_persona : "
            f"{telemetry['wall_time_seconds_avg_per_persona']:.2f}s"
        ),
        f"  tokens_total          : {telemetry['tokens_total']}",
        f"  tokens_per_person     : {telemetry['tokens_avg_per_person']}",
        f"  tokens_per_persona    : {telemetry.get('tokens_avg_per_persona', 0)}",
        f"  tokens_per_task       : {telemetry['tokens_avg_per_task']}",
        f"  estimated_cost        : {usd_s}",
        f"  fetch_attempts_avg    : {telemetry.get('fetch_attempts_avg', 0.0)}",
        f"  fetch_attempts_max    : {telemetry.get('fetch_attempts_max', 0)}",
        f"  short_fetched         : {telemetry.get('persons_short_fetched', 0)}",
    ]

    by_stage = telemetry.get("by_stage") or {}
    if by_stage:
        lines += ["", "── Per-stage ──"]
        for label, key in (
            ("Task Generation", "task_generation"),
            ("Augmentation", "augmentation"),
            ("Evaluation", "evaluation"),
        ):
            stage = by_stage.get(key)
            if stage:
                lines += _format_stage_block(label, stage)

    by_persona = telemetry.get("by_persona") or {}
    if by_persona:
        lines += ["", "── Per-persona ──"]
        for persona_id in sorted(by_persona):
            lines.append(_format_persona_block(persona_id, by_persona[persona_id]))

    gate_failures = telemetry.get("gate_failures") or {}
    lines += [
        "",
        "── Paraphrase gates ──",
        (
            f"  acceptance_rate       : "
            f"{telemetry.get('paraphrase_acceptance_rate', 0.0) * 100:.1f}%"
        ),
        f"  failed_length_gate    : {gate_failures.get('length', 0)}",
        f"  failed_emoji_gate     : {gate_failures.get('emoji', 0)}",
        f"  failed_similarity_gate: {gate_failures.get('similarity', 0)}",
    ]
    return "\n".join(lines)


def telemetry_to_dict(telemetry: dict) -> dict:
    """Return the telemetry dict (passes through unchanged)."""
    return dict(telemetry)


# ---------------------------------------------------------------------------
# Section 5; L_pref preference breakdown
# ---------------------------------------------------------------------------


def format_preference_breakdown_report(preference_breakdown: dict) -> str:
    """Format the preference-breakdown section as plain text."""
    lines = ["=== L_pref Preference Breakdown ===", ""]
    if not preference_breakdown:
        lines.append("Preference breakdown : n/a  (no per-leg statistics)")
        return "\n".join(lines)
    for leg, stats in preference_breakdown.items():
        applicability = stats.get("applicable_tasks", 0)
        mean_loss = stats.get("mean_loss")
        mape = stats.get("mape")
        mean_str = "n/a" if mean_loss is None else f"{mean_loss:.4f}"
        row = f"  {leg:<32} : loss={mean_str}  applicable_tasks={applicability}"
        if mape is not None:
            row += f"  mape={mape * 100:.1f}%"
        lines.append(row)
    return "\n".join(lines)


def preference_breakdown_to_dict(preference_breakdown: dict) -> dict:
    """Return the preference-breakdown dict (passes through unchanged)."""
    return dict(preference_breakdown)


# ---------------------------------------------------------------------------
# Section 6; L_divide cohort breakdown
# ---------------------------------------------------------------------------


def format_divide_breakdown_report(divide_breakdown: dict) -> str:
    """Format the divide-breakdown section as plain text."""
    lines = ["=== L_divide Cohort Breakdown ===", ""]
    buckets = int(divide_breakdown.get("applicable_buckets", 0) or 0)
    if buckets == 0:
        lines.append("Divide breakdown : n/a  (no dividable signal in cohort)")
        return "\n".join(lines)
    persons = int(divide_breakdown.get("applicable_persons", 0) or 0)
    weeks = int(divide_breakdown.get("applicable_weeks", 0) or 0)
    lines.append(f"  applicable persons : {persons}")
    lines.append(f"  applicable weeks   : {weeks}")
    lines.append(f"  applicable buckets : {buckets}")
    lines.append("  verdict counts:")
    verdict_counts = divide_breakdown.get("verdict_counts") or {}
    for verdict, count in verdict_counts.items():
        pct = (count / buckets * 100.0) if buckets else 0.0
        lines.append(f"    {verdict:<36}: {count:>4} ({pct:>5.1f}%)")
    per_label = divide_breakdown.get("per_label") or {}
    if per_label:
        lines.append("")
        lines.append("  per dividable label:")
        for label, stats in per_label.items():
            row = (
                f"    {label:<32} "
                f"buckets={int(stats.get('buckets', 0))} "
                f"valid={int(stats.get('divided_valid', 0))} "
                f"mean_sum={float(stats.get('mean_sum_minutes', 0.0)):.1f} "
                f"mean_pieces={float(stats.get('mean_pieces_per_bucket', 0.0)):.1f}"
            )
            lines.append(row)
    return "\n".join(lines)


def divide_breakdown_to_dict(divide_breakdown: dict) -> dict:
    """Return the divide-breakdown dict (passes through unchanged)."""
    return dict(divide_breakdown)


# ---------------------------------------------------------------------------
# Top-level writer; five file pairs in *out_dir*
# ---------------------------------------------------------------------------


def aggregate_weekly_gain(persons_dir: Path) -> dict:
    """Read every `<pid>_weekly_gain.json` under `persons_dir` and aggregate.

    Returns `{"weeks": [{"week_index": int, "week_start": str,
    "avg_weighted_gain": float, "scored_persons": int}], "by_person":
    {pid: [weeks...]}}`. When no sidecars are found, returns
    `{"weeks": [], "by_person": {}}`.
    """
    persons_dir = Path(persons_dir)
    by_person: dict[str, list[dict]] = {}
    if not persons_dir.is_dir():
        return {"weeks": [], "by_person": {}}
    for path in sorted(persons_dir.glob("*_weekly_gain.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = payload.get("person_id") or path.stem.replace("_weekly_gain", "")
        rows = payload.get("weeks") or []
        if rows:
            by_person[pid] = rows
    if not by_person:
        return {"weeks": [], "by_person": {}}

    aggregate: dict[int, list[float]] = {}
    starts: dict[int, str] = {}
    for rows in by_person.values():
        for row in rows:
            idx = int(row.get("week_index", 0))
            gain = row.get("weighted_gain")
            if gain is None:
                continue
            aggregate.setdefault(idx, []).append(float(gain))
            starts.setdefault(idx, row.get("week_start", ""))
    weeks_summary = []
    for idx, values in sorted(aggregate.items()):
        weeks_summary.append(
            {
                "week_index": idx,
                "week_start": starts.get(idx, ""),
                "avg_weighted_gain": sum(values) / len(values),
                "scored_persons": len(values),
            }
        )
    return {"weeks": weeks_summary, "by_person": by_person}


# Margin a person's later weekly gain must clear over its week-1 gain to count
# as a learning onset, and the first-to-last gain lift that marks convergence.
ONSET_TAU: float = 0.05
CONVERGENCE_TAU: float = 0.05


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated q-quantile (q in [0, 1]) of a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _interquartile_mean(values: list[float]) -> float:
    """Mean of the middle 50%, trimming 25% off each tail."""
    ordered = sorted(values)
    n = len(ordered)
    cut = n // 4
    middle = ordered[cut : n - cut] if n >= 4 else ordered
    return sum(middle) / len(middle)


def ols_slope(weeks: list[int], gains: list[float]) -> float:
    """OLS slope of gain against week index; 0.0 when undefined."""
    n = len(weeks)
    if n < 2:
        return 0.0
    mean_x = sum(weeks) / n
    mean_y = sum(gains) / n
    denom = sum((x - mean_x) ** 2 for x in weeks)
    if denom == 0.0:
        return 0.0
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(weeks, gains))
    return num / denom


def _person_series(rows: list[dict]) -> tuple[list[int], list[float]]:
    """Return (week_indices, gains) sorted by week, dropping rows with no gain."""
    pairs = []
    for row in rows:
        gain = row.get("weighted_gain")
        if gain is None:
            continue
        pairs.append((int(row.get("week_index", 0)), float(gain)))
    pairs.sort()
    return [w for w, _ in pairs], [g for _, g in pairs]


def _onset_week(weeks: list[int], gains: list[float], tau: float) -> int | None:
    """First week from which every gain stays at least tau above the week-1 gain."""
    if len(gains) < 2:
        return None
    baseline = gains[0]
    for i in range(1, len(gains)):
        if all(gains[j] - baseline >= tau for j in range(i, len(gains))):
            return weeks[i]
    return None


def weekly_gain_distribution(by_person: dict[str, list[dict]]) -> list[dict]:
    """Per-week mean, median, IQM, and p25/p75 spread across persons."""
    by_week: dict[int, list[float]] = {}
    starts: dict[int, str] = {}
    for rows in by_person.values():
        for row in rows:
            gain = row.get("weighted_gain")
            if gain is None:
                continue
            idx = int(row.get("week_index", 0))
            by_week.setdefault(idx, []).append(float(gain))
            starts.setdefault(idx, row.get("week_start", ""))
    out = []
    for idx, values in sorted(by_week.items()):
        out.append(
            {
                "week_index": idx,
                "week_start": starts.get(idx, ""),
                "avg_weighted_gain": sum(values) / len(values),
                "median_weighted_gain": _percentile(values, 0.5),
                "iqm_weighted_gain": _interquartile_mean(values),
                "p25_weighted_gain": _percentile(values, 0.25),
                "p75_weighted_gain": _percentile(values, 0.75),
                "scored_persons": len(values),
            }
        )
    return out


def per_person_trajectory(
    by_person: dict[str, list[dict]],
    *,
    onset_tau: float = ONSET_TAU,
    convergence_tau: float = CONVERGENCE_TAU,
) -> dict[str, dict]:
    """Per-person onset week, first/last two-week means, delta, convergence, slope."""
    out: dict[str, dict] = {}
    for pid, rows in by_person.items():
        weeks, gains = _person_series(rows)
        if not gains:
            continue
        first2 = sum(gains[:2]) / len(gains[:2])
        last2 = sum(gains[-2:]) / len(gains[-2:])
        delta = last2 - first2
        out[pid] = {
            "onset_week": _onset_week(weeks, gains, onset_tau),
            "first2_mean": first2,
            "last2_mean": last2,
            "delta": delta,
            "converged": delta >= convergence_tau,
            "slope": ols_slope(weeks, gains),
        }
    return out


def cohort_learning_split(
    trajectory: dict[str, dict], *, convergence_tau: float = CONVERGENCE_TAU
) -> dict[str, int]:
    """Count persons whose gain improved, stayed flat, or declined."""
    converged = flat = declined = 0
    for stats in trajectory.values():
        delta = stats["delta"]
        if delta >= convergence_tau:
            converged += 1
        elif delta <= -convergence_tau:
            declined += 1
        else:
            flat += 1
    return {
        "converged": converged,
        "flat": flat,
        "declined": declined,
        "n": len(trajectory),
    }


def aggregate_weekly_gain_stats(persons_dir: Path) -> dict:
    """Bundle per-week distribution, per-person trajectory, and the cohort split."""
    by_person = aggregate_weekly_gain(persons_dir)["by_person"]
    trajectory = per_person_trajectory(by_person)
    return {
        "weeks": weekly_gain_distribution(by_person),
        "trajectory": trajectory,
        "split": cohort_learning_split(trajectory),
    }


def format_weekly_gain_report(aggregate: dict) -> str:
    """Render the aggregate weekly-gain dict as a plain-text block."""
    weeks = aggregate.get("weeks", []) if aggregate else []
    if not weeks:
        return "Weekly Scheduling Gain\n  (no per-week data)\n"
    lines = ["Weekly Scheduling Gain"]
    for row in weeks:
        lines.append(
            f"  Wk{int(row['week_index']):>2}  ({row.get('week_start', '----')})  "
            f"avg={float(row['avg_weighted_gain']):.4f}  "
            f"n={int(row['scored_persons'])}"
        )
    return "\n".join(lines) + "\n"


def _write_pair(
    out_dir: Path, basename: str, text: str, payload: dict
) -> tuple[Path, Path]:
    """Write `<basename>.txt` + `<basename>.json` and return both paths."""
    txt_path = out_dir / f"{basename}.txt"
    json_path = out_dir / f"{basename}.json"
    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return txt_path, json_path


def write_evaluation_reports(
    results: list[tuple[str, float, LossComponents]],
    out_dir: Path,
    *,
    grounding: dict | None = None,
    telemetry: dict | None = None,
    preference_breakdown: dict | None = None,
    divide_breakdown: dict | None = None,
    context_fit_breakdown: dict | None = None,
    window: dict | None = None,
    weekly_gain: dict | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Write all evaluation reports as isolated `.txt` + `.json` pairs.

    Always written:
      * `person_instance_scheduling_gain_report.{txt,json}`
      * `total_scheduling_gain.{txt,json}`

    Written only when the corresponding kwarg is supplied:
      * `ontology_grounding.{txt,json}`       (when *grounding*)
      * `telemetry.{txt,json}`                (when *telemetry*)
      * `preference_breakdown.{txt,json}`     (when *preference_breakdown*)
      * `divide_breakdown.{txt,json}`         (when *divide_breakdown*)

    Returns a dict mapping the section basename to `(txt_path,
    json_path)` for the files actually written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, tuple[Path, Path]] = {}

    written[PERSON_REPORT_BASENAME] = _write_pair(
        out_dir,
        PERSON_REPORT_BASENAME,
        format_person_gains_report(results, grounding=grounding, window=window),
        person_gains_to_dict(results, grounding=grounding, window=window),
    )
    written[TOTAL_REPORT_BASENAME] = _write_pair(
        out_dir,
        TOTAL_REPORT_BASENAME,
        format_total_gain_report(results, grounding=grounding, window=window),
        total_gain_to_dict(results, grounding=grounding, window=window),
    )
    if grounding is not None:
        written[ONTOLOGY_REPORT_BASENAME] = _write_pair(
            out_dir,
            ONTOLOGY_REPORT_BASENAME,
            format_ontology_grounding_report(grounding),
            ontology_grounding_to_dict(grounding),
        )
    if telemetry is not None:
        written[TELEMETRY_REPORT_BASENAME] = _write_pair(
            out_dir,
            TELEMETRY_REPORT_BASENAME,
            format_telemetry_report(telemetry),
            telemetry_to_dict(telemetry),
        )
    if preference_breakdown:
        written[PREFERENCE_REPORT_BASENAME] = _write_pair(
            out_dir,
            PREFERENCE_REPORT_BASENAME,
            format_preference_breakdown_report(preference_breakdown),
            preference_breakdown_to_dict(preference_breakdown),
        )
    if divide_breakdown:
        written[DIVIDE_REPORT_BASENAME] = _write_pair(
            out_dir,
            DIVIDE_REPORT_BASENAME,
            format_divide_breakdown_report(divide_breakdown),
            divide_breakdown_to_dict(divide_breakdown),
        )
    if context_fit_breakdown:
        from src.scripts.scenarios.metrics.context_fit_breakdown import (
            format_context_fit_breakdown,
        )

        written[CONTEXT_FIT_REPORT_BASENAME] = _write_pair(
            out_dir,
            CONTEXT_FIT_REPORT_BASENAME,
            format_context_fit_breakdown(context_fit_breakdown),
            context_fit_breakdown,
        )
    if weekly_gain and weekly_gain.get("weeks"):
        written[WEEKLY_GAIN_REPORT_BASENAME] = _write_pair(
            out_dir,
            WEEKLY_GAIN_REPORT_BASENAME,
            format_weekly_gain_report(weekly_gain),
            weekly_gain,
        )
    return written
