"""Aggregated multi-scenario benchmark report (Markdown).

Walks every `scenarios/<scenario_id>/<method>/evaluation/` directory
under an experiment, reads the JSON sidecars, and writes
`<experiment_dir>/benchmark_report.md` with sections for run summary,
scheduling gain, stage cost and latency, ontology grounding,
preference breakdown, pairwise A/B, and the optional prompt-component
ablation.

Reads from disk only. Missing sidecars are skipped with a placeholder
cell so a partial run still produces a usable report.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import quote

from src.scripts.scenarios.export.report_writer import (
    cohort_learning_split,
    ols_slope,
    per_person_trajectory,
    weekly_gain_distribution,
)

# Sidecar basenames; kept in sync with `report_writer.py`.
TOTAL_REPORT_BASENAME = "total_scheduling_gain"
ONTOLOGY_REPORT_BASENAME = "ontology_grounding"
TELEMETRY_REPORT_BASENAME = "telemetry"
PREFERENCE_REPORT_BASENAME = "preference_breakdown"
DIVIDE_REPORT_BASENAME = "divide_breakdown"
WEEKLY_GAIN_REPORT_BASENAME = "weekly_scheduling_gain"

BENCHMARK_REPORT_BASENAME = "benchmark_report"

# Component long-name labels; kept aligned with `report_writer.py`.
_COMPONENT_LABELS: tuple[tuple[str, str], ...] = (
    ("recommended_task_coverage", "G_cov"),
    ("task_event_and_task_task_temporal_relations", "G_cal"),
    ("user_preference_deviation", "G_pref"),
    ("intensive_task_dispersion", "G_disp"),
    ("semantic_coscheduling_merge", "G_merge"),
    ("recommended_task_spread", "G_spread"),
    ("dividable_task_split_reward", "G_divide"),
    ("user_context_recommendation_fit", "G_context"),
)

_STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("task_generation", "Task generation"),
    ("augmentation", "Augmentation"),
    ("evaluation", "Evaluation"),
)


# ---------------------------------------------------------------------------
# Disk readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    """Read a JSON file; return `None` on missing or unparseable input."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_scenario_artifacts(eval_dir: Path) -> dict[str, dict | None]:
    """Read the five evaluation sidecars from one `evaluation/` directory.

    The returned dict always carries the five keys (`total`,
    `telemetry`, `grounding`, `preference`, `divide`).
    """
    return {
        "total": _read_json(eval_dir / f"{TOTAL_REPORT_BASENAME}.json"),
        "telemetry": _read_json(eval_dir / f"{TELEMETRY_REPORT_BASENAME}.json"),
        "grounding": _read_json(eval_dir / f"{ONTOLOGY_REPORT_BASENAME}.json"),
        "preference": _read_json(eval_dir / f"{PREFERENCE_REPORT_BASENAME}.json"),
        "divide": _read_json(eval_dir / f"{DIVIDE_REPORT_BASENAME}.json"),
        "weekly": _read_json(eval_dir / f"{WEEKLY_GAIN_REPORT_BASENAME}.json"),
    }


def discover_scenario_runs(
    experiment_dir: Path,
    scenario_method_pairs: Sequence[tuple[str, str]] | None = None,
) -> list[tuple[str, str, dict[str, dict | None]]]:
    """Enumerate `(scenario_id, method, artifacts)` tuples on disk.

    Layout: `<experiment_dir>/scenarios/<scenario_id>/<method>/evaluation/`.
    Pass `scenario_method_pairs` to restrict the scan; otherwise every
    `(scenario_id, method)` directory present is included in sorted
    order.
    """
    scenarios_root = experiment_dir / "scenarios"

    if scenario_method_pairs is not None:
        # Honour the caller's explicit list even when `scenarios/` is
        # missing; the per-pair read returns the all-`None` placeholder
        # so the markdown writer can render "n/a" cells.
        pairs = list(scenario_method_pairs)
    else:
        if not scenarios_root.is_dir():
            return []
        pairs = []
        for scenario_dir in sorted(scenarios_root.iterdir()):
            if not scenario_dir.is_dir():
                continue
            for method_dir in sorted(scenario_dir.iterdir()):
                if method_dir.is_dir():
                    pairs.append((scenario_dir.name, method_dir.name))

    out: list[tuple[str, str, dict[str, dict | None]]] = []
    for scenario_id, method in pairs:
        eval_dir = scenarios_root / scenario_id / method / "evaluation"
        out.append((scenario_id, method, read_scenario_artifacts(eval_dir)))
    return out


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

# shields.io accent for the experiment badge rendered under the report title.
_BADGE_COLOR = "22bfda"

# Trailing `(ACRONYM.YY.MM.DD)` code in `experiment_name`,
# e.g. `Progressive Healthy Lifestyle Challenge (PHLC.26.07.01)`.
_EXPERIMENT_CODE_RE = re.compile(
    r"^(?P<name>.*?)\s*"
    r"\((?P<acronym>[^.()\s]+)\.(?P<yy>\d{2})\.(?P<mm>\d{2})\.(?P<dd>\d{2})\)\s*$"
)


def _shields_escape(text: str) -> str:
    """Percent-encode a string for a shields.io static-badge path segment."""
    return quote(text.replace("_", "__").replace("-", "--"), safe="")


def _experiment_badge(experiment_name: str) -> str | None:
    """Build a shields.io badge image line from a coded experiment name.

    Labels the badge with the descriptive name plus the bare acronym and
    messages it with `20YY.MM`, e.g. `... Challenge (PHLC)` / `2026.07`.
    Returns `None` when the name lacks the trailing `(ACRONYM.YY.MM.DD)` code.
    """
    m = _EXPERIMENT_CODE_RE.match(experiment_name)
    if m is None:
        return None
    label = f"{m['name'].strip()} ({m['acronym']})"
    message = f"20{m['yy']}.{m['mm']}"
    url = (
        "https://img.shields.io/badge/"
        f"{_shields_escape(label)}-{_shields_escape(message)}-{_BADGE_COLOR}.svg"
    )
    return f"![{label} {message}]({url})"


def _gain_cell(value: float | int | None) -> str:
    """Render a gain value (float in [0, 1]) or `None` as a markdown cell."""
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _usd_cell(value: float | int | None) -> str:
    if isinstance(value, (int, float)):
        return f"${value:.4f}"
    return "n/a"


def _int_cell(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.0f}"
    return str(value)


def _seconds_cell(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}s"


def _pct_cell(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _delta_cell(value: float | int | None, baseline: float | int | None) -> str:
    """Render Δ vs baseline, or `n/a` when missing."""
    if value is None or baseline is None:
        return "n/a"
    delta = float(value) - float(baseline)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


# Paper-style short codes keyed by `(method, prompt_template)`.
# `llm_agent + augment_oneshot` = SAP (Single-Agent Prompt).
# `llm_agent + augment_agent`   = ITA (Iterative-Turn Agent).
_METHOD_ACRONYMS: dict[tuple[str, str | None], str] = {
    ("greedy", None): "GRD",
    ("llm_agent", "augment_oneshot"): "SAP",
    ("llm_agent", "augment_agent"): "ITA",
    ("rl", None): "RL",
    ("human_coach", None): "HUMAN",
}

# Legend gloss for each acronym, rendered only for the method families a
# run actually contains so the report never documents an unused method
# (e.g. the legacy `ITA` never appears unless a scenario runs it).
_ACRONYM_GLOSSARY: dict[str, str] = {
    "SAP": "`SAP` = Single-Agent Prompt (one-shot `llm_agent` + `augment_oneshot`)",
    "ITA": "`ITA` = Iterative-Turn Agent (`llm_agent` + `augment_agent`)",
    "GRD": "`GRD` = greedy",
    "PTIME": "`PTIME` = PTIME",
    "RL": "`RL` = RL",
    "HUMAN": "`HUMAN` = Human coach (handcrafted schedules)",
}

# Stage letter order T, A, E used in the compact label.
_STAGE_LETTER_ORDER: tuple[tuple[str, str], ...] = (
    ("task_generator", "T"),
    ("augmenter", "A"),
    ("evaluator", "E"),
)


def method_acronym(method: str, prompt_template: str | None = None) -> str:
    """Return the short paper-style code for one augmentation method.

    For `llm_agent`, the prompt template wins. Unknown combinations
    return the raw method name uppercased.
    """
    if (method, prompt_template) in _METHOD_ACRONYMS:
        return _METHOD_ACRONYMS[(method, prompt_template)]
    if (method, None) in _METHOD_ACRONYMS:
        return _METHOD_ACRONYMS[(method, None)]
    return method.upper().replace("_", "")


def resolve_method_acronym(scenarios_cfg, scenario_id: str, method: str) -> str:
    """Resolve the prompt template from `scenarios_cfg` then delegate
    to `method_acronym`."""
    if scenarios_cfg is None:
        return method_acronym(method, None)
    scenario = next((s for s in scenarios_cfg.scenarios if s.id == scenario_id), None)
    if scenario is None:
        return method_acronym(method, None)
    method_cfg = next((m for m in scenario.augmentation if m.method == method), None)
    if method_cfg is None:
        return method_acronym(method, None)
    if method == "llm_agent":
        return method_acronym(method, method_cfg.llm_agent.prompt_template)
    return method_acronym(method, None)


def compact_label(
    method_code: str,
    stage_models: dict[str, str],
    observation_contexts: list[str] | None = None,
) -> str:
    """Build the compact label `M: <code> + <stages>: <model> + ...`.

    Stages sharing the same model are grouped (e.g.
    `M: SAP + TAE: gpt-4o-mini`). Group order is deterministic. When
    `observation_contexts` is supplied, an `obs:` segment is appended
    so scenarios that differ only in their observation budget
    (e.g. blind vs informed runs sharing every model) remain
    distinguishable in the report.
    """
    letter_for_stage = dict(_STAGE_LETTER_ORDER)
    stage_index = {stage: i for i, (stage, _) in enumerate(_STAGE_LETTER_ORDER)}

    # model to [(stage_index, stage_letter), ...] in insertion order
    groups: dict[str, list[tuple[int, str]]] = {}
    for stage, _ in _STAGE_LETTER_ORDER:
        model = stage_models.get(stage, "env-default")
        groups.setdefault(model, []).append(
            (stage_index[stage], letter_for_stage[stage])
        )

    ordered = sorted(groups.items(), key=lambda kv: min(idx for idx, _ in kv[1]))
    segments = [f"M: {method_code}"]
    for model, members in ordered:
        letters = "".join(letter for _, letter in sorted(members))
        segments.append(f"{letters}: {model}")
    # Only emit the `obs:` segment when the augmenter is given a non-
    # empty context budget; a blind augmenter keeps the legacy model-only
    # form, while an informed one carries its category list to stay
    # distinguishable from the blind run with the same stage models.
    if observation_contexts:
        segments.append("obs: " + ",".join(observation_contexts))
    return " + ".join(segments)


def resolve_observation_contexts(
    scenarios_cfg, scenario_id: str, method: str
) -> list[str] | None:
    """Return the augmenter's visible context categories for one run.

    `None` means the scenario config could not be resolved (label falls
    back to the model-only form); `[]` means the augmenter is blind.
    """
    if scenarios_cfg is None:
        return None
    scenario = next((s for s in scenarios_cfg.scenarios if s.id == scenario_id), None)
    if scenario is None:
        return None
    method_cfg = next((m for m in scenario.augmentation if m.method == method), None)
    if method_cfg is None:
        return None
    obs = getattr(method_cfg, "observation", None)
    if obs is None:
        return []
    return list(getattr(obs, "contexts", []) or [])


def _compact_run_label(
    scenarios_cfg,
    scenario_id: str,
    method: str,
) -> str:
    """Compact label for one `(scenario_id, method)` run.

    A scenario that declares an explicit `label` uses it verbatim; the
    rest fall back to the auto-derived acronym / model / context label.
    """
    scenario = next((s for s in scenarios_cfg.scenarios if s.id == scenario_id), None)
    explicit = getattr(scenario, "label", "") if scenario is not None else ""
    if explicit:
        return explicit
    return compact_label(
        resolve_method_acronym(scenarios_cfg, scenario_id, method),
        resolve_stage_models(scenarios_cfg, scenario_id, method),
        resolve_observation_contexts(scenarios_cfg, scenario_id, method),
    )


def _scenario_label(scenario_id: str, method: str) -> str:
    """Fallback `<id> / <method>` label."""
    return f"{scenario_id} / {method}"


def _label_for_section(
    scenarios_cfg,
    scenario_id: str,
    method: str,
) -> str:
    """Compact label when `scenarios_cfg` is present, else the
    fallback `<id> / <method>` form."""
    if scenarios_cfg is None:
        return _scenario_label(scenario_id, method)
    return _compact_run_label(scenarios_cfg, scenario_id, method)


def _format_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Build a GitHub-flavoured markdown table as a list of lines."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


# ---------------------------------------------------------------------------
# Stage-model resolution (config-driven labels for the run-summary table)
# ---------------------------------------------------------------------------


def resolve_stage_models(
    scenarios_cfg, scenario_id: str, method: str
) -> dict[str, str]:
    """Return the `(task_generator, augmenter, evaluator)` model labels.

    Unset overrides render as `"env-default"`.
    """
    out = {
        "task_generator": "env-default",
        "augmenter": "env-default",
        "evaluator": "env-default",
    }
    if scenarios_cfg is None:
        return out
    scenario = next((s for s in scenarios_cfg.scenarios if s.id == scenario_id), None)
    if scenario is None:
        return out
    if scenario.task_generator_model is not None:
        out["task_generator"] = scenario.task_generator_model.model
    if scenario.evaluator_model is not None:
        out["evaluator"] = scenario.evaluator_model.model
    method_cfg = next((m for m in scenario.augmentation if m.method == method), None)
    if method_cfg is not None and method == "llm_agent":
        out["augmenter"] = method_cfg.llm_agent.model
    return out


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------


def _format_legend(method_codes: set[str]) -> list[str]:
    """`## Legend`: short codes for the method families this run contains."""
    entries = [text for code, text in _ACRONYM_GLOSSARY.items() if code in method_codes]
    entries += [
        f"`{code}` = {code}"
        for code in sorted(method_codes)
        if code not in _ACRONYM_GLOSSARY
    ]
    lines = [
        "## Legend",
        "",
        "Compact label conventions used in the tables below:",
        "",
        "* **M**: augmenter method. " + "; ".join(entries) + ".",
        "* **T**, **A**, **E**: Task generator, Augmenter, Evaluator "
        "stage models, in pipeline order.",
        "* Stages sharing the same model are grouped: `TAE: gpt-4o-mini` "
        "means all three stages use that model; "
        "`TE: gpt-4o-mini + A: gpt-4.1-mini` means only the augmenter "
        "differs. `env-default` means the stage inherits its model from "
        "`.env`.",
        "",
    ]
    return lines


def _format_run_summary(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Run summary`: compact label, pinned models, scenario id."""
    header = (
        "Configuration",
        "Task generator",
        "Augmenter",
        "Evaluator",
        "Scenario id",
    )
    rows: list[list[str]] = []
    for scenario_id, method, _ in runs:
        models = resolve_stage_models(scenarios_cfg, scenario_id, method)
        rows.append(
            [
                _label_for_section(scenarios_cfg, scenario_id, method),
                models["task_generator"],
                models["augmenter"],
                models["evaluator"],
                f"`{scenario_id} / {method}`",
            ]
        )
    return ["## Run summary", "", *_format_table(header, rows), ""]


def _format_scheduling_gain(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Scheduling gain`: total gain plus the seven components."""
    header = ["Configuration", "Avg total gain", "Scored / Empty"]
    header += [label for _, label in _COMPONENT_LABELS]
    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        total = artifacts.get("total") or {}
        avg_total = total.get("average_total_gain")
        scored = total.get("scored_persons", 0)
        empty = total.get("empty_plan_persons", 0)
        gains = total.get("average_gains") or {}
        row = [
            _label_for_section(scenarios_cfg, scenario_id, method),
            _gain_cell(avg_total),
            f"{scored} / {empty}",
        ]
        for key, _ in _COMPONENT_LABELS:
            row.append(_gain_cell(gains.get(key)))
        rows.append(row)
    return ["## Scheduling gain", "", *_format_table(header, rows), ""]


def _stage_extract(
    telemetry: dict | None, stage_key: str
) -> tuple[float, int, float | None]:
    """Pull `(wall_time, tokens, usd_or_None)` for one stage; defaults to zeros."""
    if not telemetry:
        return 0.0, 0, None
    by_stage = telemetry.get("by_stage") or {}
    stage = by_stage.get(stage_key) or {}
    return (
        float(stage.get("wall_time_seconds_total", 0.0)),
        int(stage.get("tokens_total", 0)),
        stage.get("estimated_usd_total"),
    )


def _format_stage_cost_latency(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Stage cost & latency`: wall time, tokens, and USD per stage.

    The augmentation stage carries an extra reasoning column: the
    provider-reported hidden reasoning tokens (a subset of the output
    tokens, so already inside the cost column) and their USD share.
    """
    header = ["Configuration"]
    for stage_key, label in _STAGE_LABELS:
        header += [
            f"{label} wall",
            f"{label} tokens",
            f"{label} cost",
        ]
        if stage_key == "augmentation":
            header.append(f"{label} reasoning (tokens / cost)")
    header.append("Total cost")

    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        telemetry = artifacts.get("telemetry") or {}
        row = [_label_for_section(scenarios_cfg, scenario_id, method)]
        total_usd: float | None = None
        for stage_key, _ in _STAGE_LABELS:
            wall, tokens, usd = _stage_extract(telemetry, stage_key)
            row += [_seconds_cell(wall), _int_cell(tokens), _usd_cell(usd)]
            if stage_key == "augmentation":
                stage = (telemetry.get("by_stage") or {}).get(stage_key) or {}
                reasoning = int(stage.get("reasoning_tokens_total") or 0)
                reasoning_usd = stage.get("estimated_reasoning_usd_total")
                row.append(
                    f"{reasoning} / {_usd_cell(reasoning_usd)}" if reasoning else "n/a"
                )
            if isinstance(usd, (int, float)):
                total_usd = (total_usd or 0.0) + float(usd)
        # Top-level `estimated_usd_total` covers any cost not attributed
        # to a single stage.
        if isinstance(telemetry.get("estimated_usd_total"), (int, float)):
            total_usd = telemetry["estimated_usd_total"]
        row.append(_usd_cell(total_usd))
        rows.append(row)
    return ["## Stage cost & latency", "", *_format_table(header, rows), ""]


def _format_ontology_grounding(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    header = [
        "Configuration",
        "Generated",
        "Grounded",
        "Verified",
        "Persons short-fetched",
        "Persons w/ unverified URI",
    ]
    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        grounding = artifacts.get("grounding") or {}
        total = grounding.get("total")
        grounded = grounding.get("grounded")
        verified = grounding.get("verified")
        ratio = grounding.get("ratio")
        verified_ratio = grounding.get("verified_ratio")
        rows.append(
            [
                _label_for_section(scenarios_cfg, scenario_id, method),
                _int_cell(total),
                (
                    f"{_int_cell(grounded)} ({_pct_cell(ratio)})"
                    if grounded is not None
                    else "n/a"
                ),
                (
                    f"{_int_cell(verified)} ({_pct_cell(verified_ratio)})"
                    if verified is not None
                    else "n/a"
                ),
                _int_cell(grounding.get("persons_short_fetched")),
                _int_cell(grounding.get("persons_with_unverified")),
            ]
        )
    return ["## Ontology grounding", "", *_format_table(header, rows), ""]


def _format_preference_breakdown(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Preference breakdown`: per-leg L_pref mean loss.

    Columns are the union of legs seen across runs.
    """
    leg_order: list[str] = []
    seen: set[str] = set()
    for _, _, artifacts in runs:
        breakdown = artifacts.get("preference") or {}
        for leg in breakdown:
            if leg not in seen:
                seen.add(leg)
                leg_order.append(leg)
    if not leg_order:
        return [
            "## Preference breakdown",
            "",
            "_No `preference_breakdown.json` sidecars were found._",
            "",
        ]

    header = ["Configuration", *leg_order]
    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        breakdown = artifacts.get("preference") or {}
        row = [_label_for_section(scenarios_cfg, scenario_id, method)]
        for leg in leg_order:
            stats = breakdown.get(leg) or {}
            mean_loss = stats.get("mean_loss")
            if mean_loss is None:
                row.append("n/a")
            else:
                applicable = stats.get("applicable_tasks", 0)
                row.append(f"{mean_loss:.4f} (n={applicable})")
        rows.append(row)
    return ["## Preference breakdown", "", *_format_table(header, rows), ""]


def _weekly_metric_table(
    runs: Sequence[tuple[str, str, dict]],
    metric_key: str,
    scenarios_cfg=None,
) -> list[str]:
    """Build one per-week table for `metric_key`, or `[]` when absent.

    `metric_key` is the field name inside each `weekly.weeks[*]` entry
    (`avg_weighted_gain`). Columns are the union of week indices, plus a
    trailing `Δ (last - first)`.
    """
    week_set: set[int] = set()
    for _, _, artifacts in runs:
        weekly = artifacts.get("weekly") or {}
        for row in weekly.get("weeks", []) or []:
            if row.get(metric_key) is not None:
                week_set.add(int(row.get("week_index", 0)))
    if not week_set:
        return []
    weeks_sorted = sorted(week_set)
    header = ["Configuration"]
    header += [f"Wk{w}" for w in weeks_sorted]
    header.append("Δ (last - first)")
    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        weekly = artifacts.get("weekly") or {}
        by_week: dict[int, float] = {
            int(r["week_index"]): float(r[metric_key])
            for r in (weekly.get("weeks") or [])
            if r.get(metric_key) is not None
        }
        row = [_label_for_section(scenarios_cfg, scenario_id, method)]
        for w in weeks_sorted:
            row.append(_gain_cell(by_week.get(w)))
        first = by_week.get(weeks_sorted[0])
        last = by_week.get(weeks_sorted[-1])
        row.append(_delta_cell(last, first))
        rows.append(row)
    return _format_table(header, rows)


def _format_weekly_gain(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Weekly scheduling gain`: per-week masked gain for each run.

    Each cell is the per-person mean masked scheduling gain for that ISO
    week; the trailing `Δ (last - first)` exposes the learning trajectory
    (positive = the augmenter improves across weeks; only the RL
    augmenter is expected to trend up). Returns `[]` when no run carries
    a `weekly` aggregate.
    """
    table = _weekly_metric_table(runs, "avg_weighted_gain", scenarios_cfg)
    if not table:
        return []
    return [
        "## Weekly scheduling gain",
        "",
        "_Per-person mean masked scheduling gain per ISO week. The `Δ` "
        "column is the learning signal (positive = improves across "
        "weeks); only the RL augmenter is expected to trend up._",
        "",
        *table,
        "",
    ]


def _format_per_person_distribution(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Per-person learning distribution`: per-week median plus the learner split."""
    week_set: set[int] = set()
    for _, _, artifacts in runs:
        by_person = (artifacts.get("weekly") or {}).get("by_person") or {}
        for row in weekly_gain_distribution(by_person):
            week_set.add(row["week_index"])
    if not week_set:
        return []
    weeks_sorted = sorted(week_set)
    header = ["Configuration"]
    header += [f"Wk{w}" for w in weeks_sorted]
    header += ["IQM (last)", "Δ median (last - first)", "conv/n", "flat/n", "decl/n"]
    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        by_person = (artifacts.get("weekly") or {}).get("by_person") or {}
        dist = {r["week_index"]: r for r in weekly_gain_distribution(by_person)}
        if not dist:
            continue
        split = cohort_learning_split(per_person_trajectory(by_person))
        row = [_label_for_section(scenarios_cfg, scenario_id, method)]
        for w in weeks_sorted:
            cell = dist.get(w)
            row.append(_gain_cell(cell["median_weighted_gain"] if cell else None))
        last = dist.get(weeks_sorted[-1])
        first = dist.get(weeks_sorted[0])
        row.append(_gain_cell(last["iqm_weighted_gain"] if last else None))
        row.append(
            _delta_cell(
                last["median_weighted_gain"] if last else None,
                first["median_weighted_gain"] if first else None,
            )
        )
        n = split["n"]
        row += [
            f"{split['converged']}/{n}",
            f"{split['flat']}/{n}",
            f"{split['declined']}/{n}",
        ]
        rows.append(row)
    return [
        "## Per-person learning distribution",
        "",
        "_Per-week median weighted gain across persons, IQM at the final "
        "week, and the per-person converged/flat/declined split counted by "
        "each person's first-to-last gain delta._",
        "",
        *_format_table(header, rows),
        "",
    ]


def _format_cross_augmenter_weekly(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Cross-augmenter weekly trajectory`: cohort-mean gain per week plus slope."""
    week_set: set[int] = set()
    for _, _, artifacts in runs:
        for row in (artifacts.get("weekly") or {}).get("weeks", []) or []:
            if row.get("avg_weighted_gain") is not None:
                week_set.add(int(row.get("week_index", 0)))
    if not week_set:
        return []
    weeks_sorted = sorted(week_set)
    header = ["Configuration"]
    header += [f"Wk{w}" for w in weeks_sorted]
    header += ["Δ (last - first)", "slope"]
    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        by_week = {
            int(r["week_index"]): float(r["avg_weighted_gain"])
            for r in (artifacts.get("weekly") or {}).get("weeks", []) or []
            if r.get("avg_weighted_gain") is not None
        }
        if not by_week:
            continue
        present = [w for w in weeks_sorted if w in by_week]
        row = [_label_for_section(scenarios_cfg, scenario_id, method)]
        for w in weeks_sorted:
            row.append(_gain_cell(by_week.get(w)))
        row.append(_delta_cell(by_week.get(present[-1]), by_week.get(present[0])))
        slope = ols_slope(present, [by_week[w] for w in present])
        row.append(f"{slope:+.4f}")
        rows.append(row)
    return [
        "## Cross-augmenter weekly trajectory",
        "",
        "_Cohort-mean weighted gain per ISO week for every run, with the OLS "
        "`slope` over weeks. A positive slope is improvement over time; the "
        "RL run should trend up while the non-learning baselines stay near "
        "zero._",
        "",
        *_format_table(header, rows),
        "",
        "_The per-week figure is a cohort mean; the per-person distribution "
        "above is the unit-of-analysis view, so a flat mean alone does not "
        "prove that no individual learned._",
        "",
    ]


def _format_divide_breakdown(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Divide breakdown`: per-run L_divide verdict counts.

    One row per `(scenario_id, method)` run; one column per verdict.
    Cells render as `count (pct%)`; runs without signal render as `n/a`.
    """
    verdict_order = (
        "divided_valid",
        "not_divided",
        "divided_invalid_pieces_too_long",
        "divided_invalid_sum_too_low",
        "divided_invalid_sum_too_high",
    )
    any_signal = any(
        (artifacts.get("divide") or {}).get("applicable_buckets")
        for _, _, artifacts in runs
    )
    if not any_signal:
        return [
            "## Divide breakdown",
            "",
            "_No `divide_breakdown.json` sidecars with dividable signal "
            "were found._",
            "",
        ]
    header = ["Configuration", "Buckets", *verdict_order]
    rows: list[list[str]] = []
    for scenario_id, method, artifacts in runs:
        breakdown = artifacts.get("divide") or {}
        buckets = int(breakdown.get("applicable_buckets", 0) or 0)
        row = [_label_for_section(scenarios_cfg, scenario_id, method)]
        if buckets == 0:
            row.append("n/a")
            row.extend(["n/a"] * len(verdict_order))
            rows.append(row)
            continue
        row.append(str(buckets))
        counts = breakdown.get("verdict_counts") or {}
        for verdict in verdict_order:
            n = int(counts.get(verdict, 0) or 0)
            pct = (n / buckets * 100.0) if buckets else 0.0
            row.append(f"{n} ({pct:.1f}%)")
        rows.append(row)
    return ["## Divide breakdown", "", *_format_table(header, rows), ""]


def _pairwise_section(
    baseline: tuple[str, str, dict],
    other: tuple[str, str, dict],
    scenarios_cfg=None,
) -> list[str]:
    """One pairwise A/B sub-section: baseline vs other, side-by-side."""
    base_id, base_method, base_art = baseline
    other_id, other_method, other_art = other

    base_label = _label_for_section(scenarios_cfg, base_id, base_method)
    other_label = _label_for_section(scenarios_cfg, other_id, other_method)

    base_models = resolve_stage_models(scenarios_cfg, base_id, base_method)
    other_models = resolve_stage_models(scenarios_cfg, other_id, other_method)

    diffs = [
        stage
        for stage in ("task_generator", "augmenter", "evaluator")
        if base_models[stage] != other_models[stage]
    ]
    axis_text = ", ".join(diffs) if diffs else "no model changes"
    heading = f"### `{other_label}`  vs  `{base_label}`  (axis: {axis_text})"

    base_total = base_art.get("total") or {}
    other_total = other_art.get("total") or {}
    base_gains = base_total.get("average_gains") or {}
    other_gains = other_total.get("average_gains") or {}

    rows: list[list[str]] = [
        [
            "Avg total gain",
            _gain_cell(base_total.get("average_total_gain")),
            _gain_cell(other_total.get("average_total_gain")),
            _delta_cell(
                other_total.get("average_total_gain"),
                base_total.get("average_total_gain"),
            ),
        ]
    ]
    for key, label in _COMPONENT_LABELS:
        rows.append(
            [
                label,
                _gain_cell(base_gains.get(key)),
                _gain_cell(other_gains.get(key)),
                _delta_cell(other_gains.get(key), base_gains.get(key)),
            ]
        )

    base_telem = base_art.get("telemetry") or {}
    other_telem = other_art.get("telemetry") or {}
    base_usd = base_telem.get("estimated_usd_total")
    other_usd = other_telem.get("estimated_usd_total")
    base_wall = base_telem.get("wall_time_seconds_total")
    other_wall = other_telem.get("wall_time_seconds_total")
    rows.append(
        [
            "Total cost",
            _usd_cell(base_usd),
            _usd_cell(other_usd),
            _delta_cell(other_usd, base_usd),
        ]
    )
    rows.append(
        [
            "Total wall time",
            _seconds_cell(base_wall),
            _seconds_cell(other_wall),
            _delta_cell(other_wall, base_wall),
        ]
    )

    header = ("Metric", base_label, other_label, "Δ (other - baseline)")
    return [heading, "", *_format_table(header, rows), ""]


def _format_pairwise(
    runs: Sequence[tuple[str, str, dict]],
    scenarios_cfg=None,
    baseline: tuple[str, str] | None = None,
) -> list[str]:
    """`## Pairwise A/B`: one sub-section per non-baseline run."""
    if len(runs) < 2:
        return [
            "## Pairwise A/B",
            "",
            "_Need at least two `(scenario_id, method)` runs for a pairwise comparison._",
            "",
        ]

    if baseline is not None:
        baseline_run = next((r for r in runs if (r[0], r[1]) == baseline), None)
    else:
        baseline_run = runs[0]
    if baseline_run is None:
        baseline_run = runs[0]
    others = [r for r in runs if r is not baseline_run]

    lines = ["## Pairwise A/B", ""]
    base_label = _label_for_section(scenarios_cfg, baseline_run[0], baseline_run[1])
    lines.append(
        f"_Baseline: `{base_label}`. Each Δ is `(other - baseline)`; "
        "positive = other does better, negative = other does worse._"
    )
    lines.append("")
    for other in others:
        lines += _pairwise_section(baseline_run, other, scenarios_cfg)
    return lines


# ---------------------------------------------------------------------------
# Prompt-component ablation section
# ---------------------------------------------------------------------------


def _format_prompt_ablation(
    experiment_dir: Path | None,
    ablation_runs: Sequence[tuple[str, str, str]],
    scenarios_cfg=None,
) -> list[str]:
    """`## Prompt component ablation`: OLS fits per `(scenario, method)`.

    `ablation_runs` is the list of `(scenario_id, method, variant_id)`
    triples to render. Returns `[]` when nothing on disk matches.
    """
    if not ablation_runs or experiment_dir is None:
        return []
    from src.scripts.scenarios.export.ablation_analysis import aggregate_ablation

    keys = sorted({(sid, method) for sid, method, _ in ablation_runs})
    results = []
    for sid, method in keys:
        design = _design_label_for(scenarios_cfg, sid, method)
        result = aggregate_ablation(experiment_dir, sid, method, design)
        if result is not None:
            results.append(result)
    if not results:
        return []

    lines: list[str] = [
        "## Prompt component ablation",
        "",
        "_OLS main effects of each `augment_oneshot` block on the "
        "scheduling-gain responses. `beta(y)` is the additive change "
        "in `y` per unit change of the block column "
        "(`+1` = block kept, `-1` = block ablated); positive `beta` "
        "means the block HELPS the response on average._",
        "",
    ]
    lines += _format_ablation_design_summary(results, scenarios_cfg)
    lines += _format_ablation_main_effects(results, scenarios_cfg)
    lines += _format_ablation_cost_effects(results, scenarios_cfg)
    lines += _format_ablation_variant_detail(results, scenarios_cfg)
    return lines


def _design_label_for(scenarios_cfg, scenario_id: str, method: str) -> str:
    """Resolve the design name declared in the YAML, or `unknown`."""
    if scenarios_cfg is None:
        return "unknown"
    for scenario in getattr(scenarios_cfg, "scenarios", []):
        if scenario.id != scenario_id:
            continue
        for method_cfg in scenario.augmentation:
            if method_cfg.method != method:
                continue
            ablation = getattr(method_cfg.llm_agent, "prompt_ablation", None)
            return ablation.design if ablation is not None else "single"
    return "unknown"


def _format_ablation_design_summary(results, scenarios_cfg) -> list[str]:
    """One row per ablation run: scenario, method, design name, n_runs."""
    header = ("Scenario / method", "Design", "Variants evaluated")
    rows = []
    for r in results:
        label = _label_for_section(scenarios_cfg, r.scenario_id, r.method)
        rows.append([label, r.design_label, str(len(r.records))])
    return [
        "### Design summary",
        "",
        *_format_table(header, rows),
        "",
    ]


def _format_ablation_main_effects(results, scenarios_cfg) -> list[str]:
    """Per-block main-effect table for every response variable."""
    from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
        ABLATABLE_BLOCKS,
    )

    out: list[str] = ["### Per-block main effects", ""]
    response_cols: tuple[str, ...] = (
        "total",
        "G_cov",
        "G_cal",
        "G_pref",
        "G_disp",
        "G_merge",
        "G_spread",
        "G_divide",
        "G_context",
    )
    for r in results:
        label = _label_for_section(scenarios_cfg, r.scenario_id, r.method)
        out += [f"#### `{label}`", ""]
        header = ("Block", *[f"beta({c})" for c in response_cols])
        rows = []
        for block in ABLATABLE_BLOCKS:
            row = [f"`{block}`"]
            for response in response_cols:
                fit = r.fits.get(response)
                if fit is None:
                    row.append("n/a")
                    continue
                effect = next((e for e in fit.effects if e.block == block), None)
                if effect is None or effect.se == 0.0 and effect.beta == 0.0:
                    row.append("n/a")
                else:
                    sign = "+" if effect.beta >= 0 else ""
                    row.append(f"{sign}{effect.beta:.4f} ± {effect.se:.4f}")
            rows.append(row)
        out += [*_format_table(header, rows), ""]
    return out


def _format_ablation_cost_effects(results, scenarios_cfg) -> list[str]:
    """Per-block main effect on the augment-stage cost responses."""
    from src.scripts.scenarios.augmentation.prompts.augment_oneshot import (
        ABLATABLE_BLOCKS,
    )

    out: list[str] = ["### Per-block cost effects", ""]
    response_cols = ("augment_tokens", "augment_wall_seconds", "augment_usd")
    for r in results:
        label = _label_for_section(scenarios_cfg, r.scenario_id, r.method)
        out += [f"#### `{label}`", ""]
        header = (
            "Block",
            "beta(augment tokens)",
            "beta(augment wall, s)",
            "beta(augment USD)",
        )
        rows = []
        for block in ABLATABLE_BLOCKS:
            row = [f"`{block}`"]
            for response in response_cols:
                fit = r.fits.get(response)
                if fit is None:
                    row.append("n/a")
                    continue
                effect = next((e for e in fit.effects if e.block == block), None)
                if effect is None or effect.se == 0.0 and effect.beta == 0.0:
                    row.append("n/a")
                else:
                    sign = "+" if effect.beta >= 0 else ""
                    row.append(f"{sign}{effect.beta:.4f}")
            rows.append(row)
        out += [*_format_table(header, rows), ""]
    return out


def _format_ablation_variant_detail(results, scenarios_cfg) -> list[str]:
    """Collapsible per-variant detail table."""
    out: list[str] = []
    for r in results:
        label = _label_for_section(scenarios_cfg, r.scenario_id, r.method)
        out.append(f"<details><summary>Variant detail: <code>{label}</code></summary>")
        out.append("")
        header = (
            "Variant id",
            "Ablated blocks",
            "Avg total gain",
            "Augment tokens",
            "Augment wall (s)",
            "Augment USD",
        )
        rows = []
        for record in r.records:
            ablated_text = ", ".join(sorted(record.ablated)) or "(baseline)"
            rows.append(
                [
                    f"`{record.variant_id}`",
                    ablated_text,
                    _gain_cell(record.responses.get("total")),
                    _int_cell(record.augment_tokens),
                    _seconds_cell(record.augment_wall_seconds),
                    _usd_cell(record.augment_usd),
                ]
            )
        out += _format_table(header, rows)
        out += ["", "</details>", ""]
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_benchmark_markdown(
    experiment_id: str,
    runs: Sequence[tuple[str, str, dict]],
    *,
    scenarios_cfg=None,
    experiment_name: str | None = None,
    baseline: tuple[str, str] | None = None,
    generated_at: _dt.datetime | None = None,
    experiment_dir: Path | None = None,
    ablation_runs: Sequence[tuple[str, str, str]] | None = None,
) -> str:
    """Assemble the Markdown report from already-loaded `runs`.

    Pure formatter. When `experiment_name` is set it titles the
    document; otherwise the bare `experiment_id` is used.
    """
    ts = (generated_at or _dt.datetime.now(_dt.timezone.utc)).isoformat(
        timespec="seconds"
    )
    if experiment_name:
        title = f"# Benchmark report: {experiment_name}"
        subtitle = f"_Experiment id: `{experiment_id}`_"
    else:
        title = f"# Benchmark report: `{experiment_id}`"
        subtitle = ""
    lines: list[str] = [title, ""]
    badge = _experiment_badge(experiment_name) if experiment_name else None
    if badge:
        lines += [badge, ""]
    if subtitle:
        lines += [subtitle, ""]
    lines += [
        f"_Generated {ts}_",
        "",
        f"_{len(runs)} `(scenario_id, method)` run(s) aggregated._",
        "",
    ]
    if not runs:
        lines.append(
            "_No evaluation sidecars found under "
            "`scenarios/<scenario_id>/<method>/evaluation/`._"
        )
        # An ablation run can still produce a useful section even
        # when no top-level scheduling evaluations were aggregated.
        if ablation_runs:
            lines += _format_prompt_ablation(
                experiment_dir, ablation_runs, scenarios_cfg
            )
        return "\n".join(lines)

    # When every run is missing its `total_scheduling_gain.json` the
    # tables below would just be a wall of n/a cells. Surface the cause
    # loudly so the reader notices before scrolling.
    if all(artifacts.get("total") is None for _, _, artifacts in runs):
        scenario_lines = "\n".join(
            f"  * `{_label_for_section(scenarios_cfg, sid, method)}`  "
            f"(`{sid} / {method}`)"
            for sid, method, _ in runs
        )
        lines += [
            "> **No evaluation data found for any of the listed runs.**",
            ">",
            "> Each `(scenario_id, method)` directory exists but its "
            "`evaluation/total_scheduling_gain.json` is missing. The most "
            "common cause is that `augment` produced zero output (look for "
            "`augment complete: 0/N persons in 0.0s` in the logs); re-run "
            "`augment` then `evaluate` for the affected scenarios.",
            "",
            "Affected runs:",
            scenario_lines,
            "",
        ]

    if scenarios_cfg is not None:
        method_codes = {
            resolve_method_acronym(scenarios_cfg, sid, method)
            for sid, method, _ in runs
        }
        lines += _format_legend(method_codes)
    lines += _format_run_summary(runs, scenarios_cfg)
    lines += _format_scheduling_gain(runs, scenarios_cfg)
    lines += _format_weekly_gain(runs, scenarios_cfg)
    lines += _format_per_person_distribution(runs, scenarios_cfg)
    lines += _format_cross_augmenter_weekly(runs, scenarios_cfg)
    lines += _format_stage_cost_latency(runs, scenarios_cfg)
    lines += _format_ontology_grounding(runs, scenarios_cfg)
    lines += _format_preference_breakdown(runs, scenarios_cfg)
    lines += _format_divide_breakdown(runs, scenarios_cfg)
    lines += _format_pairwise(runs, scenarios_cfg, baseline)
    if ablation_runs:
        lines += _format_prompt_ablation(experiment_dir, ablation_runs, scenarios_cfg)
    return "\n".join(lines)


def write_benchmark_report(
    experiment_dir: Path,
    experiment_id: str,
    *,
    scenarios_cfg=None,
    experiment_name: str | None = None,
    scenario_method_pairs: Iterable[tuple[str, str]] | None = None,
    baseline: tuple[str, str] | None = None,
    ablation_runs: Iterable[tuple[str, str, str]] | None = None,
) -> Path:
    """Build the Markdown and write it to
    `<experiment_dir>/benchmark_report.md`.

    `experiment_name` overrides the bare `experiment_id` in the title
    when set. `ablation_runs` appends the prompt-component ablation
    section. The file is overwritten on every call.
    """
    pairs = list(scenario_method_pairs) if scenario_method_pairs else None
    runs = discover_scenario_runs(experiment_dir, pairs)
    md = build_benchmark_markdown(
        experiment_id,
        runs,
        scenarios_cfg=scenarios_cfg,
        experiment_name=experiment_name,
        baseline=baseline,
        experiment_dir=experiment_dir,
        ablation_runs=list(ablation_runs) if ablation_runs else None,
    )
    target = experiment_dir / f"{BENCHMARK_REPORT_BASENAME}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(md, encoding="utf-8")
    return target


__all__ = [
    "BENCHMARK_REPORT_BASENAME",
    "ONTOLOGY_REPORT_BASENAME",
    "PREFERENCE_REPORT_BASENAME",
    "TELEMETRY_REPORT_BASENAME",
    "TOTAL_REPORT_BASENAME",
    "compact_label",
    "method_acronym",
    "resolve_method_acronym",
    "build_benchmark_markdown",
    "discover_scenario_runs",
    "read_scenario_artifacts",
    "resolve_stage_models",
    "write_benchmark_report",
]
