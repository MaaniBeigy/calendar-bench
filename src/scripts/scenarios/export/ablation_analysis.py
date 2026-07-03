"""OLS estimation of per-block main effects for prompt ablation runs.

Reads the per-variant evaluation sidecars and fits one ordinary
least-squares regression per response variable. The design matrix is
the `+1` / `-1` matrix of evaluated variants; an intercept column is
added so each slope is the additive change in the response when the
block is kept instead of ablated.

A positive `beta_k` means the block helps the response on average;
removing it costs `2 * beta_k` units.

Standard errors come from the closed form `s^2 * (X'X)^-1`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.scripts.scenarios.augmentation.prompts.augment_oneshot import ABLATABLE_BLOCKS

_GAIN_KEYS: tuple[str, ...] = (
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

# Map compact gain key to the field name inside
# `total_scheduling_gain.json::average_gains`.
_AVG_GAIN_FIELD: dict[str, str] = {
    "G_cov": "recommended_task_coverage",
    "G_cal": "task_event_and_task_task_temporal_relations",
    "G_pref": "user_preference_deviation",
    "G_disp": "intensive_task_dispersion",
    "G_merge": "semantic_coscheduling_merge",
    "G_spread": "recommended_task_spread",
    "G_divide": "dividable_task_split_reward",
    "G_context": "user_context_recommendation_fit",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantRecord:
    """One ablation cell read from disk.

    `augment_tokens` is the total token count for the augmentation
    stage, taken from
    `telemetry.json::by_stage.augmentation.tokens_total`.
    """

    variant_id: str
    ablated: frozenset[str]
    responses: dict[str, float | None]
    augment_tokens: float | None
    augment_wall_seconds: float | None
    augment_usd: float | None


@dataclass(frozen=True)
class BlockEffect:
    """OLS coefficient and SE for one block on one response."""

    block: str
    beta: float
    se: float


@dataclass(frozen=True)
class AblationFit:
    """Result of one OLS fit over a single response variable."""

    response: str
    intercept: float
    effects: list[BlockEffect]
    n_runs: int
    residual_std: float


# ---------------------------------------------------------------------------
# Disk reader
# ---------------------------------------------------------------------------


def _read_variants_index(method_dir: Path) -> list[tuple[str, frozenset[str]]]:
    """Read the design summary written by the CLI dispatch loop.

    Returns an empty list when the file is absent or unparseable.
    """
    index_path = Path(method_dir) / "ablation" / "variants.json"
    if not index_path.is_file():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[tuple[str, frozenset[str]]] = []
    for entry in payload.get("variants", []):
        vid = entry.get("variant_id")
        if not isinstance(vid, str):
            continue
        ablated = frozenset(entry.get("ablated", []))
        out.append((vid, ablated))
    return out


def _read_one_variant(
    method_dir: Path,
    variant_id: str,
    ablated: frozenset[str],
) -> VariantRecord | None:
    """Read the evaluation sidecars for one variant directory."""
    eval_dir = Path(method_dir) / "ablation" / variant_id / "evaluation"
    total_path = eval_dir / "total_scheduling_gain.json"
    if not total_path.is_file():
        return None
    try:
        total_payload = json.loads(total_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    responses: dict[str, float | None] = {
        "total": _as_float(total_payload.get("average_total_gain"))
    }
    avg = total_payload.get("average_gains") or {}
    for key, field in _AVG_GAIN_FIELD.items():
        responses[key] = _as_float(avg.get(field))

    tel_path = eval_dir / "telemetry.json"
    augment_tokens = augment_wall = augment_usd = None
    if tel_path.is_file():
        try:
            tel = json.loads(tel_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            tel = {}
        aug_stage = (tel.get("by_stage") or {}).get("augmentation") or {}
        augment_tokens = _as_float(aug_stage.get("tokens_total"))
        augment_wall = _as_float(aug_stage.get("wall_time_seconds_total"))
        augment_usd = _as_float(aug_stage.get("estimated_usd_total"))

    return VariantRecord(
        variant_id=variant_id,
        ablated=ablated,
        responses=responses,
        augment_tokens=augment_tokens,
        augment_wall_seconds=augment_wall,
        augment_usd=augment_usd,
    )


def _as_float(value) -> float | None:
    """Coerce a numeric-like value to float; return `None` on failure."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if not math.isnan(out) else None


def read_ablation_records(method_dir: Path) -> list[VariantRecord]:
    """Read every variant of one `(scenario, method)` pair from disk.

    Variants without `total_scheduling_gain.json` are silently
    skipped.
    """
    pairs = _read_variants_index(method_dir)
    out: list[VariantRecord] = []
    for variant_id, ablated in pairs:
        record = _read_one_variant(method_dir, variant_id, ablated)
        if record is not None:
            out.append(record)
    return out


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def _design_matrix(records: Sequence[VariantRecord]) -> np.ndarray:
    """Build the `(n_runs, n_blocks)` `+1` / `-1` matrix."""
    n = len(records)
    p = len(ABLATABLE_BLOCKS)
    matrix = np.full((n, p), 1, dtype=float)
    for i, rec in enumerate(records):
        for j, block in enumerate(ABLATABLE_BLOCKS):
            if block in rec.ablated:
                matrix[i, j] = -1.0
    return matrix


def _fit_one_response(
    response_name: str,
    y_raw: Sequence[float | None],
    x: np.ndarray,
) -> AblationFit | None:
    """Fit OLS for one response; return `None` when underdetermined."""
    mask = np.array([v is not None for v in y_raw], dtype=bool)
    if mask.sum() < x.shape[1] + 1:
        return None
    y = np.array([v for v, m in zip(y_raw, mask) if m], dtype=float)
    x_used = x[mask]
    intercept_col = np.ones((x_used.shape[0], 1), dtype=float)
    design = np.hstack([intercept_col, x_used])

    # Drop columns that are constant across the sample; those blocks
    # surface as zeros so the report keeps a complete row.
    keep_cols = [True] + [bool(np.ptp(x_used[:, j])) for j in range(x_used.shape[1])]
    if not any(keep_cols[1:]):
        return None
    # Drop later columns that are perfectly aliased (identical or sign-flip)
    # to an earlier kept column; with PB-24's always-kept tail two factors can
    # collapse to the same column under the fold, which would make the design
    # matrix rank-deficient.
    for j in range(x_used.shape[1]):
        if not keep_cols[j + 1]:
            continue
        col_j = x_used[:, j]
        for i in range(j):
            if not keep_cols[i + 1]:
                continue
            col_i = x_used[:, i]
            if np.array_equal(col_i, col_j) or np.array_equal(col_i, -col_j):
                keep_cols[j + 1] = False
                break
    design_kept = design[:, keep_cols]

    coefs, *_ = np.linalg.lstsq(design_kept, y, rcond=None)
    residuals = y - design_kept @ coefs
    dof = max(1, design_kept.shape[0] - design_kept.shape[1])
    sigma2 = float(residuals @ residuals) / dof

    try:
        xtx_inv = np.linalg.inv(design_kept.T @ design_kept)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None
    var_diag = sigma2 * np.diag(xtx_inv)
    var_diag = np.clip(var_diag, 0.0, None)
    se = np.sqrt(var_diag)

    intercept = float(coefs[0])
    full_betas: list[float] = []
    full_ses: list[float] = []
    kept_idx = 1
    for j in range(len(ABLATABLE_BLOCKS)):
        if keep_cols[j + 1]:
            full_betas.append(float(coefs[kept_idx]))
            full_ses.append(float(se[kept_idx]))
            kept_idx += 1
        else:
            full_betas.append(0.0)
            full_ses.append(0.0)

    effects = [
        BlockEffect(block=block, beta=full_betas[i], se=full_ses[i])
        for i, block in enumerate(ABLATABLE_BLOCKS)
    ]
    return AblationFit(
        response=response_name,
        intercept=intercept,
        effects=effects,
        n_runs=int(mask.sum()),
        residual_std=float(math.sqrt(sigma2)),
    )


def fit_all_responses(records: Sequence[VariantRecord]) -> dict[str, AblationFit]:
    """Fit one OLS per response variable; skip those without signal.

    Covers the eight gain components plus `total`, `augment_tokens`,
    `augment_wall_seconds`, `augment_usd`. Callers must guard with
    `.get(name)` because skipped responses are absent from the dict.
    """
    if not records:
        return {}
    x = _design_matrix(records)
    out: dict[str, AblationFit] = {}
    for key in _GAIN_KEYS:
        y_raw = [rec.responses.get(key) for rec in records]
        fit = _fit_one_response(key, y_raw, x)
        if fit is not None:
            out[key] = fit
    cost_responses: dict[str, list[float | None]] = {
        "augment_tokens": [rec.augment_tokens for rec in records],
        "augment_wall_seconds": [rec.augment_wall_seconds for rec in records],
        "augment_usd": [rec.augment_usd for rec in records],
    }
    for key, y_raw in cost_responses.items():
        fit = _fit_one_response(key, y_raw, x)
        if fit is not None:
            out[key] = fit
    return out


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationResult:
    """Everything the benchmark-report writer needs for one ablation run."""

    scenario_id: str
    method: str
    design_label: str
    records: list[VariantRecord]
    fits: dict[str, AblationFit]


def aggregate_ablation(
    experiment_dir: Path,
    scenario_id: str,
    method: str,
    design_label: str,
) -> AblationResult | None:
    """Read variants for one `(scenario, method)` pair and fit every response.

    Returns `None` when no variants index is present on disk.
    """
    method_dir = Path(experiment_dir) / "scenarios" / scenario_id / method
    records = read_ablation_records(method_dir)
    if not records:
        return None
    fits = fit_all_responses(records)
    return AblationResult(
        scenario_id=scenario_id,
        method=method,
        design_label=design_label,
        records=records,
        fits=fits,
    )
