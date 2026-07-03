"""Design-matrix library for prompt-component ablation runs.

Each design returns a list of `Variant` records. A `Variant` carries a
bitstring identifier in `ABLATABLE_BLOCKS` order (`1` for ablated, `0`
for kept) plus the matching frozenset of ablated block names.

Available designs:

    single             one run, no ablation.
    leave_one_out      n+1 runs: baseline plus one block-off run each.
    plackett_burman_12 12 runs. Resolution III (main effects aliased
                       with two-factor interactions). Tests only the
                       first 11 of the 13 ablatable blocks; columns 11
                       and 12 sit at +1 throughout.
    plackett_burman_24 24 runs. PB-12 folded with its sign mirror.
                       Resolution IV against the first 11 factors;
                       columns 11 (`split_dividable_tasks`) and 12
                       (`context_block`) co-alias under the fold, so
                       their main effects share one degree of freedom.
    plackett_burman_16 16 runs over a Hadamard H16 matrix. All 13
                       ablatable blocks vary independently (each
                       column is balanced 8/8 and pairwise orthogonal),
                       so `split_dividable_tasks` and `context_block`
                       each carry their own degree of freedom. Resolution
                       III (main effects aliased with 2fis); the cheapest
                       design that estimates all 13 main effects without
                       co-aliasing.
    full_factorial     Reserved. 2^N is infeasible at current per-call
                       cost; `build_design` raises `NotImplementedError`.
    custom             user-supplied `(id, [blocks])` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.scripts.scenarios.augmentation.prompts.augment_oneshot import ABLATABLE_BLOCKS

DESIGN_NAMES: tuple[str, ...] = (
    "single",
    "leave_one_out",
    "plackett_burman_12",
    "plackett_burman_16",
    "plackett_burman_24",
    "full_factorial",
    "custom",
)

# Plackett-Burman N=12. `+1` keeps the block, `-1` ablates it.
#
# Columns 0-10 are the standard PB-12 design (6 +1 and 6 -1 per column,
# all pairs orthogonal). Columns 11 (`split_dividable_tasks`) and 12
# (`context_block`) are held at +1 in PB-12 base; PB-12 has only 12 runs
# and cannot test a 12th or 13th factor freely. Under the fold (PB-24)
# both columns become [+1]*12 followed by [-1]*12, so PB-24 tests them
# as a single aliased pair (their inner product is 24, not 0, so they
# are co-aliased). PB-24 still estimates each as an unconfounded main
# effect against the first 11 factors because cols 0-10 sum to zero in
# both halves of the fold.
_PB12: tuple[tuple[int, ...], ...] = (
    (+1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1),
    (-1, +1, +1, -1, +1, +1, +1, -1, -1, -1, +1, +1, +1),
    (+1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1, +1, +1),
    (-1, +1, -1, +1, +1, -1, +1, +1, +1, -1, -1, +1, +1),
    (-1, -1, +1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1),
    (-1, -1, -1, +1, -1, +1, +1, -1, +1, +1, +1, +1, +1),
    (+1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, +1, +1),
    (+1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, +1),
    (+1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1),
    (-1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1),
    (+1, -1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1),
)


# Hadamard H16 (Sylvester construction): row i column j equals
# (-1) ** popcount(i & j). The 13 factor columns below take j in
# {1, 2, ..., 13}, skipping the all-ones intercept column. Each
# factor column is balanced (8 +1s, 8 -1s) and pairwise orthogonal,
# so every block in ABLATABLE_BLOCKS, including `context_block` and
# `split_dividable_tasks`, gets its own degree of freedom.
_PB16: tuple[tuple[int, ...], ...] = (
    (+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1),
    (-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1),
    (+1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1),
    (-1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1),
    (+1, +1, +1, -1, -1, -1, -1, +1, +1, +1, +1, -1, -1),
    (-1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1),
    (+1, -1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1),
    (-1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1),
    (+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1),
    (-1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1),
    (+1, -1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1),
    (-1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, -1, +1),
    (+1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1),
    (-1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1),
    (+1, -1, -1, -1, -1, +1, +1, -1, -1, +1, +1, +1, +1),
    (-1, -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, +1, -1),
)


@dataclass(frozen=True)
class Variant:
    """One prompt-ablation cell with a bitstring id (`1` = ablated, `0` = kept)."""

    variant_id: str
    ablated: frozenset[str]


def _row_to_variant(row: Sequence[int]) -> Variant:
    """Build a `Variant` from one row of a PB design matrix."""
    vid = "".join("1" if x == -1 else "0" for x in row)
    ablated = frozenset(b for b, x in zip(ABLATABLE_BLOCKS, row) if x == -1)
    return Variant(variant_id=vid, ablated=ablated)


def _baseline() -> Variant:
    """All-zeros variant; the unablated baseline."""
    return Variant(variant_id="0" * len(ABLATABLE_BLOCKS), ablated=frozenset())


def _plackett_burman(folded: bool) -> list[Variant]:
    """Return PB-12, optionally folded with its sign mirror."""
    rows: list[Sequence[int]] = list(_PB12)
    if folded:
        rows.extend(tuple(-x for x in row) for row in _PB12)
    seen: set[str] = set()
    out: list[Variant] = []
    for row in rows:
        v = _row_to_variant(row)
        if v.variant_id in seen:  # pragma: no cover - defensive
            continue
        seen.add(v.variant_id)
        out.append(v)
    return out


def _plackett_burman_16() -> list[Variant]:
    """Return the 16-run Hadamard PB-16 design over all 13 factor columns."""
    seen: set[str] = set()
    out: list[Variant] = []
    for row in _PB16:
        v = _row_to_variant(row)
        if v.variant_id in seen:  # pragma: no cover - defensive
            continue
        seen.add(v.variant_id)
        out.append(v)
    return out


def _leave_one_out() -> list[Variant]:
    """Baseline plus one variant per block with that block ablated."""
    variants: list[Variant] = [_baseline()]
    for i, name in enumerate(ABLATABLE_BLOCKS):
        bits = ["0"] * len(ABLATABLE_BLOCKS)
        bits[i] = "1"
        variants.append(Variant(variant_id="".join(bits), ablated=frozenset({name})))
    return variants


def _validate_custom_entry(entry: tuple[str, Iterable[str]]) -> Variant:
    """Coerce one `(id, blocks)` pair into a `Variant`."""
    vid, blocks = entry
    if not isinstance(vid, str) or not vid:
        raise ValueError(f"custom variant id must be a non-empty string; got {vid!r}")
    block_list = list(blocks)
    unknown = sorted(set(block_list) - set(ABLATABLE_BLOCKS))
    if unknown:
        raise ValueError(
            f"custom variant {vid!r} references unknown block(s) {unknown}; "
            f"valid blocks: {ABLATABLE_BLOCKS}"
        )
    return Variant(variant_id=vid, ablated=frozenset(block_list))


def build_design(
    name: str,
    *,
    fold: bool = True,
    custom: Sequence[tuple[str, Iterable[str]]] | None = None,
) -> list[Variant]:
    """Build the variant list for a named design.

    Args:
        name: one of `DESIGN_NAMES`.
        fold: applied only to `plackett_burman_12`. When `True`, the
            design is promoted to PB-24 (folded).
        custom: required when `name == "custom"`. Each pair is
            `(variant_id, iterable_of_block_names)`.

    Returns:
        Ordered list of `Variant` records.

    Raises:
        ValueError: unknown design name or invalid custom entry.
        NotImplementedError: `full_factorial` is reserved.
    """
    if name == "single":
        return [_baseline()]
    if name == "leave_one_out":
        return _leave_one_out()
    if name == "plackett_burman_12":
        return _plackett_burman(folded=fold)
    if name == "plackett_burman_16":
        return _plackett_burman_16()
    if name == "plackett_burman_24":
        return _plackett_burman(folded=True)
    if name == "full_factorial":
        raise NotImplementedError(
            "full_factorial is reserved but not executable: a 2^N grid over "
            "the current block count is infeasible at per-call LLM cost. Use "
            "plackett_burman_24 (24 runs, unbiased main effects) or "
            "custom for a targeted follow-up."
        )
    if name == "custom":
        if not custom:
            raise ValueError("design 'custom' requires a non-empty `custom` list")
        return [_validate_custom_entry(pair) for pair in custom]
    raise ValueError(f"unknown design name {name!r}; valid: {DESIGN_NAMES}")


def design_matrix(variants: Sequence[Variant]) -> list[list[int]]:
    """Return the `+1` / `-1` design matrix for `variants`.

    Rows are variants in order; columns are blocks in `ABLATABLE_BLOCKS`
    order. `+1` keeps the block, `-1` ablates it. The intercept column
    is not included.
    """
    out: list[list[int]] = []
    for v in variants:
        out.append([-1 if name in v.ablated else +1 for name in ABLATABLE_BLOCKS])
    return out


def _verify_pb12_balance() -> None:
    """Fail loud at import time if `_PB12` is mistranscribed."""
    if len(_PB12) != 12 or any(  # pragma: no cover - guards silent edits
        len(r) != 13 for r in _PB12
    ):
        raise RuntimeError("PB12 must be 12 rows of 13 columns")
    for c in range(11):
        pos = sum(1 for row in _PB12 if row[c] == +1)
        if pos != 6:  # pragma: no cover - guards silent edits
            raise RuntimeError(f"PB12 column {c} is unbalanced: {pos} +1s, expected 6")
    for c in (11, 12):
        col = [row[c] for row in _PB12]
        if any(x != +1 for x in col):  # pragma: no cover - guards silent edits
            raise RuntimeError(f"PB12 column {c} (always-kept) must be all +1")


_verify_pb12_balance()
