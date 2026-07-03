"""Check realized characteristic distributions against the declared shape."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from src.scripts.persona.config.schema import (
    BooleanDist,
    CategoricalDist,
    PersonaConfig,
    ScipyDist,
)
from src.scripts.persona.domain.persona import Person

DEFAULT_TOLERANCE = 1


@dataclass(frozen=True, slots=True)
class CharacteristicViolation:
    """One mismatch between a declared distribution and the realized population."""

    persona_id: str
    axis: str
    kind: str  # count_mismatch | out_of_range | missing | unexpected_label
    detail: str


def check_characteristic_distributions(
    persona_config: PersonaConfig,
    persons: Iterable[Person],
    *,
    tolerance: int = DEFAULT_TOLERANCE,
) -> list[CharacteristicViolation]:
    """Compare realized per-persona counts to declared distributions."""
    by_persona: dict[str, list[Person]] = {}
    for p in persons:
        by_persona.setdefault(p.persona_id, []).append(p)

    out: list[CharacteristicViolation] = []
    for persona in persona_config.personas:
        members = by_persona.get(persona.id, [])
        for axis, dist in persona.characteristics.items():
            values = [m.characteristics.get(axis) for m in members]
            if isinstance(dist, CategoricalDist):
                out.extend(
                    _check_categorical(persona.id, axis, dist.values, values, tolerance)
                )
            elif isinstance(dist, BooleanDist):
                weights = {True: dist.p_true, False: 1.0 - dist.p_true}
                out.extend(
                    _check_categorical(persona.id, axis, weights, values, tolerance)
                )
            elif isinstance(dist, ScipyDist):  # pragma: no branch
                out.extend(_check_scipy(persona.id, axis, dist, values))
    return out


def _check_categorical(
    persona_id: str,
    axis: str,
    weights: dict,
    realized: list,
    tolerance: int,
) -> list[CharacteristicViolation]:
    """Compare per-label realized count to `round(weight * n)`."""
    n = len(realized)
    out: list[CharacteristicViolation] = []
    counts = Counter(realized)
    if None in counts:
        out.append(
            CharacteristicViolation(
                persona_id=persona_id,
                axis=axis,
                kind="missing",
                detail=f"{counts[None]} of {n} instances missing the value",
            )
        )
    for label, weight in weights.items():
        expected = int(round(weight * n))
        actual = counts.get(label, 0)
        if abs(actual - expected) > tolerance:
            out.append(
                CharacteristicViolation(
                    persona_id=persona_id,
                    axis=axis,
                    kind="count_mismatch",
                    detail=(
                        f"label {label!r}: configured {weight:.2%} -> "
                        f"expected {expected}, realized {actual}"
                    ),
                )
            )
    # A realized value outside the declared label set never matches any
    # configured weight, so the loop above cannot see it; on a pinned
    # axis it would otherwise hide inside the count tolerance.
    for label, actual in counts.items():
        if label is None or label in weights:
            continue
        out.append(
            CharacteristicViolation(
                persona_id=persona_id,
                axis=axis,
                kind="unexpected_label",
                detail=(
                    f"label {label!r} not declared in the distribution; "
                    f"realized {actual}"
                ),
            )
        )
    return out


def _check_scipy(
    persona_id: str, axis: str, dist: ScipyDist, realized: list
) -> list[CharacteristicViolation]:
    """For scipy distributions, only check clip-range bounds when set."""
    if dist.clip is None:
        return []
    out: list[CharacteristicViolation] = []
    for value in realized:
        if value is None:
            out.append(
                CharacteristicViolation(
                    persona_id=persona_id,
                    axis=axis,
                    kind="missing",
                    detail="instance missing the value",
                )
            )
            continue
        if not (dist.clip.min <= value <= dist.clip.max):
            out.append(
                CharacteristicViolation(
                    persona_id=persona_id,
                    axis=axis,
                    kind="out_of_range",
                    detail=(
                        f"value {value} outside clip [{dist.clip.min}, "
                        f"{dist.clip.max}]"
                    ),
                )
            )
    return out


__all__ = [
    "CharacteristicViolation",
    "check_characteristic_distributions",
    "DEFAULT_TOLERANCE",
]
