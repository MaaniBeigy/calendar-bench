"""Generic evaluator for event `requires:` predicates over person characteristics."""

from __future__ import annotations

from src.scripts.persona.config.schema import RolePredicate
from src.scripts.persona.domain.persona import Person


def matches_requires(person: Person, requires: dict[str, RolePredicate]) -> bool:
    """Return True iff every predicate in `requires` matches the person."""
    for axis, predicate in requires.items():
        value = person.characteristics.get(axis)
        if value is None:
            return False
        if not _predicate_matches(predicate, value):
            return False
    return True


def _predicate_matches(predicate: RolePredicate, value: object) -> bool:
    """Check one predicate against one resolved characteristic value."""
    if predicate.in_ is not None and not _membership(value, predicate.in_):
        return False
    if predicate.eq is not None and not _equal(value, predicate.eq):
        return False
    if predicate.ge is not None and not _comparable(value, predicate.ge, "ge"):
        return False
    if predicate.le is not None and not _comparable(value, predicate.le, "le"):
        return False
    if predicate.gt is not None and not _comparable(value, predicate.gt, "gt"):
        return False
    if predicate.lt is not None and not _comparable(value, predicate.lt, "lt"):
        return False
    return True


def _membership(value: object, allowed: list) -> bool:
    """`value in allowed` with a case-insensitive string fallback for bools."""
    if value in allowed:
        return True
    return _stringify(value) in {_stringify(v) for v in allowed}


def _equal(value: object, target: object) -> bool:
    """Equality with a case-insensitive string fallback for bools / scalars."""
    if value == target:
        return True
    return _stringify(value) == _stringify(target)


def _stringify(value: object) -> str:
    """Lower-case stringify so Python `True` and YAML `"true"` compare equal."""
    return str(value).lower()


def _comparable(value: object, bound: float | int, op: str) -> bool:
    """Apply a numeric comparison; booleans and strings always fail."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if op == "ge":
        return value >= bound
    if op == "le":
        return value <= bound
    if op == "gt":
        return value > bound
    return value < bound  # op == "lt"


__all__ = ["matches_requires"]
