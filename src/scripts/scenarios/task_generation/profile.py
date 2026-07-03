"""Render a person's characteristics into prompt-ready profile text."""

from __future__ import annotations

from src.scripts.persona.domain.persona import Person


def _format_value(value: str | bool | int | float) -> str:
    """Format one characteristic value for prompt text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _selected_items(person: Person, include: list[str] | None) -> list[tuple[str, str]]:
    """Return (axis, formatted value) pairs in declaration or `include` order."""
    chars = person.characteristics
    # Axes the person lacks are skipped silently; personas may declare
    # disjoint characteristic vocabularies.
    keys = list(chars) if include is None else [k for k in include if k in chars]
    return [(k, _format_value(chars[k])) for k in keys]


def profile_block(person: Person, include: list[str] | None = None) -> str:
    """Bullet-list profile block for multi-line prompt sections."""
    items = _selected_items(person, include)
    if not items:
        return "  - (none)"
    return "\n".join(f"  - {axis}: {value}" for axis, value in items)


def profile_summary(person: Person, include: list[str] | None = None) -> str:
    """One-line `axis=value; ...` profile for single-line prompt slots."""
    items = _selected_items(person, include)
    if not items:
        return "(none)"
    return "; ".join(f"{axis}={value}" for axis, value in items)


__all__ = ["profile_block", "profile_summary"]
