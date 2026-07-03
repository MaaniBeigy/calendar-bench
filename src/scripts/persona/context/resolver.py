"""Per-person enabled-context resolution.

A context member is enabled for a given person only when every
predicate in its `requires:` block matches the person's resolved
characteristics. The resolver returns the gated subset that the
planner should attempt to place.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.scripts.persona.config.schema import ContextCategory, ContextMember
from src.scripts.persona.context.schema import MUTUALLY_EXCLUSIVE_DEFAULTS
from src.scripts.persona.domain.persona import Person
from src.scripts.persona.sampling.gating import matches_requires


@dataclass(frozen=True, slots=True)
class ResolvedCategory:
    """Per-person enabled members of one context category."""

    name: str
    mutually_exclusive: bool
    members: dict[str, ContextMember]


def resolve_for_person(
    person: Person,
    categories: dict[str, ContextCategory],
) -> dict[str, ResolvedCategory]:
    """Gate context members through their `requires:` predicates."""
    out: dict[str, ResolvedCategory] = {}
    for cat_name, category in categories.items():
        enabled: dict[str, ContextMember] = {}
        for member_name, member in category.members.items():
            if matches_requires(person, member.requires):
                enabled[member_name] = member
        if not enabled:
            continue
        out[cat_name] = ResolvedCategory(
            name=cat_name,
            mutually_exclusive=_resolve_exclusivity(cat_name, category),
            members=enabled,
        )
    return out


def _resolve_exclusivity(name: str, category: ContextCategory) -> bool:
    """Honour an explicit YAML flag; otherwise use the per-category default."""
    if category.mutually_exclusive is not None:
        return category.mutually_exclusive
    return MUTUALLY_EXCLUSIVE_DEFAULTS.get(name, False)


__all__ = ["ResolvedCategory", "resolve_for_person"]
