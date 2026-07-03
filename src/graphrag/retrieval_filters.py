"""Shared registry and safe Cypher builders for attribute and scope filters.

The natural-language query planner and the explicit CLI flags both compile
down to the same primitives defined here: an allow-listed numeric or boolean
attribute filter, an ontology scope resolved from a key or a raw URI prefix,
and the HealthTasks branch / difficulty-level tokens. Keeping the allow-list
and the Cypher rendering in one place means only validated property names ever
reach a query string, so neither front-end can smuggle Cypher into the graph.

n10s stores every literal as a 1-element array, so each predicate unwraps the
stored value with head() before comparing. Ordering comparisons coerce with
toFloat so a stray string-typed literal does not silently drop the row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordering comparisons coerce with toFloat; equality comparisons read the
# unwrapped value directly so booleans and integers match without coercion.
_ORDERING_OPS = frozenset({">", ">=", "<", "<="})
_EQUALITY_OPS = frozenset({"=", "<>"})
ATTRIBUTE_OPS = _ORDERING_OPS | _EQUALITY_OPS

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FILTER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|<>|>|<|=)\s*(.+?)\s*$")


@dataclass(frozen=True)
class AttributeSpec:
    """One filterable node attribute, with its kind and home ontologies."""

    name: str
    kind: str  # "numeric" or "boolean"
    prefixes: tuple[str, ...]
    note: str


_HA = "https://w3id.org/calendar-bench/human-activities/"
_HEALTH = "https://w3id.org/calendar-bench/health/"
_EBP = "http://www.semanticweb.org/tswheeler/ontologies/2016/3/EmpowerBP#"
_CHEBI = "http://purl.obolibrary.org/obo/CHEBI_"

# Allow-list of node attributes that may be filtered or surfaced. Only names
# present here can reach a query string, so the registry doubles as the
# injection guard for both the planner and the CLI flags.
ATTRIBUTE_REGISTRY: dict[str, AttributeSpec] = {
    "metValue": AttributeSpec(
        "metValue", "numeric", (_HA,), "metabolic equivalent of task, 1.0-23.0"
    ),
    "activityCode": AttributeSpec(
        "activityCode", "numeric", (_HA,), "Compendium activity code"
    ),
    "estimatedDurationMinutes": AttributeSpec(
        "estimatedDurationMinutes", "numeric", (_HEALTH,), "task duration, 2-120 min"
    ),
    "canRepeat": AttributeSpec(
        "canRepeat", "numeric", (_HEALTH,), "weekly repeat count, 4-52"
    ),
    "minimumUserAge": AttributeSpec(
        "minimumUserAge", "numeric", (_EBP,), "lower age bound, 3-65"
    ),
    "maximumUserAge": AttributeSpec(
        "maximumUserAge", "numeric", (_EBP,), "upper age bound, 64-120"
    ),
    "appropriateForLocation": AttributeSpec(
        "appropriateForLocation", "numeric", (_EBP,), "location fit score, 0-6"
    ),
    "appropriateForSocialOpportunity": AttributeSpec(
        "appropriateForSocialOpportunity", "numeric", (_EBP,), "social fit score, 3-6"
    ),
    "appropriateForPhysicalCapability": AttributeSpec(
        "appropriateForPhysicalCapability",
        "numeric",
        (_EBP,),
        "physical capability fit score, 3-6",
    ),
    "appropriateForPhysicalOpportunity": AttributeSpec(
        "appropriateForPhysicalOpportunity",
        "numeric",
        (_EBP,),
        "physical opportunity fit score, 3-6",
    ),
    "appropriateForPsychologicalCapability": AttributeSpec(
        "appropriateForPsychologicalCapability",
        "numeric",
        (_EBP,),
        "psychological capability fit score, 3-6",
    ),
    "appropriateForAutomaticMotivation": AttributeSpec(
        "appropriateForAutomaticMotivation",
        "numeric",
        (_EBP,),
        "automatic motivation fit score, 3-9",
    ),
    "appropriateForReflectiveMotivation": AttributeSpec(
        "appropriateForReflectiveMotivation",
        "numeric",
        (_EBP,),
        "reflective motivation fit score, 3-9",
    ),
    "mass": AttributeSpec("mass", "numeric", (_CHEBI,), "molecular mass"),
    "monoisotopic_mass": AttributeSpec(
        "monoisotopic_mass", "numeric", (_CHEBI,), "monoisotopic molecular mass"
    ),
    "charge": AttributeSpec("charge", "numeric", (_CHEBI,), "net molecular charge"),
    "isConcurrent": AttributeSpec(
        "isConcurrent", "boolean", (_HEALTH,), "task may run alongside another"
    ),
    "isDividable": AttributeSpec(
        "isDividable", "boolean", (_HEALTH,), "task may be split into sittings"
    ),
}

# Friendly ontology names mapped to the URI prefix that identifies their nodes
# in the shared graph. Used by the include / exclude scope flags so a question
# can be confined to (or kept clear of) a named ontology. A raw URI prefix is
# also accepted directly, so callers are never limited to these keys.
ONTOLOGY_PREFIXES: dict[str, tuple[str, ...]] = {
    "HumanActivities": (_HA,),
    "HealthTasks": (_HEALTH,),
    "Context": ("https://w3id.org/calendar-bench/context",),
    "EmpowerBP": (_EBP,),
    "OCHV": ("http://sbmi.uth.tmc.edu/ontology/ochv#",),
    "FOODON": ("http://purl.obolibrary.org/obo/FOODON_",),
    "CHEBI": (_CHEBI,),
    "OBO": ("http://purl.obolibrary.org/obo/",),
    "EFO": ("http://www.ebi.ac.uk/efo/EFO_",),
    "BehaviourChange": ("http://humanbehaviourchange.org/ontology/",),
}

# HealthTasks domain branches, keyed by the friendly name, mapped to the OWL
# domain-class URI the retriever's branch filter walks up to.
HEALTH_BRANCHES: dict[str, str] = {
    "Nutrition": _HEALTH + "NutritionTask",
    "PhysicalActivity": _HEALTH + "PhysicalActivityTask",
    "MentalWellbeing": _HEALTH + "MentalWellbeingTask",
}

# HealthTasks difficulty levels live in the rdf:type local name as LevelN.
LEVEL_TOKENS: tuple[str, ...] = ("Level1", "Level2", "Level3", "Level4")


@dataclass(frozen=True)
class AttributeFilter:
    """A validated predicate over one registry attribute."""

    name: str
    op: str
    value: float | bool


def _coerce_value(spec: AttributeSpec, raw: str) -> float | bool:
    """Parse a raw value string into the type the attribute expects."""
    if spec.kind == "boolean":
        token = raw.strip().lower()
        if token in ("true", "1", "yes"):
            return True
        if token in ("false", "0", "no"):
            return False
        raise ValueError(f"{spec.name} is boolean; expected true/false, got {raw!r}")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{spec.name} is numeric; got non-numeric {raw!r}") from exc


def make_attribute_filter(name: str, op: str, value: object) -> AttributeFilter:
    """Validate a name/op/value triple against the registry into a filter."""
    spec = ATTRIBUTE_REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"unknown filter attribute {name!r}")
    if op not in ATTRIBUTE_OPS:
        raise ValueError(f"unknown filter operator {op!r}")
    if spec.kind == "boolean" and op not in _EQUALITY_OPS:
        raise ValueError(f"{name} is boolean; only = and <> are allowed, not {op!r}")
    coerced = _coerce_value(spec, value if isinstance(value, str) else str(value))
    return AttributeFilter(name=name, op=op, value=coerced)


def parse_attribute_filter(expr: str) -> AttributeFilter:
    """Parse a flag expression like 'metValue > 7.3' or 'isConcurrent = true'."""
    match = _FILTER_RE.match(expr)
    if match is None:
        raise ValueError(
            f"could not parse filter {expr!r}; expected 'NAME OP VALUE' "
            f"such as 'metValue > 7.3'"
        )
    name, op, raw_value = match.groups()
    return make_attribute_filter(name, op, raw_value)


def cypher_predicate(flt: AttributeFilter) -> str:
    """Render a validated filter into a safe Cypher WHERE fragment."""
    if not _IDENT_RE.match(flt.name) or flt.name not in ATTRIBUTE_REGISTRY:
        raise ValueError(f"unsafe filter attribute {flt.name!r}")
    if flt.op not in ATTRIBUTE_OPS:
        raise ValueError(f"unsafe filter operator {flt.op!r}")
    if isinstance(flt.value, bool):
        literal = "true" if flt.value else "false"
        return f"head(node.{flt.name}) {flt.op} {literal}"
    return f"toFloat(head(node.{flt.name})) {flt.op} {float(flt.value)!r}"


def resolve_scope(tokens: list[str] | tuple[str, ...]) -> list[str]:
    """Resolve ontology keys or raw URI prefixes to a deduped prefix list."""
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token.startswith(("http://", "https://")):
            prefixes: tuple[str, ...] = (token,)
        elif token in ONTOLOGY_PREFIXES:
            prefixes = ONTOLOGY_PREFIXES[token]
        else:
            raise ValueError(
                f"unknown ontology {token!r}; use one of "
                f"{sorted(ONTOLOGY_PREFIXES)} or a raw http(s) URI prefix"
            )
        for prefix in prefixes:
            if prefix not in seen:
                seen.add(prefix)
                out.append(prefix)
    return out


def resolve_branch(name: str) -> str:
    """Resolve a HealthTasks branch name (or pass a raw URI) to its class URI."""
    if name.startswith(("http://", "https://")):
        return name
    if name in HEALTH_BRANCHES:
        return HEALTH_BRANCHES[name]
    raise ValueError(
        f"unknown branch {name!r}; use one of {sorted(HEALTH_BRANCHES)} or a class URI"
    )


def resolve_level(token: str | int) -> str:
    """Resolve a level number (1-4) or a 'LevelN' token to a 'LevelN' token."""
    text = str(token).strip()
    level = text if text.startswith("Level") else f"Level{text}"
    if level not in LEVEL_TOKENS:
        raise ValueError(f"unknown level {token!r}; expected one of {LEVEL_TOKENS}")
    return level


def _ontology_for_prefix(prefix: str) -> str:
    """Return the friendly ontology name owning a prefix, or the prefix."""
    for key, prefixes in ONTOLOGY_PREFIXES.items():
        if prefix in prefixes:
            return key
    return prefix


def attribute_registry_summary() -> str:
    """Return a one-line-per-attribute summary for the planner prompt."""
    lines = [
        f"- {spec.name} ({spec.kind}, {_ontology_for_prefix(spec.prefixes[0])}): "
        f"{spec.note}"
        for spec in ATTRIBUTE_REGISTRY.values()
    ]
    return "\n".join(lines)


def ontology_prefix_summary() -> str:
    """Return a one-line-per-ontology summary for the planner prompt."""
    lines = [
        f"- {key}: {', '.join(prefixes)}" for key, prefixes in ONTOLOGY_PREFIXES.items()
    ]
    return "\n".join(lines)
