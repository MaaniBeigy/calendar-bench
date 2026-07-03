"""Turn a natural-language question into a structured GraphRAG query plan.

A plan separates the part of a question that vector search handles well (the
semantic gist, e.g. "running and jogging") from the structured constraints it
handles badly: a numeric threshold like MET over 7.3, an ontology to confine
the search to, or a HealthTasks difficulty level. The planner asks the LLM to
emit those constraints as JSON, then validates every field against the shared
registry so only known attributes and ontologies survive. Any failure (no LLM,
bad JSON, unknown fields) degrades to a plain semantic query over the original
question, so the planner can never make a question fail outright.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .retrieval_filters import (
    ATTRIBUTE_REGISTRY,
    ONTOLOGY_PREFIXES,
    AttributeFilter,
    attribute_registry_summary,
    make_attribute_filter,
    ontology_prefix_summary,
    resolve_branch,
    resolve_level,
    resolve_scope,
)

log = logging.getLogger("query_planner")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# HealthTasks branch / level filters only make sense for nodes under this
# prefix; a numeric filter on a HumanActivities-only attribute (metValue,
# activityCode) can never co-occur with them, so they are dropped when the
# scope does not include HealthTasks.
_HEALTH_PREFIX = ONTOLOGY_PREFIXES["HealthTasks"][0]


@dataclass(frozen=True)
class QueryPlan:
    """A validated retrieval plan derived from a natural-language question."""

    semantic_text: str
    include_prefixes: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()
    branches: tuple[str, ...] = ()
    levels: tuple[str, ...] = ()
    attribute_filters: tuple[AttributeFilter, ...] = ()
    instance_only: bool = False
    top_k: int | None = None
    extra: dict = field(default_factory=dict)


_PROMPT = """You translate a user's natural-language question into a JSON retrieval plan for a graph of health, activity, and nutrition ontologies.

Return ONLY a JSON object with these keys (omit a key when it does not apply):
- "semantic_text": the gist of the question to embed for vector search, with structural constraints (numbers, levels, ontology names, counts) stripped out. Keep the topical words such as "running jogging".
- "include_ontologies": list of ontology names to confine the search to.
- "exclude_ontologies": list of ontology names to keep out of the search.
- "branches": list of HealthTasks domain branches (HealthTasks ontology only): Nutrition, PhysicalActivity, MentalWellbeing.
- "levels": list of HealthTasks difficulty levels (HealthTasks ontology only), as integers 1-4.
- "attribute_filters": list of {{"name": <attribute>, "op": <one of > >= < <= = <>>, "value": <number or true/false>}}.
- "instance_only": true to restrict to authored task instances rather than ontology classes.
- "top_k": integer count of results the user asked for (e.g. "more than five" -> 6).

Available filter attributes (each lists the ontology it belongs to):
{attributes}

Available ontologies:
{ontologies}

Rules:
- Use only attribute names and ontology names from the lists above. If a constraint does not map to one of them, leave it out and fold it into semantic_text instead.
- Pick the single ontology that owns the attribute you filter on; do not also list a different ontology. "human activities" and metValue / activityCode mean the HumanActivities ontology.
- "branches" and "levels" describe HealthTasks only. Never use them with a HumanActivities attribute (metValue, activityCode) or with the HumanActivities ontology.
- Match the comparison operator to the wording: "over" / "more than" / "above" -> >, "at least" / "or more" -> >=, "under" / "less than" / "below" -> <, "at most" / "or fewer" -> <=.

Question: {question}

JSON plan:"""


def _content_of(response: object) -> str:
    """Return the text body of an LLM response, tolerating string returns."""
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response body."""
    match = _JSON_RE.search(text)
    if match is None:
        raise ValueError("no JSON object in planner response")
    return json.loads(match.group(0))


def _as_list(value: object) -> list:
    """Coerce a possibly-missing JSON field to a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _valid_filters(raw: object) -> tuple[AttributeFilter, ...]:
    """Validate the planner's attribute filters, dropping unusable ones."""
    out: list[AttributeFilter] = []
    for item in _as_list(raw):
        if not isinstance(item, dict):
            continue
        try:
            out.append(make_attribute_filter(item["name"], item["op"], item["value"]))
        except (KeyError, ValueError):
            log.debug("dropping unusable planner filter: %r", item)
    return tuple(out)


def _valid_scope(raw: object) -> tuple[str, ...]:
    """Resolve a scope list, dropping tokens that do not name an ontology."""
    out: list[str] = []
    for token in _as_list(raw):
        try:
            out.extend(resolve_scope([str(token)]))
        except ValueError:
            log.debug("dropping unknown ontology: %r", token)
    return tuple(dict.fromkeys(out))


def _valid_branches(raw: object) -> tuple[str, ...]:
    """Resolve branch names to class URIs, dropping unknown ones."""
    out: list[str] = []
    for token in _as_list(raw):
        try:
            out.append(resolve_branch(str(token)))
        except ValueError:
            log.debug("dropping unknown branch: %r", token)
    return tuple(dict.fromkeys(out))


def _valid_levels(raw: object) -> tuple[str, ...]:
    """Resolve level numbers to LevelN tokens, dropping unknown ones."""
    out: list[str] = []
    for token in _as_list(raw):
        try:
            out.append(resolve_level(token))
        except ValueError:
            log.debug("dropping unknown level: %r", token)
    return tuple(dict.fromkeys(out))


def _consistent_scope(
    filters: tuple[AttributeFilter, ...],
    include: tuple[str, ...],
    branches: tuple[str, ...],
    levels: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Reconcile a plan so its hard filters cannot cancel each other out.

    A hard attribute filter only matches nodes that carry the attribute, so
    the scope is confined to that attribute's home ontologies. HealthTasks
    branch and level filters are dropped when the resulting scope does not
    include HealthTasks, since they would otherwise empty the result (e.g. a
    metValue filter paired with a PhysicalActivityTask branch).
    """
    attr_prefixes = tuple(
        dict.fromkeys(
            prefix
            for flt in filters
            for prefix in ATTRIBUTE_REGISTRY[flt.name].prefixes
        )
    )
    if attr_prefixes:
        include = attr_prefixes
    if (branches or levels) and include and _HEALTH_PREFIX not in include:
        branches = ()
        levels = ()
    return include, branches, levels


def _valid_top_k(raw: object) -> int | None:
    """Clamp the planner's requested count to a sane range, or None."""
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return max(1, min(value, 100))


def plan_query(question: str, *, llm: object | None = None) -> QueryPlan:
    """Return a validated QueryPlan for a question, or a plain-text fallback.

    Builds an LLM if none is supplied. On any error (no LLM, bad JSON,
    unusable fields) it returns a plan that carries the original question as
    semantic text and no filters, so retrieval still runs.
    """
    fallback = QueryPlan(semantic_text=question)
    try:
        client = llm if llm is not None else _default_llm()
        prompt = _PROMPT.format(
            attributes=attribute_registry_summary(),
            ontologies=ontology_prefix_summary(),
            question=question,
        )
        data = _extract_json(_content_of(client.invoke(prompt)))
    except Exception as exc:  # noqa: BLE001 - planner must never break ask
        log.warning("query planner fell back to plain search: %s", exc)
        return fallback

    semantic = data.get("semantic_text")
    filters = _valid_filters(data.get("attribute_filters"))
    include = _valid_scope(data.get("include_ontologies"))
    branches = _valid_branches(data.get("branches"))
    levels = _valid_levels(data.get("levels"))
    include, branches, levels = _consistent_scope(filters, include, branches, levels)
    return QueryPlan(
        semantic_text=semantic if isinstance(semantic, str) and semantic else question,
        include_prefixes=include,
        exclude_prefixes=_valid_scope(data.get("exclude_ontologies")),
        branches=branches,
        levels=levels,
        attribute_filters=filters,
        instance_only=bool(data.get("instance_only", False)),
        top_k=_valid_top_k(data.get("top_k")),
    )


def _default_llm() -> object:
    """Build the env-default LLM for the planner."""
    from .llm import make_llm

    return make_llm()
