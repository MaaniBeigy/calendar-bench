"""Derive the source ontology of a RecommendedTask from its `ontology_uri`.

Replaces the dead `source_ontology` field on :class:`RecommendedTask`: the LLM
was never asked for it (it never appeared in
`task_generation/prompt_templates.py`), the parser defaulted it to
`None`, and every shipped output file carried
`"source_ontology": null`.  The two real consumers
(`metrics/allen.py`'s `ResolvedActivity.domain` and
`metrics/intensity_resolver.py`'s branch-local hint) silently fell back
to deriving the source from the URI anyway; so we promote that derivation
to a first-class one-line utility.

Prefix constants are imported, not duplicated, from the modules where they
already live (`met_lookup.HUMAN_ACTIVITIES_PREFIX` and
`retriever.HEALTH_TASK_INSTANCE_PREFIX`).
"""

from __future__ import annotations

from src.graphrag.retriever import HEALTH_TASK_INSTANCE_PREFIX
from src.scripts.scenarios.metrics.met_lookup import HUMAN_ACTIVITIES_PREFIX


def infer_source(uri: str | None) -> str | None:
    """Return `"HealthTasks"` / `"HumanActivities"` / `None` for a URI.

    Returns `None` for an unknown prefix or a `None` input.  The match
    is a prefix check against the two canonical W3ID namespaces shipped
    with this project; URIs that do not start with either prefix are
    treated as unknown.
    """
    if uri is None:
        return None
    if uri.startswith(HEALTH_TASK_INSTANCE_PREFIX):
        return "HealthTasks"
    if uri.startswith(HUMAN_ACTIVITIES_PREFIX):
        return "HumanActivities"
    return None
