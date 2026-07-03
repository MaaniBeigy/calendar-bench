"""L_ctx = 1 - |Omega|^-1 sum over placed tasks of |Z^obs_k ∩ Z^rec_k| / |Z^rec_k|.

Omega is the set of context-linked placements ê_k. Z^rec_k is the recommended
categories the simulation actually realizes for the person: a category counts
only when the person's context trace generates an episode whose ontology IRI is
one of the task's linked IRIs for that category, and the category is exposed by
the observation budget. A task is never penalized for a context the persona
never enters. Z^obs_k is the realized subset: a category whose linked member is
present during ê_k (an overlapping episode carries one of the task's linked
IRIs). Placing a task during the wrong member of a recommended category (e.g.
with_family when the task links alone) does not earn credit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import rdflib

from src.scripts.scenarios.domain.calendar import CalendarTrace
from src.scripts.scenarios.domain.context import ContextEpisode
from src.scripts.scenarios.domain.solution import SchedulingSolution
from src.scripts.scenarios.domain.task import ScheduledTask

HB_TASK_PREFIX = "https://w3id.org/calendar-bench/health/task/"
HB_CONTEXT_LINK = rdflib.URIRef("https://w3id.org/calendar-bench/health/contextLink")

#: task_uri -> {category: frozenset(context_iri)}
ContextLinks = dict[str, dict[str, frozenset[str]]]


@dataclass(frozen=True, slots=True)
class ContextFitVerdict:
    """One per-task verdict for a placement ê_k (cohort sidecar row).

    `recommended_categories` are the linked categories the simulation realizes
    for this person (the denominator base); categories the persona never
    generates are excluded. `scored_categories` narrows that to the observation
    budget when one is supplied, and `fit` is the share of scored categories
    whose linked member overlaps ê_k. `fit_full` keeps the unscoped share over
    all realizable categories.
    """

    task_label: str
    task_uri: str
    recommended_categories: tuple[str, ...]
    observed_categories: tuple[str, ...]
    scored_categories: tuple[str, ...]
    overlapped_categories: tuple[str, ...]
    fit: float
    fit_full: float


def load_context_categories_by_iri(json_path: Path) -> dict[str, str]:
    """Read context_iris.json into `{context_iri: category}`."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    context = data.get("context", data)
    out: dict[str, str] = {}
    for category, members in context.items():
        for payload in members.values():
            iri = payload.get("iri")
            if iri:
                out.setdefault(iri, category)
    return out


def load_context_links_by_uri(
    healthtasks_ttl_path: Path, categories_by_iri: dict[str, str]
) -> ContextLinks:
    """Read HealthTasks TTL `hb:contextLink` IRIs into `{task_uri: {category: {iri}}}`.

    The Context dictionary `categories_by_iri` groups each linked IRI under its
    category. Tasks with no resolvable links are dropped, mirroring the
    empty-Z^rec convention.
    """
    graph = rdflib.Graph()
    graph.parse(str(healthtasks_ttl_path), format="turtle")
    staged: dict[str, dict[str, set[str]]] = {}
    for subject, _pred, obj in graph.triples((None, HB_CONTEXT_LINK, None)):
        category = categories_by_iri.get(str(obj))
        if category is None:
            continue
        staged.setdefault(str(subject), {}).setdefault(category, set()).add(str(obj))
    return {
        uri: {cat: frozenset(iris) for cat, iris in cats.items()}
        for uri, cats in staged.items()
    }


def _overlaps(task: ScheduledTask, episode: ContextEpisode) -> bool:
    """Return True iff I(task)=[task.start, task.end) meets I(ep)=[ep.start, ep.end) on the same day."""
    if task.date != episode.date:
        return False
    return (
        episode.start_minutes < task.end_minutes
        and task.start_minutes < episode.end_minutes
    )


def per_task_fit(
    task: ScheduledTask,
    contexts: Iterable[ContextEpisode],
    context_links_by_uri: ContextLinks,
    *,
    observed_categories: frozenset[str] | None = None,
) -> ContextFitVerdict | None:
    """Per-task fit verdict for ê_k; None when ê_k has no realizable Z^rec_k.

    A recommended category is realizable only when the person's context trace
    generates an episode whose IRI the task links for that category; categories
    the persona never enters drop out of the denominator entirely. Within the
    realizable set (narrowed to `observed_categories` when supplied), a category
    is realized when an overlapping episode carries one of the linked IRIs.
    """
    uri = task.task.ontology_uri or ""
    recommended = context_links_by_uri.get(uri)
    if not recommended:
        return None
    episodes = [ep for ep in contexts if ep.ontology_uri]
    trace_iris = {ep.ontology_uri for ep in episodes}
    realizable = {cat for cat, iris in recommended.items() if iris & trace_iris}
    if not realizable:
        return None
    if observed_categories is not None:
        scored_set = realizable & observed_categories
        if not scored_set:
            return None
    else:
        scored_set = realizable
    overlapping_iris = {ep.ontology_uri for ep in episodes if _overlaps(task, ep)}
    overlapping_full = sorted(
        cat for cat in realizable if recommended[cat] & overlapping_iris
    )
    overlapping_scored = sorted(
        cat for cat in scored_set if recommended[cat] & overlapping_iris
    )
    fit_full = len(overlapping_full) / len(realizable)
    fit_scored = len(overlapping_scored) / len(scored_set)
    return ContextFitVerdict(
        task_label=task.task.label,
        task_uri=uri,
        recommended_categories=tuple(sorted(realizable)),
        observed_categories=tuple(sorted(observed_categories or ())),
        scored_categories=tuple(sorted(scored_set)),
        overlapped_categories=tuple(overlapping_scored),
        fit=fit_scored,
        fit_full=fit_full,
    )


def compute_l_context_fit(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
    context_links_by_uri: ContextLinks,
    *,
    observed_categories: frozenset[str] | None = None,
) -> float | None:
    """Return L_ctx = 1 - |Omega|^-1 sum of per_task_fit over Omega (scorable ê_k)."""
    fits: list[float] = []
    for task in solution.scheduled:
        verdict = per_task_fit(
            task,
            calendar.contexts,
            context_links_by_uri,
            observed_categories=observed_categories,
        )
        if verdict is None:
            continue
        fits.append(verdict.fit)
    if not fits:
        return None
    mean_fit = sum(fits) / len(fits)
    return 1.0 - mean_fit


def collect_verdicts(
    solution: SchedulingSolution,
    calendar: CalendarTrace,
    context_links_by_uri: ContextLinks,
    *,
    observed_categories: frozenset[str] | None = None,
) -> list[ContextFitVerdict]:
    """Return one verdict per ê_k in Omega (a non-empty realizable Z^rec_k)."""
    verdicts: list[ContextFitVerdict] = []
    for task in solution.scheduled:
        verdict = per_task_fit(
            task,
            calendar.contexts,
            context_links_by_uri,
            observed_categories=observed_categories,
        )
        if verdict is not None:
            verdicts.append(verdict)
    return verdicts


__all__ = [
    "ContextFitVerdict",
    "ContextLinks",
    "collect_verdicts",
    "compute_l_context_fit",
    "load_context_categories_by_iri",
    "load_context_links_by_uri",
    "per_task_fit",
]
