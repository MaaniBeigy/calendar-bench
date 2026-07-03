"""Three-tier ontology-aware `(recommended_task, catalog_event)` matcher.

Tiers (highest-weight wins per pair, tier-1 locks the rest):

* Tier 1 (`weight = 1.0`): literal IRI match between the task and the
  event in either ontology.
* Tier 2 (`weight = ancestor_discount × decay**hops`): bounded
  SUBCLASS-OF traversal in Neo4j.
* Tier 3 (`weight = σ ≥ threshold`): `SemanticCompatibility` score on
  the task's rich label vs the event's name + category.

Results are cached via `PreferenceCache` (Redis HASH per `(experiment,
scenario)`); stale entries are detected via content-hash mismatch and
silently recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.scripts.persona.config.schema import EventDefinition
from src.scripts.scenarios.domain.task import RecommendedTask
from src.scripts.scenarios.metrics.preference_cache import (
    CachedMapping,
    PreferenceCache,
    _content_hash,
)
from src.scripts.scenarios.metrics.semantic import SemanticCompatibility

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MappedEvent:
    """One `(event_def, weight, source_tier, evidence)` resolution."""

    event_name: str
    weight: float
    # One of: literal, cross_literal, cross_ancestor, ancestor, semantic, none.
    source_tier: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MapperConfig:
    """Tunable knobs for the four-tier matcher."""

    ancestor_discount: float = 0.6
    ancestor_max_hops: int = 2
    ancestor_decay: float = 0.7
    sigma_threshold: float = 0.70
    cross_ancestor_discount: float = 0.6
    cross_ancestor_max_hops: int = 2
    cross_ancestor_decay: float = 0.7


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------


def _bounded_ancestor_hops(
    driver: Any,
    iri_a: str,
    iri_b: str,
    *,
    max_hops: int,
    relationship: str = "SUBCLASSOF",
) -> int | None:
    """Return the minimum SUBCLASS-OF hop count between two IRIs, or `None`.

    n10s ingests ontologies with `handleRDFTypes=LABELS` so an instance
    carries its class membership as Neo4j labels, not as outbound edges.
    The walk opens at "the class node whose URI suffix matches one of
    the instance's labels", then walks SUBCLASSOF up to the target. This
    matches the pattern used by `ontology_bridge`. When both endpoints
    are already class nodes (the URI resolves to a class), the same
    pattern still works with one hop saved.

    Errors and timeouts are absorbed and surfaced as `None`; we prefer
    a degraded mapping (σ fallback) over a crash.
    """
    if not iri_a or not iri_b or driver is None:
        return None
    if iri_a == iri_b:
        return 0
    # Walk via n10s instance labels: anchor at a class node whose URI
    # local-name is in the instance's labels, then traverse SUBCLASSOF
    # to the target. Run in both directions and take the shorter hit.
    cypher = (
        "MATCH (a:Resource {uri: $iri_a}), (b:Resource {uri: $iri_b}) "
        "WITH a, b, labels(a) AS la, labels(b) AS lb "
        f"OPTIONAL MATCH p1 = (cls:Resource)-[:{relationship}*0.."
        f"{max_hops}]->(b) "
        "WHERE split(cls.uri, '/')[-1] IN la "
        f"OPTIONAL MATCH p2 = (cls2:Resource)-[:{relationship}*0.."
        f"{max_hops}]->(a) "
        "WHERE split(cls2.uri, '/')[-1] IN lb "
        "WITH coalesce(length(p1), length(p2)) AS hops "
        "WHERE hops IS NOT NULL "
        "RETURN hops ORDER BY hops ASC LIMIT 1"
    )
    try:
        with driver.session() as session:
            rec = session.run(cypher, iri_a=iri_a, iri_b=iri_b).single()
            if rec is None:
                return None
            hops = rec["hops"]
            return int(hops) if hops is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def fetch_matched_activities(
    driver: Any,
    task_iri: str,
    *,
    relationship: str = "MATCHEDACTIVITY",
) -> list[str]:
    """Return all `hb:matchedActivity` target IRIs for a task instance.

    Returns an empty list when the driver is unset, the task has no
    triples, or the query fails (degraded mode returns no targets so
    tier 1.5 defers cleanly to tier 2).
    """
    if not task_iri or driver is None:
        return []
    cypher = (
        f"MATCH (t:Resource {{uri: $task_iri}})-[:{relationship}]->(a:Resource) "
        "RETURN a.uri AS uri"
    )
    try:
        with driver.session() as session:
            return [str(rec["uri"]) for rec in session.run(cypher, task_iri=task_iri)]
    except Exception:  # pragma: no cover - defensive
        return []


# ---------------------------------------------------------------------------
# ActivityFamilyBridge; cross-ontology pre-filter for L_merge
# ---------------------------------------------------------------------------


class ActivityFamilyBridge:
    """Decide whether a `(task, host_event)` pair shares an activity family.

    Uses `hb:matchedActivity` edges plus a bounded SUBCLASS-OF walk in
    HumanActivities space. When the check returns True the caller can
    treat the pair as σ = 1.0 and skip an LLM-judge call.

    Args:
        driver: Neo4j driver; bridge degrades to "never shares" when None.
        event_ha_iri_by_label: catalog event label to `human_activity_iri`.
        max_hops: SUBCLASS-OF ancestor walk depth.
        relationship: SUBCLASSOF relationship type name in Neo4j.
        cross_relationship: matchedActivity relationship type name.
    """

    def __init__(
        self,
        *,
        driver: Any | None,
        event_ha_iri_by_label: dict[str, str] | None = None,
        max_hops: int = 2,
        relationship: str = "SUBCLASSOF",
        cross_relationship: str = "MATCHEDACTIVITY",
    ) -> None:
        self._driver = driver
        self._ha_by_label = dict(event_ha_iri_by_label or {})
        self._max_hops = int(max_hops)
        self._rel = relationship
        self._cross_rel = cross_relationship
        self._matched_cache: dict[str, list[str]] = {}
        self._family_cache: dict[tuple[str, str], bool] = {}

    def matched_activities(self, task_iri: str) -> list[str]:
        """Return cached `hb:matchedActivity` targets for the task."""
        if not task_iri or self._driver is None:
            return []
        if task_iri in self._matched_cache:
            return self._matched_cache[task_iri]
        targets = fetch_matched_activities(
            self._driver, task_iri, relationship=self._cross_rel
        )
        self._matched_cache[task_iri] = targets
        return targets

    def shares_family(self, task_iri: str, host_event_label: str) -> bool:
        """Return True iff the pair shares a family within `max_hops`."""
        if not task_iri or not host_event_label or self._driver is None:
            return False
        host_iri = self._ha_by_label.get(host_event_label)
        if not host_iri:
            return False
        key = (task_iri, host_iri)
        if key in self._family_cache:
            return self._family_cache[key]
        verdict = False
        for src_iri in self.matched_activities(task_iri):
            if src_iri == host_iri:
                verdict = True
                break
            hops = _bounded_ancestor_hops(
                self._driver,
                src_iri,
                host_iri,
                max_hops=self._max_hops,
                relationship=self._rel,
            )
            if hops is not None and hops > 0:
                verdict = True
                break
        self._family_cache[key] = verdict
        return verdict


# ---------------------------------------------------------------------------
# PreferenceMapper
# ---------------------------------------------------------------------------


class PreferenceMapper:
    """Three-tier `(recommended_task, catalog_event)` matcher.

    The mapper is constructed once per scenario (the CLI does this in
    `generate-tasks`) and consulted by every scorer.  The cache layer
    means subsequent persona-evaluations pay one `HMGET` round-trip
    rather than walking Neo4j + σ for every pair.
    """

    def __init__(
        self,
        *,
        cache: PreferenceCache,
        neo4j_driver: Any | None = None,
        semantic: SemanticCompatibility | None = None,
        cfg: MapperConfig | None = None,
        relationship: str = "SUBCLASSOF",
        cross_relationship: str = "MATCHEDACTIVITY",
        ontology_version: str = "",
    ) -> None:
        self._cache = cache
        self._driver = neo4j_driver
        self._semantic = semantic
        self._cfg = cfg or MapperConfig()
        self._rel = relationship
        self._cross_rel = cross_relationship
        self._ontology_version = ontology_version
        self._matched_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Cache hash inputs
    # ------------------------------------------------------------------

    def _hash_for(self, task: RecommendedTask, event: EventDefinition) -> str:
        """Content-hash for the (task, event) pair under current knobs."""
        return _content_hash(
            task.label,
            task.ontology_uri,
            event.name,
            event.health_task_iri,
            event.human_activity_iri,
            self._cfg.ancestor_max_hops,
            self._cfg.ancestor_discount,
            self._cfg.ancestor_decay,
            self._cfg.sigma_threshold,
            self._cfg.cross_ancestor_max_hops,
            self._cfg.cross_ancestor_discount,
            self._cfg.cross_ancestor_decay,
            self._ontology_version,
        )

    def _matched_activities(self, task_iri: str) -> list[str]:
        """Return cached `hb:matchedActivity` targets for the task."""
        if not task_iri:
            return []
        if task_iri in self._matched_cache:
            return self._matched_cache[task_iri]
        targets = fetch_matched_activities(
            self._driver, task_iri, relationship=self._cross_rel
        )
        self._matched_cache[task_iri] = targets
        return targets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def matched_events(
        self,
        task: RecommendedTask,
        events: Iterable[EventDefinition],
        *,
        experiment: str,
        scenario: str,
    ) -> list[MappedEvent]:
        """Return the list of events that match this task (weight > 0).

        Pairs are read from cache when present; misses are computed
        fresh and persisted in one pipelined `HSET`.
        """
        event_list = list(events)
        if not event_list:
            return []
        cache_pairs = [(task.label, ev.name) for ev in event_list]
        # Per-pair content hash: cache lookups need their own expected hash.
        results: list[MappedEvent] = []
        to_write: dict[tuple[str, str], CachedMapping] = {}
        cached_by_pair: dict[tuple[str, str], CachedMapping | None] = (
            self._cache.hmget_fields(experiment, scenario, cache_pairs)
        )
        for ev in event_list:
            expected_hash = self._hash_for(task, ev)
            cached = cached_by_pair.get((task.label, ev.name))
            if cached is not None and cached.content_hash == expected_hash:
                if cached.weight > 0:
                    results.append(
                        MappedEvent(
                            event_name=ev.name,
                            weight=cached.weight,
                            source_tier=cached.source_tier,
                            evidence=dict(cached.evidence),
                        )
                    )
                continue
            # Miss or stale; re-compute.
            mapping = self._compose(task, ev)
            to_write[(task.label, ev.name)] = CachedMapping(
                weight=mapping.weight,
                source_tier=mapping.source_tier,
                evidence=dict(mapping.evidence),
                content_hash=expected_hash,
            )
            if mapping.weight > 0:
                results.append(mapping)
        if to_write:
            self._cache.hset_pipeline(experiment, scenario, to_write)
        return results

    def warm_up(
        self,
        tasks: Iterable[RecommendedTask],
        events: Iterable[EventDefinition],
        *,
        experiment: str,
        scenario: str,
    ) -> int:
        """Pre-populate every `(task × event)` mapping in one pipeline.

        Returns the number of pairs computed (cache hits skipped).
        """
        task_list = list(tasks)
        event_list = list(events)
        if not task_list or not event_list:
            return 0
        all_pairs = [(t.label, e.name) for t in task_list for e in event_list]
        cached_by_pair = self._cache.hmget_fields(experiment, scenario, all_pairs)
        to_write: dict[tuple[str, str], CachedMapping] = {}
        computed = 0
        for task in task_list:
            for ev in event_list:
                expected_hash = self._hash_for(task, ev)
                cached = cached_by_pair.get((task.label, ev.name))
                if cached is not None and cached.content_hash == expected_hash:
                    continue
                mapping = self._compose(task, ev)
                to_write[(task.label, ev.name)] = CachedMapping(
                    weight=mapping.weight,
                    source_tier=mapping.source_tier,
                    evidence=dict(mapping.evidence),
                    content_hash=expected_hash,
                )
                computed += 1
        if to_write:
            self._cache.hset_pipeline(experiment, scenario, to_write)
        return computed

    # ------------------------------------------------------------------
    # Internal: tier composition
    # ------------------------------------------------------------------

    def _compose(self, task: RecommendedTask, event: EventDefinition) -> MappedEvent:
        """Walk all tiers and return the highest-weight result.

        Order: literal, cross-ontology bridge, ancestor walk, semantic.
        A literal hit short-circuits; otherwise every tier runs and the
        highest weight wins.
        """
        best = self._tier_literal(task, event)
        if best.weight >= 1.0:
            return best
        cross = self._tier_cross_ontology(task, event)
        if cross.weight > best.weight:
            best = cross
        if best.weight >= 1.0:
            return best
        ancestor = self._tier_ancestor(task, event)
        if ancestor.weight > best.weight:
            best = ancestor
        semantic = self._tier_semantic(task, event)
        if semantic.weight > best.weight:
            best = semantic
        return best

    def _tier_literal(
        self, task: RecommendedTask, event: EventDefinition
    ) -> MappedEvent:
        """Tier 1; exact IRI equality.  weight = 1.0 on hit, 0 otherwise."""
        task_uri = task.ontology_uri or ""
        if not task_uri:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        if event.health_task_iri and task_uri == event.health_task_iri:
            return MappedEvent(
                event_name=event.name,
                weight=1.0,
                source_tier="literal",
                evidence={"matched_iri": event.health_task_iri, "kind": "health_task"},
            )
        if event.human_activity_iri and task_uri == event.human_activity_iri:
            return MappedEvent(
                event_name=event.name,
                weight=1.0,
                source_tier="literal",
                evidence={
                    "matched_iri": event.human_activity_iri,
                    "kind": "human_activity",
                },
            )
        return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")

    def _tier_cross_ontology(
        self, task: RecommendedTask, event: EventDefinition
    ) -> MappedEvent:
        """Tier 1.5; bridge via `hb:matchedActivity` to HumanActivities.

        Fetches all curated `hb:matchedActivity` targets for the task,
        then for each target either matches the event's
        `human_activity_iri` directly (weight 1.0) or walks SUBCLASS-OF
        in HumanActivities space within `cross_ancestor_max_hops`
        (weight `cross_ancestor_discount * cross_ancestor_decay**hops`).
        Returns the best candidate across all targets.
        """
        target_ha = event.human_activity_iri
        if not target_ha or self._driver is None:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        targets = self._matched_activities(task.ontology_uri or "")
        if not targets:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        best_weight = 0.0
        best_tier = "none"
        best_evidence: dict[str, Any] = {}
        candidates: list[dict[str, Any]] = []
        for src_iri in targets:
            if src_iri == target_ha:
                candidates.append({"source_iri": src_iri, "hops": 0, "weight": 1.0})
                if 1.0 > best_weight:
                    best_weight = 1.0
                    best_tier = "cross_literal"
                    best_evidence = {
                        "source_iri": src_iri,
                        "target_iri": target_ha,
                        "hops": 0,
                    }
                continue
            hops = _bounded_ancestor_hops(
                self._driver,
                src_iri,
                target_ha,
                max_hops=self._cfg.cross_ancestor_max_hops,
                relationship=self._rel,
            )
            if hops is None or hops <= 0:
                continue
            weight = self._cfg.cross_ancestor_discount * (
                self._cfg.cross_ancestor_decay**hops
            )
            candidates.append({"source_iri": src_iri, "hops": hops, "weight": weight})
            if weight > best_weight or (
                weight == best_weight and best_tier != "cross_literal"
            ):
                best_weight = weight
                best_tier = "cross_ancestor"
                best_evidence = {
                    "source_iri": src_iri,
                    "target_iri": target_ha,
                    "hops": hops,
                }
        if best_weight <= 0.0:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        best_evidence["candidates"] = candidates
        return MappedEvent(
            event_name=event.name,
            weight=best_weight,
            source_tier=best_tier,
            evidence=best_evidence,
        )

    def _tier_ancestor(
        self, task: RecommendedTask, event: EventDefinition
    ) -> MappedEvent:
        """Tier 2; Neo4j SUBCLASS-OF walk.

        Tries the HealthTasks side first, then HumanActivities; whichever
        produces a shorter path wins.  Weight decays with hop count per
        `ancestor_discount × decay**hops`.  Returns weight 0 when no
        path is found within `max_hops` (or when the driver is unset).
        """
        if self._driver is None:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        task_uri = task.ontology_uri or ""
        if not task_uri:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        best_hops: int | None = None
        evidence: dict[str, Any] = {}
        for kind, ev_iri in (
            ("health_task", event.health_task_iri),
            ("human_activity", event.human_activity_iri),
        ):
            if not ev_iri or ev_iri == task_uri:
                continue
            hops = _bounded_ancestor_hops(
                self._driver,
                task_uri,
                ev_iri,
                max_hops=self._cfg.ancestor_max_hops,
                relationship=self._rel,
            )
            if hops is None or hops <= 0:
                continue
            if best_hops is None or hops < best_hops:
                best_hops = hops
                evidence = {"hops": hops, "via_iri": ev_iri, "kind": kind}
        if best_hops is None:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        weight = self._cfg.ancestor_discount * (self._cfg.ancestor_decay**best_hops)
        return MappedEvent(
            event_name=event.name,
            weight=weight,
            source_tier="ancestor",
            evidence=evidence,
        )

    def _tier_semantic(
        self, task: RecommendedTask, event: EventDefinition
    ) -> MappedEvent:
        """Tier 3; σ on rich labels.  weight = σ when ≥ threshold, else 0."""
        if self._semantic is None:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        task_text = self._task_query_text(task)
        event_text = f"{event.name}; {event.category}"
        sigma = float(self._semantic.score(task_text, event_text))
        if sigma < self._cfg.sigma_threshold:
            return MappedEvent(event_name=event.name, weight=0.0, source_tier="none")
        return MappedEvent(
            event_name=event.name,
            weight=sigma,
            source_tier="semantic",
            evidence={
                "sigma": sigma,
                "task_text": task_text,
                "event_text": event_text,
            },
        )

    @staticmethod
    def _task_query_text(task: RecommendedTask) -> str:
        """Rich query text for σ; display_name + description (when present)."""
        name = task.effective_display_name or task.label
        if task.description:
            return f"{name}; {task.description}"
        return name
