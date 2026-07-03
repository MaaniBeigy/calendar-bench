"""Vector-based retrieval with ontology-aware graph expansion.

The vector index seeds the search with semantically similar concepts, then the
retrieval Cypher pulls graph context (ancestor classes, typed neighbors,
and basic node attributes).

Two retriever variants are exposed:

* :class:`neo4j_graphrag.retrievers.VectorCypherRetriever`; used for
  open-ended Q&A over the whole indexed corpus (`ask.py`, the
  augmenter's RAG calls).
* :class:`MultiPrefixVectorCypherRetriever`; fans the search out
  across one `VectorCypherRetriever` per URI prefix and concatenates
  the results, so each ontology listed in the scenario YAML gets
  proportional representation in the LLM's context.  The task-
  generation pipeline uses this so a small ontology (HealthTasks,
  ~162 instances) is not crowded out of top-K by a large co-loaded
  one (HumanActivities, ~1210 instances).
"""

from __future__ import annotations

from typing import Any

from neo4j import Driver
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResultItem

from .retrieval_filters import AttributeFilter, cypher_predicate

VECTOR_INDEX_NAME = "concept_embedding_index"
NODE_LABEL = "EmbeddedConcept"
EMBEDDING_PROPERTY = "embedding"

# Property keys already surfaced elsewhere in the context block (as the
# heading or description) or internal to storage; everything else on a node
# is emitted on the `attributes:` line so numeric values like metValue and
# estimatedDurationMinutes reach the LLM and can be cited.
_SURFACE_EXCLUDE_KEYS = [
    "embedding",
    "uri",
    "label",
    "prefLabel",
    "altLabel",
    "title",
    "comment",
    "description",
    "definition",
    "iAO_0000115",
    "hasExactSynonym",
    "hasRelatedSynonym",
    "hasSynonym",
    "name",
]
_EXCLUDE_LITERAL = "[" + ", ".join(f"'{k}'" for k in _SURFACE_EXCLUDE_KEYS) + "]"

# URI namespace for authored HealthTasks instance nodes (`hb-tk:*`).  The
# task-generation pipeline pins its retriever to this prefix so the LLM only
# ever sees real instance URIs in context blocks; without it the vector
# index is dominated by the larger HumanActivities + BCTT + OCHV ontologies
# that share semantic neighbourhoods with "physical activity" / "nutrition".
HEALTH_TASK_INSTANCE_PREFIX = "https://w3id.org/calendar-bench/health/task/"
# ------------------------------- Ignore uncovered types ------------------------------
# Relationship-type filters. n10s with applyNeo4jNaming uppercases the local
# name without separators (e.g. SUBCLASSOF, EXACTMATCH). PARTOF/ISA aren't
# present in the imported ontologies, kept out to avoid noisy warnings.
HIERARCHY_RELS = ["SUBCLASSOF", "BROADER"]
NOISY_RELS = HIERARCHY_RELS + [
    "TYPE",
    "ANNOTATEDPROPERTY",
    "ANNOTATEDSOURCE",
    "ANNOTATEDTARGET",
]
# ------------------------------ Filter and deduplication -----------------------------
# It runs after the vector index lookup. Each ANN hit returns node and score.
# Filters blank nodes upfront and dedupes identical context strings at the end
# so duplicate skos/owl bnodes don't eat top-K slots.
RETRIEVAL_QUERY = f"""
WITH node, score
WHERE NOT node.uri STARTS WITH 'bnode://'

OPTIONAL MATCH (node)-[:{ '|'.join(HIERARCHY_RELS) }*1..2]->(parent:Resource)
WHERE NOT parent.uri STARTS WITH 'bnode://'
WITH node, score,
     [p IN collect(DISTINCT head(coalesce(parent.label, parent.prefLabel, parent.title, [parent.uri])))
        WHERE p IS NOT NULL] AS parents

OPTIONAL MATCH (node)-[r]->(neighbor:Resource)
WHERE NOT type(r) IN { NOISY_RELS }
  AND NOT neighbor.uri STARTS WITH 'bnode://'
WITH node, score, parents,
     [n IN collect(DISTINCT {{ rel: type(r),
                               name: head(coalesce(neighbor.label, neighbor.prefLabel, neighbor.title, [neighbor.uri])) }})
        WHERE n.name IS NOT NULL][..6] AS neighbors,
     [l IN labels(node) WHERE NOT l IN ['EmbeddedConcept', 'Resource', 'Class']] AS types,
     [k IN keys(node) WHERE NOT k IN { _EXCLUDE_LITERAL } | k + '=' + toString(head(node[k]))] AS attrs

WITH
  '## ' + head(coalesce(node.label, node.prefLabel, node.title, [node.uri])) +
  CASE WHEN size(types) > 0
       THEN '\\ntype: ' + apoc.text.join(types, ' / ')
       ELSE '' END +
  CASE WHEN size(parents) > 0
       THEN '\\nis-a: ' + apoc.text.join(parents, ' / ')
       ELSE '' END +
  CASE WHEN size(coalesce(node.comment, node.iAO_0000115, node.definition, node.description, [])) > 0
       THEN '\\ndescription: ' + head(coalesce(node.comment, node.iAO_0000115, node.definition, node.description, ['']))
       ELSE '' END +
  CASE WHEN size(attrs) > 0
       THEN '\\nattributes: ' + apoc.text.join(attrs, '; ')
       ELSE '' END +
  CASE WHEN size(neighbors) > 0
       THEN '\\nrelated: ' + apoc.text.join([n IN neighbors | n.rel + 'to' + n.name], '; ')
       ELSE '' END +
  '\\nuri: ' + node.uri AS context,
  score

WITH context, max(score) AS score
RETURN context, score
ORDER BY score DESC
"""

# Properties that mark a node as an *authored task instance* rather than
# an OWL class.  Mirrors the validator's `_EXISTS_QUERY` in
# `ontology_bridge.py` so the retriever and the validator agree on
# what counts as an instance; without this alignment the retriever
# happily surfaces class URIs (`HealthTask`, `EverydayTasksSelfCare`,
# …) that the validator then rejects, wasting fetch retries.
_INSTANCE_PROPERTY_GUARD = (
    "(node.title IS NOT NULL "
    "OR node.displayName IS NOT NULL "
    "OR node.estimatedDurationMinutes IS NOT NULL "
    "OR node.isConcurrent IS NOT NULL "
    "OR node.isDividable IS NOT NULL)"
)

# Allow-list for characters that may appear inside a literal-injected
# Cypher string (URI prefix, branch URI, level token).  Anything else -
# quotes, backslashes, control characters; is rejected at construction
# time so a bad caller cannot smuggle Cypher syntax into the query.
_SAFE_LITERAL_CHARS = ":/.-_~#"


def _validate_safe_literal(value: str, *, context: str) -> None:
    """Raise `ValueError` if *value* contains characters that are not
    safe to embed inside a Cypher string literal."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string; got {value!r}")
    for ch in value:
        if not (ch.isalnum() or ch in _SAFE_LITERAL_CHARS):
            raise ValueError(
                f"unsafe character {ch!r} in {context}: {value!r}; "
                f"only alphanumerics and {_SAFE_LITERAL_CHARS!r} are allowed"
            )


def _format_result(record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=record.get("context") or "",
        metadata={"score": record.get("score")},
    )


def _build_retrieval_query(
    uri_prefixes: list[str] | None,
    *,
    allowed_branches: list[str] | None = None,
    allowed_levels: list[str] | None = None,
    instance_only: bool = False,
    attribute_filters: list[AttributeFilter] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> str:
    """Return `RETRIEVAL_QUERY` extended with optional in-Cypher filters.

    All filters are AND-combined and embedded as Cypher list literals
    at construction time (no runtime injection surface; every literal
    is run through :func:`_validate_safe_literal` first).

    Args:
        uri_prefixes: when non-empty, restrict to `Resource` nodes
            whose `uri` starts with one of the listed prefixes.
        allowed_branches: when non-empty, restrict to nodes whose
            ancestor (walking `RDFTYPE|SUBCLASSOF` up to depth 10)
            has a `uri` in this list.  Used to keep the retriever
            inside the scenario's `filters.domains` (e.g. only
            `NutritionTask` / `PhysicalActivityTask`).  Implemented
            as a chained `OPTIONAL MATCH` + `count > 0` rather than
            a `WHERE EXISTS { ... }` subquery so the Cypher works on
            every Neo4j 4.x and 5.x version (the EXISTS-subquery form
            silently returned zero rows on the live graph and produced
            empty retrieval context; an earlier regression
            re-emerged after the in-Cypher filter was wired).
        allowed_levels: when non-empty, restrict to nodes whose
            ancestor URI ends with one of the listed tokens (e.g.
            `"Level1"`).  Same OPTIONAL MATCH + count pattern as
            `allowed_branches`; mirrors `task_matches_level`.
        instance_only: when True, restrict to nodes carrying at least
            one HealthTasks instance-only property (`title`,
            `displayName`, `estimatedDurationMinutes`,
            `isConcurrent`, `isDividable`).  Aligns the retriever
            with `validate_task_uri`'s `not_instance` rejection
            so OWL class nodes are never surfaced to the LLM.
        attribute_filters: when non-empty, hard-filter to nodes whose
            stored attribute satisfies the comparison, e.g.
            `head(node.metValue) > 7.3`.  Each filter is already
            validated against the attribute allow-list, so only known
            property names reach the query string.
        exclude_prefixes: when non-empty, drop `Resource` nodes whose
            `uri` starts with one of the listed prefixes.  Lets a
            caller keep a named ontology out of the search.
    """
    where_clauses: list[str] = []

    if uri_prefixes:
        for p in uri_prefixes:
            _validate_safe_literal(p, context="uri_prefix")
        prefix_list = ", ".join(f"'{p}'" for p in uri_prefixes)
        where_clauses.append(f"any(p IN [{prefix_list}] WHERE node.uri STARTS WITH p)")

    if exclude_prefixes:
        for p in exclude_prefixes:
            _validate_safe_literal(p, context="exclude_prefix")
        for p in exclude_prefixes:
            where_clauses.append(f"NOT node.uri STARTS WITH '{p}'")

    if instance_only:
        where_clauses.append(_INSTANCE_PROPERTY_GUARD)

    if attribute_filters:
        for flt in attribute_filters:
            where_clauses.append(cypher_predicate(flt))

    # Branch / level filters BOTH use `labels(node)` as the entry
    # point because n10s on the live graph is configured with
    # `handleRDFTypes = LABELS`; i.e. an instance's `rdf:type`
    # class is materialised as a Neo4j *label* on the instance, NOT
    # as a `RDFTYPE` relationship.  The previous
    # `[:RDFTYPE|SUBCLASSOF]` walk found zero edges from any
    # instance and silently returned zero rows for every retrieval
    # call (the empty-context regression where the LLM emitted the
    # literal `<URI>` placeholder).
    #
    # The corrected pattern: each instance carries the class's local
    # name as a label, and class nodes are reachable via SUBCLASSOF
    # walks from anywhere; so we open the chain at "any class node
    # whose URI suffix matches one of this instance's labels", then
    # walk SUBCLASSOF up to the branch / level token.

    branch_block = ""
    if allowed_branches:
        for b in allowed_branches:
            _validate_safe_literal(b, context="allowed_branch")
        branch_list = ", ".join(f"'{b}'" for b in allowed_branches)
        branch_block = (
            "\nWITH node, score, labels(node) AS _il\n"
            "OPTIONAL MATCH (_cls_b:Resource)"
            "-[:SUBCLASSOF*0..10]->(_branch:Resource)\n"
            f"WHERE _branch.uri IN [{branch_list}]\n"
            "  AND split(_cls_b.uri, '/')[-1] IN _il\n"
            "WITH node, score, _il, count(DISTINCT _cls_b) AS _branch_hits\n"
            "WHERE _branch_hits > 0"
        )

    level_block = ""
    if allowed_levels:
        for tok in allowed_levels:
            _validate_safe_literal(tok, context="allowed_level")
        level_list = ", ".join(f"'{t}'" for t in allowed_levels)
        # When `branch_block` already projected `_il`, reuse it;
        # otherwise bind it here so `split(...)[-1] IN _il` resolves.
        level_open = (
            "\n" if branch_block else "\nWITH node, score, labels(node) AS _il\n"
        )
        level_block = (
            level_open + "OPTIONAL MATCH (_cls_l:Resource)"
            "-[:SUBCLASSOF*0..10]->(_level:Resource)\n"
            f"WHERE any(t IN [{level_list}] WHERE _level.uri ENDS WITH t)\n"
            "  AND split(_cls_l.uri, '/')[-1] IN _il\n"
            "WITH node, score, count(DISTINCT _cls_l) AS _level_hits\n"
            "WHERE _level_hits > 0"
        )

    if not where_clauses and not branch_block and not level_block:
        return RETRIEVAL_QUERY

    cypher = RETRIEVAL_QUERY
    if where_clauses:
        addition = "\n  AND " + "\n  AND ".join(where_clauses)
        cypher = cypher.replace(
            "WHERE NOT node.uri STARTS WITH 'bnode://'",
            "WHERE NOT node.uri STARTS WITH 'bnode://'" + addition,
            1,
        )
    if branch_block or level_block:
        # Splice the branch / level blocks in BEFORE the existing
        # OPTIONAL MATCH chain that builds the context string.  We use
        # the unique blank line that separates the bnode-guard WHERE
        # from the parents OPTIONAL MATCH as the splice point so the
        # downstream context-building Cypher is left untouched.
        splice_marker = "\nOPTIONAL MATCH (node)-[:SUBCLASSOF|BROADER"
        idx = cypher.find(splice_marker)
        if idx == -1:  # pragma: no cover - defensive; RETRIEVAL_QUERY is fixed
            raise RuntimeError("RETRIEVAL_QUERY shape changed; splice marker not found")
        cypher = cypher[:idx] + branch_block + level_block + "\n" + cypher[idx:]
    return cypher


class MultiPrefixVectorCypherRetriever(Retriever):
    """Fan a vector search out across one underlying retriever per URI
    prefix and concatenate the results.

    The default `VectorCypherRetriever` returns top-K nearest
    neighbours from the *whole* embedded corpus.  When the corpus
    contains ontologies of very different sizes (e.g. 1,210
    HumanActivities instances vs. 162 HealthTasks instances) and the
    user query semantically matches the larger one, the smaller
    ontology is crowded out of top-K; even though the scenario YAML
    listed it.  This wrapper guarantees per-ontology representation by
    issuing one independent vector search per prefix, each restricted
    to its own URI namespace plus any branch / level / instance
    filters supplied.

    Subclasses `neo4j_graphrag.retrievers.base.Retriever` so the
    Pydantic validation in :class:`neo4j_graphrag.generation.GraphRAG`
    accepts it as a drop-in retriever.  `get_search_results` merges
    the underlying retrievers' raw records; the inherited
    :meth:`Retriever.search` then formats them via
    :func:`_format_result` and wraps them as a `RetrieverResult`.
    """

    # The underlying `VectorCypherRetriever` instances each run their
    # own Neo4j-version probe in `__init__`; the base `Retriever`
    # version probe would just duplicate that work (and crashes the
    # mocked-driver path used in tests).
    VERIFY_NEO4J_VERSION = False

    # The base `Retriever` exposes `index_name` as a class
    # attribute; every per-prefix child retriever shares the
    # project-wide vector index, so set it here for consistency.
    index_name = VECTOR_INDEX_NAME

    # Default multiplier applied to the user's `top_k` when calling
    # each underlying retriever.  Necessary because
    # `neo4j_graphrag`'s `get_search_query` wraps every retrieval
    # call as::
    #
    #   CALL db.index.vector.queryNodes(idx, $top_k * $eff_ratio, qv)
    #     YIELD node, score
    #   WITH node, score LIMIT $top_k     ← truncates BEFORE our filter
    #   <retrieval_query>
    #
    # i.e. the LIMIT is applied *before* our WHERE clauses get to see
    # the candidates.  `effective_search_ratio` only widens the pool
    # the vector index re-ranks on; it does NOT widen what reaches our
    # filter.  To get enough candidates past our restrictive WHEREs we
    # must request a much larger `top_k` from the underlying
    # retriever and then truncate the merged result back to what the
    # caller asked for.  50 × `top_k=20` = 1,000 surviving the
    # pre-filter LIMIT comfortably covers the 162 HealthTasks instances
    # even when HumanActivities (1,210 nodes) outranks them on raw
    # cosine.
    DEFAULT_PREFILTER_POOL_MULTIPLIER = 50

    def __init__(
        self,
        driver: Driver,
        embedder: Embedder,
        *,
        uri_prefixes: list[str],
        allowed_branches: list[str] | None = None,
        allowed_levels: list[str] | None = None,
        instance_only: bool = False,
        attribute_filters: list[AttributeFilter] | None = None,
        exclude_prefixes: list[str] | None = None,
        neo4j_database: str | None = None,
        prefilter_pool_multiplier: int | None = None,
    ) -> None:
        if not uri_prefixes:
            raise ValueError(
                "MultiPrefixVectorCypherRetriever requires at least one "
                "uri_prefix; use VectorCypherRetriever directly for the "
                "unfiltered case."
            )
        super().__init__(driver, neo4j_database)
        # Set `result_formatter` on the instance so the inherited
        # `Retriever.search()` formats merged raw records the same
        # way the underlying retrievers would.
        self.result_formatter = _format_result
        self._uri_prefixes = list(uri_prefixes)
        self._prefilter_pool_multiplier = (
            prefilter_pool_multiplier
            if prefilter_pool_multiplier is not None
            else self.DEFAULT_PREFILTER_POOL_MULTIPLIER
        )
        self._retrievers: list[VectorCypherRetriever] = [
            VectorCypherRetriever(
                driver=driver,
                index_name=VECTOR_INDEX_NAME,
                retrieval_query=_build_retrieval_query(
                    [prefix],
                    allowed_branches=allowed_branches,
                    allowed_levels=allowed_levels,
                    instance_only=instance_only,
                    attribute_filters=attribute_filters,
                    exclude_prefixes=exclude_prefixes,
                ),
                embedder=embedder,
                result_formatter=_format_result,
                neo4j_database=neo4j_database,
            )
            for prefix in uri_prefixes
        ]

    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        """Run the vector search against every per-prefix retriever and
        return the concatenated, deduplicated raw records.

        `query_text` and any other keyword arguments are forwarded
        verbatim to each underlying retriever, but `top_k` is
        multiplied by :attr:`DEFAULT_PREFILTER_POOL_MULTIPLIER` (or
        the constructor-time override) so enough candidates survive
        `neo4j_graphrag`'s `WITH node, score LIMIT $top_k` -
        which it inserts BEFORE our retrieval Cypher's WHERE clauses.
        Without the bump the LIMIT truncates to the user's requested
        `top_k` candidates, our filter then narrows them to zero,
        the LLM sees an empty context block, and falls back to
        emitting the literal `<URI>` placeholder from the prompt
        template (an earlier empty-context regression).

        Per-prefix order is preserved so the LLM sees results from
        the YAML's first-listed ontology first.  The merged record
        list is truncated to the original user-requested `top_k` so
        the LLM is not flooded with hundreds of context blocks.  The
        base `Retriever.search` then runs the records through
        `result_formatter` and wraps them as a
        :class:`RetrieverResult`.
        """
        user_top_k = int(kwargs.pop("top_k", 5))
        kwargs["top_k"] = max(user_top_k * self._prefilter_pool_multiplier, user_top_k)
        seen_contexts: set[str] = set()
        merged_records: list[Any] = []
        per_prefix_counts: list[int] = []
        for retriever in self._retrievers:
            sub = retriever.get_search_results(*args, **kwargs)
            kept = 0
            for record in sub.records:
                # Deduplicate by context string; different prefixes
                # should produce disjoint sets, but the guard keeps the
                # LLM from seeing the same block twice if someone wires
                # overlapping prefixes by mistake.
                ctx = record.get("context")
                key = ctx if isinstance(ctx, str) else str(ctx)
                if key in seen_contexts:
                    continue
                seen_contexts.add(key)
                merged_records.append(record)
                kept += 1
            per_prefix_counts.append(kept)
        # Truncate the merged result back to the user-requested
        # top_k so the LLM is not flooded with hundreds of context
        # blocks (the bumped pool was only needed to get past the
        # neo4j_graphrag pre-filter LIMIT; once filtered, the user
        # wanted `user_top_k` blocks, not a thousand).
        truncated = merged_records[:user_top_k]
        # Use `model_construct` to skip Pydantic's re-validation of
        # the records; each underlying retriever has already returned a
        # validated `RawSearchResult`, so re-checking would just cost
        # cycles and break the mocked-driver test path.
        return RawSearchResult.model_construct(
            records=truncated,
            metadata={
                "uri_prefixes": list(self._uri_prefixes),
                "per_prefix_item_counts": per_prefix_counts,
                "user_top_k": user_top_k,
                "internal_top_k": kwargs["top_k"],
                "merged_total": len(merged_records),
            },
        )


def make_retriever(
    driver: Driver,
    embedder: Embedder,
    *,
    uri_prefixes: list[str] | None = None,
    allowed_branches: list[str] | None = None,
    allowed_levels: list[str] | None = None,
    instance_only: bool = False,
    attribute_filters: list[AttributeFilter] | None = None,
    exclude_prefixes: list[str] | None = None,
):
    """Build the project's standard vector + Cypher retriever.

    The behavior depends on whether any task-generation filters are
    supplied:

    * Nothing supplied to return a plain
      :class:`VectorCypherRetriever` over the unfiltered corpus.  This
      is the legacy path (used by `ask.py` and the augmenter's RAG
      calls), where the caller wants top-K cosine similarity over the
      whole graph and `effective_search_ratio` defaults to 1.
    * Any of `uri_prefixes` / `allowed_branches` / `allowed_levels`
      / `instance_only` supplied to return a
      :class:`MultiPrefixVectorCypherRetriever`.  Even with a single
      prefix the wrapper is preferred because it injects a high
      `effective_search_ratio`: each restrictive WHERE clause can
      throw away >95% of any plain top-K cosine slice (an
      earlier empty-context regression where the LLM emitted the
      `<URI>` placeholder), so we ask the vector index for a much
      wider initial pool and let the WHERE narrow it down.

    The optional `allowed_branches` / `allowed_levels` /
    `instance_only` filters propagate into every per-prefix
    Cypher so the LLM only sees URIs that would also pass the
    downstream :func:`validate_task_uri` check.
    """
    has_any_filter = bool(
        uri_prefixes
        or allowed_branches
        or allowed_levels
        or instance_only
        or attribute_filters
        or exclude_prefixes
    )
    if has_any_filter:
        # The wrapper requires at least one prefix; when the caller
        # only supplied branch / level / instance filters (no prefix),
        # use the global namespace as the wrapper's single prefix so
        # the wrapper still handles the search-ratio bump.  In
        # practice the project always supplies a prefix for the
        # filtered path, but the fallback keeps the surface flexible.
        prefixes = list(uri_prefixes) if uri_prefixes else ["http"]
        return MultiPrefixVectorCypherRetriever(
            driver=driver,
            embedder=embedder,
            uri_prefixes=prefixes,
            allowed_branches=allowed_branches,
            allowed_levels=allowed_levels,
            instance_only=instance_only,
            attribute_filters=attribute_filters,
            exclude_prefixes=exclude_prefixes,
        )
    return VectorCypherRetriever(
        driver=driver,
        index_name=VECTOR_INDEX_NAME,
        retrieval_query=_build_retrieval_query(
            uri_prefixes,
            allowed_branches=allowed_branches,
            allowed_levels=allowed_levels,
            instance_only=instance_only,
            attribute_filters=attribute_filters,
            exclude_prefixes=exclude_prefixes,
        ),
        embedder=embedder,
        result_formatter=_format_result,
    )
