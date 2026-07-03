"""Fetch task properties from Neo4j and map branches to admissible epochs.

The HealthTasks ontology is imported into Neo4j via n10s.  After import,
each OWL concept is a `Resource` node, class hierarchy is stored as
`SUBCLASSOF` relationships, and OWL datatype properties become node
properties with their local camelCase names.

Property names used here:

    `isDividable`             ; xsd:boolean from hb:isDividable
    `isConcurrent`            ; xsd:boolean from hb:isConcurrent
    `estimatedDurationMinutes`; xsd:integer from hb:estimatedDurationMinutes
    `displayName`             ; xsd:string from hb:displayName (rare;
                                   used only if a task author wired this
                                   bespoke property).
    `title`                   ; xsd:string from dcterms:title (n10s
                                   exposes this as `Resource.title` because
                                   `handleVocabUris: 'IGNORE'`).  The
                                   HealthTasks ontology uses this property
                                   for emoji-bearing labels; `"Blend a
                                   Smoothie 🥤"` and similar.
    `label`                   ; xsd:string `Resource.label` n10s puts
                                   on the node from rdfs:label.  Plain
                                   text, no emojis.
    `description`             ; xsd:string from dcterms:description
                                   (n10s exposes as `Resource.description`).
                                   Carries the rich, authored description
                                   for HealthTasks instances.
    `comment`                 ; xsd:string `Resource.comment` n10s puts
                                   on the node from rdfs:comment.

The coalesce order for the rendered fields is:

    display_name = coalesce(displayName, title, label)
    description  = coalesce(description, comment)

so that emoji-bearing values authored via `dcterms:title` /
`dcterms:description` flow through verbatim, with `rdfs:label` /
`rdfs:comment` as plain-text fallbacks.

Because n10s is configured with `handleMultival: 'ARRAY'`, every
literal property is stored as a Neo4j list; even single-valued ones.
The `_PROPS_QUERY` wraps every returned column in `head(...)` so
the Python layer receives scalars rather than 1-element lists.

The three top-level branch URIs are:

    PhysicalActivityTask; tasks involving physical movement or exercise.
    NutritionTask       ; tasks related to eating, hydration, or meal habits.
    MentalWellbeingTask ; tasks for stress management, sleep, mindfulness.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Branch URIs (top-level sub-classes of hb:HealthTask)
# ---------------------------------------------------------------------------

_BASE = "https://w3id.org/calendar-bench/health/"

PHYSICAL_ACTIVITY_URI: str = _BASE + "PhysicalActivityTask"
NUTRITION_URI: str = _BASE + "NutritionTask"
MENTAL_WELLBEING_URI: str = _BASE + "MentalWellbeingTask"

# ---------------------------------------------------------------------------
# Top-level HealthTasks branches recognized by the ontology bridge.
# ---------------------------------------------------------------------------

_KNOWN_BRANCHES: tuple[str, ...] = (
    PHYSICAL_ACTIVITY_URI,
    NUTRITION_URI,
    MENTAL_WELLBEING_URI,
)

# ---------------------------------------------------------------------------
# Neo4j queries
# ---------------------------------------------------------------------------

_PROPS_QUERY = """
MATCH (n:Resource {uri: $uri})
RETURN
  head(n.isDividable)                              AS is_dividable,
  head(n.isConcurrent)                             AS is_concurrent,
  head(n.estimatedDurationMinutes)                 AS duration_minutes,
  head(coalesce(n.displayName, n.title, n.label))  AS display_name,
  head(coalesce(n.description, n.comment))         AS description
"""

_BRANCH_QUERY = """
MATCH (n:Resource {uri: $uri})
WITH n, labels(n) AS _il
MATCH path = (cls:Resource)-[:SUBCLASSOF*0..10]->(branch:Resource)
WHERE branch.uri IN $branches
  AND split(cls.uri, '/')[-1] IN _il
RETURN branch.uri AS branch, length(path) AS depth
ORDER BY depth ASC
LIMIT 1
"""
"""Resolve the instance's nearest top-level branch.

The live Neo4j graph imports ontologies via n10s with
`handleRDFTypes = LABELS`: an instance's `rdf:type` class is
materialised as a Neo4j *label* on the instance, NOT as a separate
`RDFTYPE` relationship.  The previous query walked
`[:RDFTYPE|SUBCLASSOF*0..10]` from the instance and found zero
edges; every `resolve_task_branch` call returned `None` and
every URI was bucketed into the `ok_branch_unresolved` graceful-
degradation path (or, when combined with the level filter, into
`wrong_level`).

The corrected pattern: the class's local name (URI suffix) is one of
the instance's labels.  We open the chain at "any class node whose
URI suffix is in the instance's labels", then walk SUBCLASSOF up to
the branch.  Confirmed live: returns the correct branch in 39 / 39
in-filter HealthTasks instances for `nutrition_l1`."""

_EXISTS_QUERY = """
MATCH (n:Resource {uri: $uri})
WHERE n.title              IS NOT NULL
   OR n.displayName        IS NOT NULL
   OR n.estimatedDurationMinutes IS NOT NULL
   OR n.isConcurrent       IS NOT NULL
   OR n.isDividable        IS NOT NULL
RETURN 1 AS hit
LIMIT 1
"""

_LEVEL_QUERY = """
MATCH (n:Resource {uri: $uri})
WITH n, labels(n) AS _il
MATCH (cls:Resource)-[:SUBCLASSOF*0..10]->(ancestor:Resource)
WHERE any(token IN $tokens WHERE ancestor.uri ENDS WITH token)
  AND split(cls.uri, '/')[-1] IN _il
RETURN 1 AS hit
LIMIT 1
"""
"""Match the instance's level via the same label-then-SUBCLASSOF chain
as :data:`_BRANCH_QUERY`.  See that docstring for the rationale on
why `RDFTYPE` traversal does not work on this n10s install."""

#: Discover the integer difficulty level (1–4) of a HealthTasks instance
#: by walking SUBCLASSOF up to a class whose URI ends with `LevelN`.
#: Returns the smallest matching `N` (closest ancestor wins on tie).
_LEVEL_NUMERIC_QUERY = """
MATCH (n:Resource {uri: $uri})
WITH n, labels(n) AS _il
MATCH (cls:Resource)-[:SUBCLASSOF*0..10]->(ancestor:Resource)
WHERE split(cls.uri, '/')[-1] IN _il
  AND any(t IN $tokens WHERE ancestor.uri ENDS WITH t)
WITH ancestor.uri AS uri
WITH [t IN $tokens WHERE uri ENDS WITH t][0] AS token
RETURN token
ORDER BY token ASC
LIMIT 1
"""

#: All four `LevelN` URI suffixes we care about.  The ordering
#: `Level1 < Level2 < … < Level4` matches the `ORDER BY token` in
#: `_LEVEL_NUMERIC_QUERY` so the closest-on-the-chain ancestor wins
#: deterministically when an instance subclasses two of them
#: (theoretically possible if an author wires multiple level mixins -
#: in practice the ontology only ever does one).
_LEVEL_TOKENS: tuple[str, ...] = ("Level1", "Level2", "Level3", "Level4")
"""Resolve to True only when the node carries at least one
*instance-only* property.  This rejects OWL class nodes such as
`hb:HealthTask` (which carry just `rdfs:label` and `rdfs:comment`)
even though they exist as `Resource` rows in Neo4j; those are not
authored task instances and were the source of the "Health Task"
regression.

The five properties checked are populated by the HealthTasks importer
on instance nodes only:

  * `n.title`           ; n10s view of `dcterms:title` (carries
                             the emoji-bearing display name)
  * `n.displayName`     ; bespoke `hb:displayName` (rare)
  * `n.estimatedDurationMinutes`,
    `n.isConcurrent`,
    `n.isDividable`     ; the three datatype properties only set on
                             concrete instance nodes."""


def _uri_lookup_variants(uri: str) -> list[str]:
    """Return the ladder of URI shapes to try against the graph, in order.

    The ontology authors instance nodes under the `hb-tk:` prefix
    (`https://w3id.org/calendar-bench/health/task/<kebab-case>`).
    LLM responses often paraphrase that to `hb:`-style snake_case
    (`https://w3id.org/calendar-bench/health/<snake_case>`).  This
    helper enumerates the literal URI plus three normalized variants so
    the lookup recovers from such mistakes:

      1. the verbatim URI;
      2. the URI with `_` to `-` in the local name;
      3. the URI with `/task/` injected before the local name;
      4. both transformations combined.

    Variants beyond #1 are appended only when they actually differ from
    earlier entries; duplicates are skipped so a well-formed URI does
    not pay for extra round-trips.
    """
    variants: list[str] = [uri]
    if "/" not in uri:
        return variants
    base, _, local = uri.rpartition("/")
    candidates = [
        uri,
        f"{base}/{local.replace('_', '-')}",
        (f"{base}/task/{local}" if not base.endswith("/task") else f"{base}/{local}"),
        (
            f"{base}/task/{local.replace('_', '-')}"
            if not base.endswith("/task")
            else f"{base}/{local.replace('_', '-')}"
        ),
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            deduped.append(cand)
    return deduped


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def task_matches_level(uri: str, session: Any, levels: list[str] | None) -> bool:
    """Return True iff *uri* is a subclass-of an ancestor whose URI ends
    with one of *levels* (e.g. `["Level1"]`).

    Walks the `SUBCLASSOF` chain via :func:`_uri_lookup_variants` so a
    URI written as snake_case still matches its kebab-case canonical form.

    When *levels* is empty or None, returns True (no level filter
    requested).
    """
    if not levels:
        return True
    for candidate in _uri_lookup_variants(uri):
        result = session.run(_LEVEL_QUERY, uri=candidate, tokens=list(levels)).single()
        if result is not None:
            return True
    return False


def fetch_difficulty_level(uri: str, session: Any) -> int:
    """Return the integer 1–4 difficulty level for a HealthTasks URI.

    Walks SUBCLASSOF up to a class whose URI ends with `LevelN` and
    returns `N`.  Returns `0` for non-HealthTasks URIs or when no
    Level ancestor is found (graceful degradation matching the rest of
    the bridge).

    The lookup tries every URI variant returned by
    :func:`_uri_lookup_variants`, so kebab/snake-case mismatches and
    missing `/task/` segments still resolve.
    """
    for candidate in _uri_lookup_variants(uri):
        result = session.run(
            _LEVEL_NUMERIC_QUERY,
            uri=candidate,
            tokens=list(_LEVEL_TOKENS),
        ).single()
        if result is None:
            continue
        token = result.get("token")
        if not token:
            continue
        # token is "LevelN"; strip the prefix to get the integer.
        try:
            return int(str(token).removeprefix("Level"))
        except (TypeError, ValueError):
            continue
    return 0


# Map `filters.domains` local names to canonical branch URIs.
BRANCH_URIS_BY_LOCAL_NAME: dict[str, str] = {
    "PhysicalActivityTask": PHYSICAL_ACTIVITY_URI,
    "NutritionTask": NUTRITION_URI,
    "MentalWellbeingTask": MENTAL_WELLBEING_URI,
}


def validate_task_uri(
    uri: str,
    session: Any,
    *,
    allowed_branches: set[str] | None = None,
    allowed_levels: list[str] | None = None,
) -> tuple[bool, str]:
    """Composite filter for the fetch-and-verify loop.

    Combines three independent checks:

      * `uri_resolves`; node exists AND carries instance properties
        (rejects OWL class nodes and bare `rdfs:label`-only nodes);
      * branch filter; when *allowed_branches* is non-empty, the URI's
        nearest top-level branch ancestor must be in the set;
      * level filter; when *allowed_levels* is non-empty, the URI must
        have an ancestor whose URI ends with one of the supplied
        tokens (`"Level1"` etc.).

    Returns `(ok, reason)` where *reason* is one of:

      `"ok"`; every requested check passed;
      `"ok_branch_unresolved"`; instance + level OK, branch could not
        be classified (graceful degradation, see below);
      `"not_instance"`; node missing or class-only;
      `"wrong_branch"`; node resolves to a KNOWN branch that is not in
        `allowed_branches`;
      `"wrong_level"`; node resolves but no ancestor matches a level
        token.

    Graceful-degradation policy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When `resolve_task_branch` returns `None` (no ancestor matches
    any of the three known branches: `PhysicalActivityTask` /
    `NutritionTask` / `MentalWellbeingTask`), the URI is **accepted
    with verdict `"ok_branch_unresolved"`** rather than rejected as
    `"wrong_branch"`.

    Why: in practice `branch is None` can mean either
      (a) the URI legitimately sits outside the three known HealthTasks
          branches (e.g. a HumanActivities `ha-act:*` URI), or
      (b) the live Neo4j graph isn't traversable end-to-end (n10s'
          rdf:type storage varies between installs).

    Rejecting on `None` would lock the user out completely whenever
    (b) occurs; a symptom where every URI was tagged
    `wrong_branch`.  Accepting under graceful degradation lets the
    pipeline produce output and surfaces the situation as a separate
    verdict in the audit log so the user can spot a runaway
    false-positive without re-running.

    The reason string is fed into the per-persona
    `tasks_proposed.json` log so the user can see WHY each URI was
    dropped without grepping the LLM-debug attempts.
    """
    if not uri_resolves(uri, session):
        return False, "not_instance"
    branch_unresolved = False
    if allowed_branches:
        branch = resolve_task_branch(uri, session)
        if branch is not None and branch not in allowed_branches:
            return False, "wrong_branch"
        if branch is None:
            branch_unresolved = True
    if allowed_levels and not task_matches_level(uri, session, allowed_levels):
        return False, "wrong_level"
    return True, "ok_branch_unresolved" if branch_unresolved else "ok"


def validate_task_uri_groups(
    uri: str,
    session: Any,
    *,
    groups: list[tuple[set[str], list[str]]],
) -> tuple[bool, str, str | None]:
    """Grouped filter; accept iff the URI matches some (branches, levels) group.

    Each group is `(allowed_branches, allowed_levels)`. A task matches a
    group when its branch is in `allowed_branches` (or the group sets no
    branch) and its level token is in `allowed_levels` (or the group sets
    no level). The branch and level are resolved once and tested against
    every group in Python, so multiple groups cost no extra Neo4j round
    trips. Returns `(ok, reason, branch)` with the same reason vocabulary
    as :func:`validate_task_uri`; `branch` is the resolved branch URI or
    `None`. An unresolved branch is accepted under the same graceful
    degradation policy, still subject to the level filter.
    """
    if not uri_resolves(uri, session):
        return False, "not_instance", None
    checks = groups or [(set(), [])]
    needs_branch = any(allowed_branches for allowed_branches, _ in checks)
    needs_level = any(allowed_levels for _, allowed_levels in checks)
    branch = resolve_task_branch(uri, session) if needs_branch else None
    branch_unresolved = needs_branch and branch is None
    level = fetch_difficulty_level(uri, session) if needs_level else 0
    branch_matched_any = False
    for allowed_branches, allowed_levels in checks:
        if not (
            not allowed_branches or branch_unresolved or branch in allowed_branches
        ):
            continue
        branch_matched_any = True
        if not allowed_levels or (level > 0 and f"Level{level}" in allowed_levels):
            return True, "ok_branch_unresolved" if branch_unresolved else "ok", branch
    return (False, "wrong_level" if branch_matched_any else "wrong_branch", branch)


def uri_resolves(uri: str, session: Any) -> bool:
    """Return True iff *uri* (or one of its tolerated lookup variants)
    matches a `Resource` node in the graph.

    Used by the task-generation pipeline to detect URIs the LLM
    fabricated.  The bridge accepts the same snake/kebab and
    `/task/` segment variants as :func:`fetch_task_properties` so a
    well-formed URI that just uses underscores is not classified as
    fabricated.
    """
    for candidate in _uri_lookup_variants(uri):
        result = session.run(_EXISTS_QUERY, uri=candidate).single()
        if result is not None:
            return True
    return False


def fetch_task_properties(uri: str, session: Any) -> dict[str, Any]:
    """Return ontology-derived properties for a HealthTasks concept URI.

    Missing or null properties are omitted from the returned dict so callers
    can distinguish "property not set" from "property set to False / 0".
    Empty-string values are also treated as missing; an authored ontology
    label is either a non-empty string or absent.

    The `display_name` field is populated from `hb:displayName` first
    and falls back to `dcterms:title` and then `rdfs:label` (n10s
    ingests these as `Resource.title` and `Resource.label`).  Whatever
    the ontology authors wrote; including any emoji; flows through to
    the caller verbatim.  The `description` field is populated from
    `dcterms:description` and falls back to `rdfs:comment`.

    The lookup tolerates two common URI mistakes by walking
    :func:`_uri_lookup_variants`: `_` instead of `-` in the local
    name, and a missing `/task/` path segment.  The first variant
    that resolves to a node wins; further variants are not queried.

    Args:
        uri: full URI of a HealthTasks concept.
        session: open Neo4j driver session.

    Returns:
        Dict whose possible keys are `is_dividable`, `is_concurrent`,
        `duration_minutes`, `display_name`, `description`; only
        those whose values are not None / not empty string.  Empty dict
        when none of the URI variants resolves to a node.
    """
    for candidate in _uri_lookup_variants(uri):
        result = session.run(_PROPS_QUERY, uri=candidate).single()
        if result is None:
            continue
        props = {
            k: v
            for k, v in result.data().items()
            if v is not None and not (isinstance(v, str) and not v.strip())
        }
        if props:
            return props
    return {}


def resolve_task_branch(uri: str, session: Any) -> str | None:
    """Walk `SUBCLASSOF` ancestors to find the nearest top-level branch URI.

    Returns one of `PHYSICAL_ACTIVITY_URI`, `NUTRITION_URI`,
    `MENTAL_WELLBEING_URI`, or `None` when the URI is not found or has
    no recognized branch ancestor.

    Like :func:`fetch_task_properties`, the lookup walks
    :func:`_uri_lookup_variants` so common URI mis-spellings still
    resolve to a branch.
    """
    for candidate in _uri_lookup_variants(uri):
        result = session.run(
            _BRANCH_QUERY,
            uri=candidate,
            branches=list(_KNOWN_BRANCHES),
        ).single()
        if result is None:
            continue
        branch = result.get("branch")
        if branch is not None:
            return branch
    return None


def enrich_task_from_ontology(uri: str, session: Any) -> dict[str, Any]:
    """Fetch the ontology-derived task fields (flags, duration, display, description, difficulty)."""
    props = fetch_task_properties(uri, session)
    props["difficulty_level"] = fetch_difficulty_level(uri, session)
    return props


# ---------------------------------------------------------------------------
# Context dictionary lookups
# ---------------------------------------------------------------------------

CONTEXT_CATEGORY_BASE: str = "https://w3id.org/calendar-bench/context/category/"

_CONTEXT_HAS_QUERY = """
MATCH (n:Resource {uri: $iri})-[:CATEGORY]->()
RETURN 1 AS hit
LIMIT 1
"""

_CONTEXT_GET_QUERY = """
MATCH (n:Resource {uri: $iri})-[:CATEGORY]->(cls:Resource)
WHERE cls.uri STARTS WITH $base
RETURN
  head(n.slug)        AS slug,
  head(n.source)      AS source,
  head(n.label)       AS label,
  head(n.definition)  AS definition,
  head(n.polarity)    AS polarity,
  head(n.instrument)  AS instrument,
  head(n.comment)     AS comment,
  head(cls.slug)      AS category
LIMIT 1
"""

_CONTEXT_CATEGORIES_QUERY = """
MATCH (cls:Resource)
WHERE cls.uri STARTS WITH $base AND cls.slug IS NOT NULL
RETURN head(cls.slug) AS slug
"""

_CONTEXT_ALL_ENTRIES_QUERY = """
MATCH (n:Resource)-[:CATEGORY]->(cls:Resource)
WHERE cls.uri STARTS WITH $base
RETURN
  n.uri              AS iri,
  head(n.slug)       AS slug,
  head(n.source)     AS source,
  head(n.label)      AS label,
  head(n.definition) AS definition,
  head(n.polarity)   AS polarity,
  head(n.instrument) AS instrument,
  head(n.comment)    AS comment,
  head(cls.slug)     AS category
"""

_CONTEXT_FIELDS: tuple[str, ...] = (
    "slug",
    "source",
    "label",
    "definition",
    "polarity",
    "instrument",
    "comment",
    "category",
)


def context_iri_exists(iri: str, session: Any) -> bool:
    """Return True iff the IRI is a Context-dictionary entry."""
    return session.run(_CONTEXT_HAS_QUERY, iri=iri).single() is not None


def fetch_context_entry(iri: str, session: Any) -> dict[str, Any]:
    """Return the Context entry by IRI, or an empty dict when absent.

    Null fields are omitted so callers can tell missing from set.
    """
    row = session.run(_CONTEXT_GET_QUERY, iri=iri, base=CONTEXT_CATEGORY_BASE).single()
    if row is None:
        return {}
    return {k: row[k] for k in _CONTEXT_FIELDS if row.get(k) is not None}


def fetch_context_categories(session: Any) -> frozenset[str]:
    """Return the set of Context category slugs declared in the graph."""
    return frozenset(
        rec["slug"]
        for rec in session.run(_CONTEXT_CATEGORIES_QUERY, base=CONTEXT_CATEGORY_BASE)
        if rec["slug"]
    )


def fetch_all_context_entries(session: Any) -> list[dict[str, Any]]:
    """Return every Context entry with its IRI, category, and core fields."""
    out: list[dict[str, Any]] = []
    for rec in session.run(_CONTEXT_ALL_ENTRIES_QUERY, base=CONTEXT_CATEGORY_BASE):
        entry = {"iri": rec["iri"]}
        for k in _CONTEXT_FIELDS:
            v = rec.get(k)
            if v is not None:
                entry[k] = v
        out.append(entry)
    return out
