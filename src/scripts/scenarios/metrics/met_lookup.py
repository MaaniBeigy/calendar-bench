"""Resolve an arbitrary activity label to its closest HumanActivities MET value.

The HumanActivities ontology carries 1,210 named activities, each tagged
with `ha:metValue` and embedded into the same vector index. `MetLookup`
answers: given a free-form activity description, what is the most-likely
MET value and which HumanActivities URI grounds it? It does so by:

  1. Embedding the query text via the supplied `embedder`.
  2. Calling Neo4j's `db.index.vector.queryNodes` against the
     `concept_embedding_index` and filtering to URIs starting with
     the HumanActivities prefix.
  3. Reading `head(metValue)` off the top-1 hit (n10s stores literals
     as 1-element lists when `handleMultival: 'ARRAY'` is set).
  4. Returning `(met_value, matched_uri, cosine)` if cosine clears
     `min_cosine`, else `(None, None, None)`.

`MatchedActivityMetIndex` shortcuts this whole pipeline for tasks that
carry curator-authored `hb:matchedActivity` triples: looks up the
matched activity's MET in the loaded HumanActivities TTL once and
serves the value at zero round-trip cost on every resolve.

A JSONL cache mirrors the resolved triplets so re-runs avoid the cost
of embedding + Neo4j round-trip for queries already seen.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

#: HumanActivities URI prefix; every activity carrying `ha:metValue`.
HUMAN_ACTIVITIES_PREFIX: str = "https://w3id.org/calendar-bench/human-activities/"

#: Neo4j vector index built by `build_embeddings.py`.
_VECTOR_INDEX_NAME: str = "concept_embedding_index"

#: Cypher: ANN over the vector index restricted to HumanActivities URIs.
#: The `score` returned by `db.index.vector.queryNodes` is cosine
#: similarity in [0, 1].  `head(n.metValue)` unwraps n10s' 1-element
#: list storage.
_QUERY = """
CALL db.index.vector.queryNodes($index, $top_k, $vector)
YIELD node, score
WITH node, score
WHERE node.uri STARTS WITH $prefix
  AND node.metValue IS NOT NULL
WITH node, score, head(node.metValue) AS met
WHERE met IS NOT NULL
RETURN node.uri AS uri, toFloat(met) AS met, score
ORDER BY score DESC
LIMIT 1
"""


@dataclass(frozen=True)
class MetMatch:
    """One resolved (query to activity) lookup record.

    Fields:
        met: the matched activity's MET value, or `None` when no match
            cleared the cosine threshold or the bridge had no entry.
        uri: the matched HumanActivities URI, or `None`.
        cosine: cosine similarity of the match in [0, 1], or `None`.
        source: where the match came from. One of `bridge` (curated
            `hb:matchedActivity` lookup), `embedding` (vector search),
            or `none` (no match).
    """

    met: float | None
    uri: str | None
    cosine: float | None
    source: str = "none"


def load_activity_met_index(ttl_path: Path) -> dict[str, float]:
    """Return `{ha-act IRI: metValue}` from a HumanActivities TTL.

    Reads every `ha:metValue` triple under the activity prefix; values
    are coerced to float. Activities without a numeric `metValue` are
    skipped.
    """
    from rdflib import Graph, URIRef

    met_predicate = URIRef("https://w3id.org/calendar-bench/human-activities/metValue")
    graph = Graph()
    graph.parse(ttl_path.as_posix(), format="turtle")
    out: dict[str, float] = {}
    for subject, _, value in graph.triples((None, met_predicate, None)):
        iri = str(subject)
        if not iri.startswith(HUMAN_ACTIVITIES_PREFIX + "activity/"):
            continue
        try:
            out[iri] = float(value)
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class MatchedActivityMetIndex:
    """Curator-authored task to MET lookup, populated from JSONL + TTL.

    `task_iri_to_activities` maps each task to its `hb:matchedActivity`
    IRIs; `activity_iri_to_met` carries the raw MET per activity. The
    `met_for_task` method returns the MET of the best (highest) candidate.
    """

    task_iri_to_activities: dict[str, tuple[str, ...]]
    activity_iri_to_met: dict[str, float]

    def met_for_task(self, task_iri: str) -> tuple[float | None, str | None]:
        """Return `(met, matched_activity_iri)` or `(None, None)` on miss."""
        if not task_iri:
            return (None, None)
        activities = self.task_iri_to_activities.get(task_iri)
        if not activities:
            return (None, None)
        best_met: float | None = None
        best_uri: str | None = None
        for iri in activities:
            met = self.activity_iri_to_met.get(iri)
            if met is None:
                continue
            if best_met is None or met > best_met:
                best_met = met
                best_uri = iri
        return (best_met, best_uri)


def load_matched_activity_index(
    matched_jsonl: Path,
    ha_ttl: Path,
) -> MatchedActivityMetIndex:
    """Build a :class:`MatchedActivityMetIndex` from on-disk artefacts."""
    met_index = load_activity_met_index(ha_ttl)
    task_map: dict[str, tuple[str, ...]] = {}
    for line in matched_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        iris = tuple(record.get("matched_activity_iris") or ())
        if not iris:
            continue
        task_map[str(record["task_iri"])] = iris
    return MatchedActivityMetIndex(
        task_iri_to_activities=task_map,
        activity_iri_to_met=met_index,
    )


class MetLookup:
    """Look up the MET value most-likely associated with a free-form label.

    Args:
        driver: open Neo4j driver.  `None` is allowed for tests that
            only exercise the cache; live lookups will then return an
            empty :class:`MetMatch`.
        embedder: callable / object exposing `embed_query(text) -> list[float]`.
            `None` disables live lookups (cache-only mode).
        cache_path: path to a `met_cache.jsonl` file.  Loaded on init
            if it already exists; new entries append atomically.
        min_cosine: minimum cosine similarity for a match to be accepted.
            Below the threshold the lookup returns `MetMatch(None, None, None)`.
        prefix: HumanActivities URI prefix (overridable for tests).
        top_k: how many ANN candidates to fetch before applying the
            HumanActivities prefix filter.  Default 16; enough head-room
            for the WHERE to find one HumanActivities hit even when the
            top-1 cosine result is from a different ontology.
    """

    def __init__(
        self,
        driver: Any = None,
        embedder: Any = None,
        *,
        cache_path: Path | None = None,
        min_cosine: float = 0.55,
        prefix: str = HUMAN_ACTIVITIES_PREFIX,
        top_k: int = 16,
        bridge: MatchedActivityMetIndex | None = None,
    ) -> None:
        self._driver = driver
        self._embedder = embedder
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._min_cosine = float(min_cosine)
        self._prefix = prefix
        self._top_k = int(top_k)
        self._bridge = bridge
        self._cache: dict[str, MetMatch] = {}
        self._cache_query_text: dict[str, str] = {}
        self._task_cache: dict[str, MetMatch] = {}
        if self._cache_path is not None and self._cache_path.exists():
            self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def met(self, query_text: str) -> MetMatch:
        """Return the best :class:`MetMatch` for `query_text`.

        Cache hit to return immediately. Cache miss to embed and search;
        write the result to disk before returning. When the embedder or
        driver is missing the lookup returns an empty match.
        """
        if not query_text or not query_text.strip():
            return MetMatch(met=None, uri=None, cosine=None, source="none")
        normalized = query_text.strip()
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        if self._embedder is None or self._driver is None:
            empty = MetMatch(met=None, uri=None, cosine=None, source="none")
            self._save_to_cache(normalized, empty)
            return empty

        vector = self._embedder.embed_query(normalized)
        record = self._run_query(vector)
        if record is None:
            match = MetMatch(met=None, uri=None, cosine=None, source="none")
        else:
            cosine = float(record.get("score", 0.0))
            if cosine < self._min_cosine:
                match = MetMatch(met=None, uri=None, cosine=cosine, source="none")
            else:
                match = MetMatch(
                    met=float(record["met"]),
                    uri=str(record["uri"]),
                    cosine=cosine,
                    source="embedding",
                )
        self._save_to_cache(normalized, match)
        return match

    def met_for_task(self, task_iri: str, *, fallback_query: str = "") -> MetMatch:
        """Return MET for a task IRI; prefer the curated bridge, then embedding.

        Tries the matched_activity bridge first (zero round-trips when
        the task is curated); on miss, falls back to a rich-text
        embedding lookup against `fallback_query`. Empty inputs return
        an empty match. Per-task results are cached in-memory.
        """
        if not task_iri:
            return (
                self.met(fallback_query)
                if fallback_query
                else MetMatch(met=None, uri=None, cosine=None, source="none")
            )
        if task_iri in self._task_cache:
            return self._task_cache[task_iri]
        if self._bridge is not None:
            met, uri = self._bridge.met_for_task(task_iri)
            if met is not None and uri is not None:
                match = MetMatch(met=met, uri=uri, cosine=None, source="bridge")
                self._task_cache[task_iri] = match
                return match
        match = (
            self.met(fallback_query)
            if fallback_query
            else MetMatch(met=None, uri=None, cosine=None, source="none")
        )
        self._task_cache[task_iri] = match
        return match

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_query(self, vector: list[float]) -> dict[str, Any] | None:
        with self._driver.session() as session:
            result = session.run(
                _QUERY,
                index=_VECTOR_INDEX_NAME,
                top_k=self._top_k,
                vector=list(vector),
                prefix=self._prefix,
            ).single()
        if result is None:
            return None
        return {
            "uri": result.get("uri"),
            "met": result.get("met"),
            "score": result.get("score"),
        }

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        try:
            text = self._cache_path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                query = str(entry["query_text"]).strip()
                self._cache[query] = MetMatch(
                    met=(float(entry["met"]) if entry.get("met") is not None else None),
                    uri=(str(entry["uri"]) if entry.get("uri") is not None else None),
                    cosine=(
                        float(entry["cosine"])
                        if entry.get("cosine") is not None
                        else None
                    ),
                    source=str(entry.get("source", "none")),
                )
                self._cache_query_text[query] = query
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    def _save_to_cache(self, query_text: str, match: MetMatch) -> None:
        self._cache[query_text] = match
        self._cache_query_text[query_text] = query_text
        if self._cache_path is None:
            return
        record = json.dumps(
            {
                "query_text": query_text,
                "met": match.met,
                "uri": match.uri,
                "cosine": match.cosine,
                "source": match.source,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as fh:
            fh.write(record + "\n")
            fh.flush()


def build_query_text(
    *,
    display_name: str,
    description: str = "",
    branch_local_name: str = "",
    duration_minutes: int | None = None,
) -> str:
    """Build the embedder query string for MET lookup.

    Format: `display_name; description; branch_local_name duration N min`
    where empty fields are skipped and the duration suffix is added when
    `duration_minutes` is known.
    """
    parts: list[str] = []
    if display_name and display_name.strip():
        parts.append(display_name.strip())
    if description and description.strip():
        parts.append(description.strip())
    if branch_local_name and branch_local_name.strip():
        parts.append(branch_local_name.strip())
    text = "; ".join(parts)
    if duration_minutes is not None and duration_minutes > 0:
        text = (
            f"{text} duration {int(duration_minutes)} min"
            if text
            else f"duration {int(duration_minutes)} min"
        )
    return text
