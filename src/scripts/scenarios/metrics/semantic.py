"""Semantic compatibility σ(η_a, η_b) ∈ [0, 1] between two eventuality labels.

σ is the semantic compatibility primitive used by the merging loss L_mrg
(σ_act vs σ_best of a placed task ê_k against the eventualities it is
co-scheduled with) and by the persona-stage alignment leg of the preference
loss L_pref. Higher σ means the two activity labels η fit together better.

Computes the score for a label pair using a priority chain:

    1. Explicit `matrix` override (constant-time dict lookup).
    2. Disk cache (`semantic_cache.jsonl`).
    3. Embedding cosine similarity via an injectable `embedding_fn`.
    4. `llm_scorer` callable fallback.

If no strategy is available the score defaults to 0.0.

Scores are symmetric: σ(a, b) == σ(b, a).  The cache key is
`frozenset({label_a, label_b})`.

Production usage
----------------
Provide an `embedding_fn` built from a Neo4j driver::

    from src.scripts.scenarios.metrics.semantic import (
        SemanticCompatibility,
        make_neo4j_embedding_fn,
    )
    from src.graphrag.neo4j_client import make_driver
    from src.graphrag.config import Neo4jSettings

    driver = make_driver(Neo4jSettings.from_env())
    sc = SemanticCompatibility(
        embedding_fn=make_neo4j_embedding_fn(driver),
        llm_scorer=lambda a, b: call_llm(a, b),
        cache_path=Path("./output/scenario/evaluation/semantic_cache.jsonl"),
    )

Unit-testing
------------
Inject a simple lambda for `embedding_fn` and `llm_scorer`; no Neo4j
or LLM required.
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

EmbeddingFn = Callable[[str], list[float] | None]
LLMScorerFn = Callable[[str, str], float]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1], clamped to avoid floating-point overshoot."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def make_neo4j_embedding_fn(driver) -> EmbeddingFn:
    """Create an `EmbeddingFn` that fetches vectors from a Neo4j `EmbeddedConcept` index.

    The query matches nodes whose `prefLabel` or `label` array contains an
    element equal to `label` (case-insensitive) and returns the stored
    `embedding` vector.
    """

    def fetch(label: str) -> list[float] | None:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n:EmbeddedConcept)
                WHERE any(lbl IN
                      coalesce(n.prefLabel, []) + coalesce(n.label, [])
                      WHERE toLower(toString(lbl)) = toLower($label))
                RETURN n.embedding AS embedding
                LIMIT 1
                """,
                label=label,
            ).single()
            if result and result["embedding"]:
                return list(result["embedding"])
        return None

    return fetch


@dataclass
class _CacheEntry:
    score: float
    strategy: str


class SemanticCompatibility:
    """Computes σ(η_a, η_b) via a four-level priority chain.

    Args:
        matrix: explicit `{(label_a, label_b): float}` overrides.
                Symmetric: `(a, b)` and `(b, a)` are both checked.
        embedding_fn: callable `label to list[float] | None`; produces the
                      embedding vector for a label.  `None` skips this stage.
        llm_scorer: callable `(label_a, label_b) to float`; called when
                    neither embedding is found.  `None` skips LLM stage.
        cache_path: path to a `semantic_cache.jsonl` file.  Loaded on init
                    if it already exists; new entries are appended atomically.
    """

    def __init__(
        self,
        *,
        matrix: dict[tuple[str, str], float] | None = None,
        embedding_fn: EmbeddingFn | None = None,
        llm_scorer: LLMScorerFn | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._matrix: dict[tuple[str, str], float] = matrix or {}
        self._embedding_fn = embedding_fn
        self._llm_scorer = llm_scorer
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[frozenset[str], _CacheEntry] = {}
        # Per-label embedding memoization; avoids re-querying the
        # backing store (Neo4j round-trip, embedding API call, …) for
        # the same label across many score() pairs.  `None` is a
        # legitimate sentinel meaning "no embedding for this label",
        # so the dict membership check (rather than truthiness) is
        # what gates the re-fetch.
        self._embedding_memo: dict[str, list[float] | None] = {}
        if self._cache_path is not None and self._cache_path.exists():
            self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_with_strategy(self, label_a: str, label_b: str) -> tuple[float, str]:
        """Return `(score, strategy)` from the same chain as `score()`.

        `strategy` is one of `matrix`, `cache`, `embedding`, `llm`, `none`
        and lets callers distinguish a real σ from the stub `none`
        default (no oracle could resolve the pair).
        """
        for key in ((label_a, label_b), (label_b, label_a)):
            if key in self._matrix:
                return float(self._matrix[key]), "matrix"
        cache_key = frozenset({label_a, label_b})
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            return entry.score, entry.strategy or "cache"
        result = self._try_embedding(label_a, label_b)
        if result is not None:
            sc, strategy = result
            self._save_to_cache(cache_key, label_a, label_b, sc, strategy)
            return sc, strategy
        if self._llm_scorer is not None:
            sc = float(self._llm_scorer(label_a, label_b))
            self._save_to_cache(cache_key, label_a, label_b, sc, "llm")
            return sc, "llm"
        self._save_to_cache(cache_key, label_a, label_b, 0.0, "none")
        return 0.0, "none"

    def score(self, label_a: str, label_b: str) -> float:
        """Return σ(label_a, label_b) via the priority chain."""
        # 1. Matrix override (both directions)
        for key in ((label_a, label_b), (label_b, label_a)):
            if key in self._matrix:
                return float(self._matrix[key])

        # 2. Cache (in-memory + disk-loaded).
        cache_key = frozenset({label_a, label_b})
        if cache_key in self._cache:
            return self._cache[cache_key].score

        # 3. Embedding cosine similarity.
        result = self._try_embedding(label_a, label_b)
        if result is not None:
            sc, strategy = result
            self._save_to_cache(cache_key, label_a, label_b, sc, strategy)
            return sc

        # 4. LLM fallback.
        if self._llm_scorer is not None:
            sc = float(self._llm_scorer(label_a, label_b))
            self._save_to_cache(cache_key, label_a, label_b, sc, "llm")
            return sc

        # 5. No-signal default; cache the 0.0 too, otherwise
        # workloads that score the same (label_a, label_b) pair
        # millions of times (compute_l_concurrent walks |scheduled| ×
        # |events| per persona) would re-pay the embedding fetch
        # round-trip on every call.  The cached 0.0 is keyed the
        # same way as a real hit; `strategy="none"` lets a
        # researcher distinguish it from a genuine zero score.
        self._save_to_cache(cache_key, label_a, label_b, 0.0, "none")
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embedding_for(self, label: str) -> list[float] | None:
        """Memoized wrapper around `self._embedding_fn`.

        The first call for a label hits the backing store; subsequent
        calls return the memoized value (which may be `None`,
        captured deliberately so we don't re-fetch labels that have
        no embedding indexed).  Caller-supplied `embedding_fn` is
        never invoked when it is `None`.
        """
        if self._embedding_fn is None:  # pragma: no cover - defensive
            # Never reached: _try_embedding short-circuits when
            # embedding_fn is None.  Kept as a safety net for any
            # future direct caller that bypasses _try_embedding.
            return None
        if label in self._embedding_memo:
            return self._embedding_memo[label]
        emb = self._embedding_fn(label)
        self._embedding_memo[label] = emb
        return emb

    def _try_embedding(self, label_a: str, label_b: str) -> tuple[float, str] | None:
        """Attempt embedding cosine similarity.  Returns (score, strategy) or None."""
        if self._embedding_fn is None:
            return None
        emb_a = self._embedding_for(label_a)
        emb_b = self._embedding_for(label_b)
        if emb_a is None or emb_b is None:
            return None
        return _cosine_similarity(emb_a, emb_b), "embedding"

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        try:
            raw = self._cache_path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                strategy = entry.get("strategy", "")
                # `none` entries are stub fall-throughs from a prior run
                # where neither embedding nor LLM scorer was wired in.
                # Re-loading them would poison a later run that has a
                # working backend; skip so the chain re-resolves.
                if strategy == "none":
                    continue
                key: frozenset[str] = frozenset({entry["label_a"], entry["label_b"]})
                self._cache[key] = _CacheEntry(
                    score=float(entry["score"]),
                    strategy=strategy,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

    def _save_to_cache(
        self,
        cache_key: frozenset[str],
        label_a: str,
        label_b: str,
        score: float,
        strategy: str,
    ) -> None:
        # In-memory cache is the hot path; always populated.  Disk
        # persistence is best-effort; an unwriteable path (read-only
        # FS, directory in place of file, …) must NOT take down the
        # scoring loop.
        self._cache[cache_key] = _CacheEntry(score=score, strategy=strategy)
        if self._cache_path is None:
            return
        record = json.dumps(
            {
                "label_a": label_a,
                "label_b": label_b,
                "score": score,
                "strategy": strategy,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as fh:
                fh.write(record + "\n")
                fh.flush()
        except OSError:
            # Unwriteable cache path; score is still memoized
            # in-memory for the rest of this process's lifetime.
            pass
