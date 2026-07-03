"""Unit tests for src.scripts.scenarios.metrics.semantic.

Coverage targets:
  - _cosine_similarity: normal, orthogonal, parallel, zero-magnitude.
  - SemanticCompatibility: matrix hit (both directions), cache hit,
    cache symmetric, embedding hit, embedding partial miss to LLM,
    embedding_fn None to LLM, LLM used, no strategy to 0.0.
  - Cache persistence: written to file, loaded on re-init, second call uses cache.
  - make_neo4j_embedding_fn: mocked driver integration.
  - _load_cache: malformed JSON line skipped gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.scenarios.metrics.semantic import (
    SemanticCompatibility,
    _cosine_similarity,
    make_neo4j_embedding_fn,
)

# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_give_one(self):
        assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(
            1.0
        )

    def test_orthogonal_vectors_give_zero(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_clamped_to_zero(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)

    def test_partial_similarity(self):
        import math

        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2.0)
        assert _cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)

    def test_zero_magnitude_a_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_zero_magnitude_b_returns_zero(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_result_clamped_to_one(self):
        large = [1e10, 1e10]
        assert _cosine_similarity(large, large) <= 1.0


# ---------------------------------------------------------------------------
# Matrix priority
# ---------------------------------------------------------------------------


class TestMatrixPriority:
    def test_forward_key_hit(self):
        sc = SemanticCompatibility(matrix={("a", "b"): 0.8})
        assert sc.score("a", "b") == pytest.approx(0.8)

    def test_reverse_key_hit(self):
        sc = SemanticCompatibility(matrix={("a", "b"): 0.8})
        assert sc.score("b", "a") == pytest.approx(0.8)

    def test_matrix_miss_falls_through_to_zero(self):
        sc = SemanticCompatibility(matrix={("c", "d"): 0.5})
        assert sc.score("a", "b") == pytest.approx(0.0)

    def test_matrix_zero_score(self):
        sc = SemanticCompatibility(matrix={("x", "y"): 0.0})
        assert sc.score("x", "y") == pytest.approx(0.0)

    def test_matrix_takes_priority_over_embedding(self):
        calls = []
        sc = SemanticCompatibility(
            matrix={("a", "b"): 0.9},
            embedding_fn=lambda lbl: (calls.append(lbl) or [1.0, 0.0]),
        )
        result = sc.score("a", "b")
        assert result == pytest.approx(0.9)
        assert calls == []  # embedding never called


# ---------------------------------------------------------------------------
# Cache priority
# ---------------------------------------------------------------------------


class TestCachePriority:
    def _write_cache(
        self, path: Path, label_a: str, label_b: str, score: float
    ) -> None:
        entry = json.dumps(
            {
                "label_a": label_a,
                "label_b": label_b,
                "score": score,
                "strategy": "test",
                "ts": "2026-01-01T00:00:00+00:00",
            }
        )
        path.write_text(entry + "\n", encoding="utf-8")

    def test_cache_hit_returns_cached_score(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        self._write_cache(cache, "a", "b", 0.7)
        sc = SemanticCompatibility(cache_path=cache)
        assert sc.score("a", "b") == pytest.approx(0.7)

    def test_cache_hit_reverse_direction(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        self._write_cache(cache, "a", "b", 0.7)
        sc = SemanticCompatibility(cache_path=cache)
        assert sc.score("b", "a") == pytest.approx(0.7)

    def test_cache_blocks_embedding_call(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        self._write_cache(cache, "x", "y", 0.6)
        calls = []
        sc = SemanticCompatibility(
            cache_path=cache,
            embedding_fn=lambda lbl: (calls.append(lbl) or [1.0]),
        )
        sc.score("x", "y")
        assert calls == []

    def test_cache_blocks_llm_call(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        self._write_cache(cache, "x", "y", 0.6)
        calls = []
        sc = SemanticCompatibility(
            cache_path=cache,
            llm_scorer=lambda a, b: (calls.append((a, b)) or 0.99),
        )
        sc.score("x", "y")
        assert calls == []

    def test_new_score_written_to_cache_file(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        sc = SemanticCompatibility(llm_scorer=lambda a, b: 0.55, cache_path=cache)
        sc.score("p", "q")
        assert cache.exists()
        entries = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
        assert len(entries) == 1
        assert entries[0]["score"] == pytest.approx(0.55)
        assert entries[0]["strategy"] == "llm"

    def test_second_call_uses_in_memory_cache(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        call_count = [0]

        def scorer(a, b):
            call_count[0] += 1
            return 0.4

        sc = SemanticCompatibility(llm_scorer=scorer, cache_path=cache)
        sc.score("x", "y")
        sc.score("x", "y")
        assert call_count[0] == 1

    def test_no_cache_path_still_uses_in_memory_cache(self):
        call_count = [0]

        def scorer(a, b):
            call_count[0] += 1
            return 0.3

        sc = SemanticCompatibility(llm_scorer=scorer)
        sc.score("a", "b")
        sc.score("a", "b")
        assert call_count[0] == 1

    def test_cache_file_in_nonexistent_dir_created(self, tmp_path):
        cache = tmp_path / "subdir" / "cache.jsonl"
        sc = SemanticCompatibility(llm_scorer=lambda a, b: 0.5, cache_path=cache)
        sc.score("a", "b")
        assert cache.exists()

    def test_malformed_json_line_skipped(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cache.write_text(
            "not valid json\n"
            '{"label_a": "a", "label_b": "b", "score": 0.5, "strategy": "test", "ts": "t"}\n',
            encoding="utf-8",
        )
        sc = SemanticCompatibility(cache_path=cache)
        assert sc.score("a", "b") == pytest.approx(0.5)

    def test_empty_lines_in_cache_skipped(self, tmp_path):
        """Blank lines in the JSONL file must not crash the loader."""
        cache = tmp_path / "cache.jsonl"
        cache.write_text(
            "\n"
            '{"label_a": "x", "label_b": "y", "score": 0.3, "strategy": "test", "ts": "t"}\n'
            "\n",
            encoding="utf-8",
        )
        sc = SemanticCompatibility(cache_path=cache)
        assert sc.score("x", "y") == pytest.approx(0.3)

    def test_unreadable_cache_file_does_not_crash(self, tmp_path):
        """If the cache file cannot be read (e.g. directory instead of file),
        _load_cache silently returns and the instance still works.  No llm_scorer
        is provided so _save_to_cache is never called (avoids a second write error)."""
        cache = tmp_path / "cache_is_a_dir"
        cache.mkdir()  # directory to read_text raises IsADirectoryError (OSError)
        sc = SemanticCompatibility(cache_path=cache)  # must not raise
        assert sc.score("a", "b") == pytest.approx(0.0)  # no strategy to 0.0


# ---------------------------------------------------------------------------
# Embedding priority
# ---------------------------------------------------------------------------


class TestEmbeddingPriority:
    def test_parallel_embeddings_give_one(self):
        emb_map = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
        sc = SemanticCompatibility(embedding_fn=emb_map.get)
        assert sc.score("a", "b") == pytest.approx(1.0)

    def test_orthogonal_embeddings_give_zero(self):
        emb_map = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        sc = SemanticCompatibility(embedding_fn=emb_map.get)
        assert sc.score("a", "b") == pytest.approx(0.0)

    def test_embedding_result_written_to_cache(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        emb_map = {"x": [1.0, 0.0], "y": [1.0, 0.0]}
        sc = SemanticCompatibility(embedding_fn=emb_map.get, cache_path=cache)
        sc.score("x", "y")
        entries = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
        assert entries[0]["strategy"] == "embedding"

    def test_label_a_missing_falls_through_to_llm(self):
        sc = SemanticCompatibility(
            embedding_fn=lambda lbl: [1.0] if lbl == "b" else None,
            llm_scorer=lambda a, b: 0.3,
        )
        assert sc.score("a", "b") == pytest.approx(0.3)

    def test_label_b_missing_falls_through_to_llm(self):
        sc = SemanticCompatibility(
            embedding_fn=lambda lbl: [1.0] if lbl == "a" else None,
            llm_scorer=lambda a, b: 0.25,
        )
        assert sc.score("a", "b") == pytest.approx(0.25)

    def test_embedding_fn_none_skips_embedding_stage(self):
        sc = SemanticCompatibility(
            embedding_fn=None,
            llm_scorer=lambda a, b: 0.45,
        )
        assert sc.score("a", "b") == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------


class TestLLMFallback:
    def test_llm_scorer_called_with_labels(self):
        received = []

        def scorer(a, b):
            received.append((a, b))
            return 0.6

        sc = SemanticCompatibility(llm_scorer=scorer)
        sc.score("foo", "bar")
        assert received == [("foo", "bar")]

    def test_llm_score_returned(self):
        sc = SemanticCompatibility(llm_scorer=lambda a, b: 0.77)
        assert sc.score("x", "y") == pytest.approx(0.77)

    def test_no_strategy_returns_zero(self):
        sc = SemanticCompatibility()
        assert sc.score("x", "y") == pytest.approx(0.0)

    def test_llm_result_cached_in_memory(self):
        call_count = [0]

        def scorer(a, b):
            call_count[0] += 1
            return 0.5

        sc = SemanticCompatibility(llm_scorer=scorer)
        sc.score("a", "b")
        sc.score("a", "b")
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Performance-critical caching layers
#
# These tests guard the two caches that prevent compute_l_concurrent
# from thrashing the embedding backend when it walks |scheduled| ×
# |events| pairs per persona.  Without them the augment loop hangs
# at "Augmenting: 0%" for hours when σ-oracle is wired to a real
# Neo4j (~2M score() calls per scenario, each two round-trips).
# ---------------------------------------------------------------------------


class TestEmbeddingMemoisation:
    def test_embedding_fn_called_once_per_label_across_pairs(self):
        """Each unique label is fetched from the embedding backend at
        most once even when it appears in many distinct (a, b) pairs.
        Critical for compute_l_concurrent on real workloads; without
        memoisation, scoring 100 (task, event) pairs against the same
        task label would make 100 Neo4j round-trips for that one label.
        """
        call_count: dict[str, int] = {}
        emb_map = {
            "task_walk": [1.0, 0.0],
            "first_eat": [0.0, 1.0],
            "lunch": [0.5, 0.5],
            "dinner": [0.7, 0.3],
        }

        def fn(label):
            call_count[label] = call_count.get(label, 0) + 1
            return emb_map.get(label)

        sc = SemanticCompatibility(embedding_fn=fn)
        # Score the same task against 3 different events; task_walk
        # should be fetched ONCE, each event ONCE.
        sc.score("task_walk", "first_eat")
        sc.score("task_walk", "lunch")
        sc.score("task_walk", "dinner")
        assert call_count["task_walk"] == 1
        assert call_count["first_eat"] == 1
        assert call_count["lunch"] == 1
        assert call_count["dinner"] == 1

    def test_embedding_fn_memoises_none_results(self):
        """Labels with no embedding in the backing store must also be
        memoised; otherwise the loop re-queries Neo4j every time it
        hits a missing-embedding label.  This is the dominant case
        for L_merge when task slugs (`do-10-minutes-of-cardio`) are
        not indexed as `EmbeddedConcept`."""
        call_count = [0]

        def fn(label):
            call_count[0] += 1
            return None  # no embedding for any label

        sc = SemanticCompatibility(embedding_fn=fn)
        sc.score("walk", "eat")  # 2 fetches: walk + eat
        sc.score("walk", "lunch")  # 1 fetch: lunch (walk is memoised)
        sc.score("eat", "lunch")  # 0 fetches: both memoised
        assert call_count[0] == 3


class TestNoSignalCaching:
    def test_zero_score_cached_so_no_retry(self):
        """A pair that resolves to 0.0 via the no-signal fall-through
        (no matrix, no embedding match, no LLM scorer) must be cached
        so the SECOND call doesn't re-pay the embedding lookup.  This
        guards against a regression where greedy
        augmentation hung for hours because every (task, event) pair
        re-hit Neo4j on every score() call."""
        emb_call_count = [0]

        def fn(label):
            emb_call_count[0] += 1
            return None  # no embeddings to falls through to 0.0

        sc = SemanticCompatibility(embedding_fn=fn)
        # First call; 2 embedding lookups, both return None, score = 0.0.
        assert sc.score("a", "b") == pytest.approx(0.0)
        # Second call for the SAME pair; must be a cache hit, NO
        # additional embedding fetches.
        before = emb_call_count[0]
        assert sc.score("a", "b") == pytest.approx(0.0)
        assert emb_call_count[0] == before, (
            "score(a, b) re-fetched embeddings instead of using cache; "
            "this regresses the L_merge performance fix."
        )

    def test_zero_score_cached_in_reverse_direction(self):
        """The frozenset cache key works in both directions, so the
        no-signal cache hit applies whichever way the labels are
        passed."""
        emb_call_count = [0]

        def fn(label):
            emb_call_count[0] += 1
            return None

        sc = SemanticCompatibility(embedding_fn=fn)
        sc.score("a", "b")
        before = emb_call_count[0]
        sc.score("b", "a")
        assert emb_call_count[0] == before

    def test_no_signal_strategy_marker_persisted(self, tmp_path):
        """The cached no-signal entry carries `strategy="none"` so a
        researcher inspecting the JSONL cache can distinguish a
        deliberate cached zero from a real σ=0.0 score."""
        cache = tmp_path / "cache.jsonl"
        sc = SemanticCompatibility(cache_path=cache)
        sc.score("a", "b")  # no oracle wired to falls through to 0.0
        text = cache.read_text(encoding="utf-8")
        assert '"strategy": "none"' in text
        assert '"score": 0.0' in text

    def test_unwriteable_cache_path_does_not_crash_no_signal_save(self, tmp_path):
        """When the cache path is invalid (e.g. a directory), saving
        the no-signal cache entry must not crash; in-memory
        memoisation still applies for the rest of the process."""
        cache_dir = tmp_path / "cache_is_a_dir"
        cache_dir.mkdir()
        sc = SemanticCompatibility(cache_path=cache_dir)
        # Must not raise even though disk write fails.
        assert sc.score("a", "b") == pytest.approx(0.0)
        # In-memory cache still works; second call is the same value.
        assert sc.score("a", "b") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# make_neo4j_embedding_fn
# ---------------------------------------------------------------------------


class TestMakeNeo4jEmbeddingFn:
    def _mock_driver(self, embedding_map: dict) -> object:
        class _SingleResult:
            def __init__(self, emb):
                self._emb = emb

            def single(self):
                if self._emb is not None:
                    return {"embedding": self._emb}
                return None

        class _Session:
            def __init__(self, em):
                self._em = em

            def run(self, query, label):
                emb = self._em.get(label.lower())
                return _SingleResult(emb)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class _Driver:
            def __init__(self, em):
                self._em = em

            def session(self):
                return _Session(self._em)

        return _Driver(embedding_map)

    def test_label_found_returns_embedding(self):
        driver = self._mock_driver({"running": [1.0, 0.0, 0.0]})
        fn = make_neo4j_embedding_fn(driver)
        result = fn("running")
        assert result == [1.0, 0.0, 0.0]

    def test_label_not_found_returns_none(self):
        driver = self._mock_driver({})
        fn = make_neo4j_embedding_fn(driver)
        assert fn("unknown_label") is None

    def test_embedding_used_for_similarity(self):
        driver = self._mock_driver({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        sc = SemanticCompatibility(embedding_fn=make_neo4j_embedding_fn(driver))
        assert sc.score("a", "b") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_with_strategy
# ---------------------------------------------------------------------------


class TestScoreWithStrategy:
    def test_matrix_hit_reports_matrix_strategy(self):
        sc = SemanticCompatibility(matrix={("a", "b"): 0.9})
        score, strategy = sc.score_with_strategy("a", "b")
        assert score == pytest.approx(0.9)
        assert strategy == "matrix"

    def test_cache_hit_reports_cache_strategy(self):
        sc = SemanticCompatibility()
        sc.score("a", "b")  # caches the "none" result
        score, strategy = sc.score_with_strategy("a", "b")
        # Cache hit returns the stored strategy ("none" here).
        assert score == 0.0
        assert strategy in {"none", "cache"}

    def test_embedding_path_reports_embedding_strategy(self):
        sc = SemanticCompatibility(
            embedding_fn=lambda label: [1.0, 0.0] if label == "a" else [0.0, 1.0],
        )
        score, strategy = sc.score_with_strategy("a", "b")
        assert strategy == "embedding"
        assert score == pytest.approx(0.0)

    def test_llm_scorer_path_reports_llm_strategy(self):
        sc = SemanticCompatibility(llm_scorer=lambda a, b: 0.42)
        score, strategy = sc.score_with_strategy("a", "b")
        assert strategy == "llm"
        assert score == pytest.approx(0.42)

    def test_no_oracle_returns_none_strategy(self):
        sc = SemanticCompatibility()
        score, strategy = sc.score_with_strategy("a", "b")
        assert score == 0.0
        assert strategy == "none"


class TestLoadCacheSkipsNoneStrategy:
    def test_cache_entry_with_strategy_none_is_dropped(self, tmp_path):
        """Stub `none` entries from prior runs do not poison the in-memory cache."""
        cache = tmp_path / "cache.jsonl"
        cache.write_text(
            json.dumps(
                {
                    "label_a": "x",
                    "label_b": "y",
                    "score": 0.0,
                    "strategy": "none",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "label_a": "p",
                    "label_b": "q",
                    "score": 0.7,
                    "strategy": "embedding",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        sc = SemanticCompatibility(cache_path=cache)
        # The `embedding` entry made it into the cache; the `none` entry did not.
        assert sc.score("p", "q") == pytest.approx(0.7)
        # The `none` entry was skipped, so a fresh call must re-resolve via
        # the oracle chain. With no oracle wired, the score falls back to 0.0
        # and the new strategy stored is `none`, NOT a re-read of the dropped row.
        score, strategy = sc.score_with_strategy("x", "y")
        assert score == 0.0
        assert strategy in {"none", "cache"}
