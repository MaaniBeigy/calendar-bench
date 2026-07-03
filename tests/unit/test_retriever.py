"""Unit tests for src.graphrag.retriever."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.graphrag.retrieval_filters import AttributeFilter
from src.graphrag.retriever import (
    EMBEDDING_PROPERTY,
    HEALTH_TASK_INSTANCE_PREFIX,
    NODE_LABEL,
    RETRIEVAL_QUERY,
    VECTOR_INDEX_NAME,
    MultiPrefixVectorCypherRetriever,
    _build_retrieval_query,
    _format_result,
    _validate_safe_literal,
    make_retriever,
)


class TestRetrievalQueryReadsDcterms:
    """The retrieval query must surface `dcterms:title` and
    `dcterms:description` so that HealthTasks instance nodes (which
    carry only those two text properties) reach the LLM context."""

    def test_title_path_includes_node_title(self):
        assert (
            "head(coalesce(node.label, node.prefLabel, node.title, [node.uri]))"
            in RETRIEVAL_QUERY
        )

    def test_description_path_includes_node_description(self):
        assert (
            "head(coalesce(node.comment, node.iAO_0000115, "
            "node.definition, node.description, ['']))" in RETRIEVAL_QUERY
        )

    def test_description_size_check_includes_description(self):
        assert (
            "size(coalesce(node.comment, node.iAO_0000115, "
            "node.definition, node.description, []))" in RETRIEVAL_QUERY
        )

    def test_parent_title_path_includes_parent_title(self):
        assert (
            "head(coalesce(parent.label, parent.prefLabel, parent.title, "
            "[parent.uri]))" in RETRIEVAL_QUERY
        )

    def test_neighbor_name_path_includes_neighbor_title(self):
        assert (
            "head(coalesce(neighbor.label, neighbor.prefLabel, "
            "neighbor.title, [neighbor.uri]))" in RETRIEVAL_QUERY
        )


class TestFormatResult:
    def test_returns_context_and_score(self):
        record = MagicMock()
        record.get.side_effect = lambda key: {
            "context": "## Drink A Glass of Water 🫗\nuri: https://example.org/x",
            "score": 0.87,
        }[key]
        item = _format_result(record)
        assert item.content == (
            "## Drink A Glass of Water 🫗\nuri: https://example.org/x"
        )
        assert item.metadata == {"score": 0.87}

    def test_empty_context_falls_back_to_blank_string(self):
        record = MagicMock()
        record.get.side_effect = lambda key: None
        item = _format_result(record)
        assert item.content == ""
        assert item.metadata == {"score": None}


class TestValidateSafeLiteral:
    def test_accepts_uri_like_string(self):
        # Should not raise.
        _validate_safe_literal("https://w3id.org/calendar-bench/health/", context="x")

    def test_rejects_quote(self):
        with pytest.raises(ValueError, match="unsafe character"):
            _validate_safe_literal("hello'world", context="x")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_safe_literal("", context="x")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_safe_literal(123, context="x")  # type: ignore[arg-type]


class TestBuildRetrievalQuery:
    def test_no_filters_returns_baseline(self):
        assert _build_retrieval_query(None) is RETRIEVAL_QUERY
        assert _build_retrieval_query([]) is RETRIEVAL_QUERY

    def test_single_prefix_inserts_any_clause(self):
        out = _build_retrieval_query([HEALTH_TASK_INSTANCE_PREFIX])
        assert (
            f"any(p IN ['{HEALTH_TASK_INSTANCE_PREFIX}'] "
            "WHERE node.uri STARTS WITH p)"
        ) in out
        # Inserted right after the bnode guard so it actually fires
        # before the OPTIONAL MATCHes consume the score.
        assert "AND any(p IN [" in out

    def test_multiple_prefixes_emit_comma_separated_list(self):
        out = _build_retrieval_query(["http://a/", "http://b/"])
        assert "['http://a/', 'http://b/']" in out

    def test_unsafe_prefix_rejected(self):
        with pytest.raises(ValueError, match="unsafe character"):
            _build_retrieval_query(["http://x/'; DROP DATABASE neo4j; //"])

    def test_instance_only_adds_property_guard(self):
        out = _build_retrieval_query(None, instance_only=True)
        assert "node.title IS NOT NULL" in out
        assert "node.estimatedDurationMinutes IS NOT NULL" in out
        assert "node.isConcurrent IS NOT NULL" in out
        # The guard sits in the WHERE chain (uses `AND` join).
        assert "AND (node.title IS NOT NULL" in out

    def test_allowed_branches_emits_label_based_block(self):
        """Branch filter walks the SUBCLASSOF chain from a class node
        whose URI suffix matches one of the instance's labels -
        because n10s on the live graph stores `rdf:type` as a Neo4j
        *label* on the instance (`handleRDFTypes = LABELS`) rather
        than as a separate `RDFTYPE` relationship.  The previous
        `[:RDFTYPE|SUBCLASSOF]` walk found zero edges from any
        instance and silently produced empty retrieval context (the
        `<URI>` placeholder regression)."""
        out = _build_retrieval_query(
            None,
            allowed_branches=[
                "https://w3id.org/calendar-bench/health/NutritionTask",
                "https://w3id.org/calendar-bench/health/PhysicalActivityTask",
            ],
        )
        # The corrected pattern: bind labels(node) once, then walk
        # SUBCLASSOF from "any class with matching URI suffix".
        assert "labels(node) AS _il" in out
        assert "(_cls_b:Resource)-[:SUBCLASSOF*0..10]->(_branch:Resource)" in out
        assert "_branch.uri IN [" in out
        assert "split(_cls_b.uri, '/')[-1] IN _il" in out
        assert "count(DISTINCT _cls_b) AS _branch_hits" in out
        assert "WHERE _branch_hits > 0" in out
        assert "NutritionTask" in out
        assert "PhysicalActivityTask" in out
        # The broken patterns must NOT be emitted.
        assert "RDFTYPE" not in out
        assert "EXISTS {" not in out

    def test_allowed_branches_unsafe_uri_rejected(self):
        with pytest.raises(ValueError, match="unsafe character"):
            _build_retrieval_query(
                None,
                allowed_branches=["http://x/'; DROP //"],
            )

    def test_allowed_levels_emits_label_based_block(self):
        out = _build_retrieval_query(None, allowed_levels=["Level1"])
        assert "labels(node) AS _il" in out
        assert "(_cls_l:Resource)-[:SUBCLASSOF*0..10]->(_level:Resource)" in out
        assert "_level.uri ENDS WITH t" in out
        assert "['Level1']" in out
        assert "split(_cls_l.uri, '/')[-1] IN _il" in out
        assert "count(DISTINCT _cls_l) AS _level_hits" in out
        assert "WHERE _level_hits > 0" in out
        assert "RDFTYPE" not in out
        assert "EXISTS {" not in out

    def test_allowed_levels_unsafe_token_rejected(self):
        with pytest.raises(ValueError, match="unsafe character"):
            _build_retrieval_query(None, allowed_levels=["Level1'; DROP //"])

    def test_branch_and_level_share_one_il_binding(self):
        """When BOTH branch and level filters are active, `_il` is
        bound once (in the branch block) and reused by the level
        block; re-binding it would shadow the carried-through
        `score` projection and break the downstream chain."""
        out = _build_retrieval_query(
            [HEALTH_TASK_INSTANCE_PREFIX],
            allowed_branches=["https://w3id.org/calendar-bench/health/NutritionTask"],
            allowed_levels=["Level1"],
        )
        assert out.count("labels(node) AS _il") == 1

    def test_level_only_binds_il_itself(self):
        """When only the level filter is supplied, the level block
        must bind `_il` itself; there is no preceding branch block
        to inherit from."""
        out = _build_retrieval_query(None, allowed_levels=["Level1"])
        assert out.count("labels(node) AS _il") == 1

    def test_all_filters_combine_correctly(self):
        out = _build_retrieval_query(
            [HEALTH_TASK_INSTANCE_PREFIX],
            allowed_branches=["https://w3id.org/calendar-bench/health/NutritionTask"],
            allowed_levels=["Level1"],
            instance_only=True,
        )
        # WHERE clause carries the simple predicates (prefix + instance).
        assert "STARTS WITH 'bnode://'" in out
        assert "any(p IN ['" in out
        assert "node.title IS NOT NULL" in out
        # Branch + level live in their own OPTIONAL MATCH blocks
        # spliced BEFORE the parents/neighbours OPTIONAL MATCH.
        branch_idx = out.find("count(DISTINCT _cls_b) AS _branch_hits")
        level_idx = out.find("count(DISTINCT _cls_l) AS _level_hits")
        parents_idx = out.find("(parent:Resource)")
        assert 0 < branch_idx < parents_idx
        assert 0 < level_idx < parents_idx


class TestMakeRetriever:
    def test_no_filters_returns_plain_retriever(self):
        """Unfiltered legacy path (`ask.py` and the augmenter's RAG
        calls) returns a plain `VectorCypherRetriever` over the whole
        corpus."""
        with patch("src.graphrag.retriever.VectorCypherRetriever") as ctor:
            make_retriever(MagicMock(), MagicMock())
        ctor.assert_called_once()
        kwargs = ctor.call_args.kwargs
        assert kwargs["retrieval_query"] == RETRIEVAL_QUERY

    def test_one_prefix_uses_wrapper(self):
        """Even with a single prefix the wrapper is preferred; it
        injects the high `effective_search_ratio` that keeps the
        per-prefix WHERE filter from returning zero rows on a
        multi-ontology vector index (the empty-context regression
        where the LLM emitted the `<URI>` placeholder)."""
        with patch("src.graphrag.retriever.VectorCypherRetriever"):
            retriever = make_retriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=[HEALTH_TASK_INSTANCE_PREFIX],
            )
        assert isinstance(retriever, MultiPrefixVectorCypherRetriever)
        assert retriever._uri_prefixes == [HEALTH_TASK_INSTANCE_PREFIX]
        assert len(retriever._retrievers) == 1

    def test_two_prefixes_returns_multi_prefix_retriever(self):
        """Two ontologies in the YAML must trigger the per-prefix
        fan-out wrapper; without it the smaller ontology is crowded
        out of top-K by raw cosine."""
        with patch("src.graphrag.retriever.VectorCypherRetriever"):
            retriever = make_retriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=["http://a/", "http://b/"],
            )
        assert isinstance(retriever, MultiPrefixVectorCypherRetriever)
        assert retriever._uri_prefixes == ["http://a/", "http://b/"]
        assert len(retriever._retrievers) == 2

    def test_filter_only_no_prefix_still_uses_wrapper(self):
        """If a caller asks for branch / level / instance-only filters
        but no prefix, the wrapper is still chosen so the search-ratio
        bump applies.  A sentinel `http` prefix is supplied so every
        candidate URI in the index passes the (very loose) prefix
        check."""
        with patch("src.graphrag.retriever.VectorCypherRetriever"):
            retriever = make_retriever(
                MagicMock(),
                MagicMock(),
                allowed_branches=["http://branch/"],
                instance_only=True,
            )
        assert isinstance(retriever, MultiPrefixVectorCypherRetriever)
        assert retriever._uri_prefixes == ["http"]

    def test_filters_propagate_to_each_per_prefix_retriever(self):
        """When two prefixes trigger the wrapper, every underlying
        retriever must receive the same branch / level / instance-only
        filters baked into its Cypher."""
        with patch("src.graphrag.retriever.VectorCypherRetriever") as ctor:
            make_retriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=["http://a/", "http://b/"],
                allowed_branches=["http://branch/"],
                allowed_levels=["Level1"],
                instance_only=True,
            )
        # Two underlying retrievers, both with the same filter clauses.
        assert ctor.call_count == 2
        for call in ctor.call_args_list:
            cypher = call.kwargs["retrieval_query"]
            assert "http://branch/" in cypher
            assert "Level1" in cypher
            assert "node.title IS NOT NULL" in cypher


class TestMultiPrefixRetriever:
    def test_constructor_rejects_empty_prefixes(self):
        with pytest.raises(ValueError, match="at least one"):
            MultiPrefixVectorCypherRetriever(MagicMock(), MagicMock(), uri_prefixes=[])

    def test_default_prefilter_pool_multiplier_is_high(self):
        """Sanity-check the default; a low multiplier combined with
        restrictive WHERE filters returns zero rows in production
        because `neo4j_graphrag` inserts `WITH node, score LIMIT
        $top_k` BEFORE our retrieval Cypher's WHERE clauses.  The
        multiplier must stay well above 1 so enough candidates
        survive the pre-filter LIMIT and reach our filter."""
        assert MultiPrefixVectorCypherRetriever.DEFAULT_PREFILTER_POOL_MULTIPLIER >= 25

    def test_get_search_results_bumps_top_k_for_underlying_retriever(self):
        """`top_k` is multiplied by the pool multiplier before the
        underlying `VectorCypherRetriever` is called; without this
        the `neo4j_graphrag` pre-filter `LIMIT $top_k` truncates
        candidates to the user-requested `top_k` BEFORE our WHERE
        clauses see them, and the WHERE then narrows the survivors to
        zero (an earlier empty-context regression where the LLM
        emitted the literal `<URI>` placeholder)."""
        rec = MagicMock()
        rec.get.side_effect = lambda key: {"context": "## hi\nuri: http://x/1"}.get(key)
        mock = MagicMock()
        mock.get_search_results.return_value = SimpleNamespace(
            records=[rec], metadata=None
        )
        with patch("src.graphrag.retriever.VectorCypherRetriever", side_effect=[mock]):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(), MagicMock(), uri_prefixes=["http://x/"]
            )
        retriever.get_search_results(query_text="q", top_k=20)
        assert mock.get_search_results.call_args.kwargs["top_k"] == (
            20 * MultiPrefixVectorCypherRetriever.DEFAULT_PREFILTER_POOL_MULTIPLIER
        )

    def test_get_search_results_truncates_back_to_user_top_k(self):
        """Once the larger pool has been narrowed by the WHERE filter,
        the merged result is truncated back to the user-requested
        `top_k` so the LLM is not flooded with hundreds of context
        blocks (the bumped pool was only needed to get past the
        `neo4j_graphrag` pre-filter LIMIT)."""

        def _record(ctx: str):
            r = MagicMock()
            r.get.side_effect = lambda key, c=ctx: {"context": c}.get(key)
            return r

        # 100 records survive the underlying retriever's WHERE filter.
        records = [_record(f"## block\nuri: http://x/{i}") for i in range(100)]
        mock = MagicMock()
        mock.get_search_results.return_value = SimpleNamespace(
            records=records, metadata=None
        )
        with patch("src.graphrag.retriever.VectorCypherRetriever", side_effect=[mock]):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(), MagicMock(), uri_prefixes=["http://x/"]
            )
        raw = retriever.get_search_results(query_text="q", top_k=5)
        assert len(raw.records) == 5  # truncated back to user's top_k
        # Metadata exposes the bump for debugging.
        assert raw.metadata["user_top_k"] == 5
        assert (
            raw.metadata["internal_top_k"]
            == 5 * MultiPrefixVectorCypherRetriever.DEFAULT_PREFILTER_POOL_MULTIPLIER
        )
        assert raw.metadata["merged_total"] == 100

    def test_constructor_prefilter_pool_multiplier_overrides_default(self):
        """The wrapper accepts `prefilter_pool_multiplier` at
        construction time so individual scenarios can dial the bump
        up or down without going through `retriever_config`."""
        with patch("src.graphrag.retriever.VectorCypherRetriever"):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=["http://a/"],
                prefilter_pool_multiplier=7,
            )
        assert retriever._prefilter_pool_multiplier == 7

    def test_subclasses_neo4j_graphrag_retriever(self):
        """`GraphRAG`'s constructor uses Pydantic validation that
        rejects anything not subclassing `Retriever`.  Without this
        inheritance the grounded pipeline crashed at construction with
        `Input should be an instance of Retriever`."""
        from neo4j_graphrag.retrievers.base import Retriever

        with patch("src.graphrag.retriever.VectorCypherRetriever"):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=["http://a/"],
            )
        assert isinstance(retriever, Retriever)

    def test_get_search_results_concatenates_per_prefix_records(self):
        """`get_search_results` calls each underlying retriever and
        concatenates their raw records so every YAML-listed ontology
        reaches the LLM, regardless of relative corpus size.  The
        inherited `Retriever.search` formats them downstream."""

        def _record(ctx: str, score: float = 0.5):
            rec = MagicMock()
            rec.get.side_effect = lambda key, c=ctx, s=score: {
                "context": c,
                "score": s,
            }.get(key)
            return rec

        mock_a, mock_b = MagicMock(), MagicMock()
        mock_a.get_search_results.return_value = SimpleNamespace(
            records=[_record("## block A\nuri: http://a/x")], metadata=None
        )
        mock_b.get_search_results.return_value = SimpleNamespace(
            records=[_record("## block B\nuri: http://b/y")], metadata=None
        )
        with patch(
            "src.graphrag.retriever.VectorCypherRetriever",
            side_effect=[mock_a, mock_b],
        ):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=["http://a/", "http://b/"],
            )
        raw = retriever.get_search_results(query_text="hello", top_k=5)
        assert len(raw.records) == 2
        assert raw.records[0].get("context") == "## block A\nuri: http://a/x"
        assert raw.records[1].get("context") == "## block B\nuri: http://b/y"
        # Forwarded kwargs must hit each underlying retriever with the
        # bumped `top_k` so neo4j_graphrag's pre-filter LIMIT does
        # not truncate before our WHERE clauses see anything.
        bumped = 5 * MultiPrefixVectorCypherRetriever.DEFAULT_PREFILTER_POOL_MULTIPLIER
        mock_a.get_search_results.assert_called_once_with(
            query_text="hello", top_k=bumped
        )
        mock_b.get_search_results.assert_called_once_with(
            query_text="hello", top_k=bumped
        )
        assert raw.metadata["uri_prefixes"] == ["http://a/", "http://b/"]
        assert raw.metadata["per_prefix_item_counts"] == [1, 1]

    def test_search_returns_retriever_result_with_formatted_items(self):
        """End-to-end: the inherited `Retriever.search` should run
        the merged raw records through `_format_result` and produce
        a proper `RetrieverResult` that `GraphRAG` consumes."""
        rec = MagicMock()
        rec.get.side_effect = lambda key: {
            "context": "## hi\nuri: http://a/1",
            "score": 0.9,
        }.get(key)
        mock = MagicMock()
        mock.get_search_results.return_value = SimpleNamespace(
            records=[rec], metadata=None
        )
        with patch(
            "src.graphrag.retriever.VectorCypherRetriever",
            side_effect=[mock],
        ):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(), MagicMock(), uri_prefixes=["http://a/"]
            )
        result = retriever.search(query_text="q")
        assert len(result.items) == 1
        assert result.items[0].content == "## hi\nuri: http://a/1"
        assert result.items[0].metadata == {"score": 0.9}
        # The base `search` injects `__retriever` so the GraphRAG
        # debug log can identify which class produced the result.
        assert result.metadata["__retriever"] == "MultiPrefixVectorCypherRetriever"
        assert result.metadata["per_prefix_item_counts"] == [1]

    def test_get_search_results_dedupes_repeated_context(self):
        """If the same context block somehow appears for two prefixes
        (overlapping or duplicate prefix config), the LLM should not
        see it twice."""

        def _record(ctx: str):
            rec = MagicMock()
            rec.get.side_effect = lambda key, c=ctx: {"context": c}.get(key)
            return rec

        same_ctx = "duplicate context"
        mock_a, mock_b = MagicMock(), MagicMock()
        mock_a.get_search_results.return_value = SimpleNamespace(
            records=[_record(same_ctx)], metadata=None
        )
        mock_b.get_search_results.return_value = SimpleNamespace(
            records=[_record(same_ctx)], metadata=None
        )
        with patch(
            "src.graphrag.retriever.VectorCypherRetriever",
            side_effect=[mock_a, mock_b],
        ):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=["http://a/", "http://b/"],
            )
        raw = retriever.get_search_results(query_text="q")
        assert len(raw.records) == 1
        # Second prefix contributed zero NEW items after dedupe.
        assert raw.metadata["per_prefix_item_counts"] == [1, 0]

    def test_get_search_results_handles_non_string_context(self):
        """`record.get('context')` may legitimately be non-string
        for unusual retrieval queries; the dedupe guard must coerce to
        a string key without crashing."""

        rec = MagicMock()
        rec.get.side_effect = lambda key: {"context": 12345}.get(key)
        mock = MagicMock()
        mock.get_search_results.return_value = SimpleNamespace(
            records=[rec], metadata=None
        )
        with patch(
            "src.graphrag.retriever.VectorCypherRetriever",
            side_effect=[mock],
        ):
            retriever = MultiPrefixVectorCypherRetriever(
                MagicMock(), MagicMock(), uri_prefixes=["http://a/"]
            )
        raw = retriever.get_search_results(query_text="q")
        assert len(raw.records) == 1


class TestContextSurfacing:
    """The context block must surface the rdf:type label and the node's
    typed attributes (metValue, estimatedDurationMinutes, ...) so numeric
    values reach the LLM and can be cited; without this the answer cannot
    report or filter on a MET or duration number."""

    def test_query_emits_type_line_from_labels(self):
        assert (
            "[l IN labels(node) WHERE NOT l IN "
            "['EmbeddedConcept', 'Resource', 'Class']] AS types" in RETRIEVAL_QUERY
        )
        assert "'\\ntype: ' + apoc.text.join(types, ' / ')" in RETRIEVAL_QUERY

    def test_query_emits_attributes_line_from_keys(self):
        assert "[k IN keys(node) WHERE NOT k IN [" in RETRIEVAL_QUERY
        assert "| k + '=' + toString(head(node[k]))] AS attrs" in RETRIEVAL_QUERY
        assert "'\\nattributes: ' + apoc.text.join(attrs, '; ')" in RETRIEVAL_QUERY

    def test_surface_exclude_keys_skip_already_shown_text(self):
        # The heading / description text keys must not be repeated on the
        # attributes line, but the numeric ones must NOT be excluded.
        assert "'embedding'" in RETRIEVAL_QUERY
        assert "'uri'" in RETRIEVAL_QUERY
        assert "'title'" in RETRIEVAL_QUERY
        assert "'metValue'" not in RETRIEVAL_QUERY


class TestAttributeAndExcludeFilters:
    def test_numeric_filter_emits_hard_predicate(self):
        out = _build_retrieval_query(
            [HEALTH_TASK_INSTANCE_PREFIX],
            attribute_filters=[AttributeFilter("metValue", ">", 7.3)],
        )
        assert "AND toFloat(head(node.metValue)) > 7.3" in out

    def test_boolean_filter_emits_head_equality(self):
        out = _build_retrieval_query(
            None,
            attribute_filters=[AttributeFilter("isConcurrent", "=", True)],
        )
        assert "AND head(node.isConcurrent) = true" in out

    def test_exclude_prefix_emits_negated_starts_with(self):
        out = _build_retrieval_query(None, exclude_prefixes=["http://x/"])
        assert "AND NOT node.uri STARTS WITH 'http://x/'" in out

    def test_exclude_prefix_unsafe_rejected(self):
        with pytest.raises(ValueError, match="unsafe character"):
            _build_retrieval_query(None, exclude_prefixes=["http://x/'; DROP //"])

    def test_attribute_filter_alone_is_not_baseline(self):
        out = _build_retrieval_query(
            None, attribute_filters=[AttributeFilter("metValue", ">", 1.0)]
        )
        assert out is not RETRIEVAL_QUERY


class TestMakeRetrieverWithNewFilters:
    def test_attribute_filter_routes_through_wrapper(self):
        with patch("src.graphrag.retriever.VectorCypherRetriever"):
            retriever = make_retriever(
                MagicMock(),
                MagicMock(),
                attribute_filters=[AttributeFilter("metValue", ">", 7.3)],
            )
        assert isinstance(retriever, MultiPrefixVectorCypherRetriever)
        assert retriever._uri_prefixes == ["http"]

    def test_exclude_prefix_routes_through_wrapper(self):
        with patch("src.graphrag.retriever.VectorCypherRetriever"):
            retriever = make_retriever(
                MagicMock(), MagicMock(), exclude_prefixes=["http://x/"]
            )
        assert isinstance(retriever, MultiPrefixVectorCypherRetriever)

    def test_new_filters_propagate_into_each_per_prefix_cypher(self):
        with patch("src.graphrag.retriever.VectorCypherRetriever") as ctor:
            make_retriever(
                MagicMock(),
                MagicMock(),
                uri_prefixes=["http://a/", "http://b/"],
                attribute_filters=[AttributeFilter("metValue", ">", 7.3)],
                exclude_prefixes=["http://drop/"],
            )
        assert ctor.call_count == 2
        for call in ctor.call_args_list:
            cypher = call.kwargs["retrieval_query"]
            assert "toFloat(head(node.metValue)) > 7.3" in cypher
            assert "NOT node.uri STARTS WITH 'http://drop/'" in cypher


class TestExportedConstants:
    def test_node_label_present(self):
        assert NODE_LABEL == "EmbeddedConcept"

    def test_embedding_property_present(self):
        assert EMBEDDING_PROPERTY == "embedding"

    def test_vector_index_name_present(self):
        assert VECTOR_INDEX_NAME == "concept_embedding_index"

    def test_health_task_instance_prefix_present(self):
        assert HEALTH_TASK_INSTANCE_PREFIX == (
            "https://w3id.org/calendar-bench/health/task/"
        )
