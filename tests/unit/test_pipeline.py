"""Unit tests for build_graphrag: verify wiring of factories + prompt template."""

from __future__ import annotations

from unittest.mock import patch

from src.graphrag.pipeline import CITATION_TEMPLATE, build_graphrag


def test_citation_template_has_required_placeholders():
    assert "{context}" in CITATION_TEMPLATE
    assert "{query_text}" in CITATION_TEMPLATE
    # Spot-check that the prompt requires citations.
    assert "[" in CITATION_TEMPLATE and "URI" in CITATION_TEMPLATE


def test_build_graphrag_calls_each_factory():
    with patch("src.graphrag.pipeline.make_driver") as drv, patch(
        "src.graphrag.pipeline.make_embedder"
    ) as emb, patch("src.graphrag.pipeline.make_llm") as llm, patch(
        "src.graphrag.pipeline.make_retriever"
    ) as ret, patch(
        "src.graphrag.pipeline.GraphRAG"
    ) as rag:
        drv.return_value = "DRIVER"
        emb.return_value = "EMBEDDER"
        llm.return_value = "LLM"
        ret.return_value = "RETRIEVER"
        rag.return_value = "RAG"

        result = build_graphrag()

    assert result == "RAG"
    drv.assert_called_once()
    emb.assert_called_once()
    llm.assert_called_once()
    ret.assert_called_once_with(
        "DRIVER",
        "EMBEDDER",
        uri_prefixes=None,
        allowed_branches=None,
        allowed_levels=None,
        instance_only=False,
        attribute_filters=None,
        exclude_prefixes=None,
    )
    rag_kwargs = rag.call_args.kwargs
    assert rag_kwargs["retriever"] == "RETRIEVER"
    assert rag_kwargs["llm"] == "LLM"
    assert rag_kwargs["prompt_template"] is not None


def test_build_graphrag_reuses_provided_driver():
    with patch("src.graphrag.pipeline.make_driver") as drv, patch(
        "src.graphrag.pipeline.make_embedder"
    ), patch("src.graphrag.pipeline.make_llm"), patch(
        "src.graphrag.pipeline.make_retriever"
    ), patch(
        "src.graphrag.pipeline.GraphRAG"
    ):
        build_graphrag(driver="EXTERNAL_DRIVER")
    drv.assert_not_called()


def test_build_graphrag_forwards_uri_prefixes_to_retriever():
    with patch("src.graphrag.pipeline.make_driver") as drv, patch(
        "src.graphrag.pipeline.make_embedder"
    ) as emb, patch("src.graphrag.pipeline.make_llm"), patch(
        "src.graphrag.pipeline.make_retriever"
    ) as ret, patch(
        "src.graphrag.pipeline.GraphRAG"
    ):
        drv.return_value = "DRIVER"
        emb.return_value = "EMBEDDER"
        build_graphrag(uri_prefixes=["https://example.org/x/"])
    ret.assert_called_once_with(
        "DRIVER",
        "EMBEDDER",
        uri_prefixes=["https://example.org/x/"],
        allowed_branches=None,
        allowed_levels=None,
        instance_only=False,
        attribute_filters=None,
        exclude_prefixes=None,
    )


def test_build_graphrag_forwards_branch_level_instance_filters():
    """The grounded path bakes the scenario's filters.domains +
    filters.difficulty + the instance-only check directly into the
    retrieval Cypher.  This test confirms the kwargs reach
    `make_retriever` so that wiring cannot regress silently."""
    with patch("src.graphrag.pipeline.make_driver") as drv, patch(
        "src.graphrag.pipeline.make_embedder"
    ) as emb, patch("src.graphrag.pipeline.make_llm"), patch(
        "src.graphrag.pipeline.make_retriever"
    ) as ret, patch(
        "src.graphrag.pipeline.GraphRAG"
    ):
        drv.return_value = "DRIVER"
        emb.return_value = "EMBEDDER"
        build_graphrag(
            uri_prefixes=["https://example.org/x/"],
            allowed_branches=["https://example.org/branch/"],
            allowed_levels=["Level1"],
            instance_only=True,
        )
    ret.assert_called_once_with(
        "DRIVER",
        "EMBEDDER",
        uri_prefixes=["https://example.org/x/"],
        allowed_branches=["https://example.org/branch/"],
        allowed_levels=["Level1"],
        instance_only=True,
        attribute_filters=None,
        exclude_prefixes=None,
    )


def test_build_graphrag_forwards_attribute_and_exclude_filters():
    """The general-purpose ask path bakes numeric attribute filters and
    ontology exclusions into the retrieval Cypher; this confirms the
    kwargs reach `make_retriever` so the wiring cannot regress."""
    from src.graphrag.retrieval_filters import AttributeFilter

    flt = AttributeFilter("metValue", ">", 7.3)
    with patch("src.graphrag.pipeline.make_driver") as drv, patch(
        "src.graphrag.pipeline.make_embedder"
    ) as emb, patch("src.graphrag.pipeline.make_llm"), patch(
        "src.graphrag.pipeline.make_retriever"
    ) as ret, patch(
        "src.graphrag.pipeline.GraphRAG"
    ):
        drv.return_value = "DRIVER"
        emb.return_value = "EMBEDDER"
        build_graphrag(
            attribute_filters=[flt],
            exclude_prefixes=["http://sbmi.uth.tmc.edu/ontology/ochv#"],
        )
    ret.assert_called_once_with(
        "DRIVER",
        "EMBEDDER",
        uri_prefixes=None,
        allowed_branches=None,
        allowed_levels=None,
        instance_only=False,
        attribute_filters=[flt],
        exclude_prefixes=["http://sbmi.uth.tmc.edu/ontology/ochv#"],
    )


def test_build_graphrag_uses_supplied_llm_and_skips_make_llm():
    """A scenario YAML can pin a per-stage model override; the resulting
    LLM is built upstream and threaded through `build_graphrag(llm=…)`.
    When supplied, `make_llm()` must NOT be called; otherwise the env
    default would silently shadow the override."""
    with patch("src.graphrag.pipeline.make_driver"), patch(
        "src.graphrag.pipeline.make_embedder"
    ), patch("src.graphrag.pipeline.make_llm") as llm_factory, patch(
        "src.graphrag.pipeline.make_retriever"
    ), patch(
        "src.graphrag.pipeline.GraphRAG"
    ) as rag:
        build_graphrag(llm="OVERRIDE_LLM")
    llm_factory.assert_not_called()
    rag_kwargs = rag.call_args.kwargs
    assert rag_kwargs["llm"] == "OVERRIDE_LLM"


def test_build_graphrag_falls_back_to_make_llm_when_no_override():
    with patch("src.graphrag.pipeline.make_driver"), patch(
        "src.graphrag.pipeline.make_embedder"
    ), patch("src.graphrag.pipeline.make_llm") as llm_factory, patch(
        "src.graphrag.pipeline.make_retriever"
    ), patch(
        "src.graphrag.pipeline.GraphRAG"
    ) as rag:
        llm_factory.return_value = "ENV_LLM"
        build_graphrag()
    llm_factory.assert_called_once_with()
    assert rag.call_args.kwargs["llm"] == "ENV_LLM"
