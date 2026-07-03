"""Unit tests for src.scripts.build_embeddings."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from neo4j.exceptions import ClientError

from src.scripts.build_embeddings import (
    DEFAULT_LOCAL_MODEL,
    FETCH,
    PRE_PASS,
    VECTOR_INDEX_NAME,
    _make_local_batch,
    _make_openai_batch,
    ensure_vector_index,
    main,
    make_embed_batch,
)

# ---------------------------------------------------------------------------
# Query-shape tests; make sure the embedder reaches HealthTasks instance
# nodes (which carry only `dcterms:title` and `dcterms:description`).
# ---------------------------------------------------------------------------


class TestPrePassEmbeddablePredicates:
    def test_pre_pass_marks_nodes_with_title(self):
        """`dcterms:title` (n10s `Resource.title`) makes a node
        embeddable; without this the `hb-tk:*` instance nodes would be
        silently skipped."""
        assert "size(coalesce(n.title, []))" in PRE_PASS

    def test_pre_pass_marks_nodes_with_description(self):
        """Same story for `dcterms:description` (n10s
        `Resource.description`)."""
        assert "size(coalesce(n.description, []))" in PRE_PASS

    def test_pre_pass_keeps_existing_predicates(self):
        for predicate in (
            "n.label",
            "n.prefLabel",
            "n.altLabel",
            "n.comment",
            "n.definition",
            "n.iAO_0000115",
        ):
            assert f"size(coalesce({predicate}, []))" in PRE_PASS, predicate


class TestFetchTextAssembly:
    def test_fetch_includes_title_in_text_join(self):
        assert "[v IN coalesce(n.title, [])" in FETCH

    def test_fetch_includes_description_in_text_join(self):
        assert "[v IN coalesce(n.description, [])" in FETCH

    def test_fetch_keeps_existing_text_sources(self):
        for predicate in (
            "n.prefLabel",
            "n.label",
            "n.altLabel",
            "n.comment",
            "n.definition",
            "n.iAO_0000115",
        ):
            assert f"[v IN coalesce({predicate}, [])" in FETCH, predicate


def test_main_returns_2_when_openai_key_missing(monkeypatch):
    """openai provider with no API key short-circuits before touching Neo4j."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main() == 2


def test_main_returns_2_when_unknown_provider(monkeypatch):
    """Unknown provider name surfaces as a config error."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "vertex")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        make_embed_batch()


def test_make_openai_batch_calls_endpoint(monkeypatch):
    """openai batch callable forwards texts and returns endpoint embeddings."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = resp
    with patch("src.scripts.build_embeddings.OpenAI", return_value=mock_client):
        embed = _make_openai_batch("text-embedding-3-small")
        out = embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["a", "b"]
    )


def test_make_openai_batch_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _make_openai_batch("any-model")


def _install_fake_sentence_transformers(
    encode_return: np.ndarray | list[list[float]],
) -> MagicMock:
    """Inject a fake `sentence_transformers` module exposing `SentenceTransformer`."""
    fake_module = types.ModuleType("sentence_transformers")
    instance = MagicMock()
    instance.encode.return_value = encode_return
    ctor = MagicMock(return_value=instance)
    fake_module.SentenceTransformer = ctor  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = fake_module
    return ctor


def test_make_local_batch_uses_sentence_transformers(monkeypatch):
    """local batch callable encodes texts with the configured model on the chosen device."""
    monkeypatch.setenv("COMPUTE_DEVICE", "cpu")
    vectors = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
    ctor = _install_fake_sentence_transformers(vectors)
    try:
        embed = _make_local_batch("acme/embed-small", batch_size=4)
        out = embed(["alpha", "beta"])
    finally:
        sys.modules.pop("sentence_transformers", None)

    ctor.assert_called_once_with("acme/embed-small", device="cpu")
    flat = [v for row in out for v in row]
    assert flat == pytest.approx([0.5, 0.5, 0.1, 0.9])
    encode_kwargs = ctor.return_value.encode.call_args.kwargs
    assert encode_kwargs["batch_size"] == 4
    assert encode_kwargs["normalize_embeddings"] is True
    assert encode_kwargs["convert_to_numpy"] is True
    assert encode_kwargs["show_progress_bar"] is False


def test_make_local_batch_raises_when_library_missing(monkeypatch):
    """Missing sentence-transformers surfaces as a clear config error."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no module named sentence_transformers")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("sentence_transformers", None)
    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        _make_local_batch("acme/embed-small", batch_size=4)


def test_make_embed_batch_dispatches_openai(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("src.scripts.build_embeddings.OpenAI", return_value=MagicMock()):
        embed, name, dim = make_embed_batch()
    assert callable(embed)
    assert name  # OpenAI model name
    assert dim > 0


def test_make_embed_batch_dispatches_local_with_default(monkeypatch):
    """`EMBEDDING_PROVIDER=local` falls back to the SOTA Qwen default model."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.delenv("LOCAL_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("COMPUTE_DEVICE", "cpu")
    ctor = _install_fake_sentence_transformers(np.zeros((1, 4), dtype=np.float32))
    try:
        embed, name, _dim = make_embed_batch()
    finally:
        sys.modules.pop("sentence_transformers", None)
    assert callable(embed)
    assert name == DEFAULT_LOCAL_MODEL
    ctor.assert_called_once_with(DEFAULT_LOCAL_MODEL, device="cpu")


def test_make_embed_batch_explicit_provider_argument(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("src.scripts.build_embeddings.OpenAI", return_value=MagicMock()):
        _, name, _ = make_embed_batch(provider="OpenAI")
    assert name  # OpenAI model name set via env


def test_ensure_vector_index_skips_when_present():
    session = MagicMock()
    show_result = MagicMock()
    show_result.single.return_value = {"name": VECTOR_INDEX_NAME}
    session.run.return_value = show_result

    ensure_vector_index(session, 1536)

    assert session.run.call_count == 1
    assert "SHOW INDEXES" in session.run.call_args.args[0]


def test_ensure_vector_index_creates_when_missing():
    session = MagicMock()
    show_result = MagicMock()
    show_result.single.return_value = None
    create_result = MagicMock()
    session.run.side_effect = [show_result, create_result]

    ensure_vector_index(session, 1536)

    assert session.run.call_count == 2
    create_query = session.run.call_args_list[1].args[0]
    assert "db.index.vector.createNodeIndex" in create_query


def test_ensure_vector_index_swallows_race_clienterror():
    session = MagicMock()
    show_result = MagicMock()
    show_result.single.return_value = None
    session.run.side_effect = [
        show_result,
        ClientError("EquivalentSchemaRule already exists for the index"),
    ]
    ensure_vector_index(session, 1536)


def test_ensure_vector_index_propagates_other_errors():
    session = MagicMock()
    show_result = MagicMock()
    show_result.single.return_value = None
    session.run.side_effect = [show_result, ClientError("disk full")]
    with pytest.raises(ClientError):
        ensure_vector_index(session, 1536)


def test_main_full_flow(monkeypatch):
    """Run main() end-to-end with mocked OpenAI + Neo4j."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")

    pre_pass_result = MagicMock()
    pre_pass_result.single.return_value = {"marked": 2}

    show_result = MagicMock()
    show_result.single.return_value = None  # Index missing to create

    fetch_batch_1 = MagicMock()
    fetch_batch_1.data.return_value = [
        {"id": "n1", "text": "label one"},
        {"id": "n2", "text": "label two"},
    ]
    fetch_batch_2 = MagicMock()
    fetch_batch_2.data.return_value = []

    fetch_calls = iter([fetch_batch_1, fetch_batch_2])

    def session_run(query, **kwargs):
        if (
            "MATCH (n:Resource)" in query
            and "SET n:" in query
            and "UnembeddedConcept" in query
        ):
            return pre_pass_result
        if "SHOW INDEXES" in query:
            return show_result
        if "createNodeIndex" in query:
            return MagicMock()
        if "MATCH (n:UnembeddedConcept)" in query:
            return next(fetch_calls)
        if "UNWIND $rows" in query:
            return MagicMock()
        return MagicMock()

    mock_session = MagicMock()
    mock_session.run.side_effect = session_run

    mock_driver = MagicMock()

    embedding_response = MagicMock()
    embedding_response.data = [
        MagicMock(embedding=[0.0] * 1536),
        MagicMock(embedding=[0.0] * 1536),
    ]
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = embedding_response

    with patch("src.scripts.build_embeddings.OpenAI", return_value=mock_openai), patch(
        "src.scripts.build_embeddings.make_driver", return_value=mock_driver
    ), patch("src.scripts.build_embeddings.session_scope") as mock_ss:
        mock_ss.return_value.__enter__.return_value = mock_session
        mock_ss.return_value.__exit__.return_value = None

        rc = main()

    assert rc == 0
    mock_driver.verify_connectivity.assert_called_once()
    mock_driver.close.assert_called_once()
    mock_openai.embeddings.create.assert_called_once()
    create_kwargs = mock_openai.embeddings.create.call_args.kwargs
    assert create_kwargs["input"] == ["label one", "label two"]


def test_main_full_batch_then_partial_covers_both_log_branches(monkeypatch):
    """Drive two iterations: a *full* batch (no progress log) + a *partial* batch
    (progress log fires). This exercises both arms of the conditional at the
    bottom of the embedding loop:
        if total % (BATCH_SIZE * 10) == 0 or len(rows) < BATCH_SIZE:
    The first batch hits the False arm (skip log, loop back); the second hits
    the True arm (log + then break on the empty third batch).
    """
    # Shrink BATCH_SIZE so two iterations are tractable inside one test.
    monkeypatch.setattr("src.scripts.build_embeddings.BATCH_SIZE", 2)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")

    pre_pass_result = MagicMock()
    pre_pass_result.single.return_value = {"marked": 3}

    # Vector index already exists: skip the create branch this run.
    show_result = MagicMock()
    show_result.single.return_value = {"name": "concept_embedding_index"}

    full_batch = MagicMock()
    full_batch.data.return_value = [
        {"id": "n1", "text": "first"},
        {"id": "n2", "text": "second"},  # len(rows) == BATCH_SIZE to False arm
    ]
    partial_batch = MagicMock()
    partial_batch.data.return_value = [
        {"id": "n3", "text": "third"},  # len(rows) <  BATCH_SIZE to True arm
    ]
    empty_batch = MagicMock()
    empty_batch.data.return_value = []

    fetch_calls = iter([full_batch, partial_batch, empty_batch])

    def session_run(query, **kwargs):
        if (
            "MATCH (n:Resource)" in query
            and "SET n:" in query
            and "UnembeddedConcept" in query
        ):
            return pre_pass_result
        if "SHOW INDEXES" in query:
            return show_result
        if "MATCH (n:UnembeddedConcept)" in query:
            return next(fetch_calls)
        return MagicMock()

    mock_session = MagicMock()
    mock_session.run.side_effect = session_run

    full_resp = MagicMock()
    full_resp.data = [MagicMock(embedding=[0.0] * 1536) for _ in range(2)]
    partial_resp = MagicMock()
    partial_resp.data = [MagicMock(embedding=[0.0] * 1536)]

    mock_openai = MagicMock()
    mock_openai.embeddings.create.side_effect = [full_resp, partial_resp]

    mock_driver = MagicMock()

    with patch("src.scripts.build_embeddings.OpenAI", return_value=mock_openai), patch(
        "src.scripts.build_embeddings.make_driver", return_value=mock_driver
    ), patch("src.scripts.build_embeddings.session_scope") as mock_ss:
        mock_ss.return_value.__enter__.return_value = mock_session
        mock_ss.return_value.__exit__.return_value = None

        rc = main()

    assert rc == 0
    # Two embedding API calls: one per non-empty batch.
    assert mock_openai.embeddings.create.call_count == 2


def test_main_local_provider_runs_end_to_end(monkeypatch):
    """`EMBEDDING_PROVIDER=local` routes the bake through sentence-transformers."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "acme/embed-small")
    monkeypatch.setenv("COMPUTE_DEVICE", "cpu")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    pre_pass_result = MagicMock()
    pre_pass_result.single.return_value = {"marked": 2}
    show_result = MagicMock()
    show_result.single.return_value = None
    fetch_batch = MagicMock()
    fetch_batch.data.return_value = [
        {"id": "n1", "text": "first"},
        {"id": "n2", "text": "second"},
    ]
    empty = MagicMock()
    empty.data.return_value = []
    fetch_calls = iter([fetch_batch, empty])

    def session_run(query, **kwargs):
        if (
            "MATCH (n:Resource)" in query
            and "SET n:" in query
            and "UnembeddedConcept" in query
        ):
            return pre_pass_result
        if "SHOW INDEXES" in query:
            return show_result
        if "createNodeIndex" in query:
            return MagicMock()
        if "MATCH (n:UnembeddedConcept)" in query:
            return next(fetch_calls)
        return MagicMock()

    mock_session = MagicMock()
    mock_session.run.side_effect = session_run
    mock_driver = MagicMock()
    vectors = np.array([[0.7, 0.7], [0.0, 1.0]], dtype=np.float32)
    ctor = _install_fake_sentence_transformers(vectors)
    try:
        with patch(
            "src.scripts.build_embeddings.make_driver", return_value=mock_driver
        ), patch("src.scripts.build_embeddings.session_scope") as mock_ss:
            mock_ss.return_value.__enter__.return_value = mock_session
            mock_ss.return_value.__exit__.return_value = None
            rc = main()
    finally:
        sys.modules.pop("sentence_transformers", None)

    assert rc == 0
    mock_driver.verify_connectivity.assert_called_once()
    mock_driver.close.assert_called_once()
    ctor.assert_called_once_with("acme/embed-small", device="cpu")
    ctor.return_value.encode.assert_called_once()


def test_main_local_provider_returns_2_when_library_missing(monkeypatch):
    """Local provider without sentence-transformers fails fast with rc=2."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    sys.modules.pop("sentence_transformers", None)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert main() == 2
