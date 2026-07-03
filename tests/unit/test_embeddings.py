"""Unit tests for the embedder factory."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.graphrag import embeddings as embeddings_mod
from src.graphrag.embeddings import (
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    _cuda_available,
    make_embedder,
    resolve_device,
)


@pytest.fixture(autouse=True)
def _clear_embedder_cache():
    """Reset the process-shared embedder cache so each test builds fresh."""
    embeddings_mod._EMBEDDER_CACHE.clear()
    yield
    embeddings_mod._EMBEDDER_CACHE.clear()


def test_make_embedder_openai(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("src.graphrag.embeddings.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = "fake"
        result = make_embedder()
    assert result == "fake"
    mock_cls.assert_called_once_with(model=EMBED_MODEL, api_key="sk-test")


def test_make_embedder_openai_missing_key_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_embedder()


def test_make_embedder_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "totally-fake")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        make_embedder()


def test_make_embedder_provider_lowercased(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "OpenAI")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("src.graphrag.embeddings.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = "fake"
        make_embedder()
    mock_cls.assert_called_once()


def test_embed_constants_exposed():
    assert isinstance(EMBED_MODEL, str) and EMBED_MODEL
    assert isinstance(EMBED_DIMENSIONS, int) and EMBED_DIMENSIONS > 0


def test_make_embedder_local_success(monkeypatch):
    """Local provider returns a SentenceTransformerEmbeddings on the resolved device."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "test/model")
    monkeypatch.setenv("COMPUTE_DEVICE", "cpu")

    import neo4j_graphrag.embeddings as nge

    fake_class = MagicMock(return_value="local_embedder")
    monkeypatch.setattr(nge, "SentenceTransformerEmbeddings", fake_class, raising=False)

    result = make_embedder()
    assert result == "local_embedder"
    fake_class.assert_called_once_with(model="test/model", device="cpu")


def test_make_embedder_local_honours_device_argument(monkeypatch):
    """Explicit device= argument wins over COMPUTE_DEVICE env."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "test/model")
    monkeypatch.setenv("COMPUTE_DEVICE", "cpu")
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: True)

    import neo4j_graphrag.embeddings as nge

    fake_class = MagicMock(return_value="local_embedder")
    monkeypatch.setattr(nge, "SentenceTransformerEmbeddings", fake_class, raising=False)

    make_embedder(device="cuda")
    fake_class.assert_called_once_with(model="test/model", device="cuda")


def test_make_embedder_local_missing_dep_raises(monkeypatch):
    """When sentence-transformers is missing, surface a RuntimeError with install hint."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

    import builtins

    real_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "neo4j_graphrag.embeddings" and "SentenceTransformerEmbeddings" in (
            fromlist or ()
        ):
            raise ImportError("simulated missing sentence-transformers")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(RuntimeError, match="sentence-transformers"):
        make_embedder()


# ---------------------------------------------------------------------------
# resolve_device + _cuda_available
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_explicit(monkeypatch):
    monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_uppercase_and_whitespace_normalised(monkeypatch):
    """Preference tokens are case- and whitespace-insensitive."""
    monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
    assert resolve_device("  CPU  ") == "cpu"


def test_resolve_device_reads_env_when_no_argument(monkeypatch):
    monkeypatch.setenv("COMPUTE_DEVICE", "cpu")
    assert resolve_device() == "cpu"


def test_resolve_device_defaults_to_auto(monkeypatch):
    """No argument, no env var, no GPU -> falls back to cpu via auto."""
    monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: False)
    assert resolve_device() == "cpu"


def test_resolve_device_auto_picks_cuda_when_available(monkeypatch):
    monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_picks_cpu_when_unavailable(monkeypatch):
    monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_cuda_returns_token_when_available(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: True)
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_cuda_with_index_returns_token_when_available(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: True)
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch with CUDA"):
        resolve_device("cuda")


def test_resolve_device_cuda_indexed_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch with CUDA"):
        resolve_device("cuda:0")


def test_resolve_device_rejects_unknown_token(monkeypatch):
    monkeypatch.delenv("COMPUTE_DEVICE", raising=False)
    with pytest.raises(ValueError, match="Unknown COMPUTE_DEVICE"):
        resolve_device("tpu")


def test_cuda_available_returns_false_when_torch_missing(monkeypatch):
    """If `import torch` raises ImportError, return False without exploding."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    # Ensure any cached torch module is gone so the patched import path fires.
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert _cuda_available() is False


def test_cuda_available_returns_true_when_torch_reports_gpu(monkeypatch):
    """Stub torch in sys.modules so `_cuda_available` returns True."""
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _cuda_available() is True


def test_cuda_available_returns_false_when_torch_reports_no_gpu(monkeypatch):
    """Stub torch in sys.modules so `_cuda_available` returns False."""
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _cuda_available() is False
