"""Embedding factory used at retrieval time.

The factory honours a `COMPUTE_DEVICE` environment variable (`cpu` /
`cuda` / `cuda:<n>` / `auto`) so the persona pipeline's
`environment.yaml` -> `compute.device` setting can flow into the
local sentence-transformers backend without threading the value
through every call site.  The CLI sets `COMPUTE_DEVICE` before
calling into the pipeline; tests can monkeypatch the env var directly.
"""

from __future__ import annotations

import os

from neo4j_graphrag.embeddings import Embedder, OpenAIEmbeddings

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


def _cuda_available() -> bool:
    """Return True iff torch is importable and reports a usable GPU.

    Kept tiny so tests can monkeypatch this single function instead of
    faking the whole torch module.  `torch` is intentionally an
    optional dependency: a CPU-only install must still be able to
    import this module without `ImportError`.
    """
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(preference: str | None = None) -> str:
    """Resolve `preference` (or `COMPUTE_DEVICE` env) to a torch device.

    Args:
        preference: `cpu` / `cuda` / `cuda:<n>` / `auto`.  When
            `None`, the `COMPUTE_DEVICE` env var is consulted; if
            that is also unset the default is `auto`.

    Returns:
        A lower-case device string ready to hand to
        `sentence_transformers.SentenceTransformer(..., device=...)`.

    Raises:
        RuntimeError: `cuda`-family device requested but no CUDA-
            enabled torch / GPU is available at runtime.
        ValueError: `preference` is not one of the recognized tokens.
    """
    raw = (preference or os.getenv("COMPUTE_DEVICE", "auto")).strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if raw == "cuda" or raw.startswith("cuda:"):
        if not _cuda_available():
            raise RuntimeError(
                f"COMPUTE_DEVICE={raw!r} requested but torch with CUDA "
                "is not available; install a CUDA build of torch or use "
                "device=cpu / device=auto."
            )
        return raw
    raise ValueError(f"Unknown COMPUTE_DEVICE: {raw!r}")


# One embedder instance per (provider, model, resolved-device) so a
# process loads the ~2.5 GB local sentence-transformers model onto the
# GPU once and shares it across the retrieval and MET / merge paths
# instead of reloading it per call site.
_EMBEDDER_CACHE: dict[tuple, "Embedder"] = {}


def _build_embedder(provider: str, device: str | None) -> "Embedder":
    """Construct a fresh Embedder for `provider` (no caching)."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for embeddings (EMBEDDING_PROVIDER=openai)."
                "Set it in .env or switch EMBEDDING_PROVIDER=local."
            )
        return OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)
    if provider == "local":
        try:
            from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc
        local_model = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        resolved = resolve_device(device)
        return SentenceTransformerEmbeddings(model=local_model, device=resolved)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def make_embedder(device: str | None = None) -> Embedder:
    """Return a process-shared single-query Embedder for the retriever path.

    Uses EMBEDDING_PROVIDER (openai | local) from .env, OpenAI is the default.
    The optional `device` argument selects the compute backend for the
    local sentence-transformers model.  When `None` (the default) the
    `COMPUTE_DEVICE` env var is consulted, falling back to `auto`.
    The OpenAI path ignores `device` because embeddings are served by
    the remote API. Instances are cached per `(provider, model, device)`
    so the heavy local model loads onto the GPU once per process.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    model = (
        EMBED_MODEL
        if provider == "openai"
        else os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    resolved = "remote" if provider == "openai" else resolve_device(device)
    key = (provider, model, resolved)
    cached = _EMBEDDER_CACHE.get(key)
    if cached is None:
        cached = _build_embedder(provider, device)
        _EMBEDDER_CACHE[key] = cached
    return cached
