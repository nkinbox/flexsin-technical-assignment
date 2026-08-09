"""Stage 3 — Embeddings.

Local `all-MiniLM-L6-v2` (384-dim) via ChromaDB's bundled ONNX runtime.

Why ONNX rather than sentence-transformers/PyTorch (execution.md §3):
  1. The dev machine runs Python 3.14, where torch wheels may not exist.
  2. No ~800 MB torch layer in the container image.
  3. RAM headroom stays comfortable on a 4 GB VM.

Same model weights either way.

The `Embedder` interface exists so the embedding backend can be swapped
(e.g. for Vertex `text-embedding-005`) without touching any other module.
"""

from __future__ import annotations

from typing import Protocol

from chromadb.utils import embedding_functions


class Embedder(Protocol):
    """Backend-agnostic embedding interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


class LocalEmbedder:
    """ONNX MiniLM embedder, run in-process.

    Keeping embedding local avoids a network round-trip per batch during
    ingestion — the difference between a large PDF indexing in seconds versus
    minutes — and costs nothing per call.
    """

    def __init__(self) -> None:
        # Model weights (~80 MB) download once to ~/.cache/chroma. The Docker
        # image pre-warms this at build time so containers start instantly and
        # do not need network access on first use.
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    @property
    def chroma_embedding_function(self):
        """The raw Chroma embedding function.

        Handed to the collection so Chroma embeds consistently on both the
        write and query paths — the same function on both sides removes any
        chance of an index/query model mismatch.
        """
        return self._fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(vector) for vector in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._fn([text])[0])


_embedder: LocalEmbedder | None = None


def get_embedder() -> LocalEmbedder:
    """Return the process-wide embedder, constructing it on first use.

    The ONNX session takes a moment to initialise, so it is built once and
    reused rather than per request.
    """
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder()
    return _embedder
