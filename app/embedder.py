"""Embeddings.

Two interchangeable backends behind one interface:

  vertex  Vertex AI embeddings. Markedly better retrieval recall, and supports
          asymmetric embedding -- a passage and a question about that passage
          are encoded with different task types, which is what they actually
          are. This is the default.

  local   In-process ONNX all-MiniLM-L6-v2. No cost, no network, no
          credentials; noticeably weaker on paraphrased questions. Useful for
          offline development and for running the test suite.

Switching providers changes both the vector space and the dimensionality, so
`config.COLLECTION_NAME` encodes the embedding configuration and a switch
starts a fresh collection rather than silently mixing incompatible vectors.
"""

from __future__ import annotations

from typing import Protocol

from app.config import (
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBEDDING_PROVIDER,
    VERTEX_EMBED_MODEL,
)


class Embedder(Protocol):
    """Backend-agnostic embedding interface."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


class VertexEmbedder:
    """Vertex AI embeddings with task-type awareness."""

    def __init__(self, model: str = VERTEX_EMBED_MODEL, dimension: int = EMBED_DIM):
        from google import genai

        from app.config import GCP_LOCATION, GCP_PROJECT
        from app.llm import LLMError

        if not GCP_PROJECT:
            raise LLMError(
                "GCP_PROJECT is not set, which Vertex embeddings require. "
                "Set it in .env, or set EMBEDDING_PROVIDER=local to embed "
                "in-process without credentials."
            )

        self.model = model
        self.dimension = dimension
        self._client = genai.Client(
            vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION
        )

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        from app.llm import LLMError

        vectors: list[list[float]] = []

        # Vertex caps texts per call, so long documents are sent in batches.
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            try:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimension,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"Vertex embedding request failed: {exc}") from exc

            vectors.extend(list(item.values) for item in response.embeddings)

        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # RETRIEVAL_DOCUMENT tells the model these are passages to be found.
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        # RETRIEVAL_QUERY encodes a question rather than a passage. Using the
        # matching pair is what makes a short question land near the long
        # passage that answers it, instead of near other short questions.
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class LocalEmbedder:
    """ONNX MiniLM embedder, run in-process.

    Uses ChromaDB's bundled ONNX runtime rather than sentence-transformers, so
    there is no PyTorch dependency -- which keeps the image small and avoids
    depending on torch wheels existing for the running Python version.
    """

    dimension = 384

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # The ONNX runtime returns numpy float32 scalars, which Chroma rejects
        # when embeddings are supplied explicitly -- coerce to plain floats.
        return [[float(value) for value in vector] for vector in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        # MiniLM is symmetric -- queries and passages share one encoding.
        return [float(value) for value in self._fn([text])[0]]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the process-wide embedder, constructing it on first use."""
    global _embedder
    if _embedder is None:
        _embedder = (
            VertexEmbedder() if EMBEDDING_PROVIDER == "vertex" else LocalEmbedder()
        )
    return _embedder


def reset_embedder() -> None:
    """Drop the cached embedder. Used by tests that swap providers."""
    global _embedder
    _embedder = None
