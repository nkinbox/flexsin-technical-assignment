"""Stage 4 — Vector store.

Embedded ChromaDB with cosine similarity and on-disk persistence.

Embedded rather than client/server: for a single-node POC it gives real vector
database semantics (cosine search, metadata filtering) with no extra container
and no server to operate. `CHROMA_PATH` points at a mounted volume so the index
survives container restarts.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from app.chunker import Chunk
from app.config import CHROMA_PATH, COLLECTION_NAME, ensure_data_dir
from app.embedder import get_embedder


@dataclass
class Retrieved:
    """A chunk returned by search, with its distance from the query.

    `distance` is cosine distance: lower is more similar. It is carried through
    rather than discarded because the relevance gate in `rag.py` depends on it.
    """

    text: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    distance: float


class VectorStore:
    """Persistent Chroma collection holding all document chunks."""

    def __init__(self, path: str = CHROMA_PATH, collection: str = COLLECTION_NAME):
        ensure_data_dir()
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            embedding_function=get_embedder().chroma_embedding_function,
            # Cosine is the right metric for normalised sentence embeddings.
            # Chroma defaults to L2, so this must be set explicitly -- and it is
            # fixed at creation time, which is why it lives here rather than in
            # a query parameter.
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Index chunks. Embedding is performed by the collection's function."""
        if not chunks:
            return 0

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
        return len(chunks)

    def query(
        self,
        text: str,
        top_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[Retrieved]:
        """Search for the chunks most similar to `text`.

        Args:
            text: Query string (already condensed, if it was a follow-up).
            top_k: Maximum chunks to return.
            doc_ids: Optional scope. When given, only these documents are
                searched -- applied as a store-level filter rather than
                post-filtering, so top_k still returns top_k results from
                within the selected scope.

        Returns:
            Chunks ordered nearest-first. May be empty.
        """
        if self._collection.count() == 0:
            return []

        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        result = self._collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where,
        )

        # Chroma nests results one level per query; we always send exactly one.
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            Retrieved(
                text=document,
                doc_id=metadata["doc_id"],
                filename=metadata["filename"],
                page_number=int(metadata["page_number"]),
                chunk_index=int(metadata["chunk_index"]),
                distance=float(distance),
            )
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def list_documents(self) -> list[dict]:
        """Summarise indexed documents: one entry per document with its size."""
        records = self._collection.get(include=["metadatas"])

        documents: dict[str, dict] = {}
        for metadata in records["metadatas"]:
            doc_id = metadata["doc_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata["filename"],
                    "chunks": 0,
                }
            documents[doc_id]["chunks"] += 1

        return sorted(documents.values(), key=lambda d: d["filename"])

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        """Total indexed chunks across all documents."""
        return self._collection.count()


_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Return the process-wide store, opening it on first use."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
