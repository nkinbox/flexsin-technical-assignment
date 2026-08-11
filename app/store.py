"""Vector store.

Embedded ChromaDB with cosine similarity and on-disk persistence.

Embeddings are computed by `app.embedder` and passed to Chroma explicitly
rather than registering a Chroma embedding function. That keeps one code path
for both providers, keeps the asymmetric document/query task types under our
control, and avoids coupling the persisted collection to a Chroma-side
embedding configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from app.chunker import Chunk
from app.config import CHROMA_PATH, COLLECTION_NAME, ensure_data_dir
from app.embedder import get_embedder


@dataclass
class Retrieved:
    """A chunk returned by search, with the evidence behind its selection.

    Carries scores from both retrievers so the relevance gate can reason about
    lexical and semantic evidence separately.
    """

    text: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int

    # Cosine distance; lower is nearer. None when the chunk was found only by
    # lexical search and never scored by the vector retriever.
    distance: float | None = None

    # Okapi BM25 score; higher is better. None when found only by vector search.
    bm25_score: float | None = None

    # Fused rank score, set by hybrid retrieval.
    fused_score: float = 0.0


class VectorStore:
    """Persistent Chroma collection holding all document chunks."""

    def __init__(self, path: str = CHROMA_PATH, collection: str = COLLECTION_NAME):
        ensure_data_dir()
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            # Cosine is the right metric for normalised sentence embeddings.
            # Chroma defaults to L2 and this is fixed at creation time.
            metadata={"hnsw:space": "cosine"},
        )
        # Invalidation counter for the BM25 index, which is derived from this
        # collection and must be rebuilt whenever the corpus changes.
        self._version = 0

    @property
    def version(self) -> int:
        """Increments on every mutation. Used to invalidate derived indexes."""
        return self._version

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and index chunks."""
        if not chunks:
            return 0

        embeddings = get_embedder().embed_documents([c.text for c in chunks])

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
            embeddings=embeddings,
        )
        self._version += 1
        return len(chunks)

    def query(
        self,
        text: str,
        top_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[Retrieved]:
        """Dense vector search.

        Args:
            text: Query string (already condensed, if it was a follow-up).
            top_k: Maximum chunks to return.
            doc_ids: Optional scope. Applied as a store-level filter, so top_k
                still returns top_k results from within the selected subset.

        Returns:
            Chunks ordered nearest-first. May be empty.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = get_embedder().embed_query(text)
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where,
        )

        # Chroma nests results one level per query; we always send exactly one.
        return [
            self._to_retrieved(document, metadata, distance=distance)
            for document, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def all_chunks(self, doc_ids: list[str] | None = None) -> list[Retrieved]:
        """Every indexed chunk, for building the lexical index.

        BM25 needs the whole corpus rather than a nearest-neighbour slice. At
        POC scale holding it in memory is unremarkable; a larger deployment
        would push lexical search into the datastore instead.
        """
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        records = self._collection.get(where=where, include=["documents", "metadatas"])

        return [
            self._to_retrieved(document, metadata)
            for document, metadata in zip(records["documents"], records["metadatas"])
        ]

    @staticmethod
    def _to_retrieved(
        document: str, metadata: dict, distance: float | None = None
    ) -> Retrieved:
        return Retrieved(
            text=document,
            doc_id=metadata["doc_id"],
            filename=metadata["filename"],
            page_number=int(metadata["page_number"]),
            chunk_index=int(metadata["chunk_index"]),
            distance=distance,
        )

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
        self._version += 1

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
