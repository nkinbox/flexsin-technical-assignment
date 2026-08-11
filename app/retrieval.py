"""Hybrid retrieval — dense vectors and BM25, fused.

Dense and lexical retrieval fail in opposite directions:

  * Embeddings match meaning. They handle paraphrase well ("time off" finding
    "vacation policy") but blur exact tokens -- an invoice number, a SKU, a
    version string, a rare surname all compress to something approximate.
  * BM25 matches terms. It nails those exact tokens and knows nothing about
    paraphrase.

A user asking a simple factual question about their own document typically
quotes its wording, which is the lexical case; a user asking conceptually is
the dense case. Running both and fusing the rankings covers both without
having to guess which kind of question arrived.

Fusion is Reciprocal Rank Fusion: each retriever contributes 1/(k + rank) for
every document it ranks. RRF combines *ranks* rather than scores, which is what
makes it usable here at all -- a cosine distance and a BM25 score have no
common scale, and normalising them would need corpus statistics that shift
every time a document is uploaded.
"""

from __future__ import annotations

from app.bm25 import BM25Index
from app.config import (
    BM25_MIN_RATIO,
    BM25_MIN_SCORE,
    CANDIDATE_POOL,
    HYBRID_SEARCH,
    RELEVANCE_THRESHOLD,
    RRF_K,
    TOP_K,
)
from app.store import Retrieved, VectorStore, get_store

# Cache: the lexical index is derived from the corpus and rebuilt only when the
# corpus changes. Keyed on the store's version counter and the document scope.
_cache: dict = {"key": None, "index": None, "chunks": []}


def _lexical_index(
    store: VectorStore, doc_ids: list[str] | None
) -> tuple[BM25Index | None, list[Retrieved]]:
    """Return a BM25 index over the (optionally scoped) corpus, cached."""
    key = (store.version, tuple(sorted(doc_ids)) if doc_ids else None)

    if _cache["key"] != key:
        chunks = store.all_chunks(doc_ids=doc_ids)
        _cache["key"] = key
        _cache["chunks"] = chunks
        _cache["index"] = BM25Index([c.text for c in chunks]) if chunks else None

    return _cache["index"], _cache["chunks"]


def reset_cache() -> None:
    """Drop the cached lexical index. Used by tests."""
    _cache["key"] = None
    _cache["index"] = None
    _cache["chunks"] = []


def _identity(chunk: Retrieved) -> tuple[str, int]:
    """Stable identity for a chunk, so the two retrievers can be matched up."""
    return (chunk.doc_id, chunk.chunk_index)


def retrieve(
    query: str,
    doc_ids: list[str] | None = None,
    top_k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    store: VectorStore | None = None,
) -> list[Retrieved]:
    """Retrieve the most relevant chunks, then apply the relevance gate.

    Returns:
        Up to `top_k` chunks that cleared the gate, best first. Empty when
        nothing was relevant enough -- which is the signal for `rag.py` to
        refuse without calling the model.
    """
    store = store or get_store()

    dense = store.query(query, top_k=CANDIDATE_POOL, doc_ids=doc_ids)

    if not HYBRID_SEARCH:
        return [c for c in dense if _passes_gate(c, threshold, 0.0)][:top_k]

    index, corpus = _lexical_index(store, doc_ids)
    lexical: list[Retrieved] = []

    if index is not None:
        for position, score in index.search(query, top_k=CANDIDATE_POOL):
            chunk = corpus[position]
            lexical.append(
                Retrieved(
                    text=chunk.text,
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    bm25_score=score,
                )
            )

    # The lexical bar is relative to the best match for THIS query, because a
    # BM25 score only means something next to its peers.
    best_bm25 = max((c.bm25_score or 0.0) for c in lexical) if lexical else 0.0

    fused = _fuse(dense, lexical)
    survivors = [c for c in fused if _passes_gate(c, threshold, best_bm25)]
    return survivors[:top_k]


def _fuse(dense: list[Retrieved], lexical: list[Retrieved]) -> list[Retrieved]:
    """Merge two ranked lists by Reciprocal Rank Fusion.

    A chunk found by both retrievers accumulates both contributions and rises
    above one found by only a single retriever -- agreement between two
    independent signals is the strongest evidence available here.
    """
    merged: dict[tuple[str, int], Retrieved] = {}

    for rank, chunk in enumerate(dense, start=1):
        entry = merged.setdefault(_identity(chunk), chunk)
        entry.distance = chunk.distance
        entry.fused_score += 1.0 / (RRF_K + rank)

    for rank, chunk in enumerate(lexical, start=1):
        identity = _identity(chunk)
        if identity in merged:
            entry = merged[identity]
            entry.bm25_score = chunk.bm25_score
        else:
            entry = merged.setdefault(identity, chunk)
        entry.fused_score += 1.0 / (RRF_K + rank)

    return sorted(merged.values(), key=lambda c: -c.fused_score)


def _passes_gate(chunk: Retrieved, threshold: float, best_bm25: float) -> bool:
    """Decide whether a chunk is relevant enough to show the model.

    A chunk qualifies on EITHER kind of evidence:

      * semantic  -- cosine distance within the threshold, or
      * lexical   -- a BM25 score clearing both an absolute floor and a share
                     of the best lexical score for this query

    The disjunction matters. Requiring both would discard the very cases hybrid
    retrieval exists to catch: an exact identifier the embedding is ambivalent
    about, or a paraphrase sharing no vocabulary with the source.

    A chunk with neither signal -- ranked by one retriever but below both bars
    -- is treated as unrelated and dropped. That is what still lets an
    off-topic question be refused despite a deliberately permissive threshold.
    """
    if chunk.distance is not None and chunk.distance <= threshold:
        return True

    if chunk.bm25_score is not None:
        floor = max(BM25_MIN_SCORE, BM25_MIN_RATIO * best_bm25)
        if chunk.bm25_score >= floor:
            return True

    return False
