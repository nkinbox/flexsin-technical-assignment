"""Hybrid retrieval tests — fusion and the relevance gate.

The gate is the first and strongest grounding layer, and hybrid retrieval is
what stops it refusing questions the documents actually answer. Both are pinned
here against a fake store, so the logic is tested without embeddings.
"""

from unittest.mock import MagicMock

import pytest

from app import retrieval
from app.retrieval import _fuse, _passes_gate, retrieve
from app.store import Retrieved


def chunk(
    text: str = "content",
    *,
    distance: float | None = None,
    bm25: float | None = None,
    index: int = 0,
    doc: str = "doc-1",
) -> Retrieved:
    return Retrieved(
        text=text,
        doc_id=doc,
        filename="handbook.pdf",
        page_number=1,
        chunk_index=index,
        distance=distance,
        bm25_score=bm25,
    )


@pytest.fixture(autouse=True)
def clear_cache():
    retrieval.reset_cache()
    yield
    retrieval.reset_cache()


# --- The gate -----------------------------------------------------------------


def test_semantic_evidence_passes():
    assert _passes_gate(chunk(distance=0.4), threshold=1.0, best_bm25=0.0) is True


def test_distant_chunk_without_lexical_evidence_fails():
    assert _passes_gate(chunk(distance=1.4), threshold=1.0, best_bm25=0.0) is False


def test_lexical_evidence_alone_passes():
    """The case hybrid retrieval exists for.

    An exact identifier the embedding is ambivalent about still reaches the
    model on lexical evidence.
    """
    assert _passes_gate(chunk(distance=1.5, bm25=6.0), threshold=1.0, best_bm25=6.0) is True


def test_weak_lexical_match_relative_to_best_fails():
    """A tail match far below the query's best lexical hit is not evidence."""
    assert _passes_gate(chunk(distance=1.5, bm25=0.01), threshold=1.0, best_bm25=9.0) is False


def test_chunk_with_no_evidence_fails():
    assert _passes_gate(chunk(), threshold=1.0, best_bm25=0.0) is False


def test_gate_is_a_disjunction():
    """Either signal suffices; requiring both would defeat the purpose."""
    assert _passes_gate(chunk(distance=0.2, bm25=None), threshold=1.0, best_bm25=0.0) is True
    assert _passes_gate(chunk(distance=None, bm25=9.0), threshold=1.0, best_bm25=9.0) is True


# --- Reciprocal Rank Fusion ---------------------------------------------------


def test_fusion_merges_by_identity():
    dense = [chunk("shared", distance=0.3, index=1)]
    lexical = [chunk("shared", bm25=5.0, index=1)]

    fused = _fuse(dense, lexical)

    assert len(fused) == 1
    # Both signals are preserved on the merged record.
    assert fused[0].distance == 0.3
    assert fused[0].bm25_score == 5.0


def test_agreement_between_retrievers_wins():
    """A chunk both retrievers rank should outrank one only a single retriever found."""
    dense = [chunk("both", distance=0.3, index=1), chunk("dense", distance=0.4, index=2)]
    lexical = [chunk("both", bm25=5.0, index=1), chunk("lex", bm25=4.0, index=3)]

    fused = _fuse(dense, lexical)

    assert fused[0].chunk_index == 1
    assert fused[0].fused_score > fused[1].fused_score


def test_fusion_keeps_lexical_only_results():
    dense = [chunk("dense", distance=0.3, index=1)]
    lexical = [chunk("lexical", bm25=8.0, index=9)]

    fused = _fuse(dense, lexical)

    assert {c.chunk_index for c in fused} == {1, 9}


def test_fusion_ranks_by_position_not_raw_score():
    """RRF combines ranks, so incomparable score scales never need normalising."""
    dense = [chunk("a", distance=0.9, index=1), chunk("b", distance=0.91, index=2)]
    fused = _fuse(dense, [])

    assert [c.chunk_index for c in fused] == [1, 2]


# --- End to end through retrieve() -------------------------------------------


def _store_with(chunks, dense_results):
    store = MagicMock()
    store.version = 1
    store.query.return_value = dense_results
    store.all_chunks.return_value = chunks
    return store


def test_retrieve_finds_exact_identifier_missed_by_vectors():
    """The headline hybrid case.

    The vector retriever ranks the wrong chunk and puts the right one out of
    reach; BM25 matches the identifier verbatim and rescues it.
    """
    corpus = [
        chunk("Invoice SKU-4421 was shipped on 3 March.", index=0),
        chunk("General shipping policy applies to all orders.", index=1),
    ]
    dense = [chunk("General shipping policy applies to all orders.",
                   distance=0.55, index=1)]

    store = _store_with(corpus, dense)
    results = retrieve("SKU-4421", store=store, top_k=5)

    assert any("SKU-4421" in c.text for c in results)


def test_retrieve_returns_empty_when_nothing_relevant():
    """The gate must still fire — this is what triggers the refusal."""
    corpus = [chunk("Bananas are yellow.", index=0)]
    dense = [chunk("Bananas are yellow.", distance=1.6, index=0)]

    store = _store_with(corpus, dense)

    assert retrieve("quarterly revenue", store=store, top_k=5) == []


def test_retrieve_respects_top_k():
    corpus = [chunk(f"shared content {i}", index=i) for i in range(10)]
    dense = [chunk(f"shared content {i}", distance=0.2, index=i) for i in range(10)]

    store = _store_with(corpus, dense)
    results = retrieve("shared", store=store, top_k=3)

    assert len(results) == 3


def test_retrieve_passes_scope_to_store():
    store = _store_with([], [])
    retrieve("question", doc_ids=["doc-a"], store=store)

    assert store.query.call_args.kwargs["doc_ids"] == ["doc-a"]
    assert store.all_chunks.call_args.kwargs["doc_ids"] == ["doc-a"]


def test_lexical_index_rebuilds_when_corpus_changes():
    """A stale index would serve documents that were since deleted."""
    store = _store_with([chunk("original", index=0)], [])
    retrieve("original", store=store)
    assert store.all_chunks.call_count == 1

    # Same version -> cached, no rebuild.
    retrieve("original", store=store)
    assert store.all_chunks.call_count == 1

    # Version bumped -> rebuild.
    store.version = 2
    retrieve("original", store=store)
    assert store.all_chunks.call_count == 2
