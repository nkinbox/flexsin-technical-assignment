"""End-to-end pipeline test with a real store and real embeddings.

Everything except the LLM is genuine here: real chunking, real ONNX embeddings,
a real ChromaDB collection on a temp directory. Only generation is mocked, so
this exercises the parts unit tests stub out — in particular that semantic
retrieval actually ranks the right chunk first, and that the relevance gate's
threshold is correctly calibrated against real embedding distances.

Marked `slow`: the ONNX model loads on first use.

    pytest tests/ -m "not slow"     # skip
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.chunker import chunk_pages
from app.extract import Page
from app.store import VectorStore

pytestmark = pytest.mark.slow


HANDBOOK = """\
Employee Handbook — Compensation

All full-time employees receive an annual salary review each January. \
The standard review cycle considers performance, market rate, and internal parity.

Health Coverage

The company covers 100% of employee medical premiums and 60% of dependent \
premiums. Dental and vision are included at no additional cost.

Remote Work Policy

Employees may work remotely up to three days per week. Fully remote \
arrangements require director-level approval and are reviewed annually.
"""

FINANCIALS = """\
Quarterly Report — Q3 2024

Total revenue for the third quarter was 4.2 million dollars, an increase of \
18% year over year. Operating expenses were 3.1 million dollars.

Net income for the quarter was 1.1 million dollars. Cash reserves stood at \
12.4 million dollars at quarter end.
"""


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A real Chroma collection seeded with two documents."""
    path = tmp_path_factory.mktemp("chroma")
    store = VectorStore(path=str(path), collection="integration")

    for doc_id, filename, text in [
        ("doc-hr", "handbook.pdf", HANDBOOK),
        ("doc-fin", "financials.pdf", FINANCIALS),
    ]:
        chunks = chunk_pages(
            [Page(text=text, page_number=1)], doc_id=doc_id, filename=filename
        )
        store.add_chunks(chunks)

    return store


def test_documents_indexed(store):
    assert store.count() > 0
    assert {d["filename"] for d in store.list_documents()} == {
        "handbook.pdf",
        "financials.pdf",
    }


def test_semantic_retrieval_ranks_correct_chunk_first(store):
    """Retrieval must work on meaning, not keyword overlap.

    "How much time can I spend working from home?" shares almost no vocabulary
    with "Employees may work remotely up to three days per week" — matching it
    is the entire point of using embeddings.
    """
    results = store.query("How much time can I spend working from home?", top_k=3)

    assert results
    assert "remotely" in results[0].text.lower()
    assert results[0].filename == "handbook.pdf"


def test_relevant_query_scores_within_threshold(store):
    """The gate's default threshold must admit genuinely relevant chunks."""
    from app.config import RELEVANCE_THRESHOLD

    results = store.query("What was total revenue?", top_k=3)

    assert results[0].distance < RELEVANCE_THRESHOLD, (
        f"Relevant chunk scored {results[0].distance:.3f}, at or beyond the "
        f"{RELEVANCE_THRESHOLD} threshold — the gate would wrongly refuse."
    )


def test_unrelated_query_scores_beyond_threshold(store):
    """And it must exclude genuinely unrelated ones.

    This is the calibration that makes the gate work. If an off-topic question
    scored inside the threshold, the gate would pass junk to the model and the
    no-hallucination guarantee would rest on the prompt alone.
    """
    from app.config import RELEVANCE_THRESHOLD

    results = store.query(
        "What is the migratory pattern of Arctic terns?", top_k=3
    )

    assert results, "Chroma returns nearest neighbours regardless of relevance"
    assert results[0].distance > RELEVANCE_THRESHOLD, (
        f"Unrelated chunk scored {results[0].distance:.3f}, inside the "
        f"{RELEVANCE_THRESHOLD} threshold — the gate would let it through."
    )


def test_document_scope_filter(store):
    """Multi-document querying: scoping must exclude other documents entirely."""
    results = store.query("What is the policy?", top_k=5, doc_ids=["doc-fin"])

    assert results
    assert all(r.doc_id == "doc-fin" for r in results)


def test_full_pipeline_refuses_unanswerable_question(store):
    """The headline behaviour, end to end against a real index.

    An off-topic question must be refused by the gate — with the real embedder
    and real vector store deciding relevance, not a mocked distance.
    """
    from app.config import REFUSAL_MESSAGE
    from app.rag import answer_question

    with patch("app.rag.get_store", return_value=store):
        with patch("app.rag.get_llm") as get_llm:
            llm = MagicMock()
            get_llm.return_value = llm

            result = answer_question("What is the boiling point of mercury?")

            assert result.gated is True
            assert result.answer == REFUSAL_MESSAGE
            llm.generate_json.assert_not_called()


def test_full_pipeline_answers_grounded_question(store):
    """And a real question retrieves real context and cites it."""
    from app.rag import answer_question

    with patch("app.rag.get_store", return_value=store):
        with patch("app.rag.get_llm") as get_llm:
            llm = MagicMock()
            llm.generate_json.return_value = {
                "answer": "Total Q3 revenue was $4.2 million.",
                "citations": [1],
                "found": True,
            }
            get_llm.return_value = llm

            result = answer_question("What was total revenue in Q3?")

            assert result.gated is False
            assert result.found is True
            assert result.verified is True
            assert result.citations

            # The prompt must carry the real retrieved text.
            prompt = llm.generate_json.call_args.kwargs["prompt"]
            assert "4.2 million" in prompt


def test_delete_removes_document(store, tmp_path):
    """Deletion should remove only the targeted document."""
    scratch = VectorStore(path=str(tmp_path / "scratch"), collection="scratch")
    chunks = chunk_pages(
        [Page(text=HANDBOOK, page_number=1)], doc_id="temp", filename="temp.pdf"
    )
    scratch.add_chunks(chunks)
    assert scratch.count() > 0

    scratch.delete_document("temp")
    assert scratch.count() == 0
