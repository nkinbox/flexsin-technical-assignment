"""End-to-end pipeline test with a real store and real embeddings.

Everything except the LLM is genuine: real chunking, real ONNX embeddings, a
real ChromaDB collection, real BM25, real fusion. Only generation is mocked.

This is where the claims that unit tests stub out get checked — that semantic
retrieval ranks the right chunk first, that lexical search rescues exact
identifiers the embedding misses, and that the gate's threshold is calibrated
against real distances rather than assumed.

Marked `slow`: the ONNX model loads on first use.

    pytest tests/ -m "not slow"     # skip
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.chunker import chunk_pages
from app.extract import Page
from app.retrieval import reset_cache, retrieve
from app.store import VectorStore

pytestmark = pytest.mark.slow


HANDBOOK = """\
Employee Handbook — Compensation

All full-time employees receive an annual salary review each January. The \
standard review cycle considers performance, market rate, and internal parity.

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

Reference: internal tracking code QR-88214-ZX applies to this filing.
"""


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A real Chroma collection seeded with two documents."""
    path = tmp_path_factory.mktemp("chroma")
    store = VectorStore(path=str(path), collection="integration_suite")

    for doc_id, filename, text in [
        ("doc-hr", "handbook.pdf", HANDBOOK),
        ("doc-fin", "financials.pdf", FINANCIALS),
    ]:
        chunks = chunk_pages(
            [Page(text=text, page_number=1)], doc_id=doc_id, filename=filename
        )
        store.add_chunks(chunks)

    return store


@pytest.fixture(autouse=True)
def _clear(store):
    reset_cache()
    yield
    reset_cache()


def test_documents_indexed(store):
    assert store.count() > 0
    assert {d["filename"] for d in store.list_documents()} == {
        "handbook.pdf",
        "financials.pdf",
    }


# --- Semantic retrieval -------------------------------------------------------


def test_paraphrased_question_finds_its_answer(store):
    """Retrieval must work on meaning, not keyword overlap.

    "How much time can I spend working from home?" shares almost no vocabulary
    with "Employees may work remotely up to three days per week" — matching it
    is the entire point of using embeddings.
    """
    results = retrieve(
        "How much time can I spend working from home?", store=store, top_k=3
    )

    assert results
    assert any("remotely" in c.text.lower() for c in results)


def test_simple_factual_question_is_answerable(store):
    results = retrieve("What was total revenue?", store=store, top_k=3)

    assert results
    assert any("4.2 million" in c.text for c in results)


# --- Lexical retrieval --------------------------------------------------------


def test_exact_identifier_is_retrievable(store):
    """The case hybrid retrieval exists for.

    An opaque tracking code carries no semantic content for an embedding to
    represent, so dense search alone is unreliable here. BM25 matches it
    verbatim.
    """
    results = retrieve("QR-88214-ZX", store=store, top_k=5)

    assert results, "hybrid retrieval should surface an exact identifier"
    assert any("QR-88214-ZX" in c.text for c in results)


def test_identifier_chunk_carries_lexical_evidence(store):
    """It should be admitted on the lexical signal, not by luck of distance."""
    results = retrieve("QR-88214-ZX", store=store, top_k=5)
    match = next(c for c in results if "QR-88214-ZX" in c.text)

    assert match.bm25_score is not None and match.bm25_score > 0


# --- The gate -----------------------------------------------------------------


def test_unrelated_question_is_gated(store):
    """Off-topic questions must still be refused.

    A lenient threshold widens what reaches the model; it must not admit
    everything, or the first grounding layer stops meaning anything.
    """
    results = retrieve(
        "What is the migratory pattern of Arctic terns?", store=store, top_k=5
    )

    assert results == []


def test_relevant_questions_are_not_gated(store):
    """The complaint this configuration addresses: refusing answerable questions."""
    for question in [
        "What was total revenue?",
        "How many days can I work remotely?",
        "Who pays for medical premiums?",
        "When is the salary review?",
        "What were operating expenses?",
    ]:
        assert retrieve(question, store=store, top_k=5), (
            f"'{question}' is answerable from the documents but was gated"
        )


# --- Scoping ------------------------------------------------------------------


def test_document_scope_excludes_other_documents(store):
    results = retrieve("What is the policy?", store=store, top_k=5, doc_ids=["doc-fin"])

    assert results
    assert all(c.doc_id == "doc-fin" for c in results)


# --- Full pipeline ------------------------------------------------------------


def test_pipeline_refuses_unanswerable_question(store):
    """The headline behaviour, against a real index."""
    from app.config import REFUSAL_MESSAGE
    from app.rag import answer_question

    with patch("app.rag.retrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda **kw: retrieve(store=store, **kw)
        with patch("app.rag.get_llm") as get_llm:
            llm = MagicMock()
            get_llm.return_value = llm

            result = answer_question("What is the boiling point of mercury?")

            assert result.gated is True
            assert result.answer == REFUSAL_MESSAGE
            llm.generate_json.assert_not_called()


def test_pipeline_answers_grounded_question(store):
    from app.rag import answer_question

    with patch("app.rag.retrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda **kw: retrieve(store=store, **kw)
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
            assert result.citations

            prompt = llm.generate_json.call_args.kwargs["prompt"]
            assert "4.2 million" in prompt


def test_delete_removes_document(tmp_path):
    scratch = VectorStore(path=str(tmp_path / "scratch"), collection="scratch_delete")
    chunks = chunk_pages(
        [Page(text=HANDBOOK, page_number=1)], doc_id="temp", filename="temp.pdf"
    )
    scratch.add_chunks(chunks)
    assert scratch.count() > 0

    scratch.delete_document("temp")
    assert scratch.count() == 0
