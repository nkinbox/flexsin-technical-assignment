"""Grounding tests — the headline evaluation criterion.

The central claim of this build is that hallucination is prevented structurally
rather than by asking the model nicely. These tests hold that claim to account:

  * When retrieval finds nothing relevant, the model is NEVER CALLED.
  * When the model reports the sources lack the answer, that is honoured.
  * Fabricated citation numbers are removed, not passed through.

Retrieval is mocked here so these tests target the RAG layer's behaviour; the
gate's own logic is covered in test_retrieval.py, and the two are exercised
together against real embeddings in test_integration.py.

Every test mocks the LLM, so the suite runs with no credentials, no network,
and no cost.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.config import REFUSAL_MESSAGE
from app.rag import answer_question
from app.store import Retrieved


def _chunk(text: str, page: int = 1, distance: float = 0.3) -> Retrieved:
    return Retrieved(
        text=text,
        doc_id="doc-1",
        filename="handbook.pdf",
        page_number=page,
        chunk_index=0,
        distance=distance,
    )


@pytest.fixture
def mock_retrieve():
    with patch("app.rag.retrieve") as retrieve:
        yield retrieve


@pytest.fixture
def mock_llm():
    with patch("app.rag.get_llm") as get_llm:
        llm = MagicMock()
        get_llm.return_value = llm
        yield llm


# --- Layer 1: the relevance gate ---------------------------------------------


def test_nothing_relevant_refuses_without_calling_model(mock_retrieve, mock_llm):
    """THE critical test.

    Retrieval found nothing that cleared either the semantic or the lexical
    bar. A naive RAG implementation would hand the model whatever the vector
    store returned -- it always returns its nearest neighbours, however
    unrelated -- and invite an invented answer. Here the model is never
    invoked, so there is nothing to invent with.
    """
    mock_retrieve.return_value = []

    result = answer_question("What is the migratory pattern of Arctic terns?")

    assert result.gated is True
    assert result.found is False
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []
    mock_llm.generate_json.assert_not_called()


def test_relevant_results_reach_the_model(mock_retrieve, mock_llm):
    mock_retrieve.return_value = [_chunk("Q3 revenue was $4.2M.")]
    mock_llm.generate_json.return_value = {
        "answer": "Q3 revenue was $4.2M.",
        "citations": [1],
        "found": True,
    }

    result = answer_question("What was Q3 revenue?")

    assert result.gated is False
    assert result.found is True
    mock_llm.generate_json.assert_called_once()


def test_retrieval_receives_query_and_scope(mock_retrieve, mock_llm):
    mock_retrieve.return_value = []

    answer_question("Question?", doc_ids=["doc-a", "doc-b"], top_k=7)

    kwargs = mock_retrieve.call_args.kwargs
    assert kwargs["query"] == "Question?"
    assert kwargs["doc_ids"] == ["doc-a", "doc-b"]
    assert kwargs["top_k"] == 7


# --- Layer 2: the model's own found flag -------------------------------------


def test_model_reporting_not_found_is_honoured(mock_retrieve, mock_llm):
    """Relevant-looking chunks that don't actually answer the question."""
    mock_retrieve.return_value = [_chunk("Revenue is discussed quarterly.")]
    mock_llm.generate_json.return_value = {
        "answer": "",
        "citations": [],
        "found": False,
    }

    result = answer_question("What was Q3 revenue?")

    assert result.found is False
    assert result.answer == REFUSAL_MESSAGE
    # Same shape as the gate's refusal — callers see one behaviour.
    assert result.citations == []


# --- Prompt construction ------------------------------------------------------


def test_sources_are_numbered_in_prompt(mock_retrieve, mock_llm):
    """Numbering gives the model a stable handle to cite."""
    mock_retrieve.return_value = [
        _chunk("First source.", page=1),
        _chunk("Second source.", page=2),
    ]
    mock_llm.generate_json.return_value = {
        "answer": "Answer.",
        "citations": [1, 2],
        "found": True,
    }

    answer_question("Question?")

    prompt = mock_llm.generate_json.call_args.kwargs["prompt"]
    assert "[1]" in prompt and "[2]" in prompt
    assert "handbook.pdf, page 1" in prompt
    assert "handbook.pdf, page 2" in prompt


def test_history_is_included_when_present(mock_retrieve, mock_llm):
    mock_retrieve.return_value = [_chunk("Source.")]
    mock_llm.generate_json.return_value = {
        "answer": "Answer.",
        "citations": [1],
        "found": True,
    }

    answer_question(
        "And the pricing?",
        history=[{"role": "user", "content": "Tell me about the Enterprise plan."}],
    )

    prompt = mock_llm.generate_json.call_args.kwargs["prompt"]
    assert "Enterprise plan" in prompt


# --- Happy path ---------------------------------------------------------------


def test_grounded_answer_resolves_citations(mock_retrieve, mock_llm):
    mock_retrieve.return_value = [
        _chunk("Q3 revenue was $4.2M.", page=7),
        _chunk("Headcount grew to 240.", page=8),
    ]
    mock_llm.generate_json.return_value = {
        "answer": "Q3 revenue was $4.2M.",
        "citations": [1],
        "found": True,
    }

    result = answer_question("What was Q3 revenue?")

    assert result.found is True
    assert result.verified is True
    assert len(result.citations) == 1

    citation = result.citations[0]
    assert citation.number == 1
    assert citation.filename == "handbook.pdf"
    assert citation.page_number == 7
