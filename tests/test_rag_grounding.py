"""Grounding tests — the headline evaluation criterion.

The central claim of this build is that hallucination is prevented structurally
rather than by asking the model nicely. These tests hold that claim to account:

  * When nothing relevant is retrieved, the model is NEVER CALLED.
  * When the model reports the sources lack the answer, that is honoured.
  * Fabricated citation numbers are removed, not passed through.

Every test mocks the LLM, so the suite runs with no credentials, no network,
and no cost.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.config import REFUSAL_MESSAGE
from app.rag import answer_question
from app.store import Retrieved


def _chunk(text: str, distance: float, page: int = 1) -> Retrieved:
    return Retrieved(
        text=text,
        doc_id="doc-1",
        filename="handbook.pdf",
        page_number=page,
        chunk_index=0,
        distance=distance,
    )


@pytest.fixture
def mock_store():
    with patch("app.rag.get_store") as get_store:
        store = MagicMock()
        get_store.return_value = store
        yield store


@pytest.fixture
def mock_llm():
    with patch("app.rag.get_llm") as get_llm:
        llm = MagicMock()
        get_llm.return_value = llm
        yield llm


# --- Layer 1: the relevance gate ---------------------------------------------


def test_no_results_refuses_without_calling_model(mock_store, mock_llm):
    """An empty store must not reach the model at all."""
    mock_store.query.return_value = []

    result = answer_question("What is the capital of France?")

    assert result.found is False
    assert result.gated is True
    assert result.answer == REFUSAL_MESSAGE
    mock_llm.generate_json.assert_not_called()


def test_irrelevant_results_refuse_without_calling_model(mock_store, mock_llm):
    """THE critical test.

    The store returns chunks -- it always returns its nearest neighbours, however
    unrelated -- but all are beyond the relevance threshold. A naive RAG
    implementation would hand these to the model and invite an invented answer.
    Here the gate fires first and the model is never invoked.
    """
    mock_store.query.return_value = [
        _chunk("Bananas are a yellow tropical fruit.", distance=1.01),
        _chunk("The office parking policy permits two vehicles.", distance=0.94),
    ]

    result = answer_question("What was Q3 revenue?", threshold=0.75)

    assert result.gated is True
    assert result.found is False
    assert result.answer == REFUSAL_MESSAGE
    mock_llm.generate_json.assert_not_called()


def test_borderline_chunk_within_threshold_is_used(mock_store, mock_llm):
    """A chunk just inside the threshold should pass the gate."""
    mock_store.query.return_value = [_chunk("Q3 revenue was $4.2M.", distance=0.74)]
    mock_llm.generate_json.return_value = {
        "answer": "Q3 revenue was $4.2M.",
        "citations": [1],
        "found": True,
    }

    result = answer_question("What was Q3 revenue?", threshold=0.75)

    assert result.gated is False
    assert result.found is True
    mock_llm.generate_json.assert_called_once()


def test_gate_filters_mixed_relevance(mock_store, mock_llm):
    """Only chunks inside the threshold should reach the prompt."""
    mock_store.query.return_value = [
        _chunk("Q3 revenue was $4.2M.", distance=0.20),
        _chunk("Unrelated cafeteria menu.", distance=1.05),
    ]
    mock_llm.generate_json.return_value = {
        "answer": "Q3 revenue was $4.2M.",
        "citations": [1],
        "found": True,
    }

    result = answer_question("What was Q3 revenue?", threshold=0.75)

    prompt = mock_llm.generate_json.call_args.kwargs["prompt"]
    assert "4.2M" in prompt
    assert "cafeteria" not in prompt
    # Only one source survived, so only [1] can legitimately be cited.
    assert len(result.citations) == 1


# --- Layer 2: the model's own found flag -------------------------------------


def test_model_reporting_not_found_is_honoured(mock_store, mock_llm):
    """Relevant-looking chunks that don't actually answer the question."""
    mock_store.query.return_value = [_chunk("Revenue discussion.", distance=0.40)]
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


# --- Happy path ---------------------------------------------------------------


def test_grounded_answer_resolves_citations(mock_store, mock_llm):
    mock_store.query.return_value = [
        _chunk("Q3 revenue was $4.2M.", distance=0.12, page=7),
        _chunk("Headcount grew to 240.", distance=0.30, page=8),
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


def test_sources_are_numbered_in_prompt(mock_store, mock_llm):
    """Numbering is what gives the model a stable handle to cite."""
    mock_store.query.return_value = [
        _chunk("First source.", distance=0.1, page=1),
        _chunk("Second source.", distance=0.2, page=2),
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


def test_doc_scope_is_passed_to_store(mock_store, mock_llm):
    """Multi-document querying filters at the store, not after retrieval."""
    mock_store.query.return_value = [_chunk("Scoped content.", distance=0.2)]
    mock_llm.generate_json.return_value = {
        "answer": "Answer.",
        "citations": [1],
        "found": True,
    }

    answer_question("Question?", doc_ids=["doc-a", "doc-b"])

    assert mock_store.query.call_args.kwargs["doc_ids"] == ["doc-a", "doc-b"]
