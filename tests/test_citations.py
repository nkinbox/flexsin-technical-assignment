"""Citation validation tests — layer 3 of the grounding defence.

A model can produce faithful prose and still emit a citation index that
corresponds to nothing. These tests cover catching that in code rather than
trusting the model not to do it.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.rag import _validate_citations, answer_question
from app.store import Retrieved


def _chunk(text: str, page: int = 1, distance: float = 0.2) -> Retrieved:
    return Retrieved(
        text=text,
        doc_id="doc-1",
        filename="report.pdf",
        page_number=page,
        chunk_index=0,
        distance=distance,
    )


# --- Unit: validation logic ---------------------------------------------------


def test_valid_citations_resolve():
    chunks = [_chunk("First.", page=1), _chunk("Second.", page=2)]
    citations, verified = _validate_citations([1, 2], chunks)

    assert verified is True
    assert [c.number for c in citations] == [1, 2]
    assert [c.page_number for c in citations] == [1, 2]


def test_out_of_range_citation_is_dropped_and_flagged():
    """The model cites [7] when only two sources exist."""
    chunks = [_chunk("First."), _chunk("Second.")]
    citations, verified = _validate_citations([1, 7], chunks)

    assert verified is False
    assert [c.number for c in citations] == [1]


def test_zero_and_negative_are_rejected():
    """Sources are 1-indexed; [0] refers to nothing."""
    chunks = [_chunk("First.")]
    citations, verified = _validate_citations([0, -1], chunks)

    assert verified is False
    assert citations == []


def test_non_integer_citation_is_rejected():
    chunks = [_chunk("First.")]
    citations, verified = _validate_citations(["abc", None], chunks)

    assert verified is False
    assert citations == []


def test_duplicate_citations_are_deduplicated():
    chunks = [_chunk("First.")]
    citations, verified = _validate_citations([1, 1, 1], chunks)

    # A repeat is untidy, not dishonest — dedupe without flagging.
    assert verified is True
    assert len(citations) == 1


def test_empty_citation_list_is_valid():
    chunks = [_chunk("First.")]
    citations, verified = _validate_citations([], chunks)

    assert verified is True
    assert citations == []


def test_string_digits_are_accepted():
    """Tolerate "1" as well as 1 — the meaning is unambiguous."""
    chunks = [_chunk("First.")]
    citations, verified = _validate_citations(["1"], chunks)

    assert verified is True
    assert len(citations) == 1


# --- Integration: through answer_question ------------------------------------


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


def test_fabricated_citation_flags_answer_unverified(mock_store, mock_llm):
    """The answer is still returned — but marked, not silently trusted."""
    mock_store.query.return_value = [_chunk("Revenue was $4.2M.", page=7)]
    mock_llm.generate_json.return_value = {
        "answer": "Revenue was $4.2M.",
        "citations": [1, 5],  # [5] was never supplied
        "found": True,
    }

    result = answer_question("What was revenue?")

    assert result.found is True
    assert result.verified is False
    assert len(result.citations) == 1
    assert result.citations[0].number == 1


def test_all_valid_citations_stay_verified(mock_store, mock_llm):
    mock_store.query.return_value = [_chunk("A.", page=1), _chunk("B.", page=2)]
    mock_llm.generate_json.return_value = {
        "answer": "Both sources agree.",
        "citations": [1, 2],
        "found": True,
    }

    result = answer_question("Question?")

    assert result.verified is True
    assert len(result.citations) == 2


def test_citation_text_matches_retrieved_chunk(mock_store, mock_llm):
    """Citation text comes from the retrieved chunk, never from the model.

    This is what makes a citation verifiable: the quoted source is the passage
    the system actually retrieved, so it cannot be reworded or invented.
    """
    mock_store.query.return_value = [_chunk("Exact source wording.", page=3)]
    mock_llm.generate_json.return_value = {
        "answer": "A paraphrase by the model.",
        "citations": [1],
        "found": True,
    }

    result = answer_question("Question?")

    assert result.citations[0].text == "Exact source wording."


def test_response_dict_shape(mock_store, mock_llm):
    """The API contract the UI relies on."""
    mock_store.query.return_value = [_chunk("Source.", page=1)]
    mock_llm.generate_json.return_value = {
        "answer": "Answer.",
        "citations": [1],
        "found": True,
    }

    payload = answer_question("Question?").to_dict()

    assert set(payload) == {"answer", "found", "verified", "gated", "citations"}
    assert set(payload["citations"][0]) == {
        "number",
        "filename",
        "page_number",
        "text",
    }
