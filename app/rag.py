"""Retrieval-Augmented Generation.

The centrepiece. Grounding is enforced in code, not merely requested in the
prompt. Three layers (execution.md §5):

  1. RELEVANCE GATE   -- if nothing retrieved is relevant enough, refuse
                         WITHOUT calling the model. A model that is never
                         invoked cannot hallucinate.
  2. STRUCTURED OUTPUT -- the model answers into a JSON schema carrying an
                         explicit `found` flag and a machine-readable citation
                         list, so nothing has to be parsed out of prose.
  3. CITATION VALIDATION -- every citation number is checked against what was
                         actually retrieved; anything invented is dropped and
                         the answer is flagged unverified.

Retrieval itself is hybrid (see app/retrieval.py): dense vectors and BM25 are
run in parallel and fused, so both paraphrased and verbatim questions find
their answer. The gate accepts a chunk on either kind of evidence, which makes
it deliberately permissive -- a question the documents genuinely answer should
not be refused, and layers 2 and 3 remain in place behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import REFUSAL_MESSAGE, RELEVANCE_THRESHOLD, TOP_K
from app.llm import LLMError, get_llm
from app.retrieval import retrieve
from app.store import Retrieved

SYSTEM_INSTRUCTION = """\
You answer questions strictly from a set of numbered source passages taken from \
the user's own documents.

Rules:
- Use ONLY the numbered sources provided. Never use outside knowledge.
- Cite the source numbers you actually used in the `citations` field.
- If the sources do not contain the answer, set `found` to false and leave \
`answer` empty. Do not guess, and do not answer from general knowledge.
- Be concise and factual. Quote figures, names, and dates exactly as written.\
"""

# Structured output schema. `found` gives an unambiguous "not in the documents"
# signal rather than a magic token to string-match, and `citations` arrives as
# integers rather than markers to regex out of prose.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer, drawn only from the numbered sources. "
            "Empty when found is false.",
        },
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Source numbers actually used to build the answer.",
        },
        "found": {
            "type": "boolean",
            "description": "True only if the sources contain the answer.",
        },
    },
    "required": ["answer", "citations", "found"],
}


@dataclass
class Citation:
    """A source the answer drew on, resolved back to its document location."""

    number: int
    filename: str
    page_number: int
    text: str


@dataclass
class Answer:
    """The result of one question."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    found: bool = True

    # False when the model returned a citation number that was never retrieved.
    # The answer is still shown, flagged, rather than silently trusted.
    verified: bool = True

    # True when the relevance gate fired -- meaning no model call was made.
    # Surfaced so the behaviour is observable in the API and in tests.
    gated: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "found": self.found,
            "verified": self.verified,
            "gated": self.gated,
            "citations": [
                {
                    "number": c.number,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "text": c.text,
                }
                for c in self.citations
            ],
        }


def answer_question(
    question: str,
    doc_ids: list[str] | None = None,
    history: list[dict] | None = None,
    top_k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
) -> Answer:
    """Answer a question from the indexed documents.

    Args:
        question: The (already condensed) query to retrieve on.
        doc_ids: Optional document scope for multi-document querying.
        history: Prior turns, as {"role", "content"} dicts, for conversational
            phrasing. History never substitutes for retrieval.
        top_k: Chunks to retrieve.
        threshold: Maximum cosine distance for a chunk to be considered
            relevant.

    Returns:
        An `Answer`. Refusals are returned, never raised.
    """
    # ---- Layer 1: hybrid retrieval and the relevance gate ---------------
    # `retrieve` runs dense and lexical search, fuses the rankings, and drops
    # anything that cleared neither bar. An empty result means the documents
    # have nothing to say about this question -- refuse here, with no model
    # call, no chance to hallucinate, and no tokens spent.
    relevant = retrieve(
        query=question, doc_ids=doc_ids, top_k=top_k, threshold=threshold
    )

    if not relevant:
        return Answer(answer=REFUSAL_MESSAGE, found=False, gated=True)

    prompt = _build_prompt(question, relevant, history)

    try:
        result = get_llm().generate_json(
            system_instruction=SYSTEM_INSTRUCTION,
            prompt=prompt,
            response_schema=ANSWER_SCHEMA,
        )
    except LLMError:
        raise

    # ---- Layer 2: the model's own found flag -----------------------------
    # Normalised into the same refusal shape the gate produces, so a caller
    # sees one consistent behaviour regardless of which layer caught it.
    if not result.get("found", False):
        return Answer(answer=REFUSAL_MESSAGE, found=False)

    # ---- Layer 3: citation validation ------------------------------------
    citations, verified = _validate_citations(result.get("citations", []), relevant)

    return Answer(
        answer=result.get("answer", "").strip() or REFUSAL_MESSAGE,
        citations=citations,
        found=True,
        verified=verified,
    )


def _build_prompt(
    question: str,
    chunks: list[Retrieved],
    history: list[dict] | None,
) -> str:
    """Assemble the prompt: numbered sources, optional history, the question.

    Sources are numbered explicitly so the model has a stable, unambiguous
    handle to cite -- the numbers here are exactly what validation checks
    against afterwards.
    """
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{number}] ({chunk.filename}, page {chunk.page_number})\n{chunk.text}"
        )
    sources = "\n\n".join(blocks)

    parts = [f"SOURCES:\n\n{sources}"]

    if history:
        turns = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
        )
        parts.append(f"CONVERSATION SO FAR:\n{turns}")

    parts.append(f"QUESTION: {question}")

    return "\n\n---\n\n".join(parts)


def _validate_citations(
    numbers: list,
    chunks: list[Retrieved],
) -> tuple[list[Citation], bool]:
    """Resolve citation numbers to sources, discarding any that were invented.

    A model can cite [7] when only five sources were supplied. That number
    corresponds to nothing, so it is dropped and the answer is marked
    unverified -- catching it in code is more reliable than trusting the model
    not to do it.

    Returns:
        (resolved citations, all_valid)
    """
    citations: list[Citation] = []
    all_valid = True
    seen: set[int] = set()

    for raw in numbers:
        try:
            number = int(raw)
        except (TypeError, ValueError):
            all_valid = False
            continue

        # Sources are 1-indexed in the prompt.
        if not (1 <= number <= len(chunks)):
            all_valid = False
            continue

        if number in seen:
            continue
        seen.add(number)

        chunk = chunks[number - 1]
        citations.append(
            Citation(
                number=number,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
            )
        )

    return citations, all_valid
