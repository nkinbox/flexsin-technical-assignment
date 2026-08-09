"""Bonus — Chat memory with retrieval-aware follow-up condensing.

Plain history is easy; the part that matters for RAG is that a follow-up
question is often useless as a *retrieval query*:

    User: "What does the Enterprise plan include?"
    User: "What about its pricing?"      <-- embeds to near-nothing

"its" carries the entire meaning, and the resulting vector is close to
meaningless. Before retrieving, such questions are rewritten into standalone
form using recent history:

    "What about its pricing?"  ->  "What is the pricing of the Enterprise plan?"

That rewrite is what gets embedded. It is the single highest-impact fix for
multi-turn retrieval quality.

History is in-process: correct for a single-user POC, and the only module that
would change if sessions needed to survive a restart.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.config import MAX_HISTORY_TURNS
from app.llm import LLMError, get_llm

CONDENSE_INSTRUCTION = """\
Rewrite the user's follow-up question as a standalone question that can be \
understood without the conversation.

Rules:
- Replace pronouns and references ("it", "that", "the company") with what they \
actually refer to, based on the conversation.
- Preserve the user's intent exactly. Do not answer, expand, or add detail.
- If the question already stands alone, return it unchanged.
- Return ONLY the rewritten question, with no preamble.\
"""

# Signals that a question probably depends on prior context.
_DEPENDENT_PATTERN = re.compile(
    r"^\s*(and|but|what about|how about|why|and what|ok|also)\b"
    r"|\b(it|its|it's|they|them|their|that|this|those|these|he|she|his|her)\b",
    re.IGNORECASE,
)

_history: dict[str, list[dict]] = defaultdict(list)


def get_history(session_id: str) -> list[dict]:
    """Return the retained turns for a session, oldest first."""
    return list(_history[session_id])


def add_turn(session_id: str, role: str, content: str) -> None:
    """Append a turn, trimming to the retention window."""
    _history[session_id].append({"role": role, "content": content})

    # Two entries per exchange (user + assistant).
    limit = MAX_HISTORY_TURNS * 2
    if len(_history[session_id]) > limit:
        _history[session_id] = _history[session_id][-limit:]


def clear(session_id: str) -> None:
    """Drop a session's history."""
    _history.pop(session_id, None)


def needs_condensing(question: str, history: list[dict]) -> bool:
    """Decide whether a rewrite is worth a model call.

    Gated deliberately: condensing every question would add a round-trip to
    turns that gain nothing from it. A question is treated as dependent when it
    contains a pronoun/reference, opens with a continuation word, or is very
    short (short questions are almost always elliptical follow-ups).
    """
    if not history:
        return False

    if len(question.split()) <= 5:
        return True

    return bool(_DEPENDENT_PATTERN.search(question))


def condense(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone retrieval query.

    Falls back to the original question if the rewrite fails or looks
    implausible -- a degraded query still retrieves something, whereas an
    exception here would fail the whole turn for a non-essential optimisation.
    """
    if not needs_condensing(question, history):
        return question

    recent = history[-4:]
    conversation = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent
    )

    prompt = (
        f"CONVERSATION:\n{conversation}\n\n"
        f"FOLLOW-UP QUESTION: {question}\n\n"
        "STANDALONE QUESTION:"
    )

    try:
        rewritten = get_llm().generate_text(
            system_instruction=CONDENSE_INSTRUCTION,
            prompt=prompt,
            max_output_tokens=256,
        )
    except LLMError:
        return question

    rewritten = rewritten.strip().strip('"')

    # Sanity check: an empty or wildly long result means the rewrite went wrong
    # (the model answered instead of rewriting, say). Prefer the original.
    if not rewritten or len(rewritten) > 4 * len(question) + 200:
        return question

    return rewritten
