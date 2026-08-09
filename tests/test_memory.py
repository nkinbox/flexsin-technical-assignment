"""Memory tests — history retention and follow-up condensing."""

from unittest.mock import MagicMock, patch

import pytest

from app import memory


@pytest.fixture(autouse=True)
def clean_session():
    memory.clear("s1")
    yield
    memory.clear("s1")


def test_history_round_trips():
    memory.add_turn("s1", "user", "What is the Enterprise plan?")
    memory.add_turn("s1", "assistant", "It costs $99/month.")

    history = memory.get_history("s1")

    assert len(history) == 2
    assert history[0]["role"] == "user"


def test_history_is_trimmed():
    for i in range(50):
        memory.add_turn("s1", "user", f"q{i}")
        memory.add_turn("s1", "assistant", f"a{i}")

    history = memory.get_history("s1")

    assert len(history) <= memory.MAX_HISTORY_TURNS * 2
    # Trimming must keep the most recent turns, which are the ones that matter
    # for condensing the next question.
    assert history[-1]["content"] == "a49"


def test_sessions_are_isolated():
    memory.add_turn("s1", "user", "private")
    assert memory.get_history("other") == []
    memory.clear("other")


# --- Condensing gate ----------------------------------------------------------


def test_no_condensing_without_history():
    assert memory.needs_condensing("What about pricing?", []) is False


def test_pronoun_question_needs_condensing():
    history = [{"role": "user", "content": "Tell me about the Enterprise plan."}]
    assert memory.needs_condensing("What about its pricing?", history) is True


def test_short_question_needs_condensing():
    """Short questions are almost always elliptical follow-ups."""
    history = [{"role": "user", "content": "Tell me about the plan."}]
    assert memory.needs_condensing("And support?", history) is True


def test_standalone_question_skips_condensing():
    """Self-contained questions shouldn't pay for a needless round-trip."""
    history = [{"role": "user", "content": "Tell me about the Enterprise plan."}]
    question = "What were the total operating expenses reported for fiscal 2024?"

    assert memory.needs_condensing(question, history) is False


# --- Condensing behaviour -----------------------------------------------------


def test_condense_rewrites_follow_up():
    history = [
        {"role": "user", "content": "Tell me about the Enterprise plan."},
        {"role": "assistant", "content": "It is our top tier."},
    ]

    with patch("app.memory.get_llm") as get_llm:
        llm = MagicMock()
        llm.generate_text.return_value = "What is the pricing of the Enterprise plan?"
        get_llm.return_value = llm

        result = memory.condense("What about its pricing?", history)

    assert result == "What is the pricing of the Enterprise plan?"


def test_condense_falls_back_on_llm_error():
    """A failed rewrite must not fail the turn — degrade to the raw question."""
    from app.llm import LLMError

    history = [{"role": "user", "content": "Tell me about the plan."}]

    with patch("app.memory.get_llm") as get_llm:
        llm = MagicMock()
        llm.generate_text.side_effect = LLMError("unavailable")
        get_llm.return_value = llm

        result = memory.condense("What about it?", history)

    assert result == "What about it?"


def test_condense_rejects_implausible_rewrite():
    """If the model answers instead of rewriting, keep the original."""
    history = [{"role": "user", "content": "Tell me about the plan."}]

    with patch("app.memory.get_llm") as get_llm:
        llm = MagicMock()
        llm.generate_text.return_value = "x" * 5000
        get_llm.return_value = llm

        result = memory.condense("What about it?", history)

    assert result == "What about it?"


def test_condense_skipped_makes_no_llm_call():
    history = [{"role": "user", "content": "Tell me about the plan."}]
    question = "What were the total operating expenses reported for fiscal 2024?"

    with patch("app.memory.get_llm") as get_llm:
        result = memory.condense(question, history)
        get_llm.assert_not_called()

    assert result == question
