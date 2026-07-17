"""Optional live LLM tests.

These tests require GEMINI_API_KEY and are skipped in CI when the key is absent.
Deterministic answer behavior is covered by mocked unit tests.
"""

import pytest

from rag_bot import config
from rag_bot.answer import answer

needs_key = pytest.mark.skipif(
    not config.LLM_API_KEY, reason="GEMINI_API_KEY is required for live LLM tests"
)


@needs_key
def test_grounded_answer_has_correct_source():
    result = answer("How much is shipping?")
    assert result["sources"]
    assert any("shipping" in source for source in result["sources"])


@needs_key
def test_unknown_fact_is_refused():
    result = answer("What time does your Chicago warehouse close?")
    lowered = result["text"].lower()
    assert any(
        marker in lowered
        for marker in [
            "could not find",
            "cannot invent",
            "do not have reliable",
            "only answer from",
        ]
    )
