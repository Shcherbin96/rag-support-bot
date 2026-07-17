"""Optional live LLM tests.

These tests require GEMINI_API_KEY and are excluded from the default pytest run
(see the `live` marker and `addopts` in pyproject.toml); run them explicitly with
`uv run pytest -m live`. Deterministic answer behavior is covered by mocked unit
tests.
"""

import pytest

from rag_bot import config
from rag_bot.answer import answer
from rag_bot.router import QueryRoute

needs_key = pytest.mark.skipif(
    not config.LLM_API_KEY, reason="GEMINI_API_KEY is required for live LLM tests"
)


@pytest.mark.live
@needs_key
def test_grounded_answer_has_correct_source():
    result = answer("How much is shipping?")
    assert result["sources"]
    assert any("shipping" in source for source in result["sources"])


@pytest.mark.live
@needs_key
def test_unknown_fact_is_refused():
    result = answer("What time does your overnight warehouse shift end?")

    # A provider_error (e.g. a 429 under quota exhaustion) also returns
    # REFUSAL_TEXT, which would satisfy the marker check below without a real
    # round-trip through the LLM. Require a genuine grounded refusal: the call
    # must have succeeded (no error_type) and gone through in-domain retrieval,
    # not a rate-limit failure or an out-of-domain short-circuit.
    assert result["error_type"] == ""
    assert result["route"] == QueryRoute.FACTUAL_IN_DOMAIN

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
