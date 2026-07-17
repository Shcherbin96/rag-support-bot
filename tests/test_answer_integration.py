"""Hermetic end-to-end test: real Chroma retrieval with a mocked LLM.

Exercises retrieval -> prompt assembly -> citation validation without any
network call to the LLM provider. Requires a built index:
    uv run python -m rag_bot.ingestion
No LLM key is needed - the model response is mocked, following the
_client()-injection pattern used in tests/test_answer_unit.py.
"""

import json
from types import SimpleNamespace

import rag_bot.answer as answer_module
from rag_bot.answer import answer
from rag_bot.retrieval import accepted_chunks, retrieve
from rag_bot.router import QueryRoute, classify_query


class _RecordingCompletions:
    """Fake completions endpoint that records every create() call's kwargs."""

    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _RecordingClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_RecordingCompletions(content))


def _quote_from(chunk_text: str) -> str:
    """Return a short, exact substring of a real chunk's text to cite verbatim.

    Stripping leading/trailing markdown bullet/emphasis characters from a line
    only trims the ends, so the result stays a literal contiguous substring of
    chunk_text - required for _parse_model_response's containment check.
    """
    for line in chunk_text.splitlines():
        candidate = line.strip().strip("-* ").strip()
        if len(candidate) > 15:
            assert candidate in chunk_text
            return candidate
    raise AssertionError(f"no usable quote line found in chunk text: {chunk_text!r}")


def test_answer_end_to_end_over_real_retrieval_with_mocked_llm(monkeypatch):
    question = "How much does standard shipping cost?"

    # Sanity-check the real (unmocked) routing/retrieval pipeline before
    # building the fake LLM response around it, so a router or relevance-
    # threshold change fails loudly here instead of as a confusing assert
    # deeper in answer().
    assert classify_query(question) == QueryRoute.FACTUAL_IN_DOMAIN

    retrieved = retrieve(question, k=answer_module.config.TOP_K)
    assert retrieved, "index must be built: uv run python -m rag_bot.ingestion"

    context_chunks = accepted_chunks(retrieved)
    assert context_chunks, "expected at least one chunk past the relevance threshold"

    shipping_chunk = next(
        (chunk for chunk in context_chunks if "shipping" in chunk["source"]), None
    )
    assert shipping_chunk is not None, "expected an accepted shipping chunk in the real index"

    quote = _quote_from(shipping_chunk["text"])
    payload = {
        "answer": f"Here is the shipping info you asked about: {quote}",
        "citations": [{"chunk_id": shipping_chunk["id"], "quote": quote}],
    }
    fake_client = _RecordingClient(json.dumps(payload))
    monkeypatch.setattr(answer_module, "_client", lambda: fake_client)

    result = answer(question)

    assert result["error_type"] == ""
    assert result["sources"], "expected non-empty sources on a grounded answer"
    assert any("shipping" in source for source in result["sources"])
    assert answer_module.CITATION_HEADER in result["text"]

    calls = fake_client.chat.completions.calls
    assert len(calls) == 1
    user_message = calls[0]["messages"][1]["content"]
    # The LLM must have been called with a context built from the real
    # retrieved-and-accepted chunks, not a stub.
    for chunk in context_chunks:
        assert chunk["id"] in user_message
    assert "shipping" in user_message.lower()
