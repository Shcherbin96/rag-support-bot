"""Deterministic unit tests for answer routing and citation validation."""

import json
from types import SimpleNamespace

import rag_bot.answer as answer_module
from rag_bot.answer import answer


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _FailingCompletions:
    def __init__(self, exc: Exception):
        self.exc = exc

    def create(self, **kwargs):
        raise self.exc


class _FakeClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


class _FailingClient:
    def __init__(self, exc: Exception):
        self.chat = SimpleNamespace(completions=_FailingCompletions(exc))


def test_out_of_domain_query_is_refused_before_retrieval(monkeypatch):
    def fail_retrieve(*args, **kwargs):
        raise AssertionError("retrieval should not be called")

    monkeypatch.setattr(answer_module, "retrieve", fail_retrieve)
    result = answer("What is the weather today?")

    assert result["sources"] == []
    assert result["route"] == "out_of_domain"
    assert result["error_type"] == ""
    assert "Nestwell" in result["text"]


def test_smalltalk_is_handled_before_retrieval(monkeypatch):
    def fail_retrieve(*args, **kwargs):
        raise AssertionError("retrieval should not be called")

    monkeypatch.setattr(answer_module, "retrieve", fail_retrieve)
    result = answer("Hello, who are you?")

    assert result["sources"] == []
    assert result["route"] == "smalltalk"
    assert result["error_type"] == ""
    assert "Nestwell" in result["text"]


def test_mixed_greeting_and_question_uses_retrieval(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Hi, how much is shipping?")

    assert result["route"] == "factual_in_domain"
    assert result["sources"] == ["shipping.md"]


def test_answer_uses_only_validated_cited_sources(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
        {"id": "chunk-2", "source": "payment.md", "distance": 0.4, "text": "We accept cards and PayPal."},
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("How much is shipping?")

    assert result["sources"] == ["shipping.md"]
    assert "Sources: shipping.md" in result["text"]
    assert "payment.md" not in result["text"]
    assert not result["error_type"]


def test_citation_footer_prefers_human_readable_title(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "02_shipping.md",
            "title": "Shipping",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("How much is shipping?")

    assert "Sources: Shipping" in result["text"]
    assert "02_shipping.md" not in result["text"]      # no raw filename shown to the user
    assert result["sources"] == ["02_shipping.md"]     # machine-facing field keeps filenames


def test_invalid_model_citation_fails_closed(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "not-retrieved", "quote": "Standard shipping costs $5.99."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("How much is shipping?")

    assert result["sources"] == []
    assert result["error_type"] == "model_contract_error"
    assert "cannot invent" in result["text"]


def test_answer_rejects_quote_not_present_in_cited_chunk(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $999.99."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("How much is shipping?")

    assert result["sources"] == []
    assert result["error_type"] == "model_contract_error"


def test_answer_rejects_numeric_claim_not_present_in_evidence(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]
    payload = {
        "answer": "Standard shipping costs $999.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("How much is shipping?")

    assert result["sources"] == []
    assert result["error_type"] == "model_contract_error"


def test_retrieval_exception_fails_closed(monkeypatch):
    def broken_retrieve(*args, **kwargs):
        raise RuntimeError("embedding backend failed")

    monkeypatch.setattr(answer_module, "retrieve", broken_retrieve)
    result = answer("How much is shipping?")

    assert result["sources"] == []
    assert result["error_type"] == "retrieval_error"


def test_transient_provider_error_is_marked_retryable(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FailingClient(RuntimeError("429 rate limit")))

    result = answer("How much is shipping?")

    assert result["error_type"] == "provider_error"
    assert result["retryable"] is True


def test_permanent_provider_error_is_not_retryable(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FailingClient(RuntimeError("invalid api key")))

    result = answer("How much is shipping?")

    assert result["error_type"] == "provider_error"
    assert result["retryable"] is False


def test_provider_timeout_is_marked_retryable(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FailingClient(RuntimeError("Request timed out.")))

    result = answer("How much is shipping?")

    assert result["error_type"] == "provider_error"
    assert result["retryable"] is True


def test_llm_timeout_is_bounded_below_bot_wait():
    # The bot wraps answer() in a 45s wait; the client timeout must fire first.
    assert 0 < answer_module.config.LLM_TIMEOUT < 45


def test_client_disables_sdk_retries(monkeypatch):
    monkeypatch.setattr(answer_module.config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(answer_module.config, "LLM_BASE_URL", "https://example.test/v1")

    client = answer_module._client()

    assert client.max_retries == 0


def test_model_contract_rejection_reason_is_logged(monkeypatch, caplog):
    chunks = [
        {"id": "chunk-1", "source": "shipping.md", "distance": 0.2, "text": "Standard shipping costs $5.99."},
    ]
    payload = {
        "answer": "Standard shipping costs $999.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    with caplog.at_level("INFO", logger="nestwell-answer"):
        result = answer("How much is shipping?")

    assert result["error_type"] == "model_contract_error"
    assert "model_contract_rejected" in caplog.text
    assert "999.99" in caplog.text
