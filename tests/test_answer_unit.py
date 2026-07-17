"""Deterministic unit tests for answer routing and citation validation."""

import json
from types import SimpleNamespace

import httpx
import openai

import rag_bot.answer as answer_module
from rag_bot.answer import AnswerError, _numbers, answer


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


class _FallbackCompletions:
    """Fake completions endpoint that rejects the first call, then succeeds."""

    def __init__(self, exc: Exception, content: str):
        self.exc = exc
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _FallbackClient:
    def __init__(self, exc: Exception, content: str):
        self.chat = SimpleNamespace(completions=_FallbackCompletions(exc, content))


def _bad_request_error(message: str) -> openai.BadRequestError:
    """Build a real openai.BadRequestError with the SDK's expected shape."""
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return openai.BadRequestError(message, response=response, body={"error": {"message": message}})


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
        {
            "id": "chunk-1",
            "source": "shipping.md",
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

    result = answer("Hi, how much is shipping?")

    assert result["route"] == "factual_in_domain"
    assert result["sources"] == ["shipping.md"]


def test_answer_uses_only_validated_cited_sources(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
        {
            "id": "chunk-2",
            "source": "payment.md",
            "distance": 0.4,
            "text": "We accept cards and PayPal.",
        },
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
    assert "02_shipping.md" not in result["text"]  # no raw filename shown to the user
    assert result["sources"] == ["02_shipping.md"]  # machine-facing field keeps filenames


def test_invalid_model_citation_fails_closed(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
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
    assert result["error_type"] == AnswerError.MODEL_CONTRACT_ERROR
    assert "cannot invent" in result["text"]


def test_answer_rejects_quote_not_present_in_cited_chunk(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
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
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
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
    assert result["error_type"] == AnswerError.RETRIEVAL_ERROR


def test_answer_error_enum_values_match_documented_set():
    assert {member.value for member in AnswerError} == {
        "missing_index",
        "retrieval_error",
        "provider_error",
        "model_contract_error",
    }


def test_missing_index_error_path_returns_enum_value(monkeypatch):
    from rag_bot.retrieval import KnowledgeBaseNotReadyError

    def raise_not_ready(*args, **kwargs):
        raise KnowledgeBaseNotReadyError("index missing")

    monkeypatch.setattr(answer_module, "retrieve", raise_not_ready)
    result = answer("How much is shipping?")

    assert result["error_type"] == AnswerError.MISSING_INDEX
    assert result["error_type"] == "missing_index"


def test_transient_provider_error_is_marked_retryable(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(
        answer_module, "_client", lambda: _FailingClient(RuntimeError("429 rate limit"))
    )

    result = answer("How much is shipping?")

    assert result["error_type"] == "provider_error"
    assert result["retryable"] is True


def test_permanent_provider_error_is_not_retryable(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(
        answer_module, "_client", lambda: _FailingClient(RuntimeError("invalid api key"))
    )

    result = answer("How much is shipping?")

    assert result["error_type"] == "provider_error"
    assert result["retryable"] is False


def test_provider_timeout_is_marked_retryable(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(
        answer_module, "_client", lambda: _FailingClient(RuntimeError("Request timed out."))
    )

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
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
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


def test_numbers_treats_comma_as_thousands_separator():
    # US store: comma separates thousands, period is the decimal point.
    assert _numbers("8,000") == _numbers("8000") == {"8000"}
    assert _numbers("$5.99") == {"5.99"}
    assert _numbers("1,299.50") == {"1299.50"}


def test_answer_accepts_number_written_without_thousands_comma(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "catalog.md",
            "distance": 0.2,
            "text": "We stock over 8,000 products across all categories.",
        },
    ]
    payload = {
        "answer": "We stock over 8000 products.",
        "citations": [{"chunk_id": "chunk-1", "quote": "over 8,000 products"}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("How many products do you stock?")

    assert result["error_type"] == ""
    assert "We stock over 8000 products." in result["text"]


def test_no_accepted_context_refuses_with_visible_log(monkeypatch, caplog):
    # All chunks are past the relevance threshold, so accepted_chunks() is empty.
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 5.0,
            "text": "Standard shipping costs $5.99.",
        },
    ]

    def fail_client():
        raise AssertionError("LLM should not be called when no context is accepted")

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", fail_client)

    with caplog.at_level("INFO", logger="nestwell-answer"):
        result = answer("How much is shipping?")

    # Controlled refusal, not a failure: error_type stays empty by contract.
    assert result["error_type"] == ""
    assert result["refusal_reason"] == "no_accepted_context"
    assert "refusal reason=no_accepted_context" in caplog.text


def test_json_mode_on_passes_response_format(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }
    fake_client = _RecordingClient(json.dumps(payload))

    monkeypatch.setattr(answer_module.config, "LLM_JSON_MODE", True)
    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: fake_client)

    result = answer("How much is shipping?")

    assert result["error_type"] == ""
    calls = fake_client.chat.completions.calls
    assert len(calls) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_json_mode_off_omits_response_format(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }
    fake_client = _RecordingClient(json.dumps(payload))

    monkeypatch.setattr(answer_module.config, "LLM_JSON_MODE", False)
    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: fake_client)

    result = answer("How much is shipping?")

    assert result["error_type"] == ""
    calls = fake_client.chat.completions.calls
    assert len(calls) == 1
    assert "response_format" not in calls[0]


def test_json_mode_bad_request_falls_back_without_response_format(monkeypatch, caplog):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]
    payload = {
        "answer": "Standard shipping costs $5.99.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Standard shipping costs $5.99."}],
    }
    exc = _bad_request_error(
        "Invalid parameter: 'response_format' is not supported for this model."
    )
    fake_client = _FallbackClient(exc, json.dumps(payload))

    monkeypatch.setattr(answer_module.config, "LLM_JSON_MODE", True)
    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: fake_client)

    with caplog.at_level("INFO", logger="nestwell-answer"):
        result = answer("How much is shipping?")

    assert result["error_type"] == ""
    assert result["sources"] == ["shipping.md"]
    calls = fake_client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert "json_mode_unsupported" in caplog.text


def test_json_mode_unrelated_bad_request_does_not_retry(monkeypatch):
    chunks = [
        {
            "id": "chunk-1",
            "source": "shipping.md",
            "distance": 0.2,
            "text": "Standard shipping costs $5.99.",
        },
    ]
    exc = _bad_request_error("Invalid API key provided.")

    class _AlwaysFailingCompletions:
        def __init__(self, exc: Exception):
            self.exc = exc
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise self.exc

    completions = _AlwaysFailingCompletions(exc)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(answer_module.config, "LLM_JSON_MODE", True)
    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: fake_client)

    result = answer("How much is shipping?")

    # Not a response_format capability gap, so it must surface as a normal
    # (non-retryable) provider_error rather than silently falling back.
    assert result["error_type"] == AnswerError.PROVIDER_ERROR
    assert result["retryable"] is False
    assert len(completions.calls) == 1
