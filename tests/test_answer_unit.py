"""Deterministic unit tests for answer routing and citation validation."""

import json
from types import SimpleNamespace

import httpx
import openai

import rag_bot.answer as answer_module
from rag_bot.answer import AnswerError, _numbers, _parse_model_response, answer


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


def test_answer_accepts_number_written_without_thousands_comma():
    # The comma-as-thousands-separator behavior lives in _parse_model_response's
    # numeric-claim check, so test it there directly. Driving the whole routed
    # answer() would drag in the semantic router, whose embedding similarities
    # vary slightly across torch/BLAS backends and can flip a borderline query's
    # route (e.g. "How many products do you stock?" routing out_of_domain on some
    # platforms) - making a deterministic unit assertion non-deterministic by OS.
    valid_chunks = {
        "chunk-1": {
            "id": "chunk-1",
            "source": "catalog.md",
            "text": "We stock over 8,000 products across all categories.",
        },
    }
    raw_json = json.dumps(
        {
            "answer": "We stock over 8000 products.",
            "citations": [{"chunk_id": "chunk-1", "quote": "over 8,000 products"}],
        }
    )

    # "8000" in the answer must be treated as covered by "8,000" in the evidence
    # quote, so parsing succeeds instead of raising on an unsupported number.
    answer_text, citations = _parse_model_response(raw_json, valid_chunks)

    assert answer_text == "We stock over 8000 products."
    assert citations == [{"chunk_id": "chunk-1", "quote": "over 8,000 products"}]


def test_answer_accepts_quote_that_drops_markdown_bold_from_chunk():
    # The knowledge base stores facts in markdown (e.g. chunk-31, contact info),
    # but the model naturally quotes the plain-text rendering. Under a plain
    # whitespace+casefold containment check, "phone: 1-800-..." is not a
    # substring of "- **phone:** 1-800-..." because of the "**" markers, which
    # produced a systematic false refusal (model_contract_error) on every
    # bold/markdown-formatted fact: phone, prices, warranty terms.
    valid_chunks = {
        "chunk-31": {
            "id": "chunk-31",
            "source": "08_contact.md",
            "text": (
                "## Support\n"
                "- **Phone:** 1-800-555-0142 (toll-free)\n"
                "- **Email:** support@nestwell.example"
            ),
        },
    }
    raw_json = json.dumps(
        {
            "answer": "You can reach us by phone at 1-800-555-0142 (toll-free).",
            "citations": [{"chunk_id": "chunk-31", "quote": "Phone: 1-800-555-0142 (toll-free)"}],
        }
    )

    answer_text, citations = _parse_model_response(raw_json, valid_chunks)

    assert answer_text == "You can reach us by phone at 1-800-555-0142 (toll-free)."
    assert citations == [{"chunk_id": "chunk-31", "quote": "Phone: 1-800-555-0142 (toll-free)"}]


def test_answer_still_rejects_a_fabricated_quote_not_in_the_chunk():
    # Anti-hallucination guarantee preserved: a quote whose content (not just
    # its markdown formatting) differs from the chunk must still be rejected.
    valid_chunks = {
        "chunk-31": {
            "id": "chunk-31",
            "source": "08_contact.md",
            "text": (
                "## Support\n"
                "- **Phone:** 1-800-555-0142 (toll-free)\n"
                "- **Email:** support@nestwell.example"
            ),
        },
    }
    raw_json = json.dumps(
        {
            "answer": "You can reach us by phone at 1-800-999-0000.",
            "citations": [{"chunk_id": "chunk-31", "quote": "Phone: 1-800-999-0000"}],
        }
    )

    try:
        _parse_model_response(raw_json, valid_chunks)
        raised = False
    except ValueError as exc:
        raised = True
        assert "not present in cited chunk" in str(exc)

    assert raised, "fabricated quote must still raise, markdown-insensitivity must not gut this"


def test_answer_accepts_quote_that_adds_markdown_emphasis_not_in_chunk():
    # Symmetric case: the model's quote adds emphasis markers that are not in
    # the chunk's plain text. Still the same underlying words, so it matches.
    valid_chunks = {
        "chunk-2": {
            "id": "chunk-2",
            "source": "02_shipping.md",
            "text": "Standard shipping — $5.99",
        },
    }
    raw_json = json.dumps(
        {
            "answer": "Standard shipping costs $5.99.",
            "citations": [{"chunk_id": "chunk-2", "quote": "**Standard shipping** — $5.99"}],
        }
    )

    answer_text, citations = _parse_model_response(raw_json, valid_chunks)

    assert answer_text == "Standard shipping costs $5.99."
    assert citations == [{"chunk_id": "chunk-2", "quote": "**Standard shipping** — $5.99"}]


def test_answer_rejects_quote_that_would_fabricate_a_number_via_merged_markdown():
    # Anti-hallucination false-accept: the chunk says "1**2**" (i.e. "1-2" with
    # markdown emphasis wrapping the "2"). If emphasis markers are stripped to
    # EMPTY STRING, "1**2**" normalizes to "12", so a citation quoting "12"
    # would wrongly be accepted as present in the chunk - fabricating a number
    # ("12") the source never stated (it says 1-2, i.e. one to two). Stripping
    # to a SPACE instead keeps "1 2" in the chunk, so "12" is not a substring
    # and the citation (and thus the whole answer) must be rejected.
    valid_chunks = {
        "chunk-9": {
            "id": "chunk-9",
            "source": "07_shipping_speed.md",
            "text": "The item ships in 1**2** business days.",
        },
    }
    raw_json = json.dumps(
        {
            "answer": "It ships in 12 business days.",
            "citations": [{"chunk_id": "chunk-9", "quote": "ships in 12 business days"}],
        }
    )

    try:
        _parse_model_response(raw_json, valid_chunks)
        raised = False
    except ValueError:
        raised = True

    assert raised, "fabricated number '12' must be rejected, not accepted via markdown-merge"


def test_answer_rejects_underscore_merged_digits():
    # Same false-accept shape with "_" instead of "*": "A_100" must not
    # normalize to a chunk containing "A100" as a substring.
    valid_chunks = {
        "chunk-10": {
            "id": "chunk-10",
            "source": "10_model_numbers.md",
            "text": "Model A_100 is the base tier.",
        },
    }
    raw_json = json.dumps(
        {
            "answer": "The base tier model is A100.",
            "citations": [{"chunk_id": "chunk-10", "quote": "Model A100 is the base tier"}],
        }
    )

    try:
        _parse_model_response(raw_json, valid_chunks)
        raised = False
    except ValueError:
        raised = True

    assert raised, "'A100' must not match a chunk that only contains 'A_100'"


def test_answer_rejects_backtick_merged_digits():
    # Same false-accept shape with a backtick code marker.
    valid_chunks = {
        "chunk-11": {
            "id": "chunk-11",
            "source": "11_codes.md",
            "text": "Reference code `5`10 is required at checkout.",
        },
    }
    raw_json = json.dumps(
        {
            "answer": "The reference code is 510.",
            "citations": [{"chunk_id": "chunk-11", "quote": "Reference code 510 is required"}],
        }
    )

    try:
        _parse_model_response(raw_json, valid_chunks)
        raised = False
    except ValueError:
        raised = True

    assert raised, "'510' must not match a chunk that only contains '`5`10'"


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
