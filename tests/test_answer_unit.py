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
    result = answer("What is the weather in Moscow?")

    assert result["sources"] == []
    assert result["route"] == "out_of_domain"
    assert result["error_type"] == ""
    assert "DomOk" in result["text"]


def test_smalltalk_is_handled_before_retrieval(monkeypatch):
    def fail_retrieve(*args, **kwargs):
        raise AssertionError("retrieval should not be called")

    monkeypatch.setattr(answer_module, "retrieve", fail_retrieve)
    result = answer("Привет! Кто ты?")

    assert result["sources"] == []
    assert result["route"] == "smalltalk"
    assert result["error_type"] == ""
    assert "ДомОк" in result["text"]


def test_mixed_greeting_and_question_uses_retrieval(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
    ]
    payload = {
        "answer": "Доставка стоит 350 рублей.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Доставка стоит 350 рублей."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Привет, сколько стоит доставка?")

    assert result["route"] == "factual_in_domain"
    assert result["sources"] == ["dostavka.md"]


def test_answer_uses_only_validated_cited_sources(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
        {"id": "chunk-2", "source": "oplata.md", "distance": 0.4, "text": "Оплата картой или СБП."},
    ]
    payload = {
        "answer": "Доставка стоит 350 рублей.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Доставка стоит 350 рублей."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Сколько стоит доставка?")

    assert result["sources"] == ["dostavka.md"]
    assert "Источники: dostavka.md" in result["text"]
    assert "oplata.md" not in result["text"]
    assert not result["error_type"]


def test_invalid_model_citation_fails_closed(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
    ]
    payload = {
        "answer": "Доставка стоит 350 рублей.",
        "citations": [{"chunk_id": "not-retrieved", "quote": "Доставка стоит 350 рублей."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Сколько стоит доставка?")

    assert result["sources"] == []
    assert result["error_type"] == "model_contract_error"
    assert "не могу выдумывать" in result["text"]


def test_answer_rejects_quote_not_present_in_cited_chunk(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
    ]
    payload = {
        "answer": "Доставка стоит 350 рублей.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Доставка стоит 99999 рублей."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Сколько стоит доставка?")

    assert result["sources"] == []
    assert result["error_type"] == "model_contract_error"


def test_answer_rejects_numeric_claim_not_present_in_evidence(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
    ]
    payload = {
        "answer": "Доставка стоит 99999 рублей.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Доставка стоит 350 рублей."}],
    }

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Сколько стоит доставка?")

    assert result["sources"] == []
    assert result["error_type"] == "model_contract_error"


def test_retrieval_exception_fails_closed(monkeypatch):
    def broken_retrieve(*args, **kwargs):
        raise RuntimeError("embedding backend failed")

    monkeypatch.setattr(answer_module, "retrieve", broken_retrieve)
    result = answer("Сколько стоит доставка?")

    assert result["sources"] == []
    assert result["error_type"] == "retrieval_error"


def test_transient_provider_error_is_marked_retryable(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FailingClient(RuntimeError("429 rate limit")))

    result = answer("Сколько стоит доставка?")

    assert result["error_type"] == "provider_error"
    assert result["retryable"] is True


def test_permanent_provider_error_is_not_retryable(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
    ]

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FailingClient(RuntimeError("invalid api key")))

    result = answer("Сколько стоит доставка?")

    assert result["error_type"] == "provider_error"
    assert result["retryable"] is False
