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


class _FakeClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def test_out_of_domain_query_is_refused_before_retrieval(monkeypatch):
    def fail_retrieve(*args, **kwargs):
        raise AssertionError("retrieval should not be called")

    monkeypatch.setattr(answer_module, "retrieve", fail_retrieve)
    result = answer("What is the weather in Moscow?")

    assert result["sources"] == []
    assert result["route"] == "out_of_domain"
    assert "DomOk" in result["text"]


def test_smalltalk_is_handled_before_retrieval(monkeypatch):
    def fail_retrieve(*args, **kwargs):
        raise AssertionError("retrieval should not be called")

    monkeypatch.setattr(answer_module, "retrieve", fail_retrieve)
    result = answer("Привет! Кто ты?")

    assert result["sources"] == []
    assert result["route"] == "smalltalk"
    assert "ДомОк" in result["text"]


def test_answer_uses_only_validated_cited_sources(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
        {"id": "chunk-2", "source": "oplata.md", "distance": 0.4, "text": "Оплата картой или СБП."},
    ]
    payload = {"answer": "Доставка стоит 350 рублей.", "citations": ["chunk-1"]}

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Сколько стоит доставка?")

    assert result["sources"] == ["dostavka.md"]
    assert "Источники: dostavka.md" in result["text"]
    assert "oplata.md" not in result["text"]


def test_invalid_model_citation_fails_closed(monkeypatch):
    chunks = [
        {"id": "chunk-1", "source": "dostavka.md", "distance": 0.2, "text": "Доставка стоит 350 рублей."},
    ]
    payload = {"answer": "Доставка стоит 350 рублей.", "citations": ["not-retrieved"]}

    monkeypatch.setattr(answer_module, "retrieve", lambda query, k: chunks)
    monkeypatch.setattr(answer_module, "_client", lambda: _FakeClient(json.dumps(payload)))

    result = answer("Сколько стоит доставка?")

    assert result["sources"] == []
    assert "не могу выдумывать" in result["text"]
