"""Unit tests for Telegram bot helper behavior."""

import asyncio
from types import SimpleNamespace

import pytest

import rag_bot.bot as bot_module


def test_greeting_is_english_and_names_the_store():
    assert bot_module.GREETING.startswith("👋 Hi.")
    assert "Nestwell" in bot_module.GREETING
    assert "How much is shipping?" in bot_module.GREETING


def test_fallback_message_is_english():
    assert "I could not process" in bot_module.FALLBACK_MESSAGE
    assert "human support agent" in bot_module.FALLBACK_MESSAGE


def test_split_for_telegram_keeps_short_message():
    assert bot_module._split_for_telegram("hello") == ["hello"]


def test_split_for_telegram_splits_long_message():
    text = "x" * (bot_module.TELEGRAM_LIMIT + 10)

    parts = bot_module._split_for_telegram(text)

    assert len(parts) == 2
    assert "".join(parts) == text
    assert all(len(part) <= bot_module.TELEGRAM_LIMIT for part in parts)


def test_validate_runtime_config_reports_missing_selected_provider_key(monkeypatch):
    monkeypatch.setattr(bot_module.config, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(bot_module.config, "LLM_PROVIDER", "nvidia")
    monkeypatch.setattr(bot_module.config, "SUPPORTED_LLM_PROVIDERS", {"gemini", "nvidia"})
    monkeypatch.setattr(bot_module.config, "LLM_API_KEY", "")
    monkeypatch.setattr(bot_module.config, "LLM_API_KEY_ENV", "NVIDIA_API_KEY")

    with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN, NVIDIA_API_KEY"):
        bot_module._validate_runtime_config()


def test_validate_runtime_config_reports_invalid_provider(monkeypatch):
    monkeypatch.setattr(bot_module.config, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(bot_module.config, "LLM_PROVIDER", "unknown")
    monkeypatch.setattr(bot_module.config, "SUPPORTED_LLM_PROVIDERS", {"gemini", "nvidia"})
    monkeypatch.setattr(bot_module.config, "LLM_API_KEY", "key")

    with pytest.raises(SystemExit, match="LLM_PROVIDER must be one of"):
        bot_module._validate_runtime_config()


def test_validate_runtime_config_accepts_required_values(monkeypatch):
    monkeypatch.setattr(bot_module.config, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(bot_module.config, "LLM_PROVIDER", "nvidia")
    monkeypatch.setattr(bot_module.config, "SUPPORTED_LLM_PROVIDERS", {"gemini", "nvidia"})
    monkeypatch.setattr(bot_module.config, "LLM_API_KEY", "key")
    monkeypatch.setattr(bot_module.config, "LLM_API_KEY_ENV", "NVIDIA_API_KEY")

    bot_module._validate_runtime_config()


def test_on_question_survives_fallback_send_failure(monkeypatch):
    # answer() succeeds, but every Telegram send raises (e.g. network down):
    # the normal reply fails, and so must the fallback send. The handler must
    # swallow both and not raise out of on_question.
    monkeypatch.setattr(
        bot_module,
        "answer",
        lambda question: {"text": "ok", "route": "smalltalk", "sources": [], "error_type": ""},
    )

    class _Bot:
        async def send_chat_action(self, *args, **kwargs):
            return None

    class _Msg:
        text = "hi"
        chat = SimpleNamespace(id=1)
        bot = _Bot()

        async def answer(self, text):
            raise RuntimeError("telegram unreachable")

    # Must complete without propagating an exception.
    asyncio.run(bot_module.on_question(_Msg()))


def test_warm_up_loads_model_and_index(monkeypatch):
    calls = []
    monkeypatch.setattr(bot_module.embeddings, "get_model", lambda: calls.append("model"))
    monkeypatch.setattr(bot_module, "retrieve", lambda query, k: calls.append("retrieve") or [])

    bot_module._warm_up()

    assert calls == ["model", "retrieve"]


def test_warm_up_swallows_errors(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(bot_module.embeddings, "get_model", boom)

    # A missing model/index must not stop the bot from starting.
    bot_module._warm_up()
