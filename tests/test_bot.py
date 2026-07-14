"""Unit tests for Telegram bot helper behavior."""

import pytest

import rag_bot.bot as bot_module


def test_split_for_telegram_keeps_short_message():
    assert bot_module._split_for_telegram("hello") == ["hello"]


def test_split_for_telegram_splits_long_message():
    text = "x" * (bot_module.TELEGRAM_LIMIT + 10)

    parts = bot_module._split_for_telegram(text)

    assert len(parts) == 2
    assert "".join(parts) == text
    assert all(len(part) <= bot_module.TELEGRAM_LIMIT for part in parts)


def test_validate_runtime_config_reports_missing_values(monkeypatch):
    monkeypatch.setattr(bot_module.config, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(bot_module.config, "LLM_API_KEY", "")

    with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN, GEMINI_API_KEY"):
        bot_module._validate_runtime_config()


def test_validate_runtime_config_accepts_required_values(monkeypatch):
    monkeypatch.setattr(bot_module.config, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(bot_module.config, "LLM_API_KEY", "key")

    bot_module._validate_runtime_config()
