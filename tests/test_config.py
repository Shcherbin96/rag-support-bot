"""Unit tests for provider-specific application configuration."""

import importlib

import rag_bot.config as config_module


def _reload_config(monkeypatch, **env):
    keys = {
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "NVIDIA_MODEL",
        "LLM_BASE_URL",
        "ANSWER_MODEL",
        "EVAL_MODELS",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config_module)


def test_defaults_to_gemini_provider(monkeypatch):
    config = _reload_config(monkeypatch, GEMINI_API_KEY="gemini-key")

    assert config.LLM_PROVIDER == "gemini"
    assert config.LLM_API_KEY == "gemini-key"
    assert config.LLM_API_KEY_ENV == "GEMINI_API_KEY"
    assert config.ANSWER_MODEL == "gemini-2.5-flash-lite"
    assert config.LLM_BASE_URL.endswith("/openai/")


def test_can_select_nvidia_provider(monkeypatch):
    config = _reload_config(
        monkeypatch,
        LLM_PROVIDER="nvidia",
        NVIDIA_API_KEY="nvidia-key",
        NVIDIA_MODEL="nvidia/test-model",
    )

    assert config.LLM_PROVIDER == "nvidia"
    assert config.LLM_API_KEY == "nvidia-key"
    assert config.LLM_API_KEY_ENV == "NVIDIA_API_KEY"
    assert config.ANSWER_MODEL == "nvidia/test-model"
    assert config.LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"


def test_answer_model_override_applies_to_selected_provider(monkeypatch):
    config = _reload_config(
        monkeypatch,
        LLM_PROVIDER="nvidia",
        NVIDIA_API_KEY="nvidia-key",
        NVIDIA_MODEL="nvidia/default-model",
        ANSWER_MODEL="nvidia/override-model",
        EVAL_MODELS="nvidia/override-model,nvidia/second-model",
    )

    assert config.ANSWER_MODEL == "nvidia/override-model"
    assert config.EVAL_MODELS == ["nvidia/override-model", "nvidia/second-model"]
