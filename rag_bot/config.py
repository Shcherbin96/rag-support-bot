"""Central configuration for the application.

Environment variables, model settings, retrieval parameters, and paths live here
so the rest of the codebase can stay provider-agnostic and easy to configure.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_LLM_PROVIDERS = {"gemini", "nvidia"}

# Credentials are loaded from the environment and never hard-coded.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Provider endpoints. Both Gemini and NVIDIA NIM are used through
# OpenAI-compatible chat-completions clients.
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Default models can be overridden from .env. For NVIDIA, set NVIDIA_MODEL to a
# model ID copied from NVIDIA Build, for example: nvidia/llama-3.1-nemotron-nano-8b-v1.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")

if LLM_PROVIDER == "nvidia":
    LLM_API_KEY = NVIDIA_API_KEY
    LLM_API_KEY_ENV = "NVIDIA_API_KEY"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", NVIDIA_BASE_URL)
    ANSWER_MODEL = os.getenv("ANSWER_MODEL", NVIDIA_MODEL)
elif LLM_PROVIDER == "gemini":
    LLM_API_KEY = GEMINI_API_KEY
    LLM_API_KEY_ENV = "GEMINI_API_KEY"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", GEMINI_BASE_URL)
    ANSWER_MODEL = os.getenv("ANSWER_MODEL", GEMINI_MODEL)
else:
    LLM_API_KEY = ""
    LLM_API_KEY_ENV = f"LLM_PROVIDER={LLM_PROVIDER}"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
    ANSWER_MODEL = os.getenv("ANSWER_MODEL", "")


def _split_models(value: str) -> list[str]:
    """Parse comma-separated model names from environment variables."""
    return [model.strip() for model in value.split(",") if model.strip()]


_default_eval_models = "gemini-2.5-flash-lite,gemini-2.5-flash" if LLM_PROVIDER == "gemini" else ANSWER_MODEL
EVAL_MODELS = _split_models(os.getenv("EVAL_MODELS", _default_eval_models))

# Local multilingual embedding model.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "knowledge_base"
CHROMA_DIR = ROOT / "chroma"

# Retrieval settings.
TOP_K = int(os.getenv("TOP_K", "4"))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.2"))

# LLM request timeout in seconds. Kept below the bot's 45s answer wait so a stalled
# provider fails closed with a retryable provider_error instead of hanging the call.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

# Ask the provider to constrain output to valid JSON via response_format. Default
# on since Gemini's OpenAI-compatible endpoint supports it; answer.py falls back
# to prompt-only JSON for providers/models that reject the parameter.
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
