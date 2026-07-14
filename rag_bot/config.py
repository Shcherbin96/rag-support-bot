"""Central configuration for the application.

Environment variables, model settings, retrieval parameters, and paths live here
so the rest of the codebase can stay provider-agnostic and easy to configure.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Credentials are loaded from the environment and never hard-coded.
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Google Gemini through its OpenAI-compatible endpoint.
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ANSWER_MODEL = "gemini-2.5-flash-lite"
EVAL_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

# Local multilingual embedding model.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "knowledge_base"
CHROMA_DIR = ROOT / "chroma"

# Retrieval settings.
TOP_K = int(os.getenv("TOP_K", "4"))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.2"))
