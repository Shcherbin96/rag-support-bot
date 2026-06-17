"""Единое место для всех настроек проекта.

Сюда складываем ключи и константы, чтобы они не были разбросаны по коду.
Меняешь модель или параметр — меняешь в одном месте.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Подхватываем ключи из файла .env (если он есть)
load_dotenv()

# --- Ключи доступа (берутся из .env, в код не вписываем) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# --- Модели ---
# Claude, который пишет ответ клиенту (haiku — быстрый и дешёвый, для поддержки хватает)
ANSWER_MODEL = "claude-haiku-4-5"
# Модель, которая превращает текст в «числовой отпечаток» (эмбеддинг). Мультиязычная — понимает русский.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Пути (вычисляются от расположения этого файла, чтобы работало на любой машине) ---
ROOT = Path(__file__).resolve().parents[1]   # корень проекта
KB_DIR = ROOT / "data" / "knowledge_base"    # тут лежат документы базы знаний (.md)
CHROMA_DIR = ROOT / "chroma"                 # тут хранится векторная база

# --- Параметры поиска ---
TOP_K = 4   # сколько самых релевантных кусков достаём из базы на каждый вопрос
