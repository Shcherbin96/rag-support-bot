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
# Имена провайдер-нейтральные: сменить провайдера = поменять эти 3 строки, код не трогаем.
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# --- Модель для ответа (Google Gemini через OpenAI-совместимый эндпоинт) ---
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ANSWER_MODEL = "gemini-2.5-flash"   # быстрая и бесплатная; для сравнения позже добавим flash-lite/pro

# --- Модель эмбеддингов (локальная, бесплатная) ---
# Превращает текст в «числовой отпечаток». Мультиязычная — понимает русский.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Пути (вычисляются от расположения этого файла, чтобы работало на любой машине) ---
ROOT = Path(__file__).resolve().parents[1]   # корень проекта
KB_DIR = ROOT / "data" / "knowledge_base"    # тут лежат документы базы знаний (.md)
CHROMA_DIR = ROOT / "chroma"                 # тут хранится векторная база

# --- Параметры поиска ---
TOP_K = 4   # сколько самых релевантных кусков достаём из базы на каждый вопрос
