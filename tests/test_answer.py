"""Тесты ответа. Требуют NVIDIA_API_KEY (ходят в модель) — без ключа пропускаются."""
import pytest

from rag_bot import config
from rag_bot.answer import answer

needs_key = pytest.mark.skipif(
    not config.LLM_API_KEY, reason="нужен GEMINI_API_KEY в .env"
)


@needs_key
def test_grounded_answer_has_correct_source():
    """Вопрос из базы → ответ ссылается на нужный раздел."""
    res = answer("сколько стоит доставка?")
    assert res["sources"]
    assert any("dostavka" in s for s in res["sources"])


@needs_key
def test_unknown_fact_is_refused():
    """Факт, которого НЕТ в базе → guardrail: честный отказ, без выдумки.

    Важно: «out-of-scope» (Tesla) бот может корректно отвергнуть по рамке магазина.
    Здесь проверяем «out-of-base» — конкретный факт, которого в базе нет.
    """
    res = answer("во сколько закрывается ваш склад в Казани?")
    low = res["text"].lower()
    assert any(w in low for w in ["не зна", "нет информ", "не наш", "уточн", "менеджер"])
