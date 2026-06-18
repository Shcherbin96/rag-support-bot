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
def test_out_of_base_question_is_refused():
    """Вопрос ВНЕ базы → guardrail: бот честно отказывается, не выдумывает."""
    res = answer("вы продаёте автомобили Tesla?")
    low = res["text"].lower()
    assert any(w in low for w in ["не зна", "не наш", "уточн", "менеджер"])
