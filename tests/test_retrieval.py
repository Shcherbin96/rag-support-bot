"""Тест поиска: проверяем, что на вопрос про доставку в топ попадает раздел доставки.

Требует построенного индекса (uv run python -m rag_bot.ingestion). LLM-ключ не нужен.
"""
from rag_bot.retrieval import retrieve


def test_retrieve_finds_delivery_section():
    results = retrieve("во сколько обойдётся привезти заказ?", k=3)

    assert len(results) == 3
    sources = [r["source"] for r in results]
    # вопрос без слова «доставка», но по смыслу должен найтись раздел доставки
    assert any("dostavka" in s for s in sources)
