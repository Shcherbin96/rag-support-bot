"""Тест нарезки на куски: не трогает сеть/модели — проверяет только логику."""
from rag_bot import config
from rag_bot.ingestion import load_chunks


def test_load_chunks_produces_more_than_files():
    chunks = load_chunks(config.KB_DIR)
    n_files = len(list(config.KB_DIR.glob("*.md")))

    assert len(chunks) >= n_files            # хотя бы по куску на файл
    assert all(c["text"] for c in chunks)    # пустых кусков нет
    assert all(c["source"].endswith(".md") for c in chunks)  # источник проставлен
