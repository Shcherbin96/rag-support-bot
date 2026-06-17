"""Шаг 1 RAG — «расставить библиотеку».

Берём документы базы знаний (.md), режем каждый на смысловые куски,
превращаем в эмбеддинги и складываем в векторную базу Chroma.
Запуск: uv run python -m rag_bot.ingestion
"""
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag_bot import config

COLLECTION = "domok"


def load_chunks(kb_dir: Path) -> list[dict]:
    """Читаем все .md и режем на куски по разделам (заголовки '## ').

    Почему режем: класть весь документ одним куском плохо — поиск вернёт
    слишком много лишнего. Маленькие куски по темам ищутся точнее.
    """
    chunks: list[dict] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parts = text.split("\n## ")              # делим по разделам
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            body = part if i == 0 else "## " + part   # возвращаем срезанный заголовок
            chunks.append({"text": body, "source": path.name})
    return chunks


def build_index() -> int:
    """Строим векторную базу из кусков. Возвращаем число кусков."""
    chunks = load_chunks(config.KB_DIR)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )
    # пересоздаём коллекцию начисто, чтобы при повторном запуске не было дублей
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, embedding_function=embed_fn)
    coll.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return coll.count()


if __name__ == "__main__":
    n = build_index()
    print(f"Готово: в индексе {n} кусков из базы знаний.")
