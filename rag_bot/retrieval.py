"""Шаг 2 RAG — «поиск».

По вопросу пользователя находим самые близкие по смыслу куски из векторной базы.
Ключи/LLM тут НЕ нужны — это чистый поиск по эмбеддингам.
"""
import chromadb
from chromadb.utils import embedding_functions

from rag_bot import config
from rag_bot.ingestion import COLLECTION


def _collection():
    """Подключаемся к уже построенной базе тем же эмбеддером, что и при ingestion.

    Важно: модель эмбеддингов на поиске должна быть ТА ЖЕ, что при загрузке —
    иначе «отпечатки» вопроса и документов будут несопоставимы.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )
    return client.get_collection(COLLECTION, embedding_function=embed_fn)


def retrieve(query: str, k: int = config.TOP_K) -> list[dict]:
    """Вернуть k самых релевантных кусков: текст, источник и «расстояние».

    Чем меньше distance — тем ближе кусок по смыслу к вопросу.
    """
    coll = _collection()
    res = coll.query(query_texts=[query], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": d, "source": m["source"], "distance": round(dist, 3)}
        for d, m, dist in zip(docs, metas, dists)
    ]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "сколько стоит доставка?"
    print(f"Вопрос: {q}\n")
    for r in retrieve(q):
        print(f"[{r['source']}] distance={r['distance']}")
        print(r["text"][:160].replace("\n", " "), "...\n")
