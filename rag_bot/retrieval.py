"""Semantic retrieval over the local Chroma knowledge-base index."""

import chromadb
from chromadb.utils import embedding_functions

from rag_bot import config
from rag_bot.ingestion import COLLECTION


def _collection():
    """Open the persisted collection with the same embedding model used at ingestion."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )
    return client.get_collection(COLLECTION, embedding_function=embed_fn)


def retrieve(query: str, k: int = config.TOP_K) -> list[dict]:
    """Return the top-k chunks with source metadata and vector distance."""
    coll = _collection()
    res = coll.query(query_texts=[query], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": doc, "source": meta["source"], "distance": round(distance, 3)}
        for doc, meta, distance in zip(docs, metas, dists)
    ]


def is_relevant(chunks: list[dict], max_distance: float = config.RETRIEVAL_MAX_DISTANCE) -> bool:
    """Return whether the best retrieved chunk is relevant enough to call the LLM."""
    return bool(chunks) and chunks[0]["distance"] <= max_distance


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "сколько стоит доставка?"
    print(f"Question: {question}\n")
    for result in retrieve(question):
        print(f"[{result['source']}] distance={result['distance']}")
        print(result["text"][:160].replace("\n", " "), "...\n")
