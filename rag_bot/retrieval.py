"""Semantic retrieval over the local Chroma knowledge-base index."""

import chromadb
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions

from rag_bot import config
from rag_bot.ingestion import COLLECTION


class KnowledgeBaseNotReadyError(RuntimeError):
    """Raised when retrieval is attempted before the Chroma index exists."""


def _collection():
    """Open the persisted collection with the same embedding model used at ingestion."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )
    try:
        return client.get_collection(COLLECTION, embedding_function=embed_fn)
    except (ValueError, ChromaError) as exc:
        raise KnowledgeBaseNotReadyError(
            "Knowledge-base index is not ready. Run: uv run python -m rag_bot.ingestion"
        ) from exc


def retrieve(query: str, k: int = config.TOP_K) -> list[dict]:
    """Return the top-k chunks with IDs, source metadata, and vector distance."""
    coll = _collection()
    result = coll.query(query_texts=[query], n_results=k)
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "id": chunk_id,
            "text": doc,
            "source": meta["source"],
            "distance": distance,
        }
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances)
    ]


def is_relevant(chunks: list[dict], max_distance: float = config.RETRIEVAL_MAX_DISTANCE) -> bool:
    """Return whether the best retrieved chunk is relevant enough to call the LLM."""
    return bool(chunks) and chunks[0]["distance"] <= max_distance


def accepted_chunks(
    chunks: list[dict], max_distance: float = config.RETRIEVAL_MAX_DISTANCE
) -> list[dict]:
    """Return only chunks that pass the relevance threshold."""
    return [chunk for chunk in chunks if chunk["distance"] <= max_distance]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How much is shipping?"
    print(f"Question: {question}\n")
    for item in retrieve(question):
        print(f"[{item['id']}] {item['source']} distance={item['distance']:.4f}")
        print(item["text"][:160].replace("\n", " "), "...\n")
