"""Semantic retrieval over the local Chroma knowledge-base index."""

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions

from rag_bot import config
from rag_bot.ingestion import COLLECTION

# Shape of a retrieved knowledge-base chunk: id, text, source, title, distance.
Chunk = dict[str, Any]


class KnowledgeBaseNotReadyError(RuntimeError):
    """Raised when retrieval is attempted before the Chroma index exists."""


def _collection() -> Collection:
    """Open the persisted collection with the same embedding model used at ingestion."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )
    try:
        # chromadb's SentenceTransformerEmbeddingFunction narrows __call__ to
        # list[str], while ClientAPI.get_collection's EmbeddingFunction protocol
        # is typed for list[str] | list[ndarray]. This is a chromadb stub
        # inconsistency (see chromadb.utils.embedding_functions), not a bug here.
        return client.get_collection(COLLECTION, embedding_function=embed_fn)  # type: ignore[arg-type]
    except (ValueError, ChromaError) as exc:
        raise KnowledgeBaseNotReadyError(
            "Knowledge-base index is not ready. Run: uv run python -m rag_bot.ingestion"
        ) from exc


def retrieve(query: str, k: int = config.TOP_K) -> list[Chunk]:
    """Return the top-k chunks with IDs, source metadata, and vector distance."""
    coll = _collection()
    result = coll.query(query_texts=[query], n_results=k)
    # documents/metadatas/distances are only None when the caller restricts
    # `include=`; we don't, so Chroma always returns all four in lockstep.
    assert result["documents"] is not None
    assert result["metadatas"] is not None
    assert result["distances"] is not None
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "id": chunk_id,
            "text": doc,
            "source": meta["source"],
            "title": meta.get("title", meta["source"]),
            "distance": distance,
        }
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=True)
    ]


def accepted_chunks(
    chunks: list[Chunk], max_distance: float = config.RETRIEVAL_MAX_DISTANCE
) -> list[Chunk]:
    """Return only chunks that pass the relevance threshold."""
    return [chunk for chunk in chunks if chunk["distance"] <= max_distance]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How much is shipping?"
    print(f"Question: {question}\n")
    for item in retrieve(question):
        print(f"[{item['id']}] {item['source']} distance={item['distance']:.4f}")
        print(item["text"][:160].replace("\n", " "), "...\n")
