"""Build the local Chroma index from Markdown knowledge-base documents."""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag_bot import config

COLLECTION = "domok"


def load_chunks(kb_dir: Path) -> list[dict]:
    """Load Markdown files and split them into section-level chunks."""
    chunks: list[dict] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        for index, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            body = part if index == 0 else "## " + part
            chunks.append({"text": body, "source": path.name})
    return chunks


def build_index() -> int:
    """Rebuild the Chroma collection from the current knowledge base."""
    chunks = load_chunks(config.KB_DIR)
    if not chunks:
        raise RuntimeError(f"No Markdown chunks found in knowledge base: {config.KB_DIR}")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )

    # Demo-friendly rebuild: recreate the collection to avoid duplicate chunks.
    # A production service should build a versioned temporary collection and switch
    # only after validation.
    try:
        client.delete_collection(COLLECTION)
    except ValueError:
        pass

    collection = client.create_collection(COLLECTION, embedding_function=embed_fn)
    collection.add(
        ids=[f"chunk-{index}" for index in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
    )
    return collection.count()


if __name__ == "__main__":
    count = build_index()
    print(f"Done: indexed {count} knowledge-base chunks.")
