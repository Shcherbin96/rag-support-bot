"""Build the local Chroma index from Markdown knowledge-base documents."""

import contextlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag_bot import config

COLLECTION = "nestwell"


def _document_title(text: str, fallback: str) -> str:
    """Return the document's H1 heading, or the filename as a fallback."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return fallback


def load_chunks(kb_dir: Path) -> list[dict[str, str]]:
    """Load Markdown files and split them into section-level chunks."""
    chunks: list[dict[str, str]] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _document_title(text, path.name)
        parts = text.split("\n## ")
        for index, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            body = part if index == 0 else "## " + part
            chunks.append({"text": body, "source": path.name, "title": title})
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
    # Missing-collection errors are expected on a clean CI runner. Other Chroma
    # failures are surfaced later when create/add/count fails.
    with contextlib.suppress(Exception):
        client.delete_collection(COLLECTION)

    # Same chromadb SentenceTransformerEmbeddingFunction stub inconsistency as
    # retrieval.py's _collection(); see the comment there.
    collection = client.create_collection(COLLECTION, embedding_function=embed_fn)  # type: ignore[arg-type]
    collection.add(
        ids=[f"chunk-{index}" for index in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"source": chunk["source"], "title": chunk["title"]} for chunk in chunks],
    )
    return collection.count()


if __name__ == "__main__":
    count = build_index()
    print(f"Done: indexed {count} knowledge-base chunks.")
