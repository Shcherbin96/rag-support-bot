"""Chunking test: touches no network or models — checks only the splitting logic."""

from rag_bot import config
from rag_bot.ingestion import load_chunks


def test_load_chunks_produces_more_than_files():
    chunks = load_chunks(config.KB_DIR)
    n_files = len(list(config.KB_DIR.glob("*.md")))

    assert len(chunks) >= n_files  # at least one chunk per file
    assert all(c["text"] for c in chunks)  # no empty chunks
    assert all(c["source"].endswith(".md") for c in chunks)  # source is set
    assert all(c["title"] and not c["title"].startswith("#") for c in chunks)  # H1 title extracted
