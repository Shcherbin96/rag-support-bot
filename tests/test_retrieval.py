"""Retrieval tests.

Requires a built index: uv run python -m rag_bot.ingestion
No LLM key is needed.
"""

from rag_bot.retrieval import retrieve


def test_retrieve_finds_shipping_section():
    results = retrieve("how much does it cost to ship my order?", k=3)

    assert len(results) == 3
    sources = [result["source"] for result in results]
    assert any("shipping" in source for source in sources)
    assert all("id" in result for result in results)
    assert all(isinstance(result["distance"], float) for result in results)
