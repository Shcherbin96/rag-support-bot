"""Unit tests for the deterministic retrieval relevance guardrail."""

from rag_bot.retrieval import is_relevant


def test_relevant_when_best_distance_is_within_threshold():
    chunks = [{"text": "delivery", "source": "delivery.md", "distance": 0.4}]
    assert is_relevant(chunks, max_distance=1.2)


def test_irrelevant_when_best_distance_exceeds_threshold():
    chunks = [{"text": "returns", "source": "returns.md", "distance": 1.5}]
    assert not is_relevant(chunks, max_distance=1.2)


def test_irrelevant_when_no_chunks_are_returned():
    assert not is_relevant([], max_distance=1.2)
