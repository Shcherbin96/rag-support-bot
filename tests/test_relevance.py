"""Unit tests for the deterministic retrieval relevance guardrail.

Coverage lives on accepted_chunks(), the function the answer pipeline actually
calls, instead of duplicating the same threshold check in a separate
relevance helper that nothing in the pipeline invoked.
"""

from rag_bot.retrieval import accepted_chunks


def test_relevant_when_best_distance_is_within_threshold():
    chunks = [{"text": "delivery", "source": "delivery.md", "distance": 0.4}]
    assert bool(accepted_chunks(chunks, max_distance=1.2))


def test_irrelevant_when_best_distance_exceeds_threshold():
    chunks = [{"text": "returns", "source": "returns.md", "distance": 1.5}]
    assert not accepted_chunks(chunks, max_distance=1.2)


def test_irrelevant_when_no_chunks_are_returned():
    assert not accepted_chunks([], max_distance=1.2)
