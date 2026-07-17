"""Model-backed tests for the shared embedding loader."""

import numpy as np

from rag_bot.embeddings import embed, get_model


def test_embed_returns_normalized_vectors():
    vecs = embed(["hello world", "another sentence"])
    assert vecs.shape[0] == 2
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_get_model_is_cached():
    assert get_model() is get_model()
