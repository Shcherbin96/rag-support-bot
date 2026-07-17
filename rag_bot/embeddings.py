"""Shared, lazily loaded sentence-transformers embedding model.

Centralizing the model here means the router does not load a second copy at
import time, and tests that avoid the semantic path stay fast (the model is
only loaded on the first embed() call).
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_bot import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Return the process-wide embedding model, loaded once on first use."""
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings so cosine similarity is a dot product."""
    return get_model().encode(list(texts), normalize_embeddings=True)
