"""Semantic query routing before retrieval and LLM calls.

The router blocks clearly unsafe or unrelated messages before retrieval, but it is
not intended to answer every domain question by itself. Explicit ecommerce and
support markers reach retrieval, where distance filtering and evidence validation
make the final decision.
"""

from enum import StrEnum
import logging
import os

import numpy as np

from rag_bot import embeddings

log = logging.getLogger("nestwell-router")


class QueryRoute(StrEnum):
    """High-level route for a user message."""

    SMALLTALK = "smalltalk"
    FACTUAL_IN_DOMAIN = "factual_in_domain"
    OUT_OF_DOMAIN = "out_of_domain"
    ADVERSARIAL = "adversarial"


IN_DOMAIN_MIN = float(os.getenv("ROUTER_IN_DOMAIN_MIN", "0.45"))
MARGIN = float(os.getenv("ROUTER_MARGIN", "0.05"))
SMALLTALK_MIN = float(os.getenv("ROUTER_SMALLTALK_MIN", "0.50"))


def _decide(s_small: float, s_in: float, s_out: float) -> QueryRoute:
    """Route from per-route best cosine similarities (fail-closed)."""
    if s_small >= SMALLTALK_MIN and s_small >= s_in and s_small >= s_out:
        return QueryRoute.SMALLTALK
    if s_in >= IN_DOMAIN_MIN and (s_in - s_out) >= MARGIN:
        return QueryRoute.FACTUAL_IN_DOMAIN
    return QueryRoute.OUT_OF_DOMAIN


ADVERSARIAL_PHRASES = {
    "system prompt",
    "developer message",
    "ignore previous",
    "ignore earlier",
    "ignore all previous",
    "ignore all earlier",
    "ignore your instructions",
    "reveal prompt",
    "reveal your prompt",
    "show prompt",
    "print prompt",
    "hidden prompt",
    "system instructions",
    "developer instructions",
}


def _normalize(text: str) -> str:
    """Normalize text for deterministic matching."""
    return " ".join(text.lower().split())


def _has_phrase(normalized: str, phrases: set[str]) -> bool:
    """Match multi-word phrase markers after normalization."""
    return any(phrase in normalized for phrase in phrases)


ANCHORS: dict[QueryRoute, list[str]] = {
    QueryRoute.FACTUAL_IN_DOMAIN: [
        "How much is shipping?",
        "When will my order arrive?",
        "Which payment methods do you accept?",
        "Can I pay with cash at pickup?",
        "Do you offer installment payments?",
        "Where is my receipt?",
        "How do I return an item?",
        "What is your refund policy?",
        "What is the warranty on this product?",
        "How do rewards points work?",
        "Is there a discount for new customers?",
        "How can I contact support?",
        "What are your support hours?",
        "Is this product in stock?",
        "Can I change or cancel my order?",
        "Do you ship internationally?",
        "Can I order a gift?",
    ],
    QueryRoute.OUT_OF_DOMAIN: [
        "What is the weather today?",
        "What is the stock market doing?",
        "Should I buy stocks?",
        "Recommend a movie to watch.",
        "How do I contact the police?",
        "How do I contact emergency services?",
        "Where are government contacts?",
        "Tell me about Amazon.",
        "Tell me about Walmart.",
        "How do I cook pasta?",
        "How do I learn Python?",
        "What is quantum physics?",
        "Where is Paris?",
        "How much is a Tesla?",
        "Can I get investment advice?",
        "Where can I buy concert tickets?",
        "I need medical help.",
    ],
    QueryRoute.SMALLTALK: [
        "Hello",
        "Hi there",
        "Good morning",
        "Thank you",
        "Who are you?",
        "What can you do?",
    ],
}

_anchor_cache: dict[QueryRoute, np.ndarray] | None = None


def _anchor_matrices(embed_fn):
    """Embed anchors once; cache only for the default (shared) embed function."""
    global _anchor_cache
    if embed_fn is embeddings.embed:
        if _anchor_cache is None:
            _anchor_cache = {r: np.asarray(embed_fn(t)) for r, t in ANCHORS.items()}
        return _anchor_cache
    return {r: np.asarray(embed_fn(t)) for r, t in ANCHORS.items()}


def classify_query(text: str, embed_fn=None) -> QueryRoute:
    """Classify a message before retrieval. Fail-closed to OUT_OF_DOMAIN."""
    if embed_fn is None:
        embed_fn = embeddings.embed

    normalized = _normalize(text)
    if not normalized:
        return QueryRoute.SMALLTALK

    if _has_phrase(normalized, ADVERSARIAL_PHRASES):
        return QueryRoute.ADVERSARIAL

    try:
        mats = _anchor_matrices(embed_fn)
        query_vec = np.asarray(embed_fn([normalized]))[0]
        sims = {route: float(np.max(mat @ query_vec)) for route, mat in mats.items()}
    except Exception:
        log.warning("router_embed_failed", exc_info=True)
        return QueryRoute.OUT_OF_DOMAIN

    return _decide(
        sims[QueryRoute.SMALLTALK],
        sims[QueryRoute.FACTUAL_IN_DOMAIN],
        sims[QueryRoute.OUT_OF_DOMAIN],
    )
