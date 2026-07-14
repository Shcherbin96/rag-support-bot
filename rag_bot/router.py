"""Deterministic query routing before retrieval and LLM calls.

The router is intentionally conservative for this demo. It prevents obviously
out-of-domain and adversarial queries from reaching semantic retrieval, because
raw vector distance alone is not a reliable domain boundary on a small knowledge
base.
"""

from enum import StrEnum


class QueryRoute(StrEnum):
    """High-level route for a user message."""

    SMALLTALK = "smalltalk"
    FACTUAL_IN_DOMAIN = "factual_in_domain"
    OUT_OF_DOMAIN = "out_of_domain"
    ADVERSARIAL = "adversarial"


SMALLTALK_MARKERS = {
    "привет",
    "здравств",
    "добрый день",
    "доброе утро",
    "добрый вечер",
    "спасибо",
    "благодарю",
    "кто ты",
    "hello",
    "hi",
    "thanks",
    "thank you",
    "who are you",
}

DOMAIN_MARKERS = {
    # Russian business-support topics.
    "достав",
    "самовывоз",
    "курьер",
    "заказ",
    "оплат",
    "сбп",
    "карт",
    "возврат",
    "обмен",
    "гарант",
    "товар",
    "бонус",
    "кэшбэк",
    "скидк",
    "акци",
    "промокод",
    "поддержк",
    "менеджер",
    "контакт",
    "магазин",
    "домок",
    "покупател",
    # English equivalents for demo and evaluation.
    "delivery",
    "shipping",
    "pickup",
    "courier",
    "order",
    "payment",
    "card",
    "return",
    "refund",
    "exchange",
    "warranty",
    "product",
    "bonus",
    "cashback",
    "discount",
    "promo",
    "support",
    "contact",
    "store",
    "customer",
}

ADVERSARIAL_MARKERS = {
    "system prompt",
    "developer message",
    "ignore previous",
    "ignore all previous",
    "reveal prompt",
    "show prompt",
    "print prompt",
    "инструкц",
    "системный промпт",
    "покажи промпт",
    "раскрой промпт",
    "игнорируй предыдущ",
}


def classify_query(text: str) -> QueryRoute:
    """Classify a user message before retrieval.

    The goal is not broad natural-language understanding. The goal is a clear,
    deterministic safety boundary for the portfolio demo.
    """
    normalized = " ".join(text.lower().split())
    if not normalized:
        return QueryRoute.SMALLTALK

    if any(marker in normalized for marker in ADVERSARIAL_MARKERS):
        return QueryRoute.ADVERSARIAL

    if any(marker in normalized for marker in SMALLTALK_MARKERS):
        return QueryRoute.SMALLTALK

    if any(marker in normalized for marker in DOMAIN_MARKERS):
        return QueryRoute.FACTUAL_IN_DOMAIN

    return QueryRoute.OUT_OF_DOMAIN
