"""Deterministic query routing before retrieval and LLM calls.

The router is intentionally conservative for this demo. It uses token and
phrase matching rather than substring matching, because short markers such as
``hi`` or ``card`` must not match words like ``shipping`` or ``cardiology``.
"""

from enum import StrEnum
import re


class QueryRoute(StrEnum):
    """High-level route for a user message."""

    SMALLTALK = "smalltalk"
    FACTUAL_IN_DOMAIN = "factual_in_domain"
    OUT_OF_DOMAIN = "out_of_domain"
    ADVERSARIAL = "adversarial"


_WORD_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)

SMALLTALK_TOKENS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thankyou",
    "привет",
    "здравствуй",
    "здравствуйте",
    "спасибо",
    "благодарю",
}

SMALLTALK_PHRASES = {
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "who are you",
    "добрый день",
    "доброе утро",
    "добрый вечер",
    "кто ты",
}

# Exact English support-domain tokens. Exact matching avoids false positives such
# as ``order`` in ``border`` and ``card`` in ``cardiology``.
EN_DOMAIN_TOKENS = {
    "delivery",
    "shipping",
    "pickup",
    "courier",
    "order",
    "orders",
    "payment",
    "payments",
    "pay",
    "card",
    "return",
    "returns",
    "refund",
    "refunds",
    "exchange",
    "warranty",
    "product",
    "products",
    "bonus",
    "bonuses",
    "cashback",
    "discount",
    "discounts",
    "promo",
    "promocode",
    "support",
    "contact",
    "contacts",
    "customer",
    "catalog",
    "catalogue",
    "kettle",
    "kettles",
    "lamp",
    "lamps",
    "textile",
    "textiles",
    "vacuum",
    "hours",
    "schedule",
    "working",
}

# Russian stems are intentionally prefix-matched for common inflections.
RU_DOMAIN_PREFIXES = {
    "достав",
    "самовывоз",
    "курьер",
    "заказ",
    "оплат",
    "сбп",
    "карт",
    "возврат",
    "верн",
    "обмен",
    "гарант",
    "товар",
    "бонус",
    "кэшбэк",
    "кешбэк",
    "скидк",
    "акци",
    "промокод",
    "поддерж",
    "менеджер",
    "контакт",
    "домок",
    "покупател",
    "каталог",
    "чайник",
    "ламп",
    "текстил",
    "пылесос",
    "час",
    "график",
}

ADVERSARIAL_PHRASES = {
    "system prompt",
    "developer message",
    "ignore previous",
    "ignore earlier",
    "ignore all previous",
    "ignore all earlier",
    "reveal prompt",
    "show prompt",
    "print prompt",
    "hidden prompt",
    "system instructions",
    "developer instructions",
    "системный промпт",
    "покажи промпт",
    "раскрой промпт",
    "скрытый промпт",
    "игнорируй предыдущ",
    "игнорируй все предыдущ",
}

OTHER_COMPANY_MARKERS = {
    "пятерочка",
    "пятёрочка",
    "магнит",
    "ozon",
    "wildberries",
    "amazon",
}


def _normalize(text: str) -> str:
    """Normalize text for deterministic matching."""
    return " ".join(text.lower().replace("ё", "е").split())


def _tokens(normalized: str) -> list[str]:
    """Return word-like tokens for exact and prefix matching."""
    return _WORD_RE.findall(normalized)


def _has_phrase(normalized: str, phrases: set[str]) -> bool:
    """Match multi-word phrase markers after normalization."""
    return any(phrase in normalized for phrase in phrases)


def _has_domain_token(tokens: list[str]) -> bool:
    """Return whether tokenized text contains a known support-domain marker."""
    if any(token in EN_DOMAIN_TOKENS for token in tokens):
        return True
    return any(
        token.startswith(prefix)
        for token in tokens
        for prefix in RU_DOMAIN_PREFIXES
    )


def classify_query(text: str) -> QueryRoute:
    """Classify a user message before retrieval.

    Priority matters:
    1. adversarial prompt requests are blocked;
    2. factual support intent wins over greeting/thanks in mixed messages;
    3. pure small-talk is handled without retrieval;
    4. everything else is refused as out-of-domain.
    """
    normalized = _normalize(text)
    if not normalized:
        return QueryRoute.SMALLTALK

    tokens = _tokens(normalized)

    if _has_phrase(normalized, ADVERSARIAL_PHRASES):
        return QueryRoute.ADVERSARIAL

    if any(marker in tokens for marker in OTHER_COMPANY_MARKERS):
        return QueryRoute.OUT_OF_DOMAIN

    if _has_domain_token(tokens):
        return QueryRoute.FACTUAL_IN_DOMAIN

    if _has_phrase(normalized, SMALLTALK_PHRASES) or any(token in SMALLTALK_TOKENS for token in tokens):
        return QueryRoute.SMALLTALK

    return QueryRoute.OUT_OF_DOMAIN
