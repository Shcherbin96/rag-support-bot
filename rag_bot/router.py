"""Deterministic query routing before retrieval and LLM calls.

The router blocks clearly unsafe or unrelated messages before retrieval, but it is
not intended to answer every domain question by itself. Explicit ecommerce and
support markers reach retrieval, where distance filtering and evidence validation
make the final decision.
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

# Phrase-level support intents that do not necessarily contain an exact domain token.
# Keep this list narrow: broad verbs like "reach" or "touch" are intentionally not
# added as standalone tokens.
DOMAIN_PHRASES = {
    "reach you",
    "reach support",
    "get in touch",
    "contact support",
    "contact you",
}

# Exact English support-domain tokens. Exact matching avoids false positives such
# as ``order`` in ``border`` and ``card`` in ``cardiology``.
EN_DOMAIN_TOKENS = {
    "delivery",
    "deliver",
    "shipping",
    "ship",
    "pickup",
    "courier",
    "order",
    "orders",
    "payment",
    "payments",
    "pay",
    "card",
    "cash",
    "installment",
    "installments",
    "receipt",
    "invoice",
    "return",
    "returns",
    "refund",
    "refunds",
    "exchange",
    "warranty",
    "product",
    "products",
    "availability",
    "available",
    "stock",
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
    "phone",
    "email",
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
    "open",
    "address",
    "gift",
    "business",
    "company",
}

# Exact Russian support-domain tokens that are too risky for prefix matching.
RU_DOMAIN_TOKENS = {
    "наличными",
    "наличные",
    "рассрочка",
    "рассрочку",
    "рассрочки",
    "долями",
    "чек",
    "чеки",
    "электронный",
    "электронного",
    "телефон",
    "почта",
    "email",
    "телеграм",
    "сдэк",
    "пвз",
    "юрлицами",
    "юрлиц",
    "счет",
    "счета",
    "счету",
    "документы",
    "подарок",
    "подарочную",
    "упаковку",
    "адрес",
    "адреса",
    "акция",
    "акции",
    "акцию",
    "карта",
    "картой",
    "карты",
    "карту",
    "сбп",
    "часы",
    "часов",
}

# Safer Russian stems for common inflections. Do not add very short or ambiguous
# prefixes here: they can create false accepts such as ``час`` → ``часто``.
RU_DOMAIN_PREFIXES = {
    "достав",
    "самовывоз",
    "курьер",
    "заказ",
    "оплат",
    "возврат",
    "вернуть",
    "обмен",
    "гарант",
    "товар",
    "бонус",
    "кэшбэк",
    "кешбэк",
    "скидк",
    "промокод",
    "поддерж",
    "менеджер",
    "контакт",
    "связ",
    "домок",
    "покупател",
    "каталог",
    "чайник",
    "ламп",
    "текстил",
    "пылесос",
    "работа",
    "рабоч",
    "изменить",
    "отменить",
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
    "пятерочке",
    "магнит",
    "ozon",
    "wildberries",
    "amazon",
}

# Multi-word unrelated domains. These are checked before positive domain markers
# so finance uses of "stock" are blocked while ecommerce availability remains in-domain.
HARD_NEGATIVE_PHRASES = {
    "stock market",
    "stock price",
    "stock prices",
    "buy stocks",
    "sell stocks",
    "learn python",
    "quantum physics",
    "recommend a movie",
    "contact the police",
    "contact police",
    "police contact",
    "police contacts",
    "emergency services",
    "government contacts",
    "contact government",
    "contact the government",
    "где купить билеты",
    "совет по инвестициям",
    "следующий матч",
    "приготовить борщ",
    "посмотреть в кино",
}

HARD_NEGATIVE_TOKENS = {
    "tesla",
    "weather",
    "cardiology",
    "border",
    "stocks",
    "crypto",
    "bitcoin",
    "medicine",
    "doctor",
    "diagnosis",
    "police",
    "emergency",
    "government",
    "football",
    "soccer",
    "recipe",
    "rain",
    "python",
    "quantum",
    "physics",
    "paris",
    "movie",
    "concert",
    "tickets",
    "match",
    "дождь",
    "дождя",
    "погода",
    "погоде",
    "тесла",
    "кардиология",
    "медицина",
    "врач",
    "диагноз",
    "полиция",
    "полицию",
    "экстренные",
    "госуслуги",
    "правительство",
    "граница",
    "границу",
    "акционерное",
    "акционерный",
    "общество",
    "картография",
    "картографию",
    "верное",
    "верный",
    "верная",
    "борщ",
    "кино",
    "инвестициям",
    "инвестиции",
    "билеты",
    "концерт",
    "матч",
}

HARD_NEGATIVE_PREFIXES = {
    "дожд",
    "погод",
    "акционер",
    "картограф",
    "медицин",
    "диагноз",
    "крипт",
    "биткоин",
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


def _has_prefix(tokens: list[str], prefixes: set[str]) -> bool:
    """Return whether any token starts with one of the provided prefixes."""
    return any(token.startswith(prefix) for token in tokens for prefix in prefixes)


def _has_domain_marker(normalized: str, tokens: list[str]) -> bool:
    """Return whether tokenized text contains a known support-domain marker."""
    if _has_phrase(normalized, DOMAIN_PHRASES):
        return True
    if any(token in EN_DOMAIN_TOKENS for token in tokens):
        return True
    if any(token in RU_DOMAIN_TOKENS for token in tokens):
        return True
    return _has_prefix(tokens, RU_DOMAIN_PREFIXES)


def _has_hard_negative(normalized: str, tokens: list[str]) -> bool:
    """Detect clearly unrelated domains that should not reach retrieval."""
    return bool(
        _has_phrase(normalized, HARD_NEGATIVE_PHRASES)
        or set(tokens) & HARD_NEGATIVE_TOKENS
        or set(tokens) & OTHER_COMPANY_MARKERS
        or _has_prefix(tokens, HARD_NEGATIVE_PREFIXES)
    )


def classify_query(text: str) -> QueryRoute:
    """Classify a user message before retrieval.

    Priority matters:
    1. adversarial prompt requests are blocked;
    2. clear unrelated domains and other companies are blocked;
    3. explicit ecommerce/support markers reach retrieval;
    4. pure small-talk is handled without retrieval;
    5. generic unknown questions and chatter are refused.
    """
    normalized = _normalize(text)
    if not normalized:
        return QueryRoute.SMALLTALK

    tokens = _tokens(normalized)

    if _has_phrase(normalized, ADVERSARIAL_PHRASES):
        return QueryRoute.ADVERSARIAL

    if _has_hard_negative(normalized, tokens):
        return QueryRoute.OUT_OF_DOMAIN

    if _has_domain_marker(normalized, tokens):
        return QueryRoute.FACTUAL_IN_DOMAIN

    if _has_phrase(normalized, SMALLTALK_PHRASES) or any(token in SMALLTALK_TOKENS for token in tokens):
        return QueryRoute.SMALLTALK

    return QueryRoute.OUT_OF_DOMAIN
