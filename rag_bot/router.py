"""Deterministic query routing before retrieval and LLM calls.

The router is intentionally conservative about clearly unsafe or unrelated
messages, but it is no longer a tiny keyword whitelist. Natural support-like
questions should reach retrieval, where the distance threshold and evidence
validation make the final decision. This reduces false refusals for paraphrases
that are present in the knowledge base.
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

# Question/action markers that make an otherwise unknown phrase worth sending to
# retrieval. Hard negatives are checked first, so unrelated questions are still
# blocked before vector search.
EN_SUPPORT_LIKE_TOKENS = {
    "can",
    "could",
    "do",
    "does",
    "how",
    "where",
    "when",
    "what",
    "which",
    "is",
    "are",
}

RU_SUPPORT_LIKE_TOKENS = {
    "можно",
    "могу",
    "можете",
    "есть",
    "ли",
    "как",
    "где",
    "когда",
    "какой",
    "какая",
    "какие",
    "какое",
    "сколько",
    "куда",
    "что",
    "купить",
    "связаться",
    "работаете",
    "работает",
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

HARD_NEGATIVE_TOKENS = {
    "tesla",
    "weather",
    "cardiology",
    "border",
    "stock",
    "crypto",
    "bitcoin",
    "medicine",
    "doctor",
    "diagnosis",
    "football",
    "soccer",
    "recipe",
    "rain",
    "дождь",
    "дождя",
    "погода",
    "погоде",
    "тесла",
    "кардиология",
    "медицина",
    "врач",
    "диагноз",
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


def _has_domain_marker(tokens: list[str]) -> bool:
    """Return whether tokenized text contains a known support-domain marker."""
    if any(token in EN_DOMAIN_TOKENS for token in tokens):
        return True
    if any(token in RU_DOMAIN_TOKENS for token in tokens):
        return True
    return _has_prefix(tokens, RU_DOMAIN_PREFIXES)


def _looks_support_like(tokens: list[str]) -> bool:
    """Detect natural support-like questions and requests.

    This is intentionally broader than the domain marker list. Retrieval and
    evidence validation remain responsible for refusing unsupported facts.
    """
    return bool(
        set(tokens) & EN_SUPPORT_LIKE_TOKENS
        or set(tokens) & RU_SUPPORT_LIKE_TOKENS
    )


def _has_hard_negative(tokens: list[str]) -> bool:
    """Detect clearly unrelated domains that should not reach retrieval."""
    return bool(
        set(tokens) & HARD_NEGATIVE_TOKENS
        or set(tokens) & OTHER_COMPANY_MARKERS
        or _has_prefix(tokens, HARD_NEGATIVE_PREFIXES)
    )


def classify_query(text: str) -> QueryRoute:
    """Classify a user message before retrieval.

    Priority matters:
    1. adversarial prompt requests are blocked;
    2. clear unrelated domains and other companies are blocked;
    3. explicit domain markers and support-like questions reach retrieval;
    4. pure small-talk is handled without retrieval;
    5. non-question unknown chatter is refused.
    """
    normalized = _normalize(text)
    if not normalized:
        return QueryRoute.SMALLTALK

    tokens = _tokens(normalized)

    if _has_phrase(normalized, ADVERSARIAL_PHRASES):
        return QueryRoute.ADVERSARIAL

    if _has_hard_negative(tokens):
        return QueryRoute.OUT_OF_DOMAIN

    if _has_domain_marker(tokens) or _looks_support_like(tokens):
        return QueryRoute.FACTUAL_IN_DOMAIN

    if _has_phrase(normalized, SMALLTALK_PHRASES) or any(token in SMALLTALK_TOKENS for token in tokens):
        return QueryRoute.SMALLTALK

    return QueryRoute.OUT_OF_DOMAIN
