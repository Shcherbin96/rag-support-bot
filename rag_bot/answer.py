"""Шаг 3 RAG — «ответ».

Берём найденные куски (retrieval) и просим модель ответить ТОЛЬКО по ним.
Тут живёт guardrail: нет ответа в кусках → честно «не знаю», без выдумок.
Модель — через NVIDIA NIM (OpenAI-совместимый API).
Запуск: uv run python -m rag_bot.answer "вопрос"
"""
from openai import OpenAI

from rag_bot import config
from rag_bot.retrieval import retrieve

# Инструкция модели — здесь задаём поведение и guardrail
SYSTEM_PROMPT = (
    "Ты — ассистент поддержки интернет-магазина «ДомОк». "
    "Если пользователь здоровается, благодарит или задаёт разговорный/мета-вопрос "
    "(например, на каком языке ты говоришь, кто ты) — ответь коротко, дружелюбно и естественно, "
    "без отказа и без ссылок на источники. "
    "Если это ФАКТИЧЕСКИЙ вопрос о магазине (доставка, оплата, возврат, гарантия, товары, бонусы и т.п.) — "
    "отвечай ТОЛЬКО на основе предоставленных фрагментов базы знаний и в конце укажи источник в скобках. "
    "Если фактического ответа в фрагментах нет — честно скажи, что не знаешь, и предложи уточнить у менеджера. "
    "Никогда не выдумывай факты, цены, сроки и условия. "
    "Отвечай кратко и НА ТОМ ЖЕ ЯЗЫКЕ, на котором задан вопрос."
)


def _client() -> OpenAI:
    # Стандартный клиент OpenAI, но base_url указывает на провайдера из config (сейчас Gemini)
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def answer(query: str, k: int = config.TOP_K) -> dict:
    """Вернуть ответ бота: {text, sources, chunks}."""
    chunks = retrieve(query, k=k)
    context = "\n\n".join(
        f"[Источник: {c['source']}]\n{c['text']}" for c in chunks
    )
    user_msg = f"Фрагменты базы знаний:\n{context}\n\nВопрос клиента: {query}"

    resp = _client().chat.completions.create(
        model=config.ANSWER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,   # низкая «температура» = меньше фантазии, больше точности
    )
    text = resp.choices[0].message.content.strip()
    sources = sorted({c["source"] for c in chunks})
    return {"text": text, "sources": sources, "chunks": chunks}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "сколько стоит доставка?"
    res = answer(q)
    print(f"❓ {q}\n")
    print(f"🤖 {res['text']}\n")
    print(f"📎 источники: {', '.join(res['sources'])}")
