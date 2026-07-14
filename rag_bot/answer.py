"""Grounded answer generation for the RAG support assistant."""

from openai import OpenAI

from rag_bot import config
from rag_bot.retrieval import is_relevant, retrieve

SYSTEM_PROMPT = (
    "You are a customer-support assistant for the ДомОк online store. "
    "For greetings, thanks, or conversational meta-questions, reply briefly and naturally without citations. "
    "For factual store questions, answer only from the supplied knowledge-base fragments and cite the source. "
    "If the answer is not present, say that you do not know and offer escalation to a human manager. "
    "Never invent facts, prices, timelines, or policies. Reply in the same language as the user."
)

REFUSAL_TEXT = {
    "ru": "В базе знаний нет надёжного ответа на этот вопрос. Уточните, пожалуйста, у менеджера.",
    "en": "I could not find a reliable answer in the knowledge base. Please check with a human support agent.",
}


def _client() -> OpenAI:
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def _looks_english(text: str) -> bool:
    latin = sum(char.isascii() and char.isalpha() for char in text)
    cyrillic = sum("а" <= char.lower() <= "я" for char in text)
    return latin > cyrillic


def answer(query: str, k: int = config.TOP_K, model: str = config.ANSWER_MODEL) -> dict:
    """Return a grounded answer plus sources and retrieved chunks."""
    chunks = retrieve(query, k=k)

    # Reject low-confidence retrieval before calling the LLM. This complements
    # the prompt guardrail and avoids grounding an answer on unrelated context.
    if not is_relevant(chunks):
        language = "en" if _looks_english(query) else "ru"
        return {"text": REFUSAL_TEXT[language], "sources": [], "chunks": chunks}

    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in chunks
    )
    user_message = f"Knowledge-base fragments:\n{context}\n\nCustomer question: {query}"

    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    sources = sorted({chunk["source"] for chunk in chunks})
    return {"text": text, "sources": sources, "chunks": chunks}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "сколько стоит доставка?"
    result = answer(question)
    print(f"Question: {question}\n")
    print(f"Answer: {result['text']}\n")
    print(f"Sources: {', '.join(result['sources']) or 'none'}")
