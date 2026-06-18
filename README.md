# RAG Support Bot

A production-minded **RAG (Retrieval-Augmented Generation) support assistant** for Telegram. It answers customer questions **strictly from a knowledge base**, cites its sources, refuses to invent facts, and replies in the user's language. Ships with an **evaluation harness** that measures answer quality and hallucinations.

Built for the fictional home-goods store **«ДомОк»** as a demo, but the knowledge base is swappable for any business.

> **Live demo:** Telegram — [@ai_demo_assistmoki_bot](https://t.me/ai_demo_assistmoki_bot)

---

## Features

- 🔎 **RAG over your documents** — chunking → embeddings → vector search (Chroma).
- 📎 **Source citations** — every factual answer references the document it came from.
- 🛡️ **Anti-hallucination guardrail** — if the answer isn't in the knowledge base, the bot says so and offers a human, instead of making things up.
- 🌍 **Bilingual** — answers in the language of the question (RU / EN), powered by multilingual embeddings.
- 🔌 **Provider-agnostic** — the LLM is called via an OpenAI-compatible client; swap Gemini / Groq / NVIDIA / OpenAI by changing config only.
- 📊 **Evaluation harness** — a test set + automated metrics (grounded accuracy, guardrail, hallucination count) that **drove the model choice with data, not vibes**.

## Architecture

```
documents (.md)
   │  ingestion: chunk + embed (multilingual)
   ▼
Chroma vector DB
   │  retrieval: top-k chunks by meaning
   ▼
answer agent  ──►  LLM (Gemini, OpenAI-compatible)
   │  prompt: "answer ONLY from these chunks, cite source, refuse if absent"
   ▼
Telegram bot (aiogram, long polling)
```

Each component is an isolated, testable module: `ingestion` · `retrieval` · `answer` · `bot`, plus an `eval` harness.

## Tech stack

Python 3.12 · Chroma + `sentence-transformers` (multilingual embeddings, local & free) · Google Gemini via OpenAI-compatible API · `aiogram` 3 · `pytest` · `uv`.

## Quick start

```bash
# 1. install deps
uv sync

# 2. add keys
cp .env.example .env          # then fill GEMINI_API_KEY and TELEGRAM_BOT_TOKEN

# 3. build the knowledge base index
uv run python -m rag_bot.ingestion

# 4. try a question from the CLI
uv run python -m rag_bot.answer "сколько стоит доставка?"

# 5. run the Telegram bot
uv run python -m rag_bot.bot
```

Get free keys: [Gemini API](https://aistudio.google.com) · Telegram token from [@BotFather](https://t.me/BotFather).

## Evaluation

```bash
PYTHONPATH=. uv run python eval/run_eval.py
```

Runs a 13-case test set (grounded / refuse / small-talk) across models and reports metrics. Results:

| Model | Passed | Hallucinations | Notes |
|---|---|---|---|
| **gemini-2.5-flash-lite** | **13/13** | **0** | ✅ chosen — lightest model, passes everything |
| gemini-2.5-flash / pro | — | — | limited by Gemini free-tier rate limits |

**Takeaway:** the smallest model solves this task with zero hallucinations, so a bigger model isn't needed — the choice is justified by metrics. Key metric for a support bot: **zero hallucinations** on out-of-base questions. See [`eval/results.md`](eval/results.md).

## Project structure

```
rag_bot/
  config.py       # all settings (keys, model, paths) in one place
  ingestion.py    # documents → chunks → embeddings → Chroma
  retrieval.py    # semantic search over the index
  answer.py       # grounded answer + citations + guardrail
  bot.py          # Telegram interface (aiogram)
data/knowledge_base/   # the «ДомОк» documents (swap for your own)
eval/             # test set + evaluation harness
tests/            # pytest: ingestion, retrieval, answer
```

## Notes

Embeddings run locally (free). The LLM uses Google Gemini's free tier — fine for a demo; for production traffic use a paid tier or self-hosted model. Knowledge base is plain Markdown — replace the files in `data/knowledge_base/` and re-run ingestion to point the bot at a different business.
