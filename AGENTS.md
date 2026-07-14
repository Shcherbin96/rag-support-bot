# Agent Instructions

This repository is a portfolio-grade RAG support assistant demo. Keep changes small, testable, and easy to explain to a hiring manager.

## Project setup

Use Python 3.12 and `uv`.

```bash
uv sync --dev
uv run python -m rag_bot.ingestion
uv run pytest -q
```

## Environment

Optional runtime variables:

- `GEMINI_API_KEY` — required only for LLM-backed answer tests, eval, CLI answers, and Telegram bot runtime.
- `TELEGRAM_BOT_TOKEN` — required only for running the Telegram bot.
- `TOP_K` — number of retrieved chunks; default `4`.
- `RETRIEVAL_MAX_DISTANCE` — maximum accepted top retrieval distance before refusing without an LLM call; default `1.2`.

CI must pass without real API keys. Tests that require live LLM access should be skipped when `GEMINI_API_KEY` is absent.

## Coding rules

- Keep public README and code comments in English.
- Do not hard-code API keys, bot tokens, user data, or real customer data.
- Prefer deterministic business logic before asking the LLM.
- Keep the RAG safety boundary explicit: retrieval relevance check first, then grounded generation, then prompt-level refusal.
- Do not commit generated Chroma DB files, local `.env`, Python caches, or temporary outputs.

## Before finishing a change

Run or document why you could not run:

```bash
uv sync --dev
uv run python -m rag_bot.ingestion
uv run pytest -q
```

For retrieval-safety changes, also manually compare at least one in-knowledge-base query and one out-of-knowledge-base query:

```bash
uv run python -m rag_bot.retrieval "сколько стоит доставка?"
uv run python -m rag_bot.retrieval "сколько стоит Tesla Model 3?"
```

## Portfolio quality bar

A change is ready when a reviewer can understand:

1. what business problem the project solves;
2. how the RAG pipeline works;
3. how hallucination risk is reduced;
4. how to run tests locally;
5. what remains a demo limitation.
