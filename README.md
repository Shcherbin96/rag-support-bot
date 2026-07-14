# RAG Support Bot

![CI](https://github.com/Shcherbin96/rag-support-bot/actions/workflows/ci.yml/badge.svg)

A production-minded **RAG (Retrieval-Augmented Generation) support assistant** for Telegram. It answers customer questions strictly from a knowledge base, cites sources, refuses to invent unsupported facts, and replies in the user's language.

Built as a portfolio demo for a fictional home-goods store, **DomOk**. The business domain is replaceable: swap the Markdown knowledge base, rebuild the index, and the same architecture can support another company.

> **Live demo:** Telegram — [@ai_demo_assistmoki_bot](https://t.me/ai_demo_assistmoki_bot)

---

## Business use case

Small companies often answer the same support questions manually: delivery, payment, returns, warranty, product conditions, and order flow. This project shows how to automate first-line support while keeping the assistant grounded in company-approved documents.

The goal is not to make a chatbot that sounds confident. The goal is to make a support assistant that knows when it does **not** know.

## Key features

- **RAG over business documents** — Markdown knowledge base → chunking → embeddings → Chroma vector search.
- **Source citations** — factual answers cite the document source used for grounding.
- **Deterministic retrieval guardrail** — weak retrieval results are refused before calling the LLM.
- **Prompt-level anti-hallucination guardrail** — the LLM is instructed to answer only from supplied fragments.
- **Bilingual behavior** — replies in Russian or English depending on the user's message.
- **Provider-agnostic LLM client** — Gemini is called through an OpenAI-compatible API; the provider can be changed in config.
- **Evaluation harness** — test-set based evaluation for grounded answers, refusal behavior, small-talk, and hallucination count.
- **Telegram interface** — `aiogram` bot with long polling for a simple live demo.
- **Docker-ready** — includes a Dockerfile for deployment on a VPS or container platform.

## Architecture

```text
documents (.md)
   │  ingestion: split by sections + embed
   ▼
Chroma vector DB
   │  retrieval: top-k chunks by semantic similarity
   │  deterministic relevance check: distance threshold
   ▼
answer agent  ──►  LLM (Gemini via OpenAI-compatible client)
   │  prompt: answer only from retrieved chunks, cite source, refuse if absent
   ▼
Telegram bot (aiogram, long polling)
```

Main modules:

```text
rag_bot/
  config.py       # environment variables, model settings, paths, retrieval threshold
  ingestion.py    # knowledge-base documents → chunks → embeddings → Chroma
  retrieval.py    # semantic search + relevance check
  answer.py       # grounded answer generation + refusal behavior
  bot.py          # Telegram interface

data/knowledge_base/   # demo business documents
eval/                  # evaluation harness and results
tests/                 # pytest tests
```

## Tech stack

Python 3.12 · Chroma · `sentence-transformers` multilingual embeddings · Google Gemini via OpenAI-compatible API · `aiogram` 3 · `pytest` · `uv` · GitHub Actions · Docker.

## Quick start

```bash
# 1. Install dependencies
uv sync --dev

# 2. Add runtime keys
cp .env.example .env
# Fill GEMINI_API_KEY and TELEGRAM_BOT_TOKEN when you want live LLM / Telegram behavior.

# 3. Build the local vector index
uv run python -m rag_bot.ingestion

# 4. Test retrieval without an LLM key
uv run python -m rag_bot.retrieval "сколько стоит доставка?"

# 5. Ask the answer agent from CLI; requires GEMINI_API_KEY
uv run python -m rag_bot.answer "сколько стоит доставка?"

# 6. Run the Telegram bot; requires GEMINI_API_KEY and TELEGRAM_BOT_TOKEN
uv run python -m rag_bot.bot
```

Get keys:

- Gemini API key: Google AI Studio.
- Telegram bot token: Telegram [@BotFather](https://t.me/BotFather).

## Configuration

Environment variables:

| Variable | Required | Default | Purpose |
|---|---:|---:|---|
| `GEMINI_API_KEY` | For LLM calls | empty | Gemini API key used through the OpenAI-compatible client. |
| `TELEGRAM_BOT_TOKEN` | For Telegram bot | empty | Telegram bot token from BotFather. |
| `TOP_K` | No | `4` | Number of retrieved chunks. |
| `RETRIEVAL_MAX_DISTANCE` | No | `1.2` | Maximum accepted top retrieval distance before refusing without an LLM call. |

The CI workflow is designed to pass without real secrets. Tests that require live LLM access are skipped when `GEMINI_API_KEY` is absent.

## Evaluation

Run the evaluation harness:

```bash
PYTHONPATH=. uv run python eval/run_eval.py
```

Current demo result:

| Model | Passed | Hallucinations | Notes |
|---|---:|---:|---|
| `gemini-2.5-flash-lite` | **13/13** | **0** | Chosen as the default model for this task. |
| `gemini-2.5-flash` / `pro` | — | — | Limited by Gemini free-tier rate limits during testing. |

Key takeaway: for this support-bot task, the lightweight model passed the available evaluation set with zero hallucinations. Model choice is based on measured task behavior, not model size.

## Retrieval safety

The assistant has two safety layers:

1. **Retrieval relevance check** — if the best retrieved chunk is too far from the query, the app refuses before calling the LLM.
2. **Grounded generation prompt** — if retrieval is accepted, the LLM receives only selected chunks and is instructed to answer only from them.

The current threshold is intentionally simple and transparent for a portfolio demo. See [`docs/retrieval-calibration.md`](docs/retrieval-calibration.md) for the calibration protocol and production notes.

## Tests and CI

Run locally:

```bash
uv sync --dev
uv run python -m rag_bot.ingestion
uv run pytest -q
```

GitHub Actions runs the same core checks: dependency installation, index build, and pytest.

## Docker

```bash
docker build -t rag-support-bot .
docker run --env-file .env rag-support-bot
```

The Docker image builds the vector index during image build and starts the Telegram bot at runtime.

## Demo limitations

This is a portfolio demo, not a production SaaS deployment. Known limitations:

- The default retrieval threshold needs calibration against a larger real query set before production use.
- The demo uses long polling for Telegram instead of webhook deployment.
- The knowledge base is small and synthetic.
- The current eval set is useful but still small; production use would require larger regression tests and monitoring.
- The LLM uses a free-tier API during development; production traffic needs paid quota or another provider.

## Why this project matters

This repository demonstrates practical AI automation skills:

- building a working RAG pipeline;
- designing guardrails against unsupported answers;
- evaluating model behavior with a test set;
- separating deterministic logic from LLM behavior;
- shipping a runnable Telegram demo;
- documenting trade-offs and limitations clearly.
