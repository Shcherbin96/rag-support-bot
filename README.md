# RAG Support Bot

![CI](https://github.com/Shcherbin96/rag-support-bot/actions/workflows/ci.yml/badge.svg)

A portfolio-grade **RAG (Retrieval-Augmented Generation) support assistant** for Telegram. It answers customer-support questions from a Markdown knowledge base, uses a deterministic domain router before retrieval, validates model citations against retrieved chunk IDs, and refuses unsupported questions instead of pretending to know.

Built as a demo for a fictional home-goods store, **DomOk**. The business domain is replaceable: swap the Markdown knowledge base, rebuild the index, and recalibrate retrieval/routing behavior for a new company.

> **Live demo:** Telegram — [@ai_demo_assistmoki_bot](https://t.me/ai_demo_assistmoki_bot). Availability is not guaranteed; the bot may be offline during development.

---

## Business use case

Small companies often answer the same support questions manually: delivery, payment, returns, warranty, product conditions, bonuses, and order flow. This project shows how to automate first-line support while keeping the assistant constrained to company-approved documents.

The goal is not to make a chatbot that sounds confident. The goal is to make a support assistant that knows when it does **not** know.

## Key features

- **Domain routing before retrieval** — small-talk, adversarial, out-of-domain, and factual support questions are separated before semantic search.
- **RAG over business documents** — Markdown knowledge base → section chunks → embeddings → Chroma vector search.
- **Retrieval relevance check** — accepted context is filtered by a configurable distance threshold.
- **Structured answer contract** — the LLM must return JSON with `answer` and cited `chunk-id` values.
- **Validated citations** — citations are accepted only if they reference retrieved chunk IDs; user-visible sources are derived from validated citations only.
- **Fail-closed behavior** — invalid JSON, missing citations, invalid citations, missing index, or model errors return a refusal instead of an unsupported answer.
- **Bilingual behavior** — small demo support for Russian and English.
- **Telegram interface** — `aiogram` bot with timeout, concurrency limit, privacy-safer logging, and friendly fallback message.
- **Evaluation harness** — test-set based evaluation for grounded answers, refusals, small-talk, hallucination count, and runtime errors.
- **Docker-ready demo** — includes a Dockerfile for containerized bot runtime.

## Architecture

```text
user message
   │
   ├─► deterministic router
   │      ├─ small-talk → direct response
   │      ├─ adversarial / out-of-domain → refusal
   │      └─ factual support question
   ▼
Chroma retrieval over Markdown knowledge base
   │
   ├─ filter accepted chunks by distance threshold
   ▼
answer agent ──► LLM via OpenAI-compatible Gemini client
   │             output contract: {"answer": "...", "citations": ["chunk-id"]}
   ▼
validate citations against retrieved context
   │
   ├─ invalid / missing citations → refusal
   └─ valid citations → user answer + cited source files
```

Main modules:

```text
rag_bot/
  config.py       # environment variables, model settings, paths, retrieval threshold
  router.py       # deterministic pre-retrieval query routing
  ingestion.py    # knowledge-base documents → chunks → embeddings → Chroma
  retrieval.py    # semantic search + relevance filtering
  answer.py       # structured grounded answer generation + citation validation
  bot.py          # Telegram interface and runtime error boundary

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

## Configuration

| Variable | Required | Default | Purpose |
|---|---:|---:|---|
| `GEMINI_API_KEY` | For LLM calls | empty | Gemini API key used through the OpenAI-compatible client. |
| `TELEGRAM_BOT_TOKEN` | For Telegram bot | empty | Telegram bot token from BotFather. |
| `TOP_K` | No | `4` | Number of retrieved chunks. |
| `RETRIEVAL_MAX_DISTANCE` | No | `1.2` | Maximum accepted chunk distance before that chunk is excluded from context. |

The CI workflow is designed to pass without real secrets. Live LLM tests are skipped when `GEMINI_API_KEY` is absent.

## Evaluation

Run the evaluation harness:

```bash
PYTHONPATH=. uv run python eval/run_eval.py
```

The report is generated from measured results and includes timestamp, commit SHA when available, retrieval threshold, pass counts, hallucination count, and runtime error count. The committed `eval/results.md` is a reproducibility note, not a permanent quality claim.

## Safety design and limitations

The project uses multiple guardrails:

1. **Domain router** rejects obvious out-of-domain and adversarial messages before retrieval.
2. **Retrieval threshold** filters accepted chunks before the LLM call.
3. **Structured output contract** asks the LLM for JSON with cited chunk IDs.
4. **Citation validation** rejects missing or invalid citations.
5. **Runtime fallback** returns a controlled message on model, parsing, retrieval, or bot errors.

Known limitations:

- The router is deterministic and intentionally simple; it should be replaced or augmented for broader production domains.
- Vector-distance thresholding alone is not a reliable domain boundary, which is why routing and citation validation are separate layers.
- The demo knowledge base is small and synthetic.
- The eval set is useful for regression checks but still limited; production use would need a larger labeled set and monitoring.
- The Telegram bot uses long polling rather than webhook deployment.
- The Dockerfile is a demo image, not a hardened production container.

## Tests and CI

Run locally:

```bash
uv sync --dev
uv run python -m rag_bot.ingestion
uv run pytest -q
```

GitHub Actions runs dependency installation, index build, and pytest. Deterministic unit tests cover routing, relevance checks, structured citation validation, and fail-closed behavior. Optional live LLM tests are skipped without `GEMINI_API_KEY`.

## Docker

```bash
docker build -t rag-support-bot .
docker run --env-file .env rag-support-bot
```

The Docker image builds the vector index during image build and starts the Telegram bot at runtime.

## Why this project matters

This repository demonstrates practical AI automation skills:

- building a working RAG pipeline;
- adding deterministic routing before retrieval;
- validating LLM citations instead of trusting plain text;
- separating deterministic safety logic from LLM behavior;
- writing CI-friendly mocked tests for AI workflows;
- documenting trade-offs and limitations clearly.
