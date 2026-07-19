# Résumé descriptions — RAG Support Bot

Two ready-to-paste variants. All claims are backed by the repository, its tests,
and CI.

---

## Short version (2–3 lines, for a "Projects" section)

> **RAG Support Bot** — a Telegram support assistant that answers from a company
> knowledge base with mandatory citations and a **fail-closed anti-hallucination
> guardrail**: every answer must quote retrieved evidence and every number must
> appear in a cited quote, or the bot refuses. On the live eval: **29/33 with 0
> hallucinations.** 78 tests, ~91% coverage, cross-platform CI, Docker, and a
> weekly automated eval. *Python, ChromaDB, sentence-transformers, aiogram,
> OpenAI-compatible LLM API, Docker, GitHub Actions.*

---

## Extended version (4–6 bullets)

**RAG Support Bot** — grounded Telegram support assistant with anti-hallucination
guardrails (personal project) · *Python, ChromaDB, sentence-transformers,
aiogram, OpenAI-compatible LLM API, Docker, GitHub Actions*

- Built a retrieval-augmented pipeline — **semantic router → vector retrieval →
  citation-bound LLM** — that **fails closed**: every answer must quote a
  retrieved chunk and every number must appear in a cited quote, else the bot
  refuses instead of inventing. **Live eval: 29/33 with 0 hallucinations.**
- Designed a **typed answer contract** (`StrEnum` + `TypedDict`), **JSON mode**
  with a graceful provider fallback, and fail-closed handling throughout
  (bounded timeouts, guarded Telegram sends, privacy-preserving fingerprint logs).
- Wrote **78 offline tests (~91% coverage, strict mypy)** and cross-platform CI
  on **Ubuntu, macOS, and Windows** — the matrix surfaced embedding-backend
  nondeterminism in the router, which I turned into a robust **accuracy gate**
  while keeping adversarial/safety cases exact.
- Implemented an **automated LLM evaluation** pipeline (offline routing-accuracy
  gate in CI + a committed live report) that **caught a real citation bug**, plus
  a **weekly scheduled eval with a zero-hallucination gate that auto-opens a
  GitHub issue** on regression.
- Packaged as a reproducible **Docker image (~2.9 GB)** — non-root, digest-pinned,
  index baked in, CPU-only `torch` on Linux (down from ~19 GB) — and released as
  **v1.0.0**.

---

### Notes for tailoring

- For **AI Automation Specialist / AI Integration Engineer**: lead with the
  provider-agnostic OpenAI-compatible integration and the automated weekly evals.
- For **AI Systems Builder / AI Implementation Engineer**: lead with the RAG
  pipeline, the fail-closed grounding contract, and the eval + CI discipline.
- Keep "**0 hallucinations** across the eval set" and "**fails closed**" — they
  are the strongest, most memorable, and fully true claims for this project.
