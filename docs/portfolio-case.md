# Portfolio case — RAG Support Bot

A written case study for a technical recruiter or hiring manager. It explains
what the project is, the engineering decisions behind it, and the measured
result — without requiring you to read the code. Every technical claim here is
backed by code, a test, or CI; nothing is inflated.

Repository: <https://github.com/Shcherbin96/rag-support-bot>

---

## 1. Project overview

A Telegram support assistant that answers questions from a company knowledge
base using retrieval-augmented generation (RAG). The whole system is built
around one behavior: **grounded answers with citations, and a fail-closed
refusal when the knowledge base doesn't cover the question.** It never invents a
policy, price, or fact.

## 2. Business problem

Support teams answer the same questions repeatedly — shipping, returns, warranty,
payment, orders. A bot can absorb that load, but a bot that *hallucinates* a
policy is worse than none: it makes wrong promises to customers in the company's
name. The value of this project is not that it answers; it is that it answers
**only when the answer is grounded in retrieved evidence**, and otherwise hands
off to a human.

## 3. My role

Sole engineer. I designed the pipeline and the guardrails, wrote all production
and test code, built the CI and evaluation, containerized it, and wrote the
documentation — as a series of small, reviewed pull requests with green CI at
each step, culminating in a tagged `v1.0.0` release.

## 4. Architecture

```
user message (Telegram)
   → semantic router      (in-domain? small talk? out-of-domain? adversarial?)
   → vector retrieval     (ChromaDB, top-k over the local KB index)
   → LLM                  (any OpenAI-compatible endpoint; JSON mode)
   → citation validation  (quote + number checks against retrieved chunks)
   → grounded answer with sources  —  or a fail-closed refusal
```

The bot runs on `aiogram` with async handling and a small thread pool; the LLM
is behind a provider-agnostic OpenAI-compatible client (Gemini by default,
NVIDIA, or any compatible endpoint via env). Every stage fails closed — an
embedding error or an empty retrieval routes to a safe refusal, never a crash.

## 5. The grounding contract (the core guardrail)

The model is required to return JSON with an answer and citations. On **every**
reply I validate three things before anything reaches the user:

1. **Cited chunk was retrieved** — a citation cannot reference a chunk that
   wasn't in the retrieved context.
2. **Quote is real** — the cited quote must actually appear in that chunk. The
   match is markdown-insensitive (the KB stores facts in `**bold**`, the model
   quotes plain text) but cannot accept a fabricated quote.
3. **Numbers are supported** — every number in the answer must appear in a cited
   quote. This is what stops an invented price, date, or percentage.

If any check fails, the reply is rejected and the bot refuses rather than send
it. This is deliberately *not* full natural-language entailment — I say so —
but it catches the factual hallucinations that actually hurt a support bot.

## 6. Reliability and error handling

- **Typed result contract** — `AnswerError` (`StrEnum`) and `AnswerResult`
  (`TypedDict`), shared between the answer path and the eval harness so their
  error taxonomies can't drift.
- **JSON mode with graceful fallback** — `response_format` is used when
  supported and falls back to prompt-only JSON when a provider rejects it.
- **Fails closed everywhere** — a bounded LLM timeout below the bot's answer
  budget (a stalled provider becomes a retryable error, not a hang); the router
  fails closed on embed errors; the Telegram fallback send is itself guarded so
  a network blip can't crash the handler.
- **Secret & privacy hygiene** — the API key is never logged; user message text
  is SHA-256-fingerprinted before logging, so raw customer messages never land
  in logs.

## 7. Testing strategy

- **78 offline tests** (no API key, no network) plus 2 `live`-marked tests
  deselected by default; **~91% coverage**, gated at 87% on the Linux CI leg.
- `ruff` lint + format and **strict `mypy`** over `rag_bot`.
- The suite covers the router decision rule and adversarial detection, the
  citation and number validation (including the markdown-insensitive quote
  match), fail-closed paths, provider configuration, and the Telegram helper.

## 8. LLM evaluation strategy

- **Offline routing-accuracy gate** runs in CI with no key: it classifies a
  labeled set with the LLM-free semantic router. Adversarial (prompt-injection)
  cases must match exactly; non-adversarial cases are graded against a **≥93%
  accuracy floor**. That floor exists because the 3-OS CI matrix surfaced that
  the embedding backend returns slightly different similarities per platform,
  which can flip a borderline case — so the test grades the classifier's
  accuracy, not the math library, while safety cases stay exact.
- **Live grounded-answer eval** (`eval/run_eval.py`, committed report in
  `eval/results.md`): **29/33 with 0 hallucinations.** The misses are the
  guardrail refusing, not the model fabricating. This eval also *found a real
  bug* — the citation check was rejecting bold-formatted facts — which I then
  fixed.
- **Scheduled weekly workflow** (`evals.yml`) with a **degradation gate**: a
  zero-hallucination invariant plus a generous pass-rate floor. A failing run
  auto-opens a GitHub issue. Continuous monitoring, not a one-off number.

## 9. CI/CD and deployment

- **CI**: ruff lint + format and strict `mypy` on Linux; the full suite on
  **Ubuntu, macOS, and Windows**; a coverage gate on Linux; a Docker build-only
  job. SHA-pinned actions, least-privilege permissions, and Dependabot for
  `github-actions`, `uv`, and `docker`.
- **Container**: a reproducible **~2.9 GB image** (`docker images` reports
  2.86 GB) — digest-pinned `python:3.12-slim` base and `uv`, `--frozen --no-dev`,
  non-root, and CPU-only `torch` on Linux (down from ~19 GB with the default
  CUDA wheel). The vector index is baked in so the container needs no network to
  start serving.
- **Release**: tagged `v1.0.0` with a `CHANGELOG.md`.

## 10. Measured result

- **Correctness:** on the live eval, **0 hallucinations across 33 cases**; 29
  answered, the rest safely refused.
- **Grounding:** every answered number traces to a cited quote from retrieved
  evidence — enforced on every reply, not sampled.
- **Quality bar:** 78 tests, ~91% coverage, strict typing, three operating
  systems, a committed live eval report, and automated weekly monitoring.

## 11. Engineering trade-offs

- **Fail closed over answer more.** I chose precision over recall: the bot
  refuses when evidence is missing rather than guessing. For support, a wrong
  confident answer is more expensive than "let me get a human."
- **Exact-quote + number validation over an LLM judge for grounding.** It is
  deterministic, cheap, and explainable, and it catches the failures that
  matter. A semantic-entailment judge would catch more but adds cost and its own
  failure modes — a future option, gated behind the eval harness.
- **A local embedding model and a local vector store.** No external embedding
  API to depend on or pay for; the trade is a heavier container, which I
  addressed with the CPU-torch image reduction.
- **An accuracy gate, not exact matching, for a fuzzy classifier.** Documented
  above — the correct test shape for a cosine-threshold router, with safety
  cases kept exact.

## 12. Limitations

- Small demo knowledge base; one embedding model; the demo is **English-only**
  (the embedding model is multilingual, but the KB, router anchors, and prompts
  are English — no multilingual behavior is exercised or tested).
- The grounding check is exact-quote + number validation, **not** full
  natural-language entailment.
- Prompt-injection defense is a deterministic gate plus the citation backstop —
  strong, but not a guarantee against every adversarial phrasing.
- Retrieval favors a single topic per query, so a compound question can be
  answered for one part and refused for the other (honest, but a UX limit).
- Human review is expected for high-stakes answers; the bot escalates when unsure.

## 13. Future improvements

- A larger, versioned knowledge base with chunk-size/overlap tuning and a
  re-ingestion pipeline.
- An LLM-as-judge grounding layer, gated behind the eval harness so it never
  becomes an unmonitored claim.
- Multi-topic / query-decomposition retrieval for compound questions.
- Multilingual support (the embedding model already allows it) with per-language
  router anchors and evals.
- Conversation memory and human-handoff integration for a real support desk.
