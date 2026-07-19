# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-07-18

First stable release. The retrieval pipeline, guardrails, tests, CI, evals, and
container image are complete and verified end-to-end against a live model.
No behavioural changes are planned for the 1.x line beyond fixes.

### What it is
- An English-language RAG support assistant for Telegram: a semantic router →
  vector retrieval over a local knowledge base → an LLM bound by a strict
  citation contract → **fail-closed refusals**. If the answer isn't grounded in
  retrieved context, the bot refuses and offers a human rather than inventing.

### Anti-hallucination guardrails
- **Strict citation contract**: every answer must quote a retrieved chunk, and
  every number in the answer must appear in a cited quote — otherwise the reply
  is rejected. Quote-matching is markdown-insensitive without letting a
  fabricated quote through.
- **Fail-closed by design**: on the live eval, **29/33 with 0 hallucinations** —
  the misses are the guardrail refusing, not the model fabricating.
- **Semantic router** with an exact, deterministic adversarial (prompt-injection)
  gate and cosine-similarity routing for everything else; fails closed on embed
  errors.

### Reliability & contracts
- Typed result contract: `AnswerError` `StrEnum` + `AnswerResult` `TypedDict`,
  shared between the answer path and the eval harness.
- **JSON mode** (default on) with graceful fallback for providers that reject
  `response_format`.
- Bounded LLM timeout (below the bot's answer budget) so a stalled provider fails
  closed; guarded Telegram fallback send; distinct logging for context-empty
  refusals; user text SHA-256-fingerprinted in logs (no raw messages, no keys).
- Provider-agnostic: any OpenAI-compatible endpoint (Gemini default, NVIDIA,
  OpenAI-compatible) via env configuration.

### Testing & CI
- **78 offline tests** (no API key, no network) plus 2 `live`-marked tests
  deselected by default; **~91% coverage** with an 87% gate on the Linux leg.
- `ruff` lint + format and **strict `mypy`** over `rag_bot`.
- Cross-platform CI: the full suite on **Ubuntu, macOS, and Windows**, a Docker
  build, SHA-pinned actions, and Dependabot for `github-actions`, `uv`, `docker`.
- The 3-OS matrix surfaced embedding-backend nondeterminism in the router, so the
  routing regression test is an **accuracy gate (≥93% of non-adversarial cases)**
  while adversarial/safety cases stay exact.

### AI evaluation
- **Offline** routing-accuracy gate runs in CI (no key).
- **Live** grounded-answer eval (`eval/run_eval.py`) with a committed report
  (`eval/results.md`, `eval/case_results.md`): 29/33, 0 hallucinations.
- **Scheduled** weekly workflow (`.github/workflows/evals.yml`) with a
  degradation gate — a **zero-hallucination invariant** plus a generous pass-rate
  floor — that auto-opens a GitHub issue on regression.

### Packaging
- Reproducible **Docker image (~2.9 GB)**: digest-pinned `python:3.12-slim` base
  and `uv`, `--frozen --no-dev`, non-root, CPU-only `torch` on Linux (down from
  ~19 GB with the default CUDA wheel), vector index baked in for offline start.
- MIT licensed.

### Known limitations (unchanged, documented in the README)
- Small demo knowledge base; one embedding model; English-only demo (the model is
  multilingual, but the KB, anchors, and prompts are English).
- The citation guardrail is exact-quote + number validation, **not** full
  natural-language entailment.
- Prompt-injection defense is a deterministic gate plus the citation backstop, not
  a guarantee against every adversarial phrasing.
- Human review is expected for high-stakes answers; the bot escalates when unsure.

[1.0.0]: https://github.com/Shcherbin96/rag-support-bot/releases/tag/v1.0.0
