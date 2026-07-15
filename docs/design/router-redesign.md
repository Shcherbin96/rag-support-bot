# Design: Semantic query router

Date: 2026-07-15 · Status: approved, pending implementation

## Problem

The pre-retrieval router (`rag_bot/router.py`) classifies each user message into
`smalltalk` / `factual_in_domain` / `out_of_domain` / `adversarial` before any
retrieval or LLM call. It currently does this with hand-maintained keyword,
prefix, and phrase lists.

This has two structural weaknesses:

1. **False refusals (recall).** A legitimate in-domain question phrased outside
   the token lists is routed to `out_of_domain` and refused. Example:
   "will my parcel arrive by Friday?" contains no listed shipping token, so the
   assistant refuses a question the knowledge base can actually answer.
2. **Poor generalization / high maintenance.** Adapting the demo to a new
   business domain means hand-editing hundreds of tokens. This contradicts the
   README claim that the domain is replaceable by swapping the knowledge base.

## Goals

- Reduce false refusals on in-domain paraphrases (recall).
- Generalize to a new domain with example phrases instead of exhaustive token
  lists.
- Keep the routing decision **LLM-free and deterministic**: runnable in CI
  without API keys, and independent of provider availability/quota.
- Preserve the project thesis: deterministic safety logic separate from LLM
  behavior, and fail-closed when uncertain.

## Non-goals

- No LLM-based routing (adds latency/cost, breaks CI-without-keys, and depends
  on provider quota).
- No trained classifier or committed model artifact (extra machinery, labeled
  data, retraining on domain change).
- No multilingual routing — the demo is English-only.

## Approach: hybrid deterministic + semantic anchor gate

Two considered alternatives were rejected:

- **Trained classifier on embeddings** — most accurate, but needs a labeled
  dataset, a training step, and artifact management; overkill for the demo.
- **Pure retrieval-distance gate** — simplest, but the README already documents
  that vector distance alone is not a reliable domain boundary (a tiny index
  returns deceptively low distances for unrelated queries → false accepts).

The chosen approach keeps a deterministic layer for security and replaces the
domain token lists with a semantic similarity gate over curated example phrases
("anchors"), embedded with the local `sentence-transformers` model already used
for retrieval.

### Routing logic

```
text → normalize
  1. empty            → SMALLTALK
  2. adversarial      → ADVERSARIAL      (deterministic regex/phrase match)
  3. semantic gate:
       embed the query with the local model
       s_small = max cosine(query, SMALLTALK anchors)
       s_in    = max cosine(query, FACTUAL_IN_DOMAIN anchors)
       s_out   = max cosine(query, OUT_OF_DOMAIN anchors)
```

Decision rule (fail-closed):

- `s_small` is the maximum and `s_small >= SMALLTALK_MIN` → `SMALLTALK`
- else `s_in >= IN_DOMAIN_MIN` **and** `(s_in - s_out) >= MARGIN`
  → `FACTUAL_IN_DOMAIN`
- else → `OUT_OF_DOMAIN`

To be accepted as in-domain, a query must be **both** absolutely close to a
domain anchor (`IN_DOMAIN_MIN`) **and** clearly closer to it than to any
off-topic anchor (`MARGIN`). Anything ambiguous falls through to
`OUT_OF_DOMAIN`, i.e. a refusal — consistent with "when unsure, do not invent".

**Why adversarial stays deterministic:** prompt-injection detection is a
security function. A regex/phrase match is reliable and auditable; fuzzy cosine
similarity is not something to trust for security, because an injection can be
phrased to look benign.

## Components

### `rag_bot/embeddings.py` (new)

A shared, lazily loaded embedding model so the router does not load a second
copy at import time and tests that avoid the semantic path stay fast.

- `get_model()` → cached `SentenceTransformer(config.EMBED_MODEL)`, loaded once
  on first use.
- `embed(texts: list[str]) -> np.ndarray` → L2-normalized embeddings so cosine
  similarity is a dot product.

### `rag_bot/router.py` (revised)

- Keep `QueryRoute`, `_normalize`, and the deterministic `ADVERSARIAL_PHRASES`
  check (pure Python, no model).
- Remove `DOMAIN_TOKENS`, `HARD_NEGATIVE_*`, `OTHER_COMPANY_MARKERS`,
  `DOMAIN_PHRASES`, and the prefix machinery.
- Add `ANCHORS: dict[QueryRoute, list[str]]` — curated example phrases:
  - ~12–18 `FACTUAL_IN_DOMAIN`: shipping, payment, returns, warranty, rewards,
    contact, order changes, product availability;
  - ~12–18 `OUT_OF_DOMAIN`: weather, finance/stocks, movies, police/emergency/
    government, other stores, cooking, general knowledge;
  - ~6 `SMALLTALK`: greetings, thanks, "who are you".
  These are examples, not exhaustive rules.
- Lazily compute and cache anchor embedding matrices (one embed pass, module
  level).
- Thresholds `IN_DOMAIN_MIN`, `MARGIN`, `SMALLTALK_MIN` as constants, overridable
  via `ROUTER_*` environment variables for calibration.
- `classify_query(text, embed_fn=None)` — `embed_fn` defaults to
  `embeddings.embed` and is injectable so unit tests can pass controlled vectors
  and exercise the decision logic without loading the model.

The public signature `classify_query(text)` is preserved (the new parameter is
optional), so `answer.py` and other callers are unchanged.

## Calibration and testing

- **Calibrate** thresholds against `eval/routing_set.yaml` (the existing labeled
  routing set), expanded with paraphrase cases the keyword router got wrong
  (e.g. "will my parcel arrive by Friday?", "do you take Apple Pay?",
  "can I send it as a present?"). Choose thresholds that maximize labeled
  accuracy with a fail-closed bias (prefer `out_of_domain` on ties).
- **Unit tests** (`test_router.py`): decision logic with an injected fake
  `embed_fn` returning controlled vectors — fast, deterministic, no model load.
  Adversarial-detection tests stay pure Python.
- **Regression test** (`test_router_routing_set.py`): run the full labeled set
  through the real semantic router (model loaded once) end to end, including the
  new paraphrase cases.
- Document the calibration method and how to recalibrate in
  `docs/retrieval-calibration.md`.

## Error handling

If embedding fails (e.g. the model cannot load), `classify_query` catches the
error, logs it, and returns `OUT_OF_DOMAIN` (a refusal) rather than raising —
consistent with the pipeline's fail-closed behavior.

## Known limitations and follow-ups

- In the bot process the embedding model may load twice: once via this shared
  loader (router) and once via Chroma's embedding function (retrieval).
  Acceptable for a demo. A follow-up can route Chroma through the shared loader
  so a single instance is shared by ingestion, retrieval, and the router.
- Anchors are still authored by hand, but they are a small set of natural
  examples rather than exhaustive keyword rules, and can be partly derived from
  the knowledge base.
- The semantic gate improves the domain boundary but, like any similarity
  threshold, is not a perfect classifier; citation validation downstream remains
  the last line of defense against unsupported answers.
