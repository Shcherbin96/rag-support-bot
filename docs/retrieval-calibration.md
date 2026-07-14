# Retrieval Calibration Notes

The project uses a deterministic relevance check before calling the LLM:

```python
chunks[0]["distance"] <= RETRIEVAL_MAX_DISTANCE
```

This is intentionally simple for a portfolio demo. It prevents the answer agent from grounding on unrelated chunks when semantic search returns weak matches.

## Current default

```text
RETRIEVAL_MAX_DISTANCE=1.2
```

The value is conservative and should be calibrated with real query distances before treating it as production-ready.

## Manual calibration protocol

Build the index:

```bash
uv run python -m rag_bot.ingestion
```

Run several in-scope queries that should retrieve a knowledge-base answer:

```bash
uv run python -m rag_bot.retrieval "сколько стоит доставка?"
uv run python -m rag_bot.retrieval "как оформить возврат?"
uv run python -m rag_bot.retrieval "есть ли гарантия на товары?"
uv run python -m rag_bot.retrieval "how can I pay for my order?"
```

Run several out-of-scope or out-of-base queries that should be refused:

```bash
uv run python -m rag_bot.retrieval "сколько стоит Tesla Model 3?"
uv run python -m rag_bot.retrieval "во сколько закрывается склад в Казани?"
uv run python -m rag_bot.retrieval "what is your CEO's phone number?"
uv run python -m rag_bot.retrieval "какая погода завтра в Москве?"
```

Record the top-1 distance for each query. A good threshold should:

- keep normal support questions below the threshold;
- put unrelated questions above the threshold;
- avoid refusing valid paraphrases;
- avoid passing unrelated context into the LLM.

## Production notes

For a real deployment, a more robust approach would combine:

- top-1 distance threshold;
- margin between top-1 and top-2 results;
- metadata filters by domain or document type;
- an evaluation set with false-positive and false-negative rates;
- periodic recalibration after changing the embedding model or knowledge base.

For this demo, the simple threshold keeps the safety behavior visible and easy to explain during interviews.
