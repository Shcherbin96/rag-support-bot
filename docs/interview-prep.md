# Interview preparation — RAG Support Bot

Answers to the questions an interviewer is likely to ask about this project.
Written in plain English (B1–B2) but technically precise. Each answer is true to
the code; none over-claims.

---

### 1. Why does the bot refuse instead of answering when it is unsure?

Because for a support bot, a confident wrong answer is worse than no answer — it
makes a false promise to a customer in the company's name. So the whole design
is "fail closed": if the answer isn't grounded in retrieved evidence, the bot
refuses and offers a human. I chose precision over recall on purpose.

### 2. How do you stop the model from inventing a price or a policy?

With a strict citation contract on every reply. The model must return citations,
and I validate three things: the cited chunk was actually retrieved, the quote
really appears in that chunk, and **every number in the answer appears in a
cited quote.** If any check fails, the reply is rejected and the bot refuses. So
a number that isn't in the evidence simply cannot reach the customer.

### 3. What is the semantic router, and why not just keywords?

The router decides what kind of question this is — in-domain, small talk,
out-of-domain, or a prompt-injection attempt — before any retrieval or model
call. Keywords are brittle: they miss paraphrases and synonyms. Instead I
compare the message to curated "anchor" phrases for each route using embedding
cosine similarity, so "where's my order" and "how do I track my delivery" route
the same way. Injection attempts are caught by an exact, deterministic check,
because safety should not depend on a fuzzy score.

### 4. How is retrieval done, and why a vector database?

The knowledge base is split into chunks, embedded with a local model, and stored
in a local Chroma vector index. At query time I embed the question and fetch the
closest chunks. A vector search finds text by *meaning*, not exact words, which
is what makes the bot answer natural questions. A distance threshold decides
whether the best chunks are relevant enough to use at all — if nothing is close,
the bot refuses.

### 5. Why is quote-matching markdown-insensitive, and how do you keep it safe?

The knowledge base stores facts in markdown, like `**Phone:** 1-800-…`, but the
model quotes the plain text, `Phone: 1-800-…`. A strict character match would
reject that correct citation. So I ignore markdown markers when comparing. The
important part: I replace a marker with a **space**, not with nothing — so
`"1**2**"` becomes `"1 2"`, never `"12"`. That means the relaxation can never
merge characters into a number the source never stated. A live eval caught the
original bug, and a later check caught this exact edge; both are now tested.

### 6. How do you defend against prompt injection?

Two layers, and I'm honest that it's bounded, not absolute. First, the router
has a deterministic check that catches common injection phrasing and routes it
to a refusal. Second — the real backstop — even if a crafted message reached the
model, the answer still has to pass the citation and number contract, so it
cannot invent facts or leak anything that isn't grounded in the knowledge base.
The worst case is a refusal, not a fabricated or leaked answer.

### 7. How are transport retries different from how you handle bad content?

Transport retries (in the SDK) are for network problems — a timeout or a rate
limit — where you resend the same request. Bad *content* is different: a reply
that broke my grounding contract will not be fixed by resending, so I don't
retry it — I refuse. I keep a bounded LLM timeout below the bot's answer budget,
so a slow provider becomes a clean retryable error instead of hanging the chat.

### 8. Why don't the tests use a real LLM?

Speed, determinism, and cost. A test that calls a real model is slow, can answer
differently each run, and needs a paid key, so it can't run on every push.
Instead the model is faked in the tests. All 78 tests run offline with no key and
no network, which is why they run on every commit across three operating
systems. The two tests that do need a real provider are marked `live` and are
skipped by default.

### 9. Why do you need live evals if you already have unit tests?

They answer different questions. Unit tests prove my code is correct with a fixed
fake reply — routing, retrieval, validation. They can't tell me whether the
*real* model, today, still produces good grounded answers. The live eval runs
real questions against the real model and scores the results, so it catches
regressions in the model's behavior and in the prompt, not just in my code.

### 10. How do you measure quality, and what did the eval catch?

The live eval scores each case: was it grounded and correctly sourced, did it
refuse when it should, and — most important — did it ever hallucinate. The
headline is **29 of 33 with 0 hallucinations.** The eval also earned its place:
it surfaced a real bug where correct answers about phone numbers and prices were
being wrongly refused, because the citation check demanded verbatim markdown. I
fixed the quote matching, and the pass rate improved honestly.

### 11. The routing test tolerates ≥93%, not 100%. Why weaken it?

I didn't weaken it — I fixed its shape. The router is a cosine-threshold
classifier, and the 3-OS CI matrix proved that the embedding backend returns
slightly different numbers on macOS versus Linux, which can flip a case sitting
right on a threshold. Demanding a bit-exact result on every case across every OS
tests the math library, not my logic. An accuracy floor is the right test for a
fuzzy classifier — and I kept the adversarial/safety cases exact, because an
injection slipping through is never acceptable margin.

### 12. What happens on a compound question with two topics?

Retrieval favors the strongest single topic, so a question like "how much is
shipping and what's your phone number" may retrieve the shipping chunks but not
the contact chunk. The bot then answers the part it has evidence for and refuses
the part it doesn't — for example, it gives shipping prices and says it doesn't
have the phone number, offering a human. That's honest and safe; improving it
with query decomposition is on the roadmap.

### 13. Why a local embedding model instead of an embedding API?

Cost, latency, and no external dependency: a local model means no per-call
embedding fee and nothing to break if a third-party API is down. The trade is a
heavier container, which I addressed by switching to CPU-only PyTorch on Linux —
that alone cut the image from about 19 GB to under 3 GB.

### 14. What would production deployment require?

A larger, versioned knowledge base with a proper re-ingestion pipeline; a shared,
warm vector store instead of a per-process one; provider rate-limit and cost
controls; conversation memory and a real human-handoff integration; and secret
management through the platform. The guardrails, typed errors, tests, and evals
that already exist are what make that step safe rather than risky — they are the
hard part, and they are done.

### 15. What are the limits of your grounding check?

It is exact-quote plus number validation, not full natural-language entailment.
So it reliably stops invented numbers, quotes, and unsupported facts, but it
does not verify that a correctly-cited sentence is *the best* or most complete
answer — only that it is grounded. I state this limitation openly rather than
claim the check understands meaning. A semantic-entailment judge would go
further, and it's on the roadmap, gated behind the eval harness.
