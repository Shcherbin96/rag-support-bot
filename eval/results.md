# Evaluation Results

Generated at: `2026-07-17T11:01:29+00:00`

Commit: `8d213a12aa5de05dd8f2890c4d3a5f66e73014ee`

Retrieval threshold: `1.2`

Test set: 33 cases (grounded / refuse / small-talk).

| Model | Passed | Grounded | Refusal | Small-talk | Hallucinations | Errors | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| google/gemini-2.5-flash-lite | 29/33 | 16/19 | 10/11 | 3/3 | 0 | 3 | 0.96s |

**Conclusion:** This eval run passed 29/33 cases, with 0 hallucination(s) and 3 runtime/model error(s). Do not use it as a quality claim until failures are investigated.

Per-case details: [`case_results.md`](case_results.md)
