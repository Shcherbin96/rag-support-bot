"""Evaluation harness for grounded answers, refusals, and small-talk.

The report is generated from measured results instead of hard-coded claims.
Run:
    uv run python eval/run_eval.py
"""

import os
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rag_bot import config
from rag_bot.answer import answer

REFUSAL_MARKERS = [
    "не зна",
    "нет информ",
    "нет надёж",
    "не могу",
    "уточн",
    "менеджер",
    "could not find",
    "cannot invent",
    "do not have reliable",
]
TEST_SET = Path(__file__).parent / "test_set.yaml"
RESULTS = Path(__file__).parent / "results.md"


def _answer_with_retry(question: str, model: str, retries: int = 3) -> dict:
    """Call answer() with retry/backoff for temporary rate-limit errors."""
    for attempt in range(retries):
        try:
            return answer(question, model=model)
        except Exception as exc:
            if "429" in str(exc) and attempt < retries - 1:
                time.sleep(20 * (attempt + 1))
                continue
            raise


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def score(case: dict, result: dict) -> bool:
    case_type = case["type"]
    text = result["text"]
    if case_type == "grounded":
        source_ok = any(case["expect_source"] in source for source in result["sources"])
        keyword_ok = str(case["expect_keyword"]).lower() in text.lower()
        return source_ok and keyword_ok
    if case_type == "refuse":
        return is_refusal(text) and not result["sources"]
    if case_type == "smalltalk":
        return not is_refusal(text) and not result["sources"]
    return False


def eval_model(model: str, cases: list[dict]) -> dict:
    by_type = {"grounded": [0, 0], "refuse": [0, 0], "smalltalk": [0, 0]}
    latencies, hallucinations, errors = [], 0, 0
    case_results = []

    for case in cases:
        case_type = case["type"]
        by_type[case_type][1] += 1
        try:
            started = time.perf_counter()
            result = _answer_with_retry(case["question"], model)
            latencies.append(time.perf_counter() - started)
            ok = score(case, result)
            error = ""
        except Exception as exc:
            errors += 1
            ok = False
            error = str(exc)
            result = {"text": f"<ERROR: {exc}>", "sources": [], "route": "error"}

        if ok:
            by_type[case_type][0] += 1
        elif case_type == "refuse" and not is_refusal(result["text"]):
            hallucinations += 1

        case_results.append(
            {
                "id": case["id"],
                "type": case_type,
                "ok": ok,
                "route": result.get("route", "unknown"),
                "sources": result.get("sources", []),
                "error": error,
            }
        )
        time.sleep(1.5)

    passed = sum(values[0] for values in by_type.values())
    total = sum(values[1] for values in by_type.values())
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    return {
        "model": model,
        "passed": passed,
        "total": total,
        "grounded": by_type["grounded"],
        "refuse": by_type["refuse"],
        "smalltalk": by_type["smalltalk"],
        "hallucinations": hallucinations,
        "avg_latency": avg_latency,
        "errors": errors,
        "cases": case_results,
    }


def _summary_sentence(results: list[dict]) -> str:
    total_hallucinations = sum(result["hallucinations"] for result in results)
    total_errors = sum(result["errors"] for result in results)
    if total_hallucinations == 0 and total_errors == 0:
        return "No hallucinations or runtime errors were observed in this eval run."
    return (
        f"This eval run observed {total_hallucinations} hallucination(s) "
        f"and {total_errors} runtime error(s); investigate before using the result as a quality claim."
    )


def main() -> None:
    cases = yaml.safe_load(TEST_SET.read_text(encoding="utf-8"))
    print(f"Cases: {len(cases)} · models: {len(config.EVAL_MODELS)}\n")

    rows = []
    for model in config.EVAL_MODELS:
        print(f"→ {model} ...")
        result = eval_model(model, cases)
        rows.append(result)
        print(
            f"   passed {result['passed']}/{result['total']} · "
            f"hallucinations {result['hallucinations']} · "
            f"~{result['avg_latency']}s · errors {result['errors']}\n"
        )

    header = "| Model | Passed | Grounded | Refusal | Small-talk | Hallucinations | Errors | Avg latency |\n"
    header += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = []
    for result in rows:
        lines.append(
            f"| {result['model']} | {result['passed']}/{result['total']} | "
            f"{result['grounded'][0]}/{result['grounded'][1]} | "
            f"{result['refuse'][0]}/{result['refuse'][1]} | "
            f"{result['smalltalk'][0]}/{result['smalltalk'][1]} | "
            f"{result['hallucinations']} | {result['errors']} | {result['avg_latency']}s |"
        )
    table = header + "\n".join(lines)
    print("\n" + table)

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    commit = os.getenv("GITHUB_SHA", "local-run")
    RESULTS.write_text(
        "# Evaluation Results\n\n"
        f"Generated at: `{timestamp}`\n\n"
        f"Commit: `{commit}`\n\n"
        f"Retrieval threshold: `{config.RETRIEVAL_MAX_DISTANCE}`\n\n"
        f"Test set: {len(cases)} cases (grounded / refuse / small-talk).\n\n"
        + table
        + "\n\n"
        + f"**Conclusion:** {_summary_sentence(rows)}\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
