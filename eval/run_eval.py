"""Evaluation harness for grounded answers, refusals, and small-talk.

The report is generated from measured results instead of hard-coded claims.
Run:
    PYTHONPATH=. uv run python eval/run_eval.py
"""

import os
import subprocess
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
PER_CASE_RESULTS = Path(__file__).parent / "case_results.md"


def _current_commit() -> str:
    """Return CI SHA when available, otherwise the local Git commit."""
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _ensure_eval_can_run() -> None:
    """Fail fast when a live LLM eval would be meaningless."""
    if config.LLM_API_KEY:
        return
    if os.getenv("EVAL_OFFLINE") == "1":
        return
    raise SystemExit(
        f"{config.LLM_API_KEY_ENV} is required for live eval with LLM_PROVIDER={config.LLM_PROVIDER}. "
        "Set EVAL_OFFLINE=1 only when intentionally testing offline/refusal behavior."
    )


def _answer_with_retry(question: str, model: str, retries: int = 3) -> dict:
    """Call answer() and retry only provider errors marked as transient."""
    result: dict | None = None
    for attempt in range(retries):
        result = answer(question, model=model)
        if (
            result.get("error_type") == "provider_error"
            and result.get("retryable")
            and attempt < retries - 1
        ):
            time.sleep(20 * (attempt + 1))
            continue
        return result
    return result or {"text": "<ERROR: no result>", "sources": [], "route": "error", "error_type": "runtime_error"}


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def score(case: dict, result: dict) -> bool:
    if result.get("error_type"):
        return False

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
        started = time.perf_counter()
        try:
            result = _answer_with_retry(case["question"], model)
            latencies.append(time.perf_counter() - started)
            error_type = result.get("error_type", "")
            if error_type:
                errors += 1
            ok = score(case, result)
            error = error_type
        except Exception as exc:
            errors += 1
            ok = False
            error = f"runtime_exception: {exc}"
            result = {"text": f"<ERROR: {exc}>", "sources": [], "route": "error", "error_type": "runtime_exception"}

        if ok:
            by_type[case_type][0] += 1
        elif case_type == "refuse" and not result.get("error_type") and not is_refusal(result["text"]):
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
    total_passed = sum(result["passed"] for result in results)
    total_cases = sum(result["total"] for result in results)
    total_hallucinations = sum(result["hallucinations"] for result in results)
    total_errors = sum(result["errors"] for result in results)

    if total_passed < total_cases:
        return (
            f"This eval run passed {total_passed}/{total_cases} cases, with "
            f"{total_hallucinations} hallucination(s) and {total_errors} runtime/model error(s). "
            "Do not use it as a quality claim until failures are investigated."
        )
    if total_hallucinations == 0 and total_errors == 0:
        return f"All {total_cases} cases passed with no observed hallucinations or runtime/model errors."
    return (
        f"All cases passed, but the run observed {total_hallucinations} hallucination(s) "
        f"and {total_errors} runtime/model error(s); investigate before using the result as a quality claim."
    )


def _case_results_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Case | Type | OK | Route | Sources | Error |",
        "|---|---|---|---:|---|---|---|",
    ]
    for model_result in rows:
        for case in model_result["cases"]:
            sources = ", ".join(case["sources"]) or "—"
            error = case["error"] or "—"
            lines.append(
                f"| {model_result['model']} | {case['id']} | {case['type']} | "
                f"{case['ok']} | {case['route']} | {sources} | {error} |"
            )
    return "\n".join(lines)


def main() -> None:
    _ensure_eval_can_run()
    cases = yaml.safe_load(TEST_SET.read_text(encoding="utf-8"))
    print(f"Cases: {len(cases)} · models: {len(config.EVAL_MODELS)}\n")
    if os.getenv("EVAL_OFFLINE") == "1" and not config.LLM_API_KEY:
        print(
            f"Offline eval mode enabled without {config.LLM_API_KEY_ENV}. "
            "Provider-backed cases are expected to fail.\n"
        )

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
    commit = _current_commit()
    RESULTS.write_text(
        "# Evaluation Results\n\n"
        f"Generated at: `{timestamp}`\n\n"
        f"Commit: `{commit}`\n\n"
        f"Retrieval threshold: `{config.RETRIEVAL_MAX_DISTANCE}`\n\n"
        f"Test set: {len(cases)} cases (grounded / refuse / small-talk).\n\n"
        + table
        + "\n\n"
        + f"**Conclusion:** {_summary_sentence(rows)}\n\n"
        + f"Per-case details: [`case_results.md`](case_results.md)\n",
        encoding="utf-8",
    )
    PER_CASE_RESULTS.write_text("# Per-case Evaluation Results\n\n" + _case_results_table(rows) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS} and {PER_CASE_RESULTS}")


if __name__ == "__main__":
    main()
