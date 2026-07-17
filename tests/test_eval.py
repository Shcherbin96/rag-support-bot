"""Unit tests for eval accounting and retry behavior."""

import pytest

import eval.run_eval as run_eval


def test_eval_fails_fast_without_selected_provider_api_key(monkeypatch):
    monkeypatch.setattr(run_eval.config, "LLM_API_KEY", "")
    monkeypatch.setattr(run_eval.config, "LLM_PROVIDER", "nvidia")
    monkeypatch.setattr(run_eval.config, "LLM_API_KEY_ENV", "NVIDIA_API_KEY")
    monkeypatch.delenv("EVAL_OFFLINE", raising=False)

    with pytest.raises(SystemExit, match="NVIDIA_API_KEY is required"):
        run_eval._ensure_eval_can_run()


def test_eval_offline_mode_allows_missing_api_key(monkeypatch):
    monkeypatch.setattr(run_eval.config, "LLM_API_KEY", "")
    monkeypatch.setenv("EVAL_OFFLINE", "1")

    run_eval._ensure_eval_can_run()


def test_answer_with_retry_does_not_retry_permanent_provider_error(monkeypatch):
    calls = []
    sleeps = []

    def fake_answer(question, model):
        calls.append((question, model))
        return {
            "text": "refusal",
            "sources": [],
            "route": "factual_in_domain",
            "error_type": "provider_error",
            "retryable": False,
        }

    monkeypatch.setattr(run_eval, "answer", fake_answer)
    monkeypatch.setattr(run_eval.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = run_eval._answer_with_retry("question", "model")

    assert result["error_type"] == "provider_error"
    assert len(calls) == 1
    assert sleeps == []


def test_answer_with_retry_retries_transient_provider_error(monkeypatch):
    responses = [
        {
            "text": "temporary failure",
            "sources": [],
            "route": "factual_in_domain",
            "error_type": "provider_error",
            "retryable": True,
        },
        {
            "text": "temporary failure",
            "sources": [],
            "route": "factual_in_domain",
            "error_type": "provider_error",
            "retryable": True,
        },
        {
            "text": "ok",
            "sources": [],
            "route": "smalltalk",
            "error_type": "",
            "retryable": False,
        },
    ]
    sleeps = []

    def fake_answer(question, model):
        return responses.pop(0)

    monkeypatch.setattr(run_eval, "answer", fake_answer)
    monkeypatch.setattr(run_eval.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = run_eval._answer_with_retry("question", "model")

    assert result["error_type"] == ""
    assert sleeps == [20, 40]


def test_eval_model_counts_errors_hallucinations_and_case_details(monkeypatch):
    responses = [
        {
            "text": "provider failed",
            "sources": [],
            "route": "factual_in_domain",
            "error_type": "provider_error",
            "retryable": False,
        },
        {
            "text": "Tesla Model 3 costs 1 dollar.",
            "sources": ["unrelated.md"],
            "route": "factual_in_domain",
            "error_type": "",
        },
        {
            "text": "Hi. I am the support assistant.",
            "sources": [],
            "route": "smalltalk",
            "error_type": "",
        },
    ]

    def fake_answer_with_retry(question, model):
        return responses.pop(0)

    cases = [
        {
            "id": "grounded_provider_error",
            "type": "grounded",
            "question": "How much is shipping?",
            "expect_source": "delivery",
            "expect_keyword": "350",
        },
        {
            "id": "refusal_hallucination",
            "type": "refuse",
            "question": "How much is a Tesla Model 3?",
        },
        {
            "id": "smalltalk_ok",
            "type": "smalltalk",
            "question": "Hello",
        },
    ]

    monkeypatch.setattr(run_eval, "_answer_with_retry", fake_answer_with_retry)
    monkeypatch.setattr(run_eval.time, "sleep", lambda seconds: None)

    result = run_eval.eval_model("fake-model", cases)

    assert result["passed"] == 1
    assert result["total"] == 3
    assert result["errors"] == 1
    assert result["hallucinations"] == 1
    assert result["grounded"] == [0, 1]
    assert result["refuse"] == [0, 1]
    assert result["smalltalk"] == [1, 1]
    assert result["cases"][0]["error"] == "provider_error"
    assert result["cases"][1]["error"] == ""
    assert result["cases"][2]["ok"] is True


def test_summary_includes_pass_rate_when_failures_exist():
    summary = run_eval._summary_sentence(
        [
            {
                "passed": 0,
                "total": 3,
                "hallucinations": 0,
                "errors": 0,
            }
        ]
    )

    assert "passed 0/3" in summary
    assert "Do not use it as a quality claim" in summary


def test_current_commit_prefers_github_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    assert run_eval._current_commit() == "abc123"


def _rows(passed: int, total: int, hallucinations: int) -> list[dict]:
    return [{"passed": passed, "total": total, "hallucinations": hallucinations}]


def test_evaluate_degradation_gate_off_by_default_even_with_hallucinations():
    # No EVAL_FAIL_UNDER (fail_under=None) must never fail, even if the run
    # contains hallucinations — manual `uv run python eval/run_eval.py` always
    # exits 0 unless the gate is explicitly activated.
    rows = _rows(passed=10, total=33, hallucinations=3)

    assert run_eval.evaluate_degradation(rows, None) == []


def test_evaluate_degradation_healthy_run_passes():
    rows = _rows(passed=30, total=33, hallucinations=0)

    assert run_eval.evaluate_degradation(rows, 0.75) == []


def test_evaluate_degradation_hallucination_hard_fails_regardless_of_pass_rate():
    # Pass-rate clears the floor comfortably, but any hallucination is a hard
    # fail — it is the safety invariant, not subject to the generous floor.
    rows = _rows(passed=33, total=33, hallucinations=1)

    failures = run_eval.evaluate_degradation(rows, 0.75)

    assert len(failures) == 1
    assert "1 hallucination(s)" in failures[0]
    assert "quality gate failed" in failures[0]


def test_evaluate_degradation_pass_rate_below_floor_fails():
    rows = _rows(passed=20, total=33, hallucinations=0)  # ~0.606

    failures = run_eval.evaluate_degradation(rows, 0.75)

    assert len(failures) == 1
    assert "pass-rate" in failures[0]


def test_evaluate_degradation_reports_both_failures_independently():
    rows = _rows(passed=10, total=33, hallucinations=2)

    failures = run_eval.evaluate_degradation(rows, 0.75)

    assert len(failures) == 2


def test_evaluate_degradation_handles_zero_total_cases():
    rows = _rows(passed=0, total=0, hallucinations=0)

    assert run_eval.evaluate_degradation(rows, 0.75) == []
