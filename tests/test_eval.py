"""Unit tests for eval accounting and retry behavior."""

import pytest

import eval.run_eval as run_eval


def test_eval_fails_fast_without_api_key(monkeypatch):
    monkeypatch.setattr(run_eval.config, "LLM_API_KEY", "")
    monkeypatch.delenv("EVAL_OFFLINE", raising=False)

    with pytest.raises(SystemExit, match="GEMINI_API_KEY is required"):
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
