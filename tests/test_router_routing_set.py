"""Regression tests for the labeled routing set."""

from pathlib import Path

import pytest
import yaml

from rag_bot.router import classify_query

ROUTING_SET = Path(__file__).resolve().parents[1] / "eval" / "routing_set.yaml"


def _cases() -> list[dict]:
    return yaml.safe_load(ROUTING_SET.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_labeled_routing_set(case):
    assert classify_query(case["question"]).value == case["expect_route"]
