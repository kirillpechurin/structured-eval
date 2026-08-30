"""`CallableClient` — a function that is `generate`, forwarded verbatim."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from structured_eval.llm.callable import CallableClient
from structured_eval.llm.exceptions import LlmInvocationError

pytestmark = pytest.mark.unit


class Verdict(BaseModel):
    """The response schema passed to `generate_with_schema`."""

    score: float
    reason: str


def test_the_call_is_forwarded_verbatim():
    """The function *is* `generate` — both arguments reach it unchanged."""
    seen: list[tuple[str, str | None]] = []

    def record(prompt: str, *, system: str | None = None) -> str:
        seen.append((prompt, system))
        return "ok"

    CallableClient(record).generate("judge this", system="be strict")

    assert seen == [("judge this", "be strict")]


def test_the_client_names_itself_after_the_function():
    def my_llm(prompt: str, *, system: str | None = None) -> str:
        return "ok"

    assert CallableClient(my_llm).model_name == "my_llm"


def test_the_name_can_be_overridden():
    def my_llm(prompt: str, *, system: str | None = None) -> str:
        return "ok"

    assert CallableClient(my_llm, model_name="explicit").model_name == "explicit"


def test_schema_support_comes_from_the_base():
    """One implemented function is enough for both halves of the interface."""

    def reply(prompt: str, *, system: str | None = None) -> str:
        return '{"score": 1.0, "reason": "grounded"}'

    assert CallableClient(reply).generate_with_schema("judge", Verdict).score == 1.0


def test_a_failing_function_is_wrapped():
    def broken(prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("connection reset")

    with pytest.raises(LlmInvocationError, match="connection reset") as excinfo:
        CallableClient(broken).generate("judge")

    assert isinstance(excinfo.value.__cause__, RuntimeError)  # original preserved
