"""The litellm-backed client, exercised against a stubbed ``litellm`` module.

The extra is never installed for the unit run: ``litellm`` is lazy-imported, so
a stub in ``sys.modules`` covers the real code path without the dependency.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from structured_eval.llm.exceptions import LlmInvocationError, LlmResponseFormatError
from structured_eval.llm.litellm import LiteLlmClient

pytestmark = pytest.mark.unit


class Verdict(BaseModel):
    score: float
    reason: str


def make_response(content: Any) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.fixture
def litellm_stub(monkeypatch):
    """A stand-in for the ``litellm`` module, recording every call."""
    stub = SimpleNamespace(
        calls=[],
        reply='{"score": 1.0, "reason": "grounded"}',
        schema_supported=False,
        error=None,
    )

    def completion(**kwargs: Any) -> Any:
        stub.calls.append(kwargs)
        if stub.error is not None:
            raise stub.error
        return make_response(stub.reply)

    stub.completion = completion
    stub.supports_response_schema = lambda model: stub.schema_supported
    monkeypatch.setitem(sys.modules, "litellm", stub)
    return stub


def test_generate_returns_the_completion_text(litellm_stub):
    litellm_stub.reply = "a plain answer"

    assert LiteLlmClient("openai/gpt-5.5").generate("judge this") == "a plain answer"


def test_model_and_messages_are_passed_through(litellm_stub):
    LiteLlmClient("anthropic/claude-opus-5").generate("judge this", system="be strict")

    call = litellm_stub.calls[0]
    assert call["model"] == "anthropic/claude-opus-5"
    assert call["messages"] == [
        {"role": "system", "content": "be strict"},
        {"role": "user", "content": "judge this"},
    ]


def test_unset_options_are_not_sent(litellm_stub):
    """Omitted knobs must leave the provider's own defaults in place."""
    LiteLlmClient("ollama/llama3").generate("judge this")

    call = litellm_stub.calls[0]
    assert "temperature" not in call
    assert "max_tokens" not in call
    assert "timeout" not in call


def test_set_options_and_extra_params_are_forwarded(litellm_stub):
    client = LiteLlmClient(
        "ollama/llama3", temperature=0.0, max_tokens=512, api_base="http://localhost"
    )

    client.generate("judge this")

    call = litellm_stub.calls[0]
    assert (call["temperature"], call["max_tokens"]) == (0.0, 512)
    assert call["api_base"] == "http://localhost"


def test_native_response_format_used_when_supported(litellm_stub):
    litellm_stub.schema_supported = True

    result = LiteLlmClient("openai/gpt-5.5").generate_with_schema("judge", Verdict)

    assert litellm_stub.calls[0]["response_format"] is Verdict
    assert result.score == 1.0


def test_native_path_leaves_the_prompt_alone(litellm_stub):
    """Constrained generation means the schema need not be repeated in the prompt."""
    litellm_stub.schema_supported = True

    LiteLlmClient("openai/gpt-5.5").generate_with_schema("judge", Verdict)

    assert "JSON Schema" not in litellm_stub.calls[0]["messages"][0]["content"]


def test_prompt_fallback_when_schema_unsupported(litellm_stub):
    litellm_stub.schema_supported = False

    result = LiteLlmClient("ollama/llama3").generate_with_schema("judge", Verdict)

    assert "response_format" not in litellm_stub.calls[0]
    assert result.reason == "grounded"


def test_fallback_puts_the_schema_in_the_prompt(litellm_stub):
    litellm_stub.schema_supported = False

    LiteLlmClient("ollama/llama3").generate_with_schema("judge", Verdict)

    assert "JSON Schema" in litellm_stub.calls[0]["messages"][0]["content"]


def test_unknown_model_falls_back_instead_of_failing(litellm_stub):
    """``supports_response_schema`` raising is a routing signal, not an error."""

    def raises(model: str) -> bool:
        raise KeyError(model)

    litellm_stub.supports_response_schema = raises

    result = LiteLlmClient("private/model").generate_with_schema("judge", Verdict)

    assert "response_format" not in litellm_stub.calls[0]
    assert result.score == 1.0


def test_native_path_still_validates_the_reply(litellm_stub):
    litellm_stub.schema_supported = True
    litellm_stub.reply = '{"score": "high"}'

    with pytest.raises(LlmResponseFormatError):
        LiteLlmClient("openai/gpt-5.5").generate_with_schema("judge", Verdict)


def test_provider_error_is_wrapped(litellm_stub):
    litellm_stub.error = RuntimeError("rate limit exceeded")

    with pytest.raises(LlmInvocationError, match="rate limit exceeded") as excinfo:
        LiteLlmClient("openai/gpt-5.5").generate("judge")

    assert isinstance(excinfo.value.__cause__, RuntimeError)  # original preserved


@pytest.mark.parametrize(
    "content", [None, "", 42], ids=["none", "empty", "not-a-string"]
)
def test_unusable_completion_content_is_reported(litellm_stub, content):
    litellm_stub.reply = content

    with pytest.raises(LlmInvocationError, match="empty completion"):
        LiteLlmClient("openai/gpt-5.5").generate("judge")


def test_unexpected_response_shape_is_reported(litellm_stub):
    litellm_stub.completion = lambda **kwargs: SimpleNamespace(choices=[])

    with pytest.raises(LlmInvocationError, match="unexpected litellm response"):
        LiteLlmClient("openai/gpt-5.5").generate("judge")


@pytest.mark.parametrize(
    "call",
    [
        lambda client: client.generate("judge"),
        lambda client: client.generate_with_schema("judge", Verdict),
    ],
    ids=["generate", "generate-with-schema"],
)
def test_missing_extra_reports_the_install_hint(monkeypatch, call):
    # Routing to the prompt fallback also needs litellm, so a missing extra
    # surfaces as an install problem there too rather than as "no native schema".
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(ImportError, match=r"structured-eval\[litellm\]"):
        call(LiteLlmClient("openai/gpt-5.5"))


def test_empty_model_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        LiteLlmClient("")
