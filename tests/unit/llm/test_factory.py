"""``resolve_client`` — one dispatch for every shape a user may pass."""

from __future__ import annotations

from typing import Any

import pytest

from structured_eval.llm.callable import CallableClient
from structured_eval.llm.chat_model import ChatModelClient
from structured_eval.llm.factory import resolve_client
from structured_eval.llm.litellm import LiteLlmClient

pytestmark = pytest.mark.unit


class FakeChatModel:
    """Shaped like a LangChain chat model, dependency-free."""

    def invoke(self, messages: Any) -> Any:
        return "ok"


def generate_fn(prompt: str, *, system: str | None = None) -> str:
    """A function with ``LlmClient.generate``'s own signature."""
    return "ok"


def test_a_client_passes_through_untouched():
    client = CallableClient(generate_fn)

    assert resolve_client(client) is client


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (FakeChatModel(), ChatModelClient),
        (generate_fn, CallableClient),
    ],
    ids=["chat-model", "callable"],
)
def test_dispatch_by_shape(spec, expected):
    assert isinstance(resolve_client(spec), expected)


def test_a_model_string_routes_to_litellm():
    resolved = resolve_client("anthropic/claude-opus-5")

    assert isinstance(resolved, LiteLlmClient)


def test_the_model_string_is_kept_verbatim():
    """The string names a *model*, not a registry key — it must survive as-is."""
    assert resolve_client("openai/gpt-5.5").model_name == "openai/gpt-5.5"


@pytest.mark.parametrize(
    "spec", [42, None, {"model": "x"}], ids=["int", "none", "dict"]
)
def test_unusable_specs_are_rejected(spec):
    with pytest.raises(TypeError, match="as an LLM client"):
        resolve_client(spec)
