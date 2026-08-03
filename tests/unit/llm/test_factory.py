"""``resolve_client`` — one dispatch for every shape a user may pass."""

from __future__ import annotations

from typing import Any

import pytest

from structured_eval.llm.callable import CallableClient
from structured_eval.llm.chat_model import ChatModelClient
from structured_eval.llm.factory import MODEL_ENV_VAR, default_client, resolve_client
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
    resolved = resolve_client("qwen/qwen3-235b-a22b-2507")

    assert isinstance(resolved, LiteLlmClient)


def test_the_model_string_is_kept_verbatim():
    """The string names a *model*, not a registry key — it must survive as-is."""
    assert resolve_client("openai/gpt-5.5").model_name == "openai/gpt-5.5"


@pytest.mark.parametrize("spec", [42, {"model": "x"}], ids=["int", "dict"])
def test_unusable_specs_are_rejected(spec):
    with pytest.raises(TypeError, match="as an LLM client"):
        resolve_client(spec)


# ── the zero-code path: nothing passed, everything from the environment ──────


def test_no_spec_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "qwen/qwen3-235b-a22b-2507")

    assert resolve_client(None).model_name == "qwen/qwen3-235b-a22b-2507"


def test_default_client_reads_the_model_from_the_environment(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "openai/gpt-5.5")
    resolved = default_client()

    assert isinstance(resolved, LiteLlmClient)
    assert resolved.model_name == "openai/gpt-5.5"


@pytest.mark.parametrize("value", ["", "   "], ids=["empty", "blank"])
def test_a_blank_model_is_treated_as_unset(monkeypatch, value):
    monkeypatch.setenv(MODEL_ENV_VAR, value)

    with pytest.raises(ValueError, match=MODEL_ENV_VAR):
        default_client()


def test_an_unset_model_is_a_configuration_error(monkeypatch):
    # No silent fallback to some default model: the judge names what graded the
    # data, or it refuses to run.
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    with pytest.raises(ValueError, match="no LLM client was given"):
        resolve_client(None)
