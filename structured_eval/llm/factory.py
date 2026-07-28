"""The one place that turns a user-supplied spec into an ``LlmClient``.

Kept apart from the clients for the same reason ``alignment/factory`` is kept
apart from the aligners: dispatch is its own job. It also keeps them free of
any reference to the litellm-backed client, which lives behind an extra.
Importing ``LiteLlmClient`` here costs nothing — that module only reaches for
``litellm`` at call time.
"""

from __future__ import annotations

from typing import Any

from structured_eval.llm.base import LlmClient
from structured_eval.llm.callable import CallableClient
from structured_eval.llm.chat_model import ChatModelClient
from structured_eval.llm.litellm import LiteLlmClient


def resolve_client(spec: LlmClient | str | Any) -> LlmClient:
    """Coerce a user-supplied spec to an ``LlmClient``.

    Accepts an ``LlmClient`` as-is, a ``"provider/model"`` string (routed to
    ``LiteLlmClient``, which needs the ``litellm`` extra), a LangChain-style
    chat model, or a function with ``generate``'s own signature. Every
    LLM-backed feature goes through this, so none re-implements the dispatch.

    Note the string form names a **model**, not a registered class — unlike
    ``resolve_metric("numeric")``, whose strings are registry keys.
    """
    if isinstance(spec, LlmClient):
        return spec
    if isinstance(spec, str):
        return LiteLlmClient(spec)
    if callable(getattr(spec, "invoke", None)):
        return ChatModelClient(spec)
    if callable(spec):
        return CallableClient(spec)
    raise TypeError(
        f"cannot use {type(spec)!r} as an LLM client: expected an LlmClient, a "
        "'provider/model' string, a chat model with .invoke(), or a callable"
    )
