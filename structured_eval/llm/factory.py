"""The one place that turns a user-supplied spec into an ``LlmClient``.

Kept apart from the clients for the same reason ``alignment/factory`` is kept
apart from the aligners: dispatch is its own job. It also keeps them free of
any reference to the litellm-backed client, which lives behind an extra.
Importing ``LiteLlmClient`` here costs nothing — that module only reaches for
``litellm`` at call time.
"""

from __future__ import annotations

import os
from typing import Any

from structured_eval.llm.base import LlmClient
from structured_eval.llm.callable import CallableClient
from structured_eval.llm.chat_model import ChatModelClient
from structured_eval.llm.litellm import LiteLlmClient

# Names the judge's model when the caller names none. Only the model: each
# provider's credentials stay in that provider's own variable, which litellm
# reads itself, so switching providers stays a one-string change.
MODEL_ENV_VAR = "STRUCTURED_EVAL_LLM_MODEL"


def default_client() -> LlmClient:
    """The zero-code path: a client configured entirely from the environment.

    The point of the feature is that trying an LLM-backed metric costs a line in
    a ``.env`` file rather than a wrapper class — set
    ``STRUCTURED_EVAL_LLM_MODEL=qwen/qwen3-235b-a22b-2507`` plus the provider's own
    key variable, and a judge metric builds its client itself.

    The ``.env`` file is *not* read here: a library quietly loading files from
    the working directory would be a surprise, so calling ``load_dotenv()`` (or
    exporting the variables) stays the caller's decision.

    An unset variable is a configuration error, not a silent fallback to some
    default model — nobody should discover which model graded their data by
    reading a bill.
    """
    model = os.environ.get(MODEL_ENV_VAR, "").strip()
    if not model:
        raise ValueError(
            f"no LLM client was given and {MODEL_ENV_VAR} is unset; either pass "
            "client=... or set it to a 'provider/model' string, e.g. "
            f"{MODEL_ENV_VAR}=openai/gpt-4o"
        )
    return LiteLlmClient(model)


def resolve_client(spec: LlmClient | str | Any | None) -> LlmClient:
    """Coerce a user-supplied spec to an ``LlmClient``.

    Accepts an ``LlmClient`` as-is, a ``"provider/model"`` string (routed to
    ``LiteLlmClient``, which needs the ``litellm`` extra), a LangChain-style
    chat model, or a function with ``generate``'s own signature. ``None`` means
    "the caller chose nothing" and falls back to ``default_client``. Every
    LLM-backed feature goes through this, so none re-implements the dispatch.

    Note the string form names a **model**, not a registered class — unlike
    ``resolve_metric("numeric")``, whose strings are registry keys.
    """
    if spec is None:
        return default_client()
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
