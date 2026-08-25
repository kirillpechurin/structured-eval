"""The one place that turns a user-supplied spec into an `LlmClient`.

Every LLM-backed feature calls `resolve_client`, so none of them re-implements
the dispatch. `default_client` is the case where the caller named nothing.
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
    a `.env` file rather than a wrapper class — set
    `STRUCTURED_EVAL_LLM_MODEL=openai/gpt-4o` plus the provider's own key
    variable, and a judge metric builds its client itself.

    The `.env` file is *not* read here: a library loading files from the working
    directory on its own would be a surprise, so calling `load_dotenv()` — or
    exporting the variables — stays the caller's decision.

    An unset variable is a configuration error, not a fallback to some default
    model.

    Returns:
        A `LiteLlmClient` for the model the environment names.

    Raises:
        ValueError: If the variable is unset or empty.

    Example:
        >>> import os
        >>> from structured_eval.llm import default_client
        >>> os.environ["STRUCTURED_EVAL_LLM_MODEL"] = "openai/gpt-4o"
        >>> default_client().model_name
        'openai/gpt-4o'
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
    """Coerce a user-supplied spec to an `LlmClient`.

    Note the string form names a **model**, not a registered class — unlike
    `resolve_metric("numeric")`, whose strings are registry keys.

    Dispatch lives apart from the clients so that none of them has to know
    about `LiteLlmClient`, which sits behind an extra. Importing it here costs
    nothing: that module reaches for `litellm` only at call time.

    Args:
        spec: An `LlmClient`, a `"provider/model"` string (routed to
            `LiteLlmClient`, which needs the `litellm` extra), a LangChain-style
            chat model, a function with `generate`'s own signature, or `None`
            for "the caller chose nothing".

    Returns:
        The client to use — the spec itself when it already is one, the adapter
        wrapping it otherwise, and `default_client()` for `None`.

    Raises:
        TypeError: If `spec` is none of the forms above.

    Example:
        >>> from structured_eval.llm import resolve_client
        >>> type(resolve_client("openai/gpt-4o")).__name__     # a model name
        'LiteLlmClient'
        >>> def my_model(prompt: str, *, system: str | None = None) -> str:
        ...     return my_sdk.complete(system=system, user=prompt).text
        >>> type(resolve_client(my_model)).__name__            # a plain function
        'CallableClient'
        >>> from langchain_openai import ChatOpenAI  # doctest: +SKIP
        >>> model = ChatOpenAI(model="gpt-4o")  # doctest: +SKIP
        >>> type(resolve_client(model)).__name__  # a chat model  # doctest: +SKIP
        'ChatModelClient'
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
