"""Provider-neutral LLM access for LLM-backed features.

The core package ships no provider SDK; what lives here is the seam:

- `LlmClient` — the interface: implement `generate` and you have a client.
- `resolve_client` — turns whatever the caller passed into an `LlmClient`;
  `default_client` is the "nothing at all" case.
- `CallableClient` / `ChatModelClient` — adapters over a bare function and a
  LangChain-style chat model.
- `LiteLlmClient` — every provider LiteLLM speaks, behind the `litellm` extra.
  Importing it costs nothing: `litellm` itself is reached for at call time.
"""

from structured_eval.llm.base import LlmClient
from structured_eval.llm.callable import CallableClient
from structured_eval.llm.chat_model import ChatModelClient
from structured_eval.llm.exceptions import (
    LlmError,
    LlmInvocationError,
    LlmResponseFormatError,
)
from structured_eval.llm.factory import MODEL_ENV_VAR, default_client, resolve_client
from structured_eval.llm.litellm import LiteLlmClient

__all__ = [
    "MODEL_ENV_VAR",
    "CallableClient",
    "ChatModelClient",
    "LiteLlmClient",
    "LlmClient",
    "LlmError",
    "LlmInvocationError",
    "LlmResponseFormatError",
    "default_client",
    "resolve_client",
]
