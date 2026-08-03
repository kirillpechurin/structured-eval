"""Provider-neutral LLM access for LLM-backed features.

The core package ships **no provider SDK**. What lives here is the seam:

- ``LlmClient`` — the interface. Implement ``generate`` and you have a client;
  override ``generate_with_schema`` (or one of its three steps) when your
  provider can constrain output.
- ``resolve_client`` — turns whatever the user passed (a client, a chat model,
  a callable, a ``"provider/model"`` string, or nothing at all) into an
  ``LlmClient``; ``default_client`` is the "nothing at all" case, configured
  from ``STRUCTURED_EVAL_LLM_MODEL``.
- ``CallableClient`` / ``ChatModelClient`` — zero-dependency adapters over a
  bare function and a LangChain-style chat model.
- ``LiteLlmClient`` — the batteries-included option, behind the ``litellm``
  extra; import it from ``structured_eval.llm.litellm``.
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

__all__ = [
    "MODEL_ENV_VAR",
    "CallableClient",
    "ChatModelClient",
    "LlmClient",
    "LlmError",
    "LlmInvocationError",
    "LlmResponseFormatError",
    "default_client",
    "resolve_client",
]
