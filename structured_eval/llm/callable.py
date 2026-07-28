"""The smallest client there is: a function that already *is* ``generate``."""

from __future__ import annotations

from typing import Protocol

from structured_eval.llm.base import LlmClient
from structured_eval.llm.exceptions import LlmInvocationError


class GenerateFn(Protocol):
    """``LlmClient.generate`` as a plain function — what ``CallableClient`` takes."""

    def __call__(self, prompt: str, *, system: str | None = None) -> str: ...


class CallableClient(LlmClient):
    """Wraps a function with ``generate``'s own signature.

    Useful for tests, local pipelines, and any provider reachable through one
    function. The call is forwarded verbatim — the function owns what to do
    with ``system``, exactly as a subclass would. Schema support comes from the
    base class: implementing this one function is enough for both methods.
    """

    def __init__(self, fn: GenerateFn, *, model_name: str = "") -> None:
        self._fn = fn
        self.model_name = model_name or getattr(fn, "__name__", "callable")

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        try:
            return self._fn(prompt, system=system)
        except Exception as exc:
            raise LlmInvocationError(f"callable client failed: {exc}") from exc
