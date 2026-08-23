"""`EvalContext` — the single owner of one sample's data during evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from structured_eval.models.config import EvalConfig  # noqa: TC001


class EvalContext(BaseModel):
    """The single owner of a sample's data.

    Every `EvalNode` in the tree holds a reference to one `EvalContext`; nothing
    is copied.

    Attributes:
        actual: The parsed document under evaluation.
        expected: The parsed reference document, if there is one.
        source: The original text faithfulness metrics grade against.
        flat_actual: `actual` pre-flattened to dot-notation paths, computed once.
        flat_expected: The same for `expected`.
        config: The configuration this evaluation runs under.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    actual: Any
    expected: Any
    source: str | None
    flat_actual: dict[str, Any]
    flat_expected: dict[str, Any]
    config: EvalConfig
