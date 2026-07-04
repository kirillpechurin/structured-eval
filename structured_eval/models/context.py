from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from structured_eval.models.config import EvalConfig  # noqa: TC001


class EvalContext(BaseModel):
    """The single owner of a sample's data.

    Every ``EvalNode`` in the tree holds a reference to one ``EvalContext``;
    nothing is copied. ``flat_actual`` / ``flat_expected`` are the documents
    pre-flattened to dot-notation paths, computed once up front.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    actual: Any
    expected: Any
    source: str | None
    flat_actual: dict[str, Any]
    flat_expected: dict[str, Any]
    config: EvalConfig
