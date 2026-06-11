from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structured_eval.core.config import EvalConfig


@dataclass
class EvalContext:
    """The single owner of a sample's data.

    Every ``EvalNode`` in the tree holds a reference to one ``EvalContext``;
    nothing is copied. ``flat_actual`` / ``flat_expected`` are the documents
    pre-flattened to dot-notation paths, computed once up front.
    """

    actual: dict | list
    expected: dict | list | None
    source: str | None
    flat_actual: dict
    flat_expected: dict
    config: "EvalConfig"
