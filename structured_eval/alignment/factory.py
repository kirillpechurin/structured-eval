"""Construction of the aligner named by an array field's `strategy`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.alignment.by_index import ByIndexAligner
from structured_eval.alignment.by_key import ByKeyAligner
from structured_eval.alignment.hungarian import HungarianAligner
from structured_eval.models.config import ArrayStrategy

if TYPE_CHECKING:
    from structured_eval.alignment.base import ArrayAligner


def make_aligner(
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX,
    params: dict[str, Any] | None = None,
) -> ArrayAligner:
    """Build the aligner for an array config's `strategy` from its `params`.

    Args:
        strategy: Which strategy to build.
        params: That aligner's constructor arguments, by name. An unknown key
            surfaces as a `TypeError` from the constructor itself.

    Returns:
        The aligner instance the strategy names.

    Example:
        >>> from structured_eval.alignment import make_aligner
        >>> from structured_eval.models import ArrayStrategy
        >>> type(make_aligner()).__name__
        'ByIndexAligner'
        >>> aligner = make_aligner(ArrayStrategy.BY_KEY, {"key": "sku"})
        >>> aligner.align([{"sku": "A"}, {"sku": "B"}], [{"sku": "B"}]).matched
        [(1, 0)]
    """
    params = params or {}
    if strategy == ArrayStrategy.BY_INDEX:
        return ByIndexAligner()
    if strategy == ArrayStrategy.HUNGARIAN:
        return HungarianAligner(**params)
    return ByKeyAligner(**params)
