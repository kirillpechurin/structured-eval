from __future__ import annotations

from typing import Any

from structured_eval.alignment.base import ArrayAligner
from structured_eval.alignment.by_index import ByIndexAligner
from structured_eval.alignment.by_key import ByKeyAligner
from structured_eval.alignment.hungarian import HungarianAligner
from structured_eval.model.config import ArrayStrategy


def make_aligner(
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX,
    params: dict[str, Any] | None = None,
) -> ArrayAligner:
    """Build the aligner for an array config's ``strategy`` from its ``params``.

    ``params`` keys match the chosen aligner's constructor arguments; an unknown
    key surfaces as a ``TypeError`` from that constructor.
    """
    params = params or {}
    if strategy == ArrayStrategy.BY_INDEX:
        return ByIndexAligner()
    if strategy == ArrayStrategy.HUNGARIAN:
        return HungarianAligner(**params)
    return ByKeyAligner(**params)
