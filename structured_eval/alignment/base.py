from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from structured_eval.utils.paths import MISSING, navigate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from structured_eval.models.nodes.array_node import ArrayMatchResult

# Sentinel for a key that cannot be extracted (absent, or element not a dict).
_MISSING_KEY = object()


def key_value(element: Any, key: str | None) -> Any:
    """The alignment key of an element: the whole element, or a named field.

    Shared by every aligner that pairs on a key (``ByKeyAligner``,
    ``HungarianAligner``). Returns ``None`` for a missing field and a private
    sentinel when ``key`` is given but the element is not a dict.
    """
    if key is None:
        return element
    if isinstance(element, dict):
        value = navigate(element, key)
        return None if value is MISSING else value
    return _MISSING_KEY


def normalize_key(key: str | Sequence[str] | None, owner: str) -> list[str] | None:
    """One key or many, as the list of field paths every keyed aligner works on.

    ``None`` passes through with its meaning intact (key on the whole element);
    a lone field name becomes a one-field list, so a single-field key is just
    the degenerate composite key. ``owner`` names the aligner in the error.
    """
    if key is None:
        return None
    if isinstance(key, str):
        return [key]
    fields = list(key)
    if not fields:
        raise ValueError(f"{owner}: key must name at least one field")
    return fields


class ArrayAligner(ABC):
    """Maps actual array items onto expected ones (the only role of a matcher).

    ``align`` returns an ``ArrayMatchResult`` with matched ``(expected_idx,
    actual_idx)`` pairs plus the unmatched expected (missed) and actual
    (spurious) indices. Value scoring of matched pairs happens later, in the
    array metrics.
    """

    @abstractmethod
    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult: ...
