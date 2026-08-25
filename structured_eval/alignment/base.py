"""The `ArrayAligner` interface and the key helpers keyed strategies share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from structured_eval.utils.paths import MISSING, navigate

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from structured_eval.models.nodes.array_node import ArrayMatchResult

# Sentinel for a key that cannot be extracted (absent, or element not a dict).
_MISSING_KEY = object()


def key_value(element: Any, key: str | None) -> Any:
    """The alignment key of an element: the whole element, or a named field.

    Shared by every aligner that pairs on a key (`ByKeyAligner`,
    `HungarianAligner`).

    Args:
        element: One array element.
        key: Field path to key on, or `None` to key on the element itself.

    Returns:
        The element itself when `key` is `None`, the field's value when it is
        there, `None` when the field is absent, and a private sentinel when
        `key` is given but the element carries no fields at all. `keyable`
        reads that sentinel back.

    Example:
        >>> from structured_eval.alignment import key_value
        >>> key_value({"sku": "A-1", "qty": 2}, "sku")
        'A-1'
        >>> key_value({"sku": "A-1"}, "warehouse") is None   # field absent
        True
        >>> key_value("A-1", None)                           # the whole element
        'A-1'
    """
    if key is None:
        return element
    if isinstance(element, dict):
        value = navigate(element, key)
        return None if value is MISSING else value
    return _MISSING_KEY


def keyable(values: Iterable[Any]) -> bool:
    """Could every part of this key be extracted from its element?

    `key_value` answers a sentinel for an element that carries no fields at all,
    and two sentinels are the same object — scoring one against another reads as
    a perfect match. Every such pair would tie at 1.0 and be claimed in index
    order, degenerating keyed alignment into alignment by position.

    An element with no key matches nothing, so the aligners ask this first.

    Args:
        values: The key values of one element, as `key_value` returned them.

    Returns:
        True when every value came from an element that could be keyed.

    Example:
        >>> from structured_eval.alignment import key_value, keyable
        >>> keyable([key_value({"sku": "A-1"}, "sku")])
        True
        >>> keyable([key_value("A-1", "sku")])  # a scalar has no field to key on
        False
    """
    return all(value is not _MISSING_KEY for value in values)


def normalize_key(key: str | Sequence[str] | None, owner: str) -> list[str] | None:
    """One key or many, as the list of field paths every keyed aligner works on.

    A lone field name becomes a one-field list, so a single-field key is just
    the degenerate composite key.

    Args:
        key: One field path, several of them, or `None`.
        owner: The aligner's class name, used in the error message.

    Returns:
        The field paths as a list, or `None` — which passes through with its
        meaning intact: key on the whole element.

    Raises:
        ValueError: If `key` is a sequence that names no field at all.
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

    An aligner decides *who pairs with whom* and nothing else: value scoring of
    the matched pairs happens later, in the array metrics. Implement `align`
    and the strategy is complete.

    Example:
        >>> from typing import Any
        >>> from structured_eval.alignment import ArrayAligner
        >>> from structured_eval.models import ArrayMatchResult, ArrayStrategy
        >>> class ReversedAligner(ArrayAligner):
        ...     def align(self, expected: list[Any],
        ...               actual: list[Any]) -> ArrayMatchResult:
        ...         n = min(len(expected), len(actual))
        ...         return ArrayMatchResult(
        ...             strategy=ArrayStrategy.BY_INDEX,
        ...             matched=[(i, len(actual) - 1 - i) for i in range(n)],
        ...             missed=list(range(n, len(expected))),
        ...             spurious=list(range(n, len(actual))),
        ...         )
        >>> ReversedAligner().align(["a", "b"], ["b", "a"]).matched
        [(0, 1), (1, 0)]
    """

    @abstractmethod
    def align(self, expected: list[Any], actual: list[Any]) -> ArrayMatchResult:
        """Pair the actual elements with the expected ones.

        Args:
            expected: The expected list.
            actual: The actual list, as the document has it.

        Returns:
            An `ArrayMatchResult` holding the `(expected_idx, actual_idx)` pairs
            plus the unmatched expected (missed) and actual (spurious) indices.
        """
