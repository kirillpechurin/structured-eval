"""JSON and JSONL parsers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from structured_eval.formats.base import ParseError

if TYPE_CHECKING:
    from collections.abc import Iterator


class JsonParser:
    """Parse a JSON string into a Python object.

    Accepts any valid JSON value — object, array, string, number, bool, null.

    Example:
        >>> from structured_eval.formats import JsonParser
        >>> JsonParser().parse('{"total": 100, "items": [1, 2]}')
        {'total': 100, 'items': [1, 2]}
        >>> JsonParser().parse("[1, 2]")
        [1, 2]
    """

    def parse(self, text: str) -> object:
        """Read `text` as a single JSON value.

        Args:
            text: The JSON document.

        Returns:
            The decoded value: a dict, a list or a scalar.

        Raises:
            ParseError: If `text` is not valid JSON.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON: {exc}") from exc


class JsonlParser:
    r"""Parse a JSONL (JSON Lines) string into an iterator of Python objects.

    Each non-empty line must be a valid JSON value; blank lines are skipped.
    Lines are decoded lazily, so a malformed one is reported when iteration
    reaches it, not when `parse` returns.

    Example:
        >>> from structured_eval.formats import JsonlParser
        >>> list(JsonlParser().parse('{"id": 1}\n\n{"id": 2}\n'))
        [{'id': 1}, {'id': 2}]
    """

    def parse(self, text: str) -> Iterator[object]:
        """Read `text` as one JSON value per non-empty line.

        Args:
            text: The JSONL document.

        Returns:
            A lazy iterator over the decoded values, in line order. Iterating it
            raises `ParseError` on the first malformed line, quoting its number.
        """
        return self._iter(text)

    def _iter(self, text: str) -> Iterator[object]:
        """Decode the lines one at a time, skipping the blank ones."""
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"Invalid JSON on line {lineno}: {exc}") from exc
