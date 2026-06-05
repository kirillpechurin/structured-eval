from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from structured_eval.parsers.base import ParseError


class JsonParser:
    """Parse a JSON string into a Python object.

    Accepts any valid JSON value (object, array, string, number, bool, null).
    Raises ParseError on malformed input.
    """

    def parse(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON: {exc}") from exc


class JsonlParser:
    """Parse a JSONL (JSON Lines) string into an iterator of Python objects.

    Each non-empty line must be a valid JSON value. Blank lines are skipped.
    Raises ParseError on the first malformed line, including the line number.
    """

    def parse(self, text: str) -> Iterator[Any]:
        return self._iter(text)

    def _iter(self, text: str) -> Iterator[Any]:
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"Invalid JSON on line {lineno}: {exc}") from exc
