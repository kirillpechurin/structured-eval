from __future__ import annotations

from typing import Any

from structured_eval.parsers.base import ParseError
from structured_eval.parsers.json_parser import JsonParser

_json = JsonParser()


def parse(raw: Any) -> tuple[Any, str | None]:
    """Coerce raw input into a Python value without raising.

    Already-structured input (dict/list/None/scalars) passes through. A string
    is parsed as JSON. Returns ``(value, None)`` on success or ``(None, message)``
    on a parse error, so the engine can surface it in ``EvalReport``.
    """
    if isinstance(raw, str):
        try:
            return _json.parse(raw), None
        except ParseError as exc:
            return None, str(exc)
    return raw, None
