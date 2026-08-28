"""render_value — how a field's value is shown to an LLM judge.

JSON rather than `str`, because the judge has to tell the string `"100"`
from the number `100`, and an empty string from a null: quoting is what
carries that distinction into the prompt.
"""

from datetime import date
from typing import Any

import pytest

from structured_eval.metrics.utils.value import render_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        ("100", '"100"'),
        (100, "100"),
        (100.5, "100.5"),
        ("", '""'),
        (None, "null"),
        (True, "true"),
        (["a", 1], '["a", 1]'),
        ({"sku": "A-1"}, '{"sku": "A-1"}'),
        ("Grüße", '"Grüße"'),  # not escaped to \uXXXX — the judge reads the text
    ],
    ids=[
        "numeric-string",
        "int",
        "float",
        "empty-string",
        "null",
        "bool",
        "list",
        "dict",
        "non-ascii",
    ],
)
def test_values_are_rendered_as_json(value: Any, rendered: str) -> None:
    assert render_value(value) == rendered


def test_a_value_json_cannot_encode_falls_back_to_its_text() -> None:
    # A parsed document can hold whatever the format produced — a YAML date, for
    # one. Showing it as text beats failing the whole evaluation over a value
    # the judge could have read perfectly well.
    assert render_value(date(2025, 1, 31)) == '"2025-01-31"'
