"""Rendering of a field's value for reports and judge prompts."""

import json
from typing import Any


def render_value(value: Any) -> str:
    """Show a field's value the way it was extracted, ``null`` included.

    JSON rather than ``str``: the judge has to tell the string ``"100"`` from
    the number ``100`` and an empty string from a null, and quoting is what
    carries that.
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)
