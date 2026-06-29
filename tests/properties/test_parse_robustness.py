"""Case G — the parse phase never throws; it reports.

LLM output is untrusted text. A malformed document must surface as a structured
``parse_error`` on the report, never as an exception out of ``evaluate``. We
fuzz with random byte-ish strings and known-pathological JSON-ish fragments and
assert the contract holds every time.
"""

import random
import string
from typing import Any

import pytest

from structured_eval import EvalConfig, ObjectF1, evaluate

from .conftest import SEEDS

pytestmark = pytest.mark.property

# Hand-picked nasties: truncated JSON, trailing commas, lone braces, NaN, etc.
PATHOLOGICAL = [
    "",
    "   ",
    "{",
    "}",
    "[",
    '{"a": }',
    '{"a": 1,}',
    '{"a": 1',
    "not json at all",
    "{'a': 1}",  # single quotes
    "NaN",
    "\x00\x01\x02",
    "{" * 200,  # deep / unbalanced
    '{"a": "unterminated',
]


@pytest.mark.parametrize("bad", PATHOLOGICAL, ids=range(len(PATHOLOGICAL)))
def test_pathological_strings_report_not_raise(bad: Any) -> None:
    report = evaluate(bad, {"a": 1}, config=EvalConfig(metrics=[ObjectF1()]))
    if report.parse_error:
        assert report.parse_error_message
    # If a fragment happens to parse (e.g. "NaN" via a lenient path), that's fine
    # too — the only forbidden outcome is an exception, which we never reach here.


@pytest.mark.parametrize("seed", SEEDS)
def test_random_garbage_reports_not_raise(seed: Any) -> None:
    rng = random.Random(seed)
    n = rng.randint(0, 40)
    alphabet = string.printable + "{}[]:,\"'"
    bad = "".join(rng.choice(alphabet) for _ in range(n))
    # Must not raise regardless of content.
    report = evaluate(bad, {"a": 1}, config=EvalConfig(metrics=[ObjectF1()]))
    assert isinstance(report.parse_error, bool)
    if report.parse_error:
        assert report.score == 0.0 or report.score is None
