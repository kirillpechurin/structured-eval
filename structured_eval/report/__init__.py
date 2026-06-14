"""Report rendering and serialization (Stage 11): console + JSON/diff."""

from structured_eval.report.console import render
from structured_eval.report.serialize import report_from_dict, report_to_dict

__all__ = ["render", "report_to_dict", "report_from_dict"]
