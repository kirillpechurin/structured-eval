"""YAML parser — requires the `yaml` extra."""

from typing import Any

from structured_eval.formats.base import ParseError


class YamlParser:
    r"""Parse a YAML string into a Python object.

    Uses `yaml.safe_load`, so arbitrary Python object construction is disabled.
    PyYAML is imported lazily, which keeps the core package importable without
    the `yaml` extra.

    Example:
        >>> from structured_eval.formats import YamlParser
        >>> YamlParser().parse("total: 100\nvendor: Acme")
        {'total': 100, 'vendor': 'Acme'}
    """

    def parse(self, text: str) -> Any:
        """Read `text` as a YAML document.

        Args:
            text: The YAML document.

        Returns:
            The loaded value: a dict, a list or a scalar.

        Raises:
            ImportError: If PyYAML is not installed.
            ParseError: If `text` is not valid YAML.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML parsing. Install it with: pip install pyyaml"
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ParseError(f"Invalid YAML: {exc}") from exc
