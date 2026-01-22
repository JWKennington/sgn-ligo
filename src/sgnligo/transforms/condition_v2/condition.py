"""Condition dispatcher for unified conditioning transform creation.

This module provides the Condition class that dispatches to the appropriate
composed transform class based on a condition type string.

Example - Simple CLI usage:
    >>> from sgnligo.transforms.condition_v2 import Condition
    >>>
    >>> # Create from command line arguments
    >>> cond = Condition.from_argv(name="condition")
    >>> pipeline.connect(source.element, cond.element)
    >>> pipeline.connect(cond.element, sink)

Example - Custom CLI arguments:
    >>> from sgnligo.transforms.condition_v2 import Condition
    >>>
    >>> # Create parser with custom arguments
    >>> parser = Condition.create_cli_parser()
    >>> parser.add_argument("--snr-threshold", type=float, default=8.0)
    >>>
    >>> # Parse and create condition
    >>> cond, args = Condition.from_parser(parser, name="pipeline")
    >>> print(f"SNR threshold: {args.snr_threshold}")

Example - Direct instantiation:
    >>> from sgnligo.transforms.condition_v2 import Condition
    >>>
    >>> cond = Condition(
    ...     condition_type="standard",
    ...     name="test",
    ...     ifos=["H1"],
    ...     input_sample_rate=16384,
    ...     whiten_sample_rate=2048,
    ... )
    >>> pipeline.connect(source.element, cond.element)
"""

from __future__ import annotations

import argparse
from typing import Any, ClassVar, Dict, List, Optional, Type

from sgnts.compose import TSComposedTransformElement

from sgnligo.transforms.condition_v2.composed_base import ComposedTransformBase
from sgnligo.transforms.condition_v2.composed_registry import (
    get_composed_transform_class,
    list_composed_transform_types,
)


class Condition:
    """Unified conditioning transform that dispatches to the appropriate class.

    This is the main entry point for CLI-based pipelines. It accepts a
    condition_type string and forwards all other parameters to the
    appropriate transform class.

    For programmatic use, you can also instantiate transform classes directly
    (e.g., StandardCondition, ZeroLatencyCondition).

    Args:
        condition_type: Type of condition ("standard", "zero-latency", etc.)
        name: Name for the composed element
        **kwargs: Parameters forwarded to the specific transform class

    Example:
        >>> # Direct instantiation
        >>> cond = Condition(
        ...     condition_type="standard",
        ...     name="test",
        ...     ifos=["H1"],
        ...     input_sample_rate=16384,
        ...     whiten_sample_rate=2048,
        ... )
        >>>
        >>> # From CLI
        >>> cond = Condition.from_argv(name="condition")
        >>>
        >>> # Use in pipeline
        >>> pipeline.connect(source.element, cond.element)
    """

    # Class metadata
    transform_type: ClassVar[str] = "condition"
    description: ClassVar[str] = "Unified conditioning transform dispatcher"

    def __init__(
        self,
        condition_type: str,
        name: str = "condition",
        **kwargs: Any,
    ) -> None:
        """Initialize Condition.

        Args:
            condition_type: Type of condition ("standard", "zero-latency", etc.)
            name: Name for the composed element
            **kwargs: Parameters forwarded to the specific transform class
        """
        self.condition_type = condition_type
        self.name = name
        self._kwargs = kwargs

        # Look up and instantiate the inner transform
        cls = get_composed_transform_class(condition_type)
        self._inner: ComposedTransformBase = cls(name=name, **kwargs)

    # --- Expose inner element for pipeline connection ---

    @property
    def element(self) -> TSComposedTransformElement:
        """The underlying TSComposedTransformElement for pipeline integration."""
        return self._inner.element

    @property
    def srcs(self) -> Dict[str, Any]:
        """Source pads of the composed element."""
        return self._inner.srcs

    @property
    def snks(self) -> Dict[str, Any]:
        """Sink pads of the composed element."""
        return self._inner.snks

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the inner transform."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    # --- CLI Support ---

    @classmethod
    def create_cli_parser(
        cls,
        prog: Optional[str] = None,
        description: Optional[str] = None,
    ) -> argparse.ArgumentParser:
        """Create CLI argument parser with options for all registered transforms.

        Use this when you need to add custom arguments to the parser.

        Example:
            >>> parser = Condition.create_cli_parser()
            >>> parser.add_argument("--snr-threshold", type=float, default=8.0)
            >>> cond = Condition.from_parser(parser, name="pipeline")

        Returns:
            ArgumentParser configured with --condition-type and all transform options
        """
        from sgnligo.transforms.condition_v2.cli import build_condition_cli_parser

        return build_condition_cli_parser(prog=prog, description=description)

    @classmethod
    def from_argv(
        cls,
        name: str = "condition",
        argv: Optional[List[str]] = None,
    ) -> "Condition":
        """Create Condition from command line arguments.

        Simple interface for when you don't need custom CLI arguments.
        Parses sys.argv (or provided argv) directly.

        Example:
            >>> # In a script called with:
            >>> # python script.py --condition-type standard --ifos H1
            >>> cond = Condition.from_argv(name="my_condition")

        Args:
            name: Name for the composed element
            argv: Command line arguments (defaults to sys.argv[1:])

        Returns:
            Condition instance
        """
        import sys

        from sgnligo.transforms.condition_v2.cli import (
            build_condition_cli_parser,
            check_condition_help_options,
            namespace_to_condition_kwargs,
        )

        # Handle --list-conditions and --help-condition before parsing
        if check_condition_help_options(argv):
            sys.exit(0)

        parser = build_condition_cli_parser()
        args = parser.parse_args(argv)
        kwargs = namespace_to_condition_kwargs(args)
        return cls(name=name, **kwargs)

    @classmethod
    def from_parser(
        cls,
        parser: argparse.ArgumentParser,
        name: str = "condition",
        argv: Optional[List[str]] = None,
    ) -> tuple["Condition", argparse.Namespace]:
        """Create Condition from a custom argument parser.

        Use this when you've added custom arguments to the parser.
        Returns both the Condition and the parsed args so you can
        access your custom arguments.

        Example:
            >>> parser = Condition.create_cli_parser()
            >>> parser.add_argument("--snr-threshold", type=float, default=8.0)
            >>> cond, args = Condition.from_parser(parser, name="pipeline")
            >>> print(f"SNR threshold: {args.snr_threshold}")

        Args:
            parser: ArgumentParser (from create_cli_parser with optional additions)
            name: Name for the composed element
            argv: Command line arguments (defaults to sys.argv[1:])

        Returns:
            Tuple of (Condition instance, parsed args namespace)
        """
        import sys

        from sgnligo.transforms.condition_v2.cli import (
            check_condition_help_options,
            namespace_to_condition_kwargs,
        )

        # Handle --list-conditions and --help-condition before parsing
        if check_condition_help_options(argv):
            sys.exit(0)

        args = parser.parse_args(argv)
        kwargs = namespace_to_condition_kwargs(args)
        return cls(name=name, **kwargs), args

    @staticmethod
    def list_conditions() -> List[str]:
        """List all available condition types."""
        return list_composed_transform_types()

    @staticmethod
    def get_condition_class(condition_type: str) -> Type[ComposedTransformBase]:
        """Get the condition class for a given type."""
        return get_composed_transform_class(condition_type)
