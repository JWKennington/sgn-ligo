"""DataSource uber element for unified source creation.

This module provides the DataSource class that dispatches to the appropriate
composed source class based on a source type string.

Example - Simple CLI usage:
    >>> from sgnligo.sources.datasource_v2 import DataSource
    >>>
    >>> # Create from command line arguments
    >>> source = DataSource.from_argv(name="data_source")
    >>> pipeline.connect(source.element, sink)

Example - Custom CLI arguments:
    >>> from sgnligo.sources.datasource_v2 import DataSource
    >>>
    >>> # Create parser with custom arguments
    >>> parser = DataSource.create_cli_parser()
    >>> parser.add_argument("--snr-threshold", type=float, default=8.0)
    >>>
    >>> # Parse and create source
    >>> source, args = DataSource.from_parser(parser, name="pipeline")
    >>> print(f"SNR threshold: {args.snr_threshold}")

Example - Direct instantiation:
    >>> from sgnligo.sources.datasource_v2 import DataSource
    >>>
    >>> source = DataSource(
    ...     data_source="white",
    ...     name="test",
    ...     ifos=["H1"],
    ...     sample_rate=4096,
    ...     t0=1000,
    ...     end=1010,
    ... )
    >>> pipeline.connect(source.element, sink)
"""

from __future__ import annotations

import argparse
from typing import Any, ClassVar, Dict, List, Optional, Type

from sgnts.compose import TSComposedSourceElement

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.composed_registry import (
    get_composed_source_class,
    list_composed_source_types,
)


class DataSource:
    """Unified data source that dispatches to the appropriate source class.

    This is the main entry point for CLI-based pipelines. It accepts a
    data_source type string and forwards all other parameters to the
    appropriate source class.

    For programmatic use, you can also instantiate source classes directly
    (e.g., WhiteSource, GWDataNoiseComposedSource).

    Args:
        data_source: Type of source ("white", "gwdata-noise", etc.)
        name: Name for the composed element
        **kwargs: Parameters forwarded to the specific source class

    Example:
        >>> # Direct instantiation
        >>> source = DataSource(
        ...     data_source="white",
        ...     name="test",
        ...     ifos=["H1"],
        ...     sample_rate=4096,
        ...     t0=1000,
        ...     end=1010,
        ... )
        >>>
        >>> # From CLI
        >>> source = DataSource.from_argv(name="data_source")
        >>>
        >>> # Use in pipeline
        >>> pipeline.connect(source.element, sink)
    """

    # Class metadata
    source_type: ClassVar[str] = "datasource"
    description: ClassVar[str] = "Unified data source dispatcher"

    def __init__(
        self,
        data_source: str,
        name: str = "datasource",
        **kwargs: Any,
    ) -> None:
        """Initialize DataSource.

        Args:
            data_source: Type of source ("white", "gwdata-noise", etc.)
            name: Name for the composed element
            **kwargs: Parameters forwarded to the specific source class
        """
        self.data_source = data_source
        self.name = name
        self._kwargs = kwargs

        # Look up and instantiate the inner source
        cls = get_composed_source_class(data_source)
        self._inner: ComposedSourceBase = cls(name=name, **kwargs)

    # --- Expose inner element for pipeline connection ---

    @property
    def element(self) -> TSComposedSourceElement:
        """The underlying TSComposedSourceElement for pipeline integration."""
        return self._inner.element

    @property
    def srcs(self) -> Dict[str, Any]:
        """Source pads of the composed element."""
        return self._inner.srcs

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the inner source."""
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
        """Create CLI argument parser with options for all registered sources.

        Use this when you need to add custom arguments to the parser.

        Example:
            >>> parser = DataSource.create_cli_parser()
            >>> parser.add_argument("--snr-threshold", type=float, default=8.0)
            >>> source = DataSource.from_parser(parser, name="pipeline")

        Returns:
            ArgumentParser configured with --data-source and all source options
        """
        from sgnligo.sources.datasource_v2.cli import build_composed_cli_parser

        return build_composed_cli_parser(prog=prog, description=description)

    @classmethod
    def from_argv(
        cls,
        name: str = "datasource",
        argv: Optional[List[str]] = None,
    ) -> "DataSource":
        """Create DataSource from command line arguments.

        Simple interface for when you don't need custom CLI arguments.
        Parses sys.argv (or provided argv) directly.

        Example:
            >>> # In a script called with:
            >>> # python script.py --data-source white --ifos H1
            >>> source = DataSource.from_argv(name="my_source")

        Args:
            name: Name for the composed element
            argv: Command line arguments (defaults to sys.argv[1:])

        Returns:
            DataSource instance
        """
        import sys

        from sgnligo.sources.datasource_v2.cli import (
            build_composed_cli_parser,
            check_composed_help_options,
            namespace_to_datasource_kwargs,
        )

        # Handle --list-sources and --help-source before parsing
        if check_composed_help_options(argv):
            sys.exit(0)

        parser = build_composed_cli_parser()
        args = parser.parse_args(argv)
        kwargs = namespace_to_datasource_kwargs(args)
        return cls(name=name, **kwargs)

    @classmethod
    def from_parser(
        cls,
        parser: argparse.ArgumentParser,
        name: str = "datasource",
        argv: Optional[List[str]] = None,
    ) -> tuple["DataSource", argparse.Namespace]:
        """Create DataSource from a custom argument parser.

        Use this when you've added custom arguments to the parser.
        Returns both the DataSource and the parsed args so you can
        access your custom arguments.

        Example:
            >>> parser = DataSource.create_cli_parser()
            >>> parser.add_argument("--snr-threshold", type=float, default=8.0)
            >>> source, args = DataSource.from_parser(parser, name="pipeline")
            >>> print(f"SNR threshold: {args.snr_threshold}")

        Args:
            parser: ArgumentParser (from create_cli_parser with optional additions)
            name: Name for the composed element
            argv: Command line arguments (defaults to sys.argv[1:])

        Returns:
            Tuple of (DataSource instance, parsed args namespace)
        """
        import sys

        from sgnligo.sources.datasource_v2.cli import (
            check_composed_help_options,
            namespace_to_datasource_kwargs,
        )

        # Handle --list-sources and --help-source before parsing
        if check_composed_help_options(argv):
            sys.exit(0)

        args = parser.parse_args(argv)
        kwargs = namespace_to_datasource_kwargs(args)
        return cls(name=name, **kwargs), args

    @staticmethod
    def list_sources() -> List[str]:
        """List all available source types."""
        return list_composed_source_types()

    @staticmethod
    def get_source_class(source_type: str) -> Type[ComposedSourceBase]:
        """Get the source class for a given type."""
        return get_composed_source_class(source_type)
