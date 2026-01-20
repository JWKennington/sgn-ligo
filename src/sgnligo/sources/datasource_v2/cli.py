"""CLI support for composed data sources.

This module provides CLI argument parsing and help generation for
the dataclass-based composed source classes.

CLI arguments are defined by mixin classes that sources inherit from.
The `build_composed_cli_parser()` function aggregates CLI arguments
from all registered sources by walking their MRO and collecting
arguments from mixins.

Example:
    >>> from sgnligo.sources.datasource_v2.cli import (
    ...     build_composed_cli_parser,
    ...     check_composed_help_options,
    ... )
    >>>
    >>> if check_composed_help_options():
    ...     sys.exit(0)
    >>>
    >>> parser = build_composed_cli_parser()
    >>> args = parser.parse_args()
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Any, Dict, List, Optional, Set, Type

from sgnligo.sources.datasource_v2.cli_mixins import CLIMixinProtocol
from sgnligo.sources.datasource_v2.composed_registry import (
    _COMPOSED_REGISTRY,
    get_composed_source_class,
    list_composed_source_types,
)


def get_source_required_fields(cls: Type) -> List[str]:
    """Get required fields for a source class (fields without defaults).

    Args:
        cls: The source class to introspect

    Returns:
        List of required field names
    """
    required = []
    for f in dataclasses.fields(cls):
        if f.name in ("name", "_composed"):
            continue
        # Field is required if it has no default and no default_factory
        has_default = f.default is not dataclasses.MISSING
        has_factory = f.default_factory is not dataclasses.MISSING
        if not has_default and not has_factory:
            required.append(f.name)
    return required


def get_source_optional_fields(cls: Type) -> Dict[str, Any]:
    """Get optional fields with their defaults for a source class.

    Args:
        cls: The source class to introspect

    Returns:
        Dict mapping field name to default value
    """
    optional = {}
    for f in dataclasses.fields(cls):
        if f.name in ("name", "_composed"):
            continue
        if f.default is not dataclasses.MISSING:
            optional[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            optional[f.name] = f.default_factory()
    return optional


def format_composed_source_help(source_type: str) -> str:
    """Generate detailed help for a specific source type.

    Args:
        source_type: The source type to show help for

    Returns:
        Formatted help string
    """
    cls = get_composed_source_class(source_type)

    lines = [
        f"usage: prog --data-source {source_type} [options]",
        "",
        f"{source_type}: {cls.description}",
        "",
    ]

    # Required fields
    required = get_source_required_fields(cls)
    if required:
        lines.append("Required Options:")
        for name in required:
            cli_name = name.replace("_", "-")
            lines.append(f"  --{cli_name}")
        lines.append("")

    # Optional fields
    optional = get_source_optional_fields(cls)
    if optional:
        lines.append("Optional Options:")
        for name, default in optional.items():
            cli_name = name.replace("_", "-")
            if default is False:
                lines.append(f"  --{cli_name}")
            elif default is not None:
                lines.append(f"  --{cli_name} (default: {default})")
            else:
                lines.append(f"  --{cli_name}")
        lines.append("")

    # Add docstring notes if available
    if cls.__doc__:
        # Extract just the description part (first paragraph)
        doc_lines = cls.__doc__.strip().split("\n\n")[0].split("\n")
        if doc_lines:
            lines.append("Description:")
            for doc_line in doc_lines:
                lines.append(f"  {doc_line.strip()}")

    return "\n".join(lines)


def format_composed_source_list() -> str:
    """Generate list of all available sources grouped by type.

    Returns:
        Formatted string listing all sources
    """
    lines = ["Available data sources:", ""]

    offline = []
    realtime = []

    for source_type in sorted(_COMPOSED_REGISTRY.keys()):
        cls = _COMPOSED_REGISTRY[source_type]
        desc = f"  {source_type:30} {cls.description}"

        if "realtime" in source_type.lower():
            realtime.append(desc)
        else:
            offline.append(desc)

    if offline:
        lines.append("Offline Sources:")
        lines.extend(offline)
        lines.append("")

    if realtime:
        lines.append("Real-time Sources:")
        lines.extend(realtime)
        lines.append("")

    lines.append("Use --help-source <name> for detailed options.")
    return "\n".join(lines)


def check_composed_help_options(argv: Optional[List[str]] = None) -> bool:
    """Check for --list-sources and --help-source before full parsing.

    Call this before parse_args() to handle help options that don't require
    --data-source to be specified.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        True if help was handled (caller should exit), False otherwise.
    """
    if argv is None:
        argv = sys.argv[1:]

    if "--list-sources" in argv:
        print(format_composed_source_list())
        return True

    if "--help-source" in argv:
        try:
            idx = argv.index("--help-source")
            source_type = argv[idx + 1]
            if source_type in _COMPOSED_REGISTRY:
                print(format_composed_source_help(source_type))
                return True
            else:
                available = ", ".join(sorted(_COMPOSED_REGISTRY.keys()))
                print(f"Unknown source type '{source_type}'. Available: {available}")
                return True
        except IndexError:
            print("--help-source requires a source type argument")
            return True

    return False


def build_composed_cli_parser(
    prog: Optional[str] = None,
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Build CLI parser by aggregating arguments from source mixins.

    This function walks the MRO of all registered source classes and
    collects CLI arguments from mixins that implement the CLIMixinProtocol.
    Duplicate arguments (same arg defined by multiple mixins) raise an error.

    Args:
        prog: Program name for help text
        description: Description for help text

    Returns:
        ArgumentParser configured with all source options

    Raises:
        ValueError: If duplicate CLI arguments are detected
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or "Process gravitational wave data",
    )

    # Main dispatch option
    source_types = list_composed_source_types()
    parser.add_argument(
        "--data-source",
        required=True,
        choices=source_types,
        help="Type of data source to use",
    )

    # Help options (always available)
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available source types",
    )
    parser.add_argument(
        "--help-source",
        metavar="SOURCE",
        help="Show help for a specific source type",
    )

    # First pass: collect all CLI mixins from all registered sources
    # We need to process mixins with MORE args first (supersets before subsets)
    # to ensure variant mixins like StateVectorOptionsMixin (4 args) are processed
    # before StateVectorOnDictOnlyMixin (3 args), so all args get registered.
    cli_mixins: List[Type] = []
    seen_mixins: Set[Type] = set()

    for _source_type, cls in _COMPOSED_REGISTRY.items():
        for base in cls.__mro__:
            if base in seen_mixins:
                continue

            # Skip if not a CLI mixin (doesn't define add_cli_arguments directly)
            if "add_cli_arguments" not in base.__dict__:
                continue
            if "get_cli_arg_names" not in base.__dict__:
                continue  # pragma: no cover

            # Skip the protocol class itself
            if base is CLIMixinProtocol:
                continue  # pragma: no cover

            cli_mixins.append(base)
            seen_mixins.add(base)

    # Sort mixins by arg count descending - supersets first
    cli_mixins.sort(key=lambda m: len(m.get_cli_arg_names()), reverse=True)

    # Second pass: add arguments, skipping mixins with overlapping args
    added_args: Dict[str, Type] = {}
    for mixin in cli_mixins:
        new_args = mixin.get_cli_arg_names()

        # Check if any args from this mixin are already added
        # If so, skip this mixin entirely - it's a variant mixin that shares
        # some args with another (e.g., GPSOptionsMixin vs GPSOptionsOptionalMixin,
        # or StateVectorOptionsMixin vs StateVectorOnDictOnlyMixin).
        # Since we sorted by arg count, the superset was already processed.
        any_args_exist = any(arg in added_args for arg in new_args)
        if any_args_exist:
            continue

        # Register all args from this mixin
        for arg in new_args:
            added_args[arg] = mixin

        # Add the arguments to the parser
        mixin.add_cli_arguments(parser)

    return parser


def namespace_to_datasource_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to DataSource kwargs.

    This function walks the MRO of the selected source class and calls
    `process_cli_args()` on each mixin to convert CLI arguments to field values.

    Args:
        args: Parsed argparse namespace

    Returns:
        Dict of kwargs for DataSource
    """
    kwargs: Dict[str, Any] = {
        "data_source": args.data_source,
    }

    # Get the source class to walk its MRO for process_cli_args
    source_type = args.data_source
    if source_type in _COMPOSED_REGISTRY:
        cls = _COMPOSED_REGISTRY[source_type]
        processed_classes: Set[Type] = set()

        # Walk MRO and call process_cli_args on each mixin
        for base in cls.__mro__:
            if base in processed_classes:
                continue  # pragma: no cover

            if not hasattr(base, "process_cli_args"):
                continue

            if base is CLIMixinProtocol:
                continue  # pragma: no cover

            mixin_kwargs = base.process_cli_args(args)
            kwargs.update(mixin_kwargs)

            processed_classes.add(base)

    return kwargs
