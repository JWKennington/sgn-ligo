"""CLI support for composed condition transforms.

This module provides CLI argument parsing and help generation for
the dataclass-based composed transform classes.

CLI arguments are defined by mixin classes that transforms inherit from.
The `build_condition_cli_parser()` function aggregates CLI arguments
from all registered transforms by walking their MRO and collecting
arguments from mixins.

Example:
    >>> from sgnligo.transforms.condition_v2.cli import (
    ...     build_condition_cli_parser,
    ...     check_condition_help_options,
    ... )
    >>>
    >>> if check_condition_help_options():
    ...     sys.exit(0)
    >>>
    >>> parser = build_condition_cli_parser()
    >>> args = parser.parse_args()
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Any, Dict, List, Optional, Set, Type

from sgnligo.transforms.condition_v2.cli_mixins import CLIMixinProtocol
from sgnligo.transforms.condition_v2.composed_registry import (
    _COMPOSED_TRANSFORM_REGISTRY,
    get_composed_transform_class,
    list_composed_transform_types,
)


def get_transform_required_fields(cls: Type) -> List[str]:
    """Get required fields for a transform class (fields without defaults).

    Args:
        cls: The transform class to introspect

    Returns:
        List of required field names
    """
    required = []
    for f in dataclasses.fields(cls):
        if f.name in ("name", "_composed", "_also_expose_pads", "_ref_psds"):
            continue
        # Field is required if it has no default and no default_factory
        has_default = f.default is not dataclasses.MISSING
        has_factory = f.default_factory is not dataclasses.MISSING
        if not has_default and not has_factory:
            required.append(f.name)
    return required


def get_transform_optional_fields(cls: Type) -> Dict[str, Any]:
    """Get optional fields with their defaults for a transform class.

    Args:
        cls: The transform class to introspect

    Returns:
        Dict mapping field name to default value
    """
    optional = {}
    for f in dataclasses.fields(cls):
        if f.name in ("name", "_composed", "_also_expose_pads", "_ref_psds"):
            continue
        if f.default is not dataclasses.MISSING:
            optional[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            optional[f.name] = f.default_factory()
    return optional


def format_condition_help(condition_type: str) -> str:
    """Generate detailed help for a specific condition type.

    Args:
        condition_type: The condition type to show help for

    Returns:
        Formatted help string
    """
    cls = get_composed_transform_class(condition_type)

    lines = [
        f"usage: prog --condition-type {condition_type} [options]",
        "",
        f"{condition_type}: {cls.description}",
        "",
    ]

    # Required fields
    required = get_transform_required_fields(cls)
    if required:
        lines.append("Required Options:")
        for name in required:
            cli_name = name.replace("_", "-")
            lines.append(f"  --{cli_name}")
        lines.append("")

    # Optional fields
    optional = get_transform_optional_fields(cls)
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


def format_condition_list() -> str:
    """Generate list of all available condition types.

    Returns:
        Formatted string listing all conditions
    """
    lines = ["Available condition types:", ""]

    standard = []
    zero_latency = []

    for condition_type in sorted(_COMPOSED_TRANSFORM_REGISTRY.keys()):
        cls = _COMPOSED_TRANSFORM_REGISTRY[condition_type]
        desc = f"  {condition_type:30} {cls.description}"

        if "zero" in condition_type.lower():
            zero_latency.append(desc)
        else:
            standard.append(desc)

    if standard:
        lines.append("Standard Conditioning:")
        lines.extend(standard)
        lines.append("")

    if zero_latency:
        lines.append("Zero-Latency Conditioning:")
        lines.extend(zero_latency)
        lines.append("")

    lines.append("Use --help-condition <name> for detailed options.")
    return "\n".join(lines)


def check_condition_help_options(argv: Optional[List[str]] = None) -> bool:
    """Check for --list-conditions and --help-condition before full parsing.

    Call this before parse_args() to handle help options that don't require
    --condition-type to be specified.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        True if help was handled (caller should exit), False otherwise.
    """
    if argv is None:
        argv = sys.argv[1:]

    if "--list-conditions" in argv:
        print(format_condition_list())
        return True

    if "--help-condition" in argv:
        try:
            idx = argv.index("--help-condition")
            condition_type = argv[idx + 1]
            if condition_type in _COMPOSED_TRANSFORM_REGISTRY:
                print(format_condition_help(condition_type))
                return True
            else:
                available = ", ".join(sorted(_COMPOSED_TRANSFORM_REGISTRY.keys()))
                print(
                    f"Unknown condition type '{condition_type}'. Available: {available}"
                )
                return True
        except IndexError:
            print("--help-condition requires a condition type argument")
            return True

    return False


def build_condition_cli_parser(
    prog: Optional[str] = None,
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Build CLI parser by aggregating arguments from transform mixins.

    This function walks the MRO of all registered transform classes and
    collects CLI arguments from mixins that implement the CLIMixinProtocol.
    Duplicate arguments (same arg defined by multiple mixins) are handled
    by processing mixins with more args first (supersets before subsets).

    Args:
        prog: Program name for help text
        description: Description for help text

    Returns:
        ArgumentParser configured with all transform options

    Raises:
        ValueError: If duplicate CLI arguments are detected
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or "Process gravitational wave data conditioning",
    )

    # Main dispatch option
    condition_types = list_composed_transform_types()
    parser.add_argument(
        "--condition-type",
        required=True,
        choices=condition_types,
        help="Type of conditioning transform to use",
    )

    # Help options (always available)
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available condition types",
    )
    parser.add_argument(
        "--help-condition",
        metavar="CONDITION",
        help="Show help for a specific condition type",
    )

    # First pass: collect all CLI mixins from all registered transforms
    # We need to process mixins with MORE args first (supersets before subsets)
    cli_mixins: List[Type] = []
    seen_mixins: Set[Type] = set()

    for _condition_type, cls in _COMPOSED_TRANSFORM_REGISTRY.items():
        for base in cls.__mro__:
            if base in seen_mixins:
                continue

            # Skip if not a CLI mixin (doesn't define add_cli_arguments directly)
            if "add_cli_arguments" not in base.__dict__:
                continue
            if "get_cli_arg_names" not in base.__dict__:
                continue  # pragma: no cover

            assert base is not CLIMixinProtocol, "Protocol should not be in MRO"
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
        # some args with another.
        any_args_exist = any(arg in added_args for arg in new_args)
        if any_args_exist:
            continue

        # Register all args from this mixin
        for arg in new_args:
            added_args[arg] = mixin

        # Add the arguments to the parser
        mixin.add_cli_arguments(parser)

    return parser


def add_condition_options_to_parser(
    parser: argparse.ArgumentParser,
    include_condition_type: bool = False,
) -> None:
    """Add condition options to an existing argument parser.

    Use this when you have an existing parser (e.g., from
    DataSource.create_cli_parser()) and want to add condition-related
    options without creating a new parser.

    This allows embedding condition options in applications that don't need the
    full --condition-type dispatch mechanism.

    Args:
        parser: ArgumentParser to add options to
        include_condition_type: If True, add --condition-type as a required
            argument. If False (default), options are added without requiring
            a condition type to be specified.

    Example:
        >>> from sgnligo.sources.datasource_v2 import DataSource
        >>> from sgnligo.transforms.condition_v2.cli import (
        ...     add_condition_options_to_parser,
        ... )
        >>>
        >>> # Start with DataSource parser
        >>> parser = DataSource.create_cli_parser()
        >>>
        >>> # Add condition options
        >>> add_condition_options_to_parser(parser)
        >>>
        >>> # Add application-specific options
        >>> parser.add_argument("--inj-file", required=True)
        >>>
        >>> args = parser.parse_args()
    """
    if include_condition_type:
        condition_types = list_composed_transform_types()
        parser.add_argument(
            "--condition-type",
            choices=condition_types,
            help="Type of conditioning transform to use",
        )

    # Collect all CLI mixins from registered transforms
    # (same logic as build_condition_cli_parser)
    cli_mixins: List[Type] = []
    seen_mixins: Set[Type] = set()

    for _condition_type, cls in _COMPOSED_TRANSFORM_REGISTRY.items():
        for base in cls.__mro__:
            if base in seen_mixins:
                continue
            if "add_cli_arguments" not in base.__dict__:
                continue
            if "get_cli_arg_names" not in base.__dict__:
                continue
            assert base is not CLIMixinProtocol, "Protocol should not be in MRO"
            cli_mixins.append(base)
            seen_mixins.add(base)

    # Sort mixins by arg count descending - supersets first
    cli_mixins.sort(key=lambda m: len(m.get_cli_arg_names()), reverse=True)

    # Add arguments, skipping mixins with overlapping args
    added_args: Dict[str, Type] = {}
    for mixin in cli_mixins:
        new_args = mixin.get_cli_arg_names()
        any_args_exist = any(arg in added_args for arg in new_args)
        if any_args_exist:
            continue
        for arg in new_args:
            added_args[arg] = mixin
        mixin.add_cli_arguments(parser)


def namespace_to_condition_kwargs(
    args: argparse.Namespace,
    condition_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert argparse namespace to Condition kwargs.

    This function walks the MRO of the selected transform class and calls
    `process_cli_args()` on each mixin to convert CLI arguments to field values.

    Args:
        args: Parsed argparse namespace
        condition_type: Override condition type. If not provided, uses
            args.condition_type. Useful for apps that hardcode the type.

    Returns:
        Dict of kwargs for Condition
    """
    # Determine condition type - explicit parameter overrides args
    if condition_type is None:
        condition_type = getattr(args, "condition_type", "standard")

    kwargs: Dict[str, Any] = {
        "condition_type": condition_type,
    }

    # Get the transform class to walk its MRO for process_cli_args
    if condition_type in _COMPOSED_TRANSFORM_REGISTRY:
        cls = _COMPOSED_TRANSFORM_REGISTRY[condition_type]
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
