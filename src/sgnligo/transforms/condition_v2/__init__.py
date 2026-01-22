"""Condition V2: Dynamically dispatched composed transform for signal conditioning.

This package provides a modern, registry-based approach to signal conditioning
that mirrors the datasource_v2 pattern. It supports multiple conditioning
strategies through a single unified interface.

Quick Start:
    >>> from sgnligo.transforms.condition_v2 import Condition
    >>>
    >>> # Create a standard conditioning transform
    >>> cond = Condition(
    ...     name="condition",
    ...     condition_type="standard",
    ...     ifos=["H1", "L1"],
    ...     input_sample_rate=16384,
    ...     whiten_sample_rate=2048,
    ... )
    >>>
    >>> # Use in a pipeline
    >>> pipeline = Pipeline()
    >>> pipeline.connect(source.element, cond.element)
    >>> pipeline.connect(cond.element, sink)

Available Condition Types:
    - "standard": Standard whitening with PSD tracking
    - "zero-latency": Zero-latency AFIR whitening

Architecture:
    The package uses a registry pattern where each condition type is a
    dataclass that inherits from ComposedTransformBase and is decorated
    with @register_composed_transform.

    CLI options are composed via mixins that define both dataclass fields
    and their corresponding argparse arguments.

Modules:
    - composed_base: Abstract base class for composed transforms
    - composed_registry: Registry for transform class lookup
    - cli_mixins: CLI argument mixins for composable options
    - condition: Dispatcher class and factory function
    - transforms/: Concrete transform implementations
"""

from sgnligo.transforms.condition_v2.composed_base import ComposedTransformBase
from sgnligo.transforms.condition_v2.composed_registry import (
    get_composed_transform_class,
    get_composed_transform_registry,
    list_composed_transform_types,
    register_composed_transform,
)

# Import transforms to trigger registration
from sgnligo.transforms.condition_v2.transforms import (
    StandardCondition,
    ZeroLatencyCondition,
)

# Import dispatcher
from sgnligo.transforms.condition_v2.condition import Condition

__all__ = [
    # Base class
    "ComposedTransformBase",
    # Registry
    "register_composed_transform",
    "get_composed_transform_class",
    "list_composed_transform_types",
    "get_composed_transform_registry",
    # Concrete transforms
    "StandardCondition",
    "ZeroLatencyCondition",
    # Dispatcher
    "Condition",
]
