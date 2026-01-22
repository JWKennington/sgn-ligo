"""Registry for dataclass-based composed transform classes.

This module provides registration and lookup for the new dataclass-based
transform classes that inherit from ComposedTransformBase.

Example:
    >>> from sgnligo.transforms.condition_v2.composed_registry import (
    ...     register_composed_transform,
    ...     get_composed_transform_class,
    ... )
    >>>
    >>> @register_composed_transform
    >>> @dataclass(kw_only=True)
    >>> class MyTransform(ComposedTransformBase):
    ...     transform_type: ClassVar[str] = "my-transform"
    ...     ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from sgnligo.transforms.condition_v2.composed_base import ComposedTransformBase

# Global registry of composed transform classes
_COMPOSED_TRANSFORM_REGISTRY: Dict[str, Type[ComposedTransformBase]] = {}


def register_composed_transform(
    cls: Type[ComposedTransformBase],
) -> Type[ComposedTransformBase]:
    """Decorator to register a composed transform class.

    Args:
        cls: The composed transform class to register

    Returns:
        The same class (unchanged)

    Raises:
        ValueError: If the transform_type is empty or already registered

    Example:
        @register_composed_transform
        @dataclass(kw_only=True)
        class StandardCondition(ComposedTransformBase):
            transform_type: ClassVar[str] = "standard"
            ...
    """
    transform_type = cls.transform_type
    if not transform_type:
        raise ValueError(f"Class {cls.__name__} must define transform_type")
    if transform_type in _COMPOSED_TRANSFORM_REGISTRY:
        raise ValueError(
            f"Transform type '{transform_type}' is already registered "
            f"(by {_COMPOSED_TRANSFORM_REGISTRY[transform_type].__name__})"
        )
    _COMPOSED_TRANSFORM_REGISTRY[transform_type] = cls
    return cls


def get_composed_transform_class(transform_type: str) -> Type[ComposedTransformBase]:
    """Get the composed transform class for a given type string.

    Args:
        transform_type: The transform type identifier

    Returns:
        The composed transform class

    Raises:
        ValueError: If the transform type is not registered
    """
    if transform_type not in _COMPOSED_TRANSFORM_REGISTRY:
        available = ", ".join(sorted(_COMPOSED_TRANSFORM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown transform type '{transform_type}'. Available: {available}"
        )
    return _COMPOSED_TRANSFORM_REGISTRY[transform_type]


def list_composed_transform_types() -> List[str]:
    """List all registered composed transform types.

    Returns:
        Sorted list of registered transform type names
    """
    return sorted(_COMPOSED_TRANSFORM_REGISTRY.keys())


def get_composed_transform_registry() -> Dict[str, Type[ComposedTransformBase]]:
    """Get the full composed transform registry.

    Returns:
        Dict mapping transform type to class
    """
    return _COMPOSED_TRANSFORM_REGISTRY.copy()
