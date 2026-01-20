"""Registry for dataclass-based composed source classes.

This module provides registration and lookup for the new dataclass-based
source classes that inherit from ComposedSourceBase.

Example:
    >>> from sgnligo.sources.datasource_v2.composed_registry import (
    ...     register_composed_source,
    ...     get_composed_source_class,
    ... )
    >>>
    >>> @register_composed_source
    >>> @dataclass
    >>> class MySource(ComposedSourceBase):
    ...     source_type: ClassVar[str] = "my-source"
    ...     ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from sgnligo.sources.composed_base import ComposedSourceBase

# Global registry of composed source classes
_COMPOSED_REGISTRY: Dict[str, Type[ComposedSourceBase]] = {}


def register_composed_source(
    cls: Type[ComposedSourceBase],
) -> Type[ComposedSourceBase]:
    """Decorator to register a composed source class.

    Args:
        cls: The composed source class to register

    Returns:
        The same class (unchanged)

    Raises:
        ValueError: If the source_type is empty or already registered

    Example:
        @register_composed_source
        @dataclass
        class WhiteSource(ComposedSourceBase):
            source_type: ClassVar[str] = "white"
            ...
    """
    source_type = cls.source_type
    if not source_type:
        raise ValueError(f"Class {cls.__name__} must define source_type")
    if source_type in _COMPOSED_REGISTRY:
        raise ValueError(
            f"Source type '{source_type}' is already registered "
            f"(by {_COMPOSED_REGISTRY[source_type].__name__})"
        )
    _COMPOSED_REGISTRY[source_type] = cls
    return cls


def get_composed_source_class(source_type: str) -> Type[ComposedSourceBase]:
    """Get the composed source class for a given type string.

    Args:
        source_type: The source type identifier

    Returns:
        The composed source class

    Raises:
        ValueError: If the source type is not registered
    """
    if source_type not in _COMPOSED_REGISTRY:
        available = ", ".join(sorted(_COMPOSED_REGISTRY.keys()))
        raise ValueError(f"Unknown source type '{source_type}'. Available: {available}")
    return _COMPOSED_REGISTRY[source_type]


def list_composed_source_types() -> List[str]:
    """List all registered composed source types.

    Returns:
        Sorted list of registered source type names
    """
    return sorted(_COMPOSED_REGISTRY.keys())


def get_composed_registry() -> Dict[str, Type[ComposedSourceBase]]:
    """Get the full composed source registry.

    Returns:
        Dict mapping source type to class
    """
    return _COMPOSED_REGISTRY.copy()
