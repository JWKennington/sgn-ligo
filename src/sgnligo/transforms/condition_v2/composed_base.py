"""Base class for composed transform elements.

This module provides an abstract base class for creating composed transform elements
that combine multiple internal elements into a single transform. Subclasses define
their parameters as dataclass fields and implement _build() to wire up the
internal elements.

Usage with pipelines:
    Composed transforms wrap a TSComposedTransformElement internally. To use them
    with Pipeline.connect(), access the inner element via the .element property:

    >>> pipeline = Pipeline()
    >>> pipeline.connect(source.element, transform.element)
    >>> pipeline.connect(transform.element, sink)

Example:
    >>> from dataclasses import dataclass
    >>> from sgnligo.transforms.condition_v2.composed_base import ComposedTransformBase
    >>>
    >>> @dataclass(kw_only=True)
    ... class MyTransform(ComposedTransformBase):
    ...     transform_type: ClassVar[str] = "my-transform"
    ...     description: ClassVar[str] = "My custom transform"
    ...
    ...     ifos: list[str]
    ...     sample_rate: int
    ...
    ...     def _build(self) -> TSComposedTransformElement:
    ...         compose = TSCompose()
    ...         # ... wire up elements
    ...         return compose.as_transform(name=self.name)
    >>>
    >>> transform = MyTransform(name="test", ifos=["H1"], sample_rate=4096)
    >>> print(transform.srcs)  # Access source pads
    >>> print(transform.snks)  # Access sink pads
"""

from __future__ import annotations

import argparse
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Set

from sgnts.compose import TSCompose, TSComposedTransformElement

from sgnligo.transforms.latency import Latency


@dataclass(kw_only=True)
class ComposedTransformBase:
    """Abstract base class for composed transform elements.

    Subclasses define their parameters as dataclass fields and implement
    _build() to create the internal composed element. Composition happens
    automatically in __post_init__.

    The resulting object behaves like a TSComposedTransformElement - it has
    .srcs for source pads and .snks for sink pads, and can be connected
    to pipelines via pipeline.connect(upstream, transform.element, downstream).

    Class Attributes:
        transform_type: String identifier for registry (e.g., "standard",
            "zero-latency"). Leave empty if the transform should not be registered.
        description: Human-readable description for help text and documentation.

    Example:
        >>> from dataclasses import dataclass
        >>> from typing import ClassVar, List
        >>> from sgnts.compose import TSCompose, TSComposedTransformElement
        >>>
        >>> @dataclass(kw_only=True)
        ... class StandardCondition(ComposedTransformBase):
        ...     transform_type: ClassVar[str] = "standard"
        ...     description: ClassVar[str] = "Standard whitening"
        ...
        ...     ifos: List[str]
        ...     input_sample_rate: int
        ...     whiten_sample_rate: int = 2048
        ...
        ...     def _build(self) -> TSComposedTransformElement:
        ...         compose = TSCompose()
        ...         for ifo in self.ifos:
        ...             whiten = Whiten(...)
        ...             compose.insert(whiten)
        ...         return compose.as_transform(name=self.name)
        >>>
        >>> cond = StandardCondition(
        ...     name="cond",
        ...     ifos=["H1", "L1"],
        ...     input_sample_rate=16384,
        ... )
        >>> print(list(cond.snks.keys()))  # ['H1', 'L1']
        >>> print(list(cond.srcs.keys()))  # ['H1', 'L1', 'spectrum_H1', 'spectrum_L1']
    """

    # Required for all composed transforms
    name: str

    # Optional latency tracking (interval in seconds, None = disabled)
    latency_interval: Optional[float] = None

    # Internal composed element (built in __post_init__)
    _composed: TSComposedTransformElement = field(init=False, repr=False)

    # Internal: pads to expose even when internally linked (for latency multilink)
    _also_expose_pads: list[str] = field(init=False, repr=False, default_factory=list)

    # Class-level metadata for registry and CLI
    # Subclasses should override these
    transform_type: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __post_init__(self) -> None:
        """Validate parameters and build the composed element."""
        self._also_expose_pads = []  # Reset before build
        self._validate()
        self._composed = self._build()

    def _validate(self) -> None:
        """Override to add validation logic. Called before _build().

        Raise ValueError with descriptive message if validation fails.

        Example:
            def _validate(self) -> None:
                if not self.ifos:
                    raise ValueError("Must specify at least one IFO")
        """
        pass

    def _add_latency_tracking(
        self,
        compose: TSCompose,
        ifo: str,
        source_element: Any,
        source_pad_name: str,
        latency_route: Optional[str] = None,
    ) -> None:
        """Add latency tracking element for a single IFO output.

        Call this in _build() after creating each output to add
        latency tracking. The source pad is connected directly to
        the Latency element. Since SGN supports multilink (one source pad
        to multiple sinks), the source pad is also registered to be exposed
        externally via `also_expose_source_pads`.

        The latency output will appear as an additional source pad named
        "{ifo}_latency".

        Args:
            compose: The TSCompose object being built
            ifo: IFO name (e.g., "H1")
            source_element: The element producing the data to track
            source_pad_name: The pad name on the source element to tap
            latency_route: Optional route name for latency output

        Example:
            def _build(self) -> TSComposedTransformElement:
                compose = TSCompose()
                for ifo in self.ifos:
                    whiten = Whiten(...)
                    compose.insert(whiten)
                    self._add_latency_tracking(compose, ifo, whiten, ifo)
                return compose.as_transform(name=self.name)
        """
        if self.latency_interval is None:
            return

        route = latency_route or f"{ifo}_condition_latency"
        latency = Latency(
            name=f"{self.name}_{ifo}_latency",
            sink_pad_names=("data",),
            source_pad_names=(f"{ifo}_latency",),
            route=route,
            interval=self.latency_interval,
        )

        # Connect source directly to latency element
        compose.connect(
            source_element,
            latency,
            link_map={"data": source_pad_name},
        )

        # Register the source pad to be exposed externally (multilink pattern)
        # Format: "element_name:src:pad_name"
        pad_full_name = f"{source_element.name}:src:{source_pad_name}"
        self._also_expose_pads.append(pad_full_name)

    @abstractmethod
    def _build(self) -> TSComposedTransformElement:
        """Build and return the composed element.

        This method wires up the internal elements using TSCompose
        and returns the result of compose.as_transform(name=self.name).

        Returns:
            TSComposedTransformElement with sink and source pads
        """
        ...

    # --- CLI argument support ---

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for latency tracking."""
        parser.add_argument(
            "--condition-latency-interval",
            type=float,
            metavar="SECONDS",
            default=None,
            help="Enable condition latency tracking with specified interval in seconds",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        """Return set of CLI argument names defined by this class."""
        return {"condition_latency_interval"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert CLI args to field values."""
        result: Dict[str, Any] = {}
        condition_latency_interval = getattr(args, "condition_latency_interval", None)
        if condition_latency_interval is not None:
            result["latency_interval"] = condition_latency_interval
        return result

    # --- Delegate to inner composed element ---

    @property
    def element(self) -> TSComposedTransformElement:
        """The underlying TSComposedTransformElement for pipeline integration.

        Use this when passing to Pipeline.connect() or other SGN operations
        that require a proper element type.

        Returns:
            The inner composed element

        Example:
            >>> pipeline = Pipeline()
            >>> pipeline.connect(source.element, transform.element)
            >>> pipeline.connect(transform.element, sink)
        """
        return self._composed

    @property
    def srcs(self) -> Dict[str, Any]:
        """Source pads of the composed element.

        Returns:
            Dictionary mapping pad names to source pad objects
        """
        return self._composed.srcs

    @property
    def snks(self) -> Dict[str, Any]:
        """Sink pads of the composed element.

        Returns:
            Dictionary mapping pad names to sink pad objects
        """
        return self._composed.snks

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the inner composed element.

        This allows composed transforms to be used anywhere a TSComposedTransformElement
        is expected, supporting any additional methods or properties.
        """
        # Avoid infinite recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._composed, name)
