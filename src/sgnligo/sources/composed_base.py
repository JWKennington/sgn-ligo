"""Base class for composed source elements.

This module provides an abstract base class for creating composed source elements
that combine multiple internal elements into a single source. Subclasses define
their parameters as dataclass fields and implement _build() to wire up the
internal elements.

Usage with pipelines:
    Composed sources wrap a TSComposedSourceElement internally. To use them
    with Pipeline.connect(), access the inner element via the .element property:

    >>> pipeline = Pipeline()
    >>> pipeline.connect(source.element, sink)

Example:
    >>> from dataclasses import dataclass
    >>> from sgnligo.sources.composed_base import ComposedSourceBase
    >>>
    >>> @dataclass
    ... class MySource(ComposedSourceBase):
    ...     source_type = "my-source"
    ...     description = "My custom source"
    ...
    ...     ifos: list[str]
    ...     sample_rate: int
    ...     t0: float
    ...     end: float
    ...
    ...     def _build(self):
    ...         compose = TSCompose()
    ...         # ... wire up elements
    ...         return compose.as_source(name=self.name)
    >>>
    >>> source = MySource(name="test", ifos=["H1"], sample_rate=4096, t0=0, end=10)
    >>> print(source.srcs)  # Access source pads
"""

from __future__ import annotations

import argparse
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Set

from sgnts.compose import TSCompose, TSComposedSourceElement

from sgnligo.transforms.latency import Latency


@dataclass(kw_only=True)
class ComposedSourceBase:
    """Abstract base class for composed source elements.

    Subclasses define their parameters as dataclass fields and implement
    _build() to create the internal composed element. Composition happens
    automatically in __post_init__.

    The resulting object behaves like a TSComposedSourceElement - it has
    .srcs for source pads and can be connected to pipelines via
    pipeline.connect(source, downstream).

    Class Attributes:
        source_type: String identifier for registry (e.g., "white", "frames").
            Leave empty if the source should not be registered.
        description: Human-readable description for help text and documentation.

    Example:
        >>> from dataclasses import dataclass
        >>> from typing import ClassVar, List
        >>> from sgnts.compose import TSCompose, TSComposedSourceElement
        >>> from sgnts.sources import FakeSeriesSource
        >>>
        >>> @dataclass(kw_only=True)
        ... class WhiteSource(ComposedSourceBase):
        ...     source_type: ClassVar[str] = "white"
        ...     description: ClassVar[str] = "Gaussian white noise"
        ...
        ...     ifos: List[str]
        ...     sample_rate: int
        ...     t0: float
        ...     end: float
        ...
        ...     def _build(self) -> TSComposedSourceElement:
        ...         compose = TSCompose()
        ...         for ifo in self.ifos:
        ...             fake = FakeSeriesSource(
        ...                 name=f"{self.name}_{ifo}",
        ...                 source_pad_names=(f"{ifo}:STRAIN",),
        ...                 rate=self.sample_rate,
        ...                 t0=self.t0,
        ...                 end=self.end,
        ...                 signal_type="white",
        ...             )
        ...             compose.insert(fake)
        ...         return compose.as_source(name=self.name)
        >>>
        >>> source = WhiteSource(
        ...     name="noise",
        ...     ifos=["H1", "L1"],
        ...     sample_rate=4096,
        ...     t0=1000,
        ...     end=1010,
        ... )
        >>> print(list(source.srcs.keys()))
        ['H1:STRAIN', 'L1:STRAIN']
    """

    # Required for all composed sources
    name: str

    # Optional latency tracking (interval in seconds, None = disabled)
    latency_interval: Optional[float] = None

    # Internal composed element (built in __post_init__)
    _composed: TSComposedSourceElement = field(init=False, repr=False)

    # Internal: pads to expose even when internally linked (for latency multilink)
    _also_expose_pads: list[str] = field(init=False, repr=False, default_factory=list)

    # Class-level metadata for registry and CLI
    # Subclasses should override these
    source_type: ClassVar[str] = ""
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
                if self.t0 >= self.end:
                    raise ValueError("t0 must be less than end")
        """
        pass

    def _add_latency_tracking(
        self,
        compose: TSCompose,
        ifo: str,
        strain_source_element: Any,
        strain_pad_name: str,
    ) -> None:
        """Add latency tracking element for a single IFO.

        Call this in _build() after creating each strain source to add
        latency tracking. The strain source pad is connected directly to
        the Latency element. Since SGN supports multilink (one source pad
        to multiple sinks), the strain pad is also registered to be exposed
        externally via `also_expose_source_pads`.

        The latency output will appear as an additional source pad named
        "{ifo}_latency".

        Args:
            compose: The TSCompose object being built
            ifo: IFO name (e.g., "H1")
            strain_source_element: The source element producing strain data
            strain_pad_name: The pad name on the source element to tap

        Example:
            def _build(self) -> TSComposedSourceElement:
                compose = TSCompose()
                for ifo in self.ifos:
                    source = FakeSeriesSource(...)
                    compose.insert(source)
                    self._add_latency_tracking(compose, ifo, source, ifo)
                return compose.as_source(
                    name=self.name,
                    also_expose_source_pads=self._also_expose_pads,
                )
        """
        if self.latency_interval is None:
            return

        latency = Latency(
            name=f"{self.name}_{ifo}_latency",
            sink_pad_names=("data",),
            source_pad_names=(f"{ifo}_latency",),
            route=f"{ifo}_datasource_latency",
            interval=self.latency_interval,
        )

        # Connect strain source directly to latency element
        compose.connect(
            strain_source_element,
            latency,
            link_map={"data": strain_pad_name},
        )

        # Register the strain pad to be exposed externally (multilink pattern)
        # Format: "element_name:src:pad_name"
        pad_full_name = f"{strain_source_element.name}:src:{strain_pad_name}"
        self._also_expose_pads.append(pad_full_name)

    @abstractmethod
    def _build(self) -> TSComposedSourceElement:
        """Build and return the composed element.

        This method wires up the internal elements using TSCompose
        and returns the result of compose.as_source(name=self.name).

        Returns:
            TSComposedSourceElement with source pads for downstream connection
        """
        ...

    # --- CLI argument support ---

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for latency tracking."""
        parser.add_argument(
            "--source-latency-interval",
            type=float,
            metavar="SECONDS",
            default=None,
            help="Enable source latency tracking with specified interval in seconds",
        )

    @classmethod
    def get_cli_arg_names(cls) -> Set[str]:
        """Return set of CLI argument names defined by this class."""
        return {"source_latency_interval"}

    @classmethod
    def process_cli_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert CLI args to field values."""
        result: Dict[str, Any] = {}
        source_latency_interval = getattr(args, "source_latency_interval", None)
        if source_latency_interval is not None:
            result["latency_interval"] = source_latency_interval
        return result

    # --- Delegate to inner composed element ---

    @property
    def element(self) -> TSComposedSourceElement:
        """The underlying TSComposedSourceElement for pipeline integration.

        Use this when passing to Pipeline.connect() or other SGN operations
        that require a proper element type.

        Returns:
            The inner composed element

        Example:
            >>> pipeline = Pipeline()
            >>> pipeline.connect(source.element, sink)
        """
        return self._composed

    @property
    def srcs(self) -> Dict[str, Any]:
        """Source pads of the composed element.

        Returns:
            Dictionary mapping pad names to source pad objects
        """
        return self._composed.srcs

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the inner composed element.

        This allows composed sources to be used anywhere a TSComposedSourceElement
        is expected, supporting any additional methods or properties.
        """
        # Avoid infinite recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._composed, name)
