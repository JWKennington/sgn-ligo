"""Composable data sources for LIGO pipelines.

This package provides dataclass-based source elements for creating
gravitational wave data source pipelines.

Example - Simple CLI usage:
    >>> from sgnligo.sources.datasource_v2 import DataSource
    >>>
    >>> # Create from command line arguments
    >>> source = DataSource.from_argv(name="data_source")
    >>> pipeline.connect(source.element, my_sink)

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
    >>> pipeline.connect(source.element, my_sink)

Example - Direct source class:
    >>> from sgnligo.sources.datasource_v2.sources import WhiteComposedSource
    >>>
    >>> source = WhiteComposedSource(
    ...     name="noise",
    ...     ifos=["H1", "L1"],
    ...     sample_rate=4096,
    ...     t0=1000,
    ...     end=1010,
    ... )
    >>> pipeline.connect(source.element, my_sink)
"""

__all__ = [
    # DataSource uber element - main public API
    "DataSource",
    # CLI mixins - for creating custom sources
    "CLIMixinProtocol",
    "GPSOptionsMixin",
    "GPSOptionsOptionalMixin",
    "ChannelOptionsMixin",
    "IfosOnlyMixin",
    "ChannelPatternOptionsMixin",
    "SampleRateOptionsMixin",
    "SegmentsOptionsMixin",
    "StateVectorOptionsMixin",
    "InjectionOptionsMixin",
    "DevShmOptionsMixin",
    "QueueTimeoutOptionsMixin",
    "FrameCacheOptionsMixin",
    "ImpulsePositionOptionsMixin",
    "VerboseOptionsMixin",
    # Registry functions - for source introspection
    "get_composed_registry",
    "get_composed_source_class",
    "list_composed_source_types",
    "register_composed_source",
]

# Import InjectedNoiseSource to register it
from sgnligo.sources import injected_noise_source  # noqa: F401

# Import sources to trigger composed registry registration
from sgnligo.sources.datasource_v2 import sources  # noqa: F401

# CLI mixins
from sgnligo.sources.datasource_v2.cli_mixins import (
    ChannelOptionsMixin,
    ChannelPatternOptionsMixin,
    CLIMixinProtocol,
    DevShmOptionsMixin,
    FrameCacheOptionsMixin,
    GPSOptionsMixin,
    GPSOptionsOptionalMixin,
    IfosOnlyMixin,
    ImpulsePositionOptionsMixin,
    InjectionOptionsMixin,
    QueueTimeoutOptionsMixin,
    SampleRateOptionsMixin,
    SegmentsOptionsMixin,
    StateVectorOptionsMixin,
    VerboseOptionsMixin,
)

# Registry
from sgnligo.sources.datasource_v2.composed_registry import (
    get_composed_registry,
    get_composed_source_class,
    list_composed_source_types,
    register_composed_source,
)

# DataSource dispatcher
from sgnligo.sources.datasource_v2.datasource import DataSource
