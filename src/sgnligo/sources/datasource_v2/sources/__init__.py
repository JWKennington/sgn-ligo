"""Composable data source classes for LIGO pipelines.

This package provides dataclass-based source elements that can be instantiated
directly or via the DataSource dispatcher.

Example:
    >>> from sgnligo.sources.datasource_v2.sources import WhiteComposedSource
    >>> source = WhiteComposedSource(
    ...     name="noise",
    ...     ifos=["H1", "L1"],
    ...     sample_rate=4096,
    ...     t0=1000,
    ...     end=1010,
    ... )
    >>> pipeline.connect(source.element, sink)
"""

from sgnligo.sources.datasource_v2.sources.devshm import DevShmComposedSource
from sgnligo.sources.datasource_v2.sources.fake import (
    ImpulseComposedSource,
    ImpulseRealtimeComposedSource,
    SinComposedSource,
    SinRealtimeComposedSource,
    WhiteComposedSource,
    WhiteRealtimeComposedSource,
)
from sgnligo.sources.datasource_v2.sources.frames import FramesComposedSource
from sgnligo.sources.datasource_v2.sources.gwdata_noise import (
    GWDataNoiseComposedSource,
    GWDataNoiseRealtimeComposedSource,
)

# Arrakis is optional (requires sgn_arrakis)
try:
    from sgnligo.sources.datasource_v2.sources.arrakis import (  # noqa: F401
        ArrakisComposedSource,
    )

    _HAS_ARRAKIS = True
except ImportError:
    _HAS_ARRAKIS = False

__all__ = [
    # Fake sources
    "WhiteComposedSource",
    "SinComposedSource",
    "ImpulseComposedSource",
    "WhiteRealtimeComposedSource",
    "SinRealtimeComposedSource",
    "ImpulseRealtimeComposedSource",
    # GWData noise sources
    "GWDataNoiseComposedSource",
    "GWDataNoiseRealtimeComposedSource",
    # Frame sources
    "FramesComposedSource",
    # DevShm sources
    "DevShmComposedSource",
]

# Add Arrakis if available
if _HAS_ARRAKIS:
    __all__.append("ArrakisComposedSource")
