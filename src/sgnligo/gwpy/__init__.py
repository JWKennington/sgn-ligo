"""GWpy integration for SGN LIGO streaming pipelines.

This subpackage provides SGN TS interfaces to GWpy utilities, enabling
gravitational wave researchers to build streaming pipelines using familiar
GWpy operations.

Components:
    converters: Bidirectional conversion between SeriesBuffer and TimeSeries
    sources: Data sources (GWOSC, NDS2, existing TimeSeries)
    transforms: Signal processing (filtering, spectrogram, Q-transform)
    sinks: Output (plotting, TimeSeries collection)

Example:
    >>> from sgnligo.gwpy import (
    ...     GWOSCSource, GWpyFilter, TimeSeriesSink
    ... )
    >>> from sgn.apps import Pipeline
    >>>
    >>> pipeline = Pipeline()
    >>> source = GWOSCSource(detector="H1", start_time=1126259462, duration=32)
    >>> filt = GWpyFilter(filter_type="bandpass", low_freq=20, high_freq=500)
    >>> sink = TimeSeriesSink(channel="H1:PROCESSED")
    >>>
    >>> pipeline.insert(source, filt, sink, link_map={...})
    >>> pipeline.run()
    >>> result = sink.get_result()  # Returns GWpy TimeSeries
"""

from sgnligo.gwpy.converters import (
    buffers_to_timeseriesdict,
    seriesbuffer_to_timeseries,
    timeseries_to_seriesbuffer,
    timeseries_to_tsframe,
    timeseriesdict_to_buffers,
    tsframe_to_timeseries,
)

# Sinks
from sgnligo.gwpy.sinks import TimeSeriesSink

# Sources
from sgnligo.gwpy.sources import GWOSCSource, TimeSeriesSource

# Transforms
from sgnligo.gwpy.transforms import GWpyFilter, GWpyQTransform, GWpySpectrogram

# from sgnligo.gwpy.sources import NDS2Source


# from sgnligo.gwpy.sinks import GWpyPlotSink

__all__ = [
    # Converters
    "seriesbuffer_to_timeseries",
    "timeseries_to_seriesbuffer",
    "tsframe_to_timeseries",
    "timeseries_to_tsframe",
    "timeseriesdict_to_buffers",
    "buffers_to_timeseriesdict",
    # Sources
    "TimeSeriesSource",
    "GWOSCSource",
    # "NDS2Source",
    # Transforms
    "GWpyFilter",
    "GWpySpectrogram",
    "GWpyQTransform",
    # Sinks
    "TimeSeriesSink",
    # "GWpyPlotSink",
]
