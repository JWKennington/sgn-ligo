"""GWpy integration for SGN LIGO streaming pipelines.

This subpackage provides SGN TS interfaces to GWpy utilities, enabling
gravitational wave researchers to build streaming pipelines using familiar
GWpy operations.

Components:
    converters: Bidirectional conversion between SeriesBuffer and TimeSeries
    sources: Data sources (NDS2, existing TimeSeries)
    transforms: Signal processing (filtering, spectrogram, Q-transform)
    sinks: Output (plotting, TimeSeries collection)

Example:
    >>> from sgnligo.gwpy import (
    ...     TimeSeriesSource, GWpyFilter, TimeSeriesSink
    ... )
    >>> from sgn.apps import Pipeline
    >>> from gwpy.timeseries import TimeSeries
    >>>
    >>> pipeline = Pipeline()
    >>> ts = TimeSeries.fetch_open_data("H1", 1126259462, 1126259494)
    >>> source = TimeSeriesSource(name="H1", timeseries=ts, buffer_duration=1.0)
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
from sgnligo.gwpy.sinks import GWpyPlotSink, TimeSeriesSink

# Sources
from sgnligo.gwpy.sources import TimeSeriesSource

# Transforms
from sgnligo.gwpy.transforms import GWpyFilter, GWpyQTransform, GWpySpectrogram

# from sgnligo.gwpy.sources import NDS2Source

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
    # "NDS2Source",
    # Transforms
    "GWpyFilter",
    "GWpySpectrogram",
    "GWpyQTransform",
    # Sinks
    "TimeSeriesSink",
    "GWpyPlotSink",
]
