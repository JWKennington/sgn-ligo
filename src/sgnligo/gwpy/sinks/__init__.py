"""GWpy-based sinks for SGN LIGO pipelines.

Sinks:
    TimeSeriesSink: Collect pipeline output into a GWpy TimeSeries
    GWpyPlotSink: Generate plots using GWpy's visualization
"""

from sgnligo.gwpy.sinks.plot_sink import GWpyPlotSink
from sgnligo.gwpy.sinks.timeseries_sink import TimeSeriesSink

__all__ = [
    "TimeSeriesSink",
    "GWpyPlotSink",
]
