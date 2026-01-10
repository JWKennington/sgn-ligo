"""GWpy-based data sources for SGN LIGO pipelines.

Sources:
    TimeSeriesSource: Wrap an existing GWpy TimeSeries for pipeline processing
    NDS2Source: Stream data from NDS2 servers
"""

from sgnligo.gwpy.sources.timeseries_source import TimeSeriesSource

# from sgnligo.gwpy.sources.nds2_source import NDS2Source

__all__ = [
    "TimeSeriesSource",
    # "NDS2Source",
]
