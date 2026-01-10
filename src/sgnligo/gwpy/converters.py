"""Conversion utilities between SGN-TS data containers and GWpy objects.

This module provides bidirectional conversion between:
- SeriesBuffer <-> gwpy.timeseries.TimeSeries
- TSFrame <-> gwpy.timeseries.TimeSeries

These conversions enable seamless integration between SGN streaming pipelines
and GWpy's batch-oriented analysis tools.
"""

from __future__ import annotations

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sgnts.base import Offset, SeriesBuffer, TSFrame


def seriesbuffer_to_timeseries(
    buf: SeriesBuffer,
    channel: str = "SGN:UNKNOWN",
    unit: str = "strain",
) -> TimeSeries:
    """Convert a SeriesBuffer to a GWpy TimeSeries.

    Args:
        buf:
            SeriesBuffer to convert
        channel:
            Channel name for the TimeSeries metadata
        unit:
            Unit string for the data (e.g., 'strain', 'm', 'Hz^-1/2')

    Returns:
        GWpy TimeSeries with equivalent data and timing

    Note:
        Gap buffers (buf.is_gap=True) are converted to TimeSeries filled with NaN.
        This preserves gap information when round-tripping through GWpy.

    Example:
        >>> buf = SeriesBuffer(
        ...     offset=Offset.fromsec(1126259462), sample_rate=4096, data=data
        ... )
        >>> ts = seriesbuffer_to_timeseries(buf, channel="H1:STRAIN")
        >>> print(ts.t0)  # LIGOTimeGPS(1126259462, 0)
    """
    # Calculate GPS t0 from offset
    # offset encodes time since offset_ref_t0 in units of MAX_RATE samples
    t0_gps = Offset.tosec(buf.offset)

    # Handle gap buffers - fill with NaN to preserve gap information
    if buf.is_gap:
        data = np.full(buf.shape, np.nan)
    else:
        # Ensure we have a numpy array
        data = np.asarray(buf.data)

    return TimeSeries(
        data,
        t0=t0_gps,
        sample_rate=buf.sample_rate,
        channel=channel,
        unit=unit,
    )


def timeseries_to_seriesbuffer(ts: TimeSeries) -> SeriesBuffer:
    """Convert a GWpy TimeSeries to a SeriesBuffer.

    Args:
        ts:
            GWpy TimeSeries to convert

    Returns:
        SeriesBuffer with equivalent data and timing

    Note:
        TimeSeries containing all NaN values are converted to gap buffers
        (data=None). This enables round-tripping of gap information.

    Example:
        >>> ts = TimeSeries.fetch_open_data('H1', 1126259462, 1126259478)
        >>> buf = timeseries_to_seriesbuffer(ts)
        >>> print(Offset.tosec(buf.offset))  # 1126259462.0
    """
    # Calculate offset from GPS t0
    t0_sec = float(ts.t0.value)
    offset = Offset.fromsec(t0_sec)

    # Get data as numpy array
    arr = np.asarray(ts.value)

    # Detect gap (all NaN) and convert to gap buffer
    data = None if np.all(np.isnan(arr)) else arr

    sample_rate = int(ts.sample_rate.value)

    return SeriesBuffer(
        offset=offset,
        sample_rate=sample_rate,
        data=data,
        shape=ts.shape,
    )


def tsframe_to_timeseries(
    frame: TSFrame,
    channel: str = "SGN:UNKNOWN",
    unit: str = "strain",
    fill_gaps: bool = True,
) -> TimeSeries:
    """Convert a TSFrame (containing multiple buffers) to a single GWpy TimeSeries.

    This concatenates all buffers in the frame into a contiguous TimeSeries.
    Gap buffers are represented as NaN regions.

    Args:
        frame:
            TSFrame to convert
        channel:
            Channel name for the TimeSeries metadata
        unit:
            Unit string for the data
        fill_gaps:
            If True (default), gap regions are filled with NaN.
            If False, raises ValueError for frames containing gaps.

    Returns:
        GWpy TimeSeries spanning the entire frame

    Raises:
        ValueError: If frame is empty or (if fill_gaps=False) contains gaps

    Example:
        >>> # After collecting data through a pipeline
        >>> ts = tsframe_to_timeseries(frame, channel="H1:PROCESSED")
        >>> ts.plot()
    """
    if len(frame.buffers) == 0:
        raise ValueError("Cannot convert empty TSFrame to TimeSeries")

    # Get frame properties
    sample_rate = frame.sample_rate
    t0_gps = Offset.tosec(frame.offset)

    # Collect data from all buffers
    if fill_gaps:
        # Use filleddata which returns NaN for gaps
        data = frame.filleddata()
    else:
        # Check for gaps
        if frame.is_gap:
            raise ValueError("Frame contains gaps and fill_gaps=False")
        data = frame.filleddata()

    return TimeSeries(
        data,
        t0=t0_gps,
        sample_rate=sample_rate,
        channel=channel,
        unit=unit,
    )


def timeseries_to_tsframe(ts: TimeSeries) -> TSFrame:
    """Convert a GWpy TimeSeries to a TSFrame with a single buffer.

    Args:
        ts:
            GWpy TimeSeries to convert

    Returns:
        TSFrame containing a single SeriesBuffer

    Example:
        >>> ts = TimeSeries.fetch_open_data('H1', 1126259462, 1126259478)
        >>> frame = timeseries_to_tsframe(ts)
        >>> pipeline.inject(frame)  # Feed into pipeline
    """
    buf = timeseries_to_seriesbuffer(ts)
    return TSFrame(buffers=[buf])


def timeseriesdict_to_buffers(
    tsd: TimeSeriesDict,
) -> dict[str, SeriesBuffer]:
    """Convert a GWpy TimeSeriesDict to a dictionary of SeriesBuffers.

    Args:
        tsd:
            GWpy TimeSeriesDict to convert

    Returns:
        Dictionary mapping channel names to SeriesBuffers

    Example:
        >>> tsd = TimeSeriesDict.read('frame.gwf', ['H1:STRAIN', 'L1:STRAIN'])
        >>> buffers = timeseriesdict_to_buffers(tsd)
        >>> h1_buf = buffers['H1:STRAIN']
    """
    return {channel: timeseries_to_seriesbuffer(ts) for channel, ts in tsd.items()}


def buffers_to_timeseriesdict(
    buffers: dict[str, SeriesBuffer],
    unit: str = "strain",
) -> TimeSeriesDict:
    """Convert a dictionary of SeriesBuffers to a GWpy TimeSeriesDict.

    Args:
        buffers:
            Dictionary mapping channel names to SeriesBuffers
        unit:
            Unit string for all TimeSeries

    Returns:
        GWpy TimeSeriesDict

    Example:
        >>> tsd = buffers_to_timeseriesdict(buffers)
        >>> tsd.write('output.gwf')
    """
    tsd = TimeSeriesDict()
    for channel, buf in buffers.items():
        tsd[channel] = seriesbuffer_to_timeseries(buf, channel=channel, unit=unit)
    return tsd
