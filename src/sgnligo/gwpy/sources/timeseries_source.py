"""TimeSeriesSource: Wrap an existing GWpy TimeSeries for pipeline processing.

This source enables researchers who have already loaded data with GWpy
to process it through SGN streaming pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sgn.base import SourcePad
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource


@dataclass
class TimeSeriesSource(TSSource):
    """Source that wraps an existing GWpy TimeSeries for pipeline processing.

    This source takes a pre-loaded GWpy TimeSeries or TimeSeriesDict and
    streams it through the pipeline in fixed-size buffers. Useful for
    researchers who have already loaded data with GWpy and want to
    process it through SGN streaming pipelines.

    Args:
        timeseries:
            GWpy TimeSeries or TimeSeriesDict to stream. If a TimeSeries,
            it will be streamed on a single pad. If a TimeSeriesDict,
            each channel becomes a separate source pad.
        buffer_duration:
            Duration of output buffers in seconds (default 1.0).
            Smaller buffers provide finer granularity but more overhead.

    Example:
        >>> from gwpy.timeseries import TimeSeries
        >>> from sgnligo.gwpy.sources import TimeSeriesSource
        >>>
        >>> # Load data with GWpy
        >>> ts = TimeSeries.fetch_open_data('H1', 1126259462, 1126259478)
        >>>
        >>> # Stream through pipeline
        >>> source = TimeSeriesSource(
        ...     name="H1_Data",
        ...     timeseries=ts,
        ...     buffer_duration=1.0,
        ... )
        >>>
        >>> # Multi-channel example
        >>> tsd = TimeSeriesDict.read('frame.gwf', ['H1:STRAIN', 'L1:STRAIN'])
        >>> source = TimeSeriesSource(
        ...     name="Multi_IFO",
        ...     timeseries=tsd,
        ...     buffer_duration=1.0,
        ... )

    Note:
        The source automatically sets t0 and end based on the TimeSeries
        span. Source pad names are derived from channel names.
    """

    timeseries: Optional[Union[TimeSeries, TimeSeriesDict]] = None
    buffer_duration: float = 1.0

    # Internal storage for multi-channel data
    _ts_dict: dict = field(default_factory=dict, init=False, repr=False)
    _current_idx: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.timeseries is None:
            raise ValueError("timeseries argument is required")

        # Convert single TimeSeries to TimeSeriesDict for uniform handling
        if isinstance(self.timeseries, TimeSeries):
            channel = str(self.timeseries.channel) or "SGN:INPUT"
            self._ts_dict = {channel: self.timeseries}
        elif isinstance(self.timeseries, TimeSeriesDict):
            self._ts_dict = dict(self.timeseries)
        else:
            raise TypeError(
                f"timeseries must be TimeSeries or TimeSeriesDict, "
                f"got {type(self.timeseries)}"
            )

        if not self._ts_dict:
            raise ValueError("timeseries cannot be empty")

        # Set source pad names from channel names
        self.source_pad_names = tuple(self._ts_dict.keys())

        # Determine t0 and end from data
        # All channels should have the same time span (GWpy convention)
        first_ts = next(iter(self._ts_dict.values()))
        self.t0 = float(first_ts.t0.value)
        data_duration = float(first_ts.duration.value)
        self.end = self.t0 + data_duration

        # Initialize base class
        super().__post_init__()

        # Configure pad parameters
        for pad in self.source_pads:
            channel = self.rsrcs[pad]
            ts = self._ts_dict[channel]
            sample_rate = int(ts.sample_rate.value)
            self.set_pad_buffer_params(pad=pad, sample_shape=(), rate=sample_rate)

        # Initialize buffer index per pad (in samples)
        self._current_idx = {pad: 0 for pad in self.source_pads}

    def new(self, pad: SourcePad) -> TSFrame:
        """Generate the next frame of data for the given pad.

        Args:
            pad:
                SourcePad to generate data for

        Returns:
            TSFrame containing the next buffer of data
        """
        channel = self.rsrcs[pad]
        ts = self._ts_dict[channel]
        sample_rate = int(ts.sample_rate.value)
        total_samples = len(ts)

        # Determine samples for this buffer
        buffer_samples = int(self.buffer_duration * sample_rate)
        start_idx = self._current_idx[pad]
        end_idx = min(start_idx + buffer_samples, total_samples)
        actual_samples = end_idx - start_idx

        # Calculate offset for this buffer
        buffer_t0 = float(ts.t0.value) + start_idx / sample_rate
        offset = Offset.fromsec(buffer_t0)

        # Extract data
        if actual_samples > 0:
            data = np.asarray(ts.value[start_idx:end_idx])
        else:
            data = None

        # Check for EOS
        is_eos = end_idx >= total_samples

        # Update index for next call
        self._current_idx[pad] = end_idx

        # Create buffer
        if data is not None and len(data) > 0:
            buf = SeriesBuffer(
                offset=offset,
                sample_rate=sample_rate,
                data=data,
                shape=(actual_samples,),
            )
        else:
            # Gap or end of data
            buf = SeriesBuffer(
                offset=offset,
                sample_rate=sample_rate,
                data=None,
                shape=(0,),
            )

        return TSFrame(
            buffers=[buf],
            EOS=is_eos,
            metadata={"channel": channel},
        )
