"""TimeSeriesSink: Collect pipeline output into a GWpy TimeSeries.

This sink enables researchers to process data through SGN streaming pipelines
and then continue analysis with GWpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sgnts.base import Offset, TSSink


@dataclass
class TimeSeriesSink(TSSink):
    """Sink that collects pipeline output into a GWpy TimeSeries.

    This sink accumulates all buffers from the pipeline and provides
    them as a GWpy TimeSeries after the pipeline completes. Useful for
    researchers who want to process data through SGN pipelines and then
    continue analysis with GWpy.

    Args:
        channel:
            Channel name for the output TimeSeries (default "SGN:OUTPUT")
        unit:
            Unit string for the output data (default "strain")
        collect_all:
            If True (default), concatenates all buffers into a single
            TimeSeries. If False, only the most recent buffer is kept.

    Attributes:
        result:
            The collected GWpy TimeSeries. Available after pipeline
            completes (when EOS is received).

    Example:
        >>> from sgnligo.gwpy.sinks import TimeSeriesSink
        >>> from sgn.apps import Pipeline
        >>>
        >>> pipeline = Pipeline()
        >>> # ... configure pipeline ...
        >>> sink = TimeSeriesSink(name="Collector", channel="H1:PROCESSED")
        >>> # ... add to pipeline ...
        >>> pipeline.run()
        >>>
        >>> # Get result as GWpy TimeSeries
        >>> ts = sink.get_result()
        >>> ts.plot()
        >>>
        >>> # Or for multi-channel pipelines
        >>> tsd = sink.get_result_dict()  # Returns TimeSeriesDict

    Note:
        Gap buffers (missing data) are represented as NaN values in the
        output TimeSeries to preserve timing information.
    """

    channel: str = "SGN:OUTPUT"
    unit: str = "strain"
    collect_all: bool = True

    # Internal storage
    _buffers: list = field(default_factory=list, init=False, repr=False)
    _sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    _first_offset: Optional[int] = field(default=None, init=False, repr=False)
    _is_complete: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._buffers = []
        self._sample_rate = None
        self._first_offset = None
        self._is_complete = False

    def internal(self):
        """Process incoming frames and collect buffers."""
        super().internal()

        for pad in self.sink_pads:
            frame = self.preparedframes.get(pad)
            if frame is None:
                continue

            # Check for EOS
            if frame.EOS:
                self._is_complete = True
                self.mark_eos(pad)

            # Collect buffers
            for buf in frame.buffers:
                if buf.shape[0] == 0:
                    # Skip zero-length buffers
                    continue

                # Track sample rate and first offset
                if self._sample_rate is None:
                    self._sample_rate = buf.sample_rate
                if self._first_offset is None:
                    self._first_offset = buf.offset

                if self.collect_all:
                    self._buffers.append(buf)
                else:
                    # Only keep the most recent buffer
                    self._buffers = [buf]

    def get_result(self) -> TimeSeries:
        """Get the collected data as a GWpy TimeSeries.

        Returns:
            GWpy TimeSeries containing all collected data

        Raises:
            ValueError: If no data has been collected

        Example:
            >>> ts = sink.get_result()
            >>> ts.plot()
            >>> ts.bandpass(20, 500).plot()
        """
        if not self._buffers:
            raise ValueError("No data collected. Run the pipeline first.")

        # Concatenate all buffer data
        all_data = []
        for buf in self._buffers:
            if buf.is_gap:
                # Fill gaps with NaN
                gap_data = np.full(buf.shape, np.nan)
                all_data.append(gap_data)
            else:
                all_data.append(np.asarray(buf.data))

        assert (
            all_data
        ), "all_data should not be empty after iterating over self._buffers"

        # Concatenate along time axis (last dimension)
        data = np.concatenate(all_data, axis=-1)

        # Calculate t0 from first offset
        t0_gps = Offset.tosec(self._first_offset)

        return TimeSeries(
            data,
            t0=t0_gps,
            sample_rate=self._sample_rate,
            channel=self.channel,
            unit=self.unit,
        )

    def get_result_dict(self) -> TimeSeriesDict:
        """Get the collected data as a GWpy TimeSeriesDict.

        This is useful when the sink has multiple input pads.

        Returns:
            GWpy TimeSeriesDict with channel as key

        Example:
            >>> tsd = sink.get_result_dict()
            >>> tsd.plot()
        """
        tsd = TimeSeriesDict()
        tsd[self.channel] = self.get_result()
        return tsd

    def clear(self):
        """Clear collected data to start fresh.

        Useful for reusing the sink in multiple pipeline runs.
        """
        self._buffers = []
        self._sample_rate = None
        self._first_offset = None
        self._is_complete = False

    @property
    def is_complete(self) -> bool:
        """Check if the pipeline has completed (EOS received)."""
        return self._is_complete

    @property
    def samples_collected(self) -> int:
        """Return the total number of samples collected."""
        return sum(buf.shape[0] for buf in self._buffers if not buf.is_gap)

    @property
    def duration_collected(self) -> float:
        """Return the total duration collected in seconds."""
        if self._sample_rate is None:
            return 0.0
        total_samples = sum(buf.shape[0] for buf in self._buffers)
        return total_samples / self._sample_rate
