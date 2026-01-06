"""A source element that fetches open data from GWOSC via gwpy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from gwpy.timeseries import TimeSeries
from sgn.base import SourcePad
from sgnts.base import Audioadapter, Offset, SeriesBuffer, TSFrame, TSSource

# Global cache to optimize repeated runs in the same process
_GWOSC_RAM_CACHE: dict = {}


@dataclass
class GWOSCSource(TSSource):
    """
    Source element that fetches gravitational wave data from GWOSC.

    Args:
        detectors: Single string (e.g., "H1") or list of detectors (e.g., ["H1", "L1"]).
        start: GPS start time.
        end: GPS end time.
        frame_type: (Optional) The specific frame type (e.g. "H1_HOFT_C01").
        sample_rate: The EXPECTED sample rate of the data in Hz.
        batch_duration: Seconds of data to fetch in a single HTTP request (e.g. 64s).
        block_duration: Duration of each output frame in seconds (e.g. 1s).
        min_request_interval: Min seconds between GWOSC calls (rate limiting).
        cache_data: If True, keep fetched data in process memory.
        verbose: If True, print debug info.
        t0: (Inherited) Automatically mapped from 'start'.
    """

    detectors: Union[str, List[str]] = "H1"
    start: float = 0.0
    end: float = 0.0
    frame_type: Optional[str] = None
    sample_rate: int = 4096
    batch_duration: float = 64.0
    block_duration: float = 1.0
    min_request_interval: float = 1.0
    cache_data: bool = True
    verbose: bool = False

    # Internal state
    _adapters: Dict[str, Audioadapter] = field(default_factory=dict, repr=False)
    _fetch_cursors: Dict[str, float] = field(default_factory=dict, repr=False)
    _last_request_time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        # 1. Normalize detectors
        if isinstance(self.detectors, str):
            self.detectors = [self.detectors]

        # SGN requirement: source_pad_names must be set before super().__post_init__
        self.source_pad_names = tuple(self.detectors)

        # 2. Map 'start' to TSSource 't0' requirement
        if self.t0 is None:
            self.t0 = self.start

        # 3. Validate times
        if self.end <= self.start:
            raise ValueError(
                f"End time ({self.end}) must be after start time ({self.start})"
            )

        # 4. Initialize Parent (TSSource)
        # This handles t0 validation, duration logic, and base initialization
        super().__post_init__()

        # 5. Initialize internal buffering
        self._adapters = {d: Audioadapter() for d in self.detectors}
        self._fetch_cursors = {d: self.start for d in self.detectors}
        self._last_request_time = 0.0

        # 6. Configure output buffers
        # TSSource uses num_samples() to determine buffer size
        for detector in self.detectors:
            self.set_pad_buffer_params(
                pad=self.srcs[detector],
                sample_shape=(),
                rate=self.sample_rate,
            )

    def num_samples(self, rate: int) -> int:
        """
        Override TSSource.num_samples to control the output frame duration.
        """
        return int(rate * self.block_duration)

    def _apply_rate_limit(self):
        """Enforce the minimum request interval globally."""
        now = time.time()
        elapsed = now - self._last_request_time
        wait = self.min_request_interval - elapsed
        if wait > 0:
            if self.verbose:
                print(f"GWOSCSource: Rate limiting, sleeping {wait:.3f}s")
            time.sleep(wait)

    def _fetch_next_batch(self, detector: str) -> None:
        """Fetch batch from GWOSC and push to Audioadapter."""
        cursor = self._fetch_cursors[detector]
        fetch_start = cursor
        fetch_end = min(cursor + self.batch_duration, self.end)

        if fetch_start >= fetch_end:
            return

        cache_key = (detector, fetch_start, fetch_end, self.sample_rate)
        data = None

        if self.cache_data and cache_key in _GWOSC_RAM_CACHE:
            if self.verbose:
                print(f"GWOSCSource: Cache Hit {detector} [{fetch_start}, {fetch_end})")
            data = _GWOSC_RAM_CACHE[cache_key]

        if data is None:
            self._apply_rate_limit()
            if self.verbose:
                print(
                    f"GWOSCSource: Fetching {detector} [{fetch_start}, {fetch_end})..."
                )

            try:
                # Dynamic kwargs to avoid passing None for frametype
                fetch_kwargs: Dict[str, Any] = {
                    "verbose": self.verbose,
                    "cache": True,
                }
                if self.frame_type is not None:
                    fetch_kwargs["frametype"] = self.frame_type

                ts = TimeSeries.fetch_open_data(
                    detector, fetch_start, fetch_end, **fetch_kwargs
                )

                fetched_rate = int(ts.sample_rate.value)
                if fetched_rate != self.sample_rate:
                    raise ValueError(
                        f"GWOSC data rate ({fetched_rate} Hz) does not match "
                        f"configured rate ({self.sample_rate} Hz). "
                        "GWOSCSource does not resample; use a Resampler element."
                    )

                data = ts.value
                self._last_request_time = time.time()

                if self.cache_data:
                    _GWOSC_RAM_CACHE[cache_key] = data

            except Exception as e:
                raise RuntimeError(
                    f"Failed to fetch GWOSC data for {detector}: {e}"
                ) from e

        # Push to Adapter
        buf = SeriesBuffer(
            offset=Offset.fromsec(fetch_start), sample_rate=self.sample_rate, data=data
        )
        self._adapters[detector].push(buf)
        self._fetch_cursors[detector] = fetch_end

    def internal(self) -> None:
        """Check buffers and fetch if needed."""
        super().internal()

        for det in self.detectors:
            adapter = self._adapters[det]
            cursor = self._fetch_cursors[det]
            needed_samples = self.num_samples(self.sample_rate)

            # Fetch if we have less than one block of data
            if cursor < self.end and adapter.size < needed_samples:
                self._fetch_next_batch(det)

    def new(self, pad: SourcePad) -> TSFrame:
        """Produce the next aligned frame."""
        detector = next(d for d in self.detectors if self.srcs[d] is pad)
        adapter = self._adapters[detector]

        # 1. Use TSSource to calculate correct offsets/EOS based on t0/end/stride
        # This advances the source element's internal clock automatically
        frame = self.prepare_frame(pad)

        # 2. Fill the frame from the adapter
        # TSSource prepared a frame with data=None. We slice real data into it.
        # Check if we have data covering this frame's time range
        if adapter.size > 0 and adapter.end_offset >= frame.end_offset:
            bufs = adapter.get_sliced_buffers((frame.offset, frame.end_offset))
            frame.set_buffers(bufs)
            adapter.flush_samples_by_end_offset(frame.end_offset)

        # Note: If we don't have enough data, we return the frame as-is
        # (with data=None).
        # This signals a gap/underrun to the pipeline, which is valid behavior
        # while waiting for the network. EOS is handled by prepare_frame automatically
        # when we pass self.end.
        return frame
