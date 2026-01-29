"""A source element that fetches open data from GWOSC via gwpy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
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
        detectors: Single string (e.g., "H1") or list of detectors.
        start: GPS start time.
        end: GPS end time.
        frame_type: (Optional) The specific frame type (e.g. "H1_HOFT_C01").
        channel: (Optional) Specific channel name (e.g. "H1:GDS-CALIB_STRAIN").
                 If provided, uses NDS2/Frames via TimeSeries.get().
                 If None, defaults to public GWOSC data via TimeSeries.fetch_open_data().
        sample_rate: The EXPECTED sample rate of the data in Hz.
        batch_duration: Seconds of data to fetch in a single HTTP request.
        block_duration: Duration of each output frame in seconds.
        min_request_interval: Min seconds between GWOSC calls (rate limiting).
        cache_data: If True, keep fetched data in process memory.
        verbose: If True, print debug info.
        t0: (Inherited) Automatically mapped from 'start'.
    """

    detectors: Union[str, List[str]] = "H1"
    start: float = 0.0
    end: float = 0.0
    frame_type: Optional[str] = None
    channel: Optional[str] = None
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
        if isinstance(self.detectors, str):
            self.detectors = [self.detectors]

        self.source_pad_names = tuple(self.detectors)

        if self.t0 is None:
            self.t0 = self.start

        if self.end <= self.start:
            raise ValueError(
                f"End time ({self.end}) must be after start time ({self.start})"
            )

        super().__post_init__()

        self._adapters = {d: Audioadapter() for d in self.detectors}
        self._fetch_cursors = {d: self.start for d in self.detectors}
        self._last_request_time = 0.0

        for detector in self.detectors:
            self.set_pad_buffer_params(
                pad=self.srcs[detector],
                sample_shape=(),
                rate=self.sample_rate,
            )

    def num_samples(self, rate: int) -> int:
        return int(rate * self.block_duration)

    def _apply_rate_limit(self):
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
                # Decide strategy: Explicit Channel -> .get(), Default -> .fetch_open_data()
                if self.channel:
                    # PROPRIETARY / NDS2 STRATEGY
                    fetch_kwargs = {
                        "verbose": self.verbose,
                        "start": fetch_start,
                        "end": fetch_end,
                    }
                    if self.frame_type:
                        fetch_kwargs["frametype"] = self.frame_type

                    ts = TimeSeries.get(self.channel, **fetch_kwargs)

                else:
                    # OPEN DATA STRATEGY (Default)
                    # This avoids 'nds2' dependency for open data users
                    fetch_kwargs = {
                        "verbose": self.verbose,
                        "cache": True,
                        "start": fetch_start,
                        "end": fetch_end,
                    }
                    if self.frame_type:
                        fetch_kwargs["frametype"] = self.frame_type

                    ts = TimeSeries.fetch_open_data(detector, **fetch_kwargs)

                # Resample if needed
                fetched_rate = int(ts.sample_rate.value)
                if fetched_rate != self.sample_rate:
                    if self.verbose:
                        print(
                            f"GWOSCSource: Resampling {fetched_rate}->{self.sample_rate}"
                        )
                    ts = ts.resample(self.sample_rate)

                data = ts.value
                self._last_request_time = time.time()

                if self.cache_data:
                    _GWOSC_RAM_CACHE[cache_key] = data

            except Exception as e:
                # --- GAP HANDLING ---
                # Check for "Cannot find" (GWOSC) or "No data found" (NDS2)
                msg = str(e).lower()
                if (
                    "cannot find" in msg
                    or "no data found" in msg
                    or "unknown datafind" in msg
                ):
                    if self.verbose:
                        print(
                            f"GWOSCSource: GAP at {fetch_start}. Creating gap buffer."
                        )

                    nsamples = int((fetch_end - fetch_start) * self.sample_rate)

                    buf = SeriesBuffer(
                        offset=Offset.fromsec(fetch_start),
                        sample_rate=self.sample_rate,
                        data=None,  # Explicit Gap
                        shape=(nsamples,),  # Defines duration
                    )
                    self._adapters[detector].push(buf)
                    self._fetch_cursors[detector] = fetch_end
                    return
                else:
                    raise RuntimeError(
                        f"Failed to fetch GWOSC data for {detector}: {e}"
                    ) from e

        # Push Valid Data
        buf = SeriesBuffer(
            offset=Offset.fromsec(fetch_start), sample_rate=self.sample_rate, data=data
        )
        self._adapters[detector].push(buf)
        self._fetch_cursors[detector] = fetch_end

    def internal(self) -> None:
        super().internal()
        for det in self.detectors:
            adapter = self._adapters[det]
            cursor = self._fetch_cursors[det]
            needed_samples = self.num_samples(self.sample_rate)
            if cursor < self.end and adapter.size < needed_samples:
                self._fetch_next_batch(det)

    def new(self, pad: SourcePad) -> TSFrame:
        detector = next(d for d in self.detectors if self.srcs[d] is pad)
        adapter = self._adapters[detector]
        frame = self.prepare_frame(pad)

        if adapter.size > 0 and adapter.end_offset >= frame.end_offset:
            bufs = adapter.get_sliced_buffers((frame.offset, frame.end_offset))
            frame.set_buffers(bufs)
            adapter.flush_samples_by_end_offset(frame.end_offset)

        return frame
