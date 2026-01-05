"""GWOSCSource: Fetch data from the Gravitational Wave Open Science Center.

Uses GWpy's TimeSeries.fetch_open_data() with TSResourceSource for
streaming pipeline integration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.error import HTTPError, URLError

import numpy as np
from sgnts.base import Offset, SeriesBuffer, TSResourceSource
from sgnts.base.time import Time

logger = logging.getLogger("sgn")


@dataclass
class GWOSCSource(TSResourceSource):
    """Source for fetching data from the Gravitational Wave Open Science Center.

    This wraps GWpy's TimeSeries.fetch_open_data() for streaming pipeline use.
    Data is fetched in chunks in a worker thread and streamed through the pipeline.

    Args:
        detector:
            Detector identifier: "H1" (Hanford), "L1" (Livingston), or "V1" (Virgo)
        start_time:
            GPS start time in seconds
        duration:
            Duration to fetch in seconds
        target_sample_rate:
            Desired sample rate (default 4096). GWOSC provides 4096 Hz and 16384 Hz.
        chunk_size:
            Size of chunks to fetch at a time in seconds (default 64).
            Larger chunks are more efficient but use more memory.
        channel:
            Optional channel name override. If None, uses default GWOSC channel.
        cache:
            Whether to cache downloaded data (default True)
        timeout:
            Network timeout in seconds for downloading data (default 120).
            The default astropy timeout of 10s is often too short for GWOSC.
        max_retries:
            Maximum number of retry attempts per chunk on network errors (default 3).
            Uses exponential backoff between retries.

    Example:
        >>> from sgnligo.gwpy.sources import GWOSCSource
        >>> from sgn.apps import Pipeline
        >>>
        >>> # Fetch GW150914 data
        >>> source = GWOSCSource(
        ...     name="H1_GWOSC",
        ...     source_pad_names=("strain",),
        ...     detector="H1",
        ...     start_time=1126259462,
        ...     duration=32,
        ... )
        >>>
        >>> # Use in pipeline
        >>> pipeline = Pipeline()
        >>> pipeline.insert(source, ...)
        >>> pipeline.run()

    Note:
        - Requires internet connection to fetch data from GWOSC
        - Data is fetched in chunks to enable streaming
        - GPS times must be within GWOSC's available data range
        - See https://gwosc.org for available data and events
    """

    detector: str = "H1"
    target_sample_rate: int = 4096
    chunk_size: int = 64
    channel: Optional[str] = None
    cache: bool = True
    timeout: int = 120
    max_retries: int = 3

    # Internal tracking
    _fetch_complete: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Validate detector
        valid_detectors = ("H1", "L1", "V1", "G1", "K1")
        if self.detector not in valid_detectors:
            raise ValueError(
                f"Invalid detector '{self.detector}'. "
                f"Must be one of {valid_detectors}"
            )

        # Validate times
        if self.start_time is None:
            raise ValueError("start_time is required for GWOSCSource")
        if self.duration is None:
            raise ValueError("duration is required for GWOSCSource")

        # Convert duration to nanoseconds for parent class
        duration_ns = int(self.duration * Time.SECONDS)

        # Store original duration in seconds for worker
        self._duration_sec = self.duration
        self.duration = duration_ns

        super().__post_init__()

    def worker_process(
        self,
        context,
        srcs,
        detector,
        start_time,
        _duration_sec,
        target_sample_rate,
        chunk_size,
        channel,
        cache,
        timeout,
        max_retries,
    ):
        """Fetch data from GWOSC in the worker thread.

        This method runs in a separate thread and fetches data in chunks,
        sending each chunk to the main thread via the output queue.

        Note: This is not a staticmethod because the framework calls
        worker_method(temp_instance, context, **worker_params) which
        requires a self parameter to receive temp_instance.

        Args:
            self: Temporary instance (not fully initialized, don't use)
            context: WorkerContext with queues and events
            srcs: Source pads dictionary from instance
            detector: Detector identifier (H1, L1, V1, etc.)
            start_time: GPS start time in seconds
            _duration_sec: Duration in seconds
            target_sample_rate: Target sample rate in Hz
            chunk_size: Chunk size in seconds
            channel: Optional channel name override
            cache: Whether to cache downloaded data
            timeout: Network timeout in seconds
            max_retries: Maximum retry attempts per chunk
        """
        from gwpy.timeseries import TimeSeries

        # Set astropy timeout for downloads
        try:
            from astropy.utils.data import conf as astropy_conf

            astropy_conf.remote_timeout = timeout
            logger.debug("GWOSCSource: Set astropy timeout to %ds", timeout)
        except ImportError:
            logger.warning("GWOSCSource: Could not set astropy timeout")

        # Get the source pad
        pad = list(srcs.values())[0]

        current_time = start_time
        end_time = start_time + _duration_sec

        logger.info(
            "GWOSCSource: Starting fetch for %s from GPS %s to %s",
            detector,
            start_time,
            end_time,
        )

        while current_time < end_time and not context.should_stop():
            # Calculate chunk end time
            chunk_end = min(current_time + chunk_size, end_time)

            # Retry loop with exponential backoff
            ts = None
            last_error = None
            for attempt in range(max_retries):
                if context.should_stop():
                    break

                try:
                    # Fetch data from GWOSC
                    if attempt == 0:
                        logger.debug(
                            "GWOSCSource: Fetching %s [%s, %s]",
                            detector,
                            current_time,
                            chunk_end,
                        )
                    else:
                        logger.info(
                            "GWOSCSource: Retry %d/%d for %s [%s, %s]",
                            attempt + 1,
                            max_retries,
                            detector,
                            current_time,
                            chunk_end,
                        )

                    if channel:
                        # Use specified channel
                        ts = TimeSeries.get(
                            channel,
                            current_time,
                            chunk_end,
                            verbose=False,
                        )
                    else:
                        # Use GWOSC open data
                        ts = TimeSeries.fetch_open_data(
                            detector,
                            current_time,
                            chunk_end,
                            sample_rate=target_sample_rate,
                            cache=cache,
                            verbose=False,
                        )
                    break  # Success, exit retry loop

                except (OSError, HTTPError, URLError) as e:
                    # Network errors - retry with backoff
                    last_error = e
                    if attempt < max_retries - 1:
                        backoff = 2**attempt  # 1s, 2s, 4s...
                        logger.warning(
                            "GWOSCSource: Network error fetching [%s, %s]: %s. "
                            "Retrying in %ds...",
                            current_time,
                            chunk_end,
                            type(e).__name__,
                            backoff,
                        )
                        time.sleep(backoff)
                    else:
                        logger.exception(
                            "GWOSCSource: Failed to fetch [%s, %s] after %d attempts",
                            current_time,
                            chunk_end,
                            max_retries,
                        )

                except Exception as e:
                    # Other errors - don't retry
                    last_error = e
                    logger.exception(
                        "GWOSCSource: Error fetching [%s, %s]",
                        current_time,
                        chunk_end,
                    )
                    break

            if ts is not None:
                # Convert to SeriesBuffer
                data = np.asarray(ts.value)
                actual_rate = int(ts.sample_rate.value)
                offset = Offset.fromsec(float(ts.t0.value))

                buf = SeriesBuffer(
                    offset=offset,
                    sample_rate=actual_rate,
                    data=data,
                    shape=data.shape,
                )

                # Send to main thread
                context.output_queue.put((pad, buf))

                logger.debug(
                    "GWOSCSource: Fetched %d samples (%ss)",
                    len(data),
                    chunk_end - current_time,
                )
            else:
                # Send gap buffer for failed chunk
                logger.warning(
                    "GWOSCSource: Sending gap for [%s, %s] due to: %s",
                    current_time,
                    chunk_end,
                    last_error,
                )
                gap_samples = int((chunk_end - current_time) * target_sample_rate)
                offset = Offset.fromsec(current_time)
                gap_buf = SeriesBuffer(
                    offset=offset,
                    sample_rate=target_sample_rate,
                    data=None,
                    shape=(gap_samples,),
                )
                context.output_queue.put((pad, gap_buf))

            current_time = chunk_end

        logger.info("GWOSCSource: Fetch complete for %s", detector)
