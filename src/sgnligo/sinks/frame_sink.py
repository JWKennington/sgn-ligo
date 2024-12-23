"""This module contains the FrameSink class, which writes time series data to .gwf files.
The formatting is done using the gwpy library.
"""

from dataclasses import dataclass, field
from typing import Sequence

from gwpy.timeseries import TimeSeries, TimeSeriesDict

from sgn.base import SGN_LOG_LEVELS, get_sgn_logger
from sgnts.base import Offset, TSSink

# TODO remove the SGN_LOG_LEVELS once
#  https://git.ligo.org/greg/sgn/-/merge_requests/65 is merged
LOGGER = get_sgn_logger(__name__, SGN_LOG_LEVELS)
FILENAME_PARAM_CHANNELS = "channels"
FILENAME_PARAM_GPS_START_TIME = "gps_start_time"
FILENAME_PARAM_DURATION = "duration"
FILENAME_PARAMS = (
    FILENAME_PARAM_CHANNELS,
    FILENAME_PARAM_GPS_START_TIME,
    FILENAME_PARAM_DURATION,
)


@dataclass
class FrameSink(TSSink):
    """A sink element that writes time series data to .gwf file

    Args:
        channels:
            Sequence[str], the instruments to write to the file
        duration:
            int, the duration of the data to write to the file
        path:
            str, the path to write the file to. Must contain parameters for:
                - {channels}, the sorted list of instruments inferred from the included
                      channels (e.g. "H1" or "H1L1")
                - {gps_start_time}, the start time of the data in GPS time
                - {duration}, the duration of the data in seconds

    Usage:
        Must use with an AdapterConfig with the duration_offsets parameter matching
        the duration of the FrameSink. For example, if the FrameSink duration is 3
        seconds, the AdapterConfig should have duration_offsets=Offset.fromsec(3).
        ```python
        from sgnts.base import AdapterConfig, Offset
        from sgnligo.sinks import FrameSink

        duration = 3  # seconds
        duration_offsets = Offset.fromsec(duration)
        snk = FrameSink(
            name="snk",
            sink_pad_names=(
                "H1",
                "L1",
            ),
            duration=duration,
            path=path_format.as_posix(),
            adapter_config=AdapterConfig(stride=duration_offsets),
        )
        ```
    """

    channels: Sequence[str] = field(default_factory=list)
    duration: int = 0
    path: str = "{channels}-{gps_start_time}-{duration}.gwf"

    def __post_init__(self):
        """Post init for setting up the FrameSink"""
        # enforce channels = sink_pad_names
        self.sink_pad_names = self.channels

        # Call parent post init
        super().__post_init__()

        # Check valid duration
        if not isinstance(self.duration, int) and self.duration > 0:
            raise ValueError(f"Duration must be an integer, got {self.duration}")

        # Check path contains parameters for duration and gps_start_time
        for param in FILENAME_PARAMS:
            if f"{{{param}}}" not in self.path:
                raise ValueError(f"Path must contain parameter {{{param}}}")

    def write(self):
        """Write a gwf file using the gwpy library. This method gets called by the
        internal pad, and only writes to the file if there are enough samples in the
        audioadapters.

        Notes:
            Single-Segment:
                Currently, the FrameSink writes a single segment of data to a .gwf file.
                Future versions will extend this to write multiple segments.
        """
        # Initialize TimeSeriesDict to hold all channels
        ts_dict = TimeSeriesDict()
        t0 = None

        # Channels
        for name, pad in zip(self.sink_pad_names, self.sink_pads):

            # Data products
            frame = self.preparedframes[pad]
            if frame is None or (frame is not None and frame.is_gap):
                LOGGER.warning("Gap detected in data. Skipping...")
                return

            # Load first buffer
            # TODO fix this indexing to handle multiple buffers as multiple segments
            data = frame.buffers[0]

            # TODO check for above todo, For now, check if the buffer has enough data
            #  for the duration, later we'll need to cumulate check across multiple
            #  segments
            exp_samples = self.duration * data.sample_rate
            if data.samples < exp_samples:
                LOGGER.warning(
                    f"Data does not contain enough samples for duration {self.duration}. Skipping..."
                )
                return

            # Compute start time in floating seconds
            # TODO this could be a new method on the SeriesBuffer class
            t0_s = Offset.offset_ref_t0 + Offset.tosec(data.offset)

            # TimeSeries
            ts = TimeSeries(
                data.data,
                t0=t0_s,
                sample_rate=data.sample_rate,
                channel=name,
            )

            # Add to TimeSeriesDict
            ts_dict[name] = ts

            # Track start time for filename
            if t0 is None:
                t0 = data.t0

        # Format filename
        filename = self.path.format(
            channels="".join(sorted(self.channels)),
            gps_start_time=t0,
            duration=self.duration,
        )

        # Write to frame
        LOGGER.info(f"Writing file {filename}...")
        ts_dict.write(filename)

    def internal(self):
        """Internal method, checks if sufficient data is present in the audioadapters to
        write to a file.

        Args:
            pad:
                SinkPad, the pad to check for enough samples
        """
        super().internal()

        # Check EOS
        for spad in self.sink_pads:
            frame = self.preparedframes[spad]
            if frame is not None and frame.EOS:
                self.mark_eos(spad)

        self.write()
