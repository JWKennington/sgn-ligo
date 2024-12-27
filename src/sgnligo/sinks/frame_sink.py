"""This module contains the FrameSink class, which writes time series data to .gwf files.
The formatting is done using the gwpy library.
"""

from dataclasses import dataclass, field
from typing import Sequence

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sgn.base import get_sgn_logger
from sgnts.base import AdapterConfig, Offset, TSSink

LOGGER = get_sgn_logger(__name__)

# filename format parameters
FILENAME_PARAMS = (
    "instruments",
    "gps_start_time",
    "duration",
)


@dataclass
class FrameSink(TSSink):
    """A sink element that writes time series data to file

    Args:
        channels:
            Sequence[str], the instruments to write to the file
        duration:
            int, the duration of the data to write to the file
        path:
            str, the path to write the frame files to.  The file name
            must contain the following format parameters (in curly braces):
            - {instruments}, the sorted list of instruments inferred from
                the included channel names (e.g. "H1" for "H1:GDS-CAL...")
            - {gps_start_time}, the start time of the data in GPS seconds
            - {duration}, the duration of the data in seconds
            The extension on the the path determines the output file
            type.  Currently ".gwf" and ".hdf5" are supported.
            default: "{instruments}-{gps_start_time}-{duration}.gwf"

    This sink element automatically creates an AdapterConfig for
    buffering the data needed to create frames of the requested
    duration.  Attempting to provide an AdapterConfig will produce a
    RuntimeError.

    """

    channels: Sequence[str] = field(default_factory=list)
    duration: int = 0
    path: str = "{instruments}-{gps_start_time}-{duration}.gwf"

    def __post_init__(self):
        """Post init for setting up the FrameSink"""
        # enforce channels = sink_pad_names
        self.sink_pad_names = self.channels

        # setup the adapter config for the audioadapter
        if self.adapter_config is not None:
            raise RuntimeError(
                "specifying AdapterConfig is not supported in this element as they are handled internally."
            )
        stride = Offset.fromsec(self.duration)
        self.adapter_config = AdapterConfig(stride=stride)

        # Call parent post init
        super().__post_init__()

        # Check valid duration
        if not isinstance(self.duration, int) or self.duration <= 0:
            raise ValueError(f"Duration must be an positive integer, got {self.duration}")

        # Check path contains parameters for duration and gps_start_time
        for param in FILENAME_PARAMS:
            if f"{{{param}}}" not in self.path:
                raise ValueError(f"Path must contain parameter {{{param}}}")

        self._instruments_str = "".join(
            sorted({chan.split(":")[0] for chan in self.channels})
        )

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

        # Channels
        for name, pad in zip(self.sink_pad_names, self.sink_pads):

            # Data products
            frame = self.preparedframes[pad]
            if frame is None or (frame is not None and frame.is_gap):
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
            t0 = Offset.offset_ref_t0 + Offset.tosec(data.offset)
            # data times should be lined up with second boundaries
            assert int(t0) == t0, f"t0 is not on second boundary: {t0}"

            # TimeSeries
            ts = TimeSeries(
                data.data,
                t0=t0,
                sample_rate=data.sample_rate,
                channel=name,
            )

            # Add to TimeSeriesDict
            ts_dict[name] = ts

        # Format filename
        filename = self.path.format(
            instruments=self._instruments_str,
            gps_start_time=f"{t0:0=10.0f}",
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
