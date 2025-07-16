"""This module contains the FrameSink class.

Writes time series data to .gwf files.
The formatting is done using the gwpy library.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from sgn.base import get_sgn_logger
from sgnts.base import AdapterConfig, Offset, TSSink

LOGGER = get_sgn_logger(__name__)

# filename format parameters
FILENAME_PARAMS = (
    "instruments",
    "description",
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
            - {description}, the description string for the frame
            - {gps_start_time}, the start time of the data in GPS seconds
            - {duration}, the duration of the data in seconds
            The extension on the the path determines the output file
            type.  Currently ".gwf" and ".hdf5" are supported.
            default: "{instruments}-{description}-{gps_start_time}-{duration}.gwf"
        force:
            bool, whether to overwrite existing files. Default: False
        description:
            str, description string to include in the filename. Default: "SGN"
        history_seconds:
            int or None, when set to a positive value, enables circular buffer
            mode that automatically deletes frame files older than this many
            seconds. Default: None (disabled)
        cleanup_interval:
            int, how often (in seconds) to check for old files to delete when
            circular buffer mode is enabled. Default: 300

    This sink element automatically creates an AdapterConfig for
    buffering the data needed to create frames of the requested
    duration.  Attempting to provide an AdapterConfig will produce a
    RuntimeError.

    """

    channels: Sequence[str] = field(default_factory=list)
    duration: int = 0
    path: str = "{instruments}-{description}-{gps_start_time}-{duration}.gwf"
    force: bool = False
    description: str = "SGN"
    history_seconds: Optional[int] = None
    cleanup_interval: int = 300

    def __post_init__(self):
        """Post init for setting up the FrameSink"""
        # enforce channels = sink_pad_names
        self.sink_pad_names = self.channels

        # setup the adapter config for the audioadapter
        if self.adapter_config is not None:
            raise RuntimeError(
                "specifying AdapterConfig is not supported in this element "
                "as they are handled internally."
            )
        stride = Offset.fromsec(self.duration)
        self.adapter_config = AdapterConfig(stride=stride)

        # Call parent post init
        super().__post_init__()

        # Check valid duration
        if not isinstance(self.duration, int) or self.duration <= 0:
            raise ValueError(
                f"Duration must be an positive integer, got {self.duration}"
            )

        # Check path contains parameters for duration and gps_start_time
        for param in FILENAME_PARAMS:
            if f"{{{param}}}" not in self.path:
                raise ValueError(f"Path must contain parameter {{{param}}}")

        self._instruments_str = "".join(
            sorted({chan.split(":")[0] for chan in self.channels})
        )

        # Initialize circular buffer tracking
        self._last_cleanup_time = None

    def _write_tsd(self, tsd):
        """Write a gwf file using the gwpy library"""
        span = tsd.span
        t0 = span.start
        assert int(t0) == t0
        t0 = int(t0)
        duration = span.end - span.start
        assert int(duration) == duration
        duration = int(duration)

        outpath = Path(
            self.path.format(
                instruments=self._instruments_str,
                gps_start_time=f"{t0:0=10.0f}",
                duration=duration,
                description=self.description,
            )
        )

        if outpath.exists():
            if self.force:
                outpath.unlink()
            else:
                raise FileExistsError(f"output file exists: {outpath}")

        LOGGER.info("Writing file %s...", outpath)
        tsd.write(outpath)

        # Check if circular buffer cleanup is needed
        if self.history_seconds is not None:
            current_time = time.time()
            if (
                self._last_cleanup_time is None
                or current_time - self._last_cleanup_time >= self.cleanup_interval
            ):
                self._cleanup_old_frames()
                self._last_cleanup_time = current_time

    def internal(self):
        """Internal method, checks if sufficient data is present in the audioadapters to
        write to a file.

        Args:
            pad:
                SinkPad, the pad to check for enough samples
        """
        super().internal()

        # Initialize TimeSeriesDict to hold all channels
        tsd = TimeSeriesDict()

        # Channels
        for name, pad in zip(self.sink_pad_names, self.sink_pads):

            # Data products
            frame = self.preparedframes[pad]
            if frame is None:
                return
            else:
                if frame.EOS:
                    self.mark_eos(pad)
                if frame.is_gap:
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
                    "Data does not contain enough samples for duration %d. Skipping",
                    self.duration,
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
            tsd[name] = ts

        self._write_tsd(tsd)

    def _cleanup_old_frames(self):
        """Clean up old frame files beyond the history window"""
        if self.history_seconds is None or self.history_seconds <= 0:
            return

        # Get current GPS time
        from sgnligo.base import now

        current_gps_time = float(now())

        # Get the directory from the path template
        path_dir = os.path.dirname(self.path)
        if not path_dir:
            path_dir = "."

        # Create a pattern to match frame files
        # Replace format placeholders with wildcards
        filename_pattern = os.path.basename(self.path)
        for param in FILENAME_PARAMS:
            filename_pattern = filename_pattern.replace(f"{{{param}}}", "*")

        pattern = os.path.join(path_dir, filename_pattern)

        # Find all matching frame files
        frame_files = glob.glob(pattern)

        deleted_count = 0
        for filepath in frame_files:
            try:
                # Extract GPS time and duration from filename
                basename = os.path.basename(filepath)

                # Try to parse GPS time and duration from filename
                # Assuming format like "H1-SGN-1234567890-32.gwf"
                parts = basename.split("-")
                if len(parts) >= 4:
                    try:
                        gps_start = float(parts[-2])
                        duration_str = parts[-1].split(".")[0]
                        frame_duration = float(duration_str)

                        # Calculate when this frame ends
                        frame_end_time = gps_start + frame_duration

                        # Delete if frame is older than history window
                        if frame_end_time < (current_gps_time - self.history_seconds):
                            os.remove(filepath)
                            deleted_count += 1
                            LOGGER.debug("Deleted old frame file: %s", filepath)
                    except (ValueError, IndexError):
                        # Skip files that don't match expected format
                        continue
            except Exception as e:
                LOGGER.warning("Error processing file %s", filepath, exc_info=e)

        if deleted_count > 0:
            LOGGER.info(
                "Circular buffer cleanup: deleted %d old frame files", deleted_count
            )
        elif len(frame_files) > 0:
            LOGGER.debug(
                "Circular buffer cleanup: checked %d files, none old enough to delete",
                len(frame_files),
            )
