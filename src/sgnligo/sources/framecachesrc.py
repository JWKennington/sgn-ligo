"""Read GW frame files from a frame cache file."""

# Copyright (C) 2024 Becca Ewing, Yun-Jing Huang

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import igwn_segments as segments
import numpy as np
from gwpy.timeseries import TimeSeriesDict
from lal import LIGOTimeGPS
from lal.utils import CacheEntry
from sgn.base import SourcePad
from sgnts.base import Audioadapter, Offset, SeriesBuffer, TSFrame, TSSource

logger = logging.getLogger("sgn")


@dataclass
class FrameReader(TSSource):
    """Read GW frame files from a frame cache file

    Args:
        channel_names:
            list[str], a list of channel names of the data, e.g.,
            ["L1:GWOSC-16KHZ_R1_STRAIN", "L1:GWOSC-16KHZ_R1_DQMASK"]. Source pads will
            be automatically generated for each channel, with channel name as pad name.
        framecache:
            str, cache file to read data from
        instrument:
            str, optional, only read gwf files from this instrument
    """

    channel_names: Optional[list[str]] = None
    framecache: Optional[str] = None
    instrument: Optional[str] = None

    def __post_init__(self):
        if len(self.source_pad_names) > 0:
            if self.source_pad_names != tuple(self.channel_names):
                raise ValueError("Expected source pad names to match channel names")
        else:
            print(f"Generating source pads from channel names {self.channel_names}...")
            self.source_pad_names = tuple(self.channel_names)

        # Sanity check
        assert isinstance(self.channel_names, list)
        assert isinstance(self.framecache, str)

        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}

        self.last_epoch = self.t0

        if self.instrument is not None:
            for c in self.channel_names:
                assert (
                    self.instrument in c
                ), "instrument provided does not match channel name"

        # init analysis segment
        self.analysis_seg = segments.segment(
            LIGOTimeGPS(self.t0), LIGOTimeGPS(self.end)
        )

        # load the cache file
        print(f"Loading {self.framecache}...")
        cache = list(map(CacheEntry, open(self.framecache)))

        if self.instrument is not None:
            # only keep files with the correct instrument
            # sometimes there are frame files with multiple instruments, in that case
            # don't filter by instrument
            cache = [
                c
                for c in cache
                for o in c.observatory
                if o in self.ifo_strings(self.instrument)
            ]

        # only keep files that intersect the analysis segment
        self.cache = []
        for c in cache:
            try:
                self.analysis_seg & c.segment
            except ValueError:
                continue
            else:
                self.cache.append(c)

        # make sure it is sorted by gps time
        self.cache.sort(key=lambda x: x.segment[0])

        # Check if there are missing segments
        segment_remaining = self.analysis_seg
        missing_segments = []
        for c in self.cache:
            if segment_remaining in c.segment:
                # the cache contains all the rest of the proposed segment
                segment_remaining = segments.segment(0, 0)
            elif segment_remaining[0] < c.segment[0]:
                # there is a discontinuity
                missing_segments.append(
                    segments.segment(segment_remaining[0], c.segment[0])
                )
                if c.segment[1] <= segment_remaining[1]:
                    segment_remaining = segments.segment(
                        c.segment[1], segment_remaining[1]
                    )
                else:
                    segment_remaining = segments.segment(0, 0)
            else:
                segment_remaining -= c.segment

        if segment_remaining:
            missing_segments.append(segment_remaining)

        if missing_segments:
            logger.getChild(self.name).warning(
                "%s has missing segment %s, padding with gaps",
                self.name,
                missing_segments,
            )

        self.A = {c: Audioadapter() for c in self.channel_names}

        # load first segment of data to read sample rate
        self.rates = {}
        self.load_gwf_data(self.cache[0])
        print(f"Sample rate per channel: {self.rates}")

        # FIXME: support multiple pads with different sample rates
        # FIXME: do we want multiple channels in one buffer?
        for pad in self.source_pads:
            self.set_pad_buffer_params(
                sample_shape=(), rate=self.rates[self.rsrcs[pad]], pad=pad
            )

        # now that we have loaded data from this frame,
        # remove it from the cache
        self.cache.pop(0)

    @staticmethod
    def ifo_strings(ifo: str) -> tuple[str, str]:
        """Make a tuple of possible ifo strings, with and without the "1" at the end.
        I dont know if the given self.instrument will be in the form of e.g., "H" or
        "H1", just make a tuple of both options for string comparison

        Args:
            ifo:
                str, the ifo name, e.g., "H" or "H1"

        Returns:
            tuple[str, str], a tuple of the ifo name with and without the "1" at the end
        """
        if ifo[-1] == "1":
            return (ifo[0], ifo)
        else:
            return (ifo, ifo + "1")

    def load_gwf_data(self, frame_file: CacheEntry):
        """Load timeseries data from a gwf frame file.

        Args:
            frame_file:
                CacheEntry, the gwf frame file to read timeseries data from

        Returns:
            dict[str, np.ndarray], a dictionary with channel names as keys and
            numpy arrays of timeseries data as values
        """

        # get first cache entry
        segment = frame_file.segment

        intersection = self.analysis_seg & segment
        start = intersection[0]
        end = intersection[1]

        # FIXME: check for gaps
        data_dict = TimeSeriesDict.read(
            frame_file.path, self.channel_names, start=start, end=end
        )

        if len(self.rates) == 0:
            for channel, data in data_dict.items():
                self.rates[channel] = int(data.sample_rate.value)

        for channel, data in data_dict.items():
            if self.last_epoch < start:
                print(
                    f"Unepected epoch: {start}, expected: {self.last_epoch}, sending "
                    "gap buffer"
                )
                self.A[channel].push(
                    SeriesBuffer(
                        offset=Offset.fromsec(float(self.last_epoch)),
                        sample_rate=self.rates[channel],
                        data=None,
                        shape=(int((start - self.last_epoch) * self.rates[channel]),),
                    )
                )
            elif self.last_epoch > start:
                raise ValueError(
                    f"Unepected epoch: {start}, expected: {self.last_epoch}, sending "
                    "gap buffer"
                )
            self.A[channel].push(
                SeriesBuffer(
                    offset=Offset.fromsec(float(start)),
                    sample_rate=self.rates[channel],
                    data=np.array(data),
                )
            )

        self.last_epoch = end

    def internal(self) -> None:
        """Check if we need to read the next gw frame file in the cache. All channels
        are read at once.
        """
        super().internal()

        # load next frame of data from disk when we have less than
        # one buffer length of data left
        read_new = False
        for channel, adapter in self.A.items():
            if adapter.size < self.num_samples(self.rates[channel]):
                read_new = True
                break

        if read_new and self.cache:
            # Read multiple channels at once
            self.load_gwf_data(self.cache[0])

            # now that we have loaded data from this frame,
            # remove it from the cache
            self.cache.pop(0)

    def new(self, pad: SourcePad) -> TSFrame:
        """New frames are created on "pad" with an instance specific count and a name
        derived from the channel name. "EOS" is set once we have procssed all data in
        the cache within the analysis segment.

        Args:
            pad:
                SourcePad, the pad for which to produce a new TSFrame

        Returns:
            TSFrame, the TSFrame that carries a list of SeriesBuffers
        """

        self.cnt[pad] += 1

        channel = self.rsrcs[pad]

        metadata = {"cnt": self.cnt[pad], "name": "'%s'" % pad.name}

        frame = self.prepare_frame(pad, metadata=metadata)

        if self.A[channel].end_offset >= frame.end_offset:
            bufs = self.A[channel].get_sliced_buffers((frame.offset, frame.end_offset))

            frame.set_buffers(bufs)

            self.A[channel].flush_samples_by_end_offset(frame.end_offset)

        return frame
