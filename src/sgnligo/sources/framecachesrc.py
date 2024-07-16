import sys

from .. base import *

from decimal import Decimal

from gwpy.timeseries import TimeSeries

from lal import LIGOTimeGPS
from lal.utils import CacheEntry
from ligo import segments

import numpy as np

from sgn.sources import *
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource, TSSlice, TSSlices
from sgnts.base.buffer import *
from sgnts.sources import *


@dataclass
class FrameReader(TSSource):
    """
    channel_name: tuple
        channel names of the data
    instrument: str
        instrument, should be one to one with channel names
    framecache: path
        cache file to read data from
    gps_start_time: int
        GPS start time to analyze
    gps_end_time: int
        GPS end time to analyze
    """

    rate: int = 2048
    channel_name: str = ""
    instrument: str = ""
    framecache: str = ""
    gps_start_time: int = None
    gps_end_time: int = None

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}

        # init analysis segment
        self.analysis_seg = segments.segment(LIGOTimeGPS(self.gps_start_time), LIGOTimeGPS(self.gps_end_time))

        # load the cache file
        print(f"Loading {self.framecache}...")
        cache = list(map(CacheEntry, open(self.framecache)))

        # only keep files with the correct instrument
        cache = [c for c in cache if c.observatory in self.ifo_strings(self.instrument)]

        # only keep files that intersect the analysis segment
        self.cache = []
        for c in cache:
            try:
                intersection = self.analysis_seg & c.segment
            except ValueError:
                continue
            else:
                self.cache.append(c)

        # make sure it is sorted by gps time
        self.cache.sort(key=lambda x: x.segment[0])

        # init arrays for time and data
        self.offsets = np.array([])
        self.data = np.array([])

        # keep track of buffer epochs
        self.last_epoch = None

    @staticmethod
    def ifo_strings(ifo):
        """
        I dont know if the given self.instrument will be in the form of
        e.g., "H" or "H1", just make a tuple of both options for string comparison
        """
        if ifo[-1] == "1":
            return (ifo[0], ifo)
        else:
            return (ifo, ifo + "1")

    def load_gwf_data(self, frame):
        """
        load timeseries data from a gwf frame file
        """

        # get first cache entry
        segment = frame.segment

        intersection = self.analysis_seg & segment
        start = intersection[0]
        end = intersection[1]

        data = TimeSeries.read(frame.path, f"{self.instrument}:{self.channel_name}", start = start, end = end)
        assert int(data.sample_rate.value) == self.rate, "Data rate does not match requested sample rate."

        # reconstruct gps times with nanosecond precision and convert to offsets
        dt = Offset.fromsec(data.dt.value)
        gps_start = int(start.gpsSeconds * 10**9)
        gps_start_ns = int(start.gpsNanoSeconds)
        gps_end = int(end.gpsSeconds * 10**9)
        gps_end_ns = int(end.gpsNanoSeconds)

        offsets = np.arange(Offset.fromns(gps_start + gps_start_ns - Offset.offset_ref_t0), Offset.fromns(gps_end + gps_end_ns - Offset.offset_ref_t0), dt)

        data = np.array(data)

        return offsets, data

    def new(self, pad):
        """
        New buffers are created on "pad" with an instance specific count and a
        name derived from the pad name. "EOS" is set once we have procssed all data
        in the cache within the analysis segment.
        """
        self.cnt[pad] += 1
        # load next frame of data from disk when we have less than
        # one buffer length of data left
        if (self.data.size <= self.num_samples) and self.cache:
             offsets, data = self.load_gwf_data(self.cache[0])

             self.offsets = np.concatenate((self.offsets, offsets))
             self.data = np.concatenate((self.data, data))

             # now that we have loaded data from this frame,
             # remove it from the cache
             self.cache.pop(0)

        # outdata is the first self.num_samples of data in the frame
        outdata = self.data[:self.num_samples]
        outoffsets = self.offsets[:self.num_samples]

        epoch = int(outoffsets[0])
        outbuf = SeriesBuffer(
            offset=epoch, sample_rate=self.rate, data=outdata, shape=outdata.shape
        )

        # remove the used data
        self.data = self.data[self.num_samples:]
        self.offsets = self.offsets[self.num_samples:]

        # update last buffer epoch
        self.last_epoch = epoch

        # EOS condition is when we have processed all data intersecting
        # the analysis segment in every frame in the cache
        # set this condition on second to last buffer so it can propagate to
        # downstream elements
        EOS = (self.data.size <= self.num_samples) and (len(self.cache) == 0)

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS
        )

