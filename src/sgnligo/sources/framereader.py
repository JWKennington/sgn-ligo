from sgn.sources import *
from sgnts.sources import *
from .. base import *

from sgnts.base.buffer import *
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource, TSSlice, TSSlices

from gwpy.timeseries import TimeSeries

import numpy as np

from lal.utils import CacheEntry

@dataclass
class FrameReader(TSSource):
    """
    num_buffers: int
        if given, sets how many buffers will be created before setting "EOS".
        otherwise, EOS is reached when all data in the cache has been processed
    channels: tuple
        the channels of the data
    channel_name: tuple
        channel names of the data
    instruments: tuple
        instruments, should be one to one with channel names
    framecache: path
        cache file to read data from
    """

    num_buffers: int = 0
    rate: int = 2048
    channels: tuple = ()
    channel_name: tuple = ()
    instruments: tuple = ()
    ngap: int = 0
    framecache: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}
        assert len(self.channel_name) == len(self.instruments), "Instruments and channels must be one to one."

        self.shape = self.channels + (self.num_samples,)

        # load a cache file
        print(f"Loading {self.framecache}...")
        self.cache = list(map(CacheEntry, open(self.framecache)))

        # init channel name and ifo(s) as user inputs
        # FIXME: support multiple ifos/channels
        self.ifo = self.instruments[0]
        self.channel = self.channel_name[0]

        # init arrays for time and data
        self.times = np.array([])
        self.data = np.array([])

    def load_gwf_data(self):
        """
        load timeseries data from a gwf frame file
        """
        this_gwf = self.cache[0]
        segment = this_gwf.segment
        path = this_gwf.path

        data = TimeSeries.read(path, f"{self.ifo}:{self.channel}")

        assert int(data.sample_rate.value) == self.rate, "Data rate does not match requested sample rate."

        # construct gps times
        dt = data.dt.value
        times = np.arange(float(segment[0]), float(segment[1]), dt)

        data = np.array(data)

        # now that we have loaded data from this frame, remove it from the cache
        self.cache.pop(0)

        return times, data

    def new(self, pad):
        """
        New buffers are created on "pad" with an instance specific count and a
        name derived from the pad name. "EOS" is set if we have surpassed the requested
        number of buffers.
        """
        self.cnt[pad] += 1
        ngap = self.ngap
        if (ngap == -1 and np.random.rand(1) > 0.5) or (
            ngap > 0 and self.cnt[pad] % ngap == 0
        ):
            outdata = None
        else:
            if self.data.size == 0:
                # load next frame of data from disk
                self.times, self.data = self.load_gwf_data()

        # outdata is the first duration = self.num_samples / self.rate seconds of data in the frame
        outdata = self.data[:self.num_samples]
        outtimes = self.times[:self.num_samples]
        epoch = outtimes[0]

        outbuf = SeriesBuffer(
            offset=Offset.fromsec(epoch - Offset.offset_ref_t0), sample_rate=self.rate, data=outdata, shape=self.shape
        )

        # pop the used data
        self.data = self.data[self.num_samples:]
        self.times = self.times[self.num_samples:]

        # EOS condition is either that weve passed num_buffers if given, or that
        # that we have processed all data in every frame in the cache
        if self.num_buffers:
            EOS=self.cnt[pad] > self.num_buffers
        else:
            EOS = (self.data.size == 0) and (len(self.cache) == 0)

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS
        )

