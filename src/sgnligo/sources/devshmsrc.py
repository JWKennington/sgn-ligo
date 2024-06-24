from collections import deque
import inotify.adapters
from gwpy.timeseries import TimeSeries
import numpy as np
import os

from sgn.sources import *
from sgnts.sources import *
from .. base import *

from .utils import *

from sgnts.base.buffer import *
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource, TSSlice, TSSlices

import time

@dataclass
class DevShmSrc(TSSource):
    """
    shared_memory_dir: str
        Shared memory directory name (full path).  Suggestion:  /dev/shm/kafka/L1_O3ReplayMDC
    wait_time: int
        Time to wait for next file.
    watch_suffix: str
        Filename suffix to watch for.
    instrument: str
        instrument, should be one to one with channel names
    channel_name: tuple
        channel name of the data
    """

    rate: int = 2048
    channel_name: tuple = ()
    instrument: tuple = ()
    shared_memory_dir: str = None
    wait_time: int = 60
    watch_suffix: str = ".gwf"

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}

        self.shape = (self.num_samples,)

        # init inotify watcher on shared memory dir
        self.inotify = inotify.adapters.Inotify() 
        self.inotify.add_watch(self.shared_memory_dir)

        self.events = deque(maxlen=300)
        self.last_t0 = None

    def poll_dir(self, timeout=1):
        """
        load data from the next file and send it off in an output buffer
        """
        events = self.inotify.event_gen(yield_nones=False, timeout_s=timeout)
        for event in events:
            (_, type_names, path, filename) = event
            if "IN_CREATE" in type_names:
                self.events.append(os.path.join(path, filename))

    def new(self, pad):
        self.cnt[pad] += 1

        self.poll_dir()

        if self.events:
            next_file = self.events[0]
            print("Next file: ", next_file)

            # load data from the file using gwpy
            data = TimeSeries.read(next_file, f"{self.instrument}:{self.channel_name}")
            assert int(data.sample_rate.value) == self.rate, "Data rate does not match requested sample rate."
            t0 = data.t0.value
            duration = data.duration.value

            data = np.array(data)
            shape = data.shape

            print(f"Buffer t0: {t0} | Time delay: {float(now()) - t0}")

            # once we have data, pop this file from the list
            self.events.popleft()

        else:
            # FIXME: if theres not a next file, wait for 60 seconds
            # and then start pushing gap buffers
            data = None
            shape = None
            t0 = self.last_t0 + 1

        # keep track of the times of the last data sent
        # FIXME: handle discontinuities here
        if self.last_t0 and t0 - self.last_t0 > duration:
            print(f"Warning: discontinuity")
        self.last_t0 = t0

        outbuf = SeriesBuffer(
            offset=Offset.fromsec(t0 - Offset.offset_ref_t0), sample_rate=self.rate, data=data, shape=shape
        )

        # online data is never EOS
        EOS = False

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
