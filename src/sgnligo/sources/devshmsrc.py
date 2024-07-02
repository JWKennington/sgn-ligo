from collections import deque
import inotify.adapters
import inotify.constants
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

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}

        self.shape = (self.num_samples,)

        # init inotify watcher on shared memory dir
        self.inotify = inotify.adapters.Inotify() 
        self.inotify.add_watch(self.shared_memory_dir, mask = inotify.constants.IN_CREATE)

        self.events = deque(maxlen=300)

        # initialize a start up time. This will be used so that we make sure to only process
        # new files that appear in the dirctory and to keep track of any gaps
        self.last_t0 = now()
        print(f"Start up t0: {self.last_t0}")

    def poll_dir(self, timeout=0.1):
        """
        poll directory for new files with inotify
        """
        events = self.inotify.event_gen(yield_nones=False, timeout_s=timeout)
        files = []

        ## this loop is hella slow
        for event in events:
            (_, type_names, path, filename) = event
            _, _, start, dur = from_T050017(filename)

            if start <= self.last_t0:
                print(f"Skipping file from before last t0: {filename}")
                continue
            print(f"Adding new file to queue: {filename}")

            files.append(os.path.join(path, filename))

        return files

    def new(self, pad):
        self.cnt[pad] += 1

        watch_start = now()
        while now() - watch_start < self.wait_time :
            new_files = self.poll_dir()
            if new_files:
                # add new events to the queue and break
                self.events.extend(new_files)
                break
        else:
            # FIXME: handle this, weve reached the timeout so need to send a gap buffer
            print(f"Reached 60 sec timeout with no new files in {self.shared_memory_dir}")
            raise

        # process next file
        next_file = self.events[0]
        print("Next file: ", next_file)

        # load data from the file using gwpy
        data = TimeSeries.read(next_file, f"{self.instrument}:{self.channel_name}")
        assert int(data.sample_rate.value) == self.rate, "Data rate does not match requested sample rate."
        t0 = data.t0.value
        duration = data.duration.value
        data = np.array(data)
        print(f"Buffer t0: {t0} | Time Now: {now()} | Time delay: {float(now()) - t0}")

        # once we have data, pop this file from the list
        self.events.popleft()

        # keep track of the times of the last data sent
        # FIXME: handle discontinuities here
        if self.last_t0 and t0 - self.last_t0 > duration:
            print(f"Warning: discontinuity")
        self.last_t0 = t0

        outbuf = SeriesBuffer(
            offset=Offset.fromsec(t0 - Offset.offset_ref_t0), sample_rate=self.rate, data=data, shape=data.shape
        )

        # online data is never EOS
        EOS = False

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
