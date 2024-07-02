import inotify.adapters
import inotify.constants
from inotify_simple import INotify, flags
from gwpy.timeseries import TimeSeries
import numpy as np
import os

from sgn.sources import *
from sgnts.sources import *
from .. base import *

import threading

from .utils import *

from sgnts.base.buffer import *
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource, TSSlice, TSSlices

import queue

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
    watch_suffix: str
        Filename suffix to watch for.
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

        self.queue = queue.Queue()

        # initialize a start up time. This will be used so that we make sure to only process
        # new files that appear in the dirctory and to keep track of any gaps
        self.last_t0 = now()
        print(f"Start up t0: {self.last_t0}")

        # Create the inotify handler
        observer = threading.Thread(
            target=self.monitor_dir,
            args=(self.queue, self.shared_memory_dir)
        )

        # Start the observer and set the stop attribute
        observer.stop = False
        observer.start()

    def monitor_dir(self, queue, watch_dir):
        """
        poll directory for new files with inotify
        """
        # init inotify watcher on shared memory dir
        i = INotify()
        i.add_watch(watch_dir, flags.CLOSE_WRITE | flags.MOVED_TO)

        # Get the current thread
        t = threading.currentThread()

        # Check if this thread should stop
        while not t.stop:
            # Loop over the events and check when a file has been created
            for event in i.read(timeout=1):
                # directory was removed, so the corresponding watch was
                # also removed
                if flags.IGNORED in flags.from_mask(event.mask):
                    break

                # ignore temporary files
                filename = event.name
                extension = os.path.splitext(filename)[1]
                if not (extension == self.watch_suffix):
                    continue

                # Add the filename to the queue
                queue.put(os.path.join(watch_dir, filename))

        # Remove the watch
        i.rm_watch(watch_dir)

    def new(self, pad):
        self.cnt[pad] += 1

        # process next file
        try:
            next_file = self.queue.get(timeout=1)
        except queue.Empty:
            # send a gap buffer
            # FIXME: should also fail at some point after a long time with no new files?
            print("Queue is empty.")
            t0 = self.last_t0 + 1 # FIXME

            outbuf = SeriesBuffer(
                offset=Offset.fromsec(t0 - Offset.offset_ref_t0), sample_rate=self.rate, data=None, is_gap=True
            )

        else:
            next_file = self.queue.get()
            print("Next file: ", next_file)

            # load data from the file using gwpy
            data = TimeSeries.read(next_file, f"{self.instrument}:{self.channel_name}")
            assert int(data.sample_rate.value) == self.rate, "Data rate does not match requested sample rate."
            t0 = data.t0.value
            duration = data.duration.value
            data = np.array(data)
            print(f"Buffer t0: {t0} | Time Now: {now()} | Time delay: {float(now()) - t0}")

            # keep track of the times of the last data sent
            # FIXME: handle discontinuities here
            if self.last_t0 and t0 - self.last_t0 > duration:
                print(f"Warning: discontinuity. last t0 = {self.last_t0} | t0 = {t0} | duration = {duration}")
            self.last_t0 = t0

            outbuf = SeriesBuffer(
                offset=Offset.fromsec(t0 - Offset.offset_ref_t0), sample_rate=self.rate, data=data, shape=data.shape
            )

        # online data is never EOS
        # FIXME but maybe there should be some kind of graceful shutdown
        EOS = False

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
