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
    buffer_duration: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}
        self.shape = (self.num_samples,)
        self.queue = queue.Queue()

        # initialize a start up time. This will be used so that we make sure to only process
        # new files that appear in the dirctory and to keep track of any gaps
        self.last_t0 = float(now())
        print(f"Start up t0: {self.last_t0}")

        # Create the inotify handler
        self.observer = threading.Thread(
            target=self.monitor_dir,
            args=(self.queue, self.shared_memory_dir)
        )

        # Start the observer and set the stop attribute
        self.observer.stop = False
        self.observer.start()

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

                # parse filename for the t0, we dont want to
                # add files to the queue if they arrive late
                _, _, t0, _ = from_T050017(filename)
                if t0 < self.last_t0:
                    pass
                else:
                    # Add the filename to the queue
                    queue.put(os.path.join(watch_dir, filename))

        # Remove the watch
        i.rm_watch(watch_dir)

    def new(self, pad):
        self.cnt[pad] += 1
        outbuf = []

        # process next file
        try:
            # Im not sure what the right timeout here is,
            # but I want to avoid a situation where get()
            # times out just before the new file arrives and
            # prematurely decides to send a gap buffer
            next_file = self.queue.get(timeout=2)
        except queue.Empty:
            if now() - self.last_t0 >= self.wait_time:
                self.observer.stop = True
                raise ValueError(f"Reached {self.wait_time} seconds with no new files in {self.shared_memory_dir}, exiting.")
            elif now() - self.last_t0 >= self.buffer_duration:
                # send a gap buffer
                if self.cnt[pad] == 1:
                    # send the first gap buffer starting from the program start up time
                    t0 = self.last_t0
                else:
                    # send subsequent gaps at self.buffer_duration intervals
                    t0 = self.last_t0 + self.buffer_duration
                shape = (self.rate * self.buffer_duration,)
                print(f"Queue is empty, sending a gap buffer at t0: {t0}")
                outbuf.append(SeriesBuffer(
                    offset=Offset.fromsec(t0 - Offset.offset_ref_t0), sample_rate=self.rate, data=None, shape=shape
                ))

                # update last t0
                self.last_t0 = t0
        else:
            # load data from the file using gwpy
            data = TimeSeries.read(next_file, f"{self.instrument}:{self.channel_name}")
            assert int(data.sample_rate.value) == self.rate, "Data rate does not match requested sample rate."
            t0 = data.t0.value
            duration = data.duration.value
            print(f"Buffer t0: {t0} | Time Now: {now()} | Time delay: {float(now()) - t0} | shape: {data.shape} | sample rate: {data.sample_rate.value}")
            data = np.array(data)

            if t0 - self.last_t0 > duration:
                # if there is a discontinuity we need to send
                # a gap buffer to cover the missing data
                # FIXME: maybe we need to make several buffers
                # all with duration = self.buffer_duration instead
                # of making one large gap buffer here if
                # gap_duration > self.buffer_duration? Im not sure
                # if anything will break downstream
                print(f"Warning: discontinuity. last t0 = {self.last_t0} | t0 = {t0} | duration = {duration}")
                gap_duration = t0 - (self.last_t0 + self.buffer_duration)
                shape = (int(gap_duration / self.buffer_duration) * self.rate,)
                outbuf.append(SeriesBuffer(
                    offset=Offset.fromsec(self.last_t0 - Offset.offset_ref_t0), sample_rate=self.rate, data=None, shape = shape
                ))
            else:
                self.last_t0 = t0
                outbuf.append(SeriesBuffer(
                    offset=Offset.fromsec(t0 - Offset.offset_ref_t0), sample_rate=self.rate, data=data, shape=data.shape
                ))

        # online data is never EOS
        # FIXME but maybe there should be some kind of graceful shutdown
        EOS = False

        return TSFrame(
            buffers=outbuf,
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
