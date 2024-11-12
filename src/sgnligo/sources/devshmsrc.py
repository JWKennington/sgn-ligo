import os
import queue
import threading
from dataclasses import dataclass

import numpy
from gwpy.timeseries import TimeSeriesDict

try:
    from inotify_simple import INotify, flags
except ImportError:
    INotify = flags = None

from sgn.base import InternalPad, SourcePad
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource

from sgnligo.base import from_T050017, now


@dataclass
class LastBuffer:
    """Keep a record of the last buffer sent

    Args:
        t0:
            int, start time of the buffer, in seconds
        end:
            int, end time of the buffer, in seconds
        is_gap:
            bool, flag the buffer as gap
    """

    t0: int
    end: int
    is_gap: bool


@dataclass
class DevShmSrc(TSSource):
    """Source element to read low-latency data streamed to /dev/shm in real-time

    Args:
        shared_memory_dir:
            str, shared memory directory name (full path). Suggestion:
            /dev/shm/kafka/L1_O3ReplayMDC
        instrument:
            str, instrument, should be one to one with channel names
        channel_name:
            str, channel name of the data
        state_channel_name:
            str, channel name of the state vector
        wait_time:
            float, time to wait for next file.
        watch_suffix:
            str, filename suffix to watch for.
        verbose:
            bool, be verbose
    """

    shared_memory_dir: str = ""
    instrument: str = ""
    channel_name: str = ""
    state_channel_name: str = ""
    wait_time: float = 60
    watch_suffix: str = ".gwf"
    verbose: bool = False

    def __post_init__(self):
        self.source_pad_names = (self.channel_name, self.state_channel_name)
        super().__post_init__()
        assert (
            self.shared_memory_dir
            and self.instrument
            and self.channel_name
            and self.state_channel_name
        )

        self.cnt = {p: 0 for p in self.source_pads}
        # FIXME: do we need to consider other rates?
        # FIXME: make this more general
        if self.instrument == "V1":
            self.rate_dict = {self.channel_name: 16384, self.state_channel_name: 1}
        else:
            self.rate_dict = {self.channel_name: 16384, self.state_channel_name: 16}
        # set assumed buffer duration based on sample rate
        # and num samples per buffer. Will fail if this does
        # not match the file duration
        self.buffer_duration = 1
        for rate in self.rate_dict.values():
            if self.num_samples(rate) / rate != 1:
                raise ValueError("Buffer duration must be 1 second.")

        self.queue = queue.Queue()

        # initialize a named tuple to track info about the previous
        # buffer sent. this will be used to make sure we dont resend
        # late data and to track discontinuities
        start = int(now())
        self.last_buffer = LastBuffer(start, start, False)
        if self.verbose:
            print(f"Start up t0: {self.last_buffer.t0}", flush=True)

        # Create the inotify handler
        self.observer = threading.Thread(
            target=self.monitor_dir, args=(self.queue, self.shared_memory_dir)
        )

        # Start the observer and set the stop attribute
        self._stop = False
        self.observer.start()

    def monitor_dir(self, queue: queue.Queue, watch_dir: str) -> None:
        """Poll directory for new files with inotify

        Args:
            queue:
                queue.Queue, the queue to add files to
            watch_dir:
                str, directory to monitor
        """
        # init inotify watcher on shared memory dir
        if INotify is None:
            raise ImportError("inotify_simple is required for DevShmSrc source.")

        i = INotify()
        i.add_watch(watch_dir, flags.CLOSE_WRITE | flags.MOVED_TO)

        # Get the current thread
        # t = threading.currentThread()

        # Check if this thread should stop
        while not self._stop:
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
                if t0 < self.last_buffer.end:
                    pass
                else:
                    # Add the filename to the queue
                    queue.put((os.path.join(watch_dir, filename), t0))

        # Remove the watch
        i.rm_watch(watch_dir)

    def internal(self, pad: InternalPad) -> None:
        """Queue files and check if we need to send out buffers of data or gaps. All
        channels are read at once.

        Args:
            pad:
                InternalPad
        """
        # get next file from queue. if its old, try again until we
        # find a new file or reach the end of the queue
        try:
            while True:
                # Im not sure what the right timeout here is,
                # but I want to avoid a situation where get()
                # times out just before the new file arrives and
                # prematurely decides to send a gap buffer
                next_file, t0 = self.queue.get(timeout=2)
                if self.verbose:
                    print(next_file, t0, flush=True)
                if t0 < self.last_buffer.end:
                    continue
                else:
                    break

        except queue.Empty:
            if now() - self.last_buffer.end >= self.wait_time:
                # FIXME: We should send out a gap buffer instead of stopping
                # FIXME: Sending out a 60 second gap buffer doesn't seem like
                #        a good idea, cannot fit tensors in memory
                # self._stop = True
                # raise ValueError(
                #    f"Reached {self.wait_time} seconds with no new files in "
                #    f"{self.shared_memory_dir}, exiting."
                # )
                self.send_gap = True
                self.send_gap_duration = self.buffer_duration
                # update last buffer
                self.last_buffer.t0 = self.last_buffer.end
                self.last_buffer.end = self.last_buffer.end + self.buffer_duration
                self.last_buffer.is_gap = True
            else:
                # send a gap buffer
                self.send_gap = True
                self.send_gap_duration = 0
                self.last_buffer.is_gap = True
        else:
            self.send_gap = False
            # load data from the file using gwpy
            self.data_dict = TimeSeriesDict.read(
                next_file, [self.channel_name, self.state_channel_name]
            )
            # update last buffer
            self.last_buffer.t0 = self.last_buffer.end
            self.last_buffer.end = self.last_buffer.t0 + self.buffer_duration
            self.last_buffer.is_gap = False

    def new(self, pad: SourcePad) -> TSFrame:
        """New frames are created on "pad" with an instance specific count and a name
        derived from the channel name. "EOS" is never set when streaming data online.

        Args:
            pad:
                SourcePad, the pad for which to produce a new TSFrame

        Returns:
            TSFrame, the TSFrame that carries a list of SeriesBuffers
        """

        self.cnt[pad] += 1
        channel = self.rsrcs[pad]
        if self.send_gap:
            if self.verbose:
                print(
                    f"{self.instrument} Queue is empty, sending a gap buffer at t0: "
                    f"{self.last_buffer.end} | Time now: {now()} | ifo: "
                    f"{self.instrument}",
                    flush=True,
                )
            shape = (int(self.send_gap_duration * self.rate_dict[channel]),)
            outbufs = [
                SeriesBuffer(
                    offset=Offset.fromsec(self.last_buffer.end - Offset.offset_ref_t0),
                    sample_rate=self.rate_dict[channel],
                    data=None,
                    shape=shape,
                )
            ]
        else:
            # Send data!
            data = self.data_dict[channel]

            # check sample rate and duration matches what we expect
            duration = data.duration.value
            assert int(data.sample_rate.value) == self.rate_dict[channel], (
                f"Data rate does not match requested sample rate. Data sample rate:"
                f" {data.sample_rate.value}, expected {self.rate_dict[channel]}"
            )
            assert (
                duration == self.buffer_duration
            ), f"File duration ({duration} sec) does not match assumed buffer duration"
            f" ({self.buffer_duration} sec)."

            t0 = data.t0.value
            assert t0 == self.last_buffer.t0
            outbufs = [
                SeriesBuffer(
                    offset=Offset.fromsec(t0 - Offset.offset_ref_t0),
                    sample_rate=self.rate_dict[channel],
                    data=numpy.array(data),
                    shape=data.shape,
                )
            ]

            if self.verbose:
                print(
                    f"{self.instrument} Buffer t0: {t0} | Time Now: {now()} |"
                    f" Time delay: {float(now()) - t0:.3e}",
                    flush=True,
                )

        # online data is never EOS
        # FIXME but maybe there should be some kind of graceful shutdown
        EOS = False

        return TSFrame(
            buffers=outbufs,
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
