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
class DevShmSrc(TSSource):
    """Source element to read low-latency data streamed to /dev/shm in real-time

    Args:
        shared_memory_dir:
            str, shared memory directory name (full path). Suggestion:
            /dev/shm/kafka/L1_O3ReplayMDC
        channel_names:
            list[str], a list of channel names of the data, e.g.,
            ["L1:GDS-CALIB_STRAIN", "L1:GDS-CALIB_STATE_VECTOR"]. Source pads will
            be automatically generated for each channel, with channel name as pad name.
        wait_time:
            float, time to wait for next file.
        watch_suffix:
            str, filename suffix to watch for.
        verbose:
            bool, be verbose
    """

    shared_memory_dir: str = ""
    channel_names: list[str] = None
    wait_time: float = 60
    watch_suffix: str = ".gwf"
    verbose: bool = False

    def __post_init__(self):
        if len(self.source_pad_names) > 0:
            if self.source_pad_names != tuple(self.channel_names):
                raise ValueError("Expected source pad names to match channel names")
        else:
            print(f"Generating source pads from channel names {self.channel_names}...")
            self.source_pad_names = tuple(self.channel_names)
        super().__post_init__()
        assert self.shared_memory_dir and self.channel_names

        self.cnt = {p: 0 for p in self.source_pads}
        self.queue = queue.Queue()

        # initialize a named tuple to track info about the previous
        # buffer sent. this will be used to make sure we dont resend
        # late data and to track discontinuities
        start = int(now())
        self.next_buffer_t0 = start
        self.next_buffer_end = start
        if self.verbose:
            print(f"Start up t0: {self.next_buffer_t0}", flush=True)

        # Create the inotify handler
        self.observer = threading.Thread(
            target=self.monitor_dir,
            args=(self.queue, self.shared_memory_dir),
            daemon=True,
        )

        # Start the observer and set the stop attribute
        self._stop = False
        self.observer.start()

        # Read in the first gwf file to get the sample rates for each channel name
        files = os.listdir(self.shared_memory_dir)
        for f in reversed(sorted(files)):
            if f.endswith(self.watch_suffix):
                file0 = self.shared_memory_dir + "/" + f
                break

        _data_dict = TimeSeriesDict.read(file0, self.channel_names)
        self.rates = {c: int(data.sample_rate.value) for c, data in _data_dict.items()}

        # set assumed buffer duration based on sample rate
        # and num samples per buffer. Will fail if this does
        # not match the file duration
        self.buffer_duration = 1
        for rate in self.rates.values():
            if self.num_samples(rate) / rate != 1:
                raise ValueError("Buffer duration must be 1 second.")

        if self.verbose:
            print("sample rates:", self.rates)

        self.data_dict = {c: None for c in self.channel_names}

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
                if t0 < self.next_buffer_t0:
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
        self.next_buffer_t0 = self.next_buffer_end
        for data in self.data_dict.values():
            if data is not None:
                # there is still data
                if self.file_t0 == self.next_buffer_t0:
                    self.discont = False
                    self.send_gap = False
                elif self.file_t0 > self.next_buffer_t0:
                    self.discont = True
                    self.send_gap = True
                    self.send_gap_duration = self.buffer_duration
                    pass
                else:
                    raise ValueError("wrong t0")
                return

        # get next file from queue. if its old, try again until we
        # find a new file or reach the end of the queue
        try:
            while True:
                # Im not sure what the right timeout here is,
                # but I want to avoid a situation where get()
                # times out just before the new file arrives and
                # prematurely decides to send a gap buffer
                next_file, t0 = self.queue.get(timeout=3)
                self.file_t0 = t0
                if self.verbose:
                    print(next_file, t0, flush=True)
                if t0 < self.next_buffer_t0:
                    continue
                elif t0 == self.next_buffer_t0:
                    self.discont = False
                    break
                else:
                    self.discont = True
                    break

        except queue.Empty:
            if now() - self.next_buffer_t0 >= self.wait_time:
                # FIXME: We should send out a gap buffer instead of stopping
                # FIXME: Sending out a 60 second gap buffer doesn't seem like
                #        a good idea, cannot fit tensors in memory
                # self._stop = True
                # raise ValueError(
                #    f"Reached {self.wait_time} seconds with no new files in "
                #    f"{self.shared_memory_dir}, exiting."
                # )
                if self.verbose:
                    print(
                        "Reached wait time, sending a gap buffer of "
                        f" {self.buffer_duration}"
                    )
                self.send_gap = True
                self.send_gap_duration = self.buffer_duration
            else:
                # send a gap buffer
                self.send_gap = True
                self.send_gap_duration = 0
        else:
            if self.discont:
                # the new file is later than the next expected t0
                # start sending gap buffers
                self.send_gap = True
                self.send_gap_duration = self.buffer_duration
                print(
                    f"discont t0 {t0} | file_t0 {self.file_t0} | next_buffer_t0 "
                    f"{self.next_buffer_t0}"
                )
            else:
                self.send_gap = False
            # load data from the file using gwpy
            self.data_dict = TimeSeriesDict.read(
                next_file,
                self.channel_names,
            )

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
                    f"{pad.name} Queue is empty, sending a gap buffer at t0: "
                    f"{self.next_buffer_t0} | Time now: {now()} | ifo: "
                    f"{pad.name} | Time delay: {now() - self.next_buffer_t0/1e9}",
                    flush=True,
                )
            shape = (int(self.send_gap_duration * self.rates[channel]),)
            outbuf = SeriesBuffer(
                offset=Offset.fromsec(self.next_buffer_t0 - Offset.offset_ref_t0),
                sample_rate=self.rates[channel],
                data=None,
                shape=shape,
            )
            self.next_buffer_end = outbuf.end / 1_000_000_000
        else:
            # Send data!
            data = self.data_dict[channel]

            # check sample rate and duration matches what we expect
            duration = data.duration.value
            assert int(data.sample_rate.value) == self.rates[channel], (
                f"Data rate does not match requested sample rate. Data sample rate:"
                f" {data.sample_rate.value}, expected {self.rates[channel]}"
            )
            assert (
                duration == self.buffer_duration
            ), f"File duration ({duration} sec) does not match assumed buffer duration"
            f" ({self.buffer_duration} sec)."

            t0 = data.t0.value
            assert (
                t0 == self.next_buffer_t0
            ), f"Name: {self.name} | t0: {t0} | next buffer t0: {self.next_buffer_t0}"
            outbuf = SeriesBuffer(
                offset=Offset.fromsec(t0 - Offset.offset_ref_t0),
                sample_rate=self.rates[channel],
                data=numpy.array(data),
                shape=data.shape,
            )
            self.next_buffer_end = outbuf.end / 1_000_000_000

            self.data_dict[channel] = None

            if self.verbose:
                print(
                    f"{pad.name} Buffer t0: {t0} | Time Now: {now()} |"
                    f" Time delay: {float(now()) - t0:.3e}",
                    flush=True,
                )

        # online data is never EOS
        # FIXME but maybe there should be some kind of graceful shutdown
        EOS = False

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
