import os
import queue
import threading
from dataclasses import dataclass

import numpy
from gwpy.timeseries import StateVector, TimeSeries

try:
    from inotify_simple import INotify, flags
except ImportError:
    INotify = flags = None

from sgnligo.base import from_T050017, now, state_vector_on_off_bits
from sgnts.base import Offset, SeriesBuffer, TSFrame
from sgnts.sources import TSSource


@dataclass
class LastBuffer:
    t0: int
    is_gap: bool


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
    state_channel_name: tuple
        channel name of the state vector
    watch_suffix: str
        Filename suffix to watch for.
    """

    rate: int = 2048
    channel_name: tuple = ()
    state_channel_name: tuple = ()
    instrument: tuple = ()
    shared_memory_dir: str = None
    wait_time: int = 60
    watch_suffix: str = ".gwf"
    state_vector_on_bits: int = None
    verbose: bool = None

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}
        self.shape = (self.num_samples,)
        self.queue = queue.Queue()

        # set assumed buffer duration based on sample rate
        # and num samples per buffer. Will fail if this does
        # not match the file duration
        self.buffer_duration = self.num_samples / self.rate

        # initialize a named tuple to track info about the previous
        # buffer sent. this will be used to make sure we dont resend
        # late data and to track discontinuities
        self.last_buffer = LastBuffer(int(now()), False)
        if self.verbose:
            print(f"Start up t0: {self.last_buffer.t0}")

        # set state vector on bits
        self.bitmask = state_vector_on_off_bits(self.state_vector_on_bits)

        # Create the inotify handler
        self.observer = threading.Thread(
            target=self.monitor_dir, args=(self.queue, self.shared_memory_dir)
        )

        # Start the observer and set the stop attribute
        self.observer.stop = False
        self.observer.start()

    def monitor_dir(self, queue, watch_dir):
        """
        poll directory for new files with inotify
        """
        # init inotify watcher on shared memory dir
        if INotify is None:
            raise ImportError("inotify_simple is required for DevShmSrc source.")

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
                if t0 < self.last_buffer.t0:
                    pass
                else:
                    # Add the filename to the queue
                    queue.put((os.path.join(watch_dir, filename), t0))

        # Remove the watch
        i.rm_watch(watch_dir)

    def new(self, pad):
        self.cnt[pad] += 1

        # get next file from queue. if its old, try again until we
        # find a new file or reach the end of the queue
        try:
            while True:
                # Im not sure what the right timeout here is,
                # but I want to avoid a situation where get()
                # times out just before the new file arrives and
                # prematurely decides to send a gap buffer
                next_file, t0 = self.queue.get(timeout=2)
                if t0 <= self.last_buffer.t0:
                    continue
                else:
                    break

        except queue.Empty:
            if now() - self.last_buffer.t0 >= self.wait_time:
                self.observer.stop = True
                raise ValueError(
                    f"Reached {self.wait_time} seconds with no new files in {self.shared_memory_dir}, exiting."
                )
            else:
                # send a gap buffer
                if self.cnt[pad] == 1:
                    # send the first gap buffer starting from the program start up time
                    t0 = self.last_buffer.t0
                else:
                    # send subsequent gaps at self.buffer_duration intervals
                    t0 = self.last_buffer.t0 + self.buffer_duration
                shape = (int(self.rate * self.buffer_duration),)
                print(
                    f"Queue is empty, sending a gap buffer at t0: {t0} | ifo: {self.instrument}"
                )
                outbufs = [
                    SeriesBuffer(
                        offset=Offset.fromsec(t0 - Offset.offset_ref_t0),
                        sample_rate=self.rate,
                        data=None,
                        shape=shape,
                    )
                ]

                # update last buffer
                self.last_buffer.t0 = t0
        else:
            # first check the state
            statedata = StateVector.read(
                next_file, f"{self.instrument}:" + self.state_channel_name
            )

            state_data = numpy.array(statedata.data)
            state_t0 = statedata.t0.value
            state_offset0 = Offset.fromsec(
                state_t0 - Offset.offset_ref_t0 / 1_000_000_000
            )
            state_duration = statedata.duration.value
            state_sample_rate = statedata.sample_rate.value
            state_nsamples = int(self.rate * statedata.dt.value)

            state_times = numpy.arange(
                state_t0, state_t0 + state_duration, statedata.dt.value
            )
            bits = numpy.array(statedata)

            state_flags = []
            for b in bits:
                b = state_vector_on_off_bits(b)
                if b & self.bitmask == self.bitmask:
                    state_flags.append(True)
                else:
                    state_flags.append(False)

            if not any(state_flags):
                # return gap of buffer duration
                if self.verbose:
                    print(f"{self.instrument}: OFF at {t0}")

                data = None
                t0 = state_t0
                shape = (int(self.rate * self.buffer_duration),)

                outbufs = [
                    SeriesBuffer(
                        offset=Offset.fromsec(t0 - Offset.offset_ref_t0),
                        sample_rate=self.rate,
                        data=data,
                        shape=shape,
                    )
                ]
            else:
                # load data from the file using gwpy
                data = TimeSeries.read(
                    next_file, f"{self.instrument}:{self.channel_name}"
                )
                t0 = data.t0.value
                duration = data.duration.value

                # check sample rate and duration matches what we expect
                assert (
                    int(data.sample_rate.value) == self.rate
                ), "Data rate does not match requested sample rate."
                assert (
                    duration == self.buffer_duration
                ), "File duration ({duration} sec) does not match assumed buffer duration ({self.buffer_duration} sec)."

                data = numpy.array(data)

                if all(state_flags):
                    if self.verbose:
                        print(f"{self.instrument}: ON at {t0}")
                    outbufs = [
                        SeriesBuffer(
                            offset=Offset.fromsec(t0 - Offset.offset_ref_t0),
                            sample_rate=self.rate,
                            data=data,
                            shape=data.shape,
                        )
                    ]

                else:
                    # we need to slice the buffer to replace data with gaps
                    # for segments where state bits dont match self.bitmask
                    if self.verbose:
                        print(f"{self.instrument}: state transition at {t0}")

                    outbufs = []
                    state0 = state_flags[0]
                    buf_offset0 = state_offset0
                    n0 = 0
                    num_samples = state_nsamples
                    for state in state_flags[1:]:
                        if state is state0:
                            num_samples += state_nsamples
                            continue
                        else:
                            # There is a state change
                            # Create the buffer for the previous state
                            shape = (num_samples,)
                            if state0 is True:
                                outbufs.append(
                                    SeriesBuffer(
                                        offset=buf_offset0,
                                        sample_rate=self.rate,
                                        data=data[n0 : n0 + num_samples],
                                        shape=shape,
                                    )
                                )
                            else:
                                outbufs.append(
                                    SeriesBuffer(
                                        offset=buf_offset0,
                                        sample_rate=self.rate,
                                        data=None,
                                        shape=shape,
                                    )
                                )
                            # reset
                            state0 = state
                            buf_offset0 += Offset.fromsamples(num_samples, self.rate)
                            n0 += num_samples
                            num_samples = state_nsamples

                    # make the last buffer
                    if state0 is True:
                        outbufs.append(
                            SeriesBuffer(
                                offset=buf_offset0,
                                sample_rate=self.rate,
                                data=data[n0 : n0 + num_samples],
                                shape=(num_samples,),
                            )
                        )
                    else:
                        outbufs.append(
                            SeriesBuffer(
                                offset=buf_offset0,
                                sample_rate=self.rate,
                                data=None,
                                shape=(num_samples,),
                            )
                        )

            if self.verbose:
                print(
                    f"Buffer t0: {t0} | Time Now: {now()} | Time delay: {float(now()) - t0:.3e} | Discont: {self.last_buffer.is_gap}"
                )

            # update last buffer
            self.last_buffer.t0 = t0

        # online data is never EOS
        # FIXME but maybe there should be some kind of graceful shutdown
        EOS = False

        return TSFrame(
            buffers=outbufs,
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
