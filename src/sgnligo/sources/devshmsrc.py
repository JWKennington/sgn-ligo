"""A source element to read low-latency data streamed to /dev/shm in real-time."""

# Copyright (C) 2022      Ron Tapia
# Copyright (C) 2024-2025 Becca Ewing, Yun-Jing Huang

from __future__ import annotations

import os
import queue
from dataclasses import dataclass

import numpy
from gwpy.timeseries import TimeSeriesDict
from sgn.base import SourcePad
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from sgnligo.base import from_T050017, now


class _FrameFileEventHandler(PatternMatchingEventHandler):
    """Custom watchdog event handler for tracking new frame files."""

    def __init__(self, queue, watch_suffix, current_watermark):
        self.queue = queue
        self.watch_suffix = watch_suffix
        self.current_watermark = current_watermark
        super().__init__(patterns=[f"*{self.watch_suffix}"])

    def on_closed(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        self._handle_event(event.dest_path)

    def _handle_event(self, path):
        extension = os.path.splitext(path)[1]
        if extension != self.watch_suffix:
            return
        # skip any files past the current watermark
        _, _, t0, _ = from_T050017(path)
        if t0 < self.current_watermark:
            return
        self.queue.put((path, t0))


@dataclass
class DevShmSource(TSSource):
    """Source element to read low-latency data streamed to /dev/shm in real-time for
    multiple ifos. A watchdog handler is created for each shared memory dir. The
    internal pad will try to sync the ifos if one is falling behind.


    Args:
        shared_memory_dirs:
            dict[str, str], a dictionary of shared memory directory name (full path)
            with ifos as keys. Suggestion: {"L1": "/dev/shm/kafka/L1_O3ReplayMDC"}
        channel_names:
            dict[str, list[str]], a dictionary of list of channel names of the data,
            with ifos as keys, e.g.,
            {"L1": ["L1:GDS-CALIB_STRAIN", "L1:GDS-CALIB_STATE_VECTOR"]}. Source pads
            will be automatically generated for each channel, with channel name as pad
            name.
        discont_wait_time:
            float, time to wait before dropping data, in seconds.
        queue_timeout:
            float, time to wait for next file from the queue, in seconds.
        watch_suffix:
            str, filename suffix to watch for.
        verbose:
            bool, be verbose
    """

    shared_memory_dirs: dict[str, str] | str | None = None
    channel_names: dict[str, list[str]] | list[str] | None = None
    discont_wait_time: float = 60
    queue_timeout: float = 1
    watch_suffix: str = ".gwf"
    verbose: bool = False

    def __post_init__(self):
        assert self.shared_memory_dirs and self.channel_names
        # coerce list-like channel names and shm dirs to dictionaries
        if not isinstance(self.channel_names, dict):
            assert not isinstance(self.shared_memory_dirs, dict)
            ifo = self.channel_names[0].split(":")[0]
            self.channel_names = {ifo: self.channel_names}
        if not isinstance(self.shared_memory_dirs, dict):
            assert len(self.channel_names.keys()) == 1
            ifo = list(self.channel_names.keys())[0]
            self.shared_memory_dirs = {ifo: self.shared_memory_dirs}

        if len(self.source_pad_names) > 0:
            if self.source_pad_names != tuple(self.channel_names):
                raise ValueError("Expected source pad names to match channel names")
        else:
            print(f"Generating source pads from channel names {self.channel_names}...")

            self.source_pad_names = tuple(
                ci for c in self.channel_names.values() for ci in c
            )

        ifos = self.shared_memory_dirs.keys()
        self.ifos = ifos
        self.queues = {}
        for ifo in ifos:
            self.queues[ifo] = queue.Queue()

        # initialize a named tuple to track info about the previous
        # buffer sent. this will be used to make sure we dont resend
        # late data and to track discontinuities
        self.reset_start = True
        start = int(now())
        self.next_buffer_t0 = {ifo: start for ifo in ifos}
        self.next_buffer_end = {ifo: start for ifo in ifos}
        if self.verbose:
            print(f"Start up t0: {self.next_buffer_t0}", flush=True)

        # Start the observer and set the stop attribute
        self._stop = False

        # Create the observer threads to watch for new frame files
        self.observer = {}
        for ifo, shared_dir in self.shared_memory_dirs.items():
            event_handler = _FrameFileEventHandler(
                self.queues[ifo], self.watch_suffix, self.next_buffer_t0[ifo]
            )
            self.observer[ifo] = Observer()
            self.observer[ifo].schedule(
                event_handler,
                path=shared_dir,
            )
            self.observer[ifo].daemon = True
            self.observer[ifo].start()

        self.rates = {}
        self.data_dict = {}
        self.send_gaps = {}
        self.file_t0 = {}
        self.discont = {}
        self.send_gap = {}
        self.send_gap_duration = {}
        # Read in the first gwf file to get the sample rates for each channel name
        for ifo in ifos:
            files = os.listdir(self.shared_memory_dirs[ifo])
            for f in reversed(sorted(files)):
                if f.endswith(self.watch_suffix):
                    file0 = self.shared_memory_dirs[ifo] + "/" + f
                    break

            _data_dict = TimeSeriesDict.read(file0, self.channel_names[ifo])
            self.rates[ifo] = {
                c: int(data.sample_rate.value) for c, data in _data_dict.items()
            }

            # set assumed buffer duration based on sample rate
            # and num samples per buffer. Will fail if this does
            # not match the file duration
            channel = self.channel_names[ifo][0]
            self.buffer_duration = _data_dict[channel].duration.value

            print("sample rates:", self.rates[ifo])

            self.data_dict[ifo] = {c: None for c in self.channel_names[ifo]}
            self.send_gaps[ifo] = False

        self.t0 = start
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}

    def internal(self) -> None:
        """Queue files and check if we need to send out buffers of data or gaps. All
        channels are read at once.
        """
        super().internal()

        if self.reset_start is True:
            # pipeline init takes too long
            start = int(now())
            self.next_buffer_t0 = {ifo: start for ifo in self.ifos}
            self.next_buffer_end = {ifo: start for ifo in self.ifos}
            self.reset_start = False

        # old_data = False
        old_data = {ifo: False for ifo in self.ifos}
        for ifo in self.ifos:
            self.next_buffer_t0[ifo] = self.next_buffer_end[ifo]
            for data in self.data_dict[ifo].values():
                if data is not None:
                    old_data[ifo] = True
                    # there is still data
                    if self.file_t0[ifo] == self.next_buffer_t0[ifo]:
                        self.discont[ifo] = False
                        self.send_gap[ifo] = False
                        print(
                            f"old data cont. | file_t0 {self.file_t0[ifo]} | "
                            f"next_buffer_t0 {self.next_buffer_t0[ifo]} | ifo: {ifo}"
                        )
                    elif self.file_t0[ifo] > self.next_buffer_t0[ifo]:
                        print(
                            f"old data discont. | file_t0 {self.file_t0[ifo]} | "
                            f"next_buffer_t0 {self.next_buffer_t0[ifo]} | ifo: {ifo}"
                        )
                        self.discont[ifo] = True
                        self.send_gap[ifo] = True
                        self.send_gap_duration[ifo] = self.buffer_duration
                    else:
                        raise ValueError("wrong t0")
                else:
                    self.discont[ifo] = False
                    self.send_gap[ifo] = True
                    self.send_gap_duration[ifo] = 0

        # get next file from queue. if its old, try again until we
        # find a new file or reach the end of the queue
        self.next_buffer_t0_min = min(self.next_buffer_t0.values())
        self.next_buffer_t0_max = max(self.next_buffer_t0.values())
        send_gap_sync = False
        self.next_buffer_t0 = dict(
            sorted(self.next_buffer_t0.items(), key=lambda item: item[1])
        )
        for ifo in self.next_buffer_t0.keys():
            if old_data[ifo] is True:
                # There is old data in the data_dict, don't read in new data
                print(ifo, "There is old data in the data_dict, skip reading new file")
                continue

            if send_gap_sync:
                # another ifo is falling behind and is preparing to send a nongap
                # send gap for this ifo so the other one can catch up
                self.discont[ifo] = False
                self.send_gap[ifo] = True
                self.send_gap_duration[ifo] = 0
                if self.verbose:
                    print(ifo, "send_gap_sync")
                continue

            try:
                while True:
                    # Im not sure what the right timeout here is,
                    # but I want to avoid a situation where get()
                    # times out just before the new file arrives and
                    # prematurely decides to send a gap buffer
                    # if send_gap_all:
                    next_file, t0 = self.queues[ifo].get(timeout=self.queue_timeout)
                    self.file_t0[ifo] = t0
                    if self.verbose:
                        print(next_file, t0, flush=True)
                        print(self.next_buffer_t0)
                    if t0 < self.next_buffer_t0[ifo]:
                        continue
                    elif t0 == self.next_buffer_t0[ifo]:
                        if self.next_buffer_t0[ifo] < self.next_buffer_t0_max:
                            # this ifo is falling behind and is preparing to send a
                            # nongap send gap for other ifos so this one can catch up
                            send_gap_sync = True
                        self.discont[ifo] = False
                        break
                    else:
                        self.discont[ifo] = True
                        break

            except queue.Empty:
                if now() - self.next_buffer_t0[ifo] >= self.discont_wait_time:
                    # FIXME: We should send out a gap buffer instead of stopping
                    # FIXME: Sending out a 60 second gap buffer doesn't seem like
                    #        a good idea, cannot fit tensors in memory
                    # self._stop = True
                    # raise ValueError(
                    #    f"Reached {self.wait_time} seconds with no new files in "
                    #    f"{self.shared_memory_dir}, exiting."
                    # )
                    # if self.verbose:
                    print(
                        f"{ifo} Reached wait time, sending a gap buffer with t0 "
                        f"{self.next_buffer_t0[ifo]} | duration {self.buffer_duration}"
                        f" | now {now()}"
                    )
                    self.send_gap[ifo] = True
                    self.send_gap_duration[ifo] = self.buffer_duration
                else:
                    # send a gap buffer
                    self.send_gap[ifo] = True
                    self.send_gap_duration[ifo] = 0
            else:
                if self.discont[ifo]:
                    # the new file is later than the next expected t0
                    # start sending gap buffers
                    self.send_gap[ifo] = True
                    self.send_gap_duration[ifo] = self.buffer_duration
                    print(
                        f"discont t0 {t0} | file_t0 {self.file_t0[ifo]} | "
                        f"next_buffer_t0 {self.next_buffer_t0[ifo]} | ifo: {ifo}"
                    )
                else:
                    self.send_gap[ifo] = False
                # load data from the file using gwpy
                assert self.channel_names is not None
                self.data_dict[ifo] = TimeSeriesDict.read(
                    next_file,
                    self.channel_names[ifo],
                )

    def new(self, pad: SourcePad) -> TSFrame:
        """New frames are created on "pad" with an instance specific count and a name
        derived from the channel name. "EOS" is set by signaled_eos().

        Args:
            pad:
                SourcePad, the pad for which to produce a new TSFrame

        Returns:
            TSFrame, the TSFrame that carries a list of SeriesBuffers
        """

        self.cnt[pad] += 1
        channel = self.rsrcs[pad]
        ifo = channel.split(":")[0]
        if self.send_gap[ifo]:
            if self.verbose:
                print(
                    f"{pad.name} Queue is empty, sending a gap buffer at t0: "
                    f"{self.next_buffer_t0[ifo]} | Durtaion: "
                    f"{self.send_gap_duration[ifo]} | Time now: {now()} | ifo: "
                    f"{pad.name} | Time delay: {now() - self.next_buffer_t0[ifo]}",
                    flush=True,
                )
            shape = (int(self.send_gap_duration[ifo] * self.rates[ifo][channel]),)
            outbuf = SeriesBuffer(
                offset=Offset.fromsec(self.next_buffer_t0[ifo] - Offset.offset_ref_t0),
                sample_rate=self.rates[ifo][channel],
                data=None,
                shape=shape,
            )
            self.next_buffer_end[ifo] = outbuf.end / 1_000_000_000
        else:
            # Send data!
            data = self.data_dict[ifo][channel]

            # check sample rate and duration matches what we expect
            duration = data.duration.value
            assert int(data.sample_rate.value) == self.rates[ifo][channel], (
                f"Data rate does not match requested sample rate. Data sample rate:"
                f" {data.sample_rate.value}, expected {self.rates[ifo][channel]}"
            )
            assert (
                duration == self.buffer_duration
            ), f"File duration ({duration} sec) does not match assumed buffer duration"
            f" ({self.buffer_duration} sec)."

            t0 = data.t0.value
            assert (
                t0 == self.next_buffer_t0[ifo]
            ), f"Name: {self.name} | t0: {t0} | next buffer t0: "
            f"{self.next_buffer_t0[ifo]}"

            outbuf = SeriesBuffer(
                offset=Offset.fromsec(t0 - Offset.offset_ref_t0),
                sample_rate=self.rates[ifo][channel],
                data=numpy.array(data),
                shape=data.shape,
            )
            self.next_buffer_end[ifo] = outbuf.end / 1_000_000_000

            self.data_dict[ifo][channel] = None

            if self.verbose:
                print(
                    f"{pad.name} Buffer t0: {t0} | Time Now: {now()} |"
                    f" Time delay: {float(now()) - t0:.3e}",
                    flush=True,
                )

        # if at EOS, stop watching for new frame files, wait for threads to finish
        EOS = (outbuf.end_offset >= self.end_offset) or self.signaled_eos()
        if EOS:
            for observer in self.observer.values():
                observer.unschedule_all()
                observer.stop()
                observer.join()

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=EOS,
        )
