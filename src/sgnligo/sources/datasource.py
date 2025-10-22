"""Datasource element utilities for LIGO pipelines."""

# Copyright (C) 2009-2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import igwn_segments as segments
import numpy as np
from igwn_ligolw import utils as ligolw_utils
from igwn_ligolw.utils import segments as ligolw_segments
from lal import LIGOTimeGPS
from sgn import Pipeline
from sgn_arrakis.source import ArrakisSource
from sgnts.sources import FakeSeriesSource, SegmentSource
from sgnts.transforms import Adder, Gate

from sgnligo.base import parse_list_to_dict, read_segments_and_values_from_file
from sgnligo.sources.devshmsrc import DevShmSource
from sgnligo.sources.framecachesrc import FrameReader
from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource
from sgnligo.transforms import BitMask, Latency

KNOWN_DATASOURCES = [
    "white",
    "sin",
    "impulse",
    "white-realtime",
    "frames",
    "devshm",
    "arrakis",
    "gwdata-noise",
    "gwdata-noise-realtime",
]
FAKE_DATASOURCES = [
    "white",
    "sin",
    "impulse",
    "white-realtime",
    "gwdata-noise",
    "gwdata-noise-realtime",
]
OFFLINE_DATASOURCES = ["white", "sin", "impulse", "frames", "gwdata-noise"]


@dataclass
class DataSourceInfo:
    """Wrapper around data source options

    Args:
        data_source:
            str, the data source, can be one of
            [white|sin|impulse|white-realtime|frames|devshm|arrakis|gwdata-noise|gwdata-noise-realtime]
        channel_name:
            list[str, ...], a list of channel names ["IFO=CHANNEL_NAME",...].
            For fake sources [white|sin|impulse|white-realtime], channel names are used
            to derive ifos.
        gps_start_time:
            float, the gps start time of the data to analyze, in seconds
        gps_end_time:
            float, the gps end time of the data to analyze, in seconds
        frame_cache:
            str, the frame cache file to read gwf frame files from. Must be provided
            when data_source is "frames"
        frame_segments_file:
            str, the name of the LIGO light-weight XML file from which to load
            frame segments. Optional iff data_source=frames
        frame_segments_name:
            str, the name of the segments to extract from the segment tables. Required
            iff frame_segments_file is given
        noiseless_inj_frame_cache:
            str, the name of the LAL cache listing the noiseless LIGO-Virgo injection
            .gwf frame files to be added to the strain data in frame_cache. (optional,
            must be provided with frame_cache)
        noiseless_inj_channel_name:
            list[str] or Dict[Detector, HostInfo], the name of the noiseless
            inj channels to process per detector (optional, must be provided with
            channel_name)
        state_channel_name:
            list, a list of state vector channel names
        state_vector_on_bits:
            int, the bit mask for the state vector data
        shared_memory_dir:
            str, the path to the shared memory directory to read low-latency data from
        discont_wait_time:
            float, the time to wait for next file before dropping data when data_source
            is "devshm", in seconds
        source_queue_timeout:
            float, the time to wait for next file from the queue before sending a
            heartbeat buffer when data_source is "devshm", in seconds.
            When data_source is "arrakis", used as the in_queue_timeout for
            ArrakisSource.
        input_sample_rate:
            int, the sample rate for fake sources [white|sin|impulse|white-realtime]
        impulse_position:
            int, the sample point position to place the impulse data point. Default -1,
            which will generate the impulse position randomly
        state_segments_file:
            str, path to file with three columns (start end value) defining state vector
            segments for gwdata-noise sources. Optional.
        state_sample_rate:
            int, sample rate for state vector channels when using gwdata-noise sources.
            Default is 16 Hz (typical for state vectors).
    """

    data_source: str
    channel_name: list[str]
    gps_start_time: Optional[float] = None
    gps_end_time: Optional[float] = None
    frame_cache: Optional[str] = None
    frame_segments_file: Optional[str] = None
    frame_segments_name: Optional[str] = None
    noiseless_inj_frame_cache: Optional[str] = None
    noiseless_inj_channel_name: Optional[list[str]] = None
    state_channel_name: Optional[list[str]] = None
    state_vector_on_bits: Optional[list[int]] = None
    shared_memory_dir: Optional[list[str]] = None
    discont_wait_time: float = 60
    source_queue_timeout: float = 1
    input_sample_rate: Optional[int] = None
    impulse_position: int = -1
    real_time: bool = False
    state_segments_file: Optional[str] = None
    state_sample_rate: int = 16

    def __post_init__(self):
        self.channel_dict = parse_list_to_dict(self.channel_name)
        print(self.channel_dict)
        self.ifos = sorted(self.channel_dict.keys())
        self.seg = None
        self.validate()
        self.all_analysis_ifos = None

    def validate(self):
        if self.data_source not in KNOWN_DATASOURCES:
            raise ValueError(
                "Unknown datasource {}, must be one of: {}".format(
                    self.data_source, ", ".join(KNOWN_DATASOURCES)
                )
            )

        if self.data_source == "devshm":
            if self.shared_memory_dir is None:
                raise ValueError(
                    "Must specify shared_memory_dir when data_source is 'devshm'"
                )
            else:
                self.shared_memory_dict = parse_list_to_dict(self.shared_memory_dir)
                if sorted(self.shared_memory_dict.keys()) != self.ifos:
                    raise ValueError(
                        "Must specify same number of shared_memory_dir as channel_name"
                    )
            if self.state_channel_name is None:
                raise ValueError(
                    "Must specify state_channel_name when data_source is 'devshm'"
                )
            else:
                self.state_channel_dict = parse_list_to_dict(self.state_channel_name)
                if sorted(self.state_channel_dict.keys()) != self.ifos:
                    raise ValueError(
                        "Must specify same number of state_channel_name as channel_name"
                    )
            if self.state_vector_on_bits is None:
                raise ValueError(
                    "Must specify state_vector_on_bits when data_source is 'devshm'"
                )
            else:
                self.state_vector_on_dict = parse_list_to_dict(
                    self.state_vector_on_bits
                )
                if sorted(self.state_vector_on_dict.keys()) != self.ifos:
                    raise ValueError(
                        "Must specify same number of state_vector_on_bits as"
                        " channel_name"
                    )

            if self.gps_start_time is not None or self.gps_end_time is not None:
                raise ValueError(
                    "Must not specify gps_start_time or gps_end_time when"
                    " data_source is 'devshm'"
                )
        elif self.data_source == "arrakis":
            # Validate state vector configuration for arrakis
            if self.state_channel_name is not None:
                self.state_channel_dict = parse_list_to_dict(self.state_channel_name)
                if sorted(self.state_channel_dict.keys()) != self.ifos:
                    raise ValueError(
                        "Must specify same number of state_channel_name as channel_name"
                    )

                if self.state_vector_on_bits is None:
                    raise ValueError(
                        "Must specify state_vector_on_bits when state_channel_name is"
                        " provided for 'arrakis'"
                    )
                else:
                    self.state_vector_on_dict = parse_list_to_dict(
                        self.state_vector_on_bits
                    )
                    if sorted(self.state_vector_on_dict.keys()) != self.ifos:
                        raise ValueError(
                            "Must specify same number of state_vector_on_bits as"
                            " channel_name"
                        )

            # Arrakis source can have optional start_time and end_time
            # If both are provided, start_time must be less than end_time
            if self.gps_start_time is not None and self.gps_end_time is not None:
                if self.gps_start_time >= self.gps_end_time:
                    raise ValueError("Must specify gps_start_time < gps_end_time")
                else:
                    self.seg = segments.segment(
                        LIGOTimeGPS(self.gps_start_time), LIGOTimeGPS(self.gps_end_time)
                    )
        # Input sample rate is not required but will default to 16384 Hz if
        # not provided
        elif self.data_source == "white-realtime":
            if self.input_sample_rate is None:
                raise ValueError(
                    "Must specify input_sample_rate when data_source is one of"
                    f" {FAKE_DATASOURCES}"
                )
        elif self.data_source == "gwdata-noise-realtime":
            # gwdata-noise-realtime doesn't require gps_end_time
            if self.gps_start_time is not None and self.gps_end_time is not None:
                if self.gps_start_time >= self.gps_end_time:
                    raise ValueError("Must specify gps_start_time < gps_end_time")
                else:
                    self.seg = segments.segment(
                        LIGOTimeGPS(self.gps_start_time),
                        LIGOTimeGPS(self.gps_end_time),
                    )
            # If gps_end_time is None, seg remains None
            # (for GWDataNoiseSource real-time mode)
        else:
            # Special case for gwdata-noise with real_time=True
            if self.data_source == "gwdata-noise" and self.real_time:
                # For real-time gwdata-noise, gps_end_time can be None
                if self.gps_start_time is not None and self.gps_end_time is not None:
                    if self.gps_start_time >= self.gps_end_time:
                        raise ValueError("Must specify gps_start_time < gps_end_time")
                    else:
                        self.seg = segments.segment(
                            LIGOTimeGPS(self.gps_start_time),
                            LIGOTimeGPS(self.gps_end_time),
                        )
                # If gps_end_time is None, seg remains None (for GWDataNoiseSource)
            elif self.gps_start_time is None or self.gps_end_time is None:
                raise ValueError(
                    "Must specify gps_start_time and gps_end_time when "
                    f"data_source is one of {OFFLINE_DATASOURCES}"
                )
            elif self.gps_start_time >= self.gps_end_time:
                raise ValueError("Must specify gps_start_time < gps_end_time")
            else:
                self.seg = segments.segment(
                    LIGOTimeGPS(self.gps_start_time), LIGOTimeGPS(self.gps_end_time)
                )

            if self.frame_segments_file is not None:
                if self.frame_segments_name is None:
                    raise ValueError(
                        "Must specify frame_segmetns_name when frame_segments_file is"
                        " given."
                    )
                elif not os.path.exists(self.frame_segments_file):
                    raise ValueError("frame segments file does not exist")

            if self.data_source == "frames":
                if self.frame_cache is None:
                    raise ValueError(
                        "Must specify frame_cache when data_source='frames'"
                    )
                elif not os.path.exists(self.frame_cache):
                    raise ValueError("Frame cahce file does not exist")

                # Validate channel name for each noiseless injection channel name
                if self.noiseless_inj_channel_name is not None:
                    self.noiseless_inj_channel_dict = parse_list_to_dict(
                        self.noiseless_inj_channel_name
                    )
                    for ifo in self.noiseless_inj_channel_dict:
                        if ifo not in self.channel_dict:
                            raise ValueError(
                                "Must specify one hoft channel_name for each"
                                " noiseless_inj_channel_name as {Detector:name}"
                            )

                # Validate noiseless injection frame cache exists
                if self.noiseless_inj_frame_cache:
                    if not os.path.exists(self.noiseless_inj_frame_cache):
                        raise ValueError("Inj frame cahce file does not exist")

            elif self.data_source in FAKE_DATASOURCES:
                # gwdata-noise and gwdata-noise-realtime determine their own
                # sample rate from PSD
                if (
                    self.data_source not in ["gwdata-noise", "gwdata-noise-realtime"]
                    and self.input_sample_rate is None
                ):
                    excluded = ["gwdata-noise", "gwdata-noise-realtime"]
                    sources = [ds for ds in FAKE_DATASOURCES if ds not in excluded]
                    raise ValueError(
                        "Must specify input_sample_rate when data_source is one of"
                        f" {sources}"
                    )

    @staticmethod
    def from_options(options):
        return DataSourceInfo(
            data_source=options.data_source,
            channel_name=options.channel_name,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
            frame_cache=options.frame_cache,
            frame_segments_file=options.frame_segments_file,
            frame_segments_name=options.frame_segments_name,
            noiseless_inj_frame_cache=options.noiseless_inj_frame_cache,
            noiseless_inj_channel_name=options.noiseless_inj_channel_name,
            state_channel_name=options.state_channel_name,
            state_vector_on_bits=options.state_vector_on_bits,
            shared_memory_dir=options.shared_memory_dir,
            discont_wait_time=options.discont_wait_time,
            source_queue_timeout=options.source_queue_timeout,
            input_sample_rate=options.input_sample_rate,
            impulse_position=options.impulse_position,
            real_time=getattr(options, "real_time", False),
            state_segments_file=getattr(options, "state_segments_file", None),
            state_sample_rate=getattr(options, "state_sample_rate", 16),
        )

    @staticmethod
    def append_options(parser):
        group = parser.add_argument_group("Data source", "Options for data source.")
        group.add_argument(
            "--data-source",
            action="store",
            required=True,
            help=f"The type of the input source. Supported: {KNOWN_DATASOURCES}",
        )
        group.add_argument(
            "--channel-name",
            metavar="ifo=channel-name",
            action="append",
            required=True,
            help="Name of the data channel to analyze. Can be given multiple times as "
            "--channel-name=IFO=CHANNEL-NAME. For fake sources, channel name is used"
            " to derive the ifo names",
        )
        group.add_argument(
            "--gps-start-time",
            metavar="seconds",
            type=int,
            help="Set the start time of the segment to analyze in GPS seconds. "
            "For frame cache data source",
        )
        group.add_argument(
            "--gps-end-time",
            metavar="seconds",
            type=int,
            help="Set the end time of the segment to analyze in GPS seconds. "
            "For frame cache data source",
        )
        group.add_argument(
            "--frame-cache",
            metavar="filename",
            help="Set the path to the frame cache file to analyze.",
        )
        group.add_argument(
            "--frame-segments-file",
            metavar="filename",
            help="Set the name of the LIGO light-weight XML file from which to load"
            " frame segments.",
        )
        group.add_argument(
            "--frame-segments-name",
            metavar="name",
            help="Set the name of the segments to extract from the segment tables."
            " Required iff --frame-segments-file is given",
        )
        group.add_argument(
            "--noiseless-inj-frame-cache",
            metavar="filename",
            help="Set the name of the LAL cache listing the noiseless LIGO-Virgo"
            " injection .gwf frame files (optional, must also provide --frame-cache).",
        )
        group.add_argument(
            "--noiseless-inj-channel-name",
            metavar="name",
            action="append",
            help="Set the name of the noiseless injection channels to process. Can be"
            " given multiple times as --channel-name=IFO=CHANNEL-NAME (optional, must"
            " also provide --channel-name per ifo)",
        )
        group.add_argument(
            "--state-channel-name",
            metavar="ifo=channel-name",
            action="append",
            help="Set the state vector channel name. "
            "Can be given multiple times as --state-channel-name=IFO=CHANNEL-NAME",
        )
        group.add_argument(
            "--state-vector-on-bits",
            metavar="ifo=number",
            action="append",
            help="Set the state vector on bits. "
            "Can be given multiple times as --state-vector-on-bits=IFO=NUMBER",
        )
        group.add_argument(
            "--shared-memory-dir",
            metavar="ifo=directory",
            action="append",
            help="Set the name of the shared memory directory. "
            "Can be given multiple times as --shared-memory-dir=IFO=DIR-NAME",
        )
        group.add_argument(
            "--discont-wait-time",
            metavar="seconds",
            type=float,
            default=60,
            help="Time to wait for new files in seconds before dropping data. "
            "Default wait time is 60 seconds.",
        )
        group.add_argument(
            "--source-queue-timeout",
            metavar="seconds",
            type=float,
            default=1,
            help="Time to wait for new files from the queue in seconds before sending "
            "a hearbeat buffer. In online mode, new files should always arrive every "
            "second, unless there are problems. Default timeout is 1 second.",
        )
        group.add_argument(
            "--input-sample-rate",
            metavar="Hz",
            type=int,
            help="Input sample rate. Required if data-source one of [white|sin| "
            "white-realtime]",
        )
        group.add_argument(
            "--impulse-position",
            type=int,
            action="store",
            help="The sample point to put the impulse at.",
        )
        group.add_argument(
            "--state-segments-file",
            metavar="filename",
            help="Path to file with three columns (start end value) defining state "
            "vector segments for gwdata-noise sources. Each row defines a time segment "
            "with a specific bitmask value.",
        )
        group.add_argument(
            "--state-sample-rate",
            metavar="Hz",
            type=int,
            default=16,
            help=(
                "Sample rate for state vector channels when using gwdata-noise "
                "sources (default: 16 Hz)"
            ),
        )
        group.add_argument(
            "--real-time",
            action="store_true",
            help="Generate data in real time (for white-realtime and "
            "gwdata-noise sources). Note: gwdata-noise-realtime data source "
            "automatically enables real-time mode.",
        )


def datasource(
    pipeline: Pipeline,
    info: DataSourceInfo,
    source_latency: bool = False,
    verbose: bool = False,
):
    """Wrapper around sgn source elements

    Args:
        pipeline:
            Pipeline, the sgn pipeline
        data_source_info:
            DataSoureInfo, the data source info object containing all the data source
            options
    """

    if info.frame_segments_file is not None:
        frame_segments = ligolw_segments.segmenttable_get_by_name(
            ligolw_utils.load_filename(
                info.frame_segments_file,
                contenthandler=ligolw_segments.LIGOLWContentHandler,
            ),
            info.frame_segments_name,
        ).coalesce()
        if info.seg is not None:
            # Clip frame segments to seek segment if it
            # exists (not required, just saves some
            # memory and I/O overhead)
            frame_segments = segments.segmentlistdict(
                (ifo, seglist & segments.segmentlist([info.seg]))
                for ifo, seglist in frame_segments.items()
            )
        for ifo, segs in frame_segments.items():
            frame_segments[ifo] = [segments.segment(s[0].ns(), s[1].ns()) for s in segs]

        # FIXME: find a better way to get the analysis ifos. In gstlal this is obtained
        # from the time-slide file
        info.all_analysis_ifos = list(frame_segments.keys())
    else:
        # if no frame segments provided, set them to an empty segment list dictionary
        frame_segments = segments.segmentlistdict((ifo, None) for ifo in info.ifos)
        info.all_analysis_ifos = info.ifos

    source_out_links = {}
    pad_names = {}
    if source_latency:
        source_latency_links: Optional[dict[Any, Any]] = {}
    else:
        source_latency_links = None

    if info.data_source == "devshm":
        source_name = "_Gate"
        channel_names = {}
        for ifo in info.ifos:
            pad_names[ifo] = ifo
            channel_name_ifo = f"{ifo}:{info.channel_dict[ifo]}"
            state_channel_name_ifo = f"{ifo}:{info.state_channel_dict[ifo]}"
            channel_names[ifo] = [channel_name_ifo, state_channel_name_ifo]
        devshm = DevShmSource(
            name="DevShm",
            channel_names=channel_names,
            shared_memory_dirs=info.shared_memory_dict,
            discont_wait_time=info.discont_wait_time,
            queue_timeout=info.source_queue_timeout,
            verbose=verbose,
        )
        pipeline.insert(devshm)
        for ifo in info.ifos:
            bit_mask = BitMask(
                name=ifo + "_Mask",
                sink_pad_names=(ifo,),
                source_pad_names=(ifo,),
                bit_mask=int(info.state_vector_on_dict[ifo]),
            )
            gate = Gate(
                name=ifo + source_name,
                sink_pad_names=("strain", "state_vector"),
                control="state_vector",
                source_pad_names=(ifo,),
            )
            info.input_sample_rate = devshm.rates[ifo][channel_names[ifo][0]]
            pipeline.insert(
                bit_mask,
                gate,
                link_map={
                    ifo + "_Gate:snk:strain": "DevShm:src:" + channel_names[ifo][0],
                    ifo + "_Mask:snk:" + ifo: "DevShm:src:" + channel_names[ifo][1],
                    ifo + "_Gate:snk:state_vector": ifo + "_Mask:src:" + ifo,
                },
            )
            source_out_links[ifo] = ifo + source_name + ":src:" + pad_names[ifo]

    elif info.data_source == "arrakis":
        # Check if state vector gating is enabled
        use_state_vector = (
            hasattr(info, "state_channel_dict") and info.state_channel_dict is not None
        )

        # Prepare for ArrakisSource which handles all channels with a single source
        _channel_names = []

        # Create channel names list and set up pad_names in one loop
        for ifo in info.ifos:
            channel_name = f"{ifo}:{info.channel_dict[ifo]}"
            _channel_names.append(channel_name)

            if use_state_vector:
                # Add state vector channel
                state_channel_name = f"{ifo}:{info.state_channel_dict[ifo]}"
                _channel_names.append(state_channel_name)
                pad_names[ifo] = ifo  # Use IFO name for gated output
            else:
                pad_names[ifo] = channel_name
                source_out_links[ifo] = f"ArrakisSource:src:{channel_name}"

        # Create a single ArrakisSource for all channels
        arrakis_source = ArrakisSource(
            name="ArrakisSource",
            source_pad_names=_channel_names,
            start_time=info.gps_start_time,
            duration=(
                None
                if info.gps_end_time is None
                else (
                    info.gps_end_time - info.gps_start_time
                    if info.gps_start_time is not None
                    else None
                )
            ),
            in_queue_timeout=int(info.source_queue_timeout),
        )
        pipeline.insert(arrakis_source)

        # If state vector gating is enabled, add BitMask and Gate transforms
        if use_state_vector:
            source_name = "_Gate"
            for ifo in info.ifos:
                channel_name = f"{ifo}:{info.channel_dict[ifo]}"
                state_channel_name = f"{ifo}:{info.state_channel_dict[ifo]}"

                bit_mask = BitMask(
                    name=ifo + "_Mask",
                    sink_pad_names=(ifo,),
                    source_pad_names=(ifo,),
                    bit_mask=int(info.state_vector_on_dict[ifo]),
                )
                gate = Gate(
                    name=ifo + source_name,
                    sink_pad_names=("strain", "state_vector"),
                    control="state_vector",
                    source_pad_names=(ifo,),
                )
                pipeline.insert(
                    bit_mask,
                    gate,
                    link_map={
                        ifo + "_Gate:snk:strain": "ArrakisSource:src:" + channel_name,
                        ifo
                        + "_Mask:snk:"
                        + ifo: "ArrakisSource:src:"
                        + state_channel_name,
                        ifo + "_Gate:snk:state_vector": ifo + "_Mask:src:" + ifo,
                    },
                )
                source_out_links[ifo] = ifo + source_name + ":src:" + pad_names[ifo]

        # For Arrakis source, we need to set a default sample rate if not provided
        if info.input_sample_rate is None:
            info.input_sample_rate = 16384  # Default LIGO sample rate for h(t) data
    else:
        for ifo in info.ifos:
            if info.data_source == "frames":
                pad_name = ifo + ":" + info.channel_dict[ifo]
                pad_names[ifo] = pad_name
                source_name = "_FrameSource"
                frame_reader = FrameReader(
                    name=ifo + source_name,
                    framecache=info.frame_cache,
                    channel_names=[
                        ifo + ":" + info.channel_dict[ifo],
                    ],
                    instrument=ifo,
                    t0=info.gps_start_time,
                    end=info.gps_end_time,
                )
                info.input_sample_rate = next(iter(frame_reader.rates.values()))
                pipeline.insert(
                    frame_reader,
                )
                if info.noiseless_inj_frame_cache is not None:
                    print("Connecting noiseless injection frame source")
                    pipeline.insert(
                        FrameReader(
                            name=ifo + "_InjSource",
                            framecache=info.noiseless_inj_frame_cache,
                            channel_names=[
                                ifo + ":" + info.noiseless_inj_channel_dict[ifo]
                            ],
                            instrument=ifo,
                            t0=info.gps_start_time,
                            end=info.gps_end_time,
                        ),
                        Adder(
                            name=ifo + "_InjAdd",
                            sink_pad_names=("frame", "inj"),
                            source_pad_names=(ifo,),
                        ),
                        link_map={
                            ifo
                            + "_InjAdd:snk:frame": ifo
                            + "_FrameSource:src:"
                            + ifo
                            + ":"
                            + info.channel_dict[ifo],
                            ifo
                            + "_InjAdd:snk:inj": ifo
                            + "_InjSource:src:"
                            + ifo
                            + ":"
                            + info.noiseless_inj_channel_dict[ifo],
                        },
                    )
                    source_name = "_InjAdd"
                    pad_names[ifo] = ifo
            elif (
                info.data_source == "gwdata-noise"
                or info.data_source == "gwdata-noise-realtime"
            ):
                # Handle GWDataNoiseSource differently as it creates all
                # channels at once
                break  # Exit the loop after setting up pad_names
            elif info.data_source == "white-realtime":
                pad_names[ifo] = ifo
                source_name = "_FakeSource"
                source_pad_names = (ifo,)
                pipeline.insert(
                    FakeSeriesSource(
                        name=ifo + "_FakeSource",
                        source_pad_names=source_pad_names,
                        rate=info.input_sample_rate,
                        real_time=True,
                        t0=info.gps_start_time,
                        end=info.gps_end_time,
                    ),
                )
            else:
                pad_names[ifo] = ifo
                source_name = "_FakeSource"
                source_pad_names = (ifo,)
                pipeline.insert(
                    FakeSeriesSource(
                        name=ifo + "_FakeSource",
                        source_pad_names=source_pad_names,
                        rate=info.input_sample_rate,
                        signal_type=info.data_source,
                        impulse_position=info.impulse_position,
                        t0=info.gps_start_time,
                        end=info.gps_end_time,
                    ),
                )

            source_out_links[ifo] = ifo + source_name + ":src:" + pad_names[ifo]

            if info.frame_segments_file is not None:
                pipeline.insert(
                    SegmentSource(
                        name=ifo + "_SegmentSource",
                        source_pad_names=(ifo,),
                        rate=info.input_sample_rate,
                        t0=info.gps_start_time,
                        end=info.gps_end_time,
                        segments=frame_segments[ifo],
                    ),
                    Gate(
                        name=ifo + "_Gate",
                        sink_pad_names=("strain", "control"),
                        source_pad_names=(ifo,),
                        control="control",
                    ),
                    link_map={
                        ifo + "_Gate:snk:strain": source_out_links[ifo],  # type: ignore
                        ifo + "_Gate:snk:control": ifo + "_SegmentSource:src:" + ifo,
                    },
                )
                assert source_out_links is not None
                source_out_links[ifo] = ifo + "_Gate:src:" + ifo

        # Handle GWDataNoiseSource after the loop since it creates all channels at once
        if (
            info.data_source == "gwdata-noise"
            or info.data_source == "gwdata-noise-realtime"
        ):
            # Prepare channel dict with full channel names for GWDataNoiseSource
            gwdata_channel_dict = {}
            for ifo in info.ifos:
                # If channel name doesn't start with IFO:, add it
                channel = info.channel_dict[ifo]
                if not channel.startswith(f"{ifo}:"):
                    channel = f"{ifo}:{channel}"
                gwdata_channel_dict[ifo] = channel

            # Determine if using real-time mode
            is_realtime = (info.data_source == "gwdata-noise-realtime") or (
                info.data_source == "gwdata-noise" and info.real_time
            )

            # GWDataNoiseSource handles all channels in a single source (strain data)
            gwdata_source = GWDataNoiseSource(
                name="GWDataNoiseSource",
                channel_dict=gwdata_channel_dict,
                t0=info.gps_start_time,
                end=info.gps_end_time,
                real_time=is_realtime,
                verbose=verbose,
            )
            pipeline.insert(gwdata_source)

            # Set up the output links for strain channels
            for ifo in info.ifos:
                # Use the full channel name from gwdata_channel_dict
                channel_name = gwdata_channel_dict[ifo]
                pad_names[ifo] = channel_name
                source_out_links[ifo] = f"GWDataNoiseSource:src:{channel_name}"

            # Also create SegmentSource for state vector channels if requested
            # This follows the same pattern as devshm: strain + state vector +
            # optional gating
            if info.state_channel_name is not None:
                # Parse state channel dict
                state_channel_dict = parse_list_to_dict(info.state_channel_name)

                # Read state segments from file if provided, otherwise use default
                if info.state_segments_file is not None:
                    state_segments, state_values = read_segments_and_values_from_file(
                        info.state_segments_file, verbose
                    )
                else:
                    # Default: single segment covering entire time range with value 3
                    # (simulating HOFT_OK + OBS_INTENT bits set)
                    if info.gps_start_time is not None:
                        start_ns = int(info.gps_start_time * 1e9)
                        if info.gps_end_time is not None:
                            end_ns = int(info.gps_end_time * 1e9)
                        else:
                            # For real-time mode without end time, use max int32
                            end_ns = int(np.iinfo(np.int32).max * 1e9)
                        state_segments = ((start_ns, end_ns),)
                        state_values = (3,)  # Default: bits 0 and 1 set
                        if verbose:
                            print(
                                "Using default state segments: single segment "
                                "with value 3"
                            )
                    else:
                        raise ValueError(
                            "Must provide either state_segments_file or gps_start_time "
                            "when using state_channel_name with gwdata-noise sources"
                        )

                # Create SegmentSource for each IFO's state vector channel
                for ifo in state_channel_dict.keys():
                    state_channel_name = state_channel_dict[ifo]
                    if not state_channel_name.startswith(f"{ifo}:"):
                        state_channel_name = f"{ifo}:{state_channel_name}"

                    # SegmentSource doesn't support None for end time
                    seg_end = (
                        info.gps_end_time
                        if info.gps_end_time is not None
                        else float(np.iinfo(np.int32).max)
                    )

                    state_source = SegmentSource(
                        name=f"{ifo}_StateSrc",
                        source_pad_names=("state",),
                        rate=info.state_sample_rate,
                        t0=info.gps_start_time,
                        end=seg_end,
                        segments=state_segments,
                        values=state_values,
                    )
                    pipeline.insert(state_source)

                    if verbose:
                        print(f"Created state vector source for {state_channel_name}")

                    # If state_vector_on_bits is specified, apply BitMask + Gate
                    # This follows the devshm pattern
                    if info.state_vector_on_bits is not None:
                        source_name = "_Gate"

                        bit_mask = BitMask(
                            name=ifo + "_Mask",
                            sink_pad_names=(ifo,),
                            source_pad_names=(ifo,),
                            bit_mask=int(info.state_vector_on_dict[ifo]),
                        )
                        gate = Gate(
                            name=ifo + source_name,
                            sink_pad_names=("strain", "state_vector"),
                            control="state_vector",
                            source_pad_names=(ifo,),
                        )

                        # Get the strain channel name from gwdata_channel_dict
                        strain_channel = gwdata_channel_dict[ifo]

                        pipeline.insert(
                            bit_mask,
                            gate,
                            link_map={
                                f"{ifo}_Gate:snk:strain": (
                                    f"GWDataNoiseSource:src:{strain_channel}"
                                ),
                                f"{ifo}_Mask:snk:{ifo}": f"{ifo}_StateSrc:src:state",
                                f"{ifo}_Gate:snk:state_vector": (
                                    f"{ifo}_Mask:src:{ifo}"
                                ),
                            },
                        )

                        # Update source_out_links to point to gated output
                        pad_names[ifo] = ifo
                        source_out_links[ifo] = ifo + source_name + ":src:" + ifo

                        if verbose:
                            print(
                                f"Applied BitMask + Gate for {ifo} with mask "
                                f"{info.state_vector_on_dict[ifo]}"
                            )

            # Set the input sample rate from the source
            if info.input_sample_rate is None:
                # GWDataNoiseSource uses detector-specific rates, default to 16384 Hz
                info.input_sample_rate = 16384

    if source_latency:
        for ifo in info.ifos:
            pipeline.insert(
                Latency(
                    name=ifo + "_SourceLatency",
                    sink_pad_names=("data",),
                    source_pad_names=("latency",),
                    route=ifo + "_datasource_latency",
                    interval=1,
                ),
                link_map={
                    ifo
                    + "_SourceLatency:snk:data": source_out_links[ifo]  # type: ignore
                },
            )
            assert source_latency_links is not None
            source_latency_links[ifo] = ifo + "_SourceLatency:src:latency"

    return source_out_links, source_latency_links
