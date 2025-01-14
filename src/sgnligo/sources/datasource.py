"""Datasource element utilities for LIGO pipelines."""

# Copyright (C) 2009-2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from lal import LIGOTimeGPS
from ligo import segments
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import segments as ligolw_segments
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource, SegmentSource
from sgnts.transforms import Adder, Gate

from sgnligo.base import parse_list_to_dict
from sgnligo.sources.devshmsrc import DevShmSource
from sgnligo.sources.framecachesrc import FrameReader
from sgnligo.transforms import BitMask, Latency

KNOWN_DATASOURCES = ["white", "sin", "impulse", "white-realtime", "frames", "devshm"]
FAKE_DATASOURCES = ["white", "sin", "impulse", "white-realtime"]
OFFLINE_DATASOURCES = ["white", "sin", "impulse", "frames"]


@dataclass
class DataSourceInfo:
    """Wrapper around data source options

    Args:
        data_source:
            str, the data source, can be one of
            [white|sin|impulse|white-realtime|frames|devshm]
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
            float, the time to wait for next file before dropping data when data_souce
            is "devshm", in seconds
        source_queue_timeout:
            float, the time to wait for next file from the queue before sending a
            hearbeat buffer when data_souce is "devshm", in seconds
        input_sample_rate:
            int, the sample rate for fake sources [white|sin|impulse|white-realtime]
        impulse_position:
            int, the sample point position to place the impulse data point. Default -1,
            which will generate the impulse position randomly
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

    def __post_init__(self):
        self.channel_dict = parse_list_to_dict(self.channel_name)
        self.ifos = sorted(self.channel_dict.keys())
        self.seg = None
        self.validate()

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
                    "Must specify shared_memory_dir when data_source='devshm'"
                )
            else:
                self.shared_memory_dict = parse_list_to_dict(self.shared_memory_dir)
                if sorted(self.shared_memory_dict.keys()) != self.ifos:
                    raise ValueError(
                        "Must specify same number of shared_memory_dir as channel_name"
                    )
            if self.state_channel_name is None:
                raise ValueError(
                    "Must specify state_channel_name when data_source='devshm'"
                )
            else:
                self.state_channel_dict = parse_list_to_dict(self.state_channel_name)
                if sorted(self.state_channel_dict.keys()) != self.ifos:
                    raise ValueError(
                        "Must specify same number of state_channel_name as channel_name"
                    )
            if self.state_vector_on_bits is None:
                raise ValueError(
                    "Must specify state_vector_on_bits when data_source='devshm'"
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
                    " data_source='devshm'"
                )
        elif self.data_source == "white-realtime":
            if self.input_sample_rate is None:
                raise ValueError(
                    "Must specify input_sample_rate when data_source is one of"
                    f" {FAKE_DATASOURCES}"
                )
        else:
            if self.gps_start_time is None or self.gps_end_time is None:
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

                # Validate noiseless injection frame cache comes with hoft frame cache
                if self.noiseless_inj_frame_cache:
                    if not self.frame_cache:
                        raise ValueError(
                            "Must specify hoft frame_cache to add to"
                            " noiseless_inj_frame_cache"
                        )
                    elif not os.path.exists(self.noiseless_inj_frame_cache):
                        raise ValueError("Inj frame cahce file does not exist")

            elif self.data_source in FAKE_DATASOURCES:
                if self.input_sample_rate is None:
                    raise ValueError(
                        "Must specify input_sample_rate when data_source is one of"
                        f" {FAKE_DATASOURCES}"
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

    source_out_links = {ifo: None for ifo in info.ifos}
    pad_names = {ifo: None for ifo in info.ifos}
    if source_latency:
        source_latency_links = {}
    else:
        source_latency_links = None
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
        elif info.data_source == "devshm":
            pad_names[ifo] = ifo
            source_name = "_Gate"
            channel_name_ifo = f"{ifo}:{info.channel_dict[ifo]}"
            state_channel_name_ifo = f"{ifo}:{info.state_channel_dict[ifo]}"
            devshm = DevShmSource(
                name=ifo + "_Devshm",
                channel_names=[channel_name_ifo, state_channel_name_ifo],
                shared_memory_dir=info.shared_memory_dict[ifo],
                discont_wait_time=info.discont_wait_time,
                queue_timeout=info.source_queue_timeout,
                verbose=verbose,
            )
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
            info.input_sample_rate = devshm.rates[channel_name_ifo]
            pipeline.insert(
                devshm,
                bit_mask,
                gate,
                link_map={
                    ifo + "_Gate:snk:strain": ifo + "_Devshm:src:" + channel_name_ifo,
                    ifo
                    + "_Mask:snk:"
                    + ifo: ifo
                    + "_Devshm:src:"
                    + state_channel_name_ifo,
                    ifo + "_Gate:snk:state_vector": ifo + "_Mask:src:" + ifo,
                },
            )
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
                    verbose=verbose,
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
                    ifo + "_Gate:snk:strain": source_out_links[ifo],
                    ifo + "_Gate:snk:control": ifo + "_SegmentSource:src:" + ifo,
                },
            )
            source_out_links[ifo] = ifo + "_Gate:src:" + ifo

        if source_latency:
            pipeline.insert(
                Latency(
                    name=ifo + "_SourceLatency",
                    sink_pad_names=("data",),
                    source_pad_names=("latency",),
                    route=ifo + "_datasource_latency",
                    interval=1,
                ),
                link_map={ifo + "_SourceLatency:snk:data": source_out_links[ifo]},
            )
            source_latency_links[ifo] = ifo + "_SourceLatency:src:latency"

    return source_out_links, source_latency_links
