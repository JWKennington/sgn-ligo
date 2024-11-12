"""Datasource element utilities for LIGO pipelines
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Optional

from sgn import Pipeline
from sgnts.sources import FakeSeriesSrc
from sgnts.transforms import Gate

from sgnligo.base import parse_list_to_dict
from sgnligo.sources.devshmsrc import DevShmSrc
from sgnligo.sources.framecachesrc import FrameReader
from sgnligo.transforms import BitMask


def parse_command_line_datasource(parser: Optional[ArgumentParser] = None):
    if parser is None:
        parser = ArgumentParser()

    group = parser.add_argument_group("Data source", "Options for data source.")
    group.add_argument(
        "--data-source",
        action="store",
        required=True,
        help="The type of the input source. Supported sources: 'white', 'sin', "
        "'impulse', 'frames', 'devshm'",
    )
    group.add_argument(
        "--channel-name",
        metavar="ifo=channel-name",
        action="append",
        required=True,
        help="Name of the data channel to analyze. Can be given multiple times as "
        "--channel-name=IFO=CHANNEL-NAME. For fake sources, channel name is used to "
        "derive the ifo names",
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
        metavar="file",
        help="Set the path to the frame cache file to analyze.",
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
        "--wait-time",
        metavar="seconds",
        type=int,
        default=60,
        help="Time to wait for new files in seconds before throwing an error. "
        "In online mode, new files should always arrive every second, unless "
        "there are problems. Default wait time is 60 seconds.",
    )
    group.add_argument(
        "--input-sample-rate",
        metavar="Hz",
        type=int,
        help="Input sample rate. Required if data-source one of [white, sin]",
    )
    group.add_argument(
        "--impulse-position",
        type=int,
        action="store",
        help="The sample point to put the impulse at.",
    )
    group.add_argument(
        "--verbose-datasource",
        action="store_true",
        help="Be verbose.",
    )

    return parser


def datasource_from_options(pipeline: Pipeline, options):
    return datasource(
        pipeline=pipeline,
        data_source=options.data_source,
        channel_name=options.channel_name,
        gps_start_time=options.gps_start_time,
        gps_end_time=options.gps_end_time,
        frame_cache=options.frame_cache,
        state_channel_name=options.state_channel_name,
        state_vector_on_bits=options.state_vector_on_bits,
        shared_memory_dir=options.shared_memory_dir,
        wait_time=options.wait_time,
        input_sample_rate=options.input_sample_rate,
        impulse_position=options.impulse_position,
        verbose=options.verbose_datasource,
    )


def datasource(
    pipeline: Pipeline,
    data_source: str,
    channel_name: list[str],
    gps_start_time: Optional[float] = None,
    gps_end_time: Optional[float] = None,
    frame_cache: Optional[str] = None,
    state_channel_name: Optional[list[str]] = None,
    state_vector_on_bits: Optional[list[int]] = None,
    shared_memory_dir: Optional[list[str]] = None,
    wait_time: float = 60,
    input_sample_rate: Optional[int] = None,
    impulse_position: int = -1,
    verbose: bool = False,
):
    """Wrapper around sgn source elements

    Args:
        pipeline:
            Pipeline, the sgn pipeline
        data_source:
            str, the data source, can be one of
            [white|sin|impulse|frames|devshm]
        channel_name:
            list[str, ...], a list of channel names ["IFO=CHANNEL_NAME",...].
            For fake sources [white|sin|impulse], channel names are used to derive ifos.
        gps_start_time:
            float, the gps start time of the data to analyze, in seconds
        gps_end_time:
            float, the gps end time of the data to analyze, in seconds
        frame_cache:
            str, the frame cache file to read gwf frame files from. Must be provided
            when data_source is "frames"
        state_channel_name:
            list, a list of state vector channel names
        state_vector_on_bits:
            int, the bit mask for the state vector data
        shared_memory_dir:
            str, the path to the shared memory directory to read low-latency data from
        wait_time:
            float, the time to wait for next file when data_souce is "devshm", in
            seconds
        input_sample_rate:
            int, the sample rate for fake sources [white|sin|impulse]
        impulse_position:
            int, the sample point position to place the impulse data point. Default -1,
            which will generate the impulse position randomly
        verbose:
            bool, be verbose
    """
    # sanity check options
    known_datasources = ["white", "sin", "impulse", "frames", "devshm"]
    fake_datasources = ["white", "sin", "impulse"]

    if data_source == "devshm":
        if shared_memory_dir is None:
            raise ValueError("Must specify shared_memory_dir when data_source='devshm'")
        elif state_channel_name is None:
            raise ValueError(
                "Must specify state_channel_name when data_source='devshm'"
            )
        elif state_vector_on_bits is None:
            raise ValueError(
                "Must specify state_vector_on_bits when data_source='devshm'"
            )
        else:
            state_channel_dict = parse_list_to_dict(state_channel_name)
            state_vector_on_dict = parse_list_to_dict(state_vector_on_bits)
            shared_memory_dict = parse_list_to_dict(shared_memory_dir)
    else:
        if gps_start_time is None or gps_end_time is None:
            raise ValueError(
                "Must specify gps_start_time and gps_end_time when "
                f"data_source is one of {fake_datasources} or 'frames'"
            )
        if data_source == "frames":
            if frame_cache is None:
                raise ValueError("Must specify frame_cache when data_source='frames'")
        elif data_source in fake_datasources:
            if input_sample_rate is None:
                raise ValueError(
                    "Must specify input_sample_rate when data_source is one of 'white',"
                    "'sin', 'impulse'"
                )
        else:
            raise ValueError(f"Unknown data source, must be one of {known_datasources}")

    channel_dict = parse_list_to_dict(channel_name)
    ifos = list(channel_dict.keys())
    source_out_links = {ifo: None for ifo in ifos}
    pad_names = {ifo: None for ifo in ifos}
    for ifo in ifos:
        if data_source == "frames":
            pad_name = ifo + ":" + channel_dict[ifo]
            pad_names[ifo] = pad_name
            source_name = "_FrameSource"
            frame_reader = FrameReader(
                name=ifo + source_name,
                framecache=frame_cache,
                channel_names=[
                    ifo + ":" + channel_dict[ifo],
                ],
                instrument=ifo,
                t0=gps_start_time,
                end=gps_end_time,
            )
            input_sample_rate = next(iter(frame_reader.rates.values()))
            pipeline.insert(frame_reader)
        elif data_source == "devshm":
            pad_names[ifo] = ifo
            source_name = "_Gate"
            channel_name_ifo = f"{ifo}:{channel_dict[ifo]}"
            state_channel_name_ifo = f"{ifo}:{state_channel_dict[ifo]}"
            devshm = DevShmSrc(
                name=ifo + "_Devshm",
                channel_name=channel_name_ifo,
                state_channel_name=state_channel_name_ifo,
                instrument=ifo,
                shared_memory_dir=shared_memory_dict[ifo],
                wait_time=wait_time,
                verbose=verbose,
            )
            bit_mask = BitMask(
                name=ifo + "_Mask",
                sink_pad_names=(ifo,),
                source_pad_names=(ifo,),
                bit_mask=int(state_vector_on_dict[ifo]),
            )
            gate = Gate(
                name=ifo + source_name,
                sink_pad_names=("strain", "state_vector"),
                control="state_vector",
                source_pad_names=(ifo,),
            )
            input_sample_rate = devshm.rate_dict[channel_name_ifo]
            pipeline.insert(
                devshm,
                bit_mask,
                gate,
                link_map={
                    ifo + "_Gate:sink:strain": ifo + "_Devshm:src:" + channel_name_ifo,
                    ifo
                    + "_Mask:sink:"
                    + ifo: ifo
                    + "_Devshm:src:"
                    + state_channel_name_ifo,
                    ifo + "_Gate:sink:state_vector": ifo + "_Mask:src:" + ifo,
                },
            )
        else:
            pad_names[ifo] = ifo
            source_name = "_FakeSource"
            if data_source == "impulse":
                source_pad_names = (ifo,)
                signal_type = "impulse"
            elif data_source == "white":
                source_pad_names = (ifo,)
                signal_type = "white"
            elif data_source == "sin":
                source_pad_names = (ifo,)
                signal_type = "sin"
            pipeline.insert(
                FakeSeriesSrc(
                    name=ifo + source_name,
                    source_pad_names=source_pad_names,
                    rate=input_sample_rate,
                    signal_type=signal_type,
                    impulse_position=impulse_position,
                    verbose=verbose,
                    t0=gps_start_time,
                    end=gps_end_time,
                ),
            )
        source_out_links[ifo] = ifo + source_name + ":src:" + pad_names[ifo]

    return source_out_links, input_sample_rate
