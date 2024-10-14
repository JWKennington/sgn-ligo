"""Datasource element utilities for LIGO pipelines
"""

from typing import Sequence

from sgn import Pipeline
from sgnligo.base import parse_list_to_dict
from sgnligo.sources import DevShmSrc, FrameReader
from sgnts.sources import FakeSeriesSrc


def datasource_from_options(pipeline: Pipeline, options, ifos):
    return datasource(
        pipeline=pipeline,
        ifos=ifos,
        channel_name=options.channel_name,
        state_channel_name=options.state_channel_name,
        state_vector_on_bits=options.state_vector_on_bits,
        shared_memory_dir=options.shared_memory_dir,
        source_buffer_duration=options.source_buffer_duration,
        sample_rate=options.sample_rate,
        data_source=options.data_source,
        frame_cache=options.frame_cache,
        gps_start_time=options.gps_start_time,
        gps_end_time=options.gps_end_time,
        wait_time=options.wait_time,
        num_buffers=options.num_buffers,
        verbose=options.verbose,
    )


def datasource(
    pipeline: Pipeline,
    ifos: Sequence[str],
    channel_name=None,
    state_channel_name=None,
    state_vector_on_bits=None,
    shared_memory_dir=None,
    source_buffer_duration=None,
    sample_rate=None,
    data_source=None,
    frame_cache: str = None,
    gps_start_time=None,
    gps_end_time=None,
    wait_time=None,
    num_buffers=None,
    verbose: bool = False,
):
    source_out_links = {ifo: None for ifo in ifos}
    channel_dict = parse_list_to_dict(channel_name)
    state_channel_dict = parse_list_to_dict(state_channel_name)
    state_vector_on_dict = parse_list_to_dict(state_vector_on_bits)
    shared_memory_dict = parse_list_to_dict(shared_memory_dir)
    for ifo in ifos:
        num_samples = int(source_buffer_duration * sample_rate)
        if data_source == "frames":
            source_name = "_FrameSource"
            pipeline.insert(
                FrameReader(
                    name=ifo + source_name,
                    source_pad_names=(ifo,),
                    rate=sample_rate,
                    num_samples=num_samples,
                    framecache=frame_cache,
                    channel_name=channel_dict[ifo],
                    instrument=ifo,
                    gps_start_time=gps_start_time,
                    gps_end_time=gps_end_time,
                ),
            )
        elif data_source == "devshm":
            source_name = "_DevShmSource"
            pipeline.insert(
                DevShmSrc(
                    name=ifo + source_name,
                    source_pad_names=(ifo,),
                    rate=16384,
                    num_samples=16384,
                    channel_name=channel_dict[ifo],
                    state_channel_name=state_channel_dict[ifo],
                    instrument=ifo,
                    shared_memory_dir=shared_memory_dict[ifo],
                    state_vector_on_bits=int(state_vector_on_dict[ifo]),
                    wait_time=wait_time,
                    # verbose=verbose,
                    verbose=True,
                ),
            )
        else:
            source_name = "_FakeSource"
            if data_source == "impulse":
                source_pad_names = (ifo,)
                signal_type = "impulse"
                impulse_position = impulse_position
            elif data_source == "white":
                source_pad_names = (ifo,)
                signal_type = "white"
                impulse_position = None
            elif data_source == "sin":
                source_pad_names = (ifo,)
                signal_type = "sin"
                impulse_position = None
            pipeline.insert(
                FakeSeriesSrc(
                    name=ifo + source_name,
                    source_pad_names=source_pad_names,
                    num_buffers=num_buffers,
                    rate=sample_rate,
                    num_samples=num_samples,
                    signal_type=signal_type,
                    impulse_position=impulse_position,
                    verbose=verbose,
                ),
            )
        source_out_links[ifo] = ifo + source_name + ":src:" + ifo

    return source_out_links
