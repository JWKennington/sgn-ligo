from sgnligo.base import parse_list_to_dict
from sgnligo.sources import FrameReader, DevShmSrc 

def datasource(pipeline, options, ifos):
    source_out_links = {ifo: None for ifo in ifos}
    channel_dict = parse_list_to_dict(options.channel_name)
    state_channel_dict = parse_list_to_dict(options.state_channel_name)
    state_vector_on_dict = parse_list_to_dict(options.state_vector_on_bits)
    shared_memory_dict = parse_list_to_dict(options.shared_memory_dir)
    for ifo in ifos:
        num_samples = int(options.source_buffer_duration * options.sample_rate)
        if options.data_source == "frames":
            source_name = "_FrameSource"
            pipeline.insert(
                FrameReader(
                    name=ifo + source_name,
                    source_pad_names=(ifo,),
                    rate=options.sample_rate,
                    num_samples=num_samples,
                    framecache=options.frame_cache,
                    channel_name=channel_dict[ifo],
                    instrument=ifo,
                    gps_start_time=options.gps_start_time,
                    gps_end_time=options.gps_end_time,
                ),
            )
        elif options.data_source == "devshm":
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
                    wait_time=options.wait_time,
                    #verbose=options.verbose,
                    verbose=True,
                ),
            )
        else:
            source_name = "_FakeSource"
            if options.data_source == "impulse":
                source_pad_names = (ifo,)
                signal_type = "impulse"
                impulse_position = options.impulse_position
            elif options.data_source == "white":
                source_pad_names = (ifo,)
                signal_type = "white"
                impulse_position = None
            elif options.data_source == "sin":
                source_pad_names = (ifo,)
                signal_type = "sin"
                impulse_position = None
            pipeline.insert(
                FakeSeriesSrc(
                    name=ifo + source_name,
                    source_pad_names=source_pad_names,
                    num_buffers=options.num_buffers,
                    rate=options.sample_rate,
                    num_samples=options.num_samples,
                    signal_type=signal_type,
                    impulse_position=options.impulse_position,
                    verbose=options.verbose,
                ),
            )
        source_out_links[ifo] = ifo + source_name + ":src:" + ifo

    return source_out_links
