from . import Whiten, HorizonDistance, Latency
from sgnts.transforms import Resampler, Threshold


def condition(pipeline, options, ifos, maxrate, input_links):
    condition_out_links = {ifo: None for ifo in ifos}
    if options.data_source == "devshm":
        latency_out_links = {ifo: None for ifo in ifos}
        horizon_out_links = None
    else:
        latency_out_links = None
        horizon_out_links = {ifo: None for ifo in ifos}
    for ifo in ifos:
        pipeline.insert(
            Resampler(
                name=ifo + "_SourceResampler",
                sink_pad_names=(ifo,),
                source_pad_names=(ifo,),
                inrate=options.sample_rate,
                outrate=maxrate,
            ),
            Whiten(
                name=ifo + "_Whitener",
                sink_pad_names=(ifo,),
                source_pad_names=(ifo, "spectrum_" + ifo),
                instrument=ifo,
                sample_rate=maxrate,
                fft_length=options.psd_fft_length,
                whitening_method=options.whitening_method,
                reference_psd=options.reference_psd,
                psd_pad_name=ifo + "_Whitener:src:spectrum_" + ifo,
            ),
            Threshold(
                name=ifo + "_Threshold",
                source_pad_names=(ifo,),
                sink_pad_names=(ifo,),
                threshold=options.ht_gate_threshold,
                startwn=maxrate // 2,
                stopwn=maxrate // 2,
                invert=True,
            ),
        )
        if options.data_source == "devshm":
            pipeline.insert(
                Latency(
                    name=ifo + "_Latency",
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    route=ifo+"_whitening_latency",
                ),
                link_map={
                    ifo + "_Latency:sink:" + ifo: ifo + "_Whitener:src:" + ifo,
                }
            )
        else:
            pipeline.insert(
                HorizonDistance(
                    name=ifo + "_Horizon",
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    m1=1.4,
                    m2=1.4,
                    fmin=10.0,
                    fmax=1000.0,
                    delta_f=1 / 16.0,
                ),
                link_map={
                    ifo + "_Horizon:sink:" + ifo: ifo + "_Whitener:src:spectrum_" + ifo,
                }
            )
        pipeline.insert(
            link_map={
                ifo + "_SourceResampler:sink:" + ifo: input_links[ifo],
                ifo + "_Whitener:sink:" + ifo: ifo + "_SourceResampler:src:" + ifo,
                ifo + "_Threshold:sink:" + ifo: ifo + "_Whitener:src:" + ifo,
            }
        )
        condition_out_links[ifo] = ifo + "_Threshold:src:" + ifo
        if options.data_source == "devshm":
            latency_out_links[ifo] = ifo + "_Latency:src:" + ifo
        else:
            horizon_out_links[ifo] = ifo + "_Horizon:src:" + ifo

    return condition_out_links, horizon_out_links, latency_out_links
