from __future__ import annotations

from argparse import ArgumentParser
from math import isinf
from typing import Optional

from sgn import Pipeline
from sgnts.transforms import Resampler, Threshold

from sgnligo.transforms.latency import Latency
from sgnligo.transforms.whiten import Whiten


def parse_command_line_condition(parser: Optional[ArgumentParser] = None):
    if parser is None:
        parser = ArgumentParser()

    group = parser.add_argument_group(
        "PSD Options", "Adjust noise spectrum estimation parameters"
    )
    group.add_argument(
        "--whitening-method",
        metavar="algorithm",
        default="gstlal",
        help="Algorithm to use for whitening the data. Supported options are 'gwpy' "
        "or 'gstlal'. Default is gstlal.",
    )
    group.add_argument(
        "--psd-fft-length",
        action="store",
        type=int,
        default=8,
        help="The fft length for psd estimation.",
    )
    group.add_argument(
        "--reference-psd",
        metavar="file",
        help="load the spectrum from this LIGO light-weight XML file (optional).",
    )
    group.add_argument(
        "--track-psd",
        action="store_true",
        help="Enable dynamic PSD tracking.  Always enabled if --reference-psd is not "
        "given.",
    )
    group.add_argument(
        "--whiten-sample-rate",
        metavar="Hz",
        action="store",
        type=int,
        default=2048,
        help="Sample rate at which to whiten the data and generate the PSD, default"
        " 2048 Hz.",
    )

    group = parser.add_argument_group("Data Qualtiy", "Adjust data quality handling")
    group.add_argument(
        "--ht-gate-threshold",
        action="store",
        type=float,
        default=float("+inf"),
        help="The gating threshold. Data above this value will be gated out.",
    )

    return parser


def condition_from_options(pipeline: Pipeline, options, ifos: list[str]):
    return condition(
        pipeline=pipeline,
        ifos=ifos,
        whiten_sample_rate=options.whiten_sample_rate,
        input_links=options.input_links,
        data_source=options.data_source,
        input_sample_rate=options.input_sample_rate,
        psd_fft_length=options.psd_fft_length,
        whitening_method=options.whitening_method,
        reference_psd=options.reference_psd,
        ht_gate_threshold=options.ht_gate_threshold,
    )


def condition(
    pipeline: Pipeline,
    data_source: str,
    input_sample_rate: int,
    whiten_sample_rate: int,
    ifos: list[str],
    input_links: list[str],
    psd_fft_length: int = 8,
    whitening_method: str = "gstlal",
    ht_gate_threshold: float = float("+inf"),
    reference_psd: Optional[str] = None,
):
    """Condition the data with whitening and gating

    Args:
        pipeline:
            Pipeline: the sgn pipeline
        data_source:
            str, the data source for the pipeline
        input_sample_rate:
            int, the sample rate of the data
        whiten_sample_rate:
            int, the sample rate to perform the whitening
        ifos:
            list[str], the ifo names
        input_links:
            the src pad names to link to this element
        psd_fft_length:
            int, the fft length for the psd calculation, in seconds
        whitening_method:
            str, the whitening method, must be either 'gwpy' or 'gstlal'
        ht_gate_threshold:
            float, the threshold above which to gate out data
        reference_psd:
            str, the filename for the reference psd used in the Whiten element
    """
    condition_out_links = {ifo: None for ifo in ifos}
    spectrum_out_links = {ifo: None for ifo in ifos}
    if data_source == "devshm":
        latency_out_links = {ifo: None for ifo in ifos}
    else:
        latency_out_links = None

    for ifo in ifos:

        # Downsample and whiten
        pipeline.insert(
            Resampler(
                name=ifo + "_SourceResampler",
                sink_pad_names=(ifo,),
                source_pad_names=(ifo,),
                inrate=input_sample_rate,
                outrate=whiten_sample_rate,
            ),
            Whiten(
                name=ifo + "_Whitener",
                sink_pad_names=(ifo,),
                instrument=ifo,
                psd_pad_name="spectrum_" + ifo,
                whiten_pad_name=ifo,
                sample_rate=whiten_sample_rate,
                fft_length=psd_fft_length,
                whitening_method=whitening_method,
                reference_psd=reference_psd,
            ),
            link_map={
                ifo + "_SourceResampler:sink:" + ifo: input_links[ifo],
                ifo + "_Whitener:sink:" + ifo: ifo + "_SourceResampler:src:" + ifo,
            },
        )
        spectrum_out_links[ifo] = ifo + "_Whitener:src:spectrum_" + ifo

        # Apply htgate
        if not isinf(ht_gate_threshold):
            pipeline.insert(
                Threshold(
                    name=ifo + "_Threshold",
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    threshold=ht_gate_threshold,
                    startwn=whiten_sample_rate // 2,
                    stopwn=whiten_sample_rate // 2,
                    invert=True,
                ),
                link_map={
                    ifo + "_Threshold:sink:" + ifo: ifo + "_Whitener:src:" + ifo,
                },
            )
            condition_out_links[ifo] = ifo + "_Threshold:src:" + ifo
        else:
            condition_out_links[ifo] = ifo + "_Whitener:src:" + ifo

        if data_source == "devshm":
            pipeline.insert(
                Latency(
                    name=ifo + "_Latency",
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    route=ifo + "_whitening_latency",
                ),
                link_map={
                    ifo + "_Latency:sink:" + ifo: ifo + "_Whitener:src:" + ifo,
                },
            )
            latency_out_links[ifo] = ifo + "_Latency:src:" + ifo

    return condition_out_links, spectrum_out_links, latency_out_links
