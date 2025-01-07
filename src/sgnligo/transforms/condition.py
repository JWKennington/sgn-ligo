"""An SGN graph to condition incoming data with whitening and gating."""

# Copyright (C) 2009-2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from math import isinf

from sgn import Pipeline
from sgnts.transforms import Resampler, Threshold

from sgnligo.transforms.latency import Latency
from sgnligo.transforms.whiten import Whiten


@dataclass
class ConditionInfo:
    """Condition options for whitening and gating

    Args:
        whiten_sample_rate:
            int, the sample rate to perform the whitening
        psd_fft_length:
            int, the fft length for the psd calculation, in seconds
        whitening_method:
            str, the whitening method, must be either 'gwpy' or 'gstlal'
        ht_gate_threshold:
            float, the threshold above which to gate out data
        reference_psd:
            str, the filename for the reference psd used in the Whiten element
        track_psd:
            bool, default True, whether to track psd in gstlal whitening method
    """

    whiten_sample_rate: int = 2048
    psd_fft_length: int = 8
    whitening_method: str = "gstlal"
    reference_psd: str = None
    ht_gate_threshold: float = float("+inf")
    track_psd: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.whitening_method not in ["gwpy", "gstlal"]:
            raise ValueError("Whitening method must be either 'gwpy' or 'gstlal'")

        if self.reference_psd is None and self.track_psd is False:
            raise ValueError("Must enable track_psd if reference_psd not provided")

    @staticmethod
    def append_options(parser: ArgumentParser):
        group = parser.add_argument_group(
            "PSD Options", "Adjust noise spectrum estimation parameters"
        )
        group.add_argument(
            "--whitening-method",
            metavar="algorithm",
            default="gstlal",
            help="Algorithm to use for whitening the data. Supported options are"
            " 'gwpy' or 'gstlal'. Default is gstlal.",
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
            default=True,
            help="Enable dynamic PSD tracking.  Always enabled if --reference-psd is"
            " not given.",
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

        group = parser.add_argument_group(
            "Data Qualtiy", "Adjust data quality handling"
        )
        group.add_argument(
            "--ht-gate-threshold",
            action="store",
            type=float,
            default=float("+inf"),
            help="The gating threshold. Data above this value will be gated out.",
        )

    @staticmethod
    def from_options(options):
        return ConditionInfo(
            whiten_sample_rate=options.whiten_sample_rate,
            psd_fft_length=options.psd_fft_length,
            whitening_method=options.whitening_method,
            reference_psd=options.reference_psd,
            ht_gate_threshold=options.ht_gate_threshold,
            track_psd=options.track_psd,
        )


def condition(
    pipeline: Pipeline,
    condition_info: ConditionInfo,
    ifos: list[str],
    data_source: str,
    input_sample_rate: int,
    input_links: list[str],
    whiten_latency: bool = False,
):
    """Condition the data with whitening and gating

    Args:
        pipeline:
            Pipeline: the sgn pipeline
        ifos:
            list[str], the ifo names
        data_source:
            str, the data source for the pipeline
        input_sample_rate:
            int, the sample rate of the data
        input_links:
            the src pad names to link to this element
    """
    condition_out_links = {ifo: None for ifo in ifos}
    spectrum_out_links = {ifo: None for ifo in ifos}
    if whiten_latency is True:
        whiten_latency_out_links = {ifo: None for ifo in ifos}
    else:
        whiten_latency_out_links = None

    for ifo in ifos:

        # Downsample and whiten
        pipeline.insert(
            Resampler(
                name=ifo + "_SourceResampler",
                sink_pad_names=(ifo,),
                source_pad_names=(ifo,),
                inrate=input_sample_rate,
                outrate=condition_info.whiten_sample_rate,
            ),
            Whiten(
                name=ifo + "_Whitener",
                sink_pad_names=(ifo,),
                instrument=ifo,
                psd_pad_name="spectrum_" + ifo,
                whiten_pad_name=ifo,
                sample_rate=condition_info.whiten_sample_rate,
                fft_length=condition_info.psd_fft_length,
                whitening_method=condition_info.whitening_method,
                reference_psd=condition_info.reference_psd,
            ),
            link_map={
                ifo + "_SourceResampler:snk:" + ifo: input_links[ifo],
                ifo + "_Whitener:snk:" + ifo: ifo + "_SourceResampler:src:" + ifo,
            },
        )
        spectrum_out_links[ifo] = ifo + "_Whitener:src:spectrum_" + ifo

        # Apply htgate
        if not isinf(condition_info.ht_gate_threshold):
            pipeline.insert(
                Threshold(
                    name=ifo + "_Threshold",
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    threshold=condition_info.ht_gate_threshold,
                    startwn=condition_info.whiten_sample_rate // 2,
                    stopwn=condition_info.whiten_sample_rate // 2,
                    invert=True,
                ),
                link_map={
                    ifo + "_Threshold:snk:" + ifo: ifo + "_Whitener:src:" + ifo,
                },
            )
            condition_out_links[ifo] = ifo + "_Threshold:src:" + ifo
        else:
            condition_out_links[ifo] = ifo + "_Whitener:src:" + ifo

        if whiten_latency is True:
            pipeline.insert(
                Latency(
                    name=ifo + "_Latency",
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    route=ifo + "_whitening_latency",
                ),
                link_map={
                    ifo + "_Latency:snk:" + ifo: ifo + "_Whitener:src:" + ifo,
                },
            )
            whiten_latency_out_links[ifo] = ifo + "_Latency:src:" + ifo

    return condition_out_links, spectrum_out_links, whiten_latency_out_links
