"""An SGN graph to condition incoming data with whitening and gating.

Zero-latency usage (production):
- Enable with the --zero-latency command-line option.
- This flag is provided by ConditionInfo.append_options and requires no code
  changes in sgnl scripts; it is parsed by sgnl/bin/inspiral.py and
  propagated through ConditionInfo to this function.
- In zero-latency mode, the Whiten element still computes and publishes the PSD
  (spectrum_* pads). The whitening used downstream comes from an AFIR branch
  driven by PSD->kernel updates via PsdToMPKernel (minimum-phase when zero_latency=True,
  linear-phase otherwise). If input_sample_rate differs from whiten_sample_rate, a
  Resampler is inserted before AFIR.
- Gating (Threshold) is applied after the whitening branch that is actually used
  downstream (after Whiten for standard mode; after AFIR for zero-latency).
- If whitening latency telemetry is requested, it is computed from the chosen
  whitening output (AFIR when zero-latency is enabled, otherwise Whiten).
"""

# Copyright (C) 2009-2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from math import isinf
from typing import Optional

# For AFIR default filters
import numpy as np
from sgn import Pipeline
from sgnts.base import EventBuffer
from sgnts.base.slice_tools import TIME_MAX
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import AdaptiveCorrelate, Resampler, Threshold

from sgnligo.psd import read_psd as _read_psd
from sgnligo.transforms.latency import Latency
from sgnligo.transforms.whiten import PsdToMPKernel, Whiten, kernel_from_psd


@dataclass
class ConditionInfo:
    """Condition options for whitening and gating

    Args:
        whiten_sample_rate:
            int, the sample rate to perform the whitening
        psd_fft_length:
            int, the fft length for the psd calculation, in seconds
        ht_gate_threshold:
            float, the threshold above which to gate out data
        reference_psd:
            str, the filename for the reference psd used in the Whiten element
        track_psd:
            bool, default True, whether to track psd
        zero_latency:
            bool, default False, enable zero-latency whitening via AFIR.
    """

    whiten_sample_rate: int = 2048
    psd_fft_length: int = 8
    reference_psd: Optional[str] = None
    ht_gate_threshold: float = float("+inf")
    track_psd: bool = True
    zero_latency: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.reference_psd is None and self.track_psd is False:
            raise ValueError("Must enable track_psd if reference_psd not provided")

    @staticmethod
    def append_options(parser: ArgumentParser):
        group = parser.add_argument_group(
            "PSD Options", "Adjust noise spectrum estimation parameters"
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
        group.add_argument(
            "--zero-latency",
            action="store_true",
            default=False,
            help="Enable zero-latency whitening using AFIR (AdaptiveCorrelate) driven "
            "by PSD->kernel updates.",
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
            reference_psd=options.reference_psd,
            ht_gate_threshold=options.ht_gate_threshold,
            track_psd=options.track_psd,
            zero_latency=getattr(options, "zero_latency", False),
        )


def condition(
    pipeline: Pipeline,
    condition_info: ConditionInfo,
    ifos: list[str],
    data_source: str,
    input_sample_rate: int,
    input_links: list[str],
    whiten_sample_rate: Optional[int] = None,
    whiten_latency: bool = False,
    highpass_filter: bool = False,
    zero_latency: bool = False,
):
    """Condition the data with whitening and gating.

    This function wires a conditioning subgraph per IFO that produces:
      - a whitened data stream suitable for downstream matched filtering
      - a PSD stream ("spectrum") for QA/monitoring and for zero-latency mode

    Two whitening modes are supported:
      1) Standard whitening (default): uses the Whiten element, which internally
         resamples to whiten_sample_rate and applies overlap-save FFT whitening.
      2) Zero-latency whitening (optional): builds an explicit, optional branch
         that converts the running PSD to a time-domain whitening FIR kernel and
         applies it with AdaptiveCorrelate (AFIR). In this mode, the Whiten
         element is still used to estimate and publish the PSD but the whitening
         output used by downstream consumers comes from AFIR. We also explicitly
         downsample the raw input stream (if needed) before AFIR to match
         whiten_sample_rate.

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
            the src pad names to link to this element (dict-like mapping by IFO)
        whiten_sample_rate:
            Optional[int], desired whitening sample rate; defaults to
            condition_info.whiten_sample_rate
        whiten_latency:
            bool, whether to emit whitening latency telemetry frames
        highpass_filter:
            bool, whether to enable the highpass inside Whiten (standard mode)
        zero_latency:
            bool, enable explicit zero-latency whitening branch using AFIR
    """
    if whiten_sample_rate is None:
        whiten_sample_rate = condition_info.whiten_sample_rate

    # Allow zero-latency to be enabled via ConditionInfo without changing callers
    if not zero_latency:
        zero_latency = getattr(condition_info, "zero_latency", False)

    condition_out_links = {ifo: None for ifo in ifos}
    spectrum_out_links = {ifo: None for ifo in ifos}
    whiten_latency_out_links = {ifo: None for ifo in ifos} if whiten_latency else None

    for ifo in ifos:
        # Always insert Whiten to compute PSD (and optionally provide standard hoft)
        whiten_name = f"{ifo}_Whitener"
        pipeline.insert(
            Whiten(
                name=whiten_name,
                sink_pad_names=(ifo,),
                instrument=ifo,
                psd_pad_name=f"spectrum_{ifo}",
                whiten_pad_name=ifo,
                input_sample_rate=input_sample_rate,
                whiten_sample_rate=whiten_sample_rate,
                fft_length=condition_info.psd_fft_length,
                reference_psd=condition_info.reference_psd,
                highpass_filter=highpass_filter,
            ),
            link_map={
                f"{whiten_name}:snk:{ifo}": input_links[ifo],  # type: ignore
            },
        )
        spectrum_out_links[ifo] = f"{whiten_name}:src:spectrum_{ifo}"  # type: ignore

        # Determine which whitening output stream downstream should consume.
        whitening_output_link = f"{whiten_name}:src:{ifo}"  # default (standard)

        if zero_latency:
            # In zero-latency mode, consume Whiten's hoft output to avoid unlinked
            # pad errors.
            null_name = f"{ifo}_NullWhiten"
            pipeline.insert(
                NullSeriesSink(
                    name=null_name,
                    sink_pad_names=(ifo,),
                ),
                link_map={
                    f"{null_name}:snk:{ifo}": f"{whiten_name}:src:{ifo}",
                },
            )

            # Enforce downsampling-only behavior before AFIR. In zero-latency mode we
            # do not permit upsampling; the Resampler, if inserted, must reduce the
            # rate from input_sample_rate to whiten_sample_rate.
            if input_sample_rate < whiten_sample_rate:
                raise ValueError(
                    "Zero-latency path requires downsampling: "
                    f"input_sample_rate={input_sample_rate} < "
                    f"whiten_sample_rate={whiten_sample_rate} is not allowed."
                )

            # Optional explicit downsampling before AFIR if rates differ
            afir_input_link = input_links[ifo]
            if input_sample_rate != whiten_sample_rate:
                resamp_name = f"{ifo}_Resampler"
                pipeline.insert(
                    Resampler(
                        name=resamp_name,
                        source_pad_names=(ifo,),
                        sink_pad_names=(ifo,),
                        inrate=input_sample_rate,
                        outrate=whiten_sample_rate,
                    ),
                    link_map={
                        f"{resamp_name}:snk:{ifo}": input_links[ifo],  # type: ignore
                    },
                )
                afir_input_link = f"{resamp_name}:src:{ifo}"  # type: ignore

            # Convert PSD updates into whitening FIR kernels
            psd2kern_name = f"{ifo}_Psd2Kernel"
            pipeline.insert(
                PsdToMPKernel(
                    name=psd2kern_name,
                    sink_pad_names=(f"spectrum_{ifo}",),
                    target_rate=whiten_sample_rate,
                    zero_latency=True,
                ),
                link_map={
                    f"{psd2kern_name}:snk:spectrum_{ifo}": spectrum_out_links[ifo],
                },
            )

            # Apply kernels with AdaptiveCorrelate (AFIR)
            afir_name = f"{ifo}_AFIR"

            # Insert AdaptiveCorrelate (AFIR) without explicit initial filters.
            # It will emit gaps until a kernel update arrives from PsdToMPKernel.
            pipeline.insert(
                AdaptiveCorrelate(
                    name=afir_name,
                    sink_pad_names=(ifo,),
                    source_pad_names=(ifo,),
                    sample_rate=whiten_sample_rate,
                    filter_sink_name="filters",
                ),
                link_map={
                    f"{afir_name}:snk:{ifo}": afir_input_link,  # time-series input
                    f"{afir_name}:snk:filters": f"{psd2kern_name}:src:filters",
                },
            )

            # Downstream consumers use AFIR output
            whitening_output_link = f"{afir_name}:src:{ifo}"

        # Apply htgate after the chosen whitening output
        if not isinf(condition_info.ht_gate_threshold):
            thresh_name = f"{ifo}_Threshold"
            pipeline.insert(
                Threshold(
                    name=thresh_name,
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    threshold=condition_info.ht_gate_threshold,
                    startwn=whiten_sample_rate // 2,
                    stopwn=whiten_sample_rate // 2,
                    invert=True,
                ),
                link_map={
                    f"{thresh_name}:snk:{ifo}": whitening_output_link,
                },
            )
            condition_out_links[ifo] = f"{thresh_name}:src:{ifo}"  # type: ignore
        else:
            condition_out_links[ifo] = whitening_output_link  # type: ignore

        # Latency telemetry: attach to the whitening output actually used
        if whiten_latency:
            lat_name = f"{ifo}_Latency"
            pipeline.insert(
                Latency(
                    name=lat_name,
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    route=f"{ifo}_whitening_latency",
                    interval=1,
                ),
                link_map={
                    f"{lat_name}:snk:{ifo}": whitening_output_link,
                },
            )
            whiten_latency_out_links[ifo] = f"{lat_name}:src:{ifo}"  # type: ignore

    return condition_out_links, spectrum_out_links, whiten_latency_out_links
