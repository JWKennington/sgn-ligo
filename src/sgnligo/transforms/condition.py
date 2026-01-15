"""An SGN graph to condition incoming data with whitening and gating.

Zero-latency usage (production):
- Enable with the --zero-latency command-line option.
- This flag is provided by ConditionInfo.append_options and requires no code
  changes in sgnl scripts; it is parsed by sgnl/bin/inspiral.py and
  propagated through ConditionInfo to this function.
- In zero-latency mode, the Whiten element still computes and publishes the PSD
  (spectrum_* pads).
- The whitening used downstream comes from a chain of two AFIR (AdaptiveCorrelate)
  elements:
    1. Whitening Stage: Driven by WhiteningKernel (Minimum Phase).
    2. Drift Correction Stage: Driven by DriftCorrectionKernel (corrects for
       differences between the live PSD and the static reference PSD).
- If input_sample_rate differs from whiten_sample_rate, a Resampler is inserted
  before the AFIR chain.
- Gating (Threshold) is applied after the final AFIR stage.
- Detailed Latency: If --detailed-latency is enabled, Latency elements are
  inserted after each major stage (Resampler, Whitening, Drift Correction) to
  measure granular processing delays.
"""

# Copyright (C) 2009-2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from math import isinf
from typing import Optional

from sgn import Pipeline
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import AdaptiveCorrelate, Resampler, Threshold

from sgnligo.psd import read_psd as _read_psd
from sgnligo.transforms.latency import Latency
from sgnligo.transforms.whiten import DriftCorrectionKernel, Whiten, WhiteningKernel


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
        detailed_latency:
            bool, default False, insert Latency probes after each major stage.
    """

    whiten_sample_rate: int = 2048
    psd_fft_length: int = 8
    reference_psd: Optional[str] = None
    ht_gate_threshold: float = float("+inf")
    track_psd: bool = True
    zero_latency: bool = False
    detailed_latency: bool = False
    drift_correction: bool = True

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
        group.add_argument(
            "--detailed-latency",
            action="store_true",
            default=False,
            help="Insert Latency elements after each major stage (Resampler, AFIRs) "
            "to measure granular pipeline latency.",
        )

        group.add_argument(
            "--no-drift-correction",
            action="store_false",
            dest="drift_correction",
            default=True,
            help="Disable the drift correction stage in zero-latency mode. "
            "Drift correction is enabled by default if a reference PSD is provided.",
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
            detailed_latency=getattr(options, "detailed_latency", False),
            drift_correction=getattr(options, "drift_correction", True),
        )


def condition(
    pipeline: Pipeline,
    condition_info: ConditionInfo,
    ifos: list[str],
    data_source: str,
    input_sample_rate: int,
    input_links: dict[str, str],
    whiten_sample_rate: Optional[int] = None,
    whiten_latency: bool = False,
    highpass_filter: bool = False,
    zero_latency: bool = False,
    detailed_latency: bool = False,
    drift_correction: bool = True,
) -> tuple[dict[str, str], dict[str, str], Optional[dict[str, str]]]:
    """Condition the data with whitening and gating.

    This function wires a conditioning subgraph per IFO.

    Args:
        pipeline: The sgn pipeline
        ifos: List of ifo names
        data_source: The data source name
        input_sample_rate: Input data rate
        input_links: Source pad names to link (dict by IFO)
        whiten_sample_rate: Target whitening rate
        whiten_latency: Enable final output latency telemetry
        highpass_filter: Enable highpass in Whiten
        zero_latency: Enable zero-latency AFIR path
        detailed_latency: Enable intermediate latency probes
        drift_correction: Enable drift correction stage in zero-latency path
    """
    if whiten_sample_rate is None:
        whiten_sample_rate = condition_info.whiten_sample_rate

    if not zero_latency:
        zero_latency = getattr(condition_info, "zero_latency", False)
    if not detailed_latency:
        detailed_latency = getattr(condition_info, "detailed_latency", False)
    if drift_correction:
        drift_correction = getattr(condition_info, "drift_correction", True)

    ref_psds = {}
    if zero_latency and condition_info.reference_psd:
        try:
            ref_psds = _read_psd(condition_info.reference_psd, verbose=True)
        except Exception as e:
            print(f"Warning: Could not load reference PSD for drift correction: {e}")

    condition_out_links: dict[str, str] = {}
    spectrum_out_links: dict[str, str] = {}
    whiten_latency_out_links: Optional[dict[str, str]] = (
        {} if (whiten_latency or detailed_latency) else None
    )

    for ifo in ifos:
        # 1. Whiten (PSD Estimation)
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
            link_map={f"{whiten_name}:snk:{ifo}": input_links[ifo]},  # type: ignore
        )
        spectrum_out_links[ifo] = f"{whiten_name}:src:spectrum_{ifo}"  # type: ignore
        whitening_output_link = f"{whiten_name}:src:{ifo}"

        if zero_latency:
            current_link = input_links[ifo]

            # A. Null Sink for unused Whiten output
            null_name = f"{ifo}_NullWhiten"
            pipeline.insert(
                NullSeriesSink(name=null_name, sink_pad_names=(ifo,)),
                link_map={f"{null_name}:snk:{ifo}": f"{whiten_name}:src:{ifo}"},
            )

            # B. Resampling
            if input_sample_rate < whiten_sample_rate:
                raise ValueError("Zero-latency path requires downsampling.")

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
                    link_map={f"{resamp_name}:snk:{ifo}": current_link},
                )
                current_link = f"{resamp_name}:src:{ifo}"

                if detailed_latency:
                    lat_name = f"{ifo}_Lat_Resamp"
                    pipeline.insert(
                        Latency(
                            name=lat_name,
                            source_pad_names=(ifo,),
                            sink_pad_names=(ifo,),
                            route=f"{ifo}_latency_resamp",
                            interval=1,
                        ),
                        link_map={f"{lat_name}:snk:{ifo}": current_link},
                    )
                    # Sidecar: Do not update current_link
                    assert whiten_latency_out_links is not None
                    whiten_latency_out_links[f"{ifo}_resamp"] = f"{lat_name}:src:{ifo}"

            # C. Whitening (AFIR 1)
            kern_whiten_name = f"{ifo}_KernWhiten"
            pipeline.insert(
                WhiteningKernel(
                    name=kern_whiten_name,
                    sink_pad_names=(f"spectrum_{ifo}",),
                    filters_pad_name="filters",
                    zero_latency=True,
                ),
                link_map={
                    f"{kern_whiten_name}:snk:spectrum_{ifo}": spectrum_out_links[ifo]
                },
            )

            afir_whiten_name = f"{ifo}_AFIR_Whiten"
            pipeline.insert(
                AdaptiveCorrelate(
                    name=afir_whiten_name,
                    sink_pad_names=(ifo,),
                    source_pad_names=(ifo,),
                    sample_rate=whiten_sample_rate,
                    filter_sink_name="filters",
                ),
                link_map={
                    f"{afir_whiten_name}:snk:{ifo}": current_link,
                    f"{afir_whiten_name}:snk:filters": f"{kern_whiten_name}"
                    f":src:filters",
                },
            )
            current_link = f"{afir_whiten_name}:src:{ifo}"

            # Check if this is the final stage (no drift correction)
            is_final_stage = ifo not in ref_psds
            # Only add detailed latency if it's NOT redundant with the general latency
            add_whiten_lat = detailed_latency and not (
                is_final_stage and whiten_latency
            )

            if add_whiten_lat:
                lat_name = f"{ifo}_Lat_Whiten"
                pipeline.insert(
                    Latency(
                        name=lat_name,
                        source_pad_names=(ifo,),
                        sink_pad_names=(ifo,),
                        route=f"{ifo}_latency_whiten",
                        interval=1,
                    ),
                    link_map={f"{lat_name}:snk:{ifo}": current_link},
                )
                assert whiten_latency_out_links is not None
                whiten_latency_out_links[f"{ifo}_whiten"] = f"{lat_name}:src:{ifo}"

            # D. Drift Correction (AFIR 2)
            if drift_correction and (ifo in ref_psds):
                kern_drift_name = f"{ifo}_KernDrift"
                pipeline.insert(
                    DriftCorrectionKernel(
                        name=kern_drift_name,
                        sink_pad_names=(f"spectrum_{ifo}",),
                        filters_pad_name="filters",
                        reference_psd=ref_psds[ifo],
                        truncation_samples=128,
                        smoothing_hz=0.5,
                    ),
                    link_map={
                        f"{kern_drift_name}:snk:spectrum_{ifo}": spectrum_out_links[ifo]
                    },
                )

                afir_drift_name = f"{ifo}_AFIR_Drift"
                pipeline.insert(
                    AdaptiveCorrelate(
                        name=afir_drift_name,
                        sink_pad_names=(ifo,),
                        source_pad_names=(ifo,),
                        sample_rate=whiten_sample_rate,
                        filter_sink_name="filters",
                    ),
                    link_map={
                        f"{afir_drift_name}:snk:{ifo}": current_link,
                        f"{afir_drift_name}:snk:filters": f"{kern_drift_name}"
                        f":src:filters",
                    },
                )
                current_link = f"{afir_drift_name}:src:{ifo}"

                # Only add if NOT redundant with general latency
                # (since Drift is final stage)
                if detailed_latency and not whiten_latency:
                    lat_name = f"{ifo}_Lat_Drift"
                    pipeline.insert(
                        Latency(
                            name=lat_name,
                            source_pad_names=(ifo,),
                            sink_pad_names=(ifo,),
                            route=f"{ifo}_latency_drift",
                            interval=1,
                        ),
                        link_map={f"{lat_name}:snk:{ifo}": current_link},
                    )
                    assert whiten_latency_out_links is not None
                    whiten_latency_out_links[f"{ifo}_drift"] = f"{lat_name}:src:{ifo}"

            whitening_output_link = current_link

        # 2. Gating
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
                link_map={f"{thresh_name}:snk:{ifo}": whitening_output_link},
            )
            condition_out_links[ifo] = f"{thresh_name}:src:{ifo}"  # type: ignore
        else:
            condition_out_links[ifo] = whitening_output_link  # type: ignore

        # 3. Final Latency
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
                link_map={f"{lat_name}:snk:{ifo}": whitening_output_link},
            )
            assert whiten_latency_out_links is not None
            whiten_latency_out_links[ifo] = f"{lat_name}:src:{ifo}"  # type: ignore

    return condition_out_links, spectrum_out_links, whiten_latency_out_links
