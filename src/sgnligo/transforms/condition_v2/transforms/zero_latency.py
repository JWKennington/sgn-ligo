"""Zero-latency conditioning transform with AFIR whitening.

This module provides the ZeroLatencyCondition class for signal conditioning
using the zero-latency AFIR (Adaptive Finite Impulse Response) path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import ClassVar, Dict

import lal
from sgnts.compose import TSCompose, TSComposedTransformElement
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import AdaptiveCorrelate, Resampler, Threshold

from sgnligo.psd import read_psd as _read_psd
from sgnligo.sources.datasource_v2.cli_mixins import IfosOnlyMixin
from sgnligo.transforms.condition_v2.cli_mixins import (
    GatingOptionsMixin,
    InputSampleRateOptionsMixin,
    LatencyTrackingOptionsMixin,
    PSDOptionsMixin,
    WhitenSampleRateOptionsMixin,
    ZeroLatencyOptionsMixin,
)
from sgnligo.transforms.condition_v2.composed_base import ComposedTransformBase
from sgnligo.transforms.condition_v2.composed_registry import (
    register_composed_transform,
)
from sgnligo.transforms.latency import Latency
from sgnligo.transforms.whiten import DriftCorrectionKernel, Whiten, WhiteningKernel


@register_composed_transform
@dataclass(kw_only=True)
class ZeroLatencyCondition(
    ComposedTransformBase,
    IfosOnlyMixin,
    InputSampleRateOptionsMixin,
    WhitenSampleRateOptionsMixin,
    PSDOptionsMixin,
    GatingOptionsMixin,
    LatencyTrackingOptionsMixin,
    ZeroLatencyOptionsMixin,
):
    """Zero-latency conditioning with AFIR whitening.

    This transform performs per-IFO signal conditioning using zero-latency
    AFIR (Adaptive Finite Impulse Response) filters:

    1. Whiten element computes the PSD (whitened output is discarded via NullSink)
    2. Optional Resampler if input_sample_rate != whiten_sample_rate
    3. WhiteningKernel + AdaptiveCorrelate for minimum-phase AFIR whitening
    4. Optional DriftCorrectionKernel + AdaptiveCorrelate for drift correction
    5. Optional threshold-based gating
    6. Optional per-stage latency tracking

    The zero-latency path uses minimum-phase filters for whitening, achieving
    lower latency than the standard path at the cost of asymmetric impulse response.

    Sink pads (inputs):
        - {ifo}: Raw strain input for each IFO at input_sample_rate

    Source pads (outputs):
        - {ifo}: Conditioned strain for each IFO at whiten_sample_rate
        - spectrum_{ifo}: PSD output for each IFO (metadata with LAL frequency series)
        - {ifo}_whitened_raw: Unused whitened output from Whiten (if not gating)
        - {ifo}_latency: (optional) Latency telemetry if whiten_latency enabled
        - {ifo}_resamp_latency: (optional) Resampler latency if detailed_latency
        - {ifo}_whiten_latency: (optional) AFIR whiten latency if detailed_latency
        - {ifo}_drift_latency: (optional) Drift correction latency if detailed

    Example:
        >>> cond = ZeroLatencyCondition(
        ...     name="condition",
        ...     ifos=["H1", "L1"],
        ...     input_sample_rate=16384,
        ...     whiten_sample_rate=2048,
        ...     psd_fft_length=8,
        ...     reference_psd="/path/to/psd.xml",
        ... )
        >>> pipeline = Pipeline()
        >>> pipeline.connect(source.element, cond.element)
        >>> pipeline.connect(cond.element, sink)
    """

    transform_type: ClassVar[str] = "zero-latency"
    description: ClassVar[str] = "Zero-latency whitening via AFIR"

    # Cache for loaded reference PSDs
    _ref_psds: Dict[str, lal.REAL8FrequencySeries] = None  # type: ignore

    def _validate(self) -> None:
        """Validate configuration."""
        if not self.ifos:
            raise ValueError("Must specify at least one IFO")
        if self.input_sample_rate < self.whiten_sample_rate:
            raise ValueError(
                "Zero-latency path requires downsampling "
                f"(input {self.input_sample_rate} >= output {self.whiten_sample_rate})"
            )
        if self.reference_psd is None and not self.track_psd:
            raise ValueError("Must enable track_psd if reference_psd not provided")

    def _load_reference_psds(self) -> Dict[str, lal.REAL8FrequencySeries]:
        """Load reference PSDs from file if drift correction is enabled."""
        if self._ref_psds is not None:
            return self._ref_psds

        self._ref_psds = {}
        if self.drift_correction and self.reference_psd:
            try:
                self._ref_psds = _read_psd(self.reference_psd, verbose=True)
            except Exception as e:
                print(
                    f"Warning: Could not load reference PSD for drift correction: {e}"
                )

        return self._ref_psds

    def _build(self) -> TSComposedTransformElement:
        """Build the zero-latency conditioning chain for each IFO."""
        compose = TSCompose()
        ref_psds = self._load_reference_psds()

        for ifo in self.ifos:
            self._build_ifo_chain(compose, ifo, ref_psds)

        return compose.as_transform(
            name=self.name,
            also_expose_source_pads=(
                self._also_expose_pads if self._also_expose_pads else None
            ),
        )

    def _build_ifo_chain(
        self,
        compose: TSCompose,
        ifo: str,
        ref_psds: Dict[str, lal.REAL8FrequencySeries],
    ) -> None:
        """Build the zero-latency conditioning chain for a single IFO.

        The data flow is:
        - Input -> Whiten (PSD only, whitened discarded via NullSink)
                -> WhiteningKernel -> AFIR
        - Input -> [Resampler] -> AFIR chain -> [Drift AFIR] -> [Threshold] -> Output
        """
        # 1. Whiten element - PSD estimation (whitened output will be discarded)
        whiten = Whiten(
            name=f"{self.name}_{ifo}_whiten",
            sink_pad_names=(ifo,),
            instrument=ifo,
            psd_pad_name=f"spectrum_{ifo}",
            whiten_pad_name=f"{ifo}_whitened_raw",  # Unique name for unused output
            input_sample_rate=self.input_sample_rate,
            whiten_sample_rate=self.whiten_sample_rate,
            fft_length=self.psd_fft_length,
            reference_psd=self.reference_psd,
            highpass_filter=False,  # Not needed in zero-latency path
        )
        compose.insert(whiten)

        # 2. NullSink to discard unused whitened output
        null_sink = NullSeriesSink(
            name=f"{self.name}_{ifo}_null",
            sink_pad_names=(f"{ifo}_whitened_raw",),
        )
        compose.connect(
            whiten,
            null_sink,
            link_map={f"{ifo}_whitened_raw": f"{ifo}_whitened_raw"},
        )

        # Track current element for the AFIR chain
        # Note: The AFIR chain receives input from the same boundary sink as Whiten
        # SGN supports fan-out (one source -> multiple sinks)

        # 3. Optional Resampler if sample rates differ
        needs_resampling = self.input_sample_rate != self.whiten_sample_rate
        resampler = None

        if needs_resampling:
            resampler = Resampler(
                name=f"{self.name}_{ifo}_resamp",
                source_pad_names=(ifo,),
                sink_pad_names=(ifo,),
                inrate=self.input_sample_rate,
                outrate=self.whiten_sample_rate,
            )
            compose.insert(resampler)

            # Optional detailed latency after resampler
            if self.detailed_latency:
                lat_resamp = Latency(
                    name=f"{self.name}_{ifo}_lat_resamp",
                    sink_pad_names=(ifo,),
                    source_pad_names=(f"{ifo}_resamp_latency",),
                    route=f"{ifo}_latency_resamp",
                    interval=1,
                )
                compose.connect(
                    resampler,
                    lat_resamp,
                    link_map={ifo: ifo},
                )
                # Register for multilink (resampler output still exposed)
                self._also_expose_pads.append(f"{resampler.name}:src:{ifo}")

        # 4. WhiteningKernel - converts PSD to FIR filter taps
        kern_whiten = WhiteningKernel(
            name=f"{self.name}_{ifo}_kern_whiten",
            sink_pad_names=(f"spectrum_{ifo}",),
            filters_pad_name="filters",
            zero_latency=True,
        )
        compose.connect(
            whiten,
            kern_whiten,
            link_map={f"spectrum_{ifo}": f"spectrum_{ifo}"},
        )
        # Expose spectrum pad externally for downstream PSD consumers
        self._also_expose_pads.append(f"{whiten.name}:src:spectrum_{ifo}")

        # 5. AFIR Whitening - adaptive correlate with whitening filter
        afir_whiten = AdaptiveCorrelate(
            name=f"{self.name}_{ifo}_afir_whiten",
            sink_pad_names=(ifo,),
            source_pad_names=(ifo,),
            sample_rate=self.whiten_sample_rate,
            filter_sink_name="filters",
        )

        # Connect resampler output (or boundary) to AFIR
        if resampler is not None:
            compose.connect(
                resampler,
                afir_whiten,
                link_map={ifo: ifo},
            )
        else:
            # AFIR gets input directly from boundary (same input as Whiten)
            compose.insert(afir_whiten)

        # Connect whitening kernel to AFIR
        compose.connect(
            kern_whiten,
            afir_whiten,
            link_map={"filters": "filters"},
        )

        current_element = afir_whiten
        current_pad = ifo

        # Optional detailed latency after AFIR whitening
        has_drift = self.drift_correction and (ifo in ref_psds)
        # Only add if not redundant with final latency
        if self.detailed_latency and not (not has_drift and self.whiten_latency):
            lat_whiten = Latency(
                name=f"{self.name}_{ifo}_lat_whiten",
                sink_pad_names=(ifo,),
                source_pad_names=(f"{ifo}_whiten_latency",),
                route=f"{ifo}_latency_whiten",
                interval=1,
            )
            compose.connect(
                current_element,
                lat_whiten,
                link_map={ifo: current_pad},
            )
            # Register for multilink
            self._also_expose_pads.append(f"{current_element.name}:src:{current_pad}")

        # 6. Optional Drift Correction (AFIR 2) - if reference PSD available
        if has_drift:
            kern_drift = DriftCorrectionKernel(
                name=f"{self.name}_{ifo}_kern_drift",
                sink_pad_names=(f"spectrum_{ifo}",),
                filters_pad_name="filters",
                reference_psd=ref_psds[ifo],
                truncation_samples=128,
                smoothing_hz=0.5,
            )
            compose.connect(
                whiten,
                kern_drift,
                link_map={f"spectrum_{ifo}": f"spectrum_{ifo}"},
            )

            afir_drift = AdaptiveCorrelate(
                name=f"{self.name}_{ifo}_afir_drift",
                sink_pad_names=(ifo,),
                source_pad_names=(ifo,),
                sample_rate=self.whiten_sample_rate,
                filter_sink_name="filters",
            )
            compose.connect(
                current_element,
                afir_drift,
                link_map={ifo: current_pad},
            )
            compose.connect(
                kern_drift,
                afir_drift,
                link_map={"filters": "filters"},
            )

            current_element = afir_drift
            current_pad = ifo

            # Optional detailed latency after drift correction
            # Only add if not redundant with final latency
            if self.detailed_latency and not self.whiten_latency:
                lat_drift = Latency(
                    name=f"{self.name}_{ifo}_lat_drift",
                    sink_pad_names=(ifo,),
                    source_pad_names=(f"{ifo}_drift_latency",),
                    route=f"{ifo}_latency_drift",
                    interval=1,
                )
                compose.connect(
                    current_element,
                    lat_drift,
                    link_map={ifo: current_pad},
                )
                # Register for multilink
                self._also_expose_pads.append(
                    f"{current_element.name}:src:{current_pad}"
                )

        # 7. Optional gating (threshold)
        if not isinf(self.ht_gate_threshold):
            threshold = Threshold(
                name=f"{self.name}_{ifo}_threshold",
                source_pad_names=(ifo,),
                sink_pad_names=(ifo,),
                threshold=self.ht_gate_threshold,
                startwn=self.whiten_sample_rate // 2,
                stopwn=self.whiten_sample_rate // 2,
                invert=True,
            )
            compose.connect(
                current_element,
                threshold,
                link_map={ifo: current_pad},
            )
            current_element = threshold
            current_pad = ifo

        # 8. Optional final latency tracking
        if self.whiten_latency:
            latency = Latency(
                name=f"{self.name}_{ifo}_latency",
                sink_pad_names=(ifo,),
                source_pad_names=(f"{ifo}_latency",),
                route=f"{ifo}_whitening_latency",
                interval=1,
            )
            compose.connect(
                current_element,
                latency,
                link_map={ifo: current_pad},
            )
            # Register for multilink (conditioned output still exposed)
            self._also_expose_pads.append(f"{current_element.name}:src:{current_pad}")
