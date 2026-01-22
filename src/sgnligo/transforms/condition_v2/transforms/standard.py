"""Standard conditioning transform with whitening and gating.

This module provides the StandardCondition class for signal conditioning
using the standard (non-zero-latency) whitening path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import ClassVar

from sgnts.compose import TSCompose, TSComposedTransformElement
from sgnts.transforms import Threshold

from sgnligo.sources.datasource_v2.cli_mixins import IfosOnlyMixin
from sgnligo.transforms.condition_v2.cli_mixins import (
    GatingOptionsMixin,
    HighpassFilterOptionsMixin,
    InputSampleRateOptionsMixin,
    LatencyTrackingOptionsMixin,
    PSDOptionsMixin,
    WhitenSampleRateOptionsMixin,
)
from sgnligo.transforms.condition_v2.composed_base import ComposedTransformBase
from sgnligo.transforms.condition_v2.composed_registry import (
    register_composed_transform,
)
from sgnligo.transforms.latency import Latency
from sgnligo.transforms.whiten import Whiten


@register_composed_transform
@dataclass(kw_only=True)
class StandardCondition(
    ComposedTransformBase,
    IfosOnlyMixin,
    InputSampleRateOptionsMixin,
    WhitenSampleRateOptionsMixin,
    PSDOptionsMixin,
    GatingOptionsMixin,
    LatencyTrackingOptionsMixin,
    HighpassFilterOptionsMixin,
):
    """Standard conditioning with Whiten element.

    This transform performs per-IFO signal conditioning:
    1. Whitening via Whiten element (PSD estimation + spectral whitening)
    2. Optional threshold-based gating
    3. Optional latency tracking

    Sink pads (inputs):
        - {ifo}: Raw strain input for each IFO at input_sample_rate

    Source pads (outputs):
        - {ifo}: Conditioned strain for each IFO at whiten_sample_rate
        - spectrum_{ifo}: PSD output for each IFO (metadata with LAL frequency series)
        - {ifo}_latency: (optional) Latency telemetry if whiten_latency enabled

    Example:
        >>> cond = StandardCondition(
        ...     name="condition",
        ...     ifos=["H1", "L1"],
        ...     input_sample_rate=16384,
        ...     whiten_sample_rate=2048,
        ...     psd_fft_length=8,
        ... )
        >>> pipeline = Pipeline()
        >>> pipeline.connect(source.element, cond.element)
        >>> pipeline.connect(cond.element, sink)
    """

    transform_type: ClassVar[str] = "standard"
    description: ClassVar[str] = "Standard whitening with PSD tracking"

    def _validate(self) -> None:
        """Validate configuration."""
        if not self.ifos:
            raise ValueError("Must specify at least one IFO")
        if self.reference_psd is None and not self.track_psd:
            raise ValueError("Must enable track_psd if reference_psd not provided")

    def _build(self) -> TSComposedTransformElement:
        """Build the standard conditioning chain for each IFO."""
        compose = TSCompose()

        for ifo in self.ifos:
            # 1. Whiten element - PSD estimation and whitening
            whiten = Whiten(
                name=f"{self.name}_{ifo}_whiten",
                sink_pad_names=(ifo,),
                instrument=ifo,
                psd_pad_name=f"spectrum_{ifo}",
                whiten_pad_name=ifo,
                input_sample_rate=self.input_sample_rate,
                whiten_sample_rate=self.whiten_sample_rate,
                fft_length=self.psd_fft_length,
                reference_psd=self.reference_psd,
                highpass_filter=self.highpass_filter,
            )
            compose.insert(whiten)

            current_element = whiten
            current_pad = ifo

            # 2. Optional gating (threshold)
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

            # 3. Optional latency tracking
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
                # Register for multilink exposure (strain still goes out as boundary)
                pad_full_name = f"{current_element.name}:src:{current_pad}"
                self._also_expose_pads.append(pad_full_name)

        return compose.as_transform(
            name=self.name,
            also_expose_source_pads=(
                self._also_expose_pads if self._also_expose_pads else None
            ),
        )
