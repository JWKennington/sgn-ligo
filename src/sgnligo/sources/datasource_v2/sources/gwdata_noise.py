"""GWData noise composed source classes.

These sources generate colored Gaussian noise with realistic LIGO PSDs,
suitable for testing and development without real detector data.

Example:
    >>> source = GWDataNoiseComposedSource(
    ...     name="noise",
    ...     ifos=["H1", "L1"],
    ...     t0=1000,
    ...     end=1010,
    ... )
    >>> pipeline.connect(source.element, sink)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np
from sgnts.compose import TSCompose, TSComposedSourceElement
from sgnts.sources import SegmentSource

from sgnligo.base import read_segments_and_values_from_file
from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.cli_mixins import (
    ChannelPatternOptionsMixin,
    GPSOptionsMixin,
    GPSOptionsOptionalMixin,
    IfosOnlyMixin,
    StateVectorOnDictOnlyMixin,
    VerboseOptionsMixin,
)
from sgnligo.sources.datasource_v2.composed_registry import register_composed_source
from sgnligo.sources.datasource_v2.sources.utils import add_state_vector_gating
from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource


@register_composed_source
@dataclass(kw_only=True)
class GWDataNoiseComposedSource(
    ComposedSourceBase,
    IfosOnlyMixin,
    GPSOptionsMixin,
    ChannelPatternOptionsMixin,
    StateVectorOnDictOnlyMixin,
    VerboseOptionsMixin,
):
    """Colored Gaussian noise source with optional state vector gating.

    Generates colored Gaussian noise with LIGO PSD for offline analysis.
    Supports optional segment-based state vector gating.

    Fields inherited from mixins:
        ifos: List of detector prefixes (from IfosOnlyMixin)
        t0: GPS start time (from GPSOptionsMixin)
        end: GPS end time (from GPSOptionsMixin)
        channel_pattern: Channel naming pattern (from ChannelPatternOptionsMixin)
        state_vector_on_dict: Bitmask dict (from StateVectorOnDictOnlyMixin)
        state_segments_file: State segments file (from StateVectorOnDictOnlyMixin)
        state_sample_rate: State vector sample rate (from StateVectorOnDictOnlyMixin)
        verbose: Enable verbose output (from VerboseOptionsMixin)

    Example:
        >>> source = GWDataNoiseComposedSource(
        ...     name="noise",
        ...     ifos=["H1", "L1"],
        ...     t0=1000,
        ...     end=1010,
        ... )
        >>> pipeline.connect(source.element, sink)
    """

    # Class metadata
    source_type: ClassVar[str] = "gwdata-noise"
    description: ClassVar[str] = "Colored Gaussian noise with LIGO PSD"

    def _validate(self) -> None:
        """Validate parameters."""
        if self.t0 >= self.end:
            raise ValueError("t0 must be less than end")

        # Validate state segments file
        if self.state_segments_file is not None:
            if not os.path.exists(self.state_segments_file):
                raise ValueError(
                    f"State segments file does not exist: {self.state_segments_file}"
                )

        # Validate state_vector_on_dict
        if self.state_vector_on_dict is not None:
            if set(self.state_vector_on_dict.keys()) != set(self.ifos):
                raise ValueError("state_vector_on_dict keys must match ifos")

    def _build_channel_dict(self) -> Dict[str, str]:
        """Build channel dict from pattern."""
        return {ifo: self.channel_pattern.format(ifo=ifo) for ifo in self.ifos}

    def _load_state_segments(
        self,
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Load state segments from file or create defaults."""
        if self.state_vector_on_dict is None:
            return None, None

        if self.state_segments_file is not None:
            state_segments, state_values = read_segments_and_values_from_file(
                self.state_segments_file, self.verbose
            )
        else:
            # Default: single segment covering entire time range with value 3
            start_ns = int(self.t0 * 1e9)
            end_ns = int(self.end * 1e9)
            state_segments = ((start_ns, end_ns),)
            state_values = (3,)  # Default: bits 0 and 1 set
            if self.verbose:
                print("Using default state segments: single segment with value 3")

        return state_segments, state_values

    def _build(self) -> TSComposedSourceElement:
        """Build the GWData noise source."""
        channel_dict = self._build_channel_dict()

        # Create the noise source
        noise_source = GWDataNoiseSource(
            name=f"{self.name}_noise",
            channel_dict=channel_dict,
            t0=self.t0,
            end=self.end,
            real_time=False,
            verbose=self.verbose,
        )

        compose = TSCompose()

        # Check if we need state vector gating
        if self.state_vector_on_dict is not None:
            state_segments, state_values = self._load_state_segments()
            assert state_segments is not None
            assert state_values is not None

            for ifo in self.ifos:
                strain_channel = channel_dict[ifo]

                # Create segment source for state vector
                state_source = SegmentSource(
                    name=f"{self.name}_{ifo}_state",
                    source_pad_names=("state",),
                    rate=self.state_sample_rate,
                    t0=self.t0,
                    end=self.end,
                    segments=state_segments,
                    values=state_values,
                )

                gate = add_state_vector_gating(
                    compose=compose,
                    strain_source=noise_source,
                    state_source=state_source,
                    ifo=ifo,
                    bit_mask=self.state_vector_on_dict[ifo],
                    strain_pad=strain_channel,
                    state_pad="state",
                    output_pad=ifo,
                )

                # Add latency tracking if configured
                self._add_latency_tracking(compose, ifo, gate, ifo)

                if self.verbose:
                    print(
                        f"Added state vector gating for {ifo} with mask "
                        f"{self.state_vector_on_dict[ifo]}"
                    )
        else:
            # No gating - just expose noise source directly
            compose.insert(noise_source)

            # Add latency tracking for each IFO
            for ifo in self.ifos:
                strain_channel = channel_dict[ifo]
                self._add_latency_tracking(compose, ifo, noise_source, strain_channel)

        return compose.as_source(
            name=self.name,
            also_expose_source_pads=self._also_expose_pads,
        )


@register_composed_source
@dataclass(kw_only=True)
class GWDataNoiseRealtimeComposedSource(
    ComposedSourceBase,
    IfosOnlyMixin,
    GPSOptionsOptionalMixin,
    ChannelPatternOptionsMixin,
    StateVectorOnDictOnlyMixin,
    VerboseOptionsMixin,
):
    """Real-time colored Gaussian noise source.

    Generates colored Gaussian noise synchronized with wall clock time.
    Time range is optional for indefinite operation.

    Fields inherited from mixins:
        ifos: List of detector prefixes (from IfosOnlyMixin)
        t0: GPS start time (optional, from GPSOptionsOptionalMixin)
        end: GPS end time (optional, from GPSOptionsOptionalMixin)
        channel_pattern: Channel naming pattern (from ChannelPatternOptionsMixin)
        state_vector_on_dict: Bitmask dict (from StateVectorOnDictOnlyMixin)
        state_segments_file: State segments file (from StateVectorOnDictOnlyMixin)
        state_sample_rate: State vector sample rate (from StateVectorOnDictOnlyMixin)
        verbose: Enable verbose output (from VerboseOptionsMixin)

    Example:
        >>> source = GWDataNoiseRealtimeComposedSource(
        ...     name="realtime_noise",
        ...     ifos=["H1"],
        ... )
        >>> pipeline.connect(source.element, sink)
    """

    # Class metadata
    source_type: ClassVar[str] = "gwdata-noise-realtime"
    description: ClassVar[str] = "Real-time colored Gaussian noise with LIGO PSD"

    def _validate(self) -> None:
        """Validate parameters."""
        if self.t0 is not None and self.end is not None and self.t0 >= self.end:
            raise ValueError("t0 must be less than end")

        # Validate state segments file
        if self.state_segments_file is not None:
            if not os.path.exists(self.state_segments_file):
                raise ValueError(
                    f"State segments file does not exist: {self.state_segments_file}"
                )

        # Validate state_vector_on_dict
        if self.state_vector_on_dict is not None:
            if set(self.state_vector_on_dict.keys()) != set(self.ifos):
                raise ValueError("state_vector_on_dict keys must match ifos")

    def _build_channel_dict(self) -> Dict[str, str]:
        """Build channel dict from pattern."""
        return {ifo: self.channel_pattern.format(ifo=ifo) for ifo in self.ifos}

    def _load_state_segments(
        self,
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Load state segments from file or create defaults."""
        if self.state_vector_on_dict is None:
            return None, None

        if self.state_segments_file is not None:
            state_segments, state_values = read_segments_and_values_from_file(
                self.state_segments_file, self.verbose
            )
        else:
            # Default: single segment covering entire time range with value 3
            if self.t0 is not None:
                start_ns = int(self.t0 * 1e9)
                if self.end is not None:
                    end_ns = int(self.end * 1e9)
                else:
                    # For real-time mode without end time
                    end_ns = int(np.iinfo(np.int32).max * 1e9)
                state_segments = ((start_ns, end_ns),)
                state_values = (3,)  # Default: bits 0 and 1 set
                if self.verbose:
                    print("Using default state segments: single segment with value 3")
            else:
                raise ValueError(
                    "Must provide either state_segments_file or t0 "
                    "when using state vector gating"
                )

        return state_segments, state_values

    def _build(self) -> TSComposedSourceElement:
        """Build the real-time GWData noise source."""
        channel_dict = self._build_channel_dict()

        # Create the noise source
        noise_source = GWDataNoiseSource(
            name=f"{self.name}_noise",
            channel_dict=channel_dict,
            t0=self.t0,
            end=self.end,
            real_time=True,
            verbose=self.verbose,
        )

        compose = TSCompose()

        # Check if we need state vector gating
        if self.state_vector_on_dict is not None:
            state_segments, state_values = self._load_state_segments()
            assert state_segments is not None
            assert state_values is not None

            # Determine end time for SegmentSource (doesn't support None)
            seg_end = (
                self.end if self.end is not None else float(np.iinfo(np.int32).max)
            )

            for ifo in self.ifos:
                strain_channel = channel_dict[ifo]

                # Create segment source for state vector
                state_source = SegmentSource(
                    name=f"{self.name}_{ifo}_state",
                    source_pad_names=("state",),
                    rate=self.state_sample_rate,
                    t0=self.t0,
                    end=seg_end,
                    segments=state_segments,
                    values=state_values,
                )

                gate = add_state_vector_gating(
                    compose=compose,
                    strain_source=noise_source,
                    state_source=state_source,
                    ifo=ifo,
                    bit_mask=self.state_vector_on_dict[ifo],
                    strain_pad=strain_channel,
                    state_pad="state",
                    output_pad=ifo,
                )

                # Add latency tracking if configured
                self._add_latency_tracking(compose, ifo, gate, ifo)

                if self.verbose:
                    print(
                        f"Added state vector gating for {ifo} with mask "
                        f"{self.state_vector_on_dict[ifo]}"
                    )
        else:
            # No gating - just expose noise source directly
            compose.insert(noise_source)

            # Add latency tracking for each IFO
            for ifo in self.ifos:
                strain_channel = channel_dict[ifo]
                self._add_latency_tracking(compose, ifo, noise_source, strain_channel)

        return compose.as_source(
            name=self.name,
            also_expose_source_pads=self._also_expose_pads,
        )
