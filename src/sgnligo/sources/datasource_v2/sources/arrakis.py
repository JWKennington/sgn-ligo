"""Arrakis composed source classes.

These sources read streaming data from for online
gravitational wave analysis.

Example:
    >>> source = ArrakisComposedSource(
    ...     name="kafka_data",
    ...     ifos=["H1", "L1"],
    ...     channel_dict={"H1": "GDS-CALIB_STRAIN", "L1": "GDS-CALIB_STRAIN"},
    ... )
    >>> pipeline.connect(source.element, sink)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from sgn_arrakis.source import ArrakisSource
from sgnts.compose import TSCompose, TSComposedSourceElement

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.cli_mixins import (
    ChannelOptionsMixin,
    GPSOptionsOptionalMixin,
    IfosFromChannelMixin,
    QueueTimeoutOptionsMixin,
    StateVectorOptionsMixin,
    VerboseOptionsMixin,
)
from sgnligo.sources.datasource_v2.composed_registry import register_composed_source
from sgnligo.sources.datasource_v2.sources.utils import add_state_vector_gating


@register_composed_source
@dataclass(kw_only=True)
class ArrakisComposedSource(
    ComposedSourceBase,
    IfosFromChannelMixin,
    ChannelOptionsMixin,
    GPSOptionsOptionalMixin,
    QueueTimeoutOptionsMixin,
    StateVectorOptionsMixin,
    VerboseOptionsMixin,
):
    """Arrakis source for streaming data.

    Reads streaming gravitational wave data from topics.
    Optionally supports state vector gating.

    Fields inherited from mixins:
        ifos: List of detector prefixes (from IfosFromChannelMixin)
        channel_dict: Dict mapping IFO to channel name (from ChannelOptionsMixin)
        t0: GPS start time (optional, from GPSOptionsOptionalMixin)
        end: GPS end time (optional, from GPSOptionsOptionalMixin)
        queue_timeout: Queue timeout (from QueueTimeoutOptionsMixin)
        state_channel_dict: State channel dict (from StateVectorOptionsMixin)
        state_vector_on_dict: Bitmask dict (from StateVectorOptionsMixin)
        state_segments_file: State segments file (from StateVectorOptionsMixin)
        state_sample_rate: State vector sample rate (from StateVectorOptionsMixin)
        verbose: Enable verbose output (from VerboseOptionsMixin)

    Example:
        >>> source = ArrakisComposedSource(
        ...     name="kafka_data",
        ...     ifos=["H1"],
        ...     channel_dict={"H1": "GDS-CALIB_STRAIN"},
        ... )
        >>> pipeline.connect(source.element, sink)
    """

    # Class metadata
    source_type: ClassVar[str] = "arrakis"
    description: ClassVar[str] = "Read from Arrakis"

    def _validate(self) -> None:
        """Validate parameters."""
        ifos_set = set(self.ifos)

        # Validate channel_dict
        if set(self.channel_dict.keys()) != ifos_set:
            raise ValueError("channel_dict keys must match ifos")

        # Validate time range if both provided
        if self.t0 is not None and self.end is not None and self.t0 >= self.end:
            raise ValueError("t0 must be less than end")

        # Validate state vector options
        if self.state_channel_dict is not None:
            if set(self.state_channel_dict.keys()) != ifos_set:
                raise ValueError("state_channel_dict keys must match ifos")
            if self.state_vector_on_dict is None:
                raise ValueError(
                    "Must specify state_vector_on_dict when state_channel_dict is set"
                )

        if self.state_vector_on_dict is not None:
            if set(self.state_vector_on_dict.keys()) != ifos_set:
                raise ValueError("state_vector_on_dict keys must match ifos")
            if self.state_channel_dict is None:
                raise ValueError(
                    "Must specify state_channel_dict when state_vector_on_dict is set"
                )

    def _build(self) -> TSComposedSourceElement:
        """Build the Arrakis source."""
        # Check if state vector gating is enabled
        use_state_vector = (
            self.state_channel_dict is not None
            and self.state_vector_on_dict is not None
        )

        # Build channel names list for ArrakisSource
        channel_names = []
        for ifo in self.ifos:
            strain_channel = f"{ifo}:{self.channel_dict[ifo]}"
            channel_names.append(strain_channel)

            if use_state_vector:
                assert self.state_channel_dict is not None  # for type checker
                state_channel = f"{ifo}:{self.state_channel_dict[ifo]}"
                channel_names.append(state_channel)

        # Calculate duration if both times provided
        duration = None
        if self.t0 is not None and self.end is not None:
            duration = self.end - self.t0

        # Create the Arrakis source
        arrakis = ArrakisSource(
            name=f"{self.name}_arrakis",
            source_pad_names=channel_names,
            start_time=self.t0,
            duration=duration,
            in_queue_timeout=int(self.queue_timeout),
        )

        compose = TSCompose()

        if use_state_vector:
            # Add state vector gating for each IFO
            assert self.state_channel_dict is not None  # for type checker
            assert self.state_vector_on_dict is not None  # for type checker
            for ifo in self.ifos:
                strain_channel = f"{ifo}:{self.channel_dict[ifo]}"
                state_channel = f"{ifo}:{self.state_channel_dict[ifo]}"

                gate = add_state_vector_gating(
                    compose=compose,
                    strain_source=arrakis,
                    state_source=arrakis,
                    ifo=ifo,
                    bit_mask=self.state_vector_on_dict[ifo],
                    strain_pad=strain_channel,
                    state_pad=state_channel,
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
            # No gating - just expose Arrakis source directly
            compose.insert(arrakis)

            # Add latency tracking for each IFO
            for ifo in self.ifos:
                strain_channel = f"{ifo}:{self.channel_dict[ifo]}"
                self._add_latency_tracking(compose, ifo, arrakis, strain_channel)

        return compose.as_source(
            name=self.name,
            also_expose_source_pads=self._also_expose_pads,
        )
