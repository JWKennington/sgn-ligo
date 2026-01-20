"""Shared memory (devshm) composed source classes.

These sources read low-latency data from shared memory for online
gravitational wave analysis.

Example:
    >>> source = DevShmComposedSource(
    ...     name="low_latency",
    ...     ifos=["H1"],
    ...     channel_dict={"H1": "GDS-CALIB_STRAIN"},
    ...     shared_memory_dict={"H1": "/dev/shm/kafka/H1_O4Replay"},
    ...     state_channel_dict={"H1": "GDS-CALIB_STATE_VECTOR"},
    ...     state_vector_on_dict={"H1": 3},
    ... )
    >>> pipeline.connect(source.element, sink)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict

from sgnts.compose import TSCompose, TSComposedSourceElement

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.cli_mixins import (
    ChannelOptionsMixin,
    DevShmOptionsMixin,
    IfosFromChannelMixin,
    QueueTimeoutOptionsMixin,
    VerboseOptionsMixin,
)
from sgnligo.sources.datasource_v2.composed_registry import register_composed_source
from sgnligo.sources.datasource_v2.sources.utils import add_state_vector_gating
from sgnligo.sources.devshmsrc import DevShmSource


@register_composed_source
@dataclass(kw_only=True)
class DevShmComposedSource(
    ComposedSourceBase,
    IfosFromChannelMixin,
    ChannelOptionsMixin,
    DevShmOptionsMixin,
    QueueTimeoutOptionsMixin,
    VerboseOptionsMixin,
):
    """Shared memory source with state vector gating.

    Reads low-latency strain data from shared memory and applies state
    vector gating to ensure only valid data is processed.

    Fields inherited from mixins:
        ifos: List of detector prefixes (from IfosFromChannelMixin)
        channel_dict: Dict mapping IFO to channel name (from ChannelOptionsMixin)
        shared_memory_dict: Dict mapping IFO to shm path (from DevShmOptionsMixin)
        discont_wait_time: Discontinuity wait time (from DevShmOptionsMixin)
        queue_timeout: Queue timeout (from QueueTimeoutOptionsMixin)
        verbose: Enable verbose output (from VerboseOptionsMixin)

    Additional required fields:
        state_channel_dict: Dict mapping IFO to state vector channel name
        state_vector_on_dict: Dict mapping IFO to bitmask for state vector

    Example:
        >>> source = DevShmComposedSource(
        ...     name="low_latency",
        ...     ifos=["H1"],
        ...     channel_dict={"H1": "GDS-CALIB_STRAIN"},
        ...     shared_memory_dict={"H1": "/dev/shm/kafka/H1_O4Replay"},
        ...     state_channel_dict={"H1": "GDS-CALIB_STATE_VECTOR"},
        ...     state_vector_on_dict={"H1": 3},
        ... )
        >>> pipeline.connect(source.element, sink)
    """

    # Required state vector fields (not in mixin because they're required here)
    state_channel_dict: Dict[str, str]
    state_vector_on_dict: Dict[str, int]

    # Class metadata
    source_type: ClassVar[str] = "devshm"
    description: ClassVar[str] = "Read from shared memory"

    def _validate(self) -> None:
        """Validate parameters."""
        ifos_set = set(self.ifos)

        # Validate channel_dict
        if set(self.channel_dict.keys()) != ifos_set:
            raise ValueError("channel_dict keys must match ifos")

        # Validate shared_memory_dict
        if set(self.shared_memory_dict.keys()) != ifos_set:
            raise ValueError("shared_memory_dict keys must match ifos")

        # Validate state_channel_dict (required for devshm)
        if set(self.state_channel_dict.keys()) != ifos_set:
            raise ValueError("state_channel_dict keys must match ifos")

        # Validate state_vector_on_dict (required for devshm)
        if set(self.state_vector_on_dict.keys()) != ifos_set:
            raise ValueError("state_vector_on_dict keys must match ifos")

    def _build(self) -> TSComposedSourceElement:
        """Build the shared memory source with state vector gating."""
        # Build channel names for DevShmSource
        # DevShmSource expects: {ifo: [strain_channel, state_channel]}
        channel_names = {}
        for ifo in self.ifos:
            strain_channel = f"{ifo}:{self.channel_dict[ifo]}"
            state_channel = f"{ifo}:{self.state_channel_dict[ifo]}"
            channel_names[ifo] = [strain_channel, state_channel]

        # Create the shared memory source
        devshm = DevShmSource(
            name=f"{self.name}_devshm",
            channel_names=channel_names,
            shared_memory_dirs=self.shared_memory_dict,
            discont_wait_time=self.discont_wait_time,
            queue_timeout=self.queue_timeout,
            verbose=self.verbose,
        )

        compose = TSCompose()

        # Add state vector gating for each IFO
        for ifo in self.ifos:
            strain_channel = f"{ifo}:{self.channel_dict[ifo]}"
            state_channel = f"{ifo}:{self.state_channel_dict[ifo]}"

            gate = add_state_vector_gating(
                compose=compose,
                strain_source=devshm,
                state_source=devshm,
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

        return compose.as_source(
            name=self.name,
            also_expose_source_pads=self._also_expose_pads,
        )
