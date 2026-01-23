"""Utility functions for composed sources.

This module contains reusable building blocks that are shared across
multiple composed source classes.
"""

from __future__ import annotations

from sgnts.compose import TSCompose
from sgnts.transforms import Gate

from sgnligo.transforms import BitMask


def add_state_vector_gating(
    compose: TSCompose,
    strain_source,
    state_source,
    ifo: str,
    bit_mask: int,
    strain_pad: str,
    state_pad: str,
    output_pad: str,
) -> Gate:
    """Add BitMask + Gate to a compose for state vector gating.

    This is the common pattern used by devshm, arrakis, and gwdata-noise sources.
    It applies a bitmask to the state vector channel, then uses a Gate to
    control the strain data based on the masked state vector.

    The pattern is:
        strain_source[strain_pad] ─────────────────┐
                                                   ├─> Gate[output_pad]
        state_source[state_pad] -> BitMask[state] ─┘

    Args:
        compose: TSCompose to add elements to (modified in-place)
        strain_source: Source element providing strain data
        state_source: Source element providing state vector data
        ifo: Interferometer prefix (e.g., "H1")
        bit_mask: Bitmask to apply to state vector
        strain_pad: Name of the strain output pad on strain_source
        state_pad: Name of the state vector output pad on state_source
        output_pad: Name for the gated output pad

    Returns:
        The Gate element for downstream use (e.g., latency tracking)
    """
    # Create BitMask to filter state vector
    mask = BitMask(
        name=f"{ifo}_Mask",
        sink_pad_names=("state",),
        source_pad_names=("state",),
        bit_mask=bit_mask,
    )

    # Create Gate to control strain based on masked state vector
    gate = Gate(
        name=f"{ifo}_Gate",
        sink_pad_names=("strain", "state_vector"),
        control="state_vector",
        source_pad_names=(output_pad,),
    )

    # Connect state_source -> BitMask
    compose.connect(
        state_source,
        mask,
        link_map={"state": state_pad},
    )

    # Connect BitMask -> Gate.state_vector
    compose.connect(
        mask,
        gate,
        link_map={"state_vector": "state"},
    )

    # Connect strain_source -> Gate.strain
    compose.connect(
        strain_source,
        gate,
        link_map={"strain": strain_pad},
    )

    return gate
