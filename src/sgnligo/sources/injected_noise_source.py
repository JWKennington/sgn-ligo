"""A composed source element that generates fake GW noise with injections.

This module provides a factory function to create a composed source element
that combines GWDataNoiseSource (fake detector noise) with SimInspiralSource
(gravitational wave injections) using an Adder to produce realistic test data.
"""

from __future__ import annotations

from typing import List, Optional

from sgnts.compose import TSCompose, TSComposedSourceElement
from sgnts.transforms import Adder

from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource
from sgnligo.sources.sim_inspiral_source import SimInspiralSource

# GWDataNoiseSource sample rate is fixed by the PSD (16384 Hz)
SAMPLE_RATE = 16384


def create_injected_noise_source(
    name: str,
    ifos: List[str],
    t0: Optional[float] = None,
    duration: Optional[float] = None,
    end: Optional[float] = None,
    injection_file: Optional[str] = None,
    test_mode: Optional[str] = None,
    f_min: float = 20.0,
    approximant_override: Optional[str] = None,
    real_time: bool = False,
    verbose: bool = False,
    output_channel_pattern: str = "{ifo}:STRAIN",
) -> TSComposedSourceElement:
    """Create a composed source that generates fake GW noise with injections.

    This factory function creates a composed source element combining:
    - GWDataNoiseSource: generates realistic colored noise with Advanced LIGO PSD
    - SimInspiralSource: generates gravitational wave injection signals
    - Adder (one per IFO): adds noise + injections together

    The result is a single source element that outputs fake GW data with
    test injections embedded, suitable for pipeline testing and development.

    Args:
        name: Name of the composed element
        ifos: List of detector prefixes (e.g., ["H1", "L1", "V1"])
        t0: GPS start time. If None and real_time=True, uses current GPS time.
        duration: Duration in seconds (mutually exclusive with end)
        end: GPS end time (mutually exclusive with duration). Can be None
            only when real_time=True for indefinite operation.
        injection_file: Path to injection file (XML or HDF5). Mutually
            exclusive with test_mode.
        test_mode: Test mode for auto-generated injections: "bns", "nsbh",
            or "bbh". Generates periodic test injections every 30 seconds.
            Mutually exclusive with injection_file.
        f_min: Minimum frequency for waveform generation in Hz (default: 20.0)
        approximant_override: Override waveform approximant for all injections
        real_time: If True, generate data synchronized with wall clock time.
            When t0 is None, syncs with actual GPS time.
        verbose: If True, print debug information from internal elements
        output_channel_pattern: Pattern for output pad names. Use {ifo} as
            placeholder for detector prefix. Default: "{ifo}:STRAIN"

    Returns:
        TSComposedSourceElement with one output pad per IFO, named according
        to output_channel_pattern (e.g., "H1:STRAIN", "L1:STRAIN")

    Raises:
        ValueError: If neither injection_file nor test_mode is specified,
            or if both are specified, or if neither duration nor end is
            specified when real_time=False.

    Example:
        >>> # With test mode (automatic BBH injections every 30s)
        >>> source = create_injected_noise_source(
        ...     name="test_data",
        ...     ifos=["H1", "L1"],
        ...     t0=1126259460,
        ...     duration=64.0,
        ...     test_mode="bbh",
        ... )

        >>> # With injection file
        >>> source = create_injected_noise_source(
        ...     name="injected_data",
        ...     ifos=["H1", "L1"],
        ...     t0=1126259460,
        ...     duration=3600.0,
        ...     injection_file="my_injections.xml",
        ...     f_min=15.0,
        ... )

        >>> # Real-time mode
        >>> source = create_injected_noise_source(
        ...     name="realtime_data",
        ...     ifos=["H1"],
        ...     real_time=True,
        ...     test_mode="bns",
        ...     verbose=True,
        ... )

        >>> # Use in pipeline
        >>> from sgn.apps import Pipeline
        >>> from sgn.sinks import CollectSink
        >>> pipeline = Pipeline()
        >>> sink = CollectSink(name="sink", sink_pad_names=["H1:STRAIN"])
        >>> pipeline.connect(source, sink)
        >>> pipeline.run()
    """
    # Validate injection source specification
    if injection_file is None and test_mode is None:
        raise ValueError("Must specify either injection_file or test_mode")
    if injection_file is not None and test_mode is not None:
        raise ValueError("Cannot specify both injection_file and test_mode")

    # Validate time specification (unless real_time mode allows indefinite)
    if not real_time and duration is None and end is None:
        raise ValueError("Must specify either duration or end when real_time=False")

    # Build channel dictionaries for internal elements
    # Noise source uses {ifo}:FAKE-STRAIN
    noise_channel_dict = {ifo: f"{ifo}:FAKE-STRAIN" for ifo in ifos}

    # Output channel names from pattern
    output_channels = [output_channel_pattern.format(ifo=ifo) for ifo in ifos]

    # Create the noise source
    noise_source = GWDataNoiseSource(
        name=f"{name}_noise",
        channel_dict=noise_channel_dict,
        t0=t0,
        duration=duration,
        end=end,
        real_time=real_time,
        verbose=verbose,
    )

    # Create the injection source
    inj_source = SimInspiralSource(
        name=f"{name}_injections",
        ifos=ifos,
        t0=t0,
        duration=duration,
        end=end,
        injection_file=injection_file,
        test_mode=test_mode,
        sample_rate=SAMPLE_RATE,
        f_min=f_min,
        approximant_override=approximant_override,
    )

    # Create one Adder per IFO to keep detector outputs separate
    adders = []
    for ifo, out_channel in zip(ifos, output_channels):
        noise_pad = f"{ifo}:FAKE-STRAIN"
        inj_pad = f"{ifo}:INJ-STRAIN"

        adder = Adder(
            name=f"{name}_adder_{ifo}",
            sink_pad_names=(noise_pad, inj_pad),
            source_pad_names=(out_channel,),
        )
        adders.append(adder)

    # Build the composed element using TSCompose
    compose = TSCompose()

    # Connect noise source and injection source to each adder
    # Implicit linking works because pad names match (e.g., H1:FAKE-STRAIN)
    for adder in adders:
        compose.connect(noise_source, adder)
        compose.connect(inj_source, adder)

    return compose.as_source(name=name)
