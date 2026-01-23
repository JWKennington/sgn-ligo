"""A composed source element that generates fake GW noise with injections.

This module provides the InjectedNoiseSource class, a composed source element
that combines GWDataNoiseSource (fake detector noise) with SimInspiralSource
(gravitational wave injections) using an Adder to produce realistic test data.

Example:
    >>> source = InjectedNoiseSource(
    ...     name="test_data",
    ...     ifos=["H1", "L1"],
    ...     t0=1126259460,
    ...     duration=64.0,
    ...     test_mode="bbh",
    ... )
    >>> # Use in pipeline
    >>> pipeline = Pipeline()
    >>> pipeline.connect(source.element, sink)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional

from sgnts.compose import TSCompose, TSComposedSourceElement
from sgnts.transforms import Adder

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.composed_registry import register_composed_source
from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource
from sgnligo.sources.sim_inspiral_source import SimInspiralSource

# GWDataNoiseSource sample rate is fixed by the PSD (16384 Hz)
SAMPLE_RATE = 16384


@register_composed_source
@dataclass
class InjectedNoiseSource(ComposedSourceBase):
    """Composed source generating fake GW noise with injections.

    Combines GWDataNoiseSource (colored detector noise) with SimInspiralSource
    (gravitational wave signals) to produce realistic test data.

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

    Example:
        >>> # With test mode (automatic BBH injections every 30s)
        >>> source = InjectedNoiseSource(
        ...     name="test_data",
        ...     ifos=["H1", "L1"],
        ...     t0=1126259460,
        ...     duration=64.0,
        ...     test_mode="bbh",
        ... )

        >>> # With injection file
        >>> source = InjectedNoiseSource(
        ...     name="injected_data",
        ...     ifos=["H1", "L1"],
        ...     t0=1126259460,
        ...     duration=3600.0,
        ...     injection_file="my_injections.xml",
        ...     f_min=15.0,
        ... )

        >>> # Real-time mode
        >>> source = InjectedNoiseSource(
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
        >>> pipeline.connect(source.element, sink)
        >>> pipeline.run()
    """

    # Required
    ifos: List[str]

    # Time specification (at least one required unless real_time=True)
    t0: Optional[float] = None
    duration: Optional[float] = None
    end: Optional[float] = None

    # Injection source (one required)
    injection_file: Optional[str] = None
    test_mode: Optional[str] = None

    # Optional parameters
    f_min: float = 20.0
    approximant_override: Optional[str] = None
    real_time: bool = False
    verbose: bool = False
    output_channel_pattern: str = "{ifo}:STRAIN"

    # Class metadata
    source_type: ClassVar[str] = "injected-noise"
    description: ClassVar[str] = "Colored noise with GW injections"

    def _validate(self) -> None:
        """Validate injection source and time specification."""
        # Validate injection source specification
        if self.injection_file is None and self.test_mode is None:
            raise ValueError("Must specify either injection_file or test_mode")
        if self.injection_file is not None and self.test_mode is not None:
            raise ValueError("Cannot specify both injection_file and test_mode")

        # Validate time specification (unless real_time mode allows indefinite)
        if not self.real_time and self.duration is None and self.end is None:
            raise ValueError("Must specify either duration or end when real_time=False")

    def _build(self) -> TSComposedSourceElement:
        """Build the composed source element."""
        # Build channel dictionaries for internal elements
        # Noise source uses {ifo}:FAKE-STRAIN
        noise_channel_dict = {ifo: f"{ifo}:FAKE-STRAIN" for ifo in self.ifos}

        # Output channel names from pattern
        output_channels = [
            self.output_channel_pattern.format(ifo=ifo) for ifo in self.ifos
        ]

        # Create the noise source
        noise_source = GWDataNoiseSource(
            name=f"{self.name}_noise",
            channel_dict=noise_channel_dict,
            t0=self.t0,
            duration=self.duration,
            end=self.end,
            real_time=self.real_time,
            verbose=self.verbose,
        )

        # Create the injection source
        inj_source = SimInspiralSource(
            name=f"{self.name}_injections",
            ifos=self.ifos,
            t0=self.t0,
            duration=self.duration,
            end=self.end,
            injection_file=self.injection_file,
            test_mode=self.test_mode,
            sample_rate=SAMPLE_RATE,
            f_min=self.f_min,
            approximant_override=self.approximant_override,
        )

        # Create one Adder per IFO to keep detector outputs separate
        adders = []
        for ifo, out_channel in zip(self.ifos, output_channels):
            noise_pad = f"{ifo}:FAKE-STRAIN"
            inj_pad = f"{ifo}:INJ-STRAIN"

            adder = Adder(
                name=f"{self.name}_adder_{ifo}",
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

        return compose.as_source(name=self.name)
