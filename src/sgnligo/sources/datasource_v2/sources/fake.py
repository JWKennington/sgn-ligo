"""Fake signal source classes (white, sin, impulse).

These sources generate synthetic test signals for pipeline development
and testing without requiring real detector data.

Example:
    >>> source = WhiteComposedSource(
    ...     name="noise",
    ...     ifos=["H1", "L1"],
    ...     sample_rate=4096,
    ...     t0=1000,
    ...     end=1010,
    ... )
    >>> pipeline.connect(source.element, sink)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

import igwn_segments as segments
from igwn_ligolw import utils as ligolw_utils
from igwn_ligolw.utils import segments as ligolw_segments
from lal import LIGOTimeGPS
from sgnts.compose import TSCompose, TSComposedSourceElement
from sgnts.sources import FakeSeriesSource, SegmentSource
from sgnts.transforms import Gate

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.cli_mixins import (
    GPSOptionsMixin,
    IfosOnlyMixin,
    ImpulsePositionOptionsMixin,
    SampleRateOptionsMixin,
    SegmentsOptionsMixin,
    VerboseOptionsMixin,
)
from sgnligo.sources.datasource_v2.composed_registry import register_composed_source


@dataclass(kw_only=True)
class FakeSourceBase(
    ComposedSourceBase,
    IfosOnlyMixin,
    SampleRateOptionsMixin,
    GPSOptionsMixin,
    SegmentsOptionsMixin,
    VerboseOptionsMixin,
):
    """Base class for fake signal sources.

    Generates synthetic signals for pipeline testing. Supports optional
    segment-based gating from LIGO XML segment files.

    Fields inherited from mixins:
        ifos: List of detector prefixes (from IfosOnlyMixin)
        sample_rate: Sample rate in Hz (from SampleRateOptionsMixin)
        t0: GPS start time (from GPSOptionsMixin)
        end: GPS end time (from GPSOptionsMixin)
        segments_file: Path to LIGO XML segments file (from SegmentsOptionsMixin)
        segments_name: Name of segments to extract (from SegmentsOptionsMixin)
        verbose: Enable verbose output (from VerboseOptionsMixin)
    """

    # Subclasses override this
    signal_type: ClassVar[str] = "white"

    def _validate(self) -> None:
        """Validate parameters."""
        if self.t0 >= self.end:
            raise ValueError("t0 must be less than end")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        # Validate segments options
        if self.segments_file is not None:
            if self.segments_name is None:
                raise ValueError("Must specify segments_name when segments_file is set")
            if not os.path.exists(self.segments_file):
                raise ValueError(f"Segments file does not exist: {self.segments_file}")

    def _load_segments(self) -> Optional[Dict[str, list]]:
        """Load and process segments from XML file."""
        if self.segments_file is None or self.segments_name is None:
            return None

        loaded_segments = ligolw_segments.segmenttable_get_by_name(
            ligolw_utils.load_filename(
                self.segments_file,
                contenthandler=ligolw_segments.LIGOLWContentHandler,
            ),
            self.segments_name,
        ).coalesce()

        # Clip to requested time range
        seg = segments.segment(LIGOTimeGPS(self.t0), LIGOTimeGPS(self.end))
        clipped_segments = segments.segmentlistdict(
            (ifo, seglist & segments.segmentlist([seg]))
            for ifo, seglist in loaded_segments.items()
        )

        # Convert to nanoseconds
        segments_dict = {}
        for ifo, segs in clipped_segments.items():
            segments_dict[ifo] = [segments.segment(s[0].ns(), s[1].ns()) for s in segs]
        return segments_dict

    def _build(self) -> TSComposedSourceElement:
        """Build the fake signal source."""
        compose = TSCompose()
        segments_dict = self._load_segments()

        for ifo in self.ifos:
            pad_name = ifo

            source = FakeSeriesSource(
                name=f"{self.name}_{ifo}",
                source_pad_names=(pad_name,),
                rate=self.sample_rate,
                signal_type=self.signal_type,
                impulse_position=getattr(self, "impulse_position", -1),
                real_time=False,
                t0=self.t0,
                end=self.end,
            )

            # Track the final output element and pad for latency tracking
            final_source = source
            final_pad = pad_name

            # Add segment gating if configured
            if segments_dict is not None and ifo in segments_dict:
                ifo_segments = segments_dict[ifo]

                if ifo_segments:
                    seg_source = SegmentSource(
                        name=f"{self.name}_{ifo}_seg",
                        source_pad_names=("control",),
                        rate=self.sample_rate,
                        t0=self.t0,
                        end=self.end,
                        segments=ifo_segments,
                    )

                    gate = Gate(
                        name=f"{self.name}_{ifo}_gate",
                        sink_pad_names=("strain", "control"),
                        control="control",
                        source_pad_names=(ifo,),
                    )

                    compose.connect(source, gate, link_map={"strain": pad_name})
                    compose.connect(seg_source, gate, link_map={"control": "control"})

                    final_source = gate
                    final_pad = ifo

                    if self.verbose:
                        print(f"Added segment gating for {ifo}")
                else:
                    compose.insert(source)
            else:
                compose.insert(source)

            # Add latency tracking if configured
            self._add_latency_tracking(compose, ifo, final_source, final_pad)

        return compose.as_source(
            name=self.name,
            also_expose_source_pads=self._also_expose_pads,
        )


@register_composed_source
@dataclass(kw_only=True)
class WhiteComposedSource(FakeSourceBase):
    """Gaussian white noise source.

    Generates uncorrelated Gaussian white noise for each IFO channel.
    Useful for basic pipeline testing where spectral characteristics
    don't matter.

    Example:
        >>> source = WhiteComposedSource(
        ...     name="noise",
        ...     ifos=["H1", "L1"],
        ...     sample_rate=4096,
        ...     t0=1000,
        ...     end=1010,
        ... )
    """

    source_type: ClassVar[str] = "white"
    description: ClassVar[str] = "Gaussian white noise"
    signal_type: ClassVar[str] = "white"


@register_composed_source
@dataclass(kw_only=True)
class SinComposedSource(FakeSourceBase):
    """Sinusoidal test signal source.

    Generates a sinusoidal signal for each IFO channel.
    Useful for testing frequency-domain processing.

    Example:
        >>> source = SinComposedSource(
        ...     name="sine",
        ...     ifos=["H1"],
        ...     sample_rate=4096,
        ...     t0=1000,
        ...     end=1010,
        ... )
    """

    source_type: ClassVar[str] = "sin"
    description: ClassVar[str] = "Sinusoidal test signal"
    signal_type: ClassVar[str] = "sin"


@register_composed_source
@dataclass(kw_only=True)
class ImpulseComposedSource(FakeSourceBase, ImpulsePositionOptionsMixin):
    """Impulse test signal source.

    Generates an impulse signal (single spike) for each IFO channel.
    Useful for testing impulse response.

    Fields inherited from mixins:
        impulse_position: Sample index for impulse (-1 for random)

    Example:
        >>> source = ImpulseComposedSource(
        ...     name="impulse",
        ...     ifos=["H1"],
        ...     sample_rate=4096,
        ...     t0=1000,
        ...     end=1010,
        ...     impulse_position=100,
        ... )
    """

    source_type: ClassVar[str] = "impulse"
    description: ClassVar[str] = "Impulse test signal"
    signal_type: ClassVar[str] = "impulse"


# --- Real-time variants ---


@dataclass(kw_only=True)
class RealtimeFakeSourceBase(
    ComposedSourceBase,
    IfosOnlyMixin,
    SampleRateOptionsMixin,
    VerboseOptionsMixin,
):
    """Base class for real-time fake signal sources.

    Real-time sources generate data synchronized with wall clock time
    and don't require an end time.

    Fields inherited from mixins:
        ifos: List of detector prefixes (from IfosOnlyMixin)
        sample_rate: Sample rate in Hz (from SampleRateOptionsMixin)
        verbose: Enable verbose output (from VerboseOptionsMixin)

    Additional fields:
        t0: GPS start time (default: 0, syncs with current time)
    """

    # Optional t0 with default (not from mixin since default is 0, not None)
    t0: float = 0

    signal_type: ClassVar[str] = "white"

    def _validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    def _build(self) -> TSComposedSourceElement:
        """Build the real-time fake signal source."""
        compose = TSCompose()

        for ifo in self.ifos:
            pad_name = ifo

            source = FakeSeriesSource(
                name=f"{self.name}_{ifo}",
                source_pad_names=(pad_name,),
                rate=self.sample_rate,
                signal_type=self.signal_type,
                impulse_position=getattr(self, "impulse_position", -1),
                real_time=True,
                t0=self.t0,
                end=None,
            )
            compose.insert(source)

            # Add latency tracking if configured
            self._add_latency_tracking(compose, ifo, source, pad_name)

        return compose.as_source(
            name=self.name,
            also_expose_source_pads=self._also_expose_pads,
        )


@register_composed_source
@dataclass(kw_only=True)
class WhiteRealtimeComposedSource(RealtimeFakeSourceBase):
    """Real-time Gaussian white noise source.

    Generates white noise synchronized with wall clock time.

    Example:
        >>> source = WhiteRealtimeComposedSource(
        ...     name="realtime_noise",
        ...     ifos=["H1"],
        ...     sample_rate=4096,
        ... )
    """

    source_type: ClassVar[str] = "white-realtime"
    description: ClassVar[str] = "Real-time Gaussian white noise"
    signal_type: ClassVar[str] = "white"


@register_composed_source
@dataclass(kw_only=True)
class SinRealtimeComposedSource(RealtimeFakeSourceBase):
    """Real-time sinusoidal test signal source.

    Generates sinusoidal signal synchronized with wall clock time.

    Example:
        >>> source = SinRealtimeComposedSource(
        ...     name="realtime_sin",
        ...     ifos=["H1"],
        ...     sample_rate=4096,
        ... )
    """

    source_type: ClassVar[str] = "sin-realtime"
    description: ClassVar[str] = "Real-time sinusoidal test signal"
    signal_type: ClassVar[str] = "sin"


@register_composed_source
@dataclass(kw_only=True)
class ImpulseRealtimeComposedSource(
    RealtimeFakeSourceBase, ImpulsePositionOptionsMixin
):
    """Real-time impulse test signal source.

    Generates impulse signals synchronized with wall clock time.

    Fields inherited from mixins:
        impulse_position: Sample index for impulse (-1 for random)

    Example:
        >>> source = ImpulseRealtimeComposedSource(
        ...     name="realtime_impulse",
        ...     ifos=["H1"],
        ...     sample_rate=4096,
        ...     impulse_position=100,
        ... )
    """

    source_type: ClassVar[str] = "impulse-realtime"
    description: ClassVar[str] = "Real-time impulse test signal"
    signal_type: ClassVar[str] = "impulse"
