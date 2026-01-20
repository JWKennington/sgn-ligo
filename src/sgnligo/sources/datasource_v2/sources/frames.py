"""Frame file composed source classes.

These sources read gravitational wave data from GWF frame files,
the standard format for LIGO/Virgo data.

Example:
    >>> source = FramesComposedSource(
    ...     name="data",
    ...     ifos=["H1", "L1"],
    ...     frame_cache="/path/to/frames.cache",
    ...     channel_dict={"H1": "GDS-CALIB_STRAIN", "L1": "GDS-CALIB_STRAIN"},
    ...     t0=1000000000,
    ...     end=1000000100,
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
from sgnts.sources import SegmentSource
from sgnts.transforms import Adder, Gate

from sgnligo.sources.composed_base import ComposedSourceBase
from sgnligo.sources.datasource_v2.cli_mixins import (
    ChannelOptionsMixin,
    FrameCacheOptionsMixin,
    GPSOptionsMixin,
    IfosFromChannelMixin,
    InjectionOptionsMixin,
    SegmentsOptionsMixin,
    VerboseOptionsMixin,
)
from sgnligo.sources.datasource_v2.composed_registry import register_composed_source
from sgnligo.sources.framecachesrc import FrameReader


@register_composed_source
@dataclass(kw_only=True)
class FramesComposedSource(
    ComposedSourceBase,
    IfosFromChannelMixin,
    FrameCacheOptionsMixin,
    ChannelOptionsMixin,
    GPSOptionsMixin,
    SegmentsOptionsMixin,
    InjectionOptionsMixin,
    VerboseOptionsMixin,
):
    """Frame file source for offline analysis.

    Reads strain data from GWF frame files specified in a LAL cache file.
    Supports optional noiseless injections and segment-based gating.

    Fields inherited from mixins:
        ifos: List of detector prefixes (from IfosFromChannelMixin)
        frame_cache: Path to LAL cache file (from FrameCacheOptionsMixin)
        channel_dict: Dict mapping IFO to channel name (from ChannelOptionsMixin)
        t0: GPS start time (from GPSOptionsMixin)
        end: GPS end time (from GPSOptionsMixin)
        segments_file: Path to LIGO XML segments file (from SegmentsOptionsMixin)
        segments_name: Segment name in XML (from SegmentsOptionsMixin)
        noiseless_inj_frame_cache: Injection frame cache (from InjectionOptionsMixin)
        noiseless_inj_channel_dict: Injection channels (from InjectionOptionsMixin)
        verbose: Enable verbose output (from VerboseOptionsMixin)

    Example:
        >>> source = FramesComposedSource(
        ...     name="data",
        ...     ifos=["H1"],
        ...     frame_cache="/path/to/frames.cache",
        ...     channel_dict={"H1": "GDS-CALIB_STRAIN"},
        ...     t0=1000000000,
        ...     end=1000000100,
        ... )
        >>> pipeline.connect(source.element, sink)
    """

    # Class metadata
    source_type: ClassVar[str] = "frames"
    description: ClassVar[str] = "Read from GWF frame files"

    def _validate(self) -> None:
        """Validate parameters."""
        if self.t0 >= self.end:
            raise ValueError("t0 must be less than end")

        # Validate frame cache
        if not os.path.exists(self.frame_cache):
            raise ValueError(f"Frame cache file does not exist: {self.frame_cache}")

        # Validate channel_dict
        if set(self.channel_dict.keys()) != set(self.ifos):
            raise ValueError("channel_dict keys must match ifos")

        # Validate segments options
        if self.segments_file is not None:
            if self.segments_name is None:
                raise ValueError("Must specify segments_name when segments_file is set")
            if not os.path.exists(self.segments_file):
                raise ValueError(f"Segments file does not exist: {self.segments_file}")

        # Validate injection options
        if self.noiseless_inj_frame_cache is not None:
            if not os.path.exists(self.noiseless_inj_frame_cache):
                raise ValueError(
                    f"Injection frame cache does not exist: "
                    f"{self.noiseless_inj_frame_cache}"
                )
            if self.noiseless_inj_channel_dict is None:
                raise ValueError(
                    "Must specify noiseless_inj_channel_dict when "
                    "noiseless_inj_frame_cache is set"
                )

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
        """Build the frame file source."""
        compose = TSCompose()
        segments_dict = self._load_segments()

        # Determine sample rate from first frame reader (will be set after creation)
        sample_rate = None

        for ifo in self.ifos:
            channel_name = f"{ifo}:{self.channel_dict[ifo]}"

            # Create main frame reader
            frame_reader = FrameReader(
                name=f"{self.name}_{ifo}_frames",
                framecache=self.frame_cache,
                channel_names=[channel_name],
                instrument=ifo,
                t0=self.t0,
                end=self.end,
            )

            # Get sample rate from first frame reader
            if sample_rate is None:
                sample_rate = next(iter(frame_reader.rates.values()))

            # Track the current output element and pad for this IFO
            current_source = frame_reader
            current_pad = channel_name

            # Add injection if configured
            if self.noiseless_inj_frame_cache and self.noiseless_inj_channel_dict:
                if ifo in self.noiseless_inj_channel_dict:
                    inj_channel = f"{ifo}:{self.noiseless_inj_channel_dict[ifo]}"

                    inj_reader = FrameReader(
                        name=f"{self.name}_{ifo}_inj",
                        framecache=self.noiseless_inj_frame_cache,
                        channel_names=[inj_channel],
                        instrument=ifo,
                        t0=self.t0,
                        end=self.end,
                    )

                    # Add frames together
                    adder = Adder(
                        name=f"{self.name}_{ifo}_add",
                        sink_pad_names=("frame", "inj"),
                        source_pad_names=(ifo,),
                    )

                    compose.connect(
                        frame_reader,
                        adder,
                        link_map={"frame": channel_name},
                    )
                    compose.connect(
                        inj_reader,
                        adder,
                        link_map={"inj": inj_channel},
                    )

                    current_source = adder
                    current_pad = ifo

                    if self.verbose:
                        print(f"Added injection for {ifo} from {inj_channel}")
            else:
                # No injection - just insert the frame reader
                compose.insert(frame_reader)

            # Add segment gating if configured
            if segments_dict is not None and ifo in segments_dict:
                ifo_segments = segments_dict[ifo]

                if ifo_segments:  # Only add gating if there are segments
                    seg_source = SegmentSource(
                        name=f"{self.name}_{ifo}_seg",
                        source_pad_names=("control",),
                        rate=sample_rate,
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

                    compose.connect(
                        current_source,
                        gate,
                        link_map={"strain": current_pad},
                    )
                    compose.connect(
                        seg_source,
                        gate,
                        link_map={"control": "control"},
                    )

                    current_source = gate
                    current_pad = ifo

                    if self.verbose:
                        print(f"Added segment gating for {ifo}")

            # Add latency tracking if configured
            self._add_latency_tracking(compose, ifo, current_source, current_pad)

        return compose.as_source(
            name=self.name,
            also_expose_source_pads=self._also_expose_pads,
        )
