"""A sink element to write coinc XML files from MockGWEventSource.

This module provides the CoincXMLSink class which receives coinc XML events
from MockGWEventSource (or similar) and writes them to disk as numbered
XML files, with optional verbose output showing event details.
"""

from __future__ import annotations

import base64
import gzip
import io
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from igwn_ligolw import lsctables
from igwn_ligolw import utils as ligolw_utils
from sgn.base import SinkElement, SinkPad
from sgn.frames import Frame

# Gzip magic header bytes (same as used internally by gzip module)
GZIP_MAGIC = b"\x1f\x8b"


def _format_gps_time(gps_sec: int, gps_ns: int) -> str:
    """Format GPS time as a readable string."""
    return f"{gps_sec}.{gps_ns:09d}"


def _summarize_coinc_xmldoc(xml_bytes: bytes) -> Dict:
    """Extract summary information from coinc XML bytes.

    Args:
        xml_bytes: Raw XML document bytes

    Returns:
        Dictionary with event summary information
    """
    # Parse the XML
    buffer = io.BytesIO(xml_bytes)
    xmldoc = ligolw_utils.load_fileobj(buffer)

    # Extract tables
    coinc_inspiral = lsctables.CoincInspiralTable.get_table(xmldoc)[0]
    sngl_inspirals = {
        row.ifo: row for row in lsctables.SnglInspiralTable.get_table(xmldoc)
    }

    # Build summary
    summary = {
        "end_time": _format_gps_time(
            coinc_inspiral.end_time, coinc_inspiral.end_time_ns
        ),
        "network_snr": coinc_inspiral.snr,
        "ifos": coinc_inspiral.ifos,
        "mchirp": coinc_inspiral.mchirp,
        "mtotal": coinc_inspiral.mass,
        "far": coinc_inspiral.combined_far,
        "triggers": {},
    }

    for ifo, sngl in sngl_inspirals.items():
        summary["triggers"][ifo] = {
            "snr": sngl.snr,
            "end_time": _format_gps_time(sngl.end_time, sngl.end_time_ns),
            "mass1": sngl.mass1,
            "mass2": sngl.mass2,
            "coa_phase": sngl.coa_phase,
        }

    return summary


def _print_event_summary(
    event_id: int,
    pipeline: str,
    summary: Dict,
    output_path: Optional[str] = None,
) -> None:
    """Print a formatted summary of a coinc event.

    Args:
        event_id: Event identifier
        pipeline: Pipeline name that produced the event
        summary: Event summary dictionary from _summarize_coinc_xmldoc
        output_path: Path where XML was written (if any)
    """
    print(f"\n{'='*60}")
    print(f"COINC EVENT {event_id} [{pipeline.upper()}]")
    print(f"{'='*60}")
    print(f"  GPS Time:     {summary['end_time']}")
    print(f"  Network SNR:  {summary['network_snr']:.2f}")
    print(f"  Detectors:    {summary['ifos']}")
    print(f"  Chirp Mass:   {summary['mchirp']:.3f} Msun")
    print(f"  Total Mass:   {summary['mtotal']:.3f} Msun")
    print(f"  FAR:          {summary['far']:.2e} Hz")

    print("\n  Single-detector triggers:")
    for ifo, trig in sorted(summary["triggers"].items()):
        print(f"    {ifo}:")
        print(f"      SNR:       {trig['snr']:.2f}")
        print(f"      End Time:  {trig['end_time']}")
        print(f"      Masses:    {trig['mass1']:.2f} + {trig['mass2']:.2f} Msun")
        print(f"      Phase:     {trig['coa_phase']:.3f} rad")

    if output_path:
        print(f"\n  Written to: {output_path}")
    print()


@dataclass
class CoincXMLSink(SinkElement):
    """Sink element to write coinc XML files from MockGWEventSource.

    Receives Frame objects containing coinc XML bytes and writes them
    to numbered XML files in a specified directory. Optionally prints
    detailed event summaries to the console.

    Args:
        output_dir: Directory to write XML files (default: "coinc_output")
        pipelines: List of pipeline pad names to accept (default: all standard)
        filename_template: Template for output filenames. Supports {event_id},
            {pipeline}, {gps_time} placeholders.
        verbose: If True, print detailed event summary for each event
        compress: If True, write gzip-compressed XML (.xml.gz)
        create_dir: If True, create output directory if it doesn't exist

    Example:
        >>> from sgnligo.sources import MockGWEventSource
        >>> from sgnligo.sinks import CoincXMLSink
        >>>
        >>> source = MockGWEventSource(
        ...     event_cadence=10.0,
        ...     t0=1000000000.0,
        ...     duration=60.0,
        ...     real_time=False,
        ... )
        >>>
        >>> sink = CoincXMLSink(
        ...     output_dir="./events",
        ...     verbose=True,
        ... )
        >>>
        >>> # Connect source pads to sink pads
        >>> for pipeline in source.source_pad_names:
        ...     source.srcs[pipeline].link(sink.snks[pipeline])
    """

    output_dir: str = "coinc_output"
    pipelines: Optional[List[str]] = None
    filename_template: str = "{pipeline}_{event_id:04d}.xml"
    verbose: bool = True
    compress: bool = False
    create_dir: bool = True

    def __post_init__(self):
        """Initialize the sink after creation."""
        # Set default pipelines
        if self.pipelines is None:
            self.pipelines = ["SGNL", "pycbc", "MBTA", "spiir"]

        # Set sink pad names
        self.sink_pad_names = list(self.pipelines)

        # Call parent's post_init
        super().__post_init__()

        # Create output directory if needed
        if self.create_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Track event counts per pipeline
        self._event_counts: Dict[str, int] = {p: 0 for p in self.pipelines}
        self._total_events = 0

        if self.verbose:
            print("CoincXMLSink initialized:")
            print(f"  Output directory: {self.output_dir}")
            print(f"  Pipelines: {self.pipelines}")
            print(f"  Filename template: {self.filename_template}")
            print(f"  Compress: {self.compress}")
            print()

    def pull(self, pad: SinkPad, frame: Frame) -> None:
        """Process an incoming frame from a pipeline pad.

        Args:
            pad: The sink pad that received the frame
            frame: The incoming Frame containing coinc XML data
        """
        # Handle EOS
        if frame.EOS:
            self.mark_eos(pad)
            return

        # Skip gap frames
        if frame.is_gap or frame.data is None:
            return

        # Get pipeline name from pad
        pipeline = self.rsnks[pad]

        # Extract data from frame
        xml_b64 = frame.data.get("xml")
        event_id = frame.data.get("event_id", self._event_counts[pipeline])

        if xml_b64 is None:
            return

        # Decode base64 data
        xml_data = base64.b64decode(xml_b64)

        # Check if data is gzip compressed
        is_gzipped = len(xml_data) >= 2 and xml_data[:2] == GZIP_MAGIC

        # Generate output filename
        gps_time = frame.data.get("gpstime", 0)
        filename = self.filename_template.format(
            event_id=event_id,
            pipeline=pipeline,
            gps_time=int(gps_time),
        )

        # Determine output format and adjust filename
        if is_gzipped:
            # Data is already gzipped - write directly as .xml.gz
            if not filename.endswith(".gz"):
                filename = filename.replace(".xml", ".xml.gz")
                if not filename.endswith(".gz"):
                    filename += ".gz"
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, "wb") as f:
                f.write(xml_data)
            # Decompress for summary parsing
            xml_bytes = gzip.decompress(xml_data)
        elif self.compress:
            # Data is not gzipped but user wants compression
            if not filename.endswith(".gz"):
                filename += ".gz"
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, "wb") as f:
                f.write(gzip.compress(xml_data))
            xml_bytes = xml_data
        else:
            # Write as plain XML
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, "wb") as f:
                f.write(xml_data)
            xml_bytes = xml_data

        # Update counts
        self._event_counts[pipeline] += 1
        self._total_events += 1

        # Print summary if verbose
        if self.verbose:
            try:
                summary = _summarize_coinc_xmldoc(xml_bytes)
                _print_event_summary(event_id, pipeline, summary, output_path)
            except Exception as e:
                print(f"Warning: Could not parse event summary: {e}")
                print(f"  Written to: {output_path}")

    def internal(self) -> None:
        """Periodic internal processing."""
        pass  # No periodic processing needed

    def get_stats(self) -> Dict:
        """Get statistics about processed events.

        Returns:
            Dictionary with event counts per pipeline and total
        """
        return {
            "per_pipeline": dict(self._event_counts),
            "total": self._total_events,
        }

    def print_summary(self) -> None:
        """Print a summary of all processed events."""
        print(f"\n{'='*60}")
        print("COINC XML SINK SUMMARY")
        print(f"{'='*60}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Total events written: {self._total_events}")
        print("\n  Events per pipeline:")
        for pipeline, count in sorted(self._event_counts.items()):
            print(f"    {pipeline}: {count}")
        print()
