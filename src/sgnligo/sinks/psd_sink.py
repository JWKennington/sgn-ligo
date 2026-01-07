"""Sink element for writing Power Spectral Densities (PSDs) to disk."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

import lal
from sgn.base import Frame, SinkElement, SinkPad

from sgnligo.psd import FormatType, PSDWriter


@dataclass
class PSDSink(SinkElement):
    """
    A sink element that captures PSDs from frame metadata and writes them to disk.

    This sink tracks the **Data Time** (GPS Epoch) of the incoming PSDs.
    If `write_interval` is set, it will write a new file every N seconds of
    *data*, ensuring consistent output regardless of processing speed.

    Args:
        fname:
            str, path template for the output file. Default: "psd.xml".

            Standard Usage (Overwrite):
                - "psd.xml": Writes the final PSD at EOS.

            Sequential Usage (History):
                - "H1-PSD-{gps}.xml": Generates unique filenames tagged by GPS time.
                - "PSD-{gps}-{now:%H%M}.txt": Combines GPS and wall-clock time.

            The file extension (.xml, .xml.gz, .npz, .txt, .npy) determines the
            format unless `output_format` is overridden.

        output_format:
            str | None, optional format override ('xml', 'npz', 'txt', etc.).

        write_interval:
            float | None, interval in **GPS seconds** between writes.
            If None (default), writing only occurs once at End-Of-Stream (EOS).
            Example: 600.0 will write a file for every 10 minutes of data.

        verbose:
            bool, print debug information when writing. Default: False.
    """

    fname: str = "psd.xml"
    output_format: Optional[FormatType] = None
    write_interval: Optional[float] = None
    verbose: bool = False

    # Internal state
    _current_psds: Dict[str, lal.REAL8FrequencySeries] = field(
        default_factory=dict, init=False
    )
    _last_write_gps: float = field(default=-1.0, init=False)

    def __post_init__(self):
        super().__post_init__()
        self._current_psds = {}
        # Initialize with negative infinity effectively, so the first PSD triggers
        # a write check (or initializes the clock) immediately.
        self._last_write_gps = -1.0

    def pull(self, pad: SinkPad, frame: Frame) -> None:
        """
        Ingest a frame, update the PSD state, and advance the data clock.
        """
        if frame.metadata and "psd" in frame.metadata:
            psd = frame.metadata["psd"]
            if psd is not None:
                self._current_psds[self.rsnks[pad]] = psd

        # Always check for EOS to allow clean shutdown
        if frame.EOS:
            self.mark_eos(pad)

    def internal(self) -> None:
        """
        Check data time progression and trigger disk I/O if interval has passed.
        """
        # 1. Check Periodic Write (Data Time Driven)
        if self.write_interval is not None and self._current_psds:
            # Determine the current "Data Time" as the max epoch seen so far
            current_gps = max(float(p.epoch) for p in self._current_psds.values())

            # Initialize clock on first data
            if self._last_write_gps < 0:
                self._last_write_gps = current_gps

            # Check if enough data time has elapsed since last write
            if (current_gps - self._last_write_gps) >= self.write_interval:
                self._flush_to_disk()
                # Update the last write time to the CURRENT data time
                self._last_write_gps = current_gps

        # 2. Check Final Flush (EOS)
        if self.at_eos:
            self._flush_to_disk()

    def _resolve_path(self) -> str:
        """
        Resolve filename template with metadata keys ({gps}, {now}).
        """
        if "{" not in self.fname:
            return self.fname

        # Default to 0 if no PSDs yet (shouldn't happen during flush)
        gps_ref = 0
        if self._current_psds:
            gps_ref = int(max(p.epoch for p in self._current_psds.values()))

        try:
            return self.fname.format(gps=gps_ref, now=datetime.datetime.now())
        except KeyError as e:
            if self.verbose:
                print(f"PSDSink Warning: Unknown format key {e} in '{self.fname}'")
            return self.fname
        except ValueError as e:
            print(f"PSDSink Error: Bad format string '{self.fname}': {e}")
            return self.fname

    def _flush_to_disk(self):
        """Write current PSDs to disk."""
        if not self._current_psds:
            if self.verbose:
                print("PSDSink: No PSDs to write.")
            return

        output_path = self._resolve_path()

        try:
            PSDWriter.write(
                output_path,
                self._current_psds,
                fmt=self.output_format,
                verbose=self.verbose,
            )
        except Exception as e:
            print(f"PSDSink Error: Failed to write PSD to {output_path}: {e}")
