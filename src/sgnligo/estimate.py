"""
High-level utilities for estimating Power Spectral Densities (PSDs) from data.

This module provides composable functions to estimate PSDs from various sources
(GWOSC, local frames, generated noise) by constructing and running an efficient
`sgn` pipeline.

Note: **Design Rationale vs. Legacy `sgnl`**
   This module is intended as a clean, composable replacement for the legacy
   `measure_psd` functionality found in `sgnl`. It was written from scratch
   to avoid dependency on the legacy `DataSourceInfo` configuration pattern,
   which is currently undergoing significant refactoring to reduce complexity.
   This implementation favors explicit dependency injection (passing `SourceElement`
   instances) over monolithic configuration objects.
"""

from __future__ import annotations

from typing import List, Optional

from sgn.apps import Pipeline
from sgn.base import SourceElement
from sgnts.base import TSSource
from sgnts.sinks import NullSeriesSink
from sgnts.transforms.resampler import Resampler

from sgnligo.sinks.psd_sink import PSDSink
from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource
from sgnligo.sources.gwosc import GWOSCSource
from sgnligo.transforms.whiten import Whiten


def infer_source_sample_rate(source: SourceElement, ifo: str) -> float:
    """
    Attempt to determine the output sample rate of a source element.

    Strategy:
    1. Check for `GWDataNoiseSource`: Rate is stored per-channel in `channel_info`.
    2. Check for `sample_rate` attribute: Common in `GWOSCSource` and
        `FakeSeriesSource`.
    3. Default to 4096.0 Hz (Standard for open data).

    Args:
        source: The source element instance.
        ifo: The interferometer name (e.g., "H1") to look up.

    Returns:
        float: The inferred sample rate in Hz.
    """
    # Case 1: GWDataNoiseSource (Rate is implicit from PSD, stored in channel_info)
    if isinstance(source, GWDataNoiseSource):
        # channel_info is keyed by IFO (e.g. "H1")
        if hasattr(source, "channel_info") and ifo in source.channel_info:
            return float(source.channel_info[ifo]["rate"])

    # Case 2: Standard Sources (GWOSC, FakeSeries) with explicit rate
    if hasattr(source, "sample_rate"):
        return float(source.sample_rate)

    # Case 3: Fallback
    # Most public LIGO data is 4096 Hz or 16384 Hz. 4096 is the safer default assumption
    # for analysis if we really don't know.
    return 4096.0


def estimate_psd(
    start: float,
    end: float,
    ifos: List[str],
    sample_rate: int,
    source_sample_rate: Optional[float] = None,
    fname: str = "psd.xml",
    fft_length: int = 4,
    write_interval: Optional[float] = None,
    source_element: Optional[TSSource] = None,
    verbose: bool = False,
) -> str:
    """
    Measure the Power Spectral Density (PSD) for a set of detectors over a specific
    time range.

    This function constructs a temporary `sgn` pipeline consisting of:
    1. A Data Source (Default: GWOSC, or user-provided)
    2. A Resampler (automatically added if source rate != analysis rate)
    3. A Whitening transform (which computes the PSD)
    4. A PSD Sink (writes the result to disk)

    Args:
        start:
            GPS start time of the data segment.
        end:
            GPS end time of the data segment.
        ifos:
            List of interferometer names to analyze (e.g., ["H1", "L1"]).
        sample_rate:
            Target sample rate (Hz) for the analysis.
        source_sample_rate:
            Optional explicit source rate. If None (default), the function attempts
            to infer it from the `source_element`.
        fname:
            Output filename. Supports formatting keys like `{gps}` for periodic writing.
            Defaults to "psd.xml" (LIGO-LW XML format).
            Extensions .npz, .txt, .npy are also supported.
        fft_length:
            Length of the FFT window in seconds for PSD estimation. Default: 4.0s.
        write_interval:
            Optional interval in GPS seconds to write periodic PSD updates.
            If None (default), the PSD is written only once at the end of the stream.
        source_element:
            Optional pre-configured `sgn` SourceElement.
            If None, a `GWOSCSource` will be created automatically to fetch open data.
            Note: The source must provide output pads named after the IFOs (e.g., "H1").
        verbose:
            If True, prints pipeline execution details and sink status.

    Returns:
        The path to the output file (same as `fname`).
    """
    pipe = Pipeline()

    # 1. Configure Source
    if source_element is None:
        # Default to GWOSC Source if none provided
        if verbose:
            print(f"Initializing GWOSC Source for {ifos}...")
        source = GWOSCSource(
            name="gwosc_source",
            start=start,
            end=end,
            detectors=ifos,
            cache_data=True,
        )
    else:
        source = source_element

    # 2. Configure Sinks
    psd_sink = PSDSink(
        name="psd_sink",
        fname=fname,
        write_interval=write_interval,
        verbose=verbose,
        sink_pad_names=ifos,
    )

    null_sink = NullSeriesSink(
        name="discard_hoft",
        sink_pad_names=[f"dump_{ifo}" for ifo in ifos],
    )

    pipe.insert(source, psd_sink, null_sink)

    # 3. Build Processing Chain per IFO
    for ifo in ifos:
        if ifo not in source.srcs:
            raise ValueError(
                f"Source element '{source.name}' does not have an "
                f"output pad named '{ifo}'"
            )

        # Track the upstream source pad for linking
        upstream_pad = source.srcs[ifo]

        # Determine Input Rate
        if source_sample_rate is not None:
            current_rate = source_sample_rate
        else:
            current_rate = infer_source_sample_rate(source, ifo)

        if verbose:
            print(
                f"IFO {ifo}: Source Rate={current_rate} Hz -> Target={sample_rate} Hz"
            )

        # A. Resampler (Conditional)
        if int(current_rate) != int(sample_rate):
            resampler = Resampler(
                name=f"resample_{ifo}",
                inrate=int(current_rate),
                outrate=int(sample_rate),
                sink_pad_names=["source"],
                source_pad_names=["sink"],
            )
            pipe.insert(resampler)

            # Link Source -> Resampler
            pipe.link({resampler.snks["source"]: upstream_pad})
            upstream_pad = resampler.srcs["sink"]

        # B. Whitener
        whitener = Whiten(
            name=f"whiten_{ifo}",
            sink_pad_names=[ifo],
            whiten_pad_name="hoft",
            psd_pad_name="psd",
            instrument=ifo,
            input_sample_rate=sample_rate,
            whiten_sample_rate=sample_rate,
            fft_length=fft_length,
            nmed=7,
            navg=64,
        )

        pipe.insert(whitener)

        # Link Upstream -> Whiten
        pipe.link({whitener.snks[ifo]: upstream_pad})

        # Whiten -> Sinks
        pipe.link(
            {
                psd_sink.snks[ifo]: whitener.srcs["psd"],
                null_sink.snks[f"dump_{ifo}"]: whitener.srcs["hoft"],
            }
        )

    # 5. Execute
    if verbose:
        print(f"Running PSD estimation pipeline from {start} to {end}...")

    pipe.run()

    if verbose:
        print(f"PSD estimation complete. Output written to {fname}")

    return fname
