"""A source element to generate gravitational wave waveforms from injection files.

This module provides the SimInspiralSource class which reads injection parameters
from XML (LIGOLW) or HDF5 files and generates time-domain waveforms projected
onto each detector with proper time delays, antenna patterns, and phase corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lal
import lalsimulation as lalsim
import numpy as np
from igwn_ligolw import lsctables, utils as ligolw_utils
from sgn.base import SourcePad
from sgnts.base import Offset, TSFrame, TSSource
from sgnts.base.time import Time


@dataclass
class InjectionParams:
    """Normalized parameters for a single injection event.

    All angular quantities are in radians, masses in solar masses,
    distance in Mpc, and times in GPS seconds.
    """

    # Masses
    mass1: float
    mass2: float

    # Spins (dimensionless)
    spin1x: float = 0.0
    spin1y: float = 0.0
    spin1z: float = 0.0
    spin2x: float = 0.0
    spin2y: float = 0.0
    spin2z: float = 0.0

    # Orbital parameters
    distance: float = 100.0  # Mpc
    inclination: float = 0.0  # radians
    coa_phase: float = 0.0  # radians
    polarization: float = 0.0  # radians (psi)
    long_asc_nodes: float = 0.0  # radians
    eccentricity: float = 0.0
    mean_per_ano: float = 0.0  # radians

    # Sky position
    ra: float = 0.0  # radians (right ascension)
    dec: float = 0.0  # radians (declination)

    # Timing
    geocent_end_time: float = 0.0  # GPS seconds (coalescence time at geocenter)

    # Waveform model
    approximant: str = "IMRPhenomD"

    # Optional: reference frequency (Hz)
    f_ref: float = 20.0


def _load_xml_injections(filepath: str) -> List[InjectionParams]:
    """Load injections from LIGOLW XML file.

    Args:
        filepath: Path to XML file (can be .xml or .xml.gz)

    Returns:
        List of InjectionParams objects
    """
    xmldoc = ligolw_utils.load_filename(filepath, verbose=False)
    sim_table = lsctables.SimInspiralTable.get_table(xmldoc)

    injections = []
    for row in sim_table:
        # Get geocentric end time
        geocent_end_time = float(row.geocent_end_time) + float(
            row.geocent_end_time_ns
        ) * 1e-9

        inj = InjectionParams(
            mass1=row.mass1,
            mass2=row.mass2,
            spin1x=row.spin1x,
            spin1y=row.spin1y,
            spin1z=row.spin1z,
            spin2x=row.spin2x,
            spin2y=row.spin2y,
            spin2z=row.spin2z,
            distance=row.distance,
            inclination=row.inclination,
            coa_phase=row.coa_phase,
            polarization=row.polarization,
            long_asc_nodes=getattr(row, "long_asc_nodes", 0.0),
            eccentricity=getattr(row, "eccentricity", 0.0) or 0.0,
            mean_per_ano=getattr(row, "mean_per_ano", 0.0) or 0.0,
            ra=row.longitude,  # LIGOLW uses longitude for RA
            dec=row.latitude,  # LIGOLW uses latitude for Dec
            geocent_end_time=geocent_end_time,
            approximant=row.waveform or "IMRPhenomD",
            f_ref=getattr(row, "f_lower", 20.0) or 20.0,
        )
        injections.append(inj)

    return injections


def _load_hdf5_injections(filepath: str) -> List[InjectionParams]:
    """Load injections from HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        List of InjectionParams objects
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to read HDF5 injection files")

    injections = []
    with h5py.File(filepath, "r") as f:
        # Common HDF5 injection file structure
        # Try to find the injections group/dataset
        if "injections" in f:
            grp = f["injections"]
        else:
            grp = f

        # Get number of injections
        n_inj = len(grp["mass1"])

        for i in range(n_inj):
            # Handle optional fields with defaults
            def get_val(key, default=0.0):
                if key in grp:
                    val = grp[key][i]
                    return float(val) if val is not None else default
                return default

            def get_str(key, default="IMRPhenomD"):
                if key in grp:
                    val = grp[key][i]
                    if isinstance(val, bytes):
                        return val.decode("utf-8")
                    return str(val) if val else default
                return default

            inj = InjectionParams(
                mass1=float(grp["mass1"][i]),
                mass2=float(grp["mass2"][i]),
                spin1x=get_val("spin1x"),
                spin1y=get_val("spin1y"),
                spin1z=get_val("spin1z"),
                spin2x=get_val("spin2x"),
                spin2y=get_val("spin2y"),
                spin2z=get_val("spin2z"),
                distance=get_val("distance", 100.0),
                inclination=get_val("inclination"),
                coa_phase=get_val("coa_phase"),
                polarization=get_val("polarization"),
                long_asc_nodes=get_val("long_asc_nodes"),
                eccentricity=get_val("eccentricity"),
                mean_per_ano=get_val("mean_per_ano"),
                ra=get_val("ra", get_val("longitude")),
                dec=get_val("dec", get_val("latitude")),
                geocent_end_time=get_val("geocent_end_time", get_val("tc")),
                approximant=get_str("approximant", get_str("waveform")),
                f_ref=get_val("f_ref", 20.0),
            )
            injections.append(inj)

    return injections


def load_injections(filepath: str) -> List[InjectionParams]:
    """Load injections from XML or HDF5 file with auto-detection.

    Args:
        filepath: Path to injection file

    Returns:
        List of InjectionParams objects
    """
    if filepath.endswith((".xml", ".xml.gz")):
        return _load_xml_injections(filepath)
    elif filepath.endswith((".hdf5", ".h5", ".hdf")):
        return _load_hdf5_injections(filepath)
    else:
        # Try XML first, then HDF5
        try:
            return _load_xml_injections(filepath)
        except Exception:
            return _load_hdf5_injections(filepath)


def estimate_waveform_duration(
    inj: InjectionParams, f_min: float
) -> tuple[float, float]:
    """Estimate the duration of a waveform before and after merger.

    Args:
        inj: Injection parameters
        f_min: Minimum frequency for waveform generation (Hz)

    Returns:
        Tuple of (pre_merger_duration, post_merger_duration) in seconds
    """
    m1_si = inj.mass1 * lal.MSUN_SI
    m2_si = inj.mass2 * lal.MSUN_SI
    mtotal_si = m1_si + m2_si

    # Chirp time (inspiral duration)
    try:
        tchirp = lalsim.SimInspiralChirpTimeBound(
            f_min, m1_si, m2_si, inj.spin1z, inj.spin2z
        )
    except Exception:
        # Fallback estimate using leading-order chirp time
        mchirp = (m1_si * m2_si) ** (3.0 / 5.0) / mtotal_si ** (1.0 / 5.0)
        tchirp = (
            5.0
            / 256.0
            * (lal.C_SI**3 / (lal.G_SI * mchirp)) ** (5.0 / 3.0)
            * (np.pi * f_min) ** (-8.0 / 3.0)
        )

    # Merge time bound
    try:
        tmerge = lalsim.SimInspiralMergeTimeBound(m1_si, m2_si)
    except Exception:
        tmerge = 0.1  # Conservative estimate

    # Ringdown time bound
    try:
        # Final spin estimate (simple approximation)
        final_spin = min(abs(inj.spin1z) + abs(inj.spin2z), 0.998)
        tring = lalsim.SimInspiralRingdownTimeBound(mtotal_si, final_spin)
    except Exception:
        tring = 0.5  # Conservative estimate

    # Add safety margins
    pre_merger = tchirp + 1.0  # Extra second before
    post_merger = tmerge + tring + 0.5  # Extra half second after

    return pre_merger, post_merger


def generate_waveform_td(
    inj: InjectionParams,
    sample_rate: int,
    f_min: float,
    approximant_override: Optional[str] = None,
) -> tuple[lal.REAL8TimeSeries, lal.REAL8TimeSeries]:
    """Generate time-domain h+, hx waveforms.

    Automatically selects time-domain or frequency-domain generation
    based on the approximant, converting FD waveforms to TD via IFFT.

    Args:
        inj: Injection parameters
        sample_rate: Sample rate in Hz
        f_min: Minimum frequency in Hz
        approximant_override: Override the approximant from injection params

    Returns:
        Tuple of (hp, hc) as LAL REAL8TimeSeries objects
    """
    approximant_str = approximant_override or inj.approximant
    approximant = lalsim.GetApproximantFromString(approximant_str)

    # Convert units
    m1_si = inj.mass1 * lal.MSUN_SI
    m2_si = inj.mass2 * lal.MSUN_SI
    distance_si = inj.distance * 1e6 * lal.PC_SI  # Mpc to meters

    delta_t = 1.0 / sample_rate
    f_ref = inj.f_ref

    # Check if approximant is natively time-domain
    is_td = lalsim.SimInspiralImplementedTDApproximants(approximant)

    if is_td:
        # Generate time-domain waveform directly
        hp, hc = lalsim.SimInspiralTD(
            m1_si,
            m2_si,
            inj.spin1x,
            inj.spin1y,
            inj.spin1z,
            inj.spin2x,
            inj.spin2y,
            inj.spin2z,
            distance_si,
            inj.inclination,
            inj.coa_phase,
            inj.long_asc_nodes,
            inj.eccentricity,
            inj.mean_per_ano,
            delta_t,
            f_min,
            f_ref,
            lal.CreateDict(),
            approximant,
        )
    else:
        # Generate frequency-domain and convert to time-domain
        # Estimate duration to set appropriate frequency resolution
        pre_dur, post_dur = estimate_waveform_duration(inj, f_min)
        total_duration = pre_dur + post_dur

        # Pad to power of 2 for efficient FFT
        n_samples = int(np.ceil(total_duration * sample_rate))
        n_samples = int(2 ** np.ceil(np.log2(n_samples)))
        delta_f = sample_rate / n_samples
        f_max = sample_rate / 2.0

        hp_fd, hc_fd = lalsim.SimInspiralFD(
            m1_si,
            m2_si,
            inj.spin1x,
            inj.spin1y,
            inj.spin1z,
            inj.spin2x,
            inj.spin2y,
            inj.spin2z,
            distance_si,
            inj.inclination,
            inj.coa_phase,
            inj.long_asc_nodes,
            inj.eccentricity,
            inj.mean_per_ano,
            delta_f,
            f_min,
            f_max,
            f_ref,
            lal.CreateDict(),
            approximant,
        )

        # Convert to time-domain using IFFT
        hp = _fd_to_td(hp_fd, delta_t)
        hc = _fd_to_td(hc_fd, delta_t)

    return hp, hc


def _fd_to_td(
    h_fd: lal.COMPLEX16FrequencySeries, delta_t: float
) -> lal.REAL8TimeSeries:
    """Convert frequency-domain waveform to time-domain.

    Args:
        h_fd: Frequency-domain waveform
        delta_t: Time step (1/sample_rate)

    Returns:
        Time-domain waveform as REAL8TimeSeries
    """
    # Get the data
    data_fd = h_fd.data.data

    # Number of time samples (real FFT)
    n_freq = len(data_fd)
    n_time = 2 * (n_freq - 1)

    # Perform IFFT
    # For real output, we need to handle the Hermitian symmetry
    data_td = np.fft.irfft(data_fd, n=n_time)

    # Create time series
    sample_rate = 1.0 / delta_t
    h_td = lal.CreateREAL8TimeSeries(
        h_fd.name,
        h_fd.epoch,
        h_fd.f0,
        delta_t,
        lal.DimensionlessUnit,
        len(data_td),
    )
    h_td.data.data[:] = data_td * sample_rate  # Normalization for IFFT

    return h_td


def project_to_detector(
    hp: lal.REAL8TimeSeries,
    hc: lal.REAL8TimeSeries,
    inj: InjectionParams,
    ifo: str,
) -> lal.REAL8TimeSeries:
    """Project h+, hx waveforms onto a detector.

    Uses LALSimulation's accurate projection which handles:
    - Antenna response (F+, Fx)
    - Light travel time delays
    - Phase corrections

    Args:
        hp: Plus polarization time series
        hc: Cross polarization time series
        inj: Injection parameters (for sky position, polarization)
        ifo: Interferometer prefix (H1, L1, V1, etc.)

    Returns:
        Detector strain as REAL8TimeSeries
    """
    # Get detector
    detector = lal.cached_detector_by_prefix[ifo]

    # Use LALSimulation's accurate detector strain function
    strain = lalsim.SimDetectorStrainREAL8TimeSeries(
        hp,
        hc,
        inj.ra,
        inj.dec,
        inj.polarization,
        detector,
    )

    return strain


@dataclass
class CachedWaveform:
    """Cached waveform data for an injection."""

    injection_id: int
    start_gps: float  # GPS time when waveform starts
    end_gps: float  # GPS time when waveform ends
    strain: Dict[str, np.ndarray]  # Per-IFO strain data {ifo: array}
    sample_rate: int


class WaveformCache:
    """Manages waveform generation and caching.

    Waveforms are generated on-demand and cached until they're fully
    consumed (i.e., the pipeline has moved past the waveform's end time).
    """

    def __init__(
        self,
        injections: List[InjectionParams],
        ifos: List[str],
        sample_rate: int,
        f_min: float,
        approximant_override: Optional[str] = None,
    ):
        """Initialize the waveform cache.

        Args:
            injections: List of injection parameters
            ifos: List of interferometer prefixes
            sample_rate: Output sample rate in Hz
            f_min: Minimum frequency for waveform generation
            approximant_override: Override approximant for all injections
        """
        self.injections = injections
        self.ifos = ifos
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.approximant_override = approximant_override
        self.cache: Dict[int, CachedWaveform] = {}

        # Pre-compute injection time windows for fast overlap queries
        self._injection_windows: List[tuple[float, float]] = []
        for inj in injections:
            pre_dur, post_dur = estimate_waveform_duration(inj, f_min)
            start = inj.geocent_end_time - pre_dur
            end = inj.geocent_end_time + post_dur
            self._injection_windows.append((start, end))

    def get_overlapping_injections(
        self, buf_start: float, buf_end: float
    ) -> List[int]:
        """Find injections that overlap the buffer time window.

        Args:
            buf_start: Buffer start GPS time
            buf_end: Buffer end GPS time

        Returns:
            List of injection indices that overlap the buffer
        """
        overlapping = []
        for i, (inj_start, inj_end) in enumerate(self._injection_windows):
            if inj_start < buf_end and inj_end > buf_start:
                overlapping.append(i)
        return overlapping

    def _generate_and_cache(self, inj_id: int) -> None:
        """Generate waveform for an injection and cache it.

        Args:
            inj_id: Index of injection in self.injections
        """
        inj = self.injections[inj_id]

        # Generate h+, hx waveforms
        hp, hc = generate_waveform_td(
            inj, self.sample_rate, self.f_min, self.approximant_override
        )

        # The waveform epoch is relative to the merger time (typically negative)
        # LAL waveforms have epoch indicating time of first sample relative to
        # the reference point (which is usually the coalescence/merger time)
        # So absolute GPS time = geocent_end_time + epoch
        epoch_seconds = float(hp.epoch.gpsSeconds) + float(hp.epoch.gpsNanoSeconds) * 1e-9
        wf_start_gps = inj.geocent_end_time + epoch_seconds

        # Waveform duration
        n_samples = hp.data.length
        wf_duration = n_samples / self.sample_rate
        wf_end_gps = wf_start_gps + wf_duration

        # Project onto each detector
        strain_dict = {}
        for ifo in self.ifos:
            detector_strain = project_to_detector(hp, hc, inj, ifo)
            strain_dict[ifo] = detector_strain.data.data.copy()

        # Cache the waveform
        self.cache[inj_id] = CachedWaveform(
            injection_id=inj_id,
            start_gps=wf_start_gps,
            end_gps=wf_end_gps,
            strain=strain_dict,
            sample_rate=self.sample_rate,
        )

    def get_waveform_slice(
        self,
        inj_id: int,
        ifo: str,
        buf_start: float,
        buf_end: float,
    ) -> tuple[np.ndarray, int]:
        """Get the waveform slice for a specific buffer window.

        Args:
            inj_id: Injection index
            ifo: Interferometer prefix
            buf_start: Buffer start GPS time
            buf_end: Buffer end GPS time

        Returns:
            Tuple of (waveform_slice, start_sample_in_buffer)
            where start_sample_in_buffer is the index in the output buffer
            where this slice should begin
        """
        if inj_id not in self.cache:
            self._generate_and_cache(inj_id)

        cached = self.cache[inj_id]
        strain = cached.strain[ifo]

        # Calculate overlap region
        overlap_start = max(buf_start, cached.start_gps)
        overlap_end = min(buf_end, cached.end_gps)

        if overlap_start >= overlap_end:
            # No overlap
            return np.array([]), 0

        # Sample indices in waveform array
        wf_start_idx = int(
            (overlap_start - cached.start_gps) * self.sample_rate
        )
        wf_end_idx = int((overlap_end - cached.start_gps) * self.sample_rate)

        # Clamp to valid range
        wf_start_idx = max(0, wf_start_idx)
        wf_end_idx = min(len(strain), wf_end_idx)

        # Sample index in output buffer where slice begins
        buf_start_idx = int((overlap_start - buf_start) * self.sample_rate)
        buf_start_idx = max(0, buf_start_idx)

        return strain[wf_start_idx:wf_end_idx], buf_start_idx

    def cleanup_expired(self, current_gps: float) -> None:
        """Remove waveforms that are fully consumed.

        Args:
            current_gps: Current GPS time of the pipeline
        """
        expired = [k for k, v in self.cache.items() if v.end_gps < current_gps]
        for k in expired:
            del self.cache[k]


@dataclass
class SimInspiralSource(TSSource):
    """Source element that generates GW waveforms from an injection file.

    Reads injection parameters from XML (LIGOLW SimInspiralTable) or HDF5 files
    and generates time-domain waveforms projected onto each detector with proper
    time delays, antenna patterns, and phase corrections using LALSimulation.

    The source outputs zeros + summed injection signals on separate pads for
    each interferometer. Combine with a noise source using an Adder transform
    to create realistic data with injections.

    Args:
        injection_file: Path to injection file (XML or HDF5)
        ifos: List of interferometer prefixes, e.g., ["H1", "L1", "V1"]
        sample_rate: Output sample rate in Hz (default: 16384)
        f_min: Minimum frequency for waveform generation in Hz (default: 10)
        approximant_override: Override approximant for all injections (optional)

    Example:
        >>> source = SimInspiralSource(
        ...     name="Injections",
        ...     injection_file="injections.xml",
        ...     ifos=["H1", "L1"],
        ...     t0=1234567890,
        ...     end=1234567900,
        ... )
    """

    injection_file: Optional[str] = None
    ifos: Optional[List[str]] = None
    sample_rate: int = 16384
    f_min: float = 10.0
    approximant_override: Optional[str] = None

    # Internal state (not user-configurable)
    _waveform_cache: WaveformCache = field(init=False, repr=False)
    _injections: List[InjectionParams] = field(
        init=False, repr=False, default_factory=list
    )
    _channel_dict: Dict[str, str] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self):
        """Initialize the source."""
        if self.injection_file is None:
            raise ValueError("injection_file must be specified")

        if self.ifos is None:
            self.ifos = ["H1", "L1"]

        # Create channel names for source pads
        self._channel_dict = {ifo: f"{ifo}:INJ-STRAIN" for ifo in self.ifos}
        self.source_pad_names = list(self._channel_dict.values())

        # Load injections
        self._injections = load_injections(self.injection_file)

        # Initialize waveform cache
        self._waveform_cache = WaveformCache(
            injections=self._injections,
            ifos=self.ifos,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            approximant_override=self.approximant_override,
        )

        # Call parent's post_init
        super().__post_init__()

        # Set buffer params for each pad
        for ifo, channel in self._channel_dict.items():
            pad = self.srcs[channel]
            self.set_pad_buffer_params(
                pad=pad,
                sample_shape=(),
                rate=self.sample_rate,
            )

    def _ifo_from_pad(self, pad: SourcePad) -> str:
        """Get IFO prefix from pad."""
        for ifo, channel in self._channel_dict.items():
            if self.srcs[channel] == pad:
                return ifo
        raise ValueError(f"Unknown pad: {pad}")

    def new(self, pad: SourcePad) -> TSFrame:
        """Generate a new frame with injection signals.

        Args:
            pad: Source pad requesting new data

        Returns:
            TSFrame containing injection signals (zeros + waveforms)
        """
        # Get frame prepared by base class
        frame = self.prepare_frame(pad)
        buffer = frame.buffers[0]

        # Determine IFO from pad
        ifo = self._ifo_from_pad(pad)

        # Get buffer time window
        # Convert from Offset to GPS seconds
        # Offset is stored as seconds * sample_stride(16384), so divide by that
        # to get GPS time
        stride_16k = Offset.sample_stride(16384)
        buf_start = buffer.offset / stride_16k
        buf_end = buffer.end_offset / stride_16k

        # Get number of samples from the buffer's expected shape
        # The buffer shape is set by set_pad_buffer_params and prepare_frame
        num_samples = buffer.shape[0]

        # Start with zeros
        output = np.zeros(num_samples, dtype=np.float64)

        # Find and sum all overlapping injections
        overlapping = self._waveform_cache.get_overlapping_injections(
            buf_start, buf_end
        )
        for inj_id in overlapping:
            wf_slice, start_idx = self._waveform_cache.get_waveform_slice(
                inj_id, ifo, buf_start, buf_end
            )
            if len(wf_slice) > 0:
                # Add waveform slice to output
                end_idx = start_idx + len(wf_slice)
                # Handle edge cases where slice might extend beyond buffer
                actual_end = min(end_idx, num_samples)
                slice_len = actual_end - start_idx
                output[start_idx:actual_end] += wf_slice[:slice_len]

        buffer.set_data(output)
        return frame

    def internal(self) -> None:
        """Internal processing - cleanup expired waveforms."""
        super().internal()

        # Cleanup expired waveforms from cache
        # Convert from Offset to GPS seconds
        stride_16k = Offset.sample_stride(16384)
        current_gps = self.current_end / stride_16k
        self._waveform_cache.cleanup_expired(current_gps)
