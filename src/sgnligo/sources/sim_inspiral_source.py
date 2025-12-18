"""A source element to generate gravitational wave waveforms from injection files.

This module provides the SimInspiralSource class which reads injection parameters
from XML (LIGOLW) or HDF5 files and generates time-domain waveforms projected
onto each detector with proper time delays, antenna patterns, and phase corrections.

All injection parameters are stored internally as LAL dictionaries (lal.Dict) in
SI units, which allows pass-through of all LAL-supported parameters including
tidal deformability, higher-order modes, and approximant-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lal
import lalsimulation as lalsim
import numpy as np
from igwn_ligolw import lsctables
from igwn_ligolw import utils as ligolw_utils
from sgn.base import SourcePad
from sgnts.base import Offset, TSFrame, TSSource

# =============================================================================
# Required and Optional Injection Parameters
# =============================================================================
#
# These define what SimInspiralSource expects in injection dictionaries.
# All values are in SI units (masses in kg, distances in meters, angles in radians).

REQUIRED_FIELDS: Dict[str, type] = {
    "mass1": float,  # Primary mass [kg]
    "mass2": float,  # Secondary mass [kg]
    "distance": float,  # Luminosity distance [m]
    "ra": float,  # Right ascension [rad]
    "dec": float,  # Declination [rad]
    "psi": float,  # Polarization angle [rad]
    "t_co_gps": float,  # Coalescence GPS time [s]
    "approximant": str,  # Waveform approximant (e.g., "IMRPhenomD")
}

OPTIONAL_FIELDS: Dict[str, float] = {
    "spin1x": 0.0,  # Primary spin x-component (dimensionless)
    "spin1y": 0.0,  # Primary spin y-component (dimensionless)
    "spin1z": 0.0,  # Primary spin z-component (dimensionless)
    "spin2x": 0.0,  # Secondary spin x-component (dimensionless)
    "spin2y": 0.0,  # Secondary spin y-component (dimensionless)
    "spin2z": 0.0,  # Secondary spin z-component (dimensionless)
    "inclination": 0.0,  # Inclination angle [rad] (0 = face-on)
    "phi_ref": 0.0,  # Reference phase [rad]
    "eccentricity": 0.0,  # Orbital eccentricity (0 = circular)
    "long_asc_nodes": 0.0,  # Longitude of ascending nodes [rad]
    "mean_per_ano": 0.0,  # Mean periastron anomaly [rad]
    "f_ref": 20.0,  # Reference frequency [Hz]
}


def _validate_injection(params: lal.Dict, index: int = 0) -> None:
    """Validate that an injection dict has all required fields.

    This should be called immediately after loading an injection to fail fast
    with a clear error message rather than failing later during waveform generation.

    Args:
        params: LAL dictionary with injection parameters
        index: Injection index for error messages

    Raises:
        ValueError: If a required field is missing
    """
    missing = []
    for field_name, field_type in REQUIRED_FIELDS.items():
        try:
            if field_type == str:
                lal.DictLookupStringValue(params, field_name)
            else:
                lal.DictLookupREAL8Value(params, field_name)
        except (KeyError, RuntimeError):
            missing.append(field_name)

    if missing:
        raise ValueError(
            f"Injection {index}: Missing required fields: {', '.join(missing)}"
        )


def _get_dict_real8(d: lal.Dict, key: str, default: Optional[float] = None) -> float:
    """Get REAL8 value from LAL dict.

    For optional fields, pass a default value (typically from OPTIONAL_FIELDS).
    For required fields that have already been validated via _validate_injection(),
    pass default=None - if the key is somehow missing, the LAL lookup value of
    0.0 will be returned (though this shouldn't happen for validated dicts).

    Args:
        d: LAL dictionary
        key: Key to look up
        default: Value to return if key not found (None uses LAL's default of 0.0)

    Returns:
        The value from the dict, or default if not found
    """
    try:
        return lal.DictLookupREAL8Value(d, key)
    except (KeyError, RuntimeError):
        return default if default is not None else 0.0


def _get_dict_string(d: lal.Dict, key: str, default: str = "") -> str:
    """Get string value from LAL dict with optional default.

    Args:
        d: LAL dictionary
        key: Key to look up
        default: Value to return if key not found

    Returns:
        The value from the dict, or default if not found
    """
    try:
        return lal.DictLookupStringValue(d, key)
    except (KeyError, RuntimeError):
        return default


def _load_xml_injections(filepath: str) -> List[lal.Dict]:
    """Load injections from LIGOLW XML file as LAL dicts.

    Converts XML SimInspiralTable rows to LAL dictionaries with parameters
    in SI units (masses in kg, distance in meters).

    Args:
        filepath: Path to XML file (can be .xml or .xml.gz)

    Returns:
        List of LAL dictionaries containing injection parameters
    """
    xmldoc = ligolw_utils.load_filename(filepath, verbose=False)
    sim_table = lsctables.SimInspiralTable.get_table(xmldoc)

    injections = []
    for row in sim_table:
        params = lal.CreateDict()

        # Masses - convert from solar masses to SI (kg)
        lal.DictInsertREAL8Value(params, "mass1", row.mass1 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params, "mass2", row.mass2 * lal.MSUN_SI)

        # Spins (dimensionless)
        lal.DictInsertREAL8Value(params, "spin1x", row.spin1x)
        lal.DictInsertREAL8Value(params, "spin1y", row.spin1y)
        lal.DictInsertREAL8Value(params, "spin1z", row.spin1z)
        lal.DictInsertREAL8Value(params, "spin2x", row.spin2x)
        lal.DictInsertREAL8Value(params, "spin2y", row.spin2y)
        lal.DictInsertREAL8Value(params, "spin2z", row.spin2z)

        # Distance - convert from Mpc to SI (meters)
        lal.DictInsertREAL8Value(params, "distance", row.distance * 1e6 * lal.PC_SI)

        # Angular quantities (already in radians)
        lal.DictInsertREAL8Value(params, "inclination", row.inclination)
        lal.DictInsertREAL8Value(params, "phi_ref", row.coa_phase)
        lal.DictInsertREAL8Value(params, "psi", row.polarization)

        # Orbital parameters
        lal.DictInsertREAL8Value(
            params, "long_asc_nodes", getattr(row, "long_asc_nodes", 0.0) or 0.0
        )
        lal.DictInsertREAL8Value(
            params, "eccentricity", getattr(row, "eccentricity", 0.0) or 0.0
        )
        lal.DictInsertREAL8Value(
            params, "mean_per_ano", getattr(row, "mean_per_ano", 0.0) or 0.0
        )

        # Sky position - LIGOLW uses longitude/latitude naming
        lal.DictInsertREAL8Value(params, "ra", row.longitude)
        lal.DictInsertREAL8Value(params, "dec", row.latitude)

        # Timing - coalescence time at geocenter
        geocent_end_time = (
            float(row.geocent_end_time) + float(row.geocent_end_time_ns) * 1e-9
        )
        lal.DictInsertREAL8Value(params, "t_co_gps", geocent_end_time)

        # Waveform model (required - validation will catch if missing)
        if row.waveform:
            lal.DictInsertStringValue(params, "approximant", row.waveform)
        lal.DictInsertREAL8Value(
            params,
            "f_ref",
            getattr(row, "f_lower", OPTIONAL_FIELDS["f_ref"])
            or OPTIONAL_FIELDS["f_ref"],
        )

        injections.append(params)

    # Validate all injections have required fields
    for i, params in enumerate(injections):
        _validate_injection(params, index=i)

    return injections


def _load_lal_h5_injections(filepath: str) -> List[lal.Dict]:
    """Load injections from LAL-format HDF5 file.

    Uses LALSimulation's SimInspiralInjectionSequenceFromH5File to load
    injections. The returned dicts are already in the correct format
    with SI units.

    Args:
        filepath: Path to LAL-format HDF5 file

    Returns:
        List of LAL dictionaries containing injection parameters

    Raises:
        ValueError: If any injection is missing required fields
    """
    seq = lalsim.SimInspiralInjectionSequenceFromH5File(filepath)
    injections = [lal.DictSequenceGet(seq, i) for i in range(seq.length)]

    # Validate all injections have required fields
    for i, params in enumerate(injections):
        _validate_injection(params, index=i)

    return injections


def load_injections(filepath: str) -> List[lal.Dict]:
    """Load injections from XML or LAL H5 file with auto-detection.

    Supports LIGOLW XML (.xml, .xml.gz) and LAL HDF5 (.h5, .hdf5, .hdf) formats.
    The HDF5 format must be the official LAL format with 'cbc_waveform_params' group.

    Args:
        filepath: Path to injection file

    Returns:
        List of LAL dictionaries containing injection parameters in SI units
    """
    if filepath.endswith((".xml", ".xml.gz")):
        return _load_xml_injections(filepath)
    elif filepath.endswith((".hdf5", ".h5", ".hdf")):
        return _load_lal_h5_injections(filepath)
    else:
        # Try XML first, then LAL H5
        try:
            return _load_xml_injections(filepath)
        except Exception:
            return _load_lal_h5_injections(filepath)


def estimate_waveform_duration(params: lal.Dict, f_min: float) -> tuple[float, float]:
    """Estimate the duration of a waveform before and after merger.

    Args:
        params: LAL dict with injection parameters (masses in SI units)
        f_min: Minimum frequency for waveform generation (Hz)

    Returns:
        Tuple of (pre_merger_duration, post_merger_duration) in seconds
    """
    # Masses are already in SI units from the dict
    m1_si = _get_dict_real8(params, "mass1")
    m2_si = _get_dict_real8(params, "mass2")
    spin1z = _get_dict_real8(params, "spin1z")
    spin2z = _get_dict_real8(params, "spin2z")

    mtotal_si = m1_si + m2_si

    # Chirp time (inspiral duration)
    try:
        tchirp = lalsim.SimInspiralChirpTimeBound(f_min, m1_si, m2_si, spin1z, spin2z)
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
        final_spin = min(abs(spin1z) + abs(spin2z), 0.998)
        tring = lalsim.SimInspiralRingdownTimeBound(mtotal_si, final_spin)
    except Exception:
        tring = 0.5  # Conservative estimate

    # Add safety margins
    pre_merger = tchirp + 1.0  # Extra second before
    post_merger = tmerge + tring + 0.5  # Extra half second after

    return pre_merger, post_merger


def generate_waveform_td(
    params: lal.Dict,
    sample_rate: int,
    f_min: float,
    approximant_override: Optional[str] = None,
) -> tuple[lal.REAL8TimeSeries, lal.REAL8TimeSeries]:
    """Generate time-domain h+, hx waveforms from LAL parameter dict.

    Uses LALSimulation's unified generator interface which automatically
    handles both time-domain and frequency-domain approximants. All
    parameters from the input dict are passed through to the generator.

    Args:
        params: LAL dict with injection parameters (masses/distance in SI units)
        sample_rate: Sample rate in Hz
        f_min: Minimum frequency in Hz
        approximant_override: Override the approximant from the dict

    Returns:
        Tuple of (hp, hc) as LAL REAL8TimeSeries objects
    """
    # Get or override approximant (validated at load time)
    approximant_str = approximant_override or _get_dict_string(params, "approximant")
    approximant = lalsim.GetApproximantFromString(approximant_str)

    # Add runtime parameters to dict (these are not stored in injection files)
    lalsim.SimInspiralWaveformParamsInsertDeltaT(params, 1.0 / sample_rate)
    lalsim.SimInspiralWaveformParamsInsertF22Start(params, f_min)

    # Use f_ref from dict if present, otherwise use f_min
    f_ref = _get_dict_real8(params, "f_ref", f_min)
    lalsim.SimInspiralWaveformParamsInsertF22Ref(params, f_ref)

    # Create generator and add conditioning for FD->TD conversion
    gen = lalsim.SimInspiralChooseGenerator(approximant, None)
    lalsim.SimInspiralGeneratorAddStandardConditioning(gen)

    # Generate TD waveform - all parameters from dict are used
    hp, hc = lalsim.SimInspiralGenerateTDWaveform(params, gen)

    return hp, hc


def project_to_detector(
    hp: lal.REAL8TimeSeries,
    hc: lal.REAL8TimeSeries,
    params: lal.Dict,
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
        params: LAL dict with sky position and polarization
        ifo: Interferometer prefix (H1, L1, V1, etc.)

    Returns:
        Detector strain as REAL8TimeSeries
    """
    # Get detector
    detector = lal.cached_detector_by_prefix[ifo]

    # Extract sky position and polarization (validated at load time)
    ra = _get_dict_real8(params, "ra")
    dec = _get_dict_real8(params, "dec")
    psi = _get_dict_real8(params, "psi")

    # Use LALSimulation's accurate detector strain function
    strain = lalsim.SimDetectorStrainREAL8TimeSeries(
        hp,
        hc,
        ra,
        dec,
        psi,
        detector,
    )

    return strain


@dataclass
class CachedWaveform:
    """Cached waveform data for an injection.

    Each detector has its own LAL REAL8TimeSeries which includes:
    - epoch: GPS time of first sample (accounts for light travel time)
    - deltaT: Sample spacing (1/sample_rate)
    - data: The strain values

    Using LAL TimeSeries preserves sub-sample timing information needed
    for proper interpolation when injecting into buffers.
    """

    injection_id: int
    strain: Dict[str, lal.REAL8TimeSeries]  # Per-IFO strain TimeSeries {ifo: series}

    def get_end_gps(self, ifo: str) -> float:
        """Get the GPS end time for a specific IFO."""
        ts = self.strain[ifo]
        epoch = float(ts.epoch)
        duration = ts.data.length * ts.deltaT
        return epoch + duration

    def get_max_end_gps(self) -> float:
        """Get the latest end GPS time across all IFOs."""
        return max(self.get_end_gps(ifo) for ifo in self.strain)


class WaveformCache:
    """Manages waveform generation and caching.

    Waveforms are generated on-demand and cached until they're fully
    consumed (i.e., the pipeline has moved past the waveform's end time).
    """

    def __init__(
        self,
        injections: List[lal.Dict],
        ifos: List[str],
        sample_rate: int,
        f_min: float,
        approximant_override: Optional[str] = None,
    ):
        """Initialize the waveform cache.

        Args:
            injections: List of LAL dicts with injection parameters
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
        for params in injections:
            pre_dur, post_dur = estimate_waveform_duration(params, f_min)
            t_co_gps = _get_dict_real8(params, "t_co_gps")
            start = t_co_gps - pre_dur
            end = t_co_gps + post_dur
            self._injection_windows.append((start, end))

    def get_overlapping_injections(self, buf_start: float, buf_end: float) -> List[int]:
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
        params = self.injections[inj_id]

        # Generate h+, hx waveforms
        hp, hc = generate_waveform_td(
            params, self.sample_rate, self.f_min, self.approximant_override
        )

        # Get coalescence time from dict (validated at load time) and shift
        # waveform epochs to absolute GPS time before projection. The generated
        # waveforms have epochs relative to coalescence (typically negative).
        t_co_gps = _get_dict_real8(params, "t_co_gps")
        hp.epoch += t_co_gps
        hc.epoch += t_co_gps

        # Project onto each detector
        # Each detector has different timing due to light travel time delays
        # applied by SimDetectorStrainREAL8TimeSeries
        strain_dict: Dict[str, lal.REAL8TimeSeries] = {}

        for ifo in self.ifos:
            strain_dict[ifo] = project_to_detector(hp, hc, params, ifo)

        # Cache the waveform
        self.cache[inj_id] = CachedWaveform(
            injection_id=inj_id,
            strain=strain_dict,
        )

    def add_injection_to_target(
        self,
        inj_id: int,
        ifo: str,
        target: lal.REAL8TimeSeries,
    ) -> None:
        """Add an injection waveform to a target buffer using proper interpolation.

        Uses SimAddInjectionREAL8TimeSeries which performs sub-sample
        re-interpolation in the frequency domain to properly align the
        injection epoch to integer sample boundaries in the target.

        Args:
            inj_id: Injection index
            ifo: Interferometer prefix
            target: Target LAL REAL8TimeSeries to add injection into (modified in place)
        """
        if inj_id not in self.cache:
            self._generate_and_cache(inj_id)

        cached = self.cache[inj_id]
        source_strain = cached.strain[ifo]

        # SimAddInjectionREAL8TimeSeries handles:
        # - Finding overlapping region based on epochs
        # - Sub-sample interpolation for proper alignment
        # - Adding only the overlapping portion to target
        # The source may be modified (padded for alignment) but remains reusable
        lalsim.SimAddInjectionREAL8TimeSeries(target, source_strain, None)

    def cleanup_expired(self, current_gps: float) -> None:
        """Remove waveforms that are fully consumed.

        A waveform is expired when all IFOs have moved past it.

        Args:
            current_gps: Current GPS time of the pipeline
        """
        expired = []
        for k, v in self.cache.items():
            # Use the latest end time across all IFOs
            if v.get_max_end_gps() < current_gps:
                expired.append(k)
        for k in expired:
            del self.cache[k]


@dataclass
class SimInspiralSource(TSSource):
    """Source element that generates GW waveforms from an injection file.

    Reads injection parameters from XML (LIGOLW SimInspiralTable) or HDF5 files
    and generates time-domain waveforms projected onto each detector with proper
    time delays, antenna patterns, and phase corrections using LALSimulation.

    All injection parameters are stored as LAL dictionaries, which allows
    pass-through of any LAL-supported parameters including tidal deformability,
    higher-order modes, and approximant-specific settings.

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
    _injections: List[lal.Dict] = field(init=False, repr=False, default_factory=list)
    _channel_dict: Dict[str, str] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Initialize the source."""
        if self.injection_file is None:
            raise ValueError("injection_file must be specified")

        if self.ifos is None:
            self.ifos = ["H1", "L1"]

        # Create channel names for source pads
        self._channel_dict = {ifo: f"{ifo}:INJ-STRAIN" for ifo in self.ifos}
        self.source_pad_names = list(self._channel_dict.values())

        # Load injections as LAL dicts
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
        for _ifo, channel in self._channel_dict.items():
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
        # SGNTS Offset is normalized to Offset.MAX_RATE internally, regardless of
        # the source's sample rate. This is a core SGNTS design choice.
        stride_max = Offset.sample_stride(Offset.MAX_RATE)
        buf_start = buffer.offset / stride_max
        buf_end = buffer.end_offset / stride_max

        # Get number of samples from the buffer's expected shape
        num_samples = buffer.shape[0]

        # Create a LAL REAL8TimeSeries as target for injections
        # This preserves exact GPS epoch for proper sub-sample interpolation
        target = lal.CreateREAL8TimeSeries(
            f"{ifo}:STRAIN",
            lal.LIGOTimeGPS(buf_start),
            0.0,  # f0
            1.0 / self.sample_rate,  # deltaT
            lal.StrainUnit,
            num_samples,
        )
        target.data.data[:] = 0.0

        # Find and add all overlapping injections
        # SimAddInjectionREAL8TimeSeries handles sub-sample interpolation
        overlapping = self._waveform_cache.get_overlapping_injections(
            buf_start, buf_end
        )
        for inj_id in overlapping:
            self._waveform_cache.add_injection_to_target(inj_id, ifo, target)

        # Copy result to output buffer
        buffer.set_data(target.data.data)
        return frame

    def internal(self) -> None:
        """Internal processing - cleanup expired waveforms."""
        super().internal()

        # Cleanup expired waveforms from cache
        # Convert from Offset to GPS seconds
        # SGNTS Offset is normalized to Offset.MAX_RATE internally
        stride_max = Offset.sample_stride(Offset.MAX_RATE)
        current_gps = self.current_end / stride_max
        self._waveform_cache.cleanup_expired(current_gps)
