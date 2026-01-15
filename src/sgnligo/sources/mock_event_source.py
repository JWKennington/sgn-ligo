"""A source element to generate mock GW events with GraceDB-compatible coinc XML.

This module provides the MockGWEventSource class which generates realistic mock
gravitational wave detection events with proper SNR, timing, and phase calculations.
Events are output as coinc XML files (in-memory bytes) on pipeline-named pads
(sgnl, pycbc, mbta, spiir) with configurable latencies to simulate real pipelines.
"""

from __future__ import annotations

import io
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import lal
import lal.series as lalseries
import numpy
from igwn_ligolw import ligolw, lsctables
from igwn_ligolw import utils as ligolw_utils
from igwn_ligolw.utils import process as ligolw_process
from lal import LIGOTimeGPS

# Import ligolw_param with deprecation warning suppressed
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="igwn_ligolw.param")
    from igwn_ligolw.ligolw import Param as ligolw_param

from sgn.base import SourcePad
from sgn.frames import Frame
from sgnts.base import TSSource

from sgnligo.base import now
from sgnligo.psd import HorizonDistance, effective_distance_factor, fake_gwdata_psd

# =============================================================================
# Constants
# =============================================================================

# State College, PA coordinates (default overhead position)
STATE_COLLEGE_LAT_RAD = math.radians(40.7934)
STATE_COLLEGE_LON_RAD = math.radians(-77.8600)

# Source type parameters: masses in solar masses, distance in Mpc
SOURCE_TYPE_PARAMS = {
    "bns": {
        "mass1_range": (1.0, 2.5),
        "mass2_range": (1.0, 2.5),
        "distance_range": (10.0, 300.0),
    },
    "nsbh": {
        "mass1_range": (3.0, 30.0),
        "mass2_range": (1.0, 2.5),
        "distance_range": (50.0, 500.0),
    },
    "bbh": {
        "mass1_range": (5.0, 80.0),
        "mass2_range": (5.0, 80.0),
        "distance_range": (100.0, 2000.0),
    },
}

# Default pipeline latencies (mean, std) in seconds
DEFAULT_PIPELINE_LATENCIES = {
    "sgnl": (6.0, 1.0),
    "pycbc": (30.0, 5.0),
    "mbta": (10.0, 2.0),
    "spiir": (8.0, 1.5),
}

# Effective bandwidth for timing uncertainty (Hz)
# Typical values: BBH ~50 Hz, BNS ~100 Hz
EFFECTIVE_BANDWIDTH_HZ = 80.0


# =============================================================================
# LIGOLW Content Handler Setup
# =============================================================================

# Note: igwn_ligolw now handles custom table support automatically
# when lsctables is imported - no decorators needed


# =============================================================================
# Internal Data Classes
# =============================================================================


@dataclass
class _SingleTrigger:
    """Per-detector trigger parameters."""

    ifo: str
    end_time: float  # GPS seconds
    snr: float
    coa_phase: float  # radians
    mass1: float  # solar masses
    mass2: float  # solar masses
    spin1z: float = 0.0
    spin2z: float = 0.0
    eff_distance: Optional[float] = None  # Mpc
    channel: Optional[str] = None

    @property
    def end_time_int(self) -> int:
        return int(self.end_time)

    @property
    def end_time_ns(self) -> int:
        return int((self.end_time - int(self.end_time)) * 1e9)

    @property
    def mtotal(self) -> float:
        return self.mass1 + self.mass2

    @property
    def eta(self) -> float:
        return (self.mass1 * self.mass2) / (self.mtotal**2)

    @property
    def mchirp(self) -> float:
        return self.mtotal * (self.eta ** (3.0 / 5.0))


@dataclass
class _CoincEvent:
    """Coincident event parameters."""

    event_id: int
    t_co_gps: float  # Geocentric coalescence time
    triggers: List[_SingleTrigger]
    far: float = 1e-10  # False alarm rate in Hz
    source_type: str = "bns"

    @property
    def network_snr(self) -> float:
        return math.sqrt(sum(t.snr**2 for t in self.triggers))

    @property
    def ifos(self) -> List[str]:
        return [t.ifo for t in self.triggers]


# =============================================================================
# Physics Helper Functions
# =============================================================================


def _calculate_overhead_ra(gps_time: float, longitude_rad: float) -> float:
    """Calculate right ascension for a source directly overhead at a location.

    For a source to be at zenith, its RA must equal the local sidereal time.
    LST = GMST + longitude (where longitude is positive east, negative west)

    Args:
        gps_time: GPS time of observation
        longitude_rad: Observer longitude in radians (negative for west)

    Returns:
        Right ascension in radians, normalized to [0, 2π)
    """
    gmst = lal.GreenwichMeanSiderealTime(gps_time)
    ra = (gmst + longitude_rad) % (2 * math.pi)
    return ra


def _compute_optimal_snr(
    mass1_msun: float,
    mass2_msun: float,
    distance_mpc: float,
    ra: float,
    dec: float,
    psi: float,
    inclination: float,
    t_co_gps: float,
    ifo: str,
    psd: lal.REAL8FrequencySeries,
    f_min: float = 20.0,
    f_max: float = 1024.0,
) -> float:
    """Compute expected optimal SNR at a detector from source parameters.

    Uses horizon distance scaling for efficiency:
    SNR = D_horizon(SNR=1) / D_effective

    Args:
        mass1_msun: Primary mass in solar masses
        mass2_msun: Secondary mass in solar masses
        distance_mpc: Luminosity distance in Mpc
        ra: Right ascension in radians
        dec: Declination in radians
        psi: Polarization angle in radians
        inclination: Orbital inclination in radians
        t_co_gps: Coalescence GPS time
        ifo: Detector name (H1, L1, V1)
        psd: Power spectral density
        f_min: Minimum frequency for integration (Hz)
        f_max: Maximum frequency for integration (Hz)

    Returns:
        Expected optimal SNR at the detector
    """
    # Create HorizonDistance calculator
    horizon_calc = HorizonDistance(
        f_min=f_min,
        f_max=f_max,
        delta_f=psd.deltaF,
        m1=mass1_msun,
        m2=mass2_msun,
    )

    # Get horizon distance at SNR=1 (in Mpc)
    d_horizon_snr1_mpc, _ = horizon_calc(psd, snr=1.0)

    # Compute antenna response
    detector = lal.cached_detector_by_prefix[ifo]
    gps_time = LIGOTimeGPS(t_co_gps)
    fplus, fcross = lal.ComputeDetAMResponse(
        detector.response, ra, dec, psi, gps_time.gpsSeconds
    )

    # Effective distance factor (accounts for inclination + antenna pattern)
    d_eff_factor = effective_distance_factor(inclination, fplus, fcross)

    # Effective distance in Mpc
    d_eff_mpc = distance_mpc * d_eff_factor

    # Optimal SNR = D_horizon / D_effective
    if d_eff_mpc > 0:
        snr_optimal = d_horizon_snr1_mpc / d_eff_mpc
    else:
        snr_optimal = 0.0

    return snr_optimal


def _compute_time_delays(
    ra: float,
    dec: float,
    t_co_gps: float,
    ifos: List[str],
) -> Dict[str, float]:
    """Compute time delays from geocenter to each detector.

    The coalescence time at each detector is: t_det = t_geo + dt_det

    Args:
        ra: Right ascension in radians
        dec: Declination in radians
        t_co_gps: Geocentric coalescence GPS time
        ifos: List of detector names

    Returns:
        Dictionary mapping IFO name to time delay in seconds
    """
    gps_time = LIGOTimeGPS(t_co_gps)
    time_delays = {}

    for ifo in ifos:
        detector = lal.cached_detector_by_prefix[ifo]
        # Time delay from Earth center to detector (seconds)
        dt = lal.TimeDelayFromEarthCenter(
            detector.location,  # Earth-fixed XYZ coordinates (meters)
            ra,
            dec,
            gps_time,
        )
        time_delays[ifo] = dt

    return time_delays


def _compute_phases(
    ra: float,
    dec: float,
    psi: float,
    inclination: float,
    phi_geo: float,
    t_co_gps: float,
    time_delays: Dict[str, float],
    ifos: List[str],
    f_co: float = 100.0,
) -> Dict[str, float]:
    """Compute coalescence phases at each detector.

    phi_det = phi_geo + 2*pi*f_co*dt + phi_antenna

    Args:
        ra: Right ascension in radians
        dec: Declination in radians
        psi: Polarization angle in radians
        inclination: Orbital inclination in radians
        phi_geo: Geocentric reference phase in radians
        t_co_gps: Coalescence GPS time
        time_delays: Time delays per detector
        ifos: List of detector names
        f_co: Coalescence frequency in Hz

    Returns:
        Dictionary mapping IFO name to coalescence phase in radians
    """
    gps_time = LIGOTimeGPS(t_co_gps)
    phases = {}
    cos_i = math.cos(inclination)
    h_plus_amp = (1 + cos_i**2) / 2
    h_cross_amp = cos_i

    for ifo in ifos:
        detector = lal.cached_detector_by_prefix[ifo]

        # Time delay phase contribution
        omega_gw = 2 * math.pi * f_co
        phi_time_delay = omega_gw * time_delays[ifo]

        # Antenna pattern phase contribution
        fplus, fcross = lal.ComputeDetAMResponse(
            detector.response, ra, dec, psi, gps_time.gpsSeconds
        )

        # Phase from antenna mixing
        phi_antenna = (
            math.atan2(fcross * h_cross_amp, fplus * h_plus_amp)
            if (fplus * h_plus_amp != 0 or fcross * h_cross_amp != 0)
            else 0.0
        )

        # Total phase at detector (wrapped to [0, 2pi))
        phi_det = (phi_geo + phi_time_delay + phi_antenna) % (2 * math.pi)
        phases[ifo] = phi_det

    return phases


def _add_noise_fluctuations(
    snr_true: float,
    t_true: float,
    phi_true: float,
    sigma_f: float = EFFECTIVE_BANDWIDTH_HZ,
) -> Tuple[float, float, float]:
    """Add realistic noise fluctuations to recovered parameters.

    Based on Fisher matrix / Cramer-Rao bounds:
    - Timing: sigma_t ~ 1 / (2*pi*sigma_f*SNR)
    - Phase: sigma_phi ~ 1 / SNR
    - SNR measurement: sigma_rho ~ 1 (unit Gaussian, independent of SNR)

    Args:
        snr_true: True optimal SNR
        t_true: True coalescence time
        phi_true: True coalescence phase
        sigma_f: Effective bandwidth in Hz

    Returns:
        Tuple of (snr_measured, t_measured, phi_measured)
    """
    # SNR measurement noise is approximately unit Gaussian
    snr_measured = snr_true + numpy.random.normal(0, 1.0)

    # Timing noise
    if snr_true > 0:
        sigma_t = 1.0 / (2 * math.pi * sigma_f * snr_true)
        t_measured = t_true + numpy.random.normal(0, sigma_t)
    else:
        t_measured = t_true

    # Phase noise
    if snr_true > 0:
        sigma_phi = 1.0 / snr_true
        phi_measured = (phi_true + numpy.random.normal(0, sigma_phi)) % (2 * math.pi)
    else:
        phi_measured = phi_true

    return snr_measured, t_measured, phi_measured


def _apply_template_mismatch(
    snr_optimal: float,
    mass1_true: float,
    mass2_true: float,
    min_match: float = 0.97,
) -> Tuple[float, float, float]:
    """Apply template bank mismatch effects.

    Models the discrete template bank coverage:
    - Match sampled from Uniform(min_match, 1.0)
    - SNR recovery: SNR_recovered = match * SNR_optimal
    - Masses biased proportionally to sqrt(mismatch)

    Args:
        snr_optimal: Optimal SNR before mismatch
        mass1_true: True primary mass
        mass2_true: True secondary mass
        min_match: Template bank minimum match

    Returns:
        Tuple of (snr_recovered, mass1_template, mass2_template)
    """
    # Sample match from template bank coverage distribution
    match = numpy.random.uniform(min_match, 1.0)

    # SNR reduction from mismatch
    snr_recovered = match * snr_optimal

    # Mass bias from template mismatch
    mchirp_true = (mass1_true * mass2_true) ** (3.0 / 5.0) / (
        mass1_true + mass2_true
    ) ** (1.0 / 5.0)
    mismatch = 1 - match

    # Template grid spacing causes chirp mass error ~ sqrt(mismatch)
    mchirp_bias = (
        numpy.random.normal(0, 0.01 * math.sqrt(mismatch / 0.03)) if mismatch > 0 else 0
    )
    mchirp_template = mchirp_true * (1 + mchirp_bias)

    # Convert back to component masses (assuming same mass ratio)
    eta = mass1_true * mass2_true / (mass1_true + mass2_true) ** 2
    mtotal_template = mchirp_template / (eta ** (3.0 / 5.0))
    mass1_template = mtotal_template * (1 + math.sqrt(max(0, 1 - 4 * eta))) / 2
    mass2_template = mtotal_template - mass1_template

    return snr_recovered, mass1_template, mass2_template


# =============================================================================
# XML Generation Helper Functions
# =============================================================================


def _add_table_with_n_rows(xmldoc, tblcls, n):
    """Add a table to xmldoc with n rows initialized to default values."""
    table = lsctables.New(tblcls)
    for _ in range(n):
        row = table.RowType()
        for col, _type in table.validcolumns.items():
            col = col.split(":")[-1]
            setattr(row, col, "" if _type == "lstring" else 0)
        table.append(row)
    xmldoc.childNodes[-1].appendChild(table)
    return table


def _ns_to_gps(ns):
    """Convert nanoseconds to LIGOTimeGPS."""
    return LIGOTimeGPS(int(ns // 1_000_000_000), int(ns % 1_000_000_000))


def _compute_waveform_acf(
    mass1: float,
    mass2: float,
    psd: lal.REAL8FrequencySeries,
    sample_rate: float = 2048.0,
    f_min: float = 20.0,
    f_max: float = 1024.0,
) -> numpy.ndarray:
    """Compute the autocorrelation function of the matched filter output.

    The ACF is the inverse FFT of |h(f)|^2 / S_n(f), which gives the
    expected shape of the SNR time series around the peak.

    Args:
        mass1: Primary mass in solar masses
        mass2: Secondary mass in solar masses
        psd: Power spectral density
        sample_rate: Output sample rate in Hz
        f_min: Minimum frequency for waveform generation
        f_max: Maximum frequency for waveform generation

    Returns:
        Complex ACF array normalized to unit peak magnitude
    """
    import lalsimulation

    # Generate waveform at the PSD's frequency resolution
    delta_f = psd.deltaF
    hp, _hc = lalsimulation.SimInspiralFD(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        0.0,
        0.0,
        0.0,  # spin1
        0.0,
        0.0,
        0.0,  # spin2
        1.0,  # distance (1 meter)
        0.0,  # inclination
        0.0,  # reference phase
        0.0,  # longitude of ascending nodes
        0.0,  # eccentricity
        0.0,  # mean anomaly
        delta_f,
        f_min,
        f_max + delta_f,
        100.0,  # reference frequency
        None,  # LAL dictionary
        lalsimulation.GetApproximantFromString("IMRPhenomD"),
    )

    # Build |h(f)|^2 / S_n(f) integrand
    n_freqs = int(f_max / delta_f) + 1
    integrand = numpy.zeros(n_freqs, dtype=numpy.complex128)

    # Map waveform frequencies to PSD frequencies
    k_min = int(f_min / delta_f)
    k_max = min(n_freqs, hp.data.length)

    for k in range(k_min, k_max):
        f = k * delta_f
        psd_idx = int((f - psd.f0) / psd.deltaF)
        if 0 <= psd_idx < psd.data.length and psd.data.data[psd_idx] > 0:
            # h(f) is complex, we want h(f) * conj(h(f)) / S_n(f) for the ACF
            # But for the shape we just need |h(f)|^2 / S_n(f)
            integrand[k] = numpy.abs(hp.data.data[k]) ** 2 / psd.data.data[psd_idx]

    # IFFT to get ACF in time domain
    # The ACF is real and symmetric, but we compute it as complex for generality
    n_time = int(sample_rate / delta_f)  # Number of time samples
    acf_full = numpy.fft.irfft(integrand, n=n_time)

    # Normalize to unit peak
    peak_idx = numpy.argmax(numpy.abs(acf_full))
    if numpy.abs(acf_full[peak_idx]) > 0:
        acf_full = acf_full / numpy.abs(acf_full[peak_idx])

    return acf_full.astype(numpy.complex64)


def _create_snr_timeseries(
    snr_peak: complex,
    t_peak: float,
    mass1: float,
    mass2: float,
    psd: lal.REAL8FrequencySeries,
    sample_rate: float = 2048.0,
    half_duration: float = 0.1,
) -> lal.COMPLEX8TimeSeries:
    """Create an SNR time series using the waveform autocorrelation function.

    The ACF gives the physically correct shape for the matched filter SNR
    time series, capturing the inspiral-merger-ringdown structure.

    Args:
        snr_peak: Complex SNR at peak (|snr_peak| = SNR, arg = phase)
        t_peak: GPS time of peak
        mass1: Primary mass in solar masses
        mass2: Secondary mass in solar masses
        psd: Power spectral density for the detector
        sample_rate: Sample rate in Hz (default: 2048)
        half_duration: Duration on each side of peak in seconds (default: 0.1)

    Returns:
        LAL COMPLEX8TimeSeries
    """
    delta_t = 1.0 / sample_rate
    half_length = int(half_duration * sample_rate)
    length = 2 * half_length + 1

    # Compute the ACF for this template
    acf_full = _compute_waveform_acf(mass1, mass2, psd, sample_rate)

    # Extract window around peak (ACF peak is at index 0 after irfft)
    # We need samples from -half_length to +half_length
    acf_window = numpy.zeros(length, dtype=numpy.complex64)

    # Positive lags (0 to half_length) come from start of acf_full
    acf_window[half_length : half_length + half_length + 1] = acf_full[
        : half_length + 1
    ]
    # Negative lags (-half_length to -1) come from end of acf_full
    acf_window[:half_length] = acf_full[-half_length:]

    # Scale by peak SNR (magnitude and phase)
    snr_data = snr_peak * acf_window

    # Create LAL time series
    epoch = t_peak - half_length * delta_t
    ts = lal.CreateCOMPLEX8TimeSeries(
        name="snr",
        epoch=LIGOTimeGPS(epoch),
        f0=0.0,
        deltaT=delta_t,
        sampleUnits=lal.DimensionlessUnit,
        length=length,
    )
    ts.data.data[:] = snr_data

    return ts


def _truncate_psd(
    psd: lal.REAL8FrequencySeries,
    f_max: float = 1024.0,
) -> lal.REAL8FrequencySeries:
    """Truncate a PSD to a maximum frequency.

    Args:
        psd: Input PSD
        f_max: Maximum frequency to keep (Hz)

    Returns:
        Truncated PSD
    """
    n_bins = int((f_max - psd.f0) / psd.deltaF) + 1
    n_bins = min(n_bins, psd.data.length)

    truncated = lal.CreateREAL8FrequencySeries(
        psd.name,
        psd.epoch,
        psd.f0,
        psd.deltaF,
        psd.sampleUnits,
        n_bins,
    )
    truncated.data.data[:] = psd.data.data[:n_bins]

    return truncated


def _build_coinc_xmldoc(
    event: _CoincEvent,
    pipeline: str,
    psds: Dict[str, lal.REAL8FrequencySeries],
    include_snr_series: bool = True,
) -> ligolw.Document:
    """Build a complete LIGOLW coinc document for a mock event.

    Creates all required tables for GraceDB submission:
    ProcessTable, TimeSlideTable, CoincDefTable, CoincTable,
    CoincInspiralTable, SnglInspiralTable, CoincMapTable,
    and optionally COMPLEX8TimeSeries for SNR snippets.

    Args:
        event: CoincEvent with triggers for each detector
        pipeline: Pipeline name (sgnl, pycbc, mbta, spiir)
        psds: Dictionary of PSDs per detector
        include_snr_series: Whether to include SNR time series

    Returns:
        Complete LIGOLW Document
    """
    # Initialize document
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())

    # Add process table
    ligolw_process.register_to_xmldoc(
        xmldoc,
        f"mock-{pipeline}",
        {},
        instruments=event.ifos,
        is_online=True,
    )
    process_id = lsctables.ProcessTable.get_table(xmldoc)[0].process_id

    # Add time slide table (zero offsets)
    time_slide_table = _add_table_with_n_rows(
        xmldoc, lsctables.TimeSlideTable, len(event.ifos)
    )
    for n, ifo in enumerate(event.ifos):
        time_slide_table[n].instrument = ifo
        time_slide_table[n].process_id = process_id
        time_slide_table[n].offset = 0.0

    # Add coinc def table
    coinc_def_table = _add_table_with_n_rows(xmldoc, lsctables.CoincDefTable, 1)
    coinc_def_table[0].description = "sngl_inspiral<-->sngl_inspiral coincidences"
    coinc_def_table[0].search = "inspiral"

    # Add sngl inspiral table and coinc map table
    sngl_inspiral_table = _add_table_with_n_rows(
        xmldoc, lsctables.SnglInspiralTable, len(event.triggers)
    )
    coinc_map_table = _add_table_with_n_rows(
        xmldoc, lsctables.CoincMapTable, len(event.triggers)
    )

    for n, trigger in enumerate(event.triggers):
        row = sngl_inspiral_table[n]
        row.event_id = n
        row.process_id = process_id
        row.ifo = trigger.ifo
        row.search = "inspiral"
        row.channel = trigger.channel or f"{trigger.ifo}:GDS-CALIB_STRAIN"

        # Timing
        row.end_time = trigger.end_time_int
        row.end_time_ns = trigger.end_time_ns

        # Masses
        row.mass1 = trigger.mass1
        row.mass2 = trigger.mass2
        row.mtotal = trigger.mtotal
        row.mchirp = trigger.mchirp
        row.eta = trigger.eta

        # Spins
        row.spin1x = 0.0
        row.spin1y = 0.0
        row.spin1z = trigger.spin1z
        row.spin2x = 0.0
        row.spin2y = 0.0
        row.spin2z = trigger.spin2z

        # SNR and phase
        row.snr = trigger.snr
        row.coa_phase = trigger.coa_phase

        if trigger.eff_distance is not None:
            row.eff_distance = trigger.eff_distance

        # Coinc map entry
        coinc_map_table[n].event_id = row.event_id
        coinc_map_table[n].table_name = "sngl_inspiral"

        # Add SNR time series if requested
        if include_snr_series and trigger.ifo in psds:
            snr_complex = trigger.snr * numpy.exp(1j * trigger.coa_phase)
            ts = _create_snr_timeseries(
                snr_complex,
                trigger.end_time,
                trigger.mass1,
                trigger.mass2,
                psds[trigger.ifo],
            )
            snr_ts_element = lalseries.build_COMPLEX8TimeSeries(ts)
            snr_ts_element.appendChild(
                ligolw_param.from_pyvalue("event_id", row.event_id)
            )
            xmldoc.childNodes[-1].appendChild(snr_ts_element)

    # Add coinc inspiral table
    coinc_inspiral_table = _add_table_with_n_rows(
        xmldoc, lsctables.CoincInspiralTable, 1
    )
    ref_trigger = event.triggers[0]
    coinc_inspiral_table[0].end_time = int(event.t_co_gps)
    coinc_inspiral_table[0].end_time_ns = int(
        (event.t_co_gps - int(event.t_co_gps)) * 1e9
    )
    coinc_inspiral_table[0].mass = ref_trigger.mtotal
    coinc_inspiral_table[0].mchirp = ref_trigger.mchirp
    coinc_inspiral_table[0].snr = event.network_snr
    coinc_inspiral_table[0].combined_far = event.far
    coinc_inspiral_table[0].false_alarm_rate = 0.0
    coinc_inspiral_table[0].ifos = ",".join(sorted(event.ifos))

    # Add coinc event table
    coinc_event_table = _add_table_with_n_rows(xmldoc, lsctables.CoincTable, 1)
    coinc_event_table[0].instruments = ",".join(sorted(event.ifos))
    coinc_event_table[0].process_id = process_id
    coinc_event_table[0].nevents = len(event.triggers)

    # Add PSDs to document (truncated to 1024 Hz for smaller file size)
    truncated_psds = {ifo: _truncate_psd(psd) for ifo, psd in psds.items()}
    lal.series.make_psd_xmldoc(truncated_psds, xmldoc.childNodes[-1])

    return xmldoc


def _serialize_xmldoc(xmldoc: ligolw.Document) -> bytes:
    """Serialize LIGOLW document to bytes."""
    buffer = io.BytesIO()
    ligolw_utils.write_fileobj(xmldoc, buffer)
    return buffer.getvalue()


# =============================================================================
# Main Source Element
# =============================================================================


@dataclass
class MockGWEventSource(TSSource):
    """Source element to generate mock GW events with GraceDB-compatible coinc XML.

    Generates realistic gravitational wave detection events with proper SNR,
    timing, and phase calculations. Events are output as coinc XML files
    (in-memory bytes) on pipeline-named pads (sgnl, pycbc, mbta, spiir) with
    configurable latencies to simulate real pipeline behavior.

    The source synchronizes with real GPS time when real_time=True and generates
    events at a configurable cadence (default: every 20 seconds).

    Args:
        event_cadence: Seconds between coalescence times (default: 20.0)
        t0: Start GPS time. If None, uses current GPS time.
        end: End GPS time. If None and real_time=True, runs indefinitely.
        duration: Duration in seconds. Cannot combine with end.
        ifos: List of detectors (default: ["H1", "L1", "V1"])
        source_types: List of source types to generate (default: ["bns", "nsbh", "bbh"])
        source_weights: Weights for source type selection (default: [0.6, 0.2, 0.2])
        pipeline_latencies: Dict mapping pipeline name to (mean, std) latency in seconds
        template_min_match: Template bank minimum match (default: 0.97)
        snr_threshold: Minimum SNR for detection (default: 4.0)
        sky_position: "overhead" for State College overhead, "random" for uniform sky
        latitude_rad: Observer latitude for overhead position (default: State College)
        longitude_rad: Observer longitude for overhead position (default: State College)
        include_snr_series: Include SNR time series in XML (default: True)
        real_time: If True, sync with wall clock time
        verbose: If True, print additional information

    Example:
        >>> source = MockGWEventSource(
        ...     event_cadence=20.0,
        ...     ifos=["H1", "L1", "V1"],
        ...     real_time=True,
        ... )
        >>> # Connect to pipeline - each pad produces EventFrames with XML bytes
        >>> # source.srcs["sgnl"] -> events with ~6s latency
        >>> # source.srcs["pycbc"] -> events with ~30s latency
    """

    # Event scheduling
    event_cadence: float = 20.0

    # Detector configuration
    ifos: List[str] = field(default_factory=lambda: ["H1", "L1", "V1"])

    # Source type distribution
    source_types: List[str] = field(default_factory=lambda: ["bns", "nsbh", "bbh"])
    source_weights: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])

    # Pipeline latencies (mean, std) in seconds
    pipeline_latencies: Optional[Dict[str, Tuple[float, float]]] = None

    # Noise model parameters
    template_min_match: float = 0.97
    snr_threshold: float = 4.0

    # Sky position
    sky_position: str = "overhead"  # "overhead" or "random"
    latitude_rad: float = STATE_COLLEGE_LAT_RAD
    longitude_rad: float = STATE_COLLEGE_LON_RAD

    # Output options
    include_snr_series: bool = True

    # Real-time mode
    real_time: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Initialize the source after creation."""
        # Validate parameters
        if not self.real_time and self.end is None and self.duration is None:
            raise ValueError(
                "When real_time is False, either end or duration must be specified"
            )

        # Set default pipeline latencies
        if self.pipeline_latencies is None:
            self.pipeline_latencies = DEFAULT_PIPELINE_LATENCIES.copy()

        # Set source pad names to pipeline names
        self.source_pad_names = list(self.pipeline_latencies.keys())

        # Set t0 if not provided
        if self.t0 is None:
            self.t0 = int(now())
            if self.verbose:
                print(f"Using current GPS time as start: {self.t0}")

        # Call parent's post_init
        super().__post_init__()

        # Initialize PSDs
        self._psds = fake_gwdata_psd(self.ifos)

        # Initialize event state
        self._central_event_queue: List[_CoincEvent] = []
        self._pipeline_queues: Dict[str, List[Tuple[float, _CoincEvent]]] = {
            p: [] for p in self.source_pad_names
        }
        self._next_event_id = 0
        self._next_event_time = self.t0 + self.event_cadence
        self._events_per_pipeline: Dict[str, int] = {
            p: 0 for p in self.source_pad_names
        }

        # Real-time tracking
        if self.real_time:
            self._start_wall_time = time.time()
            self._start_gps_time = self.t0

        if self.verbose:
            print("MockGWEventSource initialized:")
            print(f"  Pipelines: {self.source_pad_names}")
            print(f"  Detectors: {self.ifos}")
            print(f"  Event cadence: {self.event_cadence}s")
            print(f"  Source types: {self.source_types}")

    def _get_current_gps(self) -> float:
        """Get current GPS time based on data progress."""
        # Use the base class tracking of current time
        return (
            self.t0 + (time.time() - self._start_wall_time)
            if self.real_time
            else self.t0
        )

    def _generate_source_params(self) -> Dict:
        """Generate random source parameters based on source type distribution."""
        # Select source type
        source_type = numpy.random.choice(
            self.source_types,
            p=numpy.array(self.source_weights) / sum(self.source_weights),
        )
        params = SOURCE_TYPE_PARAMS[source_type]

        # Generate masses
        mass1 = numpy.random.uniform(*params["mass1_range"])
        mass2 = numpy.random.uniform(*params["mass2_range"])
        # Ensure mass1 >= mass2
        if mass2 > mass1:
            mass1, mass2 = mass2, mass1

        # Generate distance
        distance = numpy.random.uniform(*params["distance_range"])

        # Generate sky position
        if self.sky_position == "overhead":
            ra = None  # Will be computed based on coalescence time
            dec = self.latitude_rad
        else:  # random
            ra = numpy.random.uniform(0, 2 * math.pi)
            dec = math.asin(numpy.random.uniform(-1, 1))

        # Other parameters
        psi = numpy.random.uniform(0, math.pi)
        inclination = math.acos(numpy.random.uniform(-1, 1))
        phi_ref = numpy.random.uniform(0, 2 * math.pi)

        return {
            "source_type": source_type,
            "mass1": mass1,
            "mass2": mass2,
            "distance": distance,
            "ra": ra,
            "dec": dec,
            "psi": psi,
            "inclination": inclination,
            "phi_ref": phi_ref,
        }

    def _generate_event(self, t_co_gps: float) -> _CoincEvent:
        """Generate a complete mock event with triggers for all detectors."""
        params = self._generate_source_params()

        # Compute RA for overhead position if needed
        if params["ra"] is None:
            params["ra"] = _calculate_overhead_ra(t_co_gps, self.longitude_rad)

        # Compute time delays
        time_delays = _compute_time_delays(
            params["ra"], params["dec"], t_co_gps, self.ifos
        )

        # Compute phases
        phases = _compute_phases(
            params["ra"],
            params["dec"],
            params["psi"],
            params["inclination"],
            params["phi_ref"],
            t_co_gps,
            time_delays,
            self.ifos,
        )

        # Generate triggers for each detector
        triggers = []
        for ifo in self.ifos:
            # Compute optimal SNR
            snr_optimal = _compute_optimal_snr(
                params["mass1"],
                params["mass2"],
                params["distance"],
                params["ra"],
                params["dec"],
                params["psi"],
                params["inclination"],
                t_co_gps,
                ifo,
                self._psds[ifo],
            )

            # Apply template mismatch
            snr_after_mismatch, mass1_template, mass2_template = (
                _apply_template_mismatch(
                    snr_optimal,
                    params["mass1"],
                    params["mass2"],
                    self.template_min_match,
                )
            )

            # Skip if below threshold
            if snr_after_mismatch < self.snr_threshold:
                continue

            # True time and phase at detector
            t_det_true = t_co_gps + time_delays[ifo]
            phi_det_true = phases[ifo]

            # Add noise fluctuations
            snr_measured, t_det_measured, phi_measured = _add_noise_fluctuations(
                snr_after_mismatch, t_det_true, phi_det_true
            )

            # Ensure SNR stays above threshold
            snr_measured = max(snr_measured, self.snr_threshold)

            trigger = _SingleTrigger(
                ifo=ifo,
                end_time=t_det_measured,
                snr=snr_measured,
                coa_phase=phi_measured,
                mass1=mass1_template,
                mass2=mass2_template,
                eff_distance=params["distance"],
            )
            triggers.append(trigger)

        # Only create event if at least 2 detectors saw it
        if len(triggers) < 2:
            # Try again with closer distance
            params["distance"] /= 2
            return self._generate_event(t_co_gps)

        event = _CoincEvent(
            event_id=self._next_event_id,
            t_co_gps=t_co_gps,
            triggers=triggers,
            far=numpy.random.exponential(1e-8),
            source_type=params["source_type"],
        )
        self._next_event_id += 1

        if self.verbose:
            print(
                f"Generated event {event.event_id}: {params['source_type']} "
                f"at t={t_co_gps:.1f}, network SNR={event.network_snr:.1f}, "
                f"IFOs={event.ifos}"
            )

        return event

    def _distribute_event_to_pipelines(self, event: _CoincEvent, current_gps: float):
        """Distribute an event to pipeline queues with appropriate latencies."""
        assert self.pipeline_latencies is not None  # Set in __post_init__
        for pipeline, (mean_latency, std_latency) in self.pipeline_latencies.items():
            # Sample latency from Gaussian (clamped to positive)
            latency = max(1.0, numpy.random.normal(mean_latency, std_latency))
            report_time = event.t_co_gps + latency

            # Add to pipeline queue
            self._pipeline_queues[pipeline].append((report_time, event))

            if self.verbose:
                print(
                    f"  -> {pipeline} will report at t={report_time:.1f} "
                    f"(latency={latency:.1f}s)"
                )

    def _maybe_generate_events(self, current_gps: float):
        """Generate new events if it's time."""
        while current_gps >= self._next_event_time:
            event = self._generate_event(self._next_event_time)
            self._central_event_queue.append(event)
            self._distribute_event_to_pipelines(event, current_gps)
            self._next_event_time += self.event_cadence

    def _get_ready_events(self, pipeline: str, current_gps: float) -> List[_CoincEvent]:
        """Get events ready to report for a pipeline."""
        ready = []
        remaining = []

        for report_time, event in self._pipeline_queues[pipeline]:
            if current_gps >= report_time:
                ready.append(event)
            else:
                remaining.append((report_time, event))

        self._pipeline_queues[pipeline] = remaining
        return ready

    def new(self, pad: SourcePad) -> Frame:
        """Generate a new frame for the specified pad.

        Returns a Frame containing coinc XML bytes if events are ready,
        or a gap frame otherwise.
        """
        pipeline = self.rsrcs[pad]
        current_gps = self._get_current_gps()

        # Check for ready events
        ready_events = self._get_ready_events(pipeline, current_gps)

        if ready_events:
            # Build XML for the first ready event
            event = ready_events[0]
            xmldoc = _build_coinc_xmldoc(
                event, pipeline, self._psds, self.include_snr_series
            )
            xml_bytes = _serialize_xmldoc(xmldoc)

            self._events_per_pipeline[pipeline] += 1

            if self.verbose:
                print(
                    f"{pipeline}: Output event {event.event_id} "
                    f"at GPS {current_gps:.1f}"
                )

            # Return frame with XML bytes as data
            return Frame(
                EOS=False,
                is_gap=False,
                data={
                    "xml": xml_bytes,
                    "event_id": event.event_id,
                    "pipeline": pipeline,
                },
                metadata={
                    "t_co_gps": event.t_co_gps,
                    "network_snr": event.network_snr,
                },
            )
        else:
            # Return gap frame
            return Frame(EOS=False, is_gap=True, data=None, metadata={})

    def internal(self) -> None:
        """Handle event scheduling and real-time synchronization."""
        super().internal()

        current_gps = self._get_current_gps()

        # Generate new events if needed
        self._maybe_generate_events(current_gps)

        # Real-time synchronization
        if self.real_time:
            data_elapsed = current_gps - self._start_gps_time
            wall_elapsed = time.time() - self._start_wall_time
            sleep_time = data_elapsed - wall_elapsed

            if sleep_time > 0.1:
                time.sleep(sleep_time)
            elif sleep_time < -5.0 and self.verbose:
                print(f"Warning: Running {-sleep_time:.1f}s behind real-time")
