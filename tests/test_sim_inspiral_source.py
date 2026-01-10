#!/usr/bin/env python3
"""Test coverage for sgnligo.sources.sim_inspiral_source module."""

import os
import tempfile

import lal
import numpy as np
import pytest
from sgn.apps import Pipeline
from sgnts.sinks import NullSeriesSink

from sgnligo.sources.sim_inspiral_source import (
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    STATE_COLLEGE_LAT_RAD,
    STATE_COLLEGE_LON_RAD,
    TEST_INJECTION_INTERVAL,
    SimInspiralSource,
    WaveformCache,
    _get_dict_real8,
    _get_dict_string,
    _load_lal_h5_injections,
    _load_xml_injections,
    _validate_injection,
    calculate_overhead_ra,
    estimate_waveform_duration,
    generate_test_injection,
    generate_waveform_td,
    load_injections,
    project_to_detector,
)


def create_simple_injection_dict(
    mass1: float = 1.4,
    mass2: float = 1.4,
    distance: float = 100.0,
    geocent_end_time: float = 1000000000.0,
    approximant: str = "IMRPhenomD",
) -> lal.Dict:
    """Create a simple injection dict for testing.

    Args:
        mass1: Primary mass in solar masses
        mass2: Secondary mass in solar masses
        distance: Distance in Mpc
        geocent_end_time: Coalescence time in GPS seconds
        approximant: Waveform approximant

    Returns:
        LAL dict with injection parameters in SI units
    """
    params = lal.CreateDict()

    # Masses in SI units (kg)
    lal.DictInsertREAL8Value(params, "mass1", mass1 * lal.MSUN_SI)
    lal.DictInsertREAL8Value(params, "mass2", mass2 * lal.MSUN_SI)

    # Spins (dimensionless)
    lal.DictInsertREAL8Value(params, "spin1x", 0.0)
    lal.DictInsertREAL8Value(params, "spin1y", 0.0)
    lal.DictInsertREAL8Value(params, "spin1z", 0.0)
    lal.DictInsertREAL8Value(params, "spin2x", 0.0)
    lal.DictInsertREAL8Value(params, "spin2y", 0.0)
    lal.DictInsertREAL8Value(params, "spin2z", 0.0)

    # Distance in SI units (meters)
    lal.DictInsertREAL8Value(params, "distance", distance * 1e6 * lal.PC_SI)

    # Angular quantities (radians)
    lal.DictInsertREAL8Value(params, "inclination", 0.0)
    lal.DictInsertREAL8Value(params, "phi_ref", 0.0)
    lal.DictInsertREAL8Value(params, "psi", 0.0)
    lal.DictInsertREAL8Value(params, "ra", 0.0)
    lal.DictInsertREAL8Value(params, "dec", 0.0)

    # Timing
    lal.DictInsertREAL8Value(params, "t_co_gps", geocent_end_time)

    # Waveform model
    lal.DictInsertStringValue(params, "approximant", approximant)
    lal.DictInsertREAL8Value(params, "f_ref", 20.0)

    return params


@pytest.fixture
def simple_injection():
    """Create a simple injection for testing."""
    return create_simple_injection_dict()


@pytest.fixture
def sample_xml_file():
    """Create a sample XML injection file for testing."""
    # Use correct column types matching the sim_inspiral schema
    dtd = "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt"
    xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "{dtd}">
<LIGO_LW>
    <Table Name="sim_inspiral:table">
        <Column Name="mass1" Type="real_4"/>
        <Column Name="mass2" Type="real_4"/>
        <Column Name="spin1x" Type="real_4"/>
        <Column Name="spin1y" Type="real_4"/>
        <Column Name="spin1z" Type="real_4"/>
        <Column Name="spin2x" Type="real_4"/>
        <Column Name="spin2y" Type="real_4"/>
        <Column Name="spin2z" Type="real_4"/>
        <Column Name="distance" Type="real_4"/>
        <Column Name="inclination" Type="real_4"/>
        <Column Name="coa_phase" Type="real_4"/>
        <Column Name="polarization" Type="real_4"/>
        <Column Name="longitude" Type="real_4"/>
        <Column Name="latitude" Type="real_4"/>
        <Column Name="geocent_end_time" Type="int_4s"/>
        <Column Name="geocent_end_time_ns" Type="int_4s"/>
        <Column Name="waveform" Type="lstring"/>
        <Column Name="f_lower" Type="real_4"/>
        <Stream Name="sim_inspiral:table" Type="Local" Delimiter=",">
            1.4,1.4,0.0,0.0,0.0,0.0,0.0,0.0,100.0,0.0,0.0,0.0,1.0,0.5,1000000010,0,"IMRPhenomD",20.0,
            30.0,30.0,0.0,0.0,0.1,0.0,0.0,0.1,200.0,0.5,0.0,0.1,2.0,0.3,1000000020,500000000,"IMRPhenomD",10.0,
        </Stream>
    </Table>
</LIGO_LW>
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        return f.name


class TestDictHelpers:
    """Test cases for LAL dict helper functions."""

    def test_get_dict_real8_found(self):
        """Test _get_dict_real8 when key exists."""
        d = lal.CreateDict()
        lal.DictInsertREAL8Value(d, "test_key", 42.5)
        assert _get_dict_real8(d, "test_key") == 42.5

    def test_get_dict_real8_not_found(self):
        """Test _get_dict_real8 with default when key missing."""
        d = lal.CreateDict()
        assert _get_dict_real8(d, "missing", 99.0) == 99.0

    def test_get_dict_string_found(self):
        """Test _get_dict_string when key exists."""
        d = lal.CreateDict()
        lal.DictInsertStringValue(d, "approx", "IMRPhenomD")
        assert _get_dict_string(d, "approx") == "IMRPhenomD"

    def test_get_dict_string_not_found(self):
        """Test _get_dict_string with default when key missing."""
        d = lal.CreateDict()
        assert _get_dict_string(d, "missing", "default") == "default"


class TestValidation:
    """Test cases for injection validation."""

    def test_required_fields_defined(self):
        """Test that required fields dict is properly defined."""
        assert "mass1" in REQUIRED_FIELDS
        assert "mass2" in REQUIRED_FIELDS
        assert "distance" in REQUIRED_FIELDS
        assert "ra" in REQUIRED_FIELDS
        assert "dec" in REQUIRED_FIELDS
        assert "psi" in REQUIRED_FIELDS
        assert "t_co_gps" in REQUIRED_FIELDS
        assert "approximant" in REQUIRED_FIELDS
        assert len(REQUIRED_FIELDS) == 8

    def test_optional_fields_defined(self):
        """Test that optional fields dict has reasonable defaults."""
        assert OPTIONAL_FIELDS["spin1z"] == 0.0
        assert OPTIONAL_FIELDS["inclination"] == 0.0
        assert OPTIONAL_FIELDS["f_ref"] == 20.0

    def test_validate_injection_success(self):
        """Test validation passes with all required fields."""
        params = lal.CreateDict()
        lal.DictInsertREAL8Value(params, "mass1", 1.4 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params, "mass2", 1.4 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params, "distance", 100.0 * 1e6 * lal.PC_SI)
        lal.DictInsertREAL8Value(params, "ra", 0.5)
        lal.DictInsertREAL8Value(params, "dec", 0.3)
        lal.DictInsertREAL8Value(params, "psi", 0.1)
        lal.DictInsertREAL8Value(params, "t_co_gps", 1000000000.0)
        lal.DictInsertStringValue(params, "approximant", "IMRPhenomD")
        # Should not raise
        _validate_injection(params)

    def test_validate_injection_missing_single_field(self):
        """Test validation fails with missing required field."""
        params = lal.CreateDict()
        lal.DictInsertREAL8Value(params, "mass1", 1.4 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params, "mass2", 1.4 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params, "distance", 100.0 * 1e6 * lal.PC_SI)
        lal.DictInsertREAL8Value(params, "ra", 0.5)
        lal.DictInsertREAL8Value(params, "dec", 0.3)
        lal.DictInsertREAL8Value(params, "psi", 0.1)
        lal.DictInsertStringValue(params, "approximant", "IMRPhenomD")
        # Missing t_co_gps
        with pytest.raises(ValueError, match="Missing required fields: t_co_gps"):
            _validate_injection(params)

    def test_validate_injection_missing_multiple_fields(self):
        """Test validation reports all missing fields."""
        params = lal.CreateDict()
        lal.DictInsertREAL8Value(params, "mass1", 1.4 * lal.MSUN_SI)
        # Missing everything except mass1
        with pytest.raises(ValueError, match="Missing required fields:"):
            _validate_injection(params)

    def test_validate_injection_index_in_error(self):
        """Test validation error includes injection index."""
        params = lal.CreateDict()
        lal.DictInsertREAL8Value(params, "mass1", 1.4 * lal.MSUN_SI)
        with pytest.raises(ValueError, match="Injection 5:"):
            _validate_injection(params, index=5)


class TestLoadInjections:
    """Test cases for injection loading functions."""

    def test_load_xml_injections(self, sample_xml_file):
        """Test loading injections from XML file."""
        try:
            injections = _load_xml_injections(sample_xml_file)
            assert len(injections) == 2

            # Check first injection (stored in SI units)
            inj0 = injections[0]
            mass1 = _get_dict_real8(inj0, "mass1") / lal.MSUN_SI
            assert mass1 == pytest.approx(1.4, rel=1e-6)
            distance = _get_dict_real8(inj0, "distance") / (1e6 * lal.PC_SI)
            assert distance == pytest.approx(100.0, rel=1e-6)
            geocent_end_time = _get_dict_real8(inj0, "t_co_gps")
            assert geocent_end_time == 1000000010.0

            # Check second injection
            inj1 = injections[1]
            mass1 = _get_dict_real8(inj1, "mass1") / lal.MSUN_SI
            assert mass1 == pytest.approx(30.0, rel=1e-6)
            spin1z = _get_dict_real8(inj1, "spin1z")
            assert spin1z == pytest.approx(0.1, rel=1e-6)
            # Check nanoseconds handling
            geocent_end_time = _get_dict_real8(inj1, "t_co_gps")
            assert geocent_end_time == pytest.approx(1000000020.5, rel=1e-6)
        finally:
            os.unlink(sample_xml_file)

    def test_load_injections_auto_detect_xml(self, sample_xml_file):
        """Test auto-detection for XML files."""
        try:
            injections = load_injections(sample_xml_file)
            assert len(injections) == 2
        finally:
            os.unlink(sample_xml_file)


class TestWaveformDurationEstimation:
    """Test cases for waveform duration estimation."""

    def test_estimate_bns_duration(self, simple_injection):
        """Test duration estimation for BNS."""
        pre_dur, post_dur = estimate_waveform_duration(simple_injection, f_min=10.0)
        # BNS at 10 Hz should have long inspiral (tens of seconds)
        assert pre_dur > 10.0
        assert post_dur > 0.0
        assert post_dur < 2.0  # Ringdown is short for BNS

    def test_estimate_bbh_duration(self):
        """Test duration estimation for BBH."""
        bbh = create_simple_injection_dict(mass1=30.0, mass2=30.0)
        pre_dur, post_dur = estimate_waveform_duration(bbh, f_min=10.0)
        # BBH has shorter inspiral than BNS
        assert pre_dur > 0.5
        assert pre_dur < 20.0  # Much shorter than BNS
        assert post_dur > 0.0

    def test_estimate_with_spin(self):
        """Test duration estimation with spin."""
        params = create_simple_injection_dict(mass1=10.0, mass2=10.0)
        lal.DictInsertREAL8Value(params, "spin1z", 0.9)
        lal.DictInsertREAL8Value(params, "spin2z", 0.9)
        pre_dur, post_dur = estimate_waveform_duration(params, f_min=10.0)
        # Should return reasonable values
        assert pre_dur > 0.0
        assert post_dur > 0.0


class TestWaveformGeneration:
    """Test cases for waveform generation."""

    def test_generate_waveform_td_imrphenomd(self, simple_injection):
        """Test time-domain waveform generation with IMRPhenomD."""
        hp, hc = generate_waveform_td(simple_injection, sample_rate=4096, f_min=20.0)
        assert hp is not None
        assert hc is not None
        assert hp.data.length > 0
        assert hc.data.length > 0
        # Check that waveform has non-zero amplitude
        assert np.max(np.abs(hp.data.data)) > 0

    def test_generate_waveform_td_with_override(self, simple_injection):
        """Test waveform generation with approximant override."""
        hp, hc = generate_waveform_td(
            simple_injection,
            sample_rate=4096,
            f_min=20.0,
            approximant_override="TaylorF2",
        )
        assert hp is not None
        assert hp.data.length > 0

    def test_generate_waveform_bbh(self):
        """Test waveform generation for BBH."""
        bbh = create_simple_injection_dict(mass1=30.0, mass2=30.0, distance=500.0)
        hp, hc = generate_waveform_td(bbh, sample_rate=4096, f_min=20.0)
        assert hp.data.length > 0
        # BBH should have shorter waveform than BNS
        assert hp.data.length < 100000  # Less than ~24 seconds at 4096 Hz


class TestDetectorProjection:
    """Test cases for detector projection."""

    def test_project_to_h1(self, simple_injection):
        """Test projection onto H1."""
        hp, hc = generate_waveform_td(simple_injection, sample_rate=4096, f_min=20.0)
        strain = project_to_detector(hp, hc, simple_injection, "H1")
        assert strain is not None
        assert strain.data.length > 0
        # Strain should be combination of hp and hc
        assert np.max(np.abs(strain.data.data)) > 0

    def test_project_to_l1(self, simple_injection):
        """Test projection onto L1."""
        hp, hc = generate_waveform_td(simple_injection, sample_rate=4096, f_min=20.0)
        strain = project_to_detector(hp, hc, simple_injection, "L1")
        assert strain is not None
        assert strain.data.length > 0

    def test_project_to_v1(self, simple_injection):
        """Test projection onto V1."""
        hp, hc = generate_waveform_td(simple_injection, sample_rate=4096, f_min=20.0)
        strain = project_to_detector(hp, hc, simple_injection, "V1")
        assert strain is not None
        assert strain.data.length > 0

    def test_different_detectors_give_different_strains(self):
        """Test that different detectors produce different strains."""
        # Use non-zero sky position for difference to be visible
        params = create_simple_injection_dict()
        lal.DictInsertREAL8Value(params, "ra", 1.5)
        lal.DictInsertREAL8Value(params, "dec", 0.5)
        lal.DictInsertREAL8Value(params, "psi", 0.3)

        hp, hc = generate_waveform_td(params, sample_rate=4096, f_min=20.0)

        strain_h1 = project_to_detector(hp, hc, params, "H1")
        strain_l1 = project_to_detector(hp, hc, params, "L1")

        # Strains should be different due to different detector orientations
        # and time delays. Compare near the end where signal is strongest.
        n = len(strain_h1.data.data)
        # Get the last 10000 samples (near merger) where signal is strongest
        start_idx = max(0, n - 10000)
        h1_strong = strain_h1.data.data[start_idx:]
        l1_strong = strain_l1.data.data[start_idx:]

        # The strains should not be identical (different antenna patterns)
        # But they might have similar amplitude, so check they're not exactly equal
        assert not np.array_equal(h1_strong, l1_strong)


class TestWaveformCache:
    """Test cases for WaveformCache."""

    def test_cache_init(self, simple_injection):
        """Test cache initialization."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1", "L1"],
            sample_rate=4096,
            f_min=20.0,
        )
        assert len(cache.injections) == 1
        assert "H1" in cache.ifos
        assert "L1" in cache.ifos

    def test_get_overlapping_injections_overlap(self, simple_injection):
        """Test finding overlapping injections."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Buffer that overlaps with injection (merger at 1000000000.0)
        overlapping = cache.get_overlapping_injections(
            buf_start=999999990.0, buf_end=1000000010.0
        )
        assert 0 in overlapping

    def test_get_overlapping_injections_no_overlap(self, simple_injection):
        """Test finding no overlapping injections."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Buffer far from injection
        overlapping = cache.get_overlapping_injections(
            buf_start=1100000000.0, buf_end=1100000001.0
        )
        assert len(overlapping) == 0

    def test_add_injection_to_target(self, simple_injection):
        """Test adding injection to target buffer with sub-sample interpolation."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Create target buffer around merger (geocent_end_time = 1000000000.0)
        # Buffer needs to overlap the waveform which starts before merger
        buf_start = 999999998.0  # 2 seconds before merger
        buf_end = 1000000002.0  # 2 seconds after merger
        num_samples = int((buf_end - buf_start) * 4096)

        target = lal.CreateREAL8TimeSeries(
            "H1:STRAIN",
            lal.LIGOTimeGPS(buf_start),
            0.0,
            1.0 / 4096,
            lal.StrainUnit,
            num_samples,
        )
        target.data.data[:] = 0.0

        # Add injection using new method
        cache.add_injection_to_target(inj_id=0, ifo="H1", target=target)

        # Should have non-zero samples (injection was added)
        assert np.count_nonzero(target.data.data) > 0
        # Should have significant signal (at least 1 second worth of samples)
        assert np.count_nonzero(target.data.data) > 4096

    def test_cache_cleanup(self, simple_injection):
        """Test cache cleanup of expired waveforms."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Generate and cache waveform by adding to a target
        buf_start = 999999999.0
        buf_end = 1000000001.0
        num_samples = int((buf_end - buf_start) * 4096)
        target = lal.CreateREAL8TimeSeries(
            "H1:STRAIN",
            lal.LIGOTimeGPS(buf_start),
            0.0,
            1.0 / 4096,
            lal.StrainUnit,
            num_samples,
        )
        target.data.data[:] = 0.0
        cache.add_injection_to_target(inj_id=0, ifo="H1", target=target)
        assert 0 in cache.cache

        # Cleanup with time past waveform end
        cache.cleanup_expired(current_gps=1100000000.0)
        assert 0 not in cache.cache


class TestSimInspiralSource:
    """Test cases for SimInspiralSource class."""

    def test_init_requires_injection_file_or_test_mode(self):
        """Test that injection_file or test_mode is required."""
        with pytest.raises(
            ValueError, match="Must specify either injection_file or test_mode"
        ):
            SimInspiralSource(name="TestSource", t0=1000000000.0, duration=10.0)

    def test_init_with_xml_file(self, sample_xml_file):
        """Test initialization with XML file."""
        try:
            source = SimInspiralSource(
                name="TestSource",
                injection_file=sample_xml_file,
                ifos=["H1", "L1"],
                t0=1000000000.0,
                duration=30.0,
                sample_rate=4096,
                f_min=20.0,
            )
            assert source.sample_rate == 4096
            assert source.f_min == 20.0
            assert "H1" in source.ifos
            assert len(source._injections) == 2
            assert "H1:INJ-STRAIN" in source.source_pad_names
            assert "L1:INJ-STRAIN" in source.source_pad_names
        finally:
            os.unlink(sample_xml_file)

    def test_init_default_ifos(self, sample_xml_file):
        """Test that default IFOs are H1 and L1."""
        try:
            source = SimInspiralSource(
                name="TestSource",
                injection_file=sample_xml_file,
                t0=1000000000.0,
                duration=30.0,
            )
            assert source.ifos == ["H1", "L1"]
        finally:
            os.unlink(sample_xml_file)

    def test_source_produces_output(self, sample_xml_file):
        """Test that source produces non-zero output during injection."""
        try:
            # Create a short pipeline around an injection time
            # First injection has geocent_end_time=1000000010
            # BNS waveform starts ~200s before merger
            source = SimInspiralSource(
                name="InjSource",
                injection_file=sample_xml_file,
                ifos=["H1"],
                t0=1000000005.0,  # 5 seconds before merger
                duration=10.0,  # Run for 10 seconds to capture merger
                sample_rate=4096,
                f_min=20.0,
            )

            # Create sink to capture output
            collected_data = []

            class DataCollectorSink(NullSeriesSink):
                def pull(self, pad, frame):
                    if frame.buffers:
                        collected_data.append(frame.buffers[0].data.copy())
                    return super().pull(pad, frame)

            sink = DataCollectorSink(
                name="Sink",
                sink_pad_names=["H1:INJ-STRAIN"],
            )

            pipeline = Pipeline()
            pipeline.insert(source, sink)
            pipeline.insert(
                link_map={
                    "Sink:snk:H1:INJ-STRAIN": "InjSource:src:H1:INJ-STRAIN",
                }
            )
            pipeline.run()

            # Check that we got data
            assert len(collected_data) > 0

            # At least one buffer should have non-zero data (injection signal)
            all_data = np.concatenate(collected_data)
            max_amplitude = np.max(np.abs(all_data))
            assert max_amplitude > 0, "Expected non-zero injection signal"

        finally:
            os.unlink(sample_xml_file)

    def test_source_produces_zeros_without_injection(self, sample_xml_file):
        """Test that source produces zeros when no injection overlaps."""
        try:
            # Create pipeline at time far from injections
            source = SimInspiralSource(
                name="InjSource",
                injection_file=sample_xml_file,
                ifos=["H1"],
                t0=1200000000.0,  # Far from injection times
                duration=1.0,
                sample_rate=4096,
                f_min=20.0,
            )

            collected_data = []

            class DataCollectorSink(NullSeriesSink):
                def pull(self, pad, frame):
                    if frame.buffers:
                        collected_data.append(frame.buffers[0].data.copy())
                    return super().pull(pad, frame)

            sink = DataCollectorSink(
                name="Sink",
                sink_pad_names=["H1:INJ-STRAIN"],
            )

            pipeline = Pipeline()
            pipeline.insert(source, sink)
            pipeline.insert(
                link_map={
                    "Sink:snk:H1:INJ-STRAIN": "InjSource:src:H1:INJ-STRAIN",
                }
            )
            pipeline.run()

            # All data should be zeros
            all_data = np.concatenate(collected_data)
            assert np.allclose(all_data, 0.0), "Expected all zeros far from injection"

        finally:
            os.unlink(sample_xml_file)

    def test_multi_ifo_output(self, sample_xml_file):
        """Test that multiple IFOs produce independent outputs."""
        try:
            source = SimInspiralSource(
                name="InjSource",
                injection_file=sample_xml_file,
                ifos=["H1", "L1"],
                t0=1000000008.0,
                duration=4.0,
                sample_rate=4096,
                f_min=20.0,
            )

            h1_data = []
            l1_data = []

            class MultiDataCollectorSink(NullSeriesSink):
                def pull(self, pad, frame):
                    if frame.buffers:
                        if "H1" in pad.name:
                            h1_data.append(frame.buffers[0].data.copy())
                        elif "L1" in pad.name:
                            l1_data.append(frame.buffers[0].data.copy())
                    return super().pull(pad, frame)

            sink = MultiDataCollectorSink(
                name="Sink",
                sink_pad_names=["H1:INJ-STRAIN", "L1:INJ-STRAIN"],
            )

            pipeline = Pipeline()
            pipeline.insert(source, sink)
            pipeline.insert(
                link_map={
                    "Sink:snk:H1:INJ-STRAIN": "InjSource:src:H1:INJ-STRAIN",
                    "Sink:snk:L1:INJ-STRAIN": "InjSource:src:L1:INJ-STRAIN",
                }
            )
            pipeline.run()

            # Both should have data
            assert len(h1_data) > 0
            assert len(l1_data) > 0

            # Data should be different due to different detector responses
            h1_all = np.concatenate(h1_data)
            l1_all = np.concatenate(l1_data)
            # They might be similar in amplitude but should differ in detail
            # due to antenna patterns and time delays
            # Compare non-zero signal portions (zeros dominate, can't use allclose)
            assert not np.array_equal(
                h1_all, l1_all
            ), "Detector responses should differ"

        finally:
            os.unlink(sample_xml_file)


class TestCoverageEdgeCases:
    """Test cases for edge cases and error handling to improve coverage."""

    def test_load_injections_unknown_extension_xml_fallback(self):
        """Test loading file with unknown extension falls back to XML."""
        dtd = "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt"
        xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "{dtd}">
<LIGO_LW>
    <Table Name="sim_inspiral:table">
        <Column Name="mass1" Type="real_4"/>
        <Column Name="mass2" Type="real_4"/>
        <Column Name="spin1x" Type="real_4"/>
        <Column Name="spin1y" Type="real_4"/>
        <Column Name="spin1z" Type="real_4"/>
        <Column Name="spin2x" Type="real_4"/>
        <Column Name="spin2y" Type="real_4"/>
        <Column Name="spin2z" Type="real_4"/>
        <Column Name="distance" Type="real_4"/>
        <Column Name="inclination" Type="real_4"/>
        <Column Name="coa_phase" Type="real_4"/>
        <Column Name="polarization" Type="real_4"/>
        <Column Name="longitude" Type="real_4"/>
        <Column Name="latitude" Type="real_4"/>
        <Column Name="geocent_end_time" Type="int_4s"/>
        <Column Name="geocent_end_time_ns" Type="int_4s"/>
        <Column Name="waveform" Type="lstring"/>
        <Column Name="f_lower" Type="real_4"/>
        <Stream Name="sim_inspiral:table" Type="Local" Delimiter=",">
            5.0,5.0,0,0,0,0,0,0,150,0,0,0,0,0,1000000000,0,"IMRPhenomD",20,
        </Stream>
    </Table>
</LIGO_LW>
"""
        # Create file with unknown extension
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False, mode="w") as f:
            f.write(xml_content)
            filepath = f.name

        try:
            injections = load_injections(filepath)
            assert len(injections) == 1
            mass1 = _get_dict_real8(injections[0], "mass1") / lal.MSUN_SI
            assert mass1 == pytest.approx(5.0, rel=1e-6)
        finally:
            os.unlink(filepath)

    def test_add_injection_no_overlap(self, simple_injection):
        """Test add_injection_to_target when buffer doesn't overlap waveform."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Buffer far after waveform ends (waveform ends shortly after merger)
        buf_start = 1100000000.0  # Far after merger
        buf_end = 1100000001.0
        num_samples = int((buf_end - buf_start) * 4096)

        target = lal.CreateREAL8TimeSeries(
            "H1:STRAIN",
            lal.LIGOTimeGPS(buf_start),
            0.0,
            1.0 / 4096,
            lal.StrainUnit,
            num_samples,
        )
        target.data.data[:] = 0.0

        # Add injection (should not modify target since no overlap)
        cache.add_injection_to_target(inj_id=0, ifo="H1", target=target)

        # Should remain all zeros since no overlap
        assert np.count_nonzero(target.data.data) == 0

    def test_estimate_duration_fallback_chirp(self):
        """Test duration estimation fallback for chirp time."""
        from unittest.mock import patch

        import lalsimulation as lalsim

        params = create_simple_injection_dict(mass1=10.0, mass2=10.0)
        # Mock SimInspiralChirpTimeBound to raise an exception
        with patch.object(
            lalsim, "SimInspiralChirpTimeBound", side_effect=RuntimeError("test")
        ):
            pre_dur, post_dur = estimate_waveform_duration(params, f_min=20.0)
            # Should still return valid duration using fallback formula
            assert pre_dur > 0
            assert post_dur > 0

    def test_estimate_duration_fallback_merge(self):
        """Test duration estimation fallback for merge time."""
        from unittest.mock import patch

        import lalsimulation as lalsim

        params = create_simple_injection_dict(mass1=10.0, mass2=10.0)
        # Mock SimInspiralMergeTimeBound to raise an exception
        with patch.object(
            lalsim, "SimInspiralMergeTimeBound", side_effect=RuntimeError("test")
        ):
            pre_dur, post_dur = estimate_waveform_duration(params, f_min=20.0)
            # Should still return valid duration using fallback
            assert pre_dur > 0
            assert post_dur > 0

    def test_estimate_duration_fallback_ringdown(self):
        """Test duration estimation fallback for ringdown time."""
        from unittest.mock import patch

        import lalsimulation as lalsim

        params = create_simple_injection_dict(mass1=10.0, mass2=10.0)
        # Mock SimInspiralRingdownTimeBound to raise an exception
        with patch.object(
            lalsim, "SimInspiralRingdownTimeBound", side_effect=RuntimeError("test")
        ):
            pre_dur, post_dur = estimate_waveform_duration(params, f_min=20.0)
            # Should still return valid duration using fallback
            assert pre_dur > 0
            assert post_dur > 0

    def test_ifo_from_pad_unknown(self, sample_xml_file):
        """Test _ifo_from_pad raises error for unknown pad."""
        from unittest.mock import MagicMock

        try:
            source = SimInspiralSource(
                name="TestSource",
                injection_file=sample_xml_file,
                ifos=["H1"],
                t0=1000000000.0,
                duration=1.0,
                sample_rate=4096,
                f_min=20.0,
            )

            # Create a fake pad that doesn't exist in the source
            fake_pad = MagicMock()
            fake_pad.name = "FAKE:UNKNOWN-PAD"

            with pytest.raises(ValueError, match="Unknown pad"):
                source._ifo_from_pad(fake_pad)
        finally:
            os.unlink(sample_xml_file)

    def test_load_injections_unknown_extension_lal_h5_fallback(self):
        """Test loading LAL H5 file with unknown extension falls back to LAL H5."""
        import lalsimulation as lalsim

        # Create LAL H5 file with unknown extension
        with tempfile.NamedTemporaryFile(suffix=".inj", delete=False) as f:
            filepath = f.name

        try:
            # Create a minimal LAL H5 injection file
            seq = lal.CreateDictSequence(1)
            params = lal.CreateDict()
            lal.DictInsertREAL8Value(params, "mass1", 25.0 * lal.MSUN_SI)
            lal.DictInsertREAL8Value(params, "mass2", 25.0 * lal.MSUN_SI)
            lal.DictInsertREAL8Value(params, "spin1x", 0.0)
            lal.DictInsertREAL8Value(params, "spin1y", 0.0)
            lal.DictInsertREAL8Value(params, "spin1z", 0.0)
            lal.DictInsertREAL8Value(params, "spin2x", 0.0)
            lal.DictInsertREAL8Value(params, "spin2y", 0.0)
            lal.DictInsertREAL8Value(params, "spin2z", 0.0)
            lal.DictInsertREAL8Value(params, "distance", 200.0 * 1e6 * lal.PC_SI)
            lal.DictInsertREAL8Value(params, "inclination", 0.0)
            lal.DictInsertREAL8Value(params, "phi_ref", 0.0)
            lal.DictInsertREAL8Value(params, "psi", 0.0)
            lal.DictInsertREAL8Value(params, "ra", 0.0)
            lal.DictInsertREAL8Value(params, "dec", 0.0)
            lal.DictInsertREAL8Value(params, "t_co_gps", 1000000000.0)
            lal.DictInsertStringValue(params, "approximant", "IMRPhenomD")
            lal.DictInsertREAL8Value(params, "f_ref", 20.0)
            lal.DictSequenceSet(seq, params, 0)
            lalsim.SimInspiralInjectionSequenceToH5File(seq, filepath)

            # This should fail XML parsing then fall back to LAL H5
            injections = load_injections(filepath)
            assert len(injections) == 1
            mass1 = _get_dict_real8(injections[0], "mass1") / lal.MSUN_SI
            assert mass1 == pytest.approx(25.0, rel=1e-6)
        finally:
            os.unlink(filepath)


class TestLalH5Format:
    """Test cases for LAL H5 format injection loading."""

    @pytest.fixture
    def sample_lal_h5_file(self):
        """Create a sample LAL-format HDF5 file for testing.

        LAL format has a 'cbc_waveform_params' group with injection dicts.
        """
        import lalsimulation as lalsim

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        # Create a dict sequence with injection parameters
        seq = lal.CreateDictSequence(2)

        # First injection - simple BNS
        params1 = lal.CreateDict()
        lal.DictInsertREAL8Value(params1, "mass1", 1.4 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params1, "mass2", 1.4 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params1, "spin1x", 0.0)
        lal.DictInsertREAL8Value(params1, "spin1y", 0.0)
        lal.DictInsertREAL8Value(params1, "spin1z", 0.0)
        lal.DictInsertREAL8Value(params1, "spin2x", 0.0)
        lal.DictInsertREAL8Value(params1, "spin2y", 0.0)
        lal.DictInsertREAL8Value(params1, "spin2z", 0.0)
        lal.DictInsertREAL8Value(params1, "distance", 100.0 * 1e6 * lal.PC_SI)
        lal.DictInsertREAL8Value(params1, "inclination", 0.0)
        lal.DictInsertREAL8Value(params1, "phi_ref", 0.5)
        lal.DictInsertREAL8Value(params1, "psi", 0.3)
        lal.DictInsertREAL8Value(params1, "ra", 1.0)
        lal.DictInsertREAL8Value(params1, "dec", 0.5)
        lal.DictInsertREAL8Value(params1, "t_co_gps", 1000000010.0)
        lal.DictInsertStringValue(params1, "approximant", "IMRPhenomD")
        lal.DictInsertREAL8Value(params1, "f_ref", 20.0)
        lal.DictSequenceSet(seq, params1, 0)

        # Second injection - BBH with spins
        params2 = lal.CreateDict()
        lal.DictInsertREAL8Value(params2, "mass1", 30.0 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params2, "mass2", 30.0 * lal.MSUN_SI)
        lal.DictInsertREAL8Value(params2, "spin1x", 0.0)
        lal.DictInsertREAL8Value(params2, "spin1y", 0.0)
        lal.DictInsertREAL8Value(params2, "spin1z", 0.5)
        lal.DictInsertREAL8Value(params2, "spin2x", 0.0)
        lal.DictInsertREAL8Value(params2, "spin2y", 0.0)
        lal.DictInsertREAL8Value(params2, "spin2z", -0.3)
        lal.DictInsertREAL8Value(params2, "distance", 500.0 * 1e6 * lal.PC_SI)
        lal.DictInsertREAL8Value(params2, "inclination", 0.7)
        lal.DictInsertREAL8Value(params2, "phi_ref", 1.0)
        lal.DictInsertREAL8Value(params2, "psi", 0.1)
        lal.DictInsertREAL8Value(params2, "ra", 2.5)
        lal.DictInsertREAL8Value(params2, "dec", -0.3)
        lal.DictInsertREAL8Value(params2, "t_co_gps", 1000000020.0)
        lal.DictInsertStringValue(params2, "approximant", "SEOBNRv4")
        lal.DictInsertREAL8Value(params2, "f_ref", 10.0)
        lal.DictSequenceSet(seq, params2, 1)

        # Write using LAL's H5 writer
        lalsim.SimInspiralInjectionSequenceToH5File(seq, filepath)

        return filepath

    def test_load_lal_h5_injections(self, sample_lal_h5_file):
        """Test loading injections from LAL H5 format."""
        try:
            injections = _load_lal_h5_injections(sample_lal_h5_file)
            assert len(injections) == 2

            # Check first injection (BNS) - values in SI units
            inj0 = injections[0]
            mass1 = _get_dict_real8(inj0, "mass1") / lal.MSUN_SI
            assert mass1 == pytest.approx(1.4, rel=1e-6)
            distance = _get_dict_real8(inj0, "distance") / (1e6 * lal.PC_SI)
            assert distance == pytest.approx(100.0, rel=1e-6)
            phi_ref = _get_dict_real8(inj0, "phi_ref")
            assert phi_ref == pytest.approx(0.5, rel=1e-6)
            psi = _get_dict_real8(inj0, "psi")
            assert psi == pytest.approx(0.3, rel=1e-6)
            ra = _get_dict_real8(inj0, "ra")
            assert ra == pytest.approx(1.0, rel=1e-6)
            dec = _get_dict_real8(inj0, "dec")
            assert dec == pytest.approx(0.5, rel=1e-6)
            geocent_end_time = _get_dict_real8(inj0, "t_co_gps")
            assert geocent_end_time == pytest.approx(1000000010.0, rel=1e-9)
            approximant = _get_dict_string(inj0, "approximant")
            assert approximant == "IMRPhenomD"
            f_ref = _get_dict_real8(inj0, "f_ref")
            assert f_ref == pytest.approx(20.0, rel=1e-6)

            # Check second injection (BBH)
            inj1 = injections[1]
            mass1 = _get_dict_real8(inj1, "mass1") / lal.MSUN_SI
            assert mass1 == pytest.approx(30.0, rel=1e-6)
            spin1z = _get_dict_real8(inj1, "spin1z")
            assert spin1z == pytest.approx(0.5, rel=1e-6)
            spin2z = _get_dict_real8(inj1, "spin2z")
            assert spin2z == pytest.approx(-0.3, rel=1e-6)
            distance = _get_dict_real8(inj1, "distance") / (1e6 * lal.PC_SI)
            assert distance == pytest.approx(500.0, rel=1e-6)
            inclination = _get_dict_real8(inj1, "inclination")
            assert inclination == pytest.approx(0.7, rel=1e-6)
            approximant = _get_dict_string(inj1, "approximant")
            assert approximant == "SEOBNRv4"
        finally:
            os.unlink(sample_lal_h5_file)

    def test_load_injections_auto_detects_lal_h5(self, sample_lal_h5_file):
        """Test load_injections auto-detects LAL H5 format."""
        try:
            injections = load_injections(sample_lal_h5_file)
            assert len(injections) == 2
            geocent_end_time = _get_dict_real8(injections[0], "t_co_gps")
            assert geocent_end_time == pytest.approx(1000000010.0, rel=1e-9)
        finally:
            os.unlink(sample_lal_h5_file)

    def test_sim_inspiral_source_with_lal_h5(self, sample_lal_h5_file):
        """Test SimInspiralSource works with LAL H5 format files."""
        try:
            source = SimInspiralSource(
                name="TestSource",
                injection_file=sample_lal_h5_file,
                ifos=["H1", "L1"],
                t0=1000000000.0,
                duration=30.0,
                sample_rate=4096,
                f_min=20.0,
            )
            assert len(source._injections) == 2
            # Check masses were loaded correctly (in SI units)
            mass1_0 = _get_dict_real8(source._injections[0], "mass1") / lal.MSUN_SI
            assert mass1_0 == pytest.approx(1.4, rel=1e-6)
            mass1_1 = _get_dict_real8(source._injections[1], "mass1") / lal.MSUN_SI
            assert mass1_1 == pytest.approx(30.0, rel=1e-6)
        finally:
            os.unlink(sample_lal_h5_file)


# =============================================================================
# Test Mode Tests
# =============================================================================


class TestCalculateOverheadRA:
    """Tests for calculate_overhead_ra function."""

    def test_calculate_overhead_ra_basic(self):
        """Test basic RA calculation."""
        # At some GPS time, calculate the RA for overhead position
        gps_time = 1000000000.0
        ra = calculate_overhead_ra(gps_time, STATE_COLLEGE_LON_RAD)
        # RA should be in [0, 2π)
        assert 0 <= ra < 2 * np.pi

    def test_calculate_overhead_ra_different_times(self):
        """Test that RA changes with GPS time."""
        ra1 = calculate_overhead_ra(1000000000.0, STATE_COLLEGE_LON_RAD)
        # 6 hours later (approximately 1/4 sidereal day)
        ra2 = calculate_overhead_ra(1000000000.0 + 6 * 3600, STATE_COLLEGE_LON_RAD)
        # RA should have changed by approximately π/2 (90 degrees)
        # Use modular arithmetic since RA wraps around
        diff = abs(ra2 - ra1)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        # Allow some tolerance due to sidereal vs solar time difference
        assert diff == pytest.approx(np.pi / 2, rel=0.05)

    def test_calculate_overhead_ra_different_longitudes(self):
        """Test that different longitudes give different RAs."""
        gps_time = 1000000000.0
        ra_west = calculate_overhead_ra(gps_time, -np.pi / 2)  # 90° West
        ra_east = calculate_overhead_ra(gps_time, np.pi / 2)  # 90° East
        # Difference should be π (180 degrees)
        diff = abs(ra_east - ra_west)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        assert diff == pytest.approx(np.pi, rel=1e-6)


class TestGenerateTestInjection:
    """Tests for generate_test_injection function."""

    def test_generate_bns_injection(self):
        """Test BNS test injection generation."""
        t_co = 1000000000.0
        inj = generate_test_injection("bns", t_co)

        # Check masses
        mass1 = _get_dict_real8(inj, "mass1") / lal.MSUN_SI
        mass2 = _get_dict_real8(inj, "mass2") / lal.MSUN_SI
        assert mass1 == pytest.approx(1.4, rel=1e-6)
        assert mass2 == pytest.approx(1.4, rel=1e-6)

        # Check distance
        distance = _get_dict_real8(inj, "distance") / (1e6 * lal.PC_SI)
        assert distance == pytest.approx(100.0, rel=1e-6)

        # Check sky position
        dec = _get_dict_real8(inj, "dec")
        assert dec == pytest.approx(STATE_COLLEGE_LAT_RAD, rel=1e-6)

        # Check coalescence time
        t_co_gps = _get_dict_real8(inj, "t_co_gps")
        assert t_co_gps == pytest.approx(t_co, rel=1e-9)

    def test_generate_nsbh_injection(self):
        """Test NSBH test injection generation."""
        t_co = 1000000000.0
        inj = generate_test_injection("nsbh", t_co)

        mass1 = _get_dict_real8(inj, "mass1") / lal.MSUN_SI
        mass2 = _get_dict_real8(inj, "mass2") / lal.MSUN_SI
        assert mass1 == pytest.approx(10.0, rel=1e-6)
        assert mass2 == pytest.approx(1.4, rel=1e-6)

        distance = _get_dict_real8(inj, "distance") / (1e6 * lal.PC_SI)
        assert distance == pytest.approx(200.0, rel=1e-6)

    def test_generate_bbh_injection(self):
        """Test BBH test injection generation."""
        t_co = 1000000000.0
        inj = generate_test_injection("bbh", t_co)

        mass1 = _get_dict_real8(inj, "mass1") / lal.MSUN_SI
        mass2 = _get_dict_real8(inj, "mass2") / lal.MSUN_SI
        assert mass1 == pytest.approx(30.0, rel=1e-6)
        assert mass2 == pytest.approx(30.0, rel=1e-6)

        distance = _get_dict_real8(inj, "distance") / (1e6 * lal.PC_SI)
        assert distance == pytest.approx(500.0, rel=1e-6)

    def test_injection_has_all_required_fields(self):
        """Test that generated injection has all required fields."""
        inj = generate_test_injection("bns", 1000000000.0)
        # Should not raise
        _validate_injection(inj, index=0)

    def test_injection_has_optional_fields(self):
        """Test that generated injection has optional fields set."""
        inj = generate_test_injection("bns", 1000000000.0)
        for field_name in OPTIONAL_FIELDS:
            # Should not raise - all optional fields should be present
            _get_dict_real8(inj, field_name)


class TestSimInspiralSourceTestMode:
    """Tests for SimInspiralSource test mode."""

    def test_test_mode_validation_mutual_exclusivity(self, sample_xml_file):
        """Test that injection_file and test_mode are mutually exclusive."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            SimInspiralSource(
                name="TestSource",
                injection_file=sample_xml_file,
                test_mode="bns",
                t0=1000000000.0,
                duration=10.0,
            )

    def test_test_mode_validation_requires_one(self):
        """Test that either injection_file or test_mode must be specified."""
        with pytest.raises(ValueError, match="Must specify either"):
            SimInspiralSource(
                name="TestSource",
                t0=1000000000.0,
                duration=10.0,
            )

    def test_test_mode_validation_invalid_mode(self):
        """Test that invalid test_mode raises an error."""
        with pytest.raises(ValueError, match="Invalid test_mode"):
            SimInspiralSource(
                name="TestSource",
                test_mode="invalid",
                t0=1000000000.0,
                duration=10.0,
            )

    def test_test_mode_case_insensitive(self):
        """Test that test_mode is case insensitive."""
        source = SimInspiralSource(
            name="TestSource",
            test_mode="BNS",  # uppercase
            ifos=["H1"],
            t0=1000000000.0,
            duration=10.0,
            sample_rate=4096,
            f_min=20.0,
        )
        assert source.test_mode == "bns"  # normalized to lowercase

    def test_test_mode_bns_creates_source(self):
        """Test that BNS test mode creates a valid source."""
        source = SimInspiralSource(
            name="TestSource",
            test_mode="bns",
            ifos=["H1", "L1"],
            t0=1000000000.0,
            duration=60.0,  # Long enough to capture injections
            sample_rate=4096,
            f_min=20.0,
        )
        assert source.test_mode == "bns"
        assert source._waveform_cache._test_mode == "bns"
        # Initially empty, injections generated on demand
        assert len(source._injections) == 0

    def test_test_mode_nsbh_creates_source(self):
        """Test that NSBH test mode creates a valid source."""
        source = SimInspiralSource(
            name="TestSource",
            test_mode="nsbh",
            ifos=["H1"],
            t0=1000000000.0,
            duration=60.0,
            sample_rate=4096,
            f_min=20.0,
        )
        assert source.test_mode == "nsbh"

    def test_test_mode_bbh_creates_source(self):
        """Test that BBH test mode creates a valid source."""
        source = SimInspiralSource(
            name="TestSource",
            test_mode="bbh",
            ifos=["H1"],
            t0=1000000000.0,
            duration=60.0,
            sample_rate=4096,
            f_min=20.0,
        )
        assert source.test_mode == "bbh"


class TestWaveformCacheTestMode:
    """Tests for WaveformCache with test mode."""

    def test_test_mode_generates_injections_on_demand(self):
        """Test that test mode generates injections when queried."""
        cache = WaveformCache(
            injections=[],
            ifos=["H1", "L1"],
            sample_rate=4096,
            f_min=20.0,
            test_mode="bbh",
        )

        # Initially no injections
        assert len(cache.injections) == 0

        # Query for a time range that should contain injections
        # BBH at 30-second intervals, query around t=1000000030
        cache.get_overlapping_injections(1000000025.0, 1000000035.0)

        # Should have generated at least one injection (at t=1000000030)
        assert len(cache.injections) >= 1
        assert len(cache._generated_test_times) >= 1

    def test_test_mode_injection_interval(self):
        """Test that injections are generated at correct intervals."""
        cache = WaveformCache(
            injections=[],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
            test_mode="bbh",
        )

        # Query a range spanning multiple injection times
        # From t=1000000000 to t=1000000100 should have ~3-4 injections IN the range
        # Plus additional injections before/after to account for waveform duration
        cache.get_overlapping_injections(1000000000.0, 1000000100.0)

        # All generated times should be at 30-second intervals
        for t_co in cache._generated_test_times:
            # Check that t_co is a multiple of 30
            assert t_co % TEST_INJECTION_INTERVAL == 0

        # The 30-second boundaries near our query range
        # (1000000000 is NOT a multiple of 30!)
        # Actual multiples: 999999990, 1000000020, 1000000050, 1000000080
        expected_times = {1000000020.0, 1000000050.0, 1000000080.0}
        assert expected_times.issubset(cache._generated_test_times)

    def test_test_mode_no_duplicate_injections(self):
        """Test that querying the same range twice doesn't create duplicates."""
        cache = WaveformCache(
            injections=[],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
            test_mode="bbh",
        )

        # Query same range twice
        cache.get_overlapping_injections(1000000025.0, 1000000035.0)
        initial_count = len(cache.injections)

        cache.get_overlapping_injections(1000000025.0, 1000000035.0)
        final_count = len(cache.injections)

        assert final_count == initial_count

    def test_non_test_mode_no_dynamic_generation(self):
        """Test that non-test mode doesn't generate injections dynamically."""
        inj = generate_test_injection("bns", 1000000000.0)
        cache = WaveformCache(
            injections=[inj],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
            test_mode=None,  # Not test mode
        )

        initial_count = len(cache.injections)
        cache.get_overlapping_injections(1000000025.0, 1000000035.0)
        final_count = len(cache.injections)

        # Should not have generated any new injections
        assert final_count == initial_count
