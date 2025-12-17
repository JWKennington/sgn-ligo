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
    InjectionParams,
    SimInspiralSource,
    WaveformCache,
    _fd_to_td,
    _load_hdf5_injections,
    _load_xml_injections,
    estimate_waveform_duration,
    generate_waveform_td,
    load_injections,
    project_to_detector,
)


@pytest.fixture
def simple_injection():
    """Create a simple injection for testing."""
    return InjectionParams(
        mass1=1.4,
        mass2=1.4,
        spin1z=0.0,
        spin2z=0.0,
        distance=100.0,
        inclination=0.0,
        coa_phase=0.0,
        polarization=0.0,
        ra=0.0,
        dec=0.0,
        geocent_end_time=1000000000.0,
        approximant="IMRPhenomD",
        f_ref=20.0,
    )


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


@pytest.fixture
def sample_hdf5_file():
    """Create a sample HDF5 injection file for testing."""
    import h5py

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        filepath = f.name

    with h5py.File(filepath, "w") as hf:
        grp = hf.create_group("injections")
        grp.create_dataset("mass1", data=[1.4, 30.0])
        grp.create_dataset("mass2", data=[1.4, 30.0])
        grp.create_dataset("spin1x", data=[0.0, 0.0])
        grp.create_dataset("spin1y", data=[0.0, 0.0])
        grp.create_dataset("spin1z", data=[0.0, 0.1])
        grp.create_dataset("spin2x", data=[0.0, 0.0])
        grp.create_dataset("spin2y", data=[0.0, 0.0])
        grp.create_dataset("spin2z", data=[0.0, 0.1])
        grp.create_dataset("distance", data=[100.0, 200.0])
        grp.create_dataset("inclination", data=[0.0, 0.5])
        grp.create_dataset("coa_phase", data=[0.0, 0.0])
        grp.create_dataset("polarization", data=[0.0, 0.1])
        grp.create_dataset("ra", data=[1.0, 2.0])
        grp.create_dataset("dec", data=[0.5, 0.3])
        grp.create_dataset("geocent_end_time", data=[1000000010.0, 1000000020.5])
        grp.create_dataset(
            "approximant",
            data=[b"IMRPhenomD", b"IMRPhenomD"],
        )
        grp.create_dataset("f_ref", data=[20.0, 10.0])

    return filepath


class TestInjectionParams:
    """Test cases for InjectionParams dataclass."""

    def test_defaults(self):
        """Test default values."""
        inj = InjectionParams(mass1=1.4, mass2=1.4)
        assert inj.mass1 == 1.4
        assert inj.mass2 == 1.4
        assert inj.spin1z == 0.0
        assert inj.distance == 100.0
        assert inj.approximant == "IMRPhenomD"

    def test_custom_values(self):
        """Test custom values."""
        inj = InjectionParams(
            mass1=30.0,
            mass2=30.0,
            spin1z=0.5,
            spin2z=-0.3,
            distance=500.0,
            inclination=0.7,
            ra=1.5,
            dec=0.3,
            geocent_end_time=1234567890.0,
            approximant="SEOBNRv4",
        )
        assert inj.mass1 == 30.0
        assert inj.spin1z == 0.5
        assert inj.distance == 500.0
        assert inj.approximant == "SEOBNRv4"


class TestLoadInjections:
    """Test cases for injection loading functions."""

    def test_load_xml_injections(self, sample_xml_file):
        """Test loading injections from XML file."""
        try:
            injections = _load_xml_injections(sample_xml_file)
            assert len(injections) == 2

            # Check first injection
            inj0 = injections[0]
            assert inj0.mass1 == 1.4
            assert inj0.mass2 == 1.4
            assert inj0.distance == 100.0
            assert inj0.geocent_end_time == 1000000010.0

            # Check second injection
            inj1 = injections[1]
            assert inj1.mass1 == 30.0
            assert inj1.distance == 200.0
            assert inj1.spin1z == 0.1
            # Check nanoseconds handling
            assert inj1.geocent_end_time == pytest.approx(1000000020.5, rel=1e-6)
        finally:
            os.unlink(sample_xml_file)

    def test_load_hdf5_injections(self, sample_hdf5_file):
        """Test loading injections from HDF5 file."""
        try:
            injections = load_injections(sample_hdf5_file)
            assert len(injections) == 2

            inj0 = injections[0]
            assert inj0.mass1 == 1.4
            assert inj0.distance == 100.0

            inj1 = injections[1]
            assert inj1.mass1 == 30.0
            assert inj1.spin1z == 0.1
        finally:
            os.unlink(sample_hdf5_file)

    def test_load_injections_auto_detect_xml(self, sample_xml_file):
        """Test auto-detection for XML files."""
        try:
            injections = load_injections(sample_xml_file)
            assert len(injections) == 2
        finally:
            os.unlink(sample_xml_file)

    def test_load_injections_auto_detect_hdf5(self, sample_hdf5_file):
        """Test auto-detection for HDF5 files."""
        try:
            injections = load_injections(sample_hdf5_file)
            assert len(injections) == 2
        finally:
            os.unlink(sample_hdf5_file)


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
        bbh = InjectionParams(
            mass1=30.0,
            mass2=30.0,
            geocent_end_time=1000000000.0,
        )
        pre_dur, post_dur = estimate_waveform_duration(bbh, f_min=10.0)
        # BBH has shorter inspiral than BNS
        assert pre_dur > 0.5
        assert pre_dur < 20.0  # Much shorter than BNS
        assert post_dur > 0.0

    def test_estimate_with_spin(self):
        """Test duration estimation with spin."""
        spinning = InjectionParams(
            mass1=10.0,
            mass2=10.0,
            spin1z=0.9,
            spin2z=0.9,
            geocent_end_time=1000000000.0,
        )
        pre_dur, post_dur = estimate_waveform_duration(spinning, f_min=10.0)
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
        bbh = InjectionParams(
            mass1=30.0,
            mass2=30.0,
            distance=500.0,
            geocent_end_time=1000000000.0,
            approximant="IMRPhenomD",
        )
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

    def test_different_detectors_give_different_strains(self, simple_injection):
        """Test that different detectors produce different strains."""
        # Use non-zero sky position for difference to be visible
        inj = InjectionParams(
            mass1=1.4,
            mass2=1.4,
            distance=100.0,
            ra=1.5,  # Non-zero RA
            dec=0.5,  # Non-zero Dec
            polarization=0.3,
            geocent_end_time=1000000000.0,
        )
        hp, hc = generate_waveform_td(inj, sample_rate=4096, f_min=20.0)

        strain_h1 = project_to_detector(hp, hc, inj, "H1")
        strain_l1 = project_to_detector(hp, hc, inj, "L1")

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

    def test_get_waveform_slice(self, simple_injection):
        """Test getting waveform slice for buffer."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Get slice around merger (geocent_end_time = 1000000000.0)
        # Buffer needs to overlap the waveform which starts ~200s before merger
        # The waveform ends shortly after merger (ringdown is short for BNS)
        wf_slice, start_idx = cache.get_waveform_slice(
            inj_id=0,
            ifo="H1",
            buf_start=999999998.0,  # 2 seconds before merger
            buf_end=1000000002.0,  # 2 seconds after merger
        )
        # Should have some samples (waveform might end before buffer ends)
        assert len(wf_slice) > 0
        assert start_idx >= 0
        # Should have at least 1 second of samples before merger
        assert len(wf_slice) > 4096

    def test_cache_cleanup(self, simple_injection):
        """Test cache cleanup of expired waveforms."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Generate and cache waveform
        cache.get_waveform_slice(
            inj_id=0,
            ifo="H1",
            buf_start=999999999.0,
            buf_end=1000000001.0,
        )
        assert 0 in cache.cache

        # Cleanup with time past waveform end
        cache.cleanup_expired(current_gps=1100000000.0)
        assert 0 not in cache.cache


class TestSimInspiralSource:
    """Test cases for SimInspiralSource class."""

    def test_init_requires_injection_file(self):
        """Test that injection_file is required."""
        with pytest.raises(ValueError, match="injection_file must be specified"):
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


class TestFdToTd:
    """Test cases for frequency to time domain conversion."""

    def test_fd_to_td_roundtrip(self):
        """Test that FD to TD conversion preserves signal structure."""
        # Create a simple frequency domain signal
        n_freq = 1025
        delta_f = 1.0
        sample_rate = 2 * (n_freq - 1) * delta_f

        fd_series = lal.CreateCOMPLEX16FrequencySeries(
            "test",
            lal.LIGOTimeGPS(0),
            0.0,
            delta_f,
            lal.DimensionlessUnit,
            n_freq,
        )
        # Put a sinusoid at 100 Hz
        fd_series.data.data[:] = 0.0
        fd_series.data.data[100] = 1.0 + 0.0j

        td_series = _fd_to_td(fd_series, 1.0 / sample_rate)

        # Should have the right length
        expected_length = 2 * (n_freq - 1)
        assert td_series.data.length == expected_length

        # Should have non-zero amplitude
        assert np.max(np.abs(td_series.data.data)) > 0


class TestCoverageEdgeCases:
    """Test cases for edge cases and error handling to improve coverage."""

    def test_load_hdf5_without_injections_group(self):
        """Test HDF5 file without 'injections' group (data at root)."""
        import h5py

        # Create HDF5 file with data at root (no "injections" group)
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            filepath = f.name

        try:
            with h5py.File(filepath, "w") as hf:
                # Put data at root level, not in an "injections" group
                hf.create_dataset("mass1", data=[10.0])
                hf.create_dataset("mass2", data=[10.0])
                hf.create_dataset("spin1x", data=[0.0])
                hf.create_dataset("spin1y", data=[0.0])
                hf.create_dataset("spin1z", data=[0.0])
                hf.create_dataset("spin2x", data=[0.0])
                hf.create_dataset("spin2y", data=[0.0])
                hf.create_dataset("spin2z", data=[0.0])
                hf.create_dataset("distance", data=[100.0])
                hf.create_dataset("inclination", data=[0.0])
                hf.create_dataset("coa_phase", data=[0.0])
                hf.create_dataset("polarization", data=[0.0])
                hf.create_dataset("ra", data=[0.0])
                hf.create_dataset("dec", data=[0.0])
                hf.create_dataset("geocent_end_time", data=[1000000000.0])
                hf.create_dataset("approximant", data=[b"IMRPhenomD"])
                hf.create_dataset("f_ref", data=[20.0])

            injections = _load_hdf5_injections(filepath)
            assert len(injections) == 1
            assert injections[0].mass1 == 10.0
        finally:
            os.unlink(filepath)

    def test_load_hdf5_with_string_approximant_non_bytes(self):
        """Test HDF5 string handling when value is not bytes (defensive code)."""
        from unittest.mock import patch

        import h5py

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            filepath = f.name

        try:
            # Create a valid HDF5 file first
            with h5py.File(filepath, "w") as hf:
                grp = hf.create_group("injections")
                grp.create_dataset("mass1", data=[15.0])
                grp.create_dataset("mass2", data=[15.0])
                grp.create_dataset("spin1x", data=[0.0])
                grp.create_dataset("spin1y", data=[0.0])
                grp.create_dataset("spin1z", data=[0.0])
                grp.create_dataset("spin2x", data=[0.0])
                grp.create_dataset("spin2y", data=[0.0])
                grp.create_dataset("spin2z", data=[0.0])
                grp.create_dataset("distance", data=[200.0])
                grp.create_dataset("inclination", data=[0.0])
                grp.create_dataset("coa_phase", data=[0.0])
                grp.create_dataset("polarization", data=[0.0])
                grp.create_dataset("ra", data=[0.0])
                grp.create_dataset("dec", data=[0.0])
                grp.create_dataset("geocent_end_time", data=[1000000000.0])
                # Store as bytes normally
                grp.create_dataset("approximant", data=[b"TaylorF2"])
                grp.create_dataset("f_ref", data=[20.0])

            # Now mock the h5py read to return str instead of bytes
            # to test the non-bytes branch
            original_h5_file = h5py.File

            class MockFile:
                def __init__(self, path, mode):
                    self._file = original_h5_file(path, mode)

                def __enter__(self):
                    self._file.__enter__()
                    return MockGroup(self._file)

                def __exit__(self, *args):
                    return self._file.__exit__(*args)

            class MockGroup:
                def __init__(self, grp):
                    self._grp = grp

                def __contains__(self, key):
                    return key in self._grp

                def __getitem__(self, key):
                    if key == "injections":
                        return MockGroup(self._grp[key])
                    return MockDataset(self._grp[key])

            class MockDataset:
                def __init__(self, ds):
                    self._ds = ds

                def __len__(self):
                    return len(self._ds)

                def __getitem__(self, idx):
                    val = self._ds[idx]
                    # Return str instead of bytes for approximant
                    if isinstance(val, bytes):
                        return val.decode("utf-8")  # Return str, not bytes
                    return val

            with patch.object(h5py, "File", MockFile):
                injections = _load_hdf5_injections(filepath)
                assert len(injections) == 1
                # The non-bytes branch should handle this via str()
                assert injections[0].approximant == "TaylorF2"
        finally:
            os.unlink(filepath)

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
            assert injections[0].mass1 == 5.0
        finally:
            os.unlink(filepath)

    def test_waveform_slice_no_overlap(self, simple_injection):
        """Test get_waveform_slice when buffer doesn't overlap waveform."""
        cache = WaveformCache(
            injections=[simple_injection],
            ifos=["H1"],
            sample_rate=4096,
            f_min=20.0,
        )
        # Buffer far after waveform ends (waveform ends shortly after merger)
        wf_slice, start_idx = cache.get_waveform_slice(
            inj_id=0,
            ifo="H1",
            buf_start=1100000000.0,  # Far after merger
            buf_end=1100000001.0,
        )
        # Should return empty array since no overlap
        assert len(wf_slice) == 0
        assert start_idx == 0

    def test_estimate_duration_fallback_chirp(self):
        """Test duration estimation fallback for chirp time."""
        from unittest.mock import patch

        import lalsimulation as lalsim

        inj = InjectionParams(
            mass1=10.0,
            mass2=10.0,
            geocent_end_time=1000000000.0,
        )
        # Mock SimInspiralChirpTimeBound to raise an exception
        with patch.object(
            lalsim, "SimInspiralChirpTimeBound", side_effect=RuntimeError("test")
        ):
            pre_dur, post_dur = estimate_waveform_duration(inj, f_min=20.0)
            # Should still return valid duration using fallback formula
            assert pre_dur > 0
            assert post_dur > 0

    def test_estimate_duration_fallback_merge(self):
        """Test duration estimation fallback for merge time."""
        from unittest.mock import patch

        import lalsimulation as lalsim

        inj = InjectionParams(
            mass1=10.0,
            mass2=10.0,
            geocent_end_time=1000000000.0,
        )
        # Mock SimInspiralMergeTimeBound to raise an exception
        with patch.object(
            lalsim, "SimInspiralMergeTimeBound", side_effect=RuntimeError("test")
        ):
            pre_dur, post_dur = estimate_waveform_duration(inj, f_min=20.0)
            # Should still return valid duration using fallback
            assert pre_dur > 0
            assert post_dur > 0

    def test_estimate_duration_fallback_ringdown(self):
        """Test duration estimation fallback for ringdown time."""
        from unittest.mock import patch

        import lalsimulation as lalsim

        inj = InjectionParams(
            mass1=10.0,
            mass2=10.0,
            geocent_end_time=1000000000.0,
        )
        # Mock SimInspiralRingdownTimeBound to raise an exception
        with patch.object(
            lalsim, "SimInspiralRingdownTimeBound", side_effect=RuntimeError("test")
        ):
            pre_dur, post_dur = estimate_waveform_duration(inj, f_min=20.0)
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

    def test_load_injections_unknown_extension_hdf5_fallback(self):
        """Test loading file with unknown extension falls back to HDF5."""
        import h5py

        # Create HDF5 file with unknown extension
        with tempfile.NamedTemporaryFile(suffix=".inj", delete=False) as f:
            filepath = f.name

        try:
            with h5py.File(filepath, "w") as hf:
                grp = hf.create_group("injections")
                grp.create_dataset("mass1", data=[20.0])
                grp.create_dataset("mass2", data=[20.0])
                grp.create_dataset("spin1x", data=[0.0])
                grp.create_dataset("spin1y", data=[0.0])
                grp.create_dataset("spin1z", data=[0.0])
                grp.create_dataset("spin2x", data=[0.0])
                grp.create_dataset("spin2y", data=[0.0])
                grp.create_dataset("spin2z", data=[0.0])
                grp.create_dataset("distance", data=[300.0])
                grp.create_dataset("inclination", data=[0.0])
                grp.create_dataset("coa_phase", data=[0.0])
                grp.create_dataset("polarization", data=[0.0])
                grp.create_dataset("ra", data=[0.0])
                grp.create_dataset("dec", data=[0.0])
                grp.create_dataset("geocent_end_time", data=[1000000000.0])
                grp.create_dataset("approximant", data=[b"IMRPhenomD"])
                grp.create_dataset("f_ref", data=[20.0])

            # This should fail XML parsing then fall back to HDF5
            injections = load_injections(filepath)
            assert len(injections) == 1
            assert injections[0].mass1 == 20.0
        finally:
            os.unlink(filepath)
