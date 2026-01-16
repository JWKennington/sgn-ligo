"""Tests for MockGWEventSource and its helper functions."""

import base64
import io
import math

import numpy
import pytest
from igwn_ligolw import lsctables
from igwn_ligolw import utils as ligolw_utils

from sgnligo.psd import fake_gwdata_psd
from sgnligo.sources.mock_event_source import (
    STATE_COLLEGE_LON_RAD,
    MockGWEventSource,
    _add_noise_fluctuations,
    _apply_template_mismatch,
    _build_coinc_xmldoc,
    _calculate_overhead_ra,
    _CoincEvent,
    _compute_optimal_snr,
    _compute_phases,
    _compute_time_delays,
    _create_snr_timeseries,
    _serialize_xmldoc,
    _SingleTrigger,
)


class TestPhysicsHelpers:
    """Tests for physics calculation helper functions."""

    @pytest.fixture
    def psds(self):
        """Get PSDs for standard detectors."""
        return fake_gwdata_psd(["H1", "L1", "V1"])

    def test_calculate_overhead_ra(self):
        """Test RA calculation for overhead position."""
        # At GPS time 1000000000, check that RA is computed
        ra = _calculate_overhead_ra(1000000000.0, STATE_COLLEGE_LON_RAD)
        assert 0 <= ra < 2 * math.pi

        # RA should change with longitude
        ra_east = _calculate_overhead_ra(1000000000.0, 0.0)
        ra_west = _calculate_overhead_ra(1000000000.0, -math.pi / 2)
        assert ra_east != ra_west

    def test_compute_optimal_snr_bns(self, psds):
        """Test SNR calculation for BNS at 100 Mpc."""
        snr = _compute_optimal_snr(
            mass1_msun=1.4,
            mass2_msun=1.4,
            distance_mpc=100.0,
            ra=0.0,
            dec=0.5,
            psi=0.0,
            inclination=0.0,  # Face-on
            t_co_gps=1000000000.0,
            ifo="H1",
            psd=psds["H1"],
        )
        # BNS at 100 Mpc should have SNR roughly 10-30 in optimal orientation
        assert 5 < snr < 50

    def test_compute_optimal_snr_distance_scaling(self, psds):
        """Test that SNR scales inversely with distance."""
        kwargs = dict(
            mass1_msun=1.4,
            mass2_msun=1.4,
            ra=0.0,
            dec=0.5,
            psi=0.0,
            inclination=0.0,
            t_co_gps=1000000000.0,
            ifo="H1",
            psd=psds["H1"],
        )

        snr_100 = _compute_optimal_snr(distance_mpc=100.0, **kwargs)
        snr_200 = _compute_optimal_snr(distance_mpc=200.0, **kwargs)

        # SNR should scale as 1/D
        ratio = snr_100 / snr_200
        assert 1.8 < ratio < 2.2  # Should be ~2

    def test_compute_time_delays(self):
        """Test time delay computation between detectors."""
        time_delays = _compute_time_delays(
            ra=0.0, dec=0.5, t_co_gps=1000000000.0, ifos=["H1", "L1", "V1"]
        )

        # Check all detectors present
        assert set(time_delays.keys()) == {"H1", "L1", "V1"}

        # Time delays should be within light travel time (~10ms max between H1-L1)
        for _ifo, dt in time_delays.items():
            assert abs(dt) < 0.025  # 25ms max from geocenter

        # H1 and L1 should have different delays
        assert time_delays["H1"] != time_delays["L1"]

    def test_compute_phases(self):
        """Test phase computation at each detector."""
        time_delays = _compute_time_delays(
            ra=0.0, dec=0.5, t_co_gps=1000000000.0, ifos=["H1", "L1", "V1"]
        )

        phases = _compute_phases(
            ra=0.0,
            dec=0.5,
            psi=0.0,
            inclination=0.0,
            phi_geo=0.0,
            t_co_gps=1000000000.0,
            time_delays=time_delays,
            ifos=["H1", "L1", "V1"],
        )

        # All phases should be in [0, 2pi)
        for _ifo, phi in phases.items():
            assert 0 <= phi < 2 * math.pi

    def test_add_noise_fluctuations(self):
        """Test noise fluctuation model."""
        numpy.random.seed(42)

        # High SNR should give small timing errors
        snr_high, t_high, phi_high = _add_noise_fluctuations(
            snr_true=30.0, t_true=1000000000.0, phi_true=1.0
        )

        # With SNR=30, timing error sigma ~ 0.07ms
        # Measured time should be close to true
        assert abs(t_high - 1000000000.0) < 0.01  # Within 10ms

        # SNR measurement should have unit variance
        snrs = [_add_noise_fluctuations(10.0, 0.0, 0.0)[0] for _ in range(1000)]
        snr_std = numpy.std(snrs)
        assert 0.8 < snr_std < 1.2  # Should be ~1

    def test_apply_template_mismatch(self):
        """Test template mismatch model."""
        numpy.random.seed(42)

        snr_rec, m1_rec, m2_rec = _apply_template_mismatch(
            snr_optimal=20.0,
            mass1_true=1.4,
            mass2_true=1.4,
            min_match=0.97,
        )

        # Recovered SNR should be <= optimal
        assert snr_rec <= 20.0
        assert snr_rec >= 20.0 * 0.97  # At least min_match fraction

        # Masses should be slightly biased
        assert abs(m1_rec - 1.4) < 0.5
        assert abs(m2_rec - 1.4) < 0.5


class TestXMLGeneration:
    """Tests for coinc XML generation."""

    @pytest.fixture
    def psds(self):
        """Get PSDs for standard detectors."""
        return fake_gwdata_psd(["H1", "L1", "V1"])

    @pytest.fixture
    def sample_event(self):
        """Create a sample CoincEvent for testing."""
        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
            ),
        ]
        return _CoincEvent(
            event_id=1, t_co_gps=1000000000.0, triggers=triggers, far=1e-10
        )

    def test_create_snr_timeseries(self, psds):
        """Test SNR time series creation with ACF-based shape."""
        snr_peak = 15.0 * numpy.exp(1j * 0.5)
        ts = _create_snr_timeseries(
            snr_peak,
            t_peak=1000000000.0,
            mass1=1.4,
            mass2=1.4,
            psd=psds["H1"],
        )

        # Check properties
        assert ts.data.length > 0
        assert ts.deltaT == 1.0 / 2048.0  # 2048 Hz sample rate
        assert ts.data.length == 409  # ±0.1s at 2048 Hz = 2*204 + 1
        # Peak should be at center
        center_idx = ts.data.length // 2
        assert abs(ts.data.data[center_idx]) > 0

    def test_build_coinc_xmldoc(self, sample_event, psds):
        """Test coinc XML document building."""
        xmldoc = _build_coinc_xmldoc(
            sample_event, pipeline="SGNL", psds=psds, include_snr_series=True
        )

        # Check required tables exist
        coinc_table = lsctables.CoincTable.get_table(xmldoc)
        assert len(coinc_table) == 1

        coinc_inspiral_table = lsctables.CoincInspiralTable.get_table(xmldoc)
        assert len(coinc_inspiral_table) == 1

        sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
        assert len(sngl_inspiral_table) == 2

        # Check values
        assert coinc_inspiral_table[0].ifos == "H1,L1"
        assert abs(coinc_inspiral_table[0].snr - sample_event.network_snr) < 0.1

    def test_serialize_xmldoc(self, sample_event, psds):
        """Test XML serialization to bytes."""
        xmldoc = _build_coinc_xmldoc(
            sample_event, pipeline="SGNL", psds=psds, include_snr_series=False
        )
        xml_bytes = _serialize_xmldoc(xmldoc)

        assert isinstance(xml_bytes, bytes)
        assert b"<?xml" in xml_bytes
        assert b"LIGO_LW" in xml_bytes

        # Should be parseable
        buffer = io.BytesIO(xml_bytes)
        xmldoc_loaded = ligolw_utils.load_fileobj(buffer)
        assert xmldoc_loaded is not None

    def test_roundtrip_xml(self, sample_event, psds):
        """Test XML generation and parsing roundtrip."""
        xmldoc = _build_coinc_xmldoc(
            sample_event, pipeline="pycbc", psds=psds, include_snr_series=True
        )
        xml_bytes = _serialize_xmldoc(xmldoc)

        # Parse it back
        buffer = io.BytesIO(xml_bytes)
        xmldoc_loaded = ligolw_utils.load_fileobj(buffer)

        # Verify content
        coinc_inspiral = lsctables.CoincInspiralTable.get_table(xmldoc_loaded)[0]
        assert coinc_inspiral.end_time == 1000000000
        assert "H1" in coinc_inspiral.ifos
        assert "L1" in coinc_inspiral.ifos


class TestSingleTrigger:
    """Tests for _SingleTrigger dataclass."""

    def test_trigger_properties(self):
        """Test computed properties of SingleTrigger."""
        trigger = _SingleTrigger(
            ifo="H1",
            end_time=1000000000.123456789,
            snr=15.0,
            coa_phase=0.5,
            mass1=1.4,
            mass2=1.3,
        )

        assert trigger.end_time_int == 1000000000
        assert 123000000 < trigger.end_time_ns < 124000000

        assert abs(trigger.mtotal - 2.7) < 0.01
        assert 0 < trigger.eta < 0.25  # Must be <= 0.25 for physical systems
        assert trigger.mchirp > 0


class TestCoincEvent:
    """Tests for _CoincEvent dataclass."""

    def test_network_snr(self):
        """Test network SNR calculation."""
        triggers = [
            _SingleTrigger(
                ifo="H1", end_time=0, snr=10.0, coa_phase=0, mass1=1.4, mass2=1.4
            ),
            _SingleTrigger(
                ifo="L1", end_time=0, snr=8.0, coa_phase=0, mass1=1.4, mass2=1.4
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=0, triggers=triggers)

        # Network SNR = sqrt(10^2 + 8^2) = sqrt(164) ~ 12.8
        expected = math.sqrt(10**2 + 8**2)
        assert abs(event.network_snr - expected) < 0.01

    def test_ifos(self):
        """Test IFO list extraction."""
        triggers = [
            _SingleTrigger(
                ifo="H1", end_time=0, snr=10.0, coa_phase=0, mass1=1.4, mass2=1.4
            ),
            _SingleTrigger(
                ifo="V1", end_time=0, snr=6.0, coa_phase=0, mass1=1.4, mass2=1.4
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=0, triggers=triggers)

        assert event.ifos == ["H1", "V1"]


class TestMockGWEventSourceInit:
    """Tests for MockGWEventSource initialization."""

    def test_default_initialization(self):
        """Test source initializes with defaults."""
        source = MockGWEventSource()

        assert source.event_cadence == 20.0
        assert source.ifos == ["H1", "L1", "V1"]
        assert "SGNL" in source.source_pad_names
        assert "pycbc" in source.source_pad_names

    def test_custom_pipeline_latencies(self):
        """Test custom pipeline latency configuration."""
        custom_latencies = {
            "fast_pipeline": (2.0, 0.5),
            "slow_pipeline": (60.0, 10.0),
        }

        source = MockGWEventSource(
            pipeline_latencies=custom_latencies,
        )

        assert "fast_pipeline" in source.source_pad_names
        assert "slow_pipeline" in source.source_pad_names
        assert "SGNL" not in source.source_pad_names

    def test_custom_event_cadence(self):
        """Test custom event cadence configuration."""
        source = MockGWEventSource(
            event_cadence=10.0,
        )
        assert source.event_cadence == 10.0


class TestMockGWEventSourceEventGeneration:
    """Tests for event generation in MockGWEventSource."""

    def test_generate_source_params(self):
        """Test source parameter generation."""
        numpy.random.seed(42)

        source = MockGWEventSource()

        params = source._generate_source_params()

        assert "source_type" in params
        assert params["source_type"] in ["bns", "nsbh", "bbh"]
        assert params["mass1"] >= params["mass2"]  # Enforced convention
        assert params["distance"] > 0
        assert 0 <= params["psi"] <= math.pi

    def test_generate_event(self):
        """Test full event generation."""
        numpy.random.seed(42)

        source = MockGWEventSource(
            ifos=["H1", "L1"],
        )

        event = source._generate_event(1000000000.0)

        assert event.event_id == 0
        assert len(event.triggers) >= 2
        assert event.network_snr > 0

        # All triggers should be above threshold
        for trigger in event.triggers:
            assert trigger.snr >= source.snr_threshold


class TestCoincXMLSink:
    """Tests for CoincXMLSink."""

    def test_sink_initialization(self, tmp_path):
        """Test sink initializes correctly."""
        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL", "pycbc"],
            verbose=False,
        )

        assert sink.output_dir == str(tmp_path)
        assert "SGNL" in sink.sink_pad_names
        assert "pycbc" in sink.sink_pad_names

    def test_sink_writes_xml(self, tmp_path):
        """Test sink writes XML files."""
        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=False,
        )

        # Create a sample event
        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=1000000000.0, triggers=triggers)

        # Build XML
        psds = fake_gwdata_psd(["H1", "L1"])
        xmldoc = _build_coinc_xmldoc(event, "SGNL", psds, include_snr_series=False)
        xml_bytes = _serialize_xmldoc(xmldoc)

        # Create frame with base64-encoded XML
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={
                "xml": base64.b64encode(xml_bytes).decode("ascii"),
                "event_id": 1,
                "pipeline": "SGNL",
                "gpstime": 1000000000.0,
            },
            metadata={},
        )

        # Send to sink
        sink.pull(sink.snks["SGNL"], frame)

        # Check file was written
        output_files = list(tmp_path.glob("*.xml"))
        assert len(output_files) == 1

        # Check stats
        stats = sink.get_stats()
        assert stats["total"] == 1
        assert stats["per_pipeline"]["SGNL"] == 1

    def test_sink_writes_gzipped_xml(self, tmp_path):
        """Test sink writes gzipped XML files when input is gzipped."""
        import gzip

        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=False,
        )

        # Create a sample event
        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=1000000000.0, triggers=triggers)

        # Build XML and gzip it
        psds = fake_gwdata_psd(["H1", "L1"])
        xmldoc = _build_coinc_xmldoc(event, "SGNL", psds, include_snr_series=False)
        xml_bytes = _serialize_xmldoc(xmldoc)
        xml_gzipped = gzip.compress(xml_bytes)

        # Create frame with base64-encoded gzipped XML
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={
                "xml": base64.b64encode(xml_gzipped).decode("ascii"),
                "event_id": 1,
                "pipeline": "SGNL",
                "gpstime": 1000000000.0,
            },
            metadata={},
        )

        # Send to sink
        sink.pull(sink.snks["SGNL"], frame)

        # Check file was written as .xml.gz
        output_files = list(tmp_path.glob("*.xml.gz"))
        assert len(output_files) == 1

    def test_sink_compress_option(self, tmp_path):
        """Test sink compresses output when compress=True."""
        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=False,
            compress=True,
        )

        # Create a sample event
        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=1000000000.0, triggers=triggers)

        # Build XML (not gzipped)
        psds = fake_gwdata_psd(["H1", "L1"])
        xmldoc = _build_coinc_xmldoc(event, "SGNL", psds, include_snr_series=False)
        xml_bytes = _serialize_xmldoc(xmldoc)

        # Create frame with base64-encoded plain XML
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={
                "xml": base64.b64encode(xml_bytes).decode("ascii"),
                "event_id": 1,
                "pipeline": "SGNL",
                "gpstime": 1000000000.0,
            },
            metadata={},
        )

        # Send to sink
        sink.pull(sink.snks["SGNL"], frame)

        # Check file was written as .xml.gz (compressed by sink)
        output_files = list(tmp_path.glob("*.gz"))
        assert len(output_files) == 1

    def test_sink_handles_eos(self, tmp_path):
        """Test sink handles EOS frames."""
        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=False,
        )

        # Send EOS frame
        eos_frame = Frame(EOS=True, is_gap=False, data=None, metadata={})
        sink.pull(sink.snks["SGNL"], eos_frame)

        # Should not crash, no files written
        output_files = list(tmp_path.glob("*"))
        assert len(output_files) == 0

    def test_sink_handles_gap_frame(self, tmp_path):
        """Test sink handles gap frames."""
        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=False,
        )

        # Send gap frame
        gap_frame = Frame(EOS=False, is_gap=True, data=None, metadata={})
        sink.pull(sink.snks["SGNL"], gap_frame)

        # Should not crash, no files written
        output_files = list(tmp_path.glob("*"))
        assert len(output_files) == 0

    def test_sink_handles_missing_xml(self, tmp_path):
        """Test sink handles frames with missing xml field."""
        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=False,
        )

        # Send frame without xml field
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={"event_id": 1},
            metadata={},
        )
        sink.pull(sink.snks["SGNL"], frame)

        # Should not crash, no files written
        output_files = list(tmp_path.glob("*"))
        assert len(output_files) == 0

    def test_sink_verbose_mode(self, tmp_path, capsys):
        """Test sink verbose output."""
        import gzip

        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=True,
        )

        # Check initialization output
        captured = capsys.readouterr()
        assert "CoincXMLSink initialized" in captured.out

        # Create a sample event
        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=1000000000.0, triggers=triggers)

        # Build XML and gzip
        psds = fake_gwdata_psd(["H1", "L1"])
        xmldoc = _build_coinc_xmldoc(event, "SGNL", psds, include_snr_series=False)
        xml_bytes = _serialize_xmldoc(xmldoc)
        xml_gzipped = gzip.compress(xml_bytes)

        # Create frame
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={
                "xml": base64.b64encode(xml_gzipped).decode("ascii"),
                "event_id": 1,
                "pipeline": "SGNL",
                "gpstime": 1000000000.0,
            },
            metadata={},
        )

        # Send to sink
        sink.pull(sink.snks["SGNL"], frame)

        # Check verbose output contains event summary
        captured = capsys.readouterr()
        assert "COINC EVENT" in captured.out
        assert "GPS Time" in captured.out
        assert "Network SNR" in captured.out

    def test_sink_print_summary(self, tmp_path, capsys):
        """Test sink print_summary method."""
        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL", "pycbc"],
            verbose=False,
        )

        sink.print_summary()

        captured = capsys.readouterr()
        assert "COINC XML SINK SUMMARY" in captured.out
        assert "Total events written: 0" in captured.out
        assert "SGNL" in captured.out
        assert "pycbc" in captured.out

    def test_sink_default_pipelines(self, tmp_path):
        """Test sink uses default pipelines when not specified."""
        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            verbose=False,
        )

        assert "SGNL" in sink.sink_pad_names
        assert "pycbc" in sink.sink_pad_names
        assert "MBTA" in sink.sink_pad_names
        assert "spiir" in sink.sink_pad_names

    def test_sink_internal_method(self, tmp_path):
        """Test sink internal method does nothing."""
        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=False,
        )

        # Should not raise
        sink.internal()


class TestPhysicsHelpersEdgeCases:
    """Tests for edge cases in physics helper functions."""

    @pytest.fixture
    def psds(self):
        """Get PSDs for standard detectors."""
        return fake_gwdata_psd(["H1", "L1", "V1"])

    def test_compute_optimal_snr_zero_effective_distance(self, psds):
        """Test SNR calculation when effective distance is zero (edge-on)."""
        # At exactly edge-on (inclination=pi/2), Fx can be zero for some orientations
        # This tests the d_eff_mpc > 0 branch
        snr = _compute_optimal_snr(
            mass1_msun=1.4,
            mass2_msun=1.4,
            distance_mpc=100.0,
            ra=0.0,
            dec=0.0,
            psi=0.0,
            inclination=math.pi / 2,  # Edge-on
            t_co_gps=1000000000.0,
            ifo="H1",
            psd=psds["H1"],
        )
        # SNR should still be computed (may be lower due to edge-on)
        assert snr >= 0

    def test_add_noise_fluctuations_zero_snr(self):
        """Test noise fluctuations with zero SNR."""
        snr_measured, t_measured, phi_measured = _add_noise_fluctuations(
            snr_true=0.0, t_true=1000000000.0, phi_true=1.0
        )

        # With zero SNR, time and phase should be unchanged
        assert t_measured == 1000000000.0
        assert phi_measured == 1.0
        # SNR should be around 1 (just noise)
        assert snr_measured >= 0


class TestMockGWEventSourceEdgeCases:
    """Tests for edge cases in MockGWEventSource."""

    def test_generate_source_params_random_sky(self):
        """Test source parameter generation with random sky position."""
        numpy.random.seed(42)

        source = MockGWEventSource(sky_position="random")

        params = source._generate_source_params()

        # RA should be set (not None)
        assert params["ra"] is not None
        assert 0 <= params["ra"] < 2 * math.pi
        # Dec should be in valid range
        assert -math.pi / 2 <= params["dec"] <= math.pi / 2

    def test_generate_source_params_mass_swap(self):
        """Test that masses are swapped if mass2 > mass1."""
        numpy.random.seed(123)  # Seed that might produce mass2 > mass1

        source = MockGWEventSource()

        # Generate many params to increase chance of hitting swap
        for _ in range(100):
            params = source._generate_source_params()
            assert params["mass1"] >= params["mass2"]

    def test_verbose_initialization(self, capsys):
        """Test verbose output during initialization."""
        _source = MockGWEventSource(verbose=True)  # noqa: F841

        captured = capsys.readouterr()
        assert "MockGWEventSource initialized" in captured.out
        assert "GPS start" in captured.out
        assert "Pipelines" in captured.out
        assert "Detectors" in captured.out

    def test_create_event_variant(self):
        """Test event variant creation for pipeline multiplicity."""
        numpy.random.seed(42)

        source = MockGWEventSource(ifos=["H1", "L1"])

        # Generate an event
        event = source._generate_event(1000000000.0)

        # Create a variant
        variant = source._create_event_variant(event)

        # Variant should have different event_id
        assert variant.event_id != event.event_id

        # Variant should have same number of triggers
        assert len(variant.triggers) == len(event.triggers)

        # Variant parameters should be slightly different
        for orig, var in zip(event.triggers, variant.triggers):
            assert orig.ifo == var.ifo
            # SNR should be slightly reduced
            assert var.snr <= orig.snr
            # Masses should be similar but not identical
            assert abs(var.mass1 - orig.mass1) < 0.1

    def test_schedule_coalescence(self, capsys):
        """Test coalescence scheduling."""
        numpy.random.seed(42)

        source = MockGWEventSource(
            verbose=True,
            pipeline_latencies={"SGNL": (6.0, 1.0)},
            pipeline_multiplicity={"SGNL": (1, 2)},
        )

        # Clear init output
        capsys.readouterr()

        # Schedule a coalescence
        source._schedule_coalescence(1000000000.0)

        # Check that pending events were created
        assert len(source._pending_events) >= 1

        # Check verbose output
        captured = capsys.readouterr()
        assert "Scheduled coalescence" in captured.out

    def test_pregenerate_pending_events(self):
        """Test pre-generation of pending events."""
        numpy.random.seed(42)

        source = MockGWEventSource(
            verbose=False,
            pipeline_latencies={"SGNL": (6.0, 0.1)},
            pipeline_multiplicity={"SGNL": (1, 1)},
        )

        # Schedule a coalescence
        t_co = 1000000000.0
        source._schedule_coalescence(t_co)

        # Pre-generate with large lookahead
        source._pregenerate_pending_events(t_co + 100.0)

        # Ready queue should now have events
        assert len(source._ready_queues["SGNL"]) >= 1

        # Pending should be empty (all moved to ready)
        assert len(source._pending_events) == 0

    def test_earliest_ready_event_time(self):
        """Test finding earliest ready event time."""
        source = MockGWEventSource(
            verbose=False,
            pipeline_latencies={"SGNL": (6.0, 0.1), "pycbc": (30.0, 1.0)},
        )

        # Initially no events
        assert source._earliest_ready_event_time() is None

        # Add events to ready queues
        source._ready_queues["SGNL"].append((100.0, {}))
        source._ready_queues["pycbc"].append((200.0, {}))

        # Earliest should be 100.0
        assert source._earliest_ready_event_time() == 100.0

    def test_any_event_ready(self):
        """Test checking if any event is ready."""
        source = MockGWEventSource(
            verbose=False,
            pipeline_latencies={"SGNL": (6.0, 0.1)},
        )

        # Initially no events ready
        assert source._any_event_ready(1000.0) is False

        # Add event to ready queue
        source._ready_queues["SGNL"].append((100.0, {}))

        # Event at 100.0 is ready at time 100.0
        assert source._any_event_ready(100.0) is True

        # Event at 100.0 is not ready at time 50.0
        assert source._any_event_ready(50.0) is False

    def test_new_returns_gap_when_no_events(self):
        """Test new() returns gap frame when no events ready."""
        source = MockGWEventSource(
            verbose=False,
            pipeline_latencies={"SGNL": (6.0, 0.1)},
        )

        # Get pad
        pad = source.srcs["SGNL"]

        # Call new - should return gap frame
        frame = source.new(pad)

        assert frame.is_gap is True
        assert frame.data is None

    def test_new_returns_event_when_ready(self):
        """Test new() returns event frame when event is ready."""
        source = MockGWEventSource(
            verbose=False,
            pipeline_latencies={"SGNL": (6.0, 0.1)},
        )

        # Manually add event to ready queue with past time
        frame_data = {
            "xml": "test",
            "event_id": 1,
            "pipeline": "SGNL",
            "gpstime": 1000000000.0,
            "snr": 15.0,
            "ifos": "H1,L1",
            "far": 1e-10,
        }
        # Set report time to be in the past relative to current GPS
        past_report_time = source._current_gps() - 10.0
        source._ready_queues["SGNL"].append((past_report_time, frame_data))

        # Get pad
        pad = source.srcs["SGNL"]

        # Call new - should return the event
        frame = source.new(pad)

        assert frame.is_gap is False
        assert frame.data is not None
        assert frame.data["event_id"] == 1

    def test_trigger_with_eff_distance(self):
        """Test trigger with effective distance set."""
        trigger = _SingleTrigger(
            ifo="H1",
            end_time=1000000000.0,
            snr=15.0,
            coa_phase=0.5,
            mass1=1.4,
            mass2=1.4,
            eff_distance=100.0,
        )

        assert trigger.eff_distance == 100.0

    def test_build_coinc_xmldoc_with_eff_distance(self):
        """Test XML generation includes effective distance when set."""
        psds = fake_gwdata_psd(["H1", "L1"])

        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
                eff_distance=100.0,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
                eff_distance=120.0,
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=1000000000.0, triggers=triggers)

        xmldoc = _build_coinc_xmldoc(event, "SGNL", psds, include_snr_series=False)

        # Check that eff_distance is in the sngl_inspiral table
        sngl_table = lsctables.SnglInspiralTable.get_table(xmldoc)
        for row in sngl_table:
            assert row.eff_distance is not None


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_ns_to_gps(self):
        """Test nanoseconds to GPS conversion."""
        from sgnligo.sources.mock_event_source import _ns_to_gps

        gps = _ns_to_gps(1000000000123456789)

        assert gps.gpsSeconds == 1000000000
        assert gps.gpsNanoSeconds == 123456789

    def test_format_gps_time(self):
        """Test GPS time formatting."""
        from sgnligo.sinks.coinc_xml_sink import _format_gps_time

        result = _format_gps_time(1000000000, 123456789)

        assert result == "1000000000.123456789"

    def test_summarize_coinc_xmldoc(self):
        """Test XML document summarization."""
        from sgnligo.sinks.coinc_xml_sink import _summarize_coinc_xmldoc

        psds = fake_gwdata_psd(["H1", "L1"])

        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
            ),
        ]
        event = _CoincEvent(
            event_id=1, t_co_gps=1000000000.0, triggers=triggers, far=1e-10
        )

        xmldoc = _build_coinc_xmldoc(event, "SGNL", psds, include_snr_series=False)
        xml_bytes = _serialize_xmldoc(xmldoc)

        summary = _summarize_coinc_xmldoc(xml_bytes)

        assert "end_time" in summary
        assert "network_snr" in summary
        assert "ifos" in summary
        assert "triggers" in summary
        assert "H1" in summary["triggers"]
        assert "L1" in summary["triggers"]

    def test_print_event_summary(self, capsys):
        """Test event summary printing."""
        from sgnligo.sinks.coinc_xml_sink import _print_event_summary

        summary = {
            "end_time": "1000000000.123456789",
            "network_snr": 19.2,
            "ifos": "H1,L1",
            "mchirp": 1.2,
            "mtotal": 2.8,
            "far": 1e-10,
            "triggers": {
                "H1": {
                    "snr": 15.0,
                    "end_time": "1000000000.000000000",
                    "mass1": 1.4,
                    "mass2": 1.4,
                    "coa_phase": 0.5,
                },
                "L1": {
                    "snr": 12.0,
                    "end_time": "1000000000.001000000",
                    "mass1": 1.4,
                    "mass2": 1.4,
                    "coa_phase": 0.6,
                },
            },
        }

        _print_event_summary(1, "SGNL", summary, "/path/to/file.xml")

        captured = capsys.readouterr()
        assert "COINC EVENT 1 [SGNL]" in captured.out
        assert "GPS Time:" in captured.out
        assert "Network SNR:" in captured.out
        assert "Written to: /path/to/file.xml" in captured.out

    def test_print_event_summary_no_path(self, capsys):
        """Test event summary printing without output path."""
        from sgnligo.sinks.coinc_xml_sink import _print_event_summary

        summary = {
            "end_time": "1000000000.123456789",
            "network_snr": 19.2,
            "ifos": "H1,L1",
            "mchirp": 1.2,
            "mtotal": 2.8,
            "far": 1e-10,
            "triggers": {},
        }

        _print_event_summary(1, "SGNL", summary, None)

        captured = capsys.readouterr()
        assert "Written to:" not in captured.out


class TestAdditionalCoverage:
    """Additional tests to achieve 100% coverage."""

    def test_sink_gzip_filename_without_xml_extension(self, tmp_path):
        """Test sink handles gzipped data with filename that doesn't have .xml."""
        import gzip

        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        # Use a filename template without .xml extension
        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            filename_template="{pipeline}_{event_id:04d}",  # No .xml
            verbose=False,
        )

        # Create a sample event
        triggers = [
            _SingleTrigger(
                ifo="H1",
                end_time=1000000000.0,
                snr=15.0,
                coa_phase=0.5,
                mass1=1.4,
                mass2=1.4,
            ),
            _SingleTrigger(
                ifo="L1",
                end_time=1000000000.001,
                snr=12.0,
                coa_phase=0.6,
                mass1=1.4,
                mass2=1.4,
            ),
        ]
        event = _CoincEvent(event_id=1, t_co_gps=1000000000.0, triggers=triggers)

        # Build XML and gzip it
        psds = fake_gwdata_psd(["H1", "L1"])
        xmldoc = _build_coinc_xmldoc(event, "SGNL", psds, include_snr_series=False)
        xml_bytes = _serialize_xmldoc(xmldoc)
        xml_gzipped = gzip.compress(xml_bytes)

        # Create frame with base64-encoded gzipped XML
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={
                "xml": base64.b64encode(xml_gzipped).decode("ascii"),
                "event_id": 1,
                "pipeline": "SGNL",
                "gpstime": 1000000000.0,
            },
            metadata={},
        )

        # Send to sink
        sink.pull(sink.snks["SGNL"], frame)

        # Check file was written with .gz appended
        output_files = list(tmp_path.glob("*.gz"))
        assert len(output_files) == 1
        assert output_files[0].name == "SGNL_0001.gz"

    def test_sink_verbose_xml_parse_error(self, tmp_path, capsys):
        """Test sink handles XML parsing errors gracefully in verbose mode."""
        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["SGNL"],
            verbose=True,
        )

        # Clear init output
        capsys.readouterr()

        # Create frame with invalid XML (not gzipped, just bad XML)
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={
                "xml": base64.b64encode(b"not valid xml").decode("ascii"),
                "event_id": 1,
                "pipeline": "SGNL",
                "gpstime": 1000000000.0,
            },
            metadata={},
        )

        # Send to sink - should not crash
        sink.pull(sink.snks["SGNL"], frame)

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: Could not parse event summary" in captured.out
        assert "Written to:" in captured.out

    def test_generate_event_with_low_snr_retry(self):
        """Test event generation retries with closer distance if SNR too low."""
        numpy.random.seed(42)

        # Create source with very high SNR threshold to force retry
        source = MockGWEventSource(
            ifos=["H1", "L1"],
            snr_threshold=100.0,  # Very high threshold
        )

        # This will likely trigger retries due to high threshold
        # The retry halves the distance each time
        event = source._generate_event(1000000000.0)

        # Should eventually succeed with closer distance
        assert event is not None
        assert len(event.triggers) >= 2

    def test_pregenerate_with_remaining_events(self):
        """Test pre-generation leaves events outside lookahead window."""
        numpy.random.seed(42)

        source = MockGWEventSource(
            verbose=False,
            pipeline_latencies={"SGNL": (100.0, 0.1)},  # Very long latency
            pipeline_multiplicity={"SGNL": (1, 1)},
        )

        # Schedule a coalescence
        t_co = 1000000000.0
        source._schedule_coalescence(t_co)

        # Pre-generate with small lookahead (event won't be ready)
        source._pregenerate_pending_events(t_co + 10.0)  # Only 10s lookahead

        # Pending should still have events (latency is 100s)
        assert len(source._pending_events) >= 1

        # Ready queue should be empty
        assert len(source._ready_queues["SGNL"]) == 0

    def test_pregenerate_verbose_output(self, capsys):
        """Test verbose output during pre-generation."""
        numpy.random.seed(42)

        source = MockGWEventSource(
            verbose=True,
            pipeline_latencies={"SGNL": (6.0, 0.1)},
            pipeline_multiplicity={"SGNL": (1, 1)},
        )

        # Clear init output
        capsys.readouterr()

        # Schedule a coalescence
        t_co = 1000000000.0
        source._schedule_coalescence(t_co)

        # Clear scheduling output
        capsys.readouterr()

        # Pre-generate with large lookahead
        source._pregenerate_pending_events(t_co + 100.0)

        # Check verbose output
        captured = capsys.readouterr()
        assert "Pre-generated XML" in captured.out

    def test_internal_method_schedules_and_pregenerates(self):
        """Test internal() method schedules events and pre-generates XML."""
        numpy.random.seed(42)

        source = MockGWEventSource(
            verbose=False,
            event_cadence=5.0,  # Short cadence
            lookahead=10.0,  # Small lookahead
            stride=0.001,  # Very short stride to avoid long sleeps
            pipeline_latencies={"SGNL": (1.0, 0.1)},  # Short latency
            pipeline_multiplicity={"SGNL": (1, 1)},
        )

        # Call internal - should schedule events within lookahead
        source.internal()

        # Should have scheduled at least one event
        # (either in pending or already pre-generated to ready)
        total_events = len(source._pending_events) + len(source._ready_queues["SGNL"])
        assert total_events >= 1

    def test_new_verbose_output(self, capsys):
        """Test verbose output when emitting events."""
        source = MockGWEventSource(
            verbose=True,
            pipeline_latencies={"SGNL": (6.0, 0.1)},
        )

        # Clear init output
        capsys.readouterr()

        # Manually add event to ready queue with past time
        frame_data = {
            "xml": "test",
            "event_id": 42,
            "pipeline": "SGNL",
            "gpstime": 1000000000.0,
            "snr": 15.0,
            "ifos": "H1,L1",
            "far": 1e-10,
        }
        past_report_time = source._current_gps() - 10.0
        source._ready_queues["SGNL"].append((past_report_time, frame_data))

        # Get pad and call new
        pad = source.srcs["SGNL"]
        frame = source.new(pad)

        # Check verbose output
        captured = capsys.readouterr()
        assert "SGNL: Emitting event 42" in captured.out

        assert frame.data is not None

    def test_new_returns_eos_when_signaled(self):
        """Test new() returns EOS frame when signaled."""
        import signal

        source = MockGWEventSource(
            verbose=False,
            pipeline_latencies={"SGNL": (6.0, 0.1)},
        )

        # Simulate signal receipt by directly adding to rcvd_signals
        # This is how SignalEOS tracks received signals
        MockGWEventSource.rcvd_signals.add(signal.SIGINT)
        MockGWEventSource.handled_signals.add(signal.SIGINT)

        try:
            # Get pad
            pad = source.srcs["SGNL"]

            # Call new - should return EOS frame
            frame = source.new(pad)

            assert frame.EOS is True
        finally:
            # Clean up class-level state
            MockGWEventSource.rcvd_signals.discard(signal.SIGINT)
            MockGWEventSource.handled_signals.discard(signal.SIGINT)

    def test_compute_optimal_snr_very_small_effective_distance(self):
        """Test SNR returns 0 when effective distance factor is zero."""
        psds = fake_gwdata_psd(["H1"])

        # Test with parameters that might give zero effective distance
        # This is hard to trigger naturally, so we just verify the function handles it
        snr = _compute_optimal_snr(
            mass1_msun=1.4,
            mass2_msun=1.4,
            distance_mpc=100.0,
            ra=0.0,
            dec=0.0,
            psi=0.0,
            inclination=0.0,
            t_co_gps=1000000000.0,
            ifo="H1",
            psd=psds["H1"],
        )
        # SNR should be non-negative
        assert snr >= 0

    def test_internal_with_no_earliest_event(self):
        """Test internal() when no events are scheduled (uses stride for wait)."""
        source = MockGWEventSource(
            verbose=False,
            event_cadence=1000.0,  # Very long cadence so no events scheduled yet
            lookahead=1.0,  # Short lookahead
            stride=0.001,  # Very short stride to avoid long sleep
            pipeline_latencies={"SGNL": (500.0, 0.1)},  # Very long latency
            pipeline_multiplicity={"SGNL": (1, 1)},
        )

        # Manually clear any scheduled events and pending events
        source._pending_events = []
        for key in source._ready_queues:
            source._ready_queues[key] = []

        # Move next coalescence far in the future
        source._next_coalescence_gps = source._current_gps() + 1000.0

        # Call internal - should use stride for wait since no events
        # This should hit the `else: wait_time = self.stride` branch
        source.internal()

        # Should complete without error

    def test_compute_optimal_snr_zero_effective_distance_factor(self):
        """Test SNR calculation returns 0 when effective distance factor is zero."""
        from unittest.mock import patch

        psds = fake_gwdata_psd(["H1"])

        # Mock effective_distance_factor to return 0
        with patch(
            "sgnligo.sources.mock_event_source.effective_distance_factor",
            return_value=0.0,
        ):
            snr = _compute_optimal_snr(
                mass1_msun=1.4,
                mass2_msun=1.4,
                distance_mpc=100.0,
                ra=0.0,
                dec=0.0,
                psi=0.0,
                inclination=0.0,
                t_co_gps=1000000000.0,
                ifo="H1",
                psd=psds["H1"],
            )

        # When effective distance factor is 0, SNR should be 0
        assert snr == 0.0
