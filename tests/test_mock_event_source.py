"""Tests for MockGWEventSource and its helper functions."""

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
            sample_event, pipeline="sgnl", psds=psds, include_snr_series=True
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
            sample_event, pipeline="sgnl", psds=psds, include_snr_series=False
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
        source = MockGWEventSource(
            t0=1000000000.0,
            duration=100.0,
            real_time=False,
        )

        assert source.event_cadence == 20.0
        assert source.ifos == ["H1", "L1", "V1"]
        assert "sgnl" in source.source_pad_names
        assert "pycbc" in source.source_pad_names

    def test_custom_pipeline_latencies(self):
        """Test custom pipeline latency configuration."""
        custom_latencies = {
            "fast_pipeline": (2.0, 0.5),
            "slow_pipeline": (60.0, 10.0),
        }

        source = MockGWEventSource(
            t0=1000000000.0,
            duration=100.0,
            real_time=False,
            pipeline_latencies=custom_latencies,
        )

        assert "fast_pipeline" in source.source_pad_names
        assert "slow_pipeline" in source.source_pad_names
        assert "sgnl" not in source.source_pad_names

    def test_requires_end_or_duration(self):
        """Test that non-realtime mode requires end or duration."""
        with pytest.raises(ValueError, match="either end or duration"):
            MockGWEventSource(
                t0=1000000000.0,
                real_time=False,
            )


class TestMockGWEventSourceEventGeneration:
    """Tests for event generation in MockGWEventSource."""

    def test_generate_source_params(self):
        """Test source parameter generation."""
        numpy.random.seed(42)

        source = MockGWEventSource(
            t0=1000000000.0,
            duration=100.0,
            real_time=False,
        )

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
            t0=1000000000.0,
            duration=100.0,
            real_time=False,
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
            pipelines=["sgnl", "pycbc"],
            verbose=False,
        )

        assert sink.output_dir == str(tmp_path)
        assert "sgnl" in sink.sink_pad_names
        assert "pycbc" in sink.sink_pad_names

    def test_sink_writes_xml(self, tmp_path):
        """Test sink writes XML files."""
        from sgn.frames import Frame

        from sgnligo.sinks.coinc_xml_sink import CoincXMLSink

        sink = CoincXMLSink(
            output_dir=str(tmp_path),
            pipelines=["sgnl"],
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
        xmldoc = _build_coinc_xmldoc(event, "sgnl", psds, include_snr_series=False)
        xml_bytes = _serialize_xmldoc(xmldoc)

        # Create frame
        frame = Frame(
            EOS=False,
            is_gap=False,
            data={"xml": xml_bytes, "event_id": 1, "pipeline": "sgnl"},
            metadata={"t_co_gps": 1000000000.0},
        )

        # Send to sink
        sink.pull(sink.snks["sgnl"], frame)

        # Check file was written
        output_files = list(tmp_path.glob("*.xml"))
        assert len(output_files) == 1

        # Check stats
        stats = sink.get_stats()
        assert stats["total"] == 1
        assert stats["per_pipeline"]["sgnl"] == 1
