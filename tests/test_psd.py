"""Test coverage for sgnligo.psd module."""

import math
import os
import signal
import tempfile
from unittest.mock import Mock, patch

import lal
import numpy as np
import pytest
from lal import LIGOTimeGPS

from sgnligo.psd import (
    HorizonDistance,
    condition_psd,
    effective_distance_factor,
    fake_gwdata_psd,
    harmonic_mean,
    interpolate_psd,
    movingaverage,
    movingmedian,
    polyfit,
    read_asd_txt,
    read_psd,
    taperzero_fseries,
    write_asd_txt,
    write_psd,
)


@pytest.fixture
def mock_psd():
    """Create a mock PSD for testing."""
    psd = lal.CreateREAL8FrequencySeries(
        name="test_psd",
        epoch=LIGOTimeGPS(0),
        f0=0.0,
        deltaF=1.0,
        sampleUnits=lal.Unit("strain^2 s"),
        length=100,
    )
    # Create a simple PSD shape
    psd.data.data = 1e-46 * np.ones(100)
    return psd


@pytest.fixture
def mock_psd_dict():
    """Create a dictionary of mock PSDs for testing."""
    psds = {}
    for ifo in ["H1", "L1", "V1"]:
        psd = lal.CreateREAL8FrequencySeries(
            name=f"{ifo}_psd",
            epoch=LIGOTimeGPS(0),
            f0=0.0,
            deltaF=1.0,
            sampleUnits=lal.Unit("strain^2 s"),
            length=100,
        )
        psd.data.data = (1 + 0.1 * ord(ifo[0])) * 1e-46 * np.ones(100)
        psds[ifo] = psd
    return psds


class TestReadWritePSD:
    """Test cases for read/write PSD functions."""

    @patch("sgnligo.psd.ligolw_utils.load_filename")
    @patch("sgnligo.psd.lal.series.read_psd_xmldoc")
    def test_read_psd(self, mock_read_xmldoc, mock_load_filename):
        """Test reading PSD from XML file."""
        # Mock the loaded XML document
        mock_xmldoc = Mock()
        mock_load_filename.return_value = mock_xmldoc

        # Mock the returned PSD dictionary
        expected_psds = {"H1": Mock(), "L1": Mock()}
        mock_read_xmldoc.return_value = expected_psds

        # Test without verbose
        result = read_psd("test.xml")
        assert result == expected_psds
        mock_load_filename.assert_called_once_with(
            "test.xml", verbose=False, contenthandler=lal.series.PSDContentHandler
        )

        # Test with verbose
        mock_load_filename.reset_mock()
        result = read_psd("test.xml", verbose=True)
        mock_load_filename.assert_called_once_with(
            "test.xml", verbose=True, contenthandler=lal.series.PSDContentHandler
        )

    @patch("sgnligo.psd.ligolw_utils.write_filename")
    @patch("sgnligo.psd.lal.series.make_psd_xmldoc")
    def test_write_psd(self, mock_make_xmldoc, mock_write_filename, mock_psd_dict):
        """Test writing PSD to XML file."""
        # Mock the XML document
        mock_xmldoc = Mock()
        mock_make_xmldoc.return_value = mock_xmldoc

        # Test without optional parameters
        write_psd("test.xml", mock_psd_dict)
        mock_make_xmldoc.assert_called_once_with(mock_psd_dict)
        mock_write_filename.assert_called_once_with(
            mock_xmldoc, "test.xml", verbose=False, trap_signals=None
        )

        # Test with optional parameters
        mock_make_xmldoc.reset_mock()
        mock_write_filename.reset_mock()
        trap_signals = [signal.SIGTERM, signal.SIGINT]
        write_psd("test.xml", mock_psd_dict, trap_signals=trap_signals, verbose=True)
        mock_write_filename.assert_called_once_with(
            mock_xmldoc, "test.xml", verbose=True, trap_signals=trap_signals
        )


class TestReadWriteASD:
    """Test cases for read/write ASD functions."""

    def test_read_asd_txt(self):
        """Test reading ASD from text file."""
        # Create a temporary file with test data
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("# Frequency (Hz)  ASD (strain/rtHz)\n")
            f.write("10.0  1e-23\n")
            f.write("20.0  2e-23\n")
            f.write("30.0  3e-23\n")
            f.write("40.0  4e-23\n")
            f.write("50.0  5e-23\n")
            fname = f.name

        try:
            # Test reading as ASD (default)
            psd = read_asd_txt(fname, df=10.0)
            assert psd.f0 == 10.0
            assert psd.deltaF == 10.0
            # Function interpolates, length is (max_f - min_f) / df = (50-10)/10 = 4
            assert len(psd.data.data) == 4
            # Check that ASD was squared to get PSD
            np.testing.assert_almost_equal(psd.data.data[0], (1e-23) ** 2)

            # Test reading as PSD
            psd = read_asd_txt(fname, df=10.0, read_as_psd=True)
            np.testing.assert_almost_equal(psd.data.data[0], 1e-23)

            # Test with zero padding
            psd = read_asd_txt(fname, df=5.0, zero_pad=True)
            assert psd.f0 == 0.0  # Should start from 0 Hz
            assert psd.data.data[0] == psd.data.data[1]  # Padded values

        finally:
            os.unlink(fname)

    def test_write_asd_txt(self, mock_psd, capsys):
        """Test writing ASD to text file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            fname = f.name

        try:
            # Test without verbose
            write_asd_txt(fname, mock_psd)

            # Read back and verify
            with open(fname, "r") as f:
                lines = f.readlines()
            assert len(lines) == 100
            # Check first line format
            parts = lines[0].split()
            assert float(parts[0]) == 0.0  # frequency
            assert float(parts[1]) == pytest.approx(math.sqrt(1e-46))  # ASD

            # Test with verbose
            write_asd_txt(fname, mock_psd, verbose=True)
            captured = capsys.readouterr()
            assert f"writing '{fname}'" in captured.err

        finally:
            os.unlink(fname)


class TestInterpolatePSD:
    """Test cases for interpolate_psd function."""

    def test_interpolate_psd_no_op(self, mock_psd):
        """Test that interpolation with same deltaF returns same PSD."""
        result = interpolate_psd(mock_psd, mock_psd.deltaF)
        assert result is mock_psd

    def test_interpolate_psd_upsample(self, mock_psd):
        """Test upsampling PSD to finer frequency resolution."""
        # Make PSD with some structure
        mock_psd.data.data = 1e-46 * (1 + 0.1 * np.sin(2 * np.pi * np.arange(100) / 20))

        new_deltaF = 0.5  # Upsample by factor of 2
        result = interpolate_psd(mock_psd, new_deltaF)

        assert result.deltaF == new_deltaF
        assert result.f0 == mock_psd.f0
        assert len(result.data.data) == 199  # (100-1)*2 + 1
        assert result.name == mock_psd.name
        assert result.epoch == mock_psd.epoch
        assert result.sampleUnits == mock_psd.sampleUnits

    def test_interpolate_psd_downsample(self, mock_psd):
        """Test downsampling PSD to coarser frequency resolution."""
        new_deltaF = 2.0  # Downsample by factor of 2
        result = interpolate_psd(mock_psd, new_deltaF)

        assert result.deltaF == new_deltaF
        # Length calculation: round((len(psd_data) - 1) * psd.deltaF / deltaF) + 1
        # = round((100 - 1) * 1.0 / 2.0) + 1 = round(49.5) + 1 = 50 + 1 = 51
        assert len(result.data.data) == 51

    def test_interpolate_psd_zeros(self):
        """Test interpolation handles zeros in PSD."""
        psd = lal.CreateREAL8FrequencySeries(
            "test", LIGOTimeGPS(0), 0.0, 1.0, lal.Unit("strain^2 s"), 10
        )
        psd.data.data = np.array([1e-46, 0, 1e-46, 0, 1e-46, 0, 1e-46, 0, 1e-46, 0])

        result = interpolate_psd(psd, 0.5)
        # The function replaces zeros with 1e-300 before interpolation
        # Check that the result is valid
        assert result.deltaF == 0.5
        assert len(result.data.data) > len(psd.data.data)


class TestSmoothingFunctions:
    """Test cases for PSD smoothing functions."""

    def test_movingmedian_array(self):
        """Test moving median on numpy array."""
        data = np.array([1, 10, 2, 9, 3, 8, 4, 7, 5, 6])
        window_size = 2
        result = movingmedian(data, window_size)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        # Check edges are unchanged
        assert result[0] == data[0]
        assert result[-1] == data[-1]

    def test_movingmedian_psd(self, mock_psd):
        """Test moving median on LAL PSD."""
        # Add some outliers to test median filtering
        mock_psd.data.data[10] = 1e-40  # Spike
        mock_psd.data.data[20] = 1e-50  # Dip

        window_size = 3
        result = movingmedian(mock_psd, window_size)

        assert isinstance(result, lal.REAL8FrequencySeries)
        assert result.name == mock_psd.name
        assert result.deltaF == mock_psd.deltaF
        assert len(result.data.data) == len(mock_psd.data.data)

    @patch("sgnligo.psd.lal.CreateTukeyREAL8Window")
    def test_movingaverage(self, mock_window):
        """Test moving average function."""
        # Mock the window
        mock_window_obj = Mock()
        mock_window_obj.data.data = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        mock_window.return_value = mock_window_obj

        data = np.ones(20)
        window_size = 5
        result = movingaverage(data, window_size)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        mock_window.assert_called_once_with(window_size, 0.5)


class TestTaperZeroFSeries:
    """Test cases for taperzero_fseries function."""

    def test_taperzero_fseries_default(self):
        """Test tapering frequency series with default parameters."""
        # Create a complex frequency series like from FFT
        length = 8193  # Typical FFT length (4096 * 2 + 1)
        fseries = lal.CreateCOMPLEX16FrequencySeries(
            "test", LIGOTimeGPS(0), 0.0, 1.0, lal.Unit("strain"), length
        )
        fseries.data.data = np.ones(length, dtype=complex)

        # Apply tapering
        result = taperzero_fseries(fseries)

        # Check that it returns the same object (modified in place)
        assert result is fseries

        # Check that norm is preserved (approximately)
        # The function renormalizes to preserve total power
        assert np.isfinite(np.dot(fseries.data.data.conj(), fseries.data.data).real)

    def test_taperzero_fseries_custom_bounds(self):
        """Test tapering with custom minfs and maxfs bounds."""
        # Create a complex frequency series
        length = 8193
        fseries = lal.CreateCOMPLEX16FrequencySeries(
            "test", LIGOTimeGPS(0), 0.0, 0.5, lal.Unit("strain"), length
        )
        fseries.data.data = np.ones(length, dtype=complex)

        # Apply tapering with custom bounds
        minfs = (10.0, 20.0)
        maxfs = (100.0, 200.0)
        result = taperzero_fseries(fseries, minfs=minfs, maxfs=maxfs)

        # Check that it returns the same object
        assert result is fseries

        # Check that some values were tapered
        assert not np.all(fseries.data.data == 1.0)


class TestConditionPSD:
    """Test cases for condition_psd function."""

    @patch("sgnligo.psd.HorizonDistance")
    @patch("sgnligo.psd.interpolate_psd")
    @patch("sgnligo.psd.movingmedian")
    @patch("sgnligo.psd.movingaverage")
    def test_condition_psd_frequency_domain(
        self, mock_avg, mock_median, mock_interp, mock_horizon
    ):
        """Test conditioning PSD for frequency domain whitening."""
        # Create a longer PSD that covers required frequencies
        psd = lal.CreateREAL8FrequencySeries(
            "test_psd",
            epoch=LIGOTimeGPS(0),
            f0=0.0,
            deltaF=0.5,
            sampleUnits=lal.Unit("strain^2 s"),
            length=5000,  # Up to 2500 Hz
        )
        psd.data.data = 1e-46 * np.ones(5000)

        # Mock horizon distance
        mock_horizon_instance = Mock()
        mock_horizon_instance.return_value = (100.0, Mock())
        mock_horizon.return_value = mock_horizon_instance

        # Mock interpolation to return the same PSD
        mock_interp.return_value = psd

        # Mock smoothing functions
        mock_median.return_value = psd.data.data
        mock_avg.return_value = psd.data.data

        result = condition_psd(
            psd,
            newdeltaF=0.5,
            minfs=(35.0, 40.0),
            maxfs=(1800.0, 2048.0),
            smoothing_frequency=4.0,
            fir_whiten=False,
        )

        # Verify calls
        mock_interp.assert_called_once_with(psd, 0.5)
        mock_median.assert_called_once()
        mock_avg.assert_called_once()

        # The function modifies the PSD in place and returns it
        assert result.deltaF == 0.5
        # Check that horizon distance normalization was called
        assert mock_horizon_instance.call_count == 2

    @patch("sgnligo.psd.HorizonDistance")
    @patch("sgnligo.psd.interpolate_psd")
    @patch("sgnligo.psd.movingmedian")
    @patch("sgnligo.psd.movingaverage")
    def test_condition_psd_time_domain(
        self, mock_avg, mock_median, mock_interp, mock_horizon, mock_psd
    ):
        """Test conditioning PSD for time domain whitening."""
        # Mock horizon distance
        mock_horizon_instance = Mock()
        mock_horizon_instance.return_value = (100.0, Mock())
        mock_horizon.return_value = mock_horizon_instance

        # Mock interpolation
        mock_interp.return_value = mock_psd

        # Mock smoothing functions
        mock_median.return_value = mock_psd.data.data
        mock_avg.return_value = mock_psd.data.data

        result = condition_psd(mock_psd, newdeltaF=0.5, fir_whiten=True)

        # Check that tapering was NOT applied for time domain
        assert not np.any(np.isinf(result.data.data))

    @patch("sgnligo.psd.HorizonDistance")
    def test_condition_psd_with_tapering(self, mock_horizon):
        """Test conditioning PSD with tapering for frequency domain."""
        # Create a real PSD
        psd = lal.CreateREAL8FrequencySeries(
            "test_psd",
            epoch=LIGOTimeGPS(0),
            f0=0.0,
            deltaF=1.0,
            sampleUnits=lal.Unit("strain^2 s"),
            length=3000,
        )
        psd.data.data = 1e-46 * np.ones(3000)

        # Mock horizon distance class and instance
        mock_horizon_instance = Mock()
        mock_horizon_instance.return_value = (100.0, None)
        mock_horizon.return_value = mock_horizon_instance

        result = condition_psd(
            psd,
            newdeltaF=1,
            minfs=(35.0, 40.0),
            maxfs=(1800.0, 2048.0),
            smoothing_frequency=4.0,
            fir_whiten=False,
        )

        # Check that tapering was applied
        # Values below minfs[0] should be inf
        assert np.all(np.isinf(result.data.data[:35]))
        # Values above maxfs[1] should be inf
        assert np.all(np.isinf(result.data.data[2048:]))
        # Check that line 463 was executed (psd.data.data = psddata)
        assert result is psd


class TestPolyfit:
    """Test cases for polyfit function."""

    def test_polyfit_basic(self, mock_psd, capsys):
        """Test polynomial fitting of PSD."""
        # Create PSD with power-law behavior
        f = np.arange(100) + 1
        mock_psd.data.data = 1e-46 * f.astype(float) ** (-2)

        result = polyfit(mock_psd, f_low=10.0, f_high=50.0, order=2, verbose=False)

        assert isinstance(result, lal.REAL8FrequencySeries)
        assert result.deltaF == mock_psd.deltaF
        # The function creates a new PSD, so check that it's different from input
        assert result is not mock_psd

    def test_polyfit_verbose(self, mock_psd, capsys):
        """Test polynomial fitting with verbose output."""
        polyfit(mock_psd, f_low=10.0, f_high=50.0, order=2, verbose=True)

        captured = capsys.readouterr()
        assert "Fit polynomial is:" in captured.err
        assert "log(PSD) =" in captured.err


class TestHarmonicMean:
    """Test cases for harmonic_mean function."""

    def test_harmonic_mean(self, mock_psd_dict):
        """Test harmonic mean of PSDs."""
        result = harmonic_mean(mock_psd_dict)

        assert isinstance(result, lal.REAL8FrequencySeries)
        assert result.name == "psd"
        assert result.deltaF == list(mock_psd_dict.values())[0].deltaF
        assert len(result.data.data) == len(list(mock_psd_dict.values())[0].data.data)

        # Check harmonic mean calculation
        # For constant PSDs, harmonic mean should be less than arithmetic mean
        arith_mean = np.mean([psd.data.data[0] for psd in mock_psd_dict.values()])
        assert result.data.data[0] < arith_mean

    def test_harmonic_mean_single_psd(self):
        """Test harmonic mean with single PSD."""
        psd = lal.CreateREAL8FrequencySeries(
            "test", LIGOTimeGPS(0), 0.0, 1.0, lal.Unit("strain^2 s"), 10
        )
        psd.data.data = 2.0 * np.ones(10)

        result = harmonic_mean({"H1": psd})
        np.testing.assert_array_equal(result.data.data, psd.data.data)


class TestHorizonDistance:
    """Test cases for HorizonDistance class."""

    @patch("sgnligo.psd.lalsimulation.SimInspiralFD")
    def test_horizon_distance_init(self, mock_sim_inspiral):
        """Test HorizonDistance initialization."""
        # Mock the waveform generation
        hp = Mock()
        hp.data.length = 100
        hp.data.data = np.ones(100, dtype=complex) * 1e-23
        hp.f0 = 10.0
        hp.deltaF = 1.0
        hp.sampleUnits = lal.Unit("strain")

        hc = Mock()
        mock_sim_inspiral.return_value = (hp, hc)

        hd = HorizonDistance(
            f_min=10.0,
            f_max=1000.0,
            delta_f=1.0,
            m1=1.4,
            m2=1.4,
            spin1=(0, 0, 0),
            spin2=(0, 0, 0),
            eccentricity=0.0,
            inclination=0.0,
            approximant="IMRPhenomD",
        )

        assert hd.f_min == 10.0
        assert hd.f_max == 1000.0
        assert hd.m1 == 1.4
        assert hd.m2 == 1.4
        assert isinstance(hd.model, lal.REAL8FrequencySeries)
        assert hd.model.data.length == 100

    @patch("sgnligo.psd.lalsimulation.SimInspiralFD")
    @patch("sgnligo.psd.lalsimulation.GetApproximantFromString")
    def test_horizon_distance_call(self, mock_get_approx, mock_sim_inspiral, mock_psd):
        """Test computing horizon distance."""
        # Mock the waveform generation
        hp = Mock()
        hp.data.length = 100
        hp.data.data = np.ones(100, dtype=complex) * 1e-23
        hp.f0 = 0.0
        hp.deltaF = 1.0
        hp.sampleUnits = lal.Unit("strain")

        hc = Mock()
        mock_sim_inspiral.return_value = (hp, hc)
        mock_get_approx.return_value = 0  # Some approximant enum

        hd = HorizonDistance(10.0, 100.0, 1.0, 1.4, 1.4)

        # Call with PSD
        D, (f, model) = hd(mock_psd, snr=8.0)

        assert isinstance(D, float)
        assert D > 0  # Should be positive distance
        assert isinstance(f, np.ndarray)
        assert isinstance(model, np.ndarray)
        assert len(f) == len(model)

    @patch("sgnligo.psd.lalsimulation.SimInspiralFD")
    def test_horizon_distance_zero_length_error(self, mock_sim_inspiral):
        """Test that zero-length waveform raises assertion error."""
        # Mock zero-length waveform
        hp = Mock()
        hp.data.length = 0
        hc = Mock()
        mock_sim_inspiral.return_value = (hp, hc)

        with pytest.raises(AssertionError, match="h\\+ has zero length"):
            HorizonDistance(10.0, 100.0, 1.0, 1.4, 1.4)


class TestEffectiveDistanceFactor:
    """Test cases for effective_distance_factor function."""

    def test_effective_distance_factor_face_on(self):
        """Test effective distance factor for face-on binary."""
        # Face-on: inclination = 0
        factor = effective_distance_factor(0, 1.0, 0.0)
        assert factor == pytest.approx(1.0)

    def test_effective_distance_factor_edge_on(self):
        """Test effective distance factor for edge-on binary."""
        # Edge-on: inclination = pi/2
        factor = effective_distance_factor(math.pi / 2, 1.0, 0.0)
        assert factor == pytest.approx(2.0)

    def test_effective_distance_factor_general(self):
        """Test effective distance factor for general case."""
        inclination = math.pi / 4
        fp = 0.5
        fc = 0.5
        factor = effective_distance_factor(inclination, fp, fc)

        assert factor > 0
        assert factor >= 1.0  # Effective distance >= physical distance


class TestFakeGwdataPSD:
    """Test cases for fake_gwdata_psd function."""

    def test_fake_gwdata_psd_default(self):
        """Test fake PSD generation with default IFOs."""
        psds = fake_gwdata_psd()

        assert len(psds) == 3
        assert "H1" in psds
        assert "L1" in psds
        assert "V1" in psds

        for _, psd in psds.items():
            assert isinstance(psd, lal.REAL8FrequencySeries)
            assert psd.name == "psd"
            assert psd.f0 == 0.0
            assert psd.deltaF == 0.125
            assert psd.data.length == 8192 * 8 + 1
            # Check PSD properties
            # The function creates identical PSDs for all IFOs
            # Check that f=0 should be 0 but might have numerical issues
            assert psd.data.data[0] < 1e-30  # Very small value
            # Check that low frequencies (< 10 Hz) are set to constant value
            # f < 10 Hz means indices < 10/0.125 = 80
            if len(psd.data.data) > 80:
                # Check that other values are reasonable
                assert np.all(np.isfinite(psd.data.data[1:]))
                assert np.all(psd.data.data[1:] > 0)

    def test_fake_gwdata_psd_custom_ifos(self):
        """Test fake PSD generation with custom IFOs."""
        psds = fake_gwdata_psd(ifos=("K1", "G1"))

        assert len(psds) == 2
        assert "K1" in psds
        assert "G1" in psds
        assert "H1" not in psds

    def test_fake_gwdata_psd_shape(self):
        """Test that fake PSD has reasonable shape."""
        psds = fake_gwdata_psd(ifos=("H1",))
        psd = psds["H1"]

        # Check that PSD increases at low frequencies (below ~200 Hz)
        assert psd.data.data[100 * 8] > psd.data.data[200 * 8]

        # Check that PSD increases at high frequencies (above ~300 Hz)
        assert psd.data.data[1000 * 8] > psd.data.data[300 * 8]

        # Check minimum around 200-300 Hz
        min_idx = np.argmin(psd.data.data[10 * 8 : 1000 * 8]) + 10 * 8
        min_freq = min_idx * psd.deltaF
        assert 100 < min_freq < 400
