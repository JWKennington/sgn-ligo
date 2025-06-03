"""Test coverage for sgnligo.kernels module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestPSDFirKernel:
    """Test cases for PSDFirKernel class."""

    def test_init(self):
        """Test PSDFirKernel initialization."""
        from sgnligo.kernels import PSDFirKernel

        kernel = PSDFirKernel()
        assert kernel.revplan is None
        assert kernel.fwdplan is None
        assert kernel.target_phase is None
        assert kernel.target_phase_mask is None

    def test_init_with_params(self):
        """Test PSDFirKernel initialization with parameters."""
        from sgnligo.kernels import PSDFirKernel

        phase = np.array([1, 2, 3])
        mask = np.array([0.5, 0.6, 0.7])
        kernel = PSDFirKernel(
            revplan="rev", fwdplan="fwd", target_phase=phase, target_phase_mask=mask
        )
        assert kernel.revplan == "rev"
        assert kernel.fwdplan == "fwd"
        np.testing.assert_array_equal(kernel.target_phase, phase)
        np.testing.assert_array_equal(kernel.target_phase_mask, mask)

    @patch("sgnligo.kernels.lal")
    @patch("sgnligo.kernels.HorizonDistance")
    def test_set_phase(self, mock_horizon, mock_lal):
        """Test set_phase method."""
        from sgnligo.kernels import PSDFirKernel

        # Create PSDFirKernel instance
        psd_fir_kernel = PSDFirKernel()

        # Mock PSD
        mock_psd = Mock()
        mock_psd.f0 = 0.0
        mock_psd.deltaF = 1.0
        mock_psd.data = Mock()
        mock_psd.data.data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Mock the methods that set_phase calls
        with patch.object(
            psd_fir_kernel,
            "psd_to_linear_phase_whitening_fir_kernel",
            return_value=(np.array([1, 2, 3]), 1, 100),
        ):
            with patch.object(
                psd_fir_kernel,
                "linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel",
                return_value=(np.array([4, 5, 6]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            ):

                # Mock HorizonDistance
                mock_horizon_instance = Mock()
                mock_horizon_instance.return_value = (
                    10.0,  # horizon distance
                    (
                        np.array([1.0, 2.0]),
                        np.array([1.0, 2.0]),
                    ),  # f_model, model (must be within PSD range)
                )
                mock_horizon.return_value = mock_horizon_instance

                # Call set_phase
                psd_fir_kernel.set_phase(mock_psd, f_low=20.0, m1=1.5, m2=1.5)

                # Verify phase and mask were set
                assert psd_fir_kernel.target_phase is not None
                assert psd_fir_kernel.target_phase_mask is not None

    @patch("sgnligo.kernels.lal")
    def test_psd_to_linear_phase_whitening_fir_kernel(self, mock_lal):
        """Test psd_to_linear_phase_whitening_fir_kernel method."""
        from sgnligo.kernels import PSDFirKernel

        # Mock LAL functions
        mock_window = Mock()
        mock_window.data.data = np.ones(9)
        mock_lal.CreateTukeyREAL8Window.return_value = mock_window

        mock_lal.CreateCOMPLEX16FrequencySeries.return_value = Mock(
            data=Mock(data=np.zeros(9, dtype=complex))
        )
        mock_lal.CreateCOMPLEX16TimeSeries.return_value = Mock(
            data=Mock(data=np.zeros(9, dtype=complex))
        )
        mock_lal.CreateReverseCOMPLEX16FFTPlan.return_value = "reverse_plan"
        mock_lal.Unit.return_value = "mock_unit"
        mock_lal.LIGOTimeGPS = Mock

        # Mock FFT operation
        def mock_fft(output, input_series, plan):
            output.data.data[:] = np.array(
                [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 4 + 0j, 3 + 0j, 2 + 0j, 1 + 0j]
            )

        mock_lal.COMPLEX16FreqTimeFFT.side_effect = mock_fft

        # Create kernel and PSD
        psd_fir_kernel = PSDFirKernel()
        mock_psd = Mock()
        mock_psd.f0 = 0.0
        mock_psd.deltaF = 1.0
        mock_psd.data = Mock()
        mock_psd.data.data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        kernel, latency, sample_rate = (
            psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(mock_psd)
        )

        # Check return types
        assert isinstance(kernel, np.ndarray)
        assert isinstance(latency, int)
        assert isinstance(sample_rate, int)
        assert sample_rate == 8
        assert latency == 4

    @patch("sgnligo.kernels.lal")
    def test_psd_to_linear_phase_whitening_fir_kernel_with_nyquist(self, mock_lal):
        """Test psd_to_linear_phase_whitening_fir_kernel with nyquist parameter."""
        from sgnligo.kernels import PSDFirKernel

        # Mock LAL functions
        mock_window = Mock()
        mock_window.data.data = np.ones(5)
        mock_lal.CreateTukeyREAL8Window.return_value = mock_window

        mock_lal.CreateCOMPLEX16FrequencySeries.return_value = Mock(
            data=Mock(data=np.zeros(5, dtype=complex))
        )
        mock_lal.CreateCOMPLEX16TimeSeries.return_value = Mock(
            data=Mock(data=np.zeros(5, dtype=complex))
        )
        mock_lal.CreateReverseCOMPLEX16FFTPlan.return_value = "reverse_plan"
        mock_lal.Unit.return_value = "mock_unit"
        mock_lal.LIGOTimeGPS = Mock

        # Mock FFT operation
        def mock_fft(output, input_series, plan):
            output.data.data[:] = np.ones(5, dtype=complex)

        mock_lal.COMPLEX16FreqTimeFFT.side_effect = mock_fft

        # Create kernel and PSD
        psd_fir_kernel = PSDFirKernel()
        mock_psd = Mock()
        mock_psd.f0 = 0.0
        mock_psd.deltaF = 1.0
        mock_psd.data = Mock()
        mock_psd.data.data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        kernel, latency, sample_rate = (
            psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(
                mock_psd, nyquist=2.0
            )
        )

        # Verify function executed without error
        assert isinstance(kernel, np.ndarray)
        assert isinstance(latency, int)
        assert isinstance(sample_rate, int)

    @patch("sgnligo.kernels.lal")
    def test_psd_to_linear_phase_whitening_fir_kernel_no_invert(self, mock_lal):
        """Test psd_to_linear_phase_whitening_fir_kernel without inverting."""
        from sgnligo.kernels import PSDFirKernel

        # Mock LAL functions
        mock_window = Mock()
        mock_window.data.data = np.ones(9)
        mock_lal.CreateTukeyREAL8Window.return_value = mock_window

        mock_lal.CreateCOMPLEX16FrequencySeries.return_value = Mock(
            data=Mock(data=np.zeros(9, dtype=complex))
        )
        mock_lal.CreateCOMPLEX16TimeSeries.return_value = Mock(
            data=Mock(data=np.zeros(9, dtype=complex))
        )
        mock_lal.CreateReverseCOMPLEX16FFTPlan.return_value = "reverse_plan"
        mock_lal.Unit.return_value = "mock_unit"
        mock_lal.LIGOTimeGPS = Mock

        # Mock FFT operation
        def mock_fft(output, input_series, plan):
            output.data.data[:] = np.ones(9, dtype=complex)

        mock_lal.COMPLEX16FreqTimeFFT.side_effect = mock_fft

        # Create kernel and PSD
        psd_fir_kernel = PSDFirKernel()
        mock_psd = Mock()
        mock_psd.f0 = 0.0
        mock_psd.deltaF = 1.0
        mock_psd.data = Mock()
        mock_psd.data.data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        kernel, latency, sample_rate = (
            psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(
                mock_psd, invert=False
            )
        )

        # Verify function executed without error
        assert isinstance(kernel, np.ndarray)
        assert isinstance(latency, int)
        assert isinstance(sample_rate, int)

    @patch("sgnligo.kernels.lal")
    def test_psd_to_linear_phase_whitening_fir_kernel_with_existing_plan(
        self, mock_lal
    ):
        """Test psd_to_linear_phase_whitening_fir_kernel with existing reverse plan."""
        from sgnligo.kernels import PSDFirKernel

        # Mock LAL functions
        mock_window = Mock()
        mock_window.data.data = np.ones(9)
        mock_lal.CreateTukeyREAL8Window.return_value = mock_window

        mock_lal.CreateCOMPLEX16FrequencySeries.return_value = Mock(
            data=Mock(data=np.zeros(9, dtype=complex))
        )
        mock_lal.CreateCOMPLEX16TimeSeries.return_value = Mock(
            data=Mock(data=np.zeros(9, dtype=complex))
        )
        mock_lal.Unit.return_value = "mock_unit"
        mock_lal.LIGOTimeGPS = Mock

        # Mock FFT operation
        def mock_fft(output, input_series, plan):
            output.data.data[:] = np.ones(9, dtype=complex)

        mock_lal.COMPLEX16FreqTimeFFT.side_effect = mock_fft

        # Create kernel with existing plan
        psd_fir_kernel = PSDFirKernel()
        psd_fir_kernel.revplan = "existing_plan"

        mock_psd = Mock()
        mock_psd.f0 = 0.0
        mock_psd.deltaF = 1.0
        mock_psd.data = Mock()
        mock_psd.data.data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        kernel, latency, sample_rate = (
            psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(mock_psd)
        )

        # Verify CreateReverseCOMPLEX16FFTPlan was not called
        mock_lal.CreateReverseCOMPLEX16FFTPlan.assert_not_called()

    def test_psd_to_linear_phase_whitening_fir_kernel_assertion(self):
        """Test psd_to_linear_phase_whitening_fir_kernel assertion for f0."""
        from sgnligo.kernels import PSDFirKernel

        psd_fir_kernel = PSDFirKernel()
        mock_psd = Mock()
        mock_psd.f0 = 1.0

        with pytest.raises(AssertionError):
            psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(mock_psd)

    @patch("sgnligo.kernels.lal")
    def test_linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(
        self, mock_lal
    ):
        """Test linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel method."""
        from sgnligo.kernels import PSDFirKernel

        # Mock LAL functions
        mock_lal.CreateForwardCOMPLEX16FFTPlan.return_value = "forward_plan"
        mock_lal.CreateReverseCOMPLEX16FFTPlan.return_value = "reverse_plan"
        mock_lal.LIGOTimeGPS = Mock
        mock_lal.Unit.return_value = "mock_unit"

        # Create mock series objects
        def create_series(**kwargs):
            series = Mock()
            series.data = Mock()
            series.data.data = np.zeros(5, dtype=complex)
            return series

        mock_lal.CreateCOMPLEX16TimeSeries.side_effect = create_series
        mock_lal.CreateCOMPLEX16FrequencySeries.side_effect = create_series

        # Mock FFT operations
        call_count = 0

        def mock_fft(output, input_series, plan):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First FFT call
                output.data.data[:] = np.array([1 + 0j, 2 + 1j, 3 + 2j, 2 - 1j, 1 + 0j])
            elif call_count == 2:  # Second FFT call (for cepstrum)
                output.data.data[:] = np.array(
                    [0.1 + 0j, 0.2 + 0j, 0.3 + 0j, 0.2 + 0j, 0.1 + 0j]
                )
            elif call_count == 3:  # Third FFT call (for theta)
                output.data.data[:] = np.array(
                    [0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j, 0.2 + 0.2j, 0.1 + 0.1j]
                )
            elif call_count == 4:  # Final FFT call
                output.data.data[:] = np.array([1 + 0j, 2 + 0j, 3 + 0j, 2 + 0j, 1 + 0j])

        mock_lal.COMPLEX16TimeFreqFFT.side_effect = mock_fft
        mock_lal.COMPLEX16FreqTimeFFT.side_effect = mock_fft

        # Create kernel
        psd_fir_kernel = PSDFirKernel()
        linear_kernel = np.array([1, 2, 3, 4, 5])
        sample_rate = 100

        # Use shorter variable name to avoid long lines
        func = (
            psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel
        )
        kernel, phase = func(linear_kernel, sample_rate)

        # Check return types
        assert isinstance(kernel, np.ndarray)
        assert isinstance(phase, np.ndarray)
        assert len(kernel) == 5
        assert len(phase) == 3

    @patch("sgnligo.kernels.lal")
    def test_linear_phase_to_minimum_phase_with_target_phase(self, mock_lal):
        """Test linear_phase_to_minimum_phase_whitening_fir_kernel with target phase."""
        from sgnligo.kernels import PSDFirKernel

        # Mock LAL functions
        mock_lal.CreateForwardCOMPLEX16FFTPlan.return_value = "forward_plan"
        mock_lal.CreateReverseCOMPLEX16FFTPlan.return_value = "reverse_plan"
        mock_lal.LIGOTimeGPS = Mock
        mock_lal.Unit.return_value = "mock_unit"

        # Create mock series objects
        def create_series(**kwargs):
            series = Mock()
            series.data = Mock()
            series.data.data = np.zeros(5, dtype=complex)
            return series

        mock_lal.CreateCOMPLEX16TimeSeries.side_effect = create_series
        mock_lal.CreateCOMPLEX16FrequencySeries.side_effect = create_series

        # Mock FFT operations
        call_count = 0

        def mock_fft(output, input_series, plan):
            nonlocal call_count
            call_count += 1
            output.data.data[:] = np.ones(5, dtype=complex) * (0.1 + 0.1j)

        mock_lal.COMPLEX16TimeFreqFFT.side_effect = mock_fft
        mock_lal.COMPLEX16FreqTimeFFT.side_effect = mock_fft

        # Create kernel with target phase
        psd_fir_kernel = PSDFirKernel()
        psd_fir_kernel.target_phase = np.array([0.1, 0.2, 0.3])
        psd_fir_kernel.target_phase_mask = np.array([0.5, 0.6, 0.7])

        linear_kernel = np.array([1, 2, 3, 4, 5])
        sample_rate = 100

        # Use shorter variable name to avoid long lines
        func = (
            psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel
        )
        kernel, phase = func(linear_kernel, sample_rate)

        # Check return types
        assert isinstance(kernel, np.ndarray)
        assert isinstance(phase, np.ndarray)

    @patch("sgnligo.kernels.lal")
    def test_linear_phase_to_minimum_phase_with_existing_plans(self, mock_lal):
        """Test linear_phase_to_minimum_phase_whitening_fir_kernel with plans."""
        from sgnligo.kernels import PSDFirKernel

        # Mock LAL functions
        mock_lal.LIGOTimeGPS = Mock
        mock_lal.Unit.return_value = "mock_unit"

        # Create mock series objects
        def create_series(**kwargs):
            series = Mock()
            series.data = Mock()
            series.data.data = np.zeros(5, dtype=complex)
            return series

        mock_lal.CreateCOMPLEX16TimeSeries.side_effect = create_series
        mock_lal.CreateCOMPLEX16FrequencySeries.side_effect = create_series
        mock_lal.CreateReverseCOMPLEX16FFTPlan.return_value = "new_reverse_plan"

        # Mock FFT operations
        def mock_fft(output, input_series, plan):
            output.data.data[:] = np.ones(5, dtype=complex) * 0.1

        mock_lal.COMPLEX16TimeFreqFFT.side_effect = mock_fft
        mock_lal.COMPLEX16FreqTimeFFT.side_effect = mock_fft

        # Create kernel with existing forward plan but no reverse plan
        psd_fir_kernel = PSDFirKernel()
        psd_fir_kernel.fwdplan = "existing_fwd"

        linear_kernel = np.array([1, 2, 3, 4, 5])
        sample_rate = 100

        # Use shorter variable name to avoid long lines
        func = (
            psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel
        )
        kernel, phase = func(linear_kernel, sample_rate)

        # Verify forward plan creation was not called but reverse was
        mock_lal.CreateForwardCOMPLEX16FFTPlan.assert_not_called()
        mock_lal.CreateReverseCOMPLEX16FFTPlan.assert_called_once()


class TestFirWhitenerKernel:
    """Test cases for fir_whitener_kernel function."""

    @patch("sgnligo.kernels.lal")
    @patch("sgnligo.kernels.PSDFirKernel")
    def test_fir_whitener_kernel(self, mock_psd_fir_kernel_class, mock_lal):
        """Test fir_whitener_kernel function."""
        from sgnligo.kernels import fir_whitener_kernel

        # Mock LAL functions
        mock_lal.CreateForwardCOMPLEX16FFTPlan.return_value = "forward_plan"
        mock_lal.LIGOTimeGPS = Mock
        mock_lal.Unit.return_value = "mock_unit"

        # Create mock series objects
        kernel_tseries = Mock()
        kernel_tseries.data = Mock()
        kernel_tseries.data.data = np.zeros(10, dtype=complex)

        kernel_fseries = Mock()
        kernel_fseries.data = Mock()
        kernel_fseries.data.data = np.zeros(10, dtype=complex)

        mock_lal.CreateCOMPLEX16TimeSeries.return_value = kernel_tseries
        mock_lal.CreateCOMPLEX16FrequencySeries.return_value = kernel_fseries

        # Mock FFT
        mock_lal.COMPLEX16TimeFreqFFT.return_value = None

        # Mock PSDFirKernel instance and its methods
        mock_kernel_instance = Mock()
        mock_kernel_instance.psd_to_linear_phase_whitening_fir_kernel.return_value = (
            np.array([1, 2, 3, 4, 5]),
            2,
            100,
        )
        # Use shorter variable name
        m = mock_kernel_instance
        linear_to_min = m.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel
        linear_to_min.return_value = (
            np.array([5, 4, 3, 2, 1]),
            np.array([0.1, 0.2]),
        )
        mock_psd_fir_kernel_class.return_value = mock_kernel_instance

        # Mock PSD
        mock_psd = Mock()

        # Call function
        result = fir_whitener_kernel(10, 1.0, 100, mock_psd)

        # Verify result
        assert result is kernel_fseries

        # Verify PSDFirKernel methods were called
        psd_method = mock_kernel_instance.psd_to_linear_phase_whitening_fir_kernel
        psd_method.assert_called_once_with(mock_psd, nyquist=50.0)
        linear_to_min.assert_called_once()

    @patch("sgnligo.kernels.lal")
    @patch("sgnligo.kernels.PSDFirKernel")
    def test_fir_whitener_kernel_short_kernel(
        self, mock_psd_fir_kernel_class, mock_lal
    ):
        """Test fir_whitener_kernel when kernel is shorter than requested length."""
        from sgnligo.kernels import fir_whitener_kernel

        # Mock LAL functions
        mock_lal.CreateForwardCOMPLEX16FFTPlan.return_value = "forward_plan"
        mock_lal.LIGOTimeGPS = Mock
        mock_lal.Unit.return_value = "mock_unit"

        # Create mock series objects
        kernel_tseries = Mock()
        kernel_tseries.data = Mock()
        kernel_tseries.data.data = np.zeros(10, dtype=complex)

        kernel_fseries = Mock()

        mock_lal.CreateCOMPLEX16TimeSeries.return_value = kernel_tseries
        mock_lal.CreateCOMPLEX16FrequencySeries.return_value = kernel_fseries
        mock_lal.COMPLEX16TimeFreqFFT.return_value = None

        # Mock PSDFirKernel instance to return a short kernel
        mock_kernel_instance = Mock()
        mock_kernel_instance.psd_to_linear_phase_whitening_fir_kernel.return_value = (
            np.array([1, 2, 3]),
            1,
            100,
        )
        # Use shorter variable name
        m = mock_kernel_instance
        linear_to_min = m.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel
        linear_to_min.return_value = (
            np.array([3, 2, 1]),
            np.array([0.1, 0.2]),
        )
        mock_psd_fir_kernel_class.return_value = mock_kernel_instance

        # Call function with longer length
        result = fir_whitener_kernel(10, 1.0, 100, Mock())

        # Verify result
        assert result is kernel_fseries
        # Verify kernel was padded (check that data was set)
        assert len(kernel_tseries.data.data) == 10

    @patch("sgnligo.kernels.lal")
    @patch("sgnligo.kernels.PSDFirKernel")
    def test_fir_whitener_kernel_long_kernel(self, mock_psd_fir_kernel_class, mock_lal):
        """Test fir_whitener_kernel when kernel is longer than requested length."""
        from sgnligo.kernels import fir_whitener_kernel

        # Mock LAL functions
        mock_lal.CreateForwardCOMPLEX16FFTPlan.return_value = "forward_plan"
        mock_lal.LIGOTimeGPS = Mock
        mock_lal.Unit.return_value = "mock_unit"

        # Create mock series objects
        kernel_tseries = Mock()
        kernel_tseries.data = Mock()
        kernel_tseries.data.data = np.zeros(5, dtype=complex)

        kernel_fseries = Mock()

        mock_lal.CreateCOMPLEX16TimeSeries.return_value = kernel_tseries
        mock_lal.CreateCOMPLEX16FrequencySeries.return_value = kernel_fseries
        mock_lal.COMPLEX16TimeFreqFFT.return_value = None

        # Mock PSDFirKernel instance to return a long kernel
        mock_kernel_instance = Mock()
        mock_kernel_instance.psd_to_linear_phase_whitening_fir_kernel.return_value = (
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            5,
            100,
        )
        # Use shorter variable name
        m = mock_kernel_instance
        linear_to_min = m.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel
        linear_to_min.return_value = (
            np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
            np.array([0.1, 0.2]),
        )
        mock_psd_fir_kernel_class.return_value = mock_kernel_instance

        # Call function with shorter length
        result = fir_whitener_kernel(5, 1.0, 100, Mock())

        # Verify result
        assert result is kernel_fseries

    def test_fir_whitener_kernel_assertion(self):
        """Test fir_whitener_kernel assertion for None PSD."""
        from sgnligo.kernels import fir_whitener_kernel

        with pytest.raises(AssertionError):
            fir_whitener_kernel(10, 1.0, 100, None)
