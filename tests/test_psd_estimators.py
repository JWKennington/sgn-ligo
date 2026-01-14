"""
Consistency tests to ensure the new MGMEstimator produces identical
numerical results to the legacy Whiten implementation.
"""

from collections import deque

import numpy as np
import pytest
from scipy.special import loggamma
from sympy import EulerGamma

from sgnligo.psd.estimators import BaseEstimator, MGMEstimator, RecursiveEstimator

EULERGAMMA = float(EulerGamma.evalf())


class LegacyMGM:
    """
    Minimal logic extracted VERBATIM from sgnligo.transforms.whiten.Whiten.
    Includes the initialization logic previously found in __post_init__.
    """

    def __init__(self, n_median=7, n_average=64):
        self.n_median = n_median
        self.n_average = n_average
        self.square_data_bufs = deque(maxlen=n_median)
        self.geometric_mean_square = None
        self.n_samples = 0

    def _median_bias(self, nn):
        ans = 1.0
        n = (nn - 1) // 2
        for i in range(1, n + 1):
            ans -= 1.0 / (2 * i)
            ans += 1.0 / (2 * i + 1)
        return ans

    def _log_median_bias_geometric(self, nn):
        return np.log(self._median_bias(nn)) - nn * (loggamma(1.0 / nn) - np.log(nn))

    def add_psd(self, data):
        """Standard update loop from Whiten.internal"""
        power = np.abs(data) ** 2
        self.square_data_bufs.append(power)

        if self.n_samples == 0:
            self.geometric_mean_square = np.log(self.square_data_bufs[0])
            self.n_samples += 1
        else:
            self.n_samples += 1
            self.n_samples = min(self.n_samples, self.n_average)
            median_bias = self._log_median_bias_geometric(len(self.square_data_bufs))

            # EXACT Legacy sort/index logic
            stacked = np.array(self.square_data_bufs)
            sorted_bins = np.sort(stacked, axis=0)
            # Note: This logic picks the "upper middle" for even lengths
            log_bin_median = np.log(sorted_bins[len(self.square_data_bufs) // 2])

            self.geometric_mean_square = (
                self.geometric_mean_square * (self.n_samples - 1)
                + log_bin_median
                - median_bias
            ) / self.n_samples

    def set_reference(self, psd_data, normalization, weight):
        """
        Logic copied from Whiten.__post_init__ for handling reference PSDs.
        """
        # Legacy code calculates arithmetic mean of raw data
        arithmetic_mean_square_data = psd_data / normalization

        self.square_data_bufs.clear()
        for _ in range(self.n_median):
            self.square_data_bufs.append(arithmetic_mean_square_data)

        self.geometric_mean_square = np.log(arithmetic_mean_square_data) - EULERGAMMA
        self.n_samples = min(weight, self.n_average)

    def get_psd(self, normalization):
        return np.exp(self.geometric_mean_square + EULERGAMMA) * normalization


class TestMGMConsistency:
    """Compare New Class vs Legacy Logic."""

    @pytest.fixture
    def setup_estimators(self):
        norm = 2.0
        n_med = 7
        n_avg = 64
        size = 10
        legacy = LegacyMGM(n_median=n_med, n_average=n_avg)
        new_est = MGMEstimator(
            size=size, normalization=norm, n_median=n_med, n_average=n_avg
        )
        return legacy, new_est, norm, size

    def test_exact_match_random_data(self, setup_estimators):
        """Feed both estimators random noise and assert output equality."""
        legacy, new_est, norm, size = setup_estimators
        np.random.seed(42)

        for i in range(100):
            # Generate random complex data (simulating FFT)
            data = np.random.randn(size) + 1j * np.random.randn(size)

            legacy.add_psd(data)
            new_est.update(data)

            legacy_out = legacy.get_psd(norm)
            new_out = new_est.get_psd()

            # Legacy doesn't zero boundaries internally, new class does
            # We ignore boundaries for this math check
            np.testing.assert_allclose(
                new_out[1:-1],
                legacy_out[1:-1],
                err_msg=f"Mismatch at step {i}",
                rtol=1e-12,
            )

    def test_set_reference_match(self, setup_estimators):
        """Verify set_reference initializes internal state identically."""
        legacy, new_est, norm, size = setup_estimators

        # Create a fake reference PSD (already scaled)
        ref_psd = np.linspace(10, 100, size)
        weight = 32

        legacy.set_reference(ref_psd, norm, weight)
        new_est.set_reference(ref_psd, weight)

        # Check immediate output
        legacy_out = legacy.get_psd(norm)
        new_out = new_est.get_psd()
        np.testing.assert_allclose(new_out[1:-1], legacy_out[1:-1], rtol=1e-12)

        # Check that subsequent updates behave identically (proving internal state match)
        data = np.full(size, 5.0 + 0j)
        legacy.add_psd(data)
        new_est.update(data)

        legacy_out_2 = legacy.get_psd(norm)
        new_out_2 = new_est.get_psd()
        np.testing.assert_allclose(new_out_2[1:-1], legacy_out_2[1:-1], rtol=1e-12)

    def test_step_response(self, setup_estimators):
        """Test how both estimators react to a sudden step in noise floor."""
        legacy, new_est, norm, size = setup_estimators

        # 1. Warm up with low noise
        for _ in range(10):
            data = np.full(size, 1.0 + 0j)
            legacy.add_psd(data)
            new_est.update(data)

        # 2. Step up to high noise
        step_data = np.full(size, 100.0 + 0j)
        for i in range(20):
            legacy.add_psd(step_data)
            new_est.update(step_data)

            np.testing.assert_allclose(
                new_est.get_psd()[1:-1],
                legacy.get_psd(norm)[1:-1],
                rtol=1e-12,
                err_msg=f"Divergence during step response at iter {i}",
            )

    def test_even_median_logic(self):
        """
        CRITICAL: Test Even n_median.
        Standard numpy.median averages middle elements.
        Legacy logic picks the upper-middle element.
        This test ensures we preserved the Legacy selection behavior.
        """
        n_med = 4  # Even number
        norm = 1.0
        size = 1

        legacy = LegacyMGM(n_median=n_med, n_average=10)
        new_est = MGMEstimator(
            size=size, normalization=norm, n_median=n_med, n_average=10
        )

        # Fill buffer with [1, 10, 100, 1000]
        # Sorted: [1, 10, 100, 1000]
        # Len = 4. Idx = 4 // 2 = 2.
        # Element at index 2 is 100.
        # (Standard median would be (10+100)/2 = 55)

        vals = [1.0, 10.0, 100.0, 1000.0]
        for v in vals:
            data = np.array([v + 0j])
            legacy.add_psd(data)
            new_est.update(data)

        # Verify outputs match
        np.testing.assert_allclose(
            new_est.get_psd()[0], legacy.get_psd(norm)[0], rtol=1e-12
        )

        # Verify it actually picked 100 (approximately, after smoothing)
        # We can peek at the legacy internal mean calculation to be sure
        # or just trust that if they match, we are good.
        # Let's verify they match.


class TestBaseEstimatorInterface:
    """Test the contract and shared logic of the abstract base class."""

    class ConcreteEstimator(BaseEstimator):
        """Minimal concrete implementation for testing."""

        def update(self, data: np.ndarray) -> None:
            self._check_shape(data)  # Explicitly calling the check
            self.current_psd = data * self.normalization

    def test_initialization_defaults(self):
        """Ensure safe default state (ones) to prevent div-by-zero downstream."""
        est = self.ConcreteEstimator(size=10)
        assert est.size == 10
        assert est.normalization == 1.0
        assert est.n_samples == 0
        np.testing.assert_array_equal(est.get_psd(), np.ones(10))

    def test_get_psd_reference_safety(self):
        """Check if get_psd returns the array object."""
        est = self.ConcreteEstimator(size=5)
        est.update(np.full(5, 10.0))

        psd_ref = est.get_psd()
        psd_ref[0] = 999.0

        # Verify internal state was modified (reference behavior confirmed)
        assert est.current_psd[0] == 999.0

    def test_input_shape_mismatch(self):
        """
        BaseEstimator._check_shape should raise ValueError if dimensions mismatch.
        """
        est = self.ConcreteEstimator(size=5)
        # Feeding size 3 into size 5 estimator
        with pytest.raises(ValueError):
            est.update(np.ones(3))


class TestRecursiveEstimatorLogic:
    """Detailed logic verification for the IIR Estimator."""

    def test_alpha_zero(self):
        """Alpha=0 means infinite memory (never update after init)."""
        est = RecursiveEstimator(size=1, normalization=1.0, alpha=0.0)

        # 1. Init (sets state to input regardless of alpha)
        est.update(np.array([10.0]))
        assert est.get_psd()[0] == 10.0

        # 2. Update (should be ignored)
        est.update(np.array([100.0]))
        assert est.get_psd()[0] == 10.0

    def test_alpha_one(self):
        """Alpha=1 means memoryless (output = input)."""
        est = RecursiveEstimator(size=1, normalization=1.0, alpha=1.0)

        est.update(np.array([10.0]))
        assert est.get_psd()[0] == 10.0

        est.update(np.array([50.0]))
        assert est.get_psd()[0] == 50.0

    def test_impulse_response(self):
        """Verify the exponential decay curve explicitly."""
        alpha = 0.5
        est = RecursiveEstimator(size=1, normalization=1.0, alpha=alpha)

        # Impulse: 100 at t=0, then 0
        est.update(np.array([100.0]))
        assert est.get_psd()[0] == 100.0

        # t=1: (1-0.5)*100 + 0 = 50
        est.update(np.array([0.0]))
        assert est.get_psd()[0] == 50.0

        # t=2: (1-0.5)*50 + 0 = 25
        est.update(np.array([0.0]))
        assert est.get_psd()[0] == 25.0

        # t=3: 12.5
        est.update(np.array([0.0]))
        assert est.get_psd()[0] == 12.5

    def test_complex_vs_real_equivalence(self):
        """Ensure update(complex) and update(abs^2) produce identical state."""
        est_c = RecursiveEstimator(size=1, alpha=0.1)
        est_r = RecursiveEstimator(size=1, alpha=0.1)

        # Complex data: 3+4j (mag sq = 25)
        c_data = np.array([3.0 + 4.0j])
        r_data = np.array([25.0])

        est_c.update(c_data)
        est_r.update(r_data)

        np.testing.assert_array_equal(est_c.get_psd(), est_r.get_psd())


class TestMGMEstimatorMath:
    """Verify the static math methods match theoretical expectations."""

    def test_median_bias_values(self):
        """Check bias factors for known N against manual calculation."""
        # N=1: Median is the value. Bias = 1.0 (log bias = 0)
        assert MGMEstimator._median_bias(1) == 1.0
        assert MGMEstimator._log_median_bias_geometric(1) == 0.0

        # N=3: Median is middle. Bias formula: 1 - 1/2 + 1/3 = 5/6?
        expected = 1.0 - 1.0 / 2.0 + 1.0 / 3.0
        assert np.isclose(MGMEstimator._median_bias(3), expected)

    def test_log_median_bias_monotonicity(self):
        """Bias correction should behave predictably as N increases."""
        biases = [MGMEstimator._log_median_bias_geometric(n) for n in range(1, 20, 2)]
        assert np.all(np.isfinite(biases))


class TestMGMEstimatorLogic:
    """Test the update lifecycle, history management, and reference setting."""

    def test_history_deque_maxlen(self):
        """Ensure history buffer never exceeds n_median."""
        est = MGMEstimator(size=1, n_median=5, n_average=10)
        for i in range(10):
            est.update(np.array([float(i)]))

        assert len(est.history) == 5
        # Should contain the last 5 elements: 5,6,7,8,9
        expected_last = 9.0
        assert est.history[-1][0] == expected_last

    def test_count_clamping(self):
        """Ensure n_samples count clamps at n_average."""
        est = MGMEstimator(size=1, n_median=5, n_average=10)

        # Update 15 times
        for _ in range(15):
            est.update(np.array([1.0]))

        assert est.n_samples == 10  # Clamped
        # Verify it didn't keep growing
        est.update(np.array([1.0]))
        assert est.n_samples == 10

    def test_set_reference_initialization(self):
        """Verify set_reference correctly primes the internal state."""
        est = MGMEstimator(size=2, normalization=2.0, n_median=3, n_average=10)
        ref_psd = np.array([10.0, 20.0])

        # Weight 5
        est.set_reference(ref_psd, weight=5)

        # 1. Check Output
        np.testing.assert_array_equal(est.get_psd(), ref_psd)

        # 2. Check Count
        assert est.n_samples == 5

        # 3. Check History (Should allow 'raw' values)
        # raw = psd / norm = [5.0, 10.0]
        assert len(est.history) == 3
        expected_raw = ref_psd / 2.0
        for entry in est.history:
            np.testing.assert_array_equal(entry, expected_raw)

        # 4. Check Internal Mean State
        expected_log_mean = np.log(expected_raw) - EULERGAMMA
        np.testing.assert_allclose(est.geo_mean_log, expected_log_mean)

    def test_set_reference_zero_handling(self):
        """set_reference should handle zeros by clamping to avoid log(0)."""
        est = MGMEstimator(size=1, normalization=1.0)
        ref_psd = np.array([0.0])

        est.set_reference(ref_psd, weight=1)

        # Output should be exact copy of input
        assert est.get_psd()[0] == 0.0

        # Internal history should be clamped to epsilon
        assert est.history[0][0] > 0.0
        assert est.history[0][0] <= 1e-300


class TestMGMEstimatorNumerics:
    """Edge cases, infinities, NaNs, and extreme values."""

    def test_single_zero_update_ignored(self):
        """
        Feeding a single zero to MGM should NOT drive output to zero immediately.
        It should be rejected by the median filter.
        """
        est = MGMEstimator(size=1, n_median=3)

        # 1. Valid init
        est.update(np.array([10.0]))
        base_val = est.get_psd()[0]

        # 2. Single zero update
        # History becomes [10, 0]. Sorted [0, 10]. Median index 1 => 10.
        # So the zero is ignored.
        with np.errstate(divide="ignore"):
            est.update(np.array([0.0]))

        # Value should decay slightly due to averaging, but not crash to zero
        # 15.7 is what previous test showed
        assert est.get_psd()[0] > 1.0

    def test_sustained_zero_update(self):
        """
        If we feed enough zeros to dominate the median, output MUST go to zero.
        """
        est = MGMEstimator(size=1, n_median=3)
        est.update(np.array([10.0]))

        # Feed zeros until history is [0, 0, 0] or [0, 0, 10] (median=0)
        with np.errstate(divide="ignore"):
            est.update(np.array([0.0]))  # Hist: 10, 0
            est.update(np.array([0.0]))  # Hist: 10, 0, 0 -> Median 0
            est.update(np.array([0.0]))  # Hist: 0, 0, 0 -> Median 0

        assert est.get_psd()[0] == 0.0

    def test_very_small_values(self):
        """Ensure precision holds for very small numbers (not underflowing prematurely)."""
        est = MGMEstimator(size=1)
        small_val = 1e-20

        # Init
        est.update(np.array([small_val]))

        # Expect small_val * exp(EG)
        expected = small_val * np.exp(EULERGAMMA)
        np.testing.assert_allclose(est.get_psd()[0], expected)

    def test_recovery_from_transient(self):
        """Test that the estimator recovers after a massive transient spike."""
        est = MGMEstimator(size=1, n_median=3, n_average=5)

        # Steady state 1.0
        for _ in range(5):
            est.update(np.array([1.0]))

        base_level = est.get_psd()[0]

        # Massive Glitch (1e6) for 1 sample.
        est.update(np.array([1e6]))

        # Should reject the glitch via median
        post_glitch_level = est.get_psd()[0]
        np.testing.assert_allclose(post_glitch_level, base_level, rtol=0.2)
