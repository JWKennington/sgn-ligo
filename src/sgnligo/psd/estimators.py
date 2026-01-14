"""
Core PSD estimation logic classes (Math only).
Refactored to use dataclasses for configuration and state management.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import loggamma
from sympy import EulerGamma

EULERGAMMA = float(EulerGamma.evalf())


@dataclass
class BaseEstimator(ABC):
    """Base class for PSD estimation logic."""

    size: int
    normalization: float = 1.0

    # Internal State (excluded from init)
    n_samples: int = field(init=False, default=0)
    current_psd: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Initialize with ones to avoid divide-by-zero before first update
        self.current_psd = np.ones(self.size)

    @abstractmethod
    def update(self, data: np.ndarray) -> None:
        """Update state with new frequency-domain data."""
        pass

    def get_psd(self) -> np.ndarray:
        """Return the current PSD estimate."""
        return self.current_psd

    def _apply_boundaries(self):
        """Force DC and Nyquist bins to zero."""
        if self.size > 0:
            self.current_psd[0] = 0
            self.current_psd[-1] = 0


@dataclass
class MGMEstimator(BaseEstimator):
    """
    Median-Geometric-Mean Estimator (Standard LIGO).
    Tracks a history of power spectra, computes the median for robustness,
    and then the geometric mean for stability.
    """
    n_median: int = 7
    n_average: int = 64

    # Internal State
    history: deque = field(init=False, repr=False, default=None)
    geo_mean_log: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    count: int = field(init=False, default=0)

    def __post_init__(self):
        super().__post_init__()
        self.history = deque(maxlen=self.n_median)

    def _median_bias(self, nn):
        ans = 1
        n = (nn - 1) // 2
        for i in range(1, n + 1):
            ans -= 1.0 / (2 * i)
            ans += 1.0 / (2 * i + 1)
        return ans

    def update(self, data: np.ndarray) -> None:
        # data is complex FFT or real power
        power = np.abs(data) ** 2 if np.iscomplexobj(data) else data

        # Scale to raw units for internal storage
        power_norm = power / self.normalization
        self.history.append(power_norm)

        if self.count == 0:
            self.geo_mean_log = np.log(power_norm)
            self.count += 1
        else:
            self.count = min(self.count + 1, self.n_average)
            # Calculate bias
            n_bufs = len(self.history)
            bias = np.log(self._median_bias(n_bufs)) - n_bufs * (
                loggamma(1 / n_bufs) - np.log(n_bufs)
            )

            # Median of history
            stacked = np.array(self.history)
            median_log = np.log(np.median(stacked, axis=0))

            # Recursive update
            self.geo_mean_log = (
                self.geo_mean_log * (self.count - 1) + median_log - bias
            ) / self.count

        self.current_psd = np.exp(self.geo_mean_log + EULERGAMMA) * self.normalization
        self._apply_boundaries()

    def set_reference(self, psd: np.ndarray, weight: int):
        """Initialize history with a reference PSD."""
        raw = psd / self.normalization
        # Avoid log(0)
        raw = np.where(raw > 0, raw, 1e-300)

        self.history.clear()
        for _ in range(self.n_median):
            self.history.append(raw)

        self.geo_mean_log = np.log(raw) - EULERGAMMA
        self.count = min(weight, self.n_average)
        self.current_psd = psd.copy()


@dataclass
class RecursiveEstimator(BaseEstimator):
    """
    Exponential Moving Average Estimator.
    S_t = (1-alpha)*S_{t-1} + alpha*|X|^2
    """
    alpha: float = 0.1

    # Internal State
    _initialized: bool = field(init=False, default=False)

    def update(self, data: np.ndarray) -> None:
        power = (
            np.abs(data) ** 2 if np.iscomplexobj(data) else data
        ) * self.normalization

        if not self._initialized:
            self.current_psd = power
            self._initialized = True
        else:
            self.current_psd = (
                1 - self.alpha
            ) * self.current_psd + self.alpha * power

        self._apply_boundaries()
