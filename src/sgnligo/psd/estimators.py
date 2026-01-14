"""
Core PSD estimation logic classes (Math only).
Refactored to remove invalid boundary zeroing and enforce input shape.
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

    # Internal State
    n_samples: int = field(init=False, default=0)
    current_psd: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Initialize with ones to avoid divide-by-zero
        self.current_psd = np.ones(self.size)

    def _check_shape(self, data: np.ndarray) -> None:
        """Validate input data shape matches estimator configuration."""
        if data.shape[-1] != self.size:
            raise ValueError(
                f"Input data size {data.shape[-1]} does not match estimator size {self.size}"
            )

    @abstractmethod
    def update(self, data: np.ndarray) -> None:
        """Update state with new frequency-domain data."""
        pass

    def get_psd(self) -> np.ndarray:
        return self.current_psd


@dataclass
class MGMEstimator(BaseEstimator):
    """
    Median-Geometric-Mean Estimator (Standard LIGO).
    """
    n_median: int = 7
    n_average: int = 64

    # Internal State
    history: deque = field(init=False, repr=False, default=None)
    geo_mean_log: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self.history = deque(maxlen=self.n_median)

    @staticmethod
    def _median_bias(nn):
        """XLALMedianBias"""
        ans = 1.0
        n = (nn - 1) // 2
        for i in range(1, n + 1):
            ans -= 1.0 / (2 * i)
            ans += 1.0 / (2 * i + 1)
        return ans

    @staticmethod
    def _log_median_bias_geometric(nn):
        """XLALLogMedianBiasGeometric"""
        return np.log(MGMEstimator._median_bias(nn)) - nn * (
            loggamma(1.0 / nn) - np.log(nn)
        )

    def update(self, data: np.ndarray) -> None:
        self._check_shape(data)

        if np.iscomplexobj(data):
            power = np.abs(data) ** 2
        else:
            power = data

        self.history.append(power)

        if self.n_samples == 0:
            self.geo_mean_log = np.log(power)
            self.n_samples += 1
        else:
            self.n_samples = min(self.n_samples + 1, self.n_average)

            bias = self._log_median_bias_geometric(len(self.history))

            # Match Legacy: use sort and integer index
            stacked = np.array(self.history)
            sorted_bins = np.sort(stacked, axis=0)
            idx = len(self.history) // 2
            log_bin_median = np.log(sorted_bins[idx])

            self.geo_mean_log = (
                self.geo_mean_log * (self.n_samples - 1) + log_bin_median - bias
            ) / self.n_samples

        self.current_psd = (
            np.exp(self.geo_mean_log + EULERGAMMA) * self.normalization
        )

    def set_reference(self, psd: np.ndarray, weight: int):
        self._check_shape(psd)

        raw = psd / self.normalization
        # Avoid log(0)
        raw = np.where(raw > 0, raw, 1e-300)

        self.history.clear()
        for _ in range(self.n_median):
            self.history.append(raw)

        self.geo_mean_log = np.log(raw) - EULERGAMMA
        self.n_samples = min(weight, self.n_average)
        self.current_psd = psd.copy()


@dataclass
class RecursiveEstimator(BaseEstimator):
    """
    Exponential Moving Average Estimator.
    """
    alpha: float = 0.1
    _initialized: bool = field(init=False, default=False)

    def update(self, data: np.ndarray) -> None:
        self._check_shape(data)

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
