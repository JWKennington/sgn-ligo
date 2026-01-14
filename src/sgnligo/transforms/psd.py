"""
SGN Elements for PSD Estimation.
Provides TSTransform wrappers around the core estimator logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import lal
import numpy as np
import scipy.signal
from sgn.base import SourcePad
from sgnts.base import AdapterConfig, Offset, SeriesBuffer, TSFrame, TSTransform

from sgnligo.psd.estimators import BaseEstimator, MGMEstimator, RecursiveEstimator


@dataclass
class PSDEstimator(TSTransform):
    """
    Base TSTransform for PSD Estimation.

    Handles buffering, windowing, and FFT of strain data, then updates
    an internal BaseEstimator instance.

    Outputs:
        TSFrame:
            buffers: Pass-through of input strain (optional) or empty.
            metadata: Contains 'psd' (LAL series) and 'psd_numpy'.
    """

    fft_length: float = 4.0
    overlap: float = 0.5
    sample_rate: int = 16384
    window_type: str = "hann"
    output_pass_through: bool = False

    # Internal state
    _estimator: BaseEstimator = field(init=False, repr=False, default=None)
    _window: np.ndarray = field(init=False, repr=False, default=None)
    _freqs: np.ndarray = field(init=False, repr=False, default=None)
    _norm_factor: float = field(init=False, repr=False, default=1.0)
    _delta_f: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        # 1. Setup AudioAdapter for overlapping windows
        n_samples = int(self.fft_length * self.sample_rate)
        stride = int(n_samples * (1 - self.overlap))
        overlap_samples = n_samples - stride

        self.adapter_config = AdapterConfig()
        self.adapter_config.stride = Offset.fromsamples(stride, self.sample_rate)
        self.adapter_config.overlap = (
            0,
            Offset.fromsamples(overlap_samples, self.sample_rate),
        )
        self.adapter_config.skip_gaps = True

        super().__post_init__()

        # 2. Prepare Window and Normalization
        self._window = scipy.signal.get_window(self.window_type, n_samples)

        # Normalization factor for one-sided PSD (Strain^2 / Hz)
        # Factor 2 for one-sided (excluding DC/Nyquist handled by estimator)
        # Scale = 2 / (Fs * S2) where S2 is sum of squared window weights
        s2 = np.sum(self._window**2)
        self._norm_factor = 2.0 / (self.sample_rate * s2)

        # 3. Setup Frequency Axis
        self._freqs = np.fft.rfftfreq(n_samples, d=1 / self.sample_rate)
        self._delta_f = self._freqs[1] - self._freqs[0]

        # 4. Initialize Specific Estimator Logic
        self._init_estimator(len(self._freqs))

    def _init_estimator(self, size: int):
        """Subclasses must override to set self._estimator."""
        raise NotImplementedError

    def new(self, pad: SourcePad) -> TSFrame:
        in_frame = self.preparedframes[self.sink_pads[0]]

        # Pass gaps
        if in_frame.is_gap or not in_frame.buffers:
            return TSFrame(is_gap=True, EOS=in_frame.EOS)

        buf = in_frame.buffers[0]
        data = buf.data

        # Verify size (Adapter should handle this, but for safety)
        if len(data) != len(self._window):
            # Startup transient or mismatched buffer
            return TSFrame(is_gap=True, EOS=in_frame.EOS)

        # 1. Window and FFT
        windowed = data * self._window
        fft_data = np.fft.rfft(windowed)

        # 2. Update Estimator
        # Note: Estimator expects Raw Power or FFT.
        # BaseEstimator applies normalization internally if configured.
        self._estimator.update(fft_data)

        # 3. Get Result
        psd_data = self._estimator.get_psd()

        # 4. Prepare Metadata (LAL Format for compatibility)
        # Determine Epoch (time of the middle of the window?)
        # Standard GstLAL practice: Epoch is start of the window.
        epoch = Offset.tosec(buf.offset)

        lal_psd = lal.CreateREAL8FrequencySeries(
            "psd",
            lal.LIGOTimeGPS(epoch),
            0.0,  # f0
            self._delta_f,
            "strain^2/Hz",
            len(psd_data),
        )
        lal_psd.data.data = psd_data

        meta = in_frame.metadata.copy() if in_frame.metadata else {}
        meta["psd"] = lal_psd
        meta["psd_numpy"] = psd_data.copy()
        meta["psd_freqs"] = self._freqs

        # 5. Output Frame
        out_buffers = in_frame.buffers if self.output_pass_through else []

        return TSFrame(buffers=out_buffers, metadata=meta, EOS=in_frame.EOS)


@dataclass
class RecursivePSD(PSDEstimator):
    """
    Fast, Low-Latency PSD Estimator using IIR decay.
    """

    alpha: float = 0.1

    def _init_estimator(self, size: int):
        self._estimator = RecursiveEstimator(
            size=size, normalization=self._norm_factor, alpha=self.alpha
        )


@dataclass
class MGMPSD(PSDEstimator):
    """
    Robust Median-Geometric-Mean Estimator (LIGO Standard).
    """

    n_median: int = 7
    n_average: int = 64

    def _init_estimator(self, size: int):
        self._estimator = MGMEstimator(
            size=size,
            normalization=self._norm_factor,
            n_median=self.n_median,
            n_average=self.n_average,
        )
