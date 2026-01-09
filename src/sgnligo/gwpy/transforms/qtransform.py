"""GWpyQTransform: Streaming Q-transform using GWpy.

Uses audio adapter framework for buffer accumulation and overlap handling.
Output sample rates are constrained to power-of-2 values for offset alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries
from sgn import validator
from sgnts.base import (
    AdapterConfig,
    Offset,
    SeriesBuffer,
    TSCollectFrame,
    TSFrame,
    TSTransform,
)
from sgnts.decorators import transform


def _compute_valid_output_rate(desired_rate: int, max_rate: int) -> int:
    """Find closest valid power-of-2 rate not exceeding max_rate.

    Args:
        desired_rate: Target rate in Hz
        max_rate: Maximum allowed rate (typically input sample rate)

    Returns:
        Closest power-of-2 rate from Offset.ALLOWED_RATES
    """
    valid = sorted(r for r in Offset.ALLOWED_RATES if r <= max_rate)
    assert valid, f"No valid rates <= {max_rate} in {Offset.ALLOWED_RATES}"
    return min(valid, key=lambda r: abs(r - desired_rate))


def _compute_tres_for_rate(target_rate: int) -> float:
    """Compute tres that produces target_rate time samples per second.

    Args:
        target_rate: Desired output sample rate in Hz

    Returns:
        Time resolution in seconds for GWpy q_transform
    """
    return 1.0 / target_rate


@dataclass
class GWpyQTransform(TSTransform):
    """Compute Q-transform using GWpy with streaming support.

    Uses the audio adapter framework for buffer accumulation and overlap
    management. Output sample rates are constrained to power-of-2 values
    for proper offset alignment.

    Args:
        qrange:
            Range of Q values as (min, max) tuple (default (4, 64))
        frange:
            Frequency range as (min, max) tuple in Hz (default (20, 1024))
        mismatch:
            Maximum allowed mismatch between tiles (default 0.2)
        logf:
            Use logarithmic frequency spacing (default True)
        tres:
            Time resolution for output in seconds. If None, computed from
            output_rate to ensure power-of-2 sample rate. (default: None)
        fres:
            Frequency resolution for output (default: auto)
        output_stride:
            Duration of output per processing cycle in seconds (default 1.0)
        output_rate:
            Output sample rate in Hz. Must be power-of-2 from
            Offset.ALLOWED_RATES. If None, defaults to 64 Hz. (default: None)
        n_sigma:
            Number of wavelet standard deviations to use for edge duration
            calculation (default 5.0). Q-transform wavelets have Gaussian
            envelopes, so 3-5 sigma captures essentially all the energy.
            5 sigma is conservative and ensures clean edges.
        input_sample_rate:
            Expected input sample rate in Hz for adapter configuration.
            Must be power-of-2. (default: 4096)

    Example:
        >>> qtrans = GWpyQTransform(
        ...     name="QTransform",
        ...     sink_pad_names=("in",),
        ...     source_pad_names=("out",),
        ...     qrange=(4, 64),
        ...     frange=(20, 500),
        ...     output_rate=64,  # Power-of-2 output rate
        ... )

    Note:
        - Q-transform result is stored in metadata["qtransform"]
        - metadata["q_frequencies"] and metadata["q_times"] contain axes
        - Edge effect overlap is computed as n_sigma * Q_max / (2*pi*f_min)
    """

    qrange: Tuple[float, float] = (4, 64)
    frange: Tuple[float, float] = (20, 1024)
    mismatch: float = 0.2
    logf: bool = True
    tres: Optional[float] = None
    fres: Optional[float] = None
    output_stride: float = 1.0
    output_rate: Optional[int] = None
    n_sigma: float = 5.0
    input_sample_rate: int = 4096

    def __post_init__(self):
        """Configure adapter for Q-transform buffering."""
        # Compute valid output rate if not specified
        if self.output_rate is None:
            self.output_rate = _compute_valid_output_rate(64, self.input_sample_rate)

        # Compute tres to achieve desired output rate if not specified
        if self.tres is None:
            self.tres = _compute_tres_for_rate(self.output_rate)

        # Q-transform edge effect: overlap = n_sigma * Q_max / (2*pi*f_min)
        # The Q-transform uses wavelets with Gaussian envelopes.
        # Using n_sigma ensures the wavelet response has decayed (Gaussian: 5σ
        # is very conservative).
        edge_duration_sec = self.n_sigma * self.qrange[1] / (2 * np.pi * self.frange[0])
        edge_samples = int(np.ceil(edge_duration_sec * self.input_sample_rate))

        # Configure adapter
        self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (
            Offset.fromsamples(edge_samples, self.input_sample_rate),
            Offset.fromsamples(edge_samples, self.input_sample_rate),
        )
        self.adapter_config.stride = Offset.fromsamples(
            int(self.output_stride * self.input_sample_rate),
            self.input_sample_rate,
        )
        self.adapter_config.skip_gaps = True
        # Pad zeros at startup to produce output immediately
        self.adapter_config.on_startup(pad_zeros=True)

        super().__post_init__()

    @validator.one_to_one
    def validate(self) -> None:
        """Validate configuration."""
        if self.qrange[0] >= self.qrange[1]:
            raise ValueError("qrange[0] must be less than qrange[1]")
        if self.frange[0] >= self.frange[1]:
            raise ValueError("frange[0] must be less than frange[1]")
        if self.output_stride <= 0:
            raise ValueError("output_stride must be positive")
        if (
            self.output_rate is not None
            and self.output_rate not in Offset.ALLOWED_RATES
        ):
            raise ValueError(
                f"output_rate {self.output_rate} must be power-of-2 from "
                f"Offset.ALLOWED_RATES: {sorted(Offset.ALLOWED_RATES)}"
            )
        # input_sample_rate validation handled by Offset.fromsamples in __post_init__

    def _estimate_freq_bins(self) -> int:
        """Estimate number of frequency bins for gap buffer shape."""
        # Rough estimate based on frange and logf
        if self.logf:
            # Log spacing typically gives ~50-200 bins
            return 100
        else:
            # Linear spacing based on fres
            if self.fres is not None:
                return int((self.frange[1] - self.frange[0]) / self.fres)
            return 100

    def _create_gap_buffer(
        self, offset: int, noffset: int, sample_rate: int
    ) -> SeriesBuffer:
        """Create a gap buffer with appropriate 2D shape."""
        n_freq = self._estimate_freq_bins()
        n_times = Offset.tosamples(noffset, self.output_rate)

        return SeriesBuffer(
            offset=offset,
            sample_rate=self.output_rate,
            data=None,
            shape=(n_freq, n_times),
        )

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Process input frame through Q-transform.

        The adapter provides overlapped/strided data automatically.
        """
        for buf in input_frame:
            if buf.is_gap:
                # Use output_frame's expected offset/noffset for gap buffers
                assert self.output_rate is not None  # Set in __post_init__
                gap_buf = self._create_gap_buffer(
                    output_frame.offset,
                    output_frame.noffset,
                    self.output_rate,
                )
                output_frame.append(gap_buf)
                continue

            # Create TimeSeries from adapter-provided data
            t0_gps = Offset.tosec(buf.offset)
            ts = TimeSeries(
                buf.data,
                t0=t0_gps,
                sample_rate=buf.sample_rate,
                channel="SGN:QTRANSFORM_INPUT",
            )

            # Calculate output segment from stride
            # The adapter provides overlapped data; we extract the center portion
            edge_duration_sec = (
                self.n_sigma * self.qrange[1] / (2 * np.pi * self.frange[0])
            )
            out_start = t0_gps + edge_duration_sec
            out_end = out_start + self.output_stride

            try:
                qtrans = ts.q_transform(
                    qrange=self.qrange,
                    frange=self.frange,
                    mismatch=self.mismatch,
                    logf=self.logf,
                    tres=self.tres,
                    fres=self.fres,
                    outseg=Segment(out_start, out_end),
                )
            except Exception:
                # Q-transform can fail for various reasons
                # Use output_frame's expected offset/noffset for gap buffers
                assert self.output_rate is not None  # Set in __post_init__
                gap_buf = self._create_gap_buffer(
                    output_frame.offset,
                    output_frame.noffset,
                    self.output_rate,
                )
                output_frame.append(gap_buf)
                continue

            # Extract 2D Q-transform data
            # GWpy Spectrogram is (n_times, n_frequencies)
            q_data = np.asarray(qtrans.value)

            # Transpose to (n_frequencies, n_times) for SGN convention
            out_data = q_data.T

            # Use output_frame's expected offset for buffer alignment
            out_buf = SeriesBuffer(
                offset=output_frame.offset,
                sample_rate=self.output_rate,  # Power-of-2 rate!
                data=out_data,
                shape=out_data.shape,
            )

            output_frame.append(out_buf)

            # Store Q-transform metadata
            output_frame.metadata["qtransform"] = qtrans
            output_frame.metadata["q_frequencies"] = np.asarray(
                qtrans.frequencies.value
            )
            output_frame.metadata["q_times"] = np.asarray(qtrans.times.value)
            output_frame.metadata["q_qrange"] = self.qrange
            output_frame.metadata["q_frange"] = self.frange
