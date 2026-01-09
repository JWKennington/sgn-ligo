"""GWpySpectrogram: Time-frequency spectrogram using GWpy.

Uses audio adapter framework for buffer accumulation and overlap handling.
Output sample rates are constrained to power-of-2 values for offset alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
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


@dataclass
class GWpySpectrogram(TSTransform):
    """Compute time-frequency spectrogram using GWpy with streaming support.

    Uses the audio adapter framework for buffer accumulation and overlap
    management. Output sample rates are constrained to power-of-2 values
    for proper offset alignment.

    Args:
        spec_stride:
            Time step between spectrogram columns in seconds (default 1.0)
        fft_length:
            FFT length in seconds (default 2.0)
        fft_overlap:
            Overlap between FFTs in seconds (default: fft_length/2)
        window:
            Window function name (default 'hann')
        nproc:
            Number of processes for parallel computation (default 1)
        output_stride:
            Duration of output per processing cycle in seconds (default 1.0).
            Controls how often spectrogram output is produced.
        output_rate:
            Output sample rate in Hz. Must be power-of-2 from
            Offset.ALLOWED_RATES. If None, computed from spec_stride. (default: None)
        input_sample_rate:
            Expected input sample rate in Hz for adapter configuration.
            Must be power-of-2. (default: 4096)

    Example:
        >>> spec = GWpySpectrogram(
        ...     name="Spectrogram",
        ...     sink_pad_names=("in",),
        ...     source_pad_names=("out",),
        ...     spec_stride=1,
        ...     fft_length=2,
        ...     output_rate=64,  # Power-of-2 output rate
        ... )

    Note:
        - Output buffer shape is (n_frequencies, n_times)
        - Spectrogram frequencies are stored in metadata["spec_frequencies"]
        - Spectrogram times are stored in metadata["spec_times"]
        - The full Spectrogram object is in metadata["spectrogram"]
    """

    spec_stride: float = 1.0
    fft_length: float = 2.0
    fft_overlap: Optional[float] = None
    window: str = "hann"
    nproc: int = 1
    output_stride: float = 1.0
    output_rate: Optional[int] = None
    input_sample_rate: int = 4096

    def __post_init__(self):
        """Configure adapter for spectrogram buffering."""
        if self.fft_overlap is None:
            self.fft_overlap = self.fft_length / 2

        # Compute valid output rate if not specified
        # For spectrogram, the natural rate is 1/spec_stride
        # Constrain to power-of-2
        if self.output_rate is None:
            desired_rate = int(1.0 / self.spec_stride) if self.spec_stride > 0 else 64
            self.output_rate = _compute_valid_output_rate(
                desired_rate, self.input_sample_rate
            )

        # Spectrogram needs fft_length worth of overlap for edge effects
        edge_samples = int(np.ceil(self.fft_length * self.input_sample_rate))

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
        if self.spec_stride <= 0:
            raise ValueError("spec_stride must be positive")
        if self.fft_length <= 0:
            raise ValueError("fft_length must be positive")
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
        # Based on fft_length and sample rate
        # Number of frequency bins is (fft_length * sample_rate) / 2 + 1
        return int(self.fft_length * self.input_sample_rate / 2) + 1

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
        """Process input frame through spectrogram.

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
                channel="SGN:SPEC_INPUT",
            )

            spec = ts.spectrogram(
                stride=self.spec_stride,
                fftlength=self.fft_length,
                overlap=self.fft_overlap,
                window=self.window,
                nproc=self.nproc,
            )

            # Extract 2D data (transpose to get time as last dimension)
            # GWpy Spectrogram is (n_times, n_frequencies)
            spec_data = np.asarray(spec.value)

            # Transpose to (n_frequencies, n_times) for SGN convention
            out_data = spec_data.T

            # Crop to the center portion (stride region, excluding overlap)
            # The adapter provides overlapped data; extract only the center stride
            output_duration_sec = Offset.tosec(output_frame.noffset)
            n_times_total = out_data.shape[-1]
            times_per_sec = (
                n_times_total / Offset.tosec(buf.end_offset - buf.offset)
                if buf.end_offset > buf.offset
                else 1.0
            )

            # Number of time samples for the output stride
            n_times_output = max(1, int(round(output_duration_sec * times_per_sec)))

            # Calculate start index (skip edge effect portion)
            edge_duration_sec = self.fft_length
            n_times_edge = int(round(edge_duration_sec * times_per_sec))
            start_idx = min(n_times_edge, n_times_total - n_times_output)
            end_idx = min(start_idx + n_times_output, n_times_total)

            # Crop the data
            out_data = out_data[..., start_idx:end_idx]
            n_times = out_data.shape[-1]

            # Calculate effective sample rate to match output_frame's expected noffset
            assert (
                output_frame.noffset > 0
            ), f"output_frame.noffset={output_frame.noffset} must be positive"
            assert n_times > 0, f"n_times={n_times} must be positive"
            effective_rate = n_times / output_duration_sec
            # Find closest power-of-2 rate
            effective_rate = _compute_valid_output_rate(
                int(round(effective_rate)), self.input_sample_rate
            )

            # Use output_frame's expected offset for buffer alignment
            out_buf = SeriesBuffer(
                offset=output_frame.offset,
                sample_rate=effective_rate,
                data=out_data,
                shape=out_data.shape,
            )

            output_frame.append(out_buf)

            # Store spectrogram metadata
            output_frame.metadata["spectrogram"] = spec
            output_frame.metadata["spec_frequencies"] = np.asarray(
                spec.frequencies.value
            )
            output_frame.metadata["spec_times"] = np.asarray(spec.times.value)
            output_frame.metadata["spec_df"] = float(spec.df.value)
            output_frame.metadata["spec_dt"] = float(spec.dt.value)
