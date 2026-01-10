"""GWpyFilter: Streaming bandpass, lowpass, highpass, and notch filtering.

Uses the audio adapter framework for buffer accumulation and overlap handling.
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


def _apply_filter_to_timeseries(
    ts: TimeSeries,
    filter_type: str,
    low_freq: Optional[float],
    high_freq: Optional[float],
    notch_freq: Optional[float],
    gpass: float,
    gstop: float,
    filtfilt: bool,
) -> TimeSeries:
    """Apply filter to TimeSeries. Helper for impulse response measurement."""
    if filter_type == "bandpass":
        return ts.bandpass(
            low_freq, high_freq, gpass=gpass, gstop=gstop, filtfilt=filtfilt
        )
    elif filter_type == "lowpass":
        return ts.lowpass(high_freq, gpass=gpass, gstop=gstop, filtfilt=filtfilt)
    elif filter_type == "highpass":
        return ts.highpass(low_freq, gpass=gpass, gstop=gstop, filtfilt=filtfilt)
    elif filter_type == "notch":
        return ts.notch(notch_freq)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def _measure_impulse_response_duration(
    filter_type: str,
    sample_rate: int,
    low_freq: Optional[float] = None,
    high_freq: Optional[float] = None,
    notch_freq: Optional[float] = None,
    notch_q: float = 30.0,
    gpass: float = 2.0,
    gstop: float = 30.0,
    filtfilt: bool = True,
    energy_threshold: float = 1e-10,
    max_duration: float = 60.0,
) -> float:
    """Measure filter edge duration by computing impulse response energy decay.

    Creates an impulse, applies the filter, and finds the time at which
    the cumulative squared magnitude of the impulse response reaches
    (1 - energy_threshold) of its total energy.

    The buffer size is iteratively doubled until the edge energy is negligible,
    ensuring we capture the true total energy.

    Args:
        filter_type: Type of filter (bandpass, lowpass, highpass, notch)
        sample_rate: Sample rate in Hz
        low_freq: Low frequency cutoff
        high_freq: High frequency cutoff
        notch_freq: Notch frequency
        notch_q: Q-factor for notch filter
        gpass: Maximum loss in passband (dB)
        gstop: Minimum attenuation in stopband (dB)
        filtfilt: Whether zero-phase filtering is used
        energy_threshold: Fraction of energy that can remain in tails (default 1e-10)
        max_duration: Maximum duration to consider in seconds (default 60.0)

    Returns:
        Edge duration in seconds where impulse response energy is essentially zero
    """
    # Start with a small initial duration and expand if needed
    if filter_type == "bandpass" and low_freq and low_freq > 0:
        initial_duration = 2.0 / low_freq
    elif filter_type == "highpass" and low_freq and low_freq > 0:
        initial_duration = 2.0 / low_freq
    elif filter_type == "lowpass" and high_freq and high_freq > 0:
        initial_duration = 2.0 / high_freq
    elif filter_type == "notch" and notch_freq and notch_freq > 0:
        initial_duration = 2.0 / notch_freq
    else:
        initial_duration = 0.5

    duration = max(initial_duration, 0.1)

    # Iteratively expand buffer until we've captured essentially all the energy
    # We check this by verifying the energy in the outer 10% of the buffer is negligible
    edge_check_fraction = 0.1
    edge_energy_threshold = (
        energy_threshold * 0.01
    )  # Edge should have << threshold energy

    while duration <= max_duration:
        n_samples = int(duration * sample_rate)
        assert n_samples >= 10, f"n_samples={n_samples} too small"

        # Create impulse at center of buffer
        impulse = np.zeros(n_samples * 2)
        impulse[n_samples] = 1.0

        # Create TimeSeries and apply filter
        ts = TimeSeries(impulse, sample_rate=sample_rate, t0=0, channel="IMPULSE")

        try:
            filtered = _apply_filter_to_timeseries(
                ts, filter_type, low_freq, high_freq, notch_freq, gpass, gstop, filtfilt
            )
        except Exception:
            # If filtering fails, fall back to current duration
            return duration

        # Get impulse response
        h = np.asarray(filtered.value)
        h_squared = h * h
        total_energy = np.sum(h_squared)

        assert total_energy > 0, "Filter produced zero energy impulse response"

        # Check energy in outer edges of buffer
        edge_samples = int(len(h) * edge_check_fraction)
        left_edge_energy = np.sum(h_squared[:edge_samples])
        right_edge_energy = np.sum(h_squared[-edge_samples:])
        edge_energy_fraction = (left_edge_energy + right_edge_energy) / total_energy

        # If edge energy is negligible, we've captured enough of the response
        if edge_energy_fraction < edge_energy_threshold:
            break

        # Otherwise, double the buffer size and try again
        duration *= 2

    # Now find where cumulative energy reaches threshold
    center = len(h) // 2
    max_radius = center

    # Binary search for the radius where we exceed threshold
    # This is more efficient than linear search for long impulse responses
    threshold = 1.0 - energy_threshold

    # Compute cumulative energy from center outward
    cumulative_energy = np.zeros(max_radius)
    running_sum = h_squared[center] if center < len(h) else 0.0
    cumulative_energy[0] = running_sum

    for i in range(1, max_radius):
        left_idx = center - i
        right_idx = center + i
        if left_idx >= 0:
            running_sum += h_squared[left_idx]
        if right_idx < len(h):
            running_sum += h_squared[right_idx]
        cumulative_energy[i] = running_sum

    # Find where cumulative energy reaches threshold
    energy_fraction = cumulative_energy / total_energy
    indices = np.where(energy_fraction >= threshold)[0]

    assert len(indices) > 0, "Cumulative energy never reached threshold"
    edge_samples = indices[0]

    edge_duration = edge_samples / sample_rate

    # Add small safety margin (10%)
    return edge_duration * 1.1


@dataclass
class GWpyFilter(TSTransform):
    """Apply GWpy filtering operations to streaming time series data.

    Wraps GWpy's TimeSeries.bandpass(), lowpass(), highpass(), and notch()
    methods for use in streaming pipelines. Uses the audio adapter framework
    for proper overlap handling to produce continuous output without edge
    artifacts.

    The adapter automatically:
    1. Accumulates data with overlap at buffer boundaries
    2. Provides overlapped buffers to the process method
    3. Manages buffer timing for seamless streaming

    Args:
        filter_type:
            Type of filter: "bandpass", "lowpass", "highpass", or "notch"
        low_freq:
            Low frequency cutoff in Hz (for bandpass, highpass)
        high_freq:
            High frequency cutoff in Hz (for bandpass, lowpass)
        notch_freq:
            Frequency to notch out in Hz (for notch filter)
        notch_q:
            Q-factor for notch filter (default 30.0)
        gpass:
            Maximum loss in the passband in dB (default 2.0)
        gstop:
            Minimum attenuation in the stopband in dB (default 30.0)
        filtfilt:
            Use zero-phase filtering (default True). Recommended for
            streaming as it produces symmetric edge effects.
        edge_duration:
            Override edge effect duration in seconds. If None, computed
            automatically by measuring the filter's impulse response.
            (default: None)
        energy_threshold:
            Fraction of impulse response energy allowed in tails when
            computing edge duration automatically (default 1e-10).
            Smaller values = more conservative edge duration.
        input_sample_rate:
            Expected input sample rate in Hz for adapter configuration.
            (default: 4096)

    Example:
        >>> # Bandpass filter 20-500 Hz
        >>> filt = GWpyFilter(
        ...     name="Bandpass",
        ...     sink_pad_names=("in",),
        ...     source_pad_names=("out",),
        ...     filter_type="bandpass",
        ...     low_freq=20,
        ...     high_freq=500,
        ... )
        >>>
        >>> # Notch out 60 Hz power line
        >>> notch = GWpyFilter(
        ...     name="Notch60Hz",
        ...     sink_pad_names=("in",),
        ...     source_pad_names=("out",),
        ...     filter_type="notch",
        ...     notch_freq=60,
        ...     notch_q=30,
        ... )

    Note:
        - Uses filtfilt=True by default for zero-phase filtering
        - Gap buffers reset the accumulator and pass through unchanged
        - Uses zero-padding at startup (zeros precede t=0)
        - Trailing overlap introduces latency (waits for next buffer)
        - Output timestamps align with input at buffer boundaries
    """

    filter_type: str = "bandpass"
    low_freq: Optional[float] = None
    high_freq: Optional[float] = None
    notch_freq: Optional[float] = None
    notch_q: float = 30.0
    gpass: float = 2.0  # Max loss in passband (dB)
    gstop: float = 30.0  # Min attenuation in stopband (dB)
    filtfilt: bool = True
    edge_duration: Optional[float] = None
    energy_threshold: float = 1e-10
    input_sample_rate: int = 4096

    def __post_init__(self):
        """Configure adapter for filter buffering with overlap."""
        # Compute edge duration by measuring impulse response
        if self.edge_duration is not None:
            edge_sec = self.edge_duration
        else:
            edge_sec = _measure_impulse_response_duration(
                filter_type=self.filter_type,
                sample_rate=self.input_sample_rate,
                low_freq=self.low_freq,
                high_freq=self.high_freq,
                notch_freq=self.notch_freq,
                notch_q=self.notch_q,
                gpass=self.gpass,
                gstop=self.gstop,
                filtfilt=self.filtfilt,
                energy_threshold=self.energy_threshold,
            )

        # Convert to samples
        edge_samples = int(np.ceil(edge_sec * self.input_sample_rate))

        # Configure adapter with overlap on both sides
        # Leading overlap: from previous buffer (or zeros at startup)
        # Trailing overlap: from next buffer (introduces latency)
        self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (
            Offset.fromsamples(edge_samples, self.input_sample_rate),
            Offset.fromsamples(edge_samples, self.input_sample_rate),
        )
        self.adapter_config.skip_gaps = True
        # Pad zeros at startup (zeros precede t=0, so output starts at t=0)
        self.adapter_config.on_startup(pad_zeros=True)

        super().__post_init__()

        # Store edge samples for use in process
        self._edge_samples = edge_samples

    @validator.one_to_one
    def validate(self) -> None:
        """Validate filter configuration."""
        if self.filter_type == "bandpass":
            if self.low_freq is None or self.high_freq is None:
                raise ValueError("bandpass filter requires low_freq and high_freq")
        elif self.filter_type == "lowpass":
            if self.high_freq is None:
                raise ValueError("lowpass filter requires high_freq")
        elif self.filter_type == "highpass":
            if self.low_freq is None:
                raise ValueError("highpass filter requires low_freq")
        elif self.filter_type == "notch":
            if self.notch_freq is None:
                raise ValueError("notch filter requires notch_freq")
        else:
            raise ValueError(
                f"Unknown filter_type: {self.filter_type}. "
                "Must be bandpass, lowpass, highpass, or notch."
            )

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Process input frame through the filter.

        The adapter provides overlapped data automatically. We filter the
        full overlapped buffer, then extract the center portion that matches
        the expected output size (output_frame.noffset).
        """
        for buf in input_frame:
            # Pass through gap buffers unchanged
            if buf.is_gap:
                # Create a gap buffer with the correct output shape
                gap_buf = SeriesBuffer(
                    offset=output_frame.offset,
                    sample_rate=buf.sample_rate,
                    data=None,
                    shape=(Offset.tosamples(output_frame.noffset, buf.sample_rate),),
                )
                output_frame.append(gap_buf)
                continue

            # Create TimeSeries from adapter-provided overlapped data
            t0_gps = Offset.tosec(buf.offset)
            ts = TimeSeries(
                buf.data,
                t0=t0_gps,
                sample_rate=buf.sample_rate,
                channel="SGN:FILTER_INPUT",
            )

            # Apply filter to the full overlapped buffer
            filtered = self._apply_filter(ts)

            # Extract the portion corresponding to output_frame's time span
            # The adapter provides overlapped data starting at buf.offset
            # We need to extract the portion starting at output_frame.offset
            # for duration output_frame.noffset

            # Calculate expected output samples
            expected_output_samples = Offset.tosamples(
                output_frame.noffset, buf.sample_rate
            )

            # Calculate extraction start index based on GPS time alignment
            # output_frame.offset is where output should start
            # buf.offset is where the overlapped input starts
            time_offset = output_frame.offset - buf.offset
            start_idx = Offset.tosamples(time_offset, buf.sample_rate)
            end_idx = start_idx + expected_output_samples

            # Bounds check
            total_samples = len(filtered.value)
            assert start_idx >= 0, f"start_idx={start_idx} is negative"
            assert (
                end_idx <= total_samples
            ), f"end_idx={end_idx} > total_samples={total_samples}"

            # Extract the valid portion
            output_data = np.asarray(filtered.value[start_idx:end_idx])

            # Create output buffer with correct offset
            out_buf = SeriesBuffer(
                offset=output_frame.offset,
                sample_rate=buf.sample_rate,
                data=output_data,
                shape=output_data.shape,
            )

            output_frame.append(out_buf)

    def _apply_filter(self, ts):
        """Apply the configured filter to a TimeSeries."""
        if self.filter_type == "bandpass":
            return ts.bandpass(
                self.low_freq,
                self.high_freq,
                gpass=self.gpass,
                gstop=self.gstop,
                filtfilt=self.filtfilt,
            )
        elif self.filter_type == "lowpass":
            return ts.lowpass(
                self.high_freq,
                gpass=self.gpass,
                gstop=self.gstop,
                filtfilt=self.filtfilt,
            )
        elif self.filter_type == "highpass":
            return ts.highpass(
                self.low_freq,
                gpass=self.gpass,
                gstop=self.gstop,
                filtfilt=self.filtfilt,
            )
        else:
            assert (
                self.filter_type == "notch"
            ), f"Unknown filter type: {self.filter_type}"
            return ts.notch(self.notch_freq)
