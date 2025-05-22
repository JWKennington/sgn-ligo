"""A source element to generate realistic noise inspired by modern GW detectors.

This module provides the GWDataNoiseSource class which generates colored noise
with spectral characteristics inspired by Advanced LIGO and Virgo detectors.
The noise is generated using FIR filtering of white noise to achieve
realistic power spectral density characteristics for testing and simulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy
from scipy import signal
from sgn.base import SourcePad
from sgnts.base import Offset, TSFrame, TSSource

from sgnligo.base import now
from sgnligo.kernels import PSDFirKernel
from sgnligo.psd import fake_gwdata_psd


def parse_psd(channel_dict):
    """Parse the PSDs for the given channels.

    Args:
        channel_dict: Dictionary mapping detector names to channel names

    Returns:
        Dictionary containing PSD information for each detector
    """
    _psd = fake_gwdata_psd(channel_dict.keys())
    out = {}
    FIRKernel = PSDFirKernel()

    for ifo, channel_name in channel_dict.items():
        assert ifo in _psd
        psd = _psd[ifo]
        nyquist = (len(psd.data.data) - 1) * psd.deltaF
        ln2nyquist = numpy.log2(nyquist)
        assert nyquist == int(nyquist)
        assert ln2nyquist == int(ln2nyquist)
        rate = int(nyquist * 2)
        stride = Offset.sample_stride(rate)

        # Create the coloring FIR kernel from reference_psd.psd_to_fir_kernel()
        fir_matrix, latency, measured_sample_rate = (
            FIRKernel.psd_to_linear_phase_whitening_fir_kernel(psd, invert=False)
        )
        out[ifo] = {
            "channel-name": channel_name,
            "rate": rate,
            "psd": psd,
            "sample-stride": stride,
            "state": numpy.random.randn(stride + len(fir_matrix) - 1),
            "fir-matrix": fir_matrix,
        }
    return out


@dataclass
class GWDataNoiseSource(TSSource):
    """Source element to generate realistic noise inspired by modern GW detectors.

    This source generates noise with spectral characteristics inspired by 
    Advanced LIGO and Virgo detectors. The noise is colored using realistic
    power spectral density curves suitable for testing and simulation purposes.

    Args:
        channel_dict:
            dict or None. If None use {"H1":"H1:FAKE-STRAIN", "L1":"L1:FAKE-STRAIN"}
        t0:
            float or None, start GPS time. If None, use current time.
        end:
            float or None, end GPS time. If None, run indefinitely.
        duration:
            float or None, duration GPS time. Cannot be combined with end. Use
            one or the other.
        real_time:
            bool, if True, generate data in real time, otherwise as fast as possible.
        verbose:
            bool, if True, print additional information.
    """

    channel_dict: Optional[dict] = None
    real_time: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Initialize the source after creation.

        This sets up the PSD, filter coefficients, and initial state for noise
        generation.
        """

        if self.channel_dict is None:
            self.channel_dict = {"H1": "H1:FAKE-STRAIN", "L1": "L1:FAKE-STRAIN"}

        self.channel_info = parse_psd(self.channel_dict)
        self.source_pad_names = [
            info["channel-name"] for info in self.channel_info.values()
        ]

        # Set proper t0 value before calling parent's __post_init__
        if self.t0 is None:
            self.t0 = int(now())

        # Call parent's post_init BEFORE setting buffer parameters
        super().__post_init__()

        # Associate the pads with the channel_info and set buffer params
        for info in self.channel_info.values():
            pad = self.srcs[info["channel-name"]]
            info.update({"pad": pad})
            self.set_pad_buffer_params(
                pad=pad,
                sample_shape=(),  # Scalar data (strain)
                rate=info["rate"],
            )

        # Initialize current_end attribute
        self._current_end = 0.0

        if self.verbose:
            if self.end is None:
                print("No end time specified, will run indefinitely")
            else:
                print(f"Will run until GPS time: {self.end}")

    def _generate_noise_chunk(self, pad: SourcePad) -> numpy.ndarray:
        """Generate a chunk of colored noise with proper continuity.

        This method applies an FIR filter to white noise, producing colored noise
        with the desired LIGO PSD. It maintains filter state between calls to ensure
        there are no discontinuities in the generated noise.

        Args:
            pad: Source pad requesting new data

        Returns:
            NumPy array containing colored noise
        """

        # Get the info for this detector
        info = [info for info in self.channel_info.values() if info["pad"] == pad][0]
        out = signal.correlate(info["state"], info["fir-matrix"], "valid")

        # Maintain state for the next call
        info["state"][: -len(out)] = info["state"][len(out) :]
        info["state"][-len(out) :] = numpy.random.randn(len(out))

        return out

    def new(self, pad: SourcePad) -> TSFrame:
        """Generate a new frame with colored noise matching LIGO PSD.

        This method is called by the base class's prepare_frame method, which manages
        the timing and buffer creation for us.

        Args:
            pad: Source pad requesting new data

        Returns:
            TSFrame containing realistic LIGO noise
        """
        # Get the frame prepared by the base class's prepare_frame method
        frame = self.prepare_frame(pad)

        # Get the buffer from the frame
        assert len(frame) == 1
        buffer = frame.buffers[0]

        # Generate noise for this channel
        noise_chunk = self._generate_noise_chunk(pad)

        # Set the data in the buffer
        buffer.set_data(noise_chunk)

        # Store the current end time of the frame
        self._current_end = frame.end

        return frame

    def internal(self) -> None:
        """Internal processing, handles real-time timing if enabled."""
        super().internal()

        if self.real_time:
            # In real-time mode, sleep until the current end time to maintain timing
            from sgnts.base.time import Time
            from sgnts.utils import gpsnow

            # Calculate the time to sleep based on current end time vs current GPS time
            sleep_time = self._current_end / Time.SECONDS - gpsnow()

            if sleep_time < 0:
                if sleep_time < -1:
                    # We're falling behind real time
                    if self.verbose:
                        print(
                            "Warning: GWDataNoiseSource falling behind real time"
                            + f"({sleep_time:.2f} s)"
                        )
            else:
                # Sleep to maintain real-time generation
                time.sleep(sleep_time)
