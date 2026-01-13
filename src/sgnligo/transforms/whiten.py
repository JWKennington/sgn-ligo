"""An element to whiten time-series data and caclulate the power spectral density."""

# Copyright (C) 2007 Bruce Allen, Duncan Brown, Jolien Creighton, Kipp Cannon,
#                    Patrick Brady, Teviet Creighton
# Copyright (C) 2007 Bernd Machenschalk, Jolien Creighton, Kipp Cannon, Drew Keppel
# Copyright (C) 2008-2016  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2024 Becca Ewing, Joshua Gonsalves, Yun-Jing Huang

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import lal
import lal.series
import numpy as np
from igwn_ligolw import utils as ligolw_utils
from scipy import interpolate
from scipy.signal import butter, sosfilt
from scipy.special import loggamma
from sgnts.base import (
    AdapterConfig,
    EventBuffer,
    EventFrame,
    Offset,
    SeriesBuffer,
    TSFrame,
    TSTransform,
)
from sgnts.base.slice_tools import TIME_MAX
from sgnts.decorators import transform
from sympy import EulerGamma
from zlw.corrections import ExtPsdDriftCorrection
from zlw.kernels import LPWhiteningFilter, MPWhiteningFilter
from zlw.window import WindowSpec

from sgnligo.base import now

EULERGAMMA = float(EulerGamma.evalf())


def Y(length, i):
    """
    https://lscsoft.docs.ligo.org/lalsuite/lal/_window_8c_source.html#l00109

    Maps the length of a window and the offset within the window to the "y"
    co-ordinate of the LAL documentation.

    Input:
    length > 0,
    0 <= i < length

    Output:
    length < 2 --> return 0.0
    i == 0 --> return -1.0
    i == (length - 1) / 2 --> return 0.0
    i == length - 1 --> return +1.0

    e.g., length = 5 (odd), then i == 2 --> return 0.0
    if length = 6 (even), then i == 2.5 --> return 0.0

    (in the latter case, obviously i can't be a non-integer, but that's the
    value it would have to be for this function to return 0.0)
    """
    length -= 1
    return (2 * i - length) / length if length > 0 else 0


def interpolate_psd(
    psd: lal.REAL8FrequencySeries, deltaF: int
) -> lal.REAL8FrequencySeries:
    """Interpolates a PSD to a target frequency resolution.

    Args:
        psd:
            lal.REAL8FrequencySeries, the PSD to interpolate
        deltaF:
            int, the target frequency resolution to interpolate to

    Returns:
        lal.REAL8FrequencySeries, the interpolated PSD

    """
    # no-op?
    if deltaF == psd.deltaF:
        return psd

    # interpolate log(PSD) with cubic spline.  note that the PSD is
    # clipped at 1e-300 to prevent nan's in the interpolator (which
    # doesn't seem to like the occasional sample being -inf)
    psd_data = psd.data.data
    psd_data = np.where(psd_data, psd_data, 1e-300)
    f = psd.f0 + np.arange(len(psd_data)) * psd.deltaF
    interp = interpolate.splrep(f, np.log(psd_data), s=0)
    f = (
        psd.f0
        + np.arange(round((len(psd_data) - 1) * psd.deltaF / deltaF) + 1) * deltaF
    )
    psd_data = np.exp(interpolate.splev(f, interp, der=0))

    # return result
    psd = lal.CreateREAL8FrequencySeries(
        name=psd.name,
        epoch=psd.epoch,
        f0=psd.f0,
        deltaF=deltaF,
        sampleUnits=psd.sampleUnits,
        length=len(psd_data),
    )
    psd.data.data = psd_data

    return psd


@dataclass
class Whiten(TSTransform):
    """Whiten input timeseries data.

    Args:
        instrument:
            str, instrument to process. Used if reference-psd is given
        sample-rate:
            int, sample rate of the data
        fft-length:
            int, length of fft in seconds used for whitening
        nmed:
            int, how many previous samples we should account for when calcualting the
            geometric mean of the psd
        navg:
            int, changes to the PSD must occur over a time scale of at least
            navg*(n/2 - z)*(1/sample_rate) *check cody's paper for more info
        reference_psd:
            file, path to reference psd xml
        psd_pad_name:
            str, pad name of the psd output source pad
        highpass_filter:
            bool, set whether to add a Butterworth highpass filter at 8Hz
            before data whitening
    """

    instrument: Optional[str] = None
    psd_pad_name: str = ""
    whiten_pad_name: str = ""
    input_sample_rate: int = 16384
    whiten_sample_rate: int = 2048
    fft_length: int = 8
    nmed: int = 7
    navg: int = 64
    reference_psd: Optional[str] = None
    highpass_filter: bool = False

    def __post_init__(self):
        assert len(self.sink_pad_names) == 1, "Only supports one sink pad"
        assert (
            len(self.source_pad_names) == 0
        ), "source_pad_names are derived from whiten_pad_name and psd_pad_name"
        assert self.whiten_pad_name and self.psd_pad_name
        self.source_pad_names = (self.whiten_pad_name, self.psd_pad_name)

        # define block overlap following arxiv:1604.04324
        self.n_input = int(self.fft_length * self.input_sample_rate)
        self.z_input = int(self.fft_length / 4 * self.input_sample_rate)
        self.hann_length_input = self.n_input - 2 * self.z_input
        overlap_input = self.hann_length_input // 2

        self.n_whiten = int(self.fft_length * self.whiten_sample_rate)
        self.z_whiten = int(self.fft_length / 4 * self.whiten_sample_rate)
        self.hann_length_whiten = self.n_whiten - 2 * self.z_whiten

        # init audio addapter
        self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (
            0,
            Offset.fromsamples(overlap_input, self.input_sample_rate),
        )
        self.adapter_config.stride = Offset.fromsamples(
            self.hann_length_input // 2, self.input_sample_rate
        )
        self.adapter_config.skip_gaps = True
        self.stride_samples = Offset.tosamples(
            self.adapter_config.stride, self.whiten_sample_rate
        )

        super().__post_init__()

        self.latest_psd = None
        self.output_frames = {p: None for p in self.source_pads}

        # set up for whitening:
        # the offset of the first output buffer
        self.first_output_offset = None

        # keep track of number of instantaneous PSDs
        # we have calculated up to navg
        self.n_samples = 0

        # set requested sampling rates
        self.delta_f_whiten = 1 / (1 / self.whiten_sample_rate) / self.n_whiten
        self.delta_t_whiten = 1 / self.whiten_sample_rate
        self.lal_normalization_constant_whiten = 2 * self.delta_f_whiten

        # store last nmed instantaneous PSDs
        self.square_data_bufs = deque(maxlen=self.nmed)
        self.prev_data = None

        # initialize window functions
        # we apply a hann window to incoming raw data
        self.hann_input, self.hann_norm_input = self.hann_window(
            self.n_input, self.z_input
        )
        self.hann_whiten, self.hann_norm_whiten = self.hann_window(
            self.n_whiten, self.z_whiten
        )

        # we apply a tukey window on whitened data if we have zero-padding
        # z_whiten must be > 0 due to stride/sample rate constraints
        assert self.z_whiten > 0, (
            f"z_whiten must be positive, got {self.z_whiten}. "
            f"Check fft_length ({self.fft_length}) and "
            f"whiten_sample_rate ({self.whiten_sample_rate})."
        )
        self.tukey = self.tukey_window(self.n_whiten, 2 * self.z_whiten / self.n_whiten)

        # load reference PSD if provided
        if self.reference_psd:
            psd = lal.series.read_psd_xmldoc(
                ligolw_utils.load_filename(
                    self.reference_psd,
                    verbose=True,
                    contenthandler=lal.series.PSDContentHandler,
                )
            )
            psd = psd[self.instrument]

            # gstlal.condition
            # def psd_units_or_resolution_changed(elem, pspec, psd):
            # make sure units are set, compute scale factor
            # FIXME: what is this units?
            # units = lal.Unit(elem.get_property("psd-units"))
            # if units == lal.DimensionlessUnit:
            #    return
            # scale = float(psd.sampleUnits / units)
            scale = 1

            # get frequency resolution and number of bins
            fnyquist = self.whiten_sample_rate / 2
            n = int(round(fnyquist / self.delta_f_whiten) + 1)

            # interpolate and rescale PSD
            psd = interpolate_psd(psd, self.delta_f_whiten)
            ref_psd_data = psd.data.data[:n] * scale

            # install PSD in buffer history
            self.set_psd(ref_psd_data, self.navg)

    def tukey_window(self, length, beta):
        """
        XLALCreateTukeyREAL8Window
        https://lscsoft.docs.ligo.org/lalsuite/lal/_window_8c_source.html#l00597

        1.0 and flat in the middle, cos^2 transition at each end, zero
        at end points, 0.0 <= beta <= 1.0 sets what fraction of the
        window is transition (0 --> rectangle window, 1 --> Hann window)
        """
        if beta < 0 or beta > 1:
            raise ValueError("Invalid value for beta")

        transition_length = round(beta * length)

        out = np.ones(length)
        for i in range((transition_length + 1) // 2):
            o = np.cos(np.pi / 2 * Y(transition_length, i)) ** 2
            out[i] = o
            out[length - 1 - i] = o

        return out

    def hann_window(self, N, Z):
        """
        Define hann window
        Parameters:
        -----------
        N: int
            Number of samples in one window block
        Z: int
            Number of samples to zero pad
        """
        # array of indices
        k = np.arange(0, N, 1)

        hann = np.zeros(N)
        hann[Z : N - Z] = (np.sin(np.pi * (k[Z : N - Z] - Z) / (N - 2 * Z))) ** 2

        # FIXME gstlal had a method of adding from the two ends of the window
        # so that small numbers weren't added to big ones
        hann_norm = np.sqrt(N / np.sum(hann**2))

        return hann, hann_norm

    def median_bias(self, nn):
        """
        XLALMedianBias
        https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l00378
        """
        ans = 1
        n = (nn - 1) // 2
        for i in range(1, n + 1):
            ans -= 1.0 / (2 * i)
            ans += 1.0 / (2 * i + 1)

        return ans

    def log_median_bias_geometric(self, nn):
        """
        XLALLogMedianBiasGeometric
        https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01423
        """
        return np.log(self.median_bias(nn)) - nn * (loggamma(1 / nn) - np.log(nn))

    def add_psd(self, fdata):
        """
        XLALPSDRegressorAdd
        https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01632
        """
        self.square_data_bufs.append(np.abs(fdata) ** 2)

        if self.n_samples == 0:
            self.geometric_mean_square = np.log(self.square_data_bufs[0])
            self.n_samples += 1
        else:
            self.n_samples += 1
            self.n_samples = min(self.n_samples, self.navg)
            median_bias = self.log_median_bias_geometric(len(self.square_data_bufs))

            # FIXME: this is how XLALPSDRegressorAdd gets the median,
            # but this is not exactly the median when the number is even.
            # numpy takes the average of the middle two, while this gets
            # the larger one
            log_bin_median = np.log(
                np.sort(self.square_data_bufs, axis=0)[len(self.square_data_bufs) // 2]
            )
            self.geometric_mean_square = (
                self.geometric_mean_square * (self.n_samples - 1)
                + log_bin_median
                - median_bias
            ) / self.n_samples

    def get_psd(self, fdata):
        """
        XLALPSDRegressorGetPSD
        https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01773
        """
        # running average mode (track-psd)
        if self.n_samples == 0:
            out = self.lal_normalization_constant_whiten * (np.abs(fdata) ** 2)

            # set DC and Nyquist terms to zero
            # FIXME: gstlal had a condition if self.f0 == 0
            out[0] = 0
            out[self.n_whiten // 2] = 0
            return out
        else:
            return (
                np.exp(self.geometric_mean_square + EULERGAMMA)
                * self.lal_normalization_constant_whiten
            )

    def set_psd(self, ref_psd_data, weight):
        """
        XLALPSDRegressorSetPSD
        https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01831
        """
        arithmetic_mean_square_data = (
            ref_psd_data / self.lal_normalization_constant_whiten
        )

        # populate the buffer history with the ref psd
        for _i in range(self.nmed):
            self.square_data_bufs.append(arithmetic_mean_square_data)

        self.geometric_mean_square = np.log(arithmetic_mean_square_data) - EULERGAMMA
        self.n_samples = min(weight, self.navg)

    def internal(self):
        """
        Whiten incoming data in segments of fft-length seconds overlapped by fft-length
        * 3/4. If the data segment has N samples, we apply a zero-padded Hann window on
        the data with zero-padding of Z = N/4 samples. The Hann window length is then
        N - 2 * Z. The output stride is hann_length / 2.

        Example:
        --------
        fft_length = 4 sec
        sample_rate = 4
        N = 16
        Z = 4
        hann_length = 8
        output_stride = 4

        -- : input data
        .. : zero-padding
        ** : hann window
        [] : output buffer
        {} : output that will be added to the next iteration


                     *
                    * *
                   *   *
                  *     *
                 *
        1)   ....--------....
            -1s  0s      2s  3s
                 {add to next}


                         *
                        * *
                       *   *
                      *     *
                     *
        2)       ....--------....
                0s   1s      3s  4s
                [out]{add to next}


                             *
                            * *
                           *   *
                          *     *
                         *
        3)           ....--------....
                    1s   2s      4s  5s
                    [out]{add to next}


        Each fft-length of data will be windowed by the zero-padded Hann window, then
        FFTed to obtain the instantaneous PSD. The instantaneous PSD will be saved to
        a queue to calculate the running geometric mean of median PSDs, see
        arxiv:1604.04324. The running geometric mean of median PSDs from the last
        iteration will be used to whiten the current windowed-fft-length of data.
        The overlap segment from the previous output will be added to current whitened
        data. Finally the first output-stride samples of the whitened data will be put
        into the output buffer.

        Note that we will only start to produce output when the output offset is equal
        to or after the first input buffer, so the first iteration is a gap buffer.

        """
        super().internal()
        # incoming frame handling
        frame = self.preparedframes[self.sink_pads[0]]
        EOS = frame.EOS
        metadata = frame.metadata
        outoffset_info = self.preparedoutoffsets

        if self.first_output_offset is None:
            self.first_output_offset = frame.offset

        padded_data_offset = outoffset_info["offset"] - Offset.fromsamples(
            self.z_whiten, self.whiten_sample_rate
        )

        # FIXME: can we make this more general?
        if padded_data_offset < self.first_output_offset:
            # we are in the startup stage, don't output yet
            outoffset = outoffset_info["offset"]
            shape = (0,)
        else:
            outoffset = padded_data_offset
            shape = (
                Offset.tosamples(outoffset_info["noffset"], self.whiten_sample_rate),
            )

        # the epoch of the psd is the mid point of the most recent fft
        # which corresponds to the end offset of the output + half hann
        # length
        # FIXME: double check

        psd_epoch = int(
            Offset.tons(outoffset + outoffset_info["noffset"])
            + self.hann_length_whiten // 2 / self.whiten_sample_rate * 1e9
        )

        # if audioadapter hasn't given us a frame, then we have to wait for more
        # data before we can whiten. send a gap buffer
        if frame.is_gap:
            if (
                outoffset_info["noffset"] != 0
                and self.prev_data is not None
                and self.prev_data.shape[-1] > 0
            ):
                # drain the output history
                output_whitened_data = self.prev_data[: self.stride_samples]
                self.prev_data = self.prev_data[self.stride_samples :]
            else:
                output_whitened_data = None
        else:
            # retrieve samples from the deque
            assert len(frame.buffers) == 1, "Multiple buffers not implemented yet."
            buf = frame.buffers[0]
            this_seg_data = buf.data

            if self.highpass_filter:
                sos = butter(
                    4, 8, btype="highpass", fs=self.input_sample_rate, output="sos"
                )
                this_seg_data = sosfilt(sos, this_seg_data)
            # apply the window function
            this_seg_data = (
                self.hann_input[self.z_input : -self.z_input]
                * this_seg_data
                * 1
                / self.input_sample_rate
                * self.hann_norm_input
            )
            this_seg_data = np.pad(this_seg_data, (self.z_input, self.z_input))

            # apply fourier transform
            freq_data = np.fft.rfft(this_seg_data)

            # get frequency bins
            freqs = np.fft.rfftfreq(this_seg_data.size, d=1 / self.input_sample_rate)

            # downsampling
            freq_data = freq_data[
                : int(
                    self.whiten_sample_rate
                    / self.input_sample_rate
                    * freq_data.shape[-1]
                )
                + 1
            ]

            # get the latest PSD
            this_psd = self.get_psd(freq_data)
            # store the latest spectrum so we can output on spectrum pad
            f0 = freqs[0]
            self.latest_psd = lal.CreateREAL8FrequencySeries(
                "new_psd",
                psd_epoch / 1e9,
                f0,
                self.delta_f_whiten,
                "s strain^2",
                len(this_psd),
            )
            self.latest_psd.data.data = this_psd

            # push freq data into psd history
            self.add_psd(freq_data)

            # Whitening
            # the DC and Nyquist terms are zero
            freq_data_whitened = np.zeros_like(freq_data)
            freq_data_whitened[1:-1] = freq_data[1:-1] * np.sqrt(
                self.lal_normalization_constant_whiten / this_psd[1:-1]
            )

            # Fourier Transform back to the time domain
            # # see arxiv: 1604.04324 (13)
            # self.delta_f scaling https://lscsoft.docs.ligo.org/lalsuite/lal/
            # _time_freq_f_f_t_8c_source.html#l00183
            whitened_data = (
                np.fft.irfft(freq_data_whitened, self.n_whiten, norm="forward")
                * self.delta_f_whiten
            )
            whitened_data *= self.delta_t_whiten * np.sqrt(np.sum(self.hann_whiten**2))

            if self.tukey is not None:
                whitened_data *= self.tukey

            # accounts for overlap by summing with prev_data over the
            # stride of the adapter
            if self.prev_data is not None:
                whitened_data[: self.prev_data.shape[-1]] += self.prev_data
            self.prev_data = whitened_data[self.stride_samples :]

            # FIXME: can we make this more general?
            # only output data up till the length of the adapter stride
            if padded_data_offset < self.first_output_offset:
                # output a gap buffer during the startup period of the whitening
                # calculation
                output_whitened_data = None
            else:
                output_whitened_data = whitened_data[: self.stride_samples]

        # passes the spectrum in metadata if the pad is the psd_pad
        # FIXME: use EventBuffers
        if output_whitened_data is None:
            metadata["psd"] = None
            metadata["navg"] = None
            metadata["n_samples"] = None
        else:
            metadata["psd"] = self.latest_psd
            metadata["navg"] = self.navg
            metadata["n_samples"] = self.n_samples

        metadata["epoch"] = psd_epoch

        self.output_frames[self.srcs[self.psd_pad_name]] = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=outoffset,
                    sample_rate=self.whiten_sample_rate,
                    data=None,
                    shape=shape,
                )
            ],
            EOS=EOS,
            metadata=metadata,
        )

        self.output_frames[self.srcs[self.whiten_pad_name]] = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=outoffset,
                    sample_rate=self.whiten_sample_rate,
                    data=output_whitened_data,
                    shape=shape,
                )
            ],
            EOS=EOS,
            metadata=metadata,
        )

    def new(self, pad):
        return self.output_frames[pad]


# --- Kernel Container ---


@dataclass
class Kernel:
    """Container for FIR taps and latency metadata."""

    fir_matrix: np.ndarray
    latency: int


def kernel_from_psd(
    psd: lal.REAL8FrequencySeries,
    zero_latency: bool = False,
    window_spec: Optional[WindowSpec] = None,
) -> Kernel:
    """Build a time-domain whitening kernel from a one-sided PSD using ZLW."""
    psd_data = np.asarray(psd.data.data, dtype=np.float64)
    df = psd.deltaF
    f0 = psd.f0

    n_fft = 2 * (len(psd_data) - 1)
    fs = 2.0 * (f0 + (len(psd_data) - 1) * df)

    if zero_latency:
        # Minimum Phase: Causal, Latency = 0
        filt = MPWhiteningFilter(psd=psd_data, fs=fs, n_fft=n_fft)
        taps = filt.impulse_response(window=window_spec)
        latency = 0
    else:
        # Linear Phase: Symmetric, Latency = Group Delay
        # We must explicitly set the delay to center the impulse for a causal filter.
        # Length of impulse response typically defaults to n_fft
        # unless windowed shorter, but here we rely on standard n_fft length.

        # Center of the filter in samples
        delay_samples = (n_fft - 1) / 2.0
        delay_seconds = delay_samples / fs

        filt = LPWhiteningFilter(psd=psd_data, fs=fs, n_fft=n_fft, delay=delay_seconds)
        taps = filt.impulse_response(window=window_spec)

        # The peak of the LP filter is now at index `delay_samples`
        latency = int(delay_samples)

    # Normalize energy
    norm = np.linalg.norm(taps)
    if norm > 0:
        taps /= norm

    return Kernel(fir_matrix=taps, latency=latency)


def correction_kernel_from_psds(
    psd_live: lal.REAL8FrequencySeries,
    psd_ref: lal.REAL8FrequencySeries,
    truncation_samples: int = 1024,
    smoothing_hz: float = 0.0,
) -> Kernel:
    """Generate a drift correction kernel using ZLW Adjoint logic."""
    assert len(psd_live.data.data) == len(psd_ref.data.data)

    n_bins = len(psd_live.data.data)
    fs = 2.0 * (n_bins - 1) * psd_live.deltaF

    corrector = ExtPsdDriftCorrection(
        freqs=np.linspace(0, fs / 2, n_bins),
        psd_ref=np.asarray(psd_live.data.data, dtype=float),
        psd_live=np.asarray(psd_ref.data.data, dtype=float),
        fs=fs,
    )

    k_circ = corrector.compute_adjoint_kernel(
        smoothing_hz=smoothing_hz,
        truncate_samples=truncation_samples,
        check_aliasing=False,
    )

    L = truncation_samples
    N = len(k_circ)
    taps = np.zeros(L, dtype=np.float64)

    if L > 1:
        taps[0 : L - 1] = k_circ[N - (L - 1) : N]

    taps[L - 1] = k_circ[0]

    return Kernel(fir_matrix=taps, latency=L - 1)


@dataclass
class WhiteningKernel(TSTransform):
    """Generates a live whitening kernel (1/sqrt(S_live))."""

    filters_pad_name: str = "filters"
    zero_latency: bool = False
    window_spec: Optional[WindowSpec] = None
    min_update_interval: Optional[int] = None
    similarity_threshold: float = 0.9999
    verbose: bool = False

    @property
    def static_source_pads(self) -> list[str]:
        return [self.filters_pad_name]

    def configure(self) -> None:
        self.output_frame_types[self.filters_pad_name] = EventFrame
        self._latest_epoch: Optional[int] = None
        self._latest_send_ns: Optional[int] = None
        self._last_psd_data: Optional[np.ndarray] = None
        self.source_pad = self.srcs[self.filters_pad_name]

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: EventFrame) -> None:
        output_frame.is_gap = True
        live_psd = input_frame.metadata.get("psd", None)
        epoch = input_frame.metadata.get("epoch", None)

        if epoch is None or live_psd is None:
            return

        if self._latest_epoch is not None and epoch <= self._latest_epoch:
            return

        if self.min_update_interval:
            curr_ns = now().ns()
            if (
                self._latest_send_ns
                and (curr_ns - self._latest_send_ns) < self.min_update_interval
            ):
                return
            self._latest_send_ns = curr_ns

        self._latest_epoch = epoch

        # --- Optimization: Check PSD Similarity BEFORE Computation ---
        current_psd_data = np.asarray(live_psd.data.data)

        if self._last_psd_data is not None and len(current_psd_data) == len(
            self._last_psd_data
        ):
            diff_norm = np.linalg.norm(current_psd_data - self._last_psd_data)
            ref_norm = np.linalg.norm(current_psd_data)

            # If relative change is within tolerance, skip update
            if ref_norm > 0 and (diff_norm / ref_norm) < (
                1.0 - self.similarity_threshold
            ):
                if self.verbose:
                    print(f"WhiteningKernel: PSD stable at {epoch}, skipping update.")
                return

        # Update cache (important to copy to ensure persistence)
        self._last_psd_data = current_psd_data.copy()

        try:
            k = kernel_from_psd(
                live_psd, zero_latency=self.zero_latency, window_spec=self.window_spec
            )
            taps = k.fir_matrix

            # Update frame bounds to Infinite
            output_frame.noffset = int(TIME_MAX) - output_frame.offset

            buf = EventBuffer(
                offset=output_frame.offset,
                noffset=output_frame.noffset,
                data=[np.asarray([taps])],
            )
            output_frame.append(buf)
            output_frame.is_gap = False

            if self.verbose:
                print(f"WhiteningKernel: Updated at {epoch}, lat={k.latency}")

        except Exception as e:
            if self.verbose:
                print(f"WhiteningKernel Error: {e}")


@dataclass
class DriftCorrectionKernel(TSTransform):
    """Generates a correction kernel using ZLW Adjoint logic."""

    filters_pad_name: str = "filters"
    reference_psd: Optional[lal.REAL8FrequencySeries] = None
    truncation_samples: int = 1024
    smoothing_hz: float = 0.0
    min_update_interval: Optional[int] = None
    similarity_threshold: float = 0.9999
    verbose: bool = False

    @property
    def static_source_pads(self) -> list[str]:
        return [self.filters_pad_name]

    def configure(self) -> None:
        self.output_frame_types[self.filters_pad_name] = EventFrame
        self._latest_epoch: Optional[int] = None
        self._latest_send_ns: Optional[int] = None
        self._last_psd_data: Optional[np.ndarray] = None
        self._interpolated_ref_psd: Optional[lal.REAL8FrequencySeries] = None
        self.source_pad = self.srcs[self.filters_pad_name]

        if self.reference_psd is None and self.verbose:
            print("DriftCorrectionKernel warning: No reference_psd provided.")

    def _get_interpolated_reference(
        self, live_psd: lal.REAL8FrequencySeries
    ) -> Optional[lal.REAL8FrequencySeries]:
        """Align the static reference PSD to the frequency grid of the live PSD."""
        target_df = live_psd.deltaF
        target_len = len(live_psd.data.data)

        # Check if cache is valid
        if self._interpolated_ref_psd is not None:
            if (
                np.isclose(self._interpolated_ref_psd.deltaF, target_df)
                and len(self._interpolated_ref_psd.data.data) == target_len
            ):
                return self._interpolated_ref_psd

        # Re-interpolate
        try:
            # 1. Resample to match frequency spacing
            interp_psd = interpolate_psd(self.reference_psd, target_df)

            # 2. Force length match (truncate or pad if f_max differs)
            interp_len = len(interp_psd.data.data)
            if interp_len != target_len:
                new_series = lal.CreateREAL8FrequencySeries(
                    interp_psd.name,
                    interp_psd.epoch,
                    interp_psd.f0,
                    interp_psd.deltaF,
                    interp_psd.sampleUnits,
                    target_len,
                )
                n_copy = min(interp_len, target_len)
                new_series.data.data[:n_copy] = interp_psd.data.data[:n_copy]
                if target_len > interp_len:
                    new_series.data.data[n_copy:] = 0.0
                interp_psd = new_series

            self._interpolated_ref_psd = interp_psd
            return interp_psd

        except Exception as e:
            if self.verbose:
                print(f"DriftCorrection Error interpolating PSD: {e}")
            return None

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: EventFrame) -> None:
        output_frame.is_gap = True
        live_psd = input_frame.metadata.get("psd", None)
        epoch = input_frame.metadata.get("epoch", None)

        if epoch is None or self.reference_psd is None or live_psd is None:
            return

        if self._latest_epoch is not None and epoch <= self._latest_epoch:
            return

        if self.min_update_interval:
            curr_ns = now().ns()
            if (
                self._latest_send_ns
                and (curr_ns - self._latest_send_ns) < self.min_update_interval
            ):
                return
            self._latest_send_ns = curr_ns

        self._latest_epoch = epoch

        # Align Reference
        ref_to_use = self._get_interpolated_reference(live_psd)
        if ref_to_use is None:
            return

        # Optimization Check
        current_psd_data = np.asarray(live_psd.data.data)
        if self._last_psd_data is not None and len(current_psd_data) == len(
            self._last_psd_data
        ):
            diff_norm = np.linalg.norm(current_psd_data - self._last_psd_data)
            ref_norm = np.linalg.norm(current_psd_data)
            if ref_norm > 0 and (diff_norm / ref_norm) < (
                1.0 - self.similarity_threshold
            ):
                return

        self._last_psd_data = current_psd_data.copy()

        try:
            k = correction_kernel_from_psds(
                psd_live=live_psd,
                psd_ref=ref_to_use,
                truncation_samples=self.truncation_samples,
                smoothing_hz=self.smoothing_hz,
            )
            taps = k.fir_matrix

            output_frame.noffset = int(TIME_MAX) - output_frame.offset

            buf = EventBuffer(
                offset=output_frame.offset,
                noffset=output_frame.noffset,
                data=[np.asarray([taps])],
            )
            output_frame.append(buf)
            output_frame.is_gap = False

            if self.verbose:
                print(f"DriftCorrection: Updated {epoch}, lat={k.latency}")

        except Exception as e:
            if self.verbose:
                import traceback

                print(f"DriftCorrectionKernel Error: {e}")
                traceback.print_exc()
