from collections import deque

from sgn.transforms import *
from sgnts.transforms import *
from sgnts.base import (
    SeriesBuffer,
    TSFrame,
    TSTransform,
    AdapterConfig,
)

from gwpy.timeseries import TimeSeries

import lal
import lal.series
from ligo.lw import utils as ligolw_utils

import numpy as np
import os
from sympy import EulerGamma
from scipy.special import loggamma
from scipy import interpolate

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
    """
    Whiten input timeseries data
    Parameters:
    -----------
    whitening-method: str
        currently supported types: (1) 'gwpy', (2) 'gstlal'
    instrument: str
        instrument to process. Used if reference-psd is given
    sample-rate: int
        sample rate of the data
    fft-length: int
        length of fft in seconds used for whitening
    nmed: int
        how many previous samples we should account for when calcualting
        the geometric mean of the psd
    navg: int
        changes to the PSD must occur over a time scale of at least
        navg*(n/2 − z)*(1/sample_rate) *check cody's paper for more info
    reference_psd: file
        path to reference psd xml
    psd_pad_name: str
        pad name of the psd output source pad
    """

    instrument: str = None
    whitening_method: str = "gwpy"
    sample_rate: int = 2048
    fft_length: int = 8
    nmed: int = 7
    navg: int = 64
    reference_psd: str = None
    psd_pad_name: str = ""

    def __post_init__(self):
        # define block overlap following arxiv:1604.04324
        self.n = int(self.fft_length * self.sample_rate)
        self.z = int(self.fft_length / 4 * self.sample_rate)
        self.hann_length = self.n - 2 * self.z
        overlap = self.hann_length // 2

        # init audio addapter
        self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (0, overlap)
        self.adapter_config.stride = self.hann_length // 2

        super().__post_init__()

        self.latest_psd = None

        # set up for gstlal method:
        if self.whitening_method == "gstlal":
            # the offset of the first output buffer
            self.first_output_offset = None

            # keep track of number of instantaneous PSDs
            # we have calculated up to navg
            self.n_samples = 0

            # set requested sampling rates
            self.delta_f = 1 / (1 / self.sample_rate) / self.n
            self.delta_t = 1 / self.sample_rate
            self.lal_normalization_constant = 2 * self.delta_f

            # store last nmed instantaneous PSDs
            self.square_data_bufs = deque(maxlen=self.nmed)
            self.prev_data = None

            # initialize window functions
            # we apply a hann window to incoming raw data
            self.hann = self.hann_window(self.n, self.z)

            # we apply a tukey window on whitened data if we have zero-padding
            if self.z:
                self.tukey = self.tukey_window(self.n, 2 * self.z / self.n)
            else:
                self.tukey = None

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
                fnyquist = self.sample_rate / 2
                n = int(round(fnyquist / self.delta_f) + 1)

                # interpolate and rescale PSD
                psd = interpolate_psd(psd, self.delta_f)
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

        n = (transition_length + 1) // 2

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
        self.hann_norm = np.sqrt(N / np.sum(hann**2))

        return hann

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
            out = self.lal_normalization_constant * (np.abs(fdata) ** 2)

            # set DC and Nyquist terms to zero
            # FIXME: gstlal had a condition if self.f0 == 0
            out[0] = 0
            out[self.n // 2] = 0
            return out
        else:
            return (
                np.exp(self.geometric_mean_square + EULERGAMMA)
                * self.lal_normalization_constant
            )

    def set_psd(self, ref_psd_data, weight):
        """
        XLALPSDRegressorSetPSD
        https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01831
        """
        arithmetic_mean_square_data = ref_psd_data / self.lal_normalization_constant

        # populate the buffer history with the ref psd
        for i in range(self.nmed):
            self.square_data_bufs.append(arithmetic_mean_square_data)

        self.geometric_mean_square = np.log(arithmetic_mean_square_data) - EULERGAMMA
        self.n_samples = min(weight, self.navg)

    def transform(self, pad):
        """
        Whiten incoming data in segments of fft-length seconds overlapped by fft-length * 3/4
        If the data segment has N samples, we apply a zero-padded Hann window on the data with
        zero-padding of Z = N/4 samples. The Hann window length is then N - 2 * Z. The output
        stride is hann_length / 2.

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
        # incoming frame handling
        outbufs = []
        frame = self.preparedframes[self.sink_pads[0]]
        EOS = frame.EOS
        metadata = frame.metadata
        outoffsets = self.preparedoutoffsets[self.sink_pads[0]]

        if self.first_output_offset is None:
            self.first_output_offset = frame.offset

        padded_data_offset = outoffsets[0]["offset"] - Offset.fromsamples(
            self.z, self.sample_rate
        )

        # FIXME: haven't tested whether this works for gwpy method
        # FIXME: can we make this more general?
        if padded_data_offset < self.first_output_offset:
            # we are in the startup stage, don't output yet
            outoffset = outoffsets[0]["offset"]
            shape = (0,)
        else:
            outoffset = padded_data_offset
            shape = (Offset.tosamples(outoffsets[0]["noffset"], self.sample_rate),)

        # passes the spectrum in metadata if the pad is the psd_pad
        if pad.name == self.psd_pad_name:
            psd = self.latest_psd
            metadata["psd"] = psd

            return TSFrame(
                buffers=[
                    SeriesBuffer(
                        offset=outoffset,
                        sample_rate=self.sample_rate,
                        data=None,
                        shape=shape,
                    )
                ],
                EOS=EOS,
                metadata=metadata,
            )

        # if audioadapter hasn't given us a frame, then we have to wait for more
        # data before we can whiten. send a gap buffer
        if frame.is_gap:
            outbufs.append(
                SeriesBuffer(
                    offset=outoffset,
                    sample_rate=self.sample_rate,
                    data=None,
                    shape=shape,
                )
            )
        else:
            # retrieve samples from the deque
            assert len(frame.buffers) == 1, "Multiple buffers not implemented yet."
            buf = frame.buffers[0]
            this_seg_data = buf.data

            if self.whitening_method == "gwpy":
                # check the type of the timeseries data.
                # transform it to a gwpy.timeseries object
                if not isinstance(this_seg_data, TimeSeries):
                    this_seg_data = TimeSeries(this_seg_data)

                # whiten it
                whitened_data = this_seg_data.whiten(
                    fftlength=self.fft_length, overlap=0, window="hann"
                )

                # transform back to a numpy array
                whitened_data = np.array(whitened_data)

            elif self.whitening_method == "gstlal":
                # apply the window function
                this_seg_data = (
                    self.hann[self.z : -self.z]
                    * this_seg_data
                    * self.delta_t
                    * self.hann_norm
                )
                this_seg_data = np.pad(this_seg_data, (self.z, self.z))

                # apply fourier transform
                freq_data = np.fft.rfft(this_seg_data)

                # get frequency bins
                freqs = np.fft.rfftfreq(this_seg_data.size, d=self.delta_t)

                # get the latest PSD
                this_psd = self.get_psd(freq_data)

                # store the latest spectrum so we can output on spectrum pad
                psd_offset = outoffsets[0]["offset"]
                psd_epoch = Offset.tosec(psd_offset)
                f0 = freqs[0]
                self.latest_psd = lal.CreateREAL8FrequencySeries(
                    "new_psd", psd_epoch, f0, self.delta_f, "s strain^2", len(this_psd)
                )
                self.latest_psd.data.data = this_psd

                # push freq data into psd history
                self.add_psd(freq_data)

                # Whitening
                # the DC and Nyquist terms are zero
                freq_data_whitened = np.zeros_like(freq_data)
                freq_data_whitened[1:-1] = freq_data[1:-1] * np.sqrt(
                    self.lal_normalization_constant / this_psd[1:-1]
                )

                # Fourier Transform back to the time domain
                # # see arxiv: 1604.04324 (13)
                # self.delta_f scaling https://lscsoft.docs.ligo.org/lalsuite/lal/_time_freq_f_f_t_8c_source.html#l00183
                whitened_data = (
                    np.fft.irfft(freq_data_whitened, self.n, norm="forward")
                    * self.delta_f
                )
                whitened_data *= self.delta_t * np.sqrt(np.sum(self.hann**2))

                if self.tukey is not None:
                    whitened_data *= self.tukey

                # accounts for overlap by summing with prev_data over the
                # stride of the adapter
                if self.prev_data is not None:
                    whitened_data[: -self.adapter_config.stride] += self.prev_data
                self.prev_data = whitened_data[self.adapter_config.stride :]

            # FIXME: haven't tested whether this works for gwpy method
            # FIXME: can we make this more general?
            if padded_data_offset < self.first_output_offset:
                # output a gap buffer during the startup period of the whitening calculation
                outdata = None
            else:
                outdata = whitened_data[: self.adapter_config.stride]

            # only output data up till the length of the adapter stride
            outbufs.append(
                SeriesBuffer(
                    offset=outoffset,
                    sample_rate=self.sample_rate,
                    data=outdata,
                    shape=shape,
                )
            )

        # return frame with the correct buffers
        return TSFrame(
            buffers=outbufs,
            EOS=EOS,
            metadata=metadata,
        )
