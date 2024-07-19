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

EULERGAMMA = float(EulerGamma.evalf())
print(EULERGAMMA)

def interpolate_psd(psd: lal.REAL8FrequencySeries, deltaF: int) -> lal.REAL8FrequencySeries:
    """Interpolates a PSD to a target frequency resolution.

    Args:
        psd:
            lal.REAL8FrequencySeries, the PSD to interpolate
        deltaF:
            int, the target frequency resolution to interpolate to

    Returns:
        lal.REAL8FrequencySeries, the interpolated PSD

    """
    #
    # no-op?
    #

    if deltaF == psd.deltaF:
        return psd

    #
    # interpolate log(PSD) with cubic spline.  note that the PSD is
    # clipped at 1e-300 to prevent nan's in the interpolator (which
    # doesn't seem to like the occasional sample being -inf)
    #

    psd_data = psd.data.data
    psd_data = np.where(psd_data, psd_data, 1e-300)
    f = psd.f0 + np.arange(len(psd_data)) * psd.deltaF
    interp = interpolate.splrep(f, np.log(psd_data), s = 0)
    f = psd.f0 + np.arange(round((len(psd_data) - 1) * psd.deltaF / deltaF) + 1) * deltaF
    psd_data = np.exp(interpolate.splev(f, interp, der = 0))

    #
    # return result
    #

    psd = lal.CreateREAL8FrequencySeries(
        name = psd.name,
        epoch = psd.epoch,
        f0 = psd.f0,
        deltaF = deltaF,
        sampleUnits = psd.sampleUnits,
        length = len(psd_data)
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
        how many previous samples we should account for when calcualting the geometric mean of the psd
    navg: int
        changes to the PSD must occur over a time scale of at least navg*(n/2 − z)*(1/sample_rate)
        *check cody's paper for more info
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
        print(f"Sample rate: {self.sample_rate}")

        # define block overlap following arxiv:1604.04324
        self.n = int(self.fft_length * self.sample_rate)
        self.z = int(self.fft_length / 4 * self.sample_rate)
        overlap = int(self.n / 2 + self.z) 

        # init audio addapter
        self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (0, overlap) 
        self.adapter_config.stride = self.n - overlap
        self.adapter_config.pad_zeros_startup = True

        super().__post_init__()

        # set up for gstlal method:
        if self.whitening_method == "gstlal":
            self.n_samples = 0
            self.delta_f = 1 / (1 / self.sample_rate) / self.n
            self.delta_t = (1 / self.sample_rate)
            print(f"delta f: {self.delta_f}")
            self.window = self.hann_window(int(self.n), int(self.z))
            self.square_data_bufs = deque(maxlen = self.nmed)
            self.prev_data = None
            self.psd_offset = None
            self.lal_normalization_constant = 2 * self.delta_f

            # load ref psd if necessary
            if self.reference_psd:
                psd = lal.series.read_psd_xmldoc(ligolw_utils.load_filename(self.reference_psd, verbose=True, contenthandler=lal.series.PSDContentHandler))
                psd = psd[self.instrument]
                ref_psd_freqs = psd.f0 + np.arange(psd.data.length) * psd.deltaF
                self.psd_geometric_mean = None
                print(f"Reference PSD frequencies: {ref_psd_freqs}")
                print(f"Reference PSD f0: {psd.f0} | epoch: {psd.epoch} | deltaF: {psd.deltaF} | sampleUnits: {psd.sampleUnits} | size: {len(psd.data.data)}")

                #def psd_units_or_resolution_changed(elem, pspec, psd):
                # make sure units are set, compute scale factor
                # FIXME: what is this units?
                #units = lal.Unit(elem.get_property("psd-units"))
                #if units == lal.DimensionlessUnit:
                #    return
                #scale = float(psd.sampleUnits / units)
                scale = 1
                # get frequency resolution and number of bins
                fnyquist = self.sample_rate / 2
                n = int(round(fnyquist / self.delta_f) + 1)
                # interpolate, rescale, and install PSD
                psd = interpolate_psd(psd, self.delta_f)
                ref_psd_data = psd.data.data[:n] * scale
                self.set_psd(ref_psd_data, self.navg) 

            else:
                self.psd_geometric_mean = None

            self.tukey = None
            if self.z:
                # use tukey window
                self.tukey = self.tukey_window(self.n, 2*self.z/self.n)
                
    def tukey_window(self, length, beta):
        """
        1.0 and flat in the middle, cos^2 transition at each end, zero
        at end points, 0.0 <= beta <= 1.0 sets what fraction of the
        window is transition (0 --> rectangle window, 1 --> Hann window)
        """
        if beta < 0 or beta > 1:
            raise ValueError("Invalide value for beta")

        transition_length = round(beta * length)

        n = (transition_length+1)//2

        out = np.ones(length)
        transition =  1/2 * (1 - np.cos(2 * np.pi * np.arange((transition_length + 1) // 2)/ transition_length))
        out[:n] = transition
        out[-n:] = np.flip(transition)
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
        hann[Z:N-Z] = (np.sin(np.pi * (k[Z:N-Z] - Z)/(N - 2*Z)))**2

        # FIXME gstlal had a method of adding from the two ends of the window
        # so that small numbers weren't added to big ones
        self.window_norm = np.sqrt(N / np.sum(hann ** 2))

        return hann

    # XLALMedianBias
    # https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l00378
    def median_bias(self, nn):
        ans = 1
        n = (nn - 1)//2
        for i in range(1,n+1):
            ans -= 1.0/(2*i);
            ans += 1.0/(2*i + 1);
        return ans

    # XLALLogMedianBiasGeometric
    # https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01423
    def log_median_bias_geometric(self, nn):
        return np.log(self.median_bias(nn)) - nn * (loggamma(1/nn) - np.log(nn))

    # XLALPSDRegressorAdd 
    # https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01632
    def add_psd(self, fdata):
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
            log_bin_median = np.log(np.sort(self.square_data_bufs, axis=0)[len(self.square_data_bufs)//2])
            self.geometric_mean_square = (self.geometric_mean_square * (self.n_samples - 1) + log_bin_median - median_bias) / self.n_samples

    # XLALPSDRegressorGetPSD
    # https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01773
    def get_psd(self, fdata):
        # running average mode (track-psd)
        if self.n_samples == 0:
            return self.lal_normalization_constant * (np.abs(fdata) ** 2)
        else:
            return np.exp(self.geometric_mean_square + EULERGAMMA) * self.lal_normalization_constant

    # XLALPSDRegressorSetPSD
    # https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01831
    def set_psd(self, ref_psd_data, weight):
        arithmetic_mean_square_data = ref_psd_data / self.lal_normalization_constant 

        # populate the buffer history with the ref psd
        for i in range(self.nmed):
            self.square_data_bufs.append(arithmetic_mean_square_data)

        self.geometric_mean_square = np.log(arithmetic_mean_square_data) - EULERGAMMA
        self.n_samples = min(weight, self.navg)


    def transform(self, pad):
            """
            Whiten incoming data in segments of fft-length seconds overlapped by fft-length / 4
            Some ascii art here would be helpful to illustrate what were doing.
            """
            # incoming frame handling
            outbufs = []
            frame = self.preparedframes[self.sink_pads[0]]
            EOS = frame.EOS
            outoffsets = self.preparedoutoffsets[self.sink_pads[0]]

            # passes the psd along with an aligned attribute if the pad is the psd_pad
            # if aligned == True, then the current offset == psd offset
            if pad.name == self.psd_pad_name:
                offset = outoffsets[0]["offset"]
                shape = (Offset.tosamples(outoffsets[0]["noffset"], self.sample_rate),)
                if self.psd_offset is None:
                    psd = None
                    aligned = None
                else:
                    psd = self.arithmetic_mean_psd
                    aligned = outoffsets[0]["offset"] == self.psd_offset

                return TSFrame(
                        buffers = [SeriesBuffer(
                                    offset=offset,
                                    sample_rate=self.sample_rate,
                                    data=None,
                                    shape=shape)],
                        EOS = EOS,
                        metadata = {"psd": psd, "aligned": aligned})

            # if audioadapter hasn't given us a frame, then we have to wait for more
            # data before we can whiten. send a gap buffer
            if frame.shape[-1] == 0:
                outbufs.append(
                    SeriesBuffer(
                        offset=outoffsets[0]["offset"],
                        sample_rate=self.sample_rate,
                        data=None,
                        shape=frame.shape,
                    )
                )
            else:
                # copy samples from the deque
                buf = frame.buffers[0]
                this_seg_data = buf.data
                
                if self.whitening_method == "gwpy":
                    # check the type of the timeseries data. 
                    # transform it to a gwpy.timeseries object
                    if not isinstance(this_seg_data, TimeSeries):
                        this_seg_data = TimeSeries(this_seg_data)
        
                    # whiten it
                    whitened_data = this_seg_data.whiten(fftlength=self.fft_length, overlap=0, window="hann")
        
                    # transform back to a numpy array
                    whitened_data = np.array(whitened_data)

                elif self.whitening_method == "gstlal":
                    # apply the window function - in gstlal we have used the
                    # Hann window but we could allow for different window
                    # functions here if we want to
                    # FIXME kinda goes for the entire gstlal whitener: some
                    # things here are different than gstlal - not good. Luckily
                    # it seems to be caused by some missing proportionality constant.
                    this_seg_data = self.window * this_seg_data * 1/self.sample_rate * self.window_norm

                    # apply fourier transform
                    freq_data = np.fft.rfft(this_seg_data)

                    # get frequency bins. FIXME: is this right?
                    timestep = (1 / self.sample_rate)
                    freqs = np.fft.rfftfreq(this_seg_data.size, d=timestep)
                    f0 = freqs[0]

                    # set DC and Nyquist terms to zero
                    freq_data[0] = 0
                    freq_data[self.n//2] = 0

                    # 
                    # PSD
                    #
                    # get the psd
                    this_psd = self.get_psd(freq_data)

                    # push freq data into psd history
                    self.add_psd(freq_data)


                    # now update the "last" geometric mean
                    # offset of most recent psd
                    #self.psd_offset = outoffsets[0]["offset"]
                    #psd_epoch = Offset.tosec(self.psd_offset)
                    #self.psd_geometric_mean = lal.CreateREAL8FrequencySeries("geometric_mean_psd", psd_epoch, f0, self.delta_f, "s strain^2", len(log_psd_geometric_mean))
                    #self.psd_geometric_mean.data.data = np.exp(log_psd_geometric_mean)
                    #for i in range(self.psd_geometric_mean.data.data.shape[-1]):
                    #    print("psd_geometric_mean",i,self.psd_geometric_mean.data.data[i])

                    # whiten by dividing data by the square root of the arithmetic PSD
                    # see arxiv: 1604.04324 (12)
                    #arithmetic_mean_psd = self.psd_geometric_mean.data.data * EULERGAMMA
                    #self.arithmetic_mean_psd = lal.CreateREAL8FrequencySeries("arithmetic_mean_psd", psd_epoch, f0, self.delta_f, "s strain^2", len(arithmetic_mean_psd))
                    #self.arithmetic_mean_psd.data.data = arithmetic_mean_psd
                    #for i in range(self.arithmetic_mean_psd.data.data.shape[-1]):
                    #    print("psd_arithmetic_mean",i,self.arithmetic_mean_psd.data.data[i])

                    #
                    # Whitening
                    #
                    # norm = 2 * self.delta_f in https://lscsoft.docs.ligo.org/lalsuite/lal/_average_spectrum_8c_source.html#l01366
                    # the DC and Nyquist terms are zero
                    freq_data_whitened = np.zeros_like(freq_data)
                    freq_data_whitened[1:-1] = freq_data[1:-1] * np.sqrt(2 * self.delta_f / this_psd[1:-1])

                    # Fourier Transform back to the time domain
                    # # see arxiv: 1604.04324 (13)
                    #
                    # np.fft.irfft default norm is "backward", which uses a normalization factor of 1/n
                    # choose norm="forward" so there is no factor, to match XLALREAL8FreqTimeFFT
                    #
                    # self.delta_f scaling https://lscsoft.docs.ligo.org/lalsuite/lal/_time_freq_f_f_t_8c_source.html#l00183
                    whitened_data = np.fft.irfft(freq_data_whitened, self.n, norm="forward")  * self.delta_f
                    whitened_data *= self.delta_t * np.sqrt(np.sum(self.window ** 2))

                    if self.tukey is not None:
                        whitened_data *= self.tukey

                    # accounts for overlap by summing with prev_data over the
                    # stride of the adapter
                    if self.prev_data is None:
                        self.prev_data = whitened_data[self.adapter_config.stride:]
                    else:
                        whitened_data[:self.adapter_config.overlap[1]] += self.prev_data
                        self.prev_data = whitened_data[self.adapter_config.stride:]
                    
                # only output data up till the length of the adapter stride      
                outbufs.append(
                    SeriesBuffer(
                        offset=outoffsets[0]["offset"],
                        sample_rate=self.sample_rate,
                        data=whitened_data[:self.adapter_config.stride],
                    )
                )

            # return frame with the correct buffers
            return TSFrame(
                            buffers = outbufs,
                            #metadata = metadata,
                            EOS = EOS)

