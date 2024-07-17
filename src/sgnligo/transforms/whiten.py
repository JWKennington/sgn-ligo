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

EULERGAMMA = np.exp(float(EulerGamma.evalf()))
print(EULERGAMMA)

@dataclass
class Whiten(TSTransform):
    """
    Whiten input timeseries data
    Parameters:
    -----------
    whitening-method: str
        currently supported types: (1) 'gwpy', (2) 'gstlal'
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
        overlap = int(self.n / 2 + self.z) 

        # init audio addapter
        self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (0, overlap) 
        self.adapter_config.stride = self.n - overlap

        super().__post_init__()

        # set up for gstlal method:
        if self.whitening_method == "gstlal":
            self.delta_f = 1 / (1 / self.sample_rate) / self.n
            self.window = self.hann_window(int(self.n), int(self.z))
            self.psd_buffer = deque(maxlen = self.nmed)
            self.prev_data = None
            self.psd_offset = None

            # load ref psd if necessary
            if self.reference_psd:
                psd = lal.series.read_psd_xmldoc(ligolw_utils.load_filename(self.reference_psd, verbose=True, contenthandler=lal.series.PSDContentHandler))

                psd_data = psd[self.instrument].data.data
                self.psd_geometric_mean = psd_data
            else:
                self.psd_geometric_mean = None


        
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
        window_norm = np.sqrt(N / np.sum(hann ** 2))

        return hann * window_norm

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
                    psd = self.psd_geometric_mean
                    aligned = outoffsets[0]["offset"] == self.psd_offset

                return TSFrame(
                        buffers = [SeriesBuffer(
                                    offset=offset,
                                    sample_rate=self.sample_rate,
                                    data=None,
                                    shape=shape)],
                        EOS = EOS,
                        metadata = {"psd": psd, "aligned": aligned})

            # FIXME: delete
            # used for plotting purposes
            offset_start = frame.buffers[0].offset
            offset_end = frame.buffers[0].offset + frame.buffers[0].noffset

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
                frame = frame.buffers[0]
                this_seg_data = frame.data
                
                # FIXME: delete
                np.savetxt(f"seg_data/seg_data_{offset_start}-{offset_end}.txt", this_seg_data)

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
                    this_seg_data = self.window * this_seg_data # * 1/self.sample_rate

                    # apply fourier transform
                    freq_data = np.fft.rfft(this_seg_data)

                    # inst. PSD is proportional to the sq. magnitude
                    # see arxiv: 1604.04324 (10)
                    psd_inst = 2 * self.delta_f * (np.abs(freq_data) ** 2) 
 
                    # keep track of last nmed instantaneous PSDs
                    self.psd_buffer.append(psd_inst)

                    # compute median of PSDs, geometric mean, and arithmetic mean of PSDs 
                    # calculate median over the last nmed instantaneous PSDs
                    psd_median = np.median(self.psd_buffer, axis=0)

                    # calculate new geometric mean
                    # FIXME this beta value appears in arxiv: 1604.04324, but idk what
                    # it is - and I havent found it in the gstlal code either
                    # see arxiv: 1604.04324 (11)
                    beta = 1  # beta value
                    log_psd_median_adjusted = np.log(psd_median / beta)
                    if self.psd_geometric_mean is not None:
                        log_psd_geometric_mean = (self.navg - 1) / self.navg * np.log(self.psd_geometric_mean) + 1 / self.navg * log_psd_median_adjusted
                    else:
                        # start up condition when we're not using a reference psd
                        log_psd_geometric_mean = log_psd_median_adjusted

                    # now update the "last" geometric mean
                    self.psd_geometric_mean = np.exp(log_psd_geometric_mean)

                    # whiten by dividing data by the square root of the arithmetic PSD
                    # see arxiv: 1604.04324 (12)
                    freq_data_whitened = freq_data / np.sqrt(self.psd_geometric_mean * EULERGAMMA)

                    # Fourier Transform back to the time domain
                    # # see arxiv: 1604.04324 (13)
                    whitened_data = np.fft.irfft(freq_data_whitened, self.n)
                    whitened_data *= 2 * self.delta_f * (1 / self.sample_rate) * np.sqrt(np.sum(self.window ** 2))

                    # accounts for overlap by summing with prev_data over the
                    # stride of the adapter
                    if not self.prev_data:
                        self.prev_data = whitened_data[self.adapter_config.stride:]
                    else:
                        whitened_data[:self.adapter_config.overlap[1]] += self.prev_data
                        self.prev_data = whitened_data[self.adapter_config.stride:]
                    
                    # offset of most recent psd
                    self.psd_offset = outoffsets[0]["offset"]

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

