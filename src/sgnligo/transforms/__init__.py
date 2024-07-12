from sgn.transforms import *
from sgnts.transforms import *
from sgnts.base import (
    SeriesBuffer,
    TSFrame,
    TSTransform,
    AdapterConfig,
)
import os


import numpy as np
from gwpy.timeseries import TimeSeries

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
    n: int
        total number of samples in a single fft
    z: int
        zero padding length
    nmed: int
        how many previous samples we should account for when calcualting the geometric mean of the psd
    navg: int
        changes to the PSD must occur over a time scale of at least navg*(n/2 − z)*(1/sample_rate)
        *check cody's paper for more info
    ref_psd: str
        file path to reference psd
    psd_pad_name: str
        pad name of the psd output source pad
    """

    whitening_method: str = "gwpy"
    sample_rate: int = 2048
    fft_length: int = 8
    nmed: int = 7
    navg: int = 64
    n: int = fft_length * sample_rate
    z: int = int(fft_length / 4. * sample_rate)

    ref_psd: str = ""
    psd_pad_name: str = ""

    psd_buffer = []
    psd_geometric_mean = []
    psd_offset = None
    prev_data = []

    def __post_init__(self):
        if self.adapter_config is None:
            self.adapter_config = AdapterConfig()
        overlap = int(self.n/2 + self.z) 
        self.adapter_config.overlap = (0, overlap) 
        self.adapter_config.stride = self.n - overlap

        super().__post_init__()
        
        self.inputs = {}

        # sanity check the whitening method given
        if self.whitening_method not in ("gwpy", "gstlal"):
            raise ValueError("Unknown whitening method, exiting.")

        # init the hann window
        self.window = self.hann_window()

        
    def hann_window(self):
        """
        Apply hann window to input data
        """
        # number of samples in one fft-length segment
        N = self.n
        Z = self.z
        k = np.arange(0, N, 1) # indices

        hann = np.zeros(N)
        hann[Z:N-Z] = (np.sin(np.pi * (k[Z:N-Z] - Z)/(N - 2*Z)))**2

        #FIXME gstlal had a method of adding from the two ends of the window so that small numbers weren't added to big ones
        self.window_norm = np.sqrt(N / np.sum(hann ** 2))
        return hann

    def transform(self, pad):
            """
            Whiten incoming data in segments of fft-length seconds overlapped by fft-length / 4
            Some ascii art here would be helpful to illustrate what were doing.
            """
            """
            metadata={
                "name": "%s -> %s" % (
                    "+".join(f.metadata["name"] for f in self.inputs.values()),
                    pad.name,
                )
            }
            """
            # incoming frame handling
            outbufs = []
            frame = self.preparedframes[self.sink_pads[0]]
            EOS=any(frame.EOS for frame in self.inputs.values()),
            outoffsets = self.preparedoutoffsets[self.sink_pads[0]]

            # passes the psd along with an aligned attribute if the pad is the psd_pad
            # if aligned == True, then the current offset == psd offset
            if pad.name == self.psd_pad_name:
                if self.psd_offset == None:
                    return TSFrame(
                            buffers = [SeriesBuffer(
                                        offset=outoffsets[0]["offset"],
                                        sample_rate=self.sample_rate,
                                        data=None,
                                        shape=frame.shape)],
                            EOS = EOS,
                            metadata = {"psd": None, "aligned": None})
                else:
                    aligned = (outoffsets[0]["offset"] == self.psd_offset)
                    return TSFrame(
                            buffers = [SeriesBuffer(
                                        offset=outoffsets[0]["offset"],
                                        sample_rate=self.sample_rate,
                                        data=None,
                                        shape=frame.shape)],
                            EOS = EOS,
                            metadata = {"psd": self.psd_geometric_mean, "aligned": aligned})

            # used for plotting purposes
            offset_start = frame.buffers[0].offset
            offset_end = frame.buffers[0].offset + frame.buffers[0].noffset

            # if audioadapter hasn't given us a frame, then waiting for more data
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
                
                #np.savetxt(f"seg_data/seg_data_{offset_start}-{offset_end}.txt", this_seg_data)

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
                    # apply the window function - in gstlal we have used the Hann window
                    # but we could allow for different window functions here if we want to

                    # FIXME kinda goes for the entire gstlal whitener: some things here are different than gstlal - not good
                    #       luckily it seems to be caused by some missing proportionality constant.
                    this_seg_data = self.window * this_seg_data * self.window_norm #* 1/self.sample_rate

                    # apply fourier transform
                    freq_data = np.fft.rfft(this_seg_data)
                    # inst. PSD is proportional to the sq. magnitude

                    # FIXME gstlal code multiply their psd by a 2*delta-f factor, but unsure where to do that
                    psd_inst = 2 * self.sample_rate/self.n * (np.abs(freq_data) ** 2) 
 
                    # compute median of PSDs, geometric mean, and arithmetic mean of PSDs 
                    # keep track of last nmed psd_insts
                    self.psd_buffer.append(psd_inst)
                    if len(self.psd_buffer) > self.nmed:
                        self.psd_buffer.pop(0)

                    # calculate median of psd_buffer
                    psd_median = np.median(self.psd_buffer, axis=0)

                    # create geometric mean template
                    if self.psd_geometric_mean == []:
                        self.psd_geometric_mean = np.ones_like(psd_median)

                    # load ref psd if necessary
                    if self.ref_psd != "":
                        self.psd_geometric_mean = np.loadtxt(self.ref_psd)
                        self.ref_psd = ""

                    # calculate geometric mean
                    # FIXME this beta value appears in cody's paper, but idk what it is - and i havent found it in the gstlal code either 
                    beta = 1  # beta value
                    log_psd_median_adjusted = np.log(psd_median / beta)
                    log_psd_geometric_mean = (self.navg - 1) / self.navg * np.log(self.psd_geometric_mean) + 1 / self.navg * log_psd_median_adjusted
                    self.psd_geometric_mean = np.exp(log_psd_geometric_mean)

                    # Estimate the arithmetic mean of the PSD
                    euler_gamma = np.exp(0.57721566490153286060)  # Euler-Mascheroni constant
                    psd_arithmetic_mean = self.psd_geometric_mean * euler_gamma

                    # whiten by dividing data by the square root of the PSD
                    # in each frequency bin
                    freq_data_whitened = freq_data / np.sqrt(psd_arithmetic_mean)

                    # Fourier Transform back to the time domain
                    whitened_data = np.fft.irfft(freq_data_whitened, self.n)
                    #np.savetxt(f"seg_data/whitened_{offset_start}-{offset_end}.txt", whitened_data)
                    
                    whitened_data *= (1/self.sample_rate) * np.sqrt(np.sum(self.window ** 2))

                    # accounts for overlap by summing with prev_data over the stride of the adapter
                    if self.prev_data == []:
                        self.prev_data = whitened_data[self.adapter_config.stride:]
                    else:
                        whitened_data[:self.adapter_config.overlap[1]] += self.prev_data
                        self.prev_data = whitened_data[self.adapter_config.stride:]
                    
                    """
                    if EOS:
                        np.savetxt(f"seg_data/psd_geo.txt", self.psd_geometric_mean)
                    """
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
