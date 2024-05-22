from sgn.transforms import *
from sgnts.transforms import *
from sgnts.base.buffer import *

import numpy as np
from gwpy.timeseries import TimeSeries

@dataclass
class Whiten(TransformElement):
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
    """

    whitening_method: str = "gwpy"
    sample_rate: int = 2048
    fft_length: int = 8

    def __post_init__(self):
        super().__post_init__()
        self.inputs = {}

        # sanity check the whitening method given
        if self.whitening_method not in ("gwpy", "gstlal"):
            raise ValueError("Unknown whitening method, exiting.")

        # set up a deque to store incoming data with audioadapter
        self.audioadapters = {}

        # track start offset of segment to whiten
        self.seg_start = None

        # init the hann window
        self.window = self.hann_window()


    def pull(self, pad, frame):
        self.inputs[pad] = frame
        # initialize audio adapter and push new data to deque
        if pad not in self.audioadapters:
            self.audioadapters[pad] = Audioadapter()
        for buf in frame:
            self.audioadapters[pad].push(buf)


        # get offset segment
        self.this_segment = (frame[0].offset, frame[-1].offset + frame[-1].noffset)
        self.this_noffset = frame[-1].noffset + frame[-1].offset - frame[0].offset

        # the first segment will start at the first buffer
        if self.seg_start is None:
            print(f"setting seg start to: {self.this_segment[0]}")
            self.seg_start = self.this_segment[0]

        print(f"{pad} this segment: {self.this_segment}")

    def hann_window(self):
        """
        Apply hann window to input data
        """
        # number of samples in one fft-length segment
        N = int(self.fft_length * self.sample_rate)
        Z = int((self.fft_length / 4.) * self.sample_rate)
        k = np.arange(0, N, 1) # indices

        hann = np.zeros(N)
        hann[Z:N-Z] = (np.sin(np.pi * (k[Z:N-Z] - Z)/(N - 2*Z)))**2
        return hann

    def transform(self, pad):
            """
            Whiten incoming data in segments of fft-length seconds overlapped by fft-length / 4
            Some ascii art here would be helpful to illustrate what were doing.
            """
            metadata={
                "name": "%s -> %s" % (
                    "+".join(f.metadata["name"] for f in self.inputs.values()),
                    pad.name,
                )
            }
            EOS=any(frame.EOS for frame in self.inputs.values()),
        
            frame = self.inputs[self.sink_pads[0]]
            frame = frame[-1]
            data = frame.data

            # number of offsets in one segment to whiten
            self.num_fft_offsets = int(self.fft_length * self.this_noffset)

            # check the length of data in the deque
            offset_start, offset_end = self.audioadapters[self.sink_pads[0]].get_available_offset_segment()
            data_length = offset_end - offset_start
            if data_length > self.num_fft_offsets:
                # flush one buffer of data from the left of the deque
                offset_to_flush = offset_start + self.this_noffset
                self.audioadapters[self.sink_pads[0]].flush_samples_by_end_offset_segment(offset_to_flush)
            elif data_length < self.num_fft_offsets:
                # we are  waiting for enough data at start up
                return TSFrame(
                                buffers = [SeriesBuffer(offset = frame.offset, noffset = 0, data = None, is_gap = True, offset_ref_t0 = frame.offset_ref_t0)],
                                metadata = metadata,                                
                                EOS = EOS)

            offset_start, offset_end = self.audioadapters[self.sink_pads[0]].get_available_offset_segment()

            # check if it is time to compute the next inst. PSD
            # ie have we shifted our data deque by fft-length / 4 since the last inst. PSD?
            if offset_start >= self.seg_start:
                # copy samples from the deque
                this_seg_data, _, _ = self.audioadapters[self.sink_pads[0]].copy_samples_by_offset_segment((offset_start, offset_end))

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

                    #np.savetxt(f"seg_data/whitened_{offset_start}-{offset_end}.txt", whitened_data)

                elif self.whitening_method == "gstlal":
                    # apply the window function - in gstlal we have used the Hann window
                    # but we could allow for different window functions here if we want to
                    N = len(this_seg_data)
                    #np.savetxt(f"seg_data/data_{offset_start}-{offset_end}.txt", this_seg_data)
                    this_seg_data = self.window * this_seg_data

                    # apply fourier transform
                    freq_data = np.fft.rfft(this_seg_data)
                    #np.savetxt(f"seg_data/ft_{offset_start}-{offset_end}.txt", freq_data)
                    # inst. PSD is proportional to the sq. magnitude
                    psd_inst = (2/(N*(1/self.sample_rate))) * (np.abs(freq_data) ** 2)
                    
                    #np.savetxt(f"seg_data/psd_inst_{offset_start}-{offset_end}.txt", psd_inst)

                    # compute median of PSDs, geometric mean, and arithmetic mean of PSDs
                    if not hasattr(self, 'psd_buffer'):
                        self.psd_buffer = []

                    nmed = 7
                    navg = 64

                    self.psd_buffer.append(psd_inst)
                    if len(self.psd_buffer) > nmed:
                        self.psd_buffer.pop(0)

                    psd_median = np.median(self.psd_buffer, axis=0)
                    #np.savetxt(f"seg_data/psd_med_{offset_start}-{offset_end}.txt", this_seg_data)

                    # Calculate the geometric mean
                    if not hasattr(self, 'psd_geometric_mean'):
                        self.psd_geometric_mean = np.ones_like(psd_median)

                    beta = 1  # beta value
                    log_psd_median_adjusted = np.log(psd_median / beta)
                    log_psd_geometric_mean = (navg - 1) / navg * np.log(self.psd_geometric_mean) + 1 / navg * log_psd_median_adjusted
                    self.psd_geometric_mean = np.exp(log_psd_geometric_mean)

                    #np.savetxt(f"seg_data/psd_geo_{offset_start}-{offset_end}.txt", self.psd_geometric_mean)

                    # Estimate the arithmetic mean of the PSD
                    euler_gamma = np.exp(0.57721566490153286060)  # Euler-Mascheroni constant
                    psd_arithmetic_mean = self.psd_geometric_mean * euler_gamma

                    # whiten by dividing data by the square root of the PSD
                    # in each frequency bin
                    freq_data_whitened = freq_data / np.sqrt(psd_arithmetic_mean)
                    
                    #np.savetxt(f"seg_data/freq_whitened_{offset_start}-{offset_end}.txt", freq_data_whitened)

                    # Fourier Transform back to the time domain
                    whitened_data = np.fft.irfft(freq_data_whitened, N)

                    #np.savetxt(f"seg_data/whitened_{offset_start}-{offset_end}.txt", whitened_data)
                # shift the next segment start by the overlap
                self.seg_start += int(self.num_fft_offsets / 4)
                print(f"setting seg start to: {self.seg_start}")

                return TSFrame(
                                buffers = [SeriesBuffer(offset = offset_start, noffset = offset_end - offset_start, data = whitened_data, offset_ref_t0 = frame.offset_ref_t0)],
                                metadata = metadata,
                                EOS = EOS)

            else:
                # its not time for the next segment yet
                return TSFrame(
                                buffers = [SeriesBuffer(offset = frame.offset, noffset = 0, data = None, is_gap=True, offset_ref_t0 = frame.offset_ref_t0)],
                                metadata = metadata,
                                EOS = EOS)
