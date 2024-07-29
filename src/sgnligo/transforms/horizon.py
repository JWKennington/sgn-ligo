import lal
from lal import LIGOTimeGPS
import lalsimulation

import math

from sgn.transforms import *
from sgnts.transforms import *
from sgnts.base import (
    SeriesBuffer,
    TSFrame,
    TSTransform,
    AdapterConfig,
)

import numpy as np

@dataclass
class HorizonDistance(TSTransform):
    """
    Compute horizon distance for an incoming PSD and a given waveform model
    """

    rate: int = 1
    fmin: int = None
    fmax: int = None
    delta_f: int = None
    m1: int = None
    m2: int = None
    spin1: tuple = (0., 0., 0.)
    spin2: tuple = (0., 0., 0.)
    eccentricity: int = 0
    inclination: int = 0
    approximant: str = "IMRPhenomD"
    range: bool = False
    snr: int = 8

    def __post_init__(self):
        super().__post_init__()

        # init waveform
        # NOTE:  the waveform models are computed up-to but not
        # including the supplied fmax parameter so we need to pass
        # (fmax + delta_f) if we want the waveform model defined
        # in the fmax bin
        hp, hc = lalsimulation.SimInspiralFD(
            self.m1 * lal.MSUN_SI, self.m2 * lal.MSUN_SI,
            self.spin1[0], self.spin1[1], self.spin1[2],
            self.spin2[0], self.spin2[1], self.spin2[2],
            1.0,    # distance (m)
            self.inclination,
            0.0,    # reference orbital phase (rad)
            0.0,    # longitude of ascending nodes (rad)
            self.eccentricity,
            0.0,    # mean anomaly of periastron
            self.delta_f,
            self.fmin,
            self.fmax + self.delta_f,
            100.,    # reference frequency (Hz)
            None,    # LAL dictionary containing accessory parameters
            lalsimulation.GetApproximantFromString(self.approximant)
        )
        assert hp.data.length > 0, "huh!?  h+ has zero length!"

        # store |h(f)|^2 for source at D = 1 m.  see (5) in
        # arXiv:1003.2481
        self.model = lal.CreateREAL8FrequencySeries(
            name = "signal spectrum",
            epoch = LIGOTimeGPS(0),
            f0 = hp.f0,
            deltaF = hp.deltaF,
            sampleUnits = hp.sampleUnits * hp.sampleUnits,
            length = hp.data.length
        )
        self.model.data.data[:] = np.abs(hp.data.data)**2.

    def compute_horizon(self, psd):
        """
        compute the horizon distance in Mpc
        """
        # frequencies at which PSD has been measured
        f = psd.f0 + np.arange(psd.data.length) * psd.deltaF

        # nearest-neighbour interpolation of waveform model
        # evaluated at PSD's frequency bins
        indexes = ((f - self.model.f0) / self.model.deltaF).round().astype("int").clip(0, self.model.data.length - 1)
        model = self.model.data.data[indexes]

        # range of indexes for integration
        kmin = (max(psd.f0, self.model.f0, self.fmin) - psd.f0) / psd.deltaF
        kmin = int(round(kmin))
        kmax = (min(psd.f0 + psd.data.length * psd.deltaF, self.model.f0 + self.model.data.length * self.model.deltaF, self.fmax) - psd.f0) / psd.deltaF
        kmax = int(round(kmax)) + 1
        assert kmin < kmax, "PSD and waveform model do not intersect"

        # SNR for source at D = 1 m <--> D in m for source w/ SNR =
        # 1.  see (3) in arXiv:1003.2481
        f = f[kmin:kmax]
        model = model[kmin:kmax]
        D = math.sqrt(4. * (model / psd.data.data[kmin:kmax]).sum() * psd.deltaF)

        # distance at desired SNR
        D /= self.snr

        # scale inspiral spectrum by distance to achieve desired SNR
        model *= 4. / D**2.

        # D in Mpc for source with specified SNR, and waveform model
        D /= (1e6 * lal.PC_SI)

        return D
         

    def transform(self, pad):
        """
        compute horizon distance
        """
        # incoming frame handling
        outbufs = []
        frame = self.preparedframes[self.sink_pads[0]]
        EOS = frame.EOS
        metadata = frame.metadata
        shape = frame.shape
        offset = frame.offset
        metadata = frame.metadata

        # get spectrum from metadata
        # FIXME: this is a hack since the PSD is a frequency series.
        psd = metadata["psd"]
        if psd is not None:
            assert isinstance(psd, lal.REAL8FrequencySeries)

            dist = self.compute_horizon(psd)
        else:
            dist = None

        # send buffer with no data, put horizon history in metadata
        outbuf = SeriesBuffer(
            offset=offset, sample_rate=frame.sample_rate, data=None, shape=shape
        )
        metadata["psd_name"] = "'%s'" % pad.name
        metadata["horizon"] =  dist
        if self.range and dist is not None:
            metadata["range"] = dist/2.25

        return TSFrame(
            buffers=[outbuf],
            metadata=metadata,
            EOS=EOS,
        )
