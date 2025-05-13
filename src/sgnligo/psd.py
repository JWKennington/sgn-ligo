"""A module for reading, writing, and measuring PSDs."""

# Copyright (C) 2010-2013  Kipp Cannon, Chad Hanna, Leo Singer
# Copyright (C) 2024 Anushka Doke, Yun-Jing Huang, Ryan Magee, Shio Sakon

from __future__ import annotations

import math
import signal
import sys
from typing import Dict, Iterable, Optional, Tuple, Union

import lal
import lal.series
import lalsimulation
import numpy
import pandas
from lal import LIGOTimeGPS
from ligo.lw import utils as ligolw_utils
from scipy import interpolate

#
# =============================================================================
#
#                               PSD Measurement
#
# =============================================================================
#


# The measure_psd function has been moved to sgnl.measure_psd


def read_psd(
    filename: str, verbose: Optional[bool] = False
) -> Dict[str, lal.REAL8FrequencySeries]:
    """Reads in an XML-formatted PSD.

    Args:
        filename:
            str, the file to read PSD(s) from
        verbose:
            bool, default False, whether to display logging messages

    Returns:
        a dictionary of lal FrequencySeries keyed by instrument

    """
    return lal.series.read_psd_xmldoc(
        ligolw_utils.load_filename(
            filename, verbose=verbose, contenthandler=lal.series.PSDContentHandler
        )
    )


def write_psd(
    filename: str,
    psddict: Dict[str, lal.REAL8FrequencySeries],
    trap_signals: Optional[Iterable[signal.Signals]] = None,
    verbose: Optional[bool] = False,
) -> None:
    """Writes an XML-formatted PSD to disk.

    Wrapper around make_psd_xmldoc() to write the XML document directly
    to a named file.

    Args:
        filename:
            str, the file to write PSD(s) to
        psds:
            Dict[str, lal.REAL8FrequencySeries], the PSD(s)
        trap_signals:
            Iterable[signal.Signal], optional, whether to attach extra signal
            handlers on write
        verbose:
            bool, default False, whether to display logging messages

    """
    ligolw_utils.write_filename(
        lal.series.make_psd_xmldoc(psddict),
        filename,
        verbose=verbose,
        trap_signals=trap_signals,
    )


def read_asd_txt(
    filename: str,
    df: float = 0.25,
    zero_pad: bool = False,
    read_as_psd: bool = False,
) -> lal.REAL8FrequencySeries:
    """Reads in a text-formatted ASD as a PSD.

    Args:
        filename:
            str, the file to read ASD(s) from
        df (Hz):
            float, default 0.25, the frequency resolution to interpolate to
        zero_pad:
            bool, default False, whether to zero-pad PSD to 0 Hz if needed.
        read_as_psd:
            bool, default False, whether to treat input as PSD rather than ASD

    Returns:
        lal.REAL8FrequencySeries, the PSD

    """
    data = numpy.loadtxt(filename, comments="#")
    psd_data = data[:, 1]
    if not read_as_psd:
        psd_data = numpy.power(psd_data, 2)

    f = data[:, 0]
    if zero_pad:
        f_pad = numpy.arange(0, f[0], df)
        psd_data = numpy.concatenate((numpy.ones(len(f_pad)) * psd_data[0], psd_data))
        f = numpy.concatenate((f_pad, f))

    uniformf = numpy.arange(f[0], f.max(), df)
    psdinterp = interpolate.interp1d(f, psd_data)
    psd_data = psdinterp(uniformf)

    psd = lal.CreateREAL8FrequencySeries(
        name="PSD",
        epoch=0,
        f0=f[0],
        deltaF=df,
        sampleUnits=lal.Unit("s strain^2"),
        length=len(psd_data),
    )
    psd.data.data = psd_data

    return psd


def write_asd_txt(
    filename: str,
    psd: lal.REAL8FrequencySeries,
    verbose: Optional[bool] = False,
) -> None:
    """Writes an text-formatted ASD to disk.

    Args:
        filename:
            str, the file to write ASD to
        psd:
            lal.REAL8FrequencySeries, the PSD
        verbose:
            bool, default False, whether to display logging messages

    """
    if verbose:
        print("writing '%s' ..." % filename, file=sys.stderr)
    with open(filename, "w") as f:
        for i, x in enumerate(psd.data.data):
            print("%.16g %.16g" % (psd.f0 + i * psd.deltaF, x**0.5), file=f)


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
    #
    # no-op?
    #

    if deltaF == psd.deltaF:
        return psd

    #
    # interpolate PSD by clipping/zero-padding time-domain impulse
    # response of equivalent whitening filter
    #

    # from scipy import fftpack
    # psd_data = psd.data.data
    # x = numpy.zeros((len(psd_data) * 2 - 2,), dtype = "double")
    # psd_data = numpy.where(psd_data, psd_data, float("inf"))
    # x[0] = 1 / psd_data[0]**.5
    # x[1::2] = 1 / psd_data[1:]**.5
    # x = fftpack.irfft(x)
    # if deltaF < psd.deltaF:
    #    x *= numpy.cos(numpy.arange(len(x)) * math.pi / (len(x) + 1))**2
    #    x = numpy.concatenate((x[:(len(x) / 2)], numpy.zeros((int(round(len(x)
    #        * psd.deltaF / deltaF)) - len(x),), dtype = "double"), x[(len(x) / 2):]))
    # else:
    #    x = numpy.concatenate((x[:(int(round(len(x) * psd.deltaF / deltaF)) / 2)],
    #        x[-(int(round(len(x) * psd.deltaF / deltaF)) / 2):]))
    #    x *= numpy.cos(numpy.arange(len(x)) * math.pi / (len(x) + 1))**2
    # x = 1 / fftpack.rfft(x)**2
    # psd_data = numpy.concatenate(([x[0]], x[1::2]))

    #
    # interpolate log(PSD) with cubic spline.  note that the PSD is
    # clipped at 1e-300 to prevent nan's in the interpolator (which
    # doesn't seem to like the occasional sample being -inf)
    #

    psd_data = psd.data.data
    psd_data = numpy.where(psd_data, psd_data, 1e-300)
    f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF
    interp = interpolate.splrep(f, numpy.log(psd_data), s=0)
    f = (
        psd.f0
        + numpy.arange(round((len(psd_data) - 1) * psd.deltaF / deltaF) + 1) * deltaF
    )
    psd_data = numpy.exp(interpolate.splev(f, interp, der=0))

    #
    # return result
    #

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


def movingmedian(
    psd: Union[numpy.ndarray, lal.REAL8FrequencySeries], window_size: int
) -> Union[numpy.ndarray, lal.REAL8FrequencySeries]:
    """Smoothen a PSD with a moving median.

    Assumes that the underlying PSD doesn't have variance, i.e., that there
    is no median / mean correction factor required.

    Args:
        psd:
            Union[numpy.ndarray, lal.REAL8FrequencySeries], the PSD to smoothen
        window_size:
            int, the size of the window used for the moving median

    Returns:
        a smoothened PSD of same type as input PSD

    """
    if isinstance(psd, numpy.ndarray):
        tmp = numpy.copy(psd)
    else:  # lal Series
        tmp = numpy.copy(psd.data.data)

    # compute rolling median
    tmp[window_size : (len(tmp) - window_size)] = numpy.array(
        pandas.Series(tmp).rolling(2 * window_size).median()[(2 * window_size - 1) : -1]
    )

    if isinstance(psd, numpy.ndarray):
        return tmp
    else:  # lal Series
        new_psd = lal.CreateREAL8FrequencySeries(
            name=psd.name,
            epoch=psd.epoch,
            f0=psd.f0,
            deltaF=psd.deltaF,
            sampleUnits=psd.sampleUnits,
            length=len(tmp),
        )
        psd.data.data = tmp
        return new_psd


def movingaverage(psd: numpy.ndarray, window_size: int) -> numpy.ndarray:
    """Smoothen a PSD with a moving average.

    Args:
        psd:
            numpy.ndarray, the PSD to smoothen
        window_size:
            int, the size of the window used for the moving median

    Returns:
        the smoothened PSD

    """
    window = lal.CreateTukeyREAL8Window(window_size, 0.5).data.data
    return numpy.convolve(psd, window, "same")


def taperzero_fseries(
    fseries: lal.REAL8FrequencySeries,
    minfs: Optional[Tuple[float, float]] = (35.0, 40.0),
    maxfs: Optional[Tuple[float, float]] = (1800.0, 2048.0),
) -> lal.REAL8FrequencySeries:
    """Taper the PSD to infinity for given min/max frequencies.

    Args:
        psd:
            lal.REAL8FrequencySeries, the PSD to taper
        minfs (Hz):
            Tuple[float, float], optional, the frequency boundaries over which to taper
            the spectrum to infinity.  i.e., frequencies below the first item in the
            tuple will have an infinite spectrum, the second item in the tuple will not
            be changed. A taper from 0 to infinity is applied in between.
        maxfs (Hz):
            Tuple[float, float], optional, the frequency boundaries over which to taper
            the spectrum to infinity.  i.e., frequencies above the second item in the
            tuple will have an infinite spectrum, the first item in the tuple will not
            be changed. A taper from 0 to infinity is applied in between.

    Returns:
        lal.REAL8FrequencySeries, the tapered PSD

    """

    #
    # store the psd horizon before tapering
    #

    data = fseries.data.data
    norm_before = numpy.dot(data.conj(), data).real

    #
    # taper to infinity to turn this psd into an effective band pass filter
    #

    deltaF = fseries.deltaF
    kmin = int(minfs[0] / deltaF)
    kmax = int(minfs[1] / deltaF)
    data[(len(data) // 2 + 1) - kmin + 1 : (len(data) // 2 + 1) + kmin] = 0.0
    data[(len(data) // 2 + 1) + kmin : (len(data) // 2 + 1) + kmax] *= (
        numpy.sin(numpy.arange(kmax - kmin) / (kmax - kmin - 1.0) * numpy.pi / 2.0) ** 4
    )
    data[(len(data) // 2 + 1) - kmax : (len(data) // 2 + 1) - kmin] *= (
        numpy.cos(numpy.arange(kmax - kmin) / (kmax - kmin - 1.0) * numpy.pi / 2.0) ** 4
    )

    kmin = int(maxfs[0] / deltaF) - 1
    kmax = int(maxfs[1] / deltaF) - 1
    data[(len(data) // 2 + 1) + kmax :] = data[: -(len(data) // 2 + 1) - kmax] = 0.0
    data[(len(data) // 2 + 1) + kmin : (len(data) // 2 + 1) + kmax] *= (
        numpy.cos(numpy.arange(kmax - kmin) / (kmax - kmin - 1.0) * numpy.pi / 2.0) ** 4
    )
    data[(len(data) // 2 + 1) - kmax : (len(data) // 2 + 1) - kmin] *= (
        numpy.sin(numpy.arange(kmax - kmin) / (kmax - kmin - 1.0) * numpy.pi / 2.0) ** 4
    )

    #
    # renormalize after tapering
    #

    fseries.data.data = data * math.sqrt(
        norm_before / numpy.dot(data.conj(), data).real
    )

    #
    # done
    #

    return fseries


def condition_psd(
    psd: lal.REAL8FrequencySeries,
    newdeltaF: int,
    minfs: Optional[Tuple[int, int]] = (35.0, 40.0),
    maxfs: Optional[Tuple[int, int]] = (1800.0, 2048.0),
    smoothing_frequency: Optional[float] = 4.0,
    fir_whiten: Optional[bool] = False,
) -> lal.REAL8FrequencySeries:
    """Condition a PSD suitable for whitening waveforms.

    Args:
        psd:
            lal.REAL8FrequencySeries, the PSD to taper
        newdeltaF (Hz):
            int, the target delta F to interpolate to
        minfs (Hz):
            Tuple[float, float], optional, the frequency boundaries over which to taper
            the spectrum to infinity.  i.e., frequencies below the first item in the
            tuple will have an infinite spectrum, the second item in the tuple will not
            be changed. A taper from 0 to infinity is applied in between.
        maxfs (Hz):
            Tuple[float, float], optional, the frequency boundaries over which to taper
            the spectrum to infinity.  i.e., frequencies above the second item in the
            tuple will have an infinite spectrum, the first item in the tuple will not
            be changed. A taper from 0 to infinity is applied in between.
        smoothing_frequency (Hz):
            float, default = 4 Hz, the target frequency resolution after smoothing.
            Lines with bandwidths << smoothing_frequency are removed via a median
            calculation.  Remaining features will be blurred out to this resolution.
        fir_whiten:
            bool, default False, whether to enable causal whitening with a time-domain
            whitening kernel vs. traditional acausal whitening

    Returns:
        lal.REAL8FrequencySeries, the conditioned PSD

    """

    #
    # store the psd horizon before conditioning
    #

    horizon_distance = HorizonDistance(minfs[1], maxfs[0], psd.deltaF, 1.4, 1.4)
    horizon_before = horizon_distance(psd, 8.0)[0]

    #
    # interpolate to new \Delta f
    #

    psd = interpolate_psd(psd, newdeltaF)

    #
    # Smooth the psd
    #

    psddata = psd.data.data
    avgwindow = int(smoothing_frequency / newdeltaF)
    psddata = movingmedian(psddata, avgwindow)
    psddata = movingaverage(psddata, avgwindow)
    psd.data.data = psddata

    #
    # Tapering psd in either side up to infinity if a frequency-domain whitener is used,
    # returns a psd without tapering otherwise. For a time-domain whitener, the tapering
    # is effectively done as a part of deriving a frequency series of the FIR-whitner
    # kernel
    #
    if not fir_whiten:
        #
        # Taper to infinity to turn this psd into an effective band pass filter
        #

        psddata = psd.data.data
        kmin = int(minfs[0] / newdeltaF)
        kmax = int(minfs[1] / newdeltaF)
        psddata[: kmin + 1] = numpy.inf
        psddata[kmin:kmax] /= (
            numpy.sin(numpy.arange(kmax - kmin) / (kmax - kmin - 1.0) * numpy.pi / 2.0)
            ** 4
        )

        kmin = int(maxfs[0] / newdeltaF)
        kmax = int(maxfs[1] / newdeltaF)
        psddata[kmax:] = numpy.inf
        psddata[kmin:kmax] /= (
            numpy.cos(numpy.arange(kmax - kmin) / (kmax - kmin - 1.0) * numpy.pi / 2.0)
            ** 4
        )

        psd.data.data = psddata

    #
    # compute the psd horizon after conditioning and renormalize
    #

    horizon_after = horizon_distance(psd, 8.0)[0]

    psddata = psd.data.data
    psd.data.data = psddata * (horizon_after / horizon_before) ** 2

    #
    # done
    #

    return psd


def polyfit(
    psd: lal.REAL8FrequencySeries,
    f_low: float,
    f_high: float,
    order: int,
    verbose: Optional[bool] = False,
) -> lal.REAL8FrequencySeries:
    """Fit a PSD to a polynomial.

    Args:
        psd:
            lal.REAL8FrequencySeries, the PSD to fit
        f_low (Hz):
            float, the low frequency to begin fitting with
        f_high (Hz):
            float, the high frequency to stop fitting with
        order:
            int, the order of the fitting polynomial
        verbose:
            bool, default false, whether to display the fit

    Returns:
        lal.REAL8FrequencySeries, the PSD fitted to a polynomial

    """
    minsample = int(f_low // psd.deltaF)
    maxsample = int(f_high // psd.deltaF)

    # f / f_min between f_min and f_max, i.e. f[0] here is 1
    f = numpy.arange(maxsample - minsample) * psd.deltaF + 1
    data = psd.data.data[minsample:maxsample]

    logf = numpy.linspace(numpy.log(f[0]), numpy.log(f[-1]), 100000)
    interp = interpolate.interp1d(numpy.log(f), numpy.log(data))
    data = interp(logf)
    p = numpy.poly1d(numpy.polyfit(logf, data, order))
    if verbose:
        print(
            "\nFit polynomial is: \n\nlog(PSD) = \n",
            p,
            "\n\nwhere x = f / f_min\n",
            file=sys.stderr,
        )
    data = numpy.exp(p(numpy.log(f)))
    olddata = psd.data.data
    olddata[minsample:maxsample] = data
    psd = lal.CreateREAL8FrequencySeries(
        name=psd.name,
        epoch=psd.epoch,
        f0=psd.f0,
        deltaF=psd.deltaF,
        sampleUnits=psd.sampleUnits,
        length=len(olddata),
    )
    psd.data.data = olddata
    return psd


def harmonic_mean(
    psddict: Dict[str, lal.REAL8FrequencySeries],
) -> lal.REAL8FrequencySeries:
    """Take the harmonic mean of a dictionary of PSDs."""
    refpsd = list(psddict.values())[0]
    psd = lal.CreateREAL8FrequencySeries(
        "psd",
        refpsd.epoch,
        0.0,
        refpsd.deltaF,
        lal.Unit("strain^2 s"),
        refpsd.data.length,
    )
    psd.data.data[:] = 0.0
    for ifo in psddict:
        psd.data.data[:] += 1.0 / psddict[ifo].data.data
    psd.data.data[:] = len(psddict) / psd.data.data[:]
    return psd


class HorizonDistance(object):
    def __init__(
        self,
        f_min,
        f_max,
        delta_f,
        m1,
        m2,
        spin1=(0.0, 0.0, 0.0),
        spin2=(0.0, 0.0, 0.0),
        eccentricity=0.0,
        inclination=0.0,
        approximant="IMRPhenomD",
    ):
        """
        Configures the horizon distance calculation for a specific
        waveform model.  The waveform is pre-computed and stored,
        so this initialization step can be time-consuming but
        computing horizon distances from measured PSDs will be
        fast.

        The waveform model's spectrum parameters, for example its
        Nyquist and frequency resolution, need not match the
        parameters for the PSDs that will ultimately be supplied
        but there are some advantages to be had in getting them to
        match.  For example, computing the waveform with a smaller
        delta_f than will be needed will require additional storage
        and consume additional CPU time for the initialization,
        while computing it with too low an f_max or too large a
        delta_f might lead to inaccurate horizon distance
        estimates.

        f_min (Hertz) sets the frequency at which the waveform
        model is to begin.

        f_max (Hertz) sets the frequency upto which the waveform's
        model is desired.

        delta_f (Hertz) sets the frequency resolution of the
        desired waveform model.

        m1, m2 (solar masses) set the component masses of the
        system to model.

        spin1, spin2 (3-component vectors, geometric units) set the
        spins of the component masses.

        eccentricity [0, 1) sets the eccentricity of the system.

        inclination (radians) sets the orbital inclination of the
        system.

        Example:

        >>> # configure for non-spinning, circular, 1.4+1.4 BNS
        >>> horizon_distance = HorizonDistance(10., 1024., 1./32., 1.4, 1.4)
        >>> # populate a PSD for testing
        >>> import lal, lalsimulation
        >>> psd = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0., 1./32.,
        ...     lal.Unit("strain^2 s"), horizon_distance.model.data.length)
        >>> lalsimulation.SimNoisePSDaLIGODesignSensitivityP1200087(psd, 0.)
        0
        >>> # compute horizon distance
        >>> D, (f, model) = horizon_distance(psd)
        >>> print("%.4g Mpc" % D)
        434.7 Mpc
        >>> # compute distance and spectrum for SNR = 25
        >>> D, (f, model) = horizon_distance(psd, 25.)
        >>> "%.4g Mpc" % D
        '139.1 Mpc'
        >>> f
        array([   10.     ,    10.03125,    10.0625 , ...,  1023.9375 ,
                1023.96875,  1024.     ])
        >>> model
        array([  8.05622865e-45,   7.99763234e-45,   7.93964216e-45, ...,
                 1.11824864e-49,   1.11815656e-49,   1.11806450e-49])

        NOTE:

        - Currently the SEOBNRv4_ROM waveform model is used, so its
          limitations with respect to masses, spins, etc., apply.
          The choice of waveform model is subject to change.
        """
        self.f_min = f_min
        self.f_max = f_max
        self.m1 = m1
        self.m2 = m2
        self.spin1 = numpy.array(spin1)
        self.spin2 = numpy.array(spin2)
        self.inclination = inclination
        self.eccentricity = eccentricity
        self.approximant = approximant
        # NOTE:  the waveform models are computed up-to but not
        # including the supplied f_max parameter so we need to pass
        # (f_max + delta_f) if we want the waveform model defined
        # in the f_max bin
        hp, hc = lalsimulation.SimInspiralFD(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            spin1[0],
            spin1[1],
            spin1[2],
            spin2[0],
            spin2[1],
            spin2[2],
            1.0,  # distance (m)
            inclination,
            0.0,  # reference orbital phase (rad)
            0.0,  # longitude of ascending nodes (rad)
            eccentricity,
            0.0,  # mean anomaly of periastron
            delta_f,
            f_min,
            f_max + delta_f,
            100.0,  # reference frequency (Hz)
            None,  # LAL dictionary containing accessory parameters
            lalsimulation.GetApproximantFromString(self.approximant),
        )
        assert hp.data.length > 0, "huh!?  h+ has zero length!"

        #
        # store |h(f)|^2 for source at D = 1 m.  see (5) in
        # arXiv:1003.2481
        #

        self.model = lal.CreateREAL8FrequencySeries(
            name="signal spectrum",
            epoch=LIGOTimeGPS(0),
            f0=hp.f0,
            deltaF=hp.deltaF,
            sampleUnits=hp.sampleUnits * hp.sampleUnits,
            length=hp.data.length,
        )
        self.model.data.data[:] = numpy.abs(hp.data.data) ** 2.0

    def __call__(self, psd, snr=8.0):
        """
        Compute the horizon distance for the configured waveform
        model given the PSD and the SNR at which the horizon is
        defined (default = 8).  Equivalently, from a PSD and an
        observed SNR compute and return the amplitude of the
        configured waveform's spectrum required to achieve that
        SNR.

        The return value is a two-element tuple.  The first element
        is the horizon distance in Mpc.  The second element is,
        itself, a two-element tuple containing two vectors giving
        the frequencies and amplitudes of the waveform model's
        spectrum scaled so as to have the given SNR.  The vectors
        are clipped to the range of frequencies that were used for
        the SNR integral.

        The parameters of the PSD, for example its Nyquist and
        frequency resolution, need not match the parameters of the
        configured waveform model.  In the event of a mismatch, the
        waveform model is resampled to the frequencies at which the
        PSD has been measured.

        The inspiral spectrum returned has the same units as the
        PSD and is normalized so that the SNR is

        SNR^2 = int (inspiral_spectrum / psd) df

        That is, the ratio of the inspiral spectrum to the PSD
        gives the spectral density of SNR^2.
        """
        #
        # frequencies at which PSD has been measured
        #

        f = psd.f0 + numpy.arange(psd.data.length) * psd.deltaF

        #
        # nearest-neighbour interpolation of waveform model
        # evaluated at PSD's frequency bins
        #

        indexes = (
            ((f - self.model.f0) / self.model.deltaF)
            .round()
            .astype("int")
            .clip(0, self.model.data.length - 1)
        )
        model = self.model.data.data[indexes]

        #
        # range of indexes for integration
        #

        kmin = (max(psd.f0, self.model.f0, self.f_min) - psd.f0) / psd.deltaF
        kmin = int(round(kmin))
        kmax = (
            min(
                psd.f0 + psd.data.length * psd.deltaF,
                self.model.f0 + self.model.data.length * self.model.deltaF,
                self.f_max,
            )
            - psd.f0
        ) / psd.deltaF
        kmax = int(round(kmax)) + 1
        assert kmin < kmax, "PSD and waveform model do not intersect"

        #
        # SNR for source at D = 1 m <--> D in m for source w/ SNR =
        # 1.  see (3) in arXiv:1003.2481
        #

        f = f[kmin:kmax]
        model = model[kmin:kmax]
        D = math.sqrt(4.0 * (model / psd.data.data[kmin:kmax]).sum() * psd.deltaF)

        #
        # distance at desired SNR
        #

        D /= snr

        #
        # scale inspiral spectrum by distance to achieve desired SNR
        #

        model *= 4.0 / D**2.0

        #
        # D in Mpc for source with specified SNR, and waveform
        # model
        #

        return D / (1e6 * lal.PC_SI), (f, model)


def effective_distance_factor(inclination, fp, fc):
    """
    Returns the ratio of effective distance to physical distance for
    compact binary mergers.  Inclination is the orbital inclination of
    the system in radians, fp and fc are the F+ and Fx antenna factors.
    See lal.ComputeDetAMResponse() for a function to compute antenna
    factors.  The effective distance is given by

    Deff = effective_distance_factor * D

    See Equation (4.3) of arXiv:0705.1514.
    """
    cos2i = math.cos(inclination) ** 2
    return 1.0 / math.sqrt(fp**2 * (1 + cos2i) ** 2 / 4 + fc**2 * cos2i)
