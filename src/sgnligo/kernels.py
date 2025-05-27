from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import lal
import numpy
from lal import LIGOTimeGPS

from sgnligo.psd import HorizonDistance


@dataclass
class PSDFirKernel:
    revplan: Optional[str] = None
    fwdplan: Optional[str] = None
    target_phase: Optional[numpy.ndarray[Any, Any]] = None
    target_phase_mask: Optional[
        numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
    ] = None

    def set_phase(
        self,
        psd: lal.REAL8FrequencySeries,
        f_low: float = 10.0,
        m1: float = 1.4,
        m2: float = 1.4,
    ) -> None:
        """
        Compute the phase response of zero-latency whitening filter
        given a reference PSD.

        """
        kernel, latency, sample_rate = self.psd_to_linear_phase_whitening_fir_kernel(
            psd
        )
        kernel, phase = (
            self.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(
                kernel, sample_rate
            )
        )

        # get merger model for SNR = 1.
        f_psd = psd.f0 + numpy.arange(len(psd.data.data)) * psd.deltaF
        horizon_distance = HorizonDistance(f_low, f_psd[-1], psd.deltaF, m1, m2)
        f_model, model = horizon_distance(psd, 1.0)[1]

        # find the range of frequency bins covered by the merger
        # model
        kmin, kmax = f_psd.searchsorted(f_model[0]), f_psd.searchsorted(f_model[-1]) + 1

        # compute SNR=1 model's (d SNR^2 / df) spectral density
        unit_snr2_density = numpy.zeros_like(phase)
        unit_snr2_density[kmin:kmax] = model / psd.data.data[kmin:kmax]

        # integrate across each frequency bin, converting to
        # snr^2/bin.  NOTE:  this step is here for the record, but
        # is commented out because it has no effect on the result
        # given the renormalization that occurs next.
        # unit_snr2_density *= psd.deltaF

        # take 16th root, then normalize so max=1.  why?  I don't
        # know, just feels good, on the whole.
        unit_snr2_density = unit_snr2_density ** (1.0 / 16)
        unit_snr2_density /= unit_snr2_density.max()

        # record phase vector and SNR^2 density vector
        self.target_phase = phase
        self.target_phase_mask = unit_snr2_density

    def psd_to_linear_phase_whitening_fir_kernel(
        self,
        psd: lal.REAL8FrequencySeries,
        invert: Optional[bool] = True,
        nyquist: Optional[float] = None,
    ) -> Tuple[numpy.ndarray, int, int]:
        """
        Compute an acausal finite impulse-response filter kernel
        from a power spectral density conforming to the LAL
        normalization convention, such that if colored Gaussian
        random noise with the given PSD is fed into an FIR filter
        using the kernel the filter's output will be zero-mean
        unit-variance Gaussian random noise.  The PSD must be
        provided as a lal.REAL8FrequencySeries object.

        The phase response of this filter is 0, just like whitening
        done in the frequency domain.

        Args:
                psd:
                        lal.REAL8FrequencySeries, the reference PSD
                invert:
                        bool, default true, whether to invert the kernel
                nyquist:
                        float, disabled by default, whether to change
                        the Nyquist frequency.

        Returns:
                Tuple[numpy.ndarray, int, int], the kernel, latency,
                sample rate pair.  The kernel is a numpy array containing
                the filter kernel, the latency is the filter latency in
                samples and the sample rate is in Hz.  The kernel and
                latency can be used, for example, with gstreamer's stock
                audiofirfilter element.

        """
        #
        # this could be relaxed with some work
        #

        assert psd.f0 == 0.0

        #
        # extract the PSD bins and determine sample rate for kernel
        #

        data = psd.data.data / 2
        sample_rate = 2 * int(round(psd.f0 + (len(data) - 1) * psd.deltaF))

        #
        # remove LAL normalization
        #

        data *= sample_rate

        #
        # change Nyquist frequency if requested.  round to nearest
        # available bin
        #

        if nyquist is not None:
            i = int(round((nyquist - psd.f0) / psd.deltaF))
            assert i < len(data)
            data = data[: i + 1]
            sample_rate = 2 * int(round(psd.f0 + (len(data) - 1) * psd.deltaF))

        #
        # compute the FIR kernel.  it always has an odd number of
        # samples and no DC offset.
        #

        data[0] = data[-1] = 0.0
        if invert:
            data_nonzeros = data != 0.0
            data[data_nonzeros] = 1.0 / data[data_nonzeros]
        # repack data:  data[0], data[1], 0, data[2], 0, ....
        tmp = numpy.zeros((2 * len(data) - 1,), dtype=data.dtype)
        tmp[len(data) - 1 :] = data
        # tmp[:len(data)] = data
        data = tmp

        kernel_fseries = lal.CreateCOMPLEX16FrequencySeries(
            name="double sided psd",
            epoch=LIGOTimeGPS(0),
            f0=0.0,
            deltaF=psd.deltaF,
            length=len(data),
            sampleUnits=lal.Unit("strain s"),
        )

        kernel_tseries = lal.CreateCOMPLEX16TimeSeries(
            name="timeseries of whitening kernel",
            epoch=LIGOTimeGPS(0.0),
            f0=0.0,
            deltaT=1.0 / sample_rate,
            length=len(data),
            sampleUnits=lal.Unit("strain"),
        )

        # FIXME check for change in length
        if self.revplan is None:
            self.revplan = lal.CreateReverseCOMPLEX16FFTPlan(len(data), 1)

        kernel_fseries.data.data = numpy.sqrt(data) + 0.0j
        lal.COMPLEX16FreqTimeFFT(kernel_tseries, kernel_fseries, self.revplan)
        kernel = kernel_tseries.data.data.real
        kernel = numpy.roll(kernel, (len(data) - 1) // 2) / sample_rate * 2

        #
        # apply a Tukey window whose flat bit is 50% of the kernel.
        # preserve the FIR kernel's square magnitude
        #

        norm_before = numpy.dot(kernel, kernel)
        kernel *= lal.CreateTukeyREAL8Window(len(data), 0.5).data.data
        kernel *= math.sqrt(norm_before / numpy.dot(kernel, kernel))

        #
        # the kernel's latency
        #

        latency = (len(data) - 1) // 2

        #
        # done
        #

        return kernel, latency, sample_rate

    def linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(
        self, linear_phase_kernel: numpy.ndarray, sample_rate: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Compute the minimum-phase response filter (zero latency)
        associated with a linear-phase response filter (latency
        equal to half the filter length).

        From "Design of Optimal Minimum-Phase Digital FIR Filters
        Using Discrete Hilbert Transforms", IEEE Trans. Signal
        Processing, vol. 48, pp. 1491-1495, May 2000.

        Args:
                linear_phase_kernel:
                        numpy.ndarray, the kernel to compute the minimum-phase kernel
                        from
                sample_rate:
                        int, the sample rate

        Returns:
                Tuple[numpy.ndarray. numpy.ndarray], the kernel and the phase response.
                The kernel is a numpy array containing the filter kernel. The kernel
                can be used, for example, with gstreamer's stock audiofirfilter element.

        """
        #
        # compute abs of FFT of kernel
        #

        # FIXME check for change in length
        if self.fwdplan is None:
            self.fwdplan = lal.CreateForwardCOMPLEX16FFTPlan(
                len(linear_phase_kernel), 1
            )
        if self.revplan is None:
            self.revplan = lal.CreateReverseCOMPLEX16FFTPlan(
                len(linear_phase_kernel), 1
            )

        deltaT = 1.0 / sample_rate
        deltaF = 1.0 / (len(linear_phase_kernel) * deltaT)
        working_length = len(linear_phase_kernel)

        kernel_tseries = lal.CreateCOMPLEX16TimeSeries(
            name="timeseries of whitening kernel",
            epoch=LIGOTimeGPS(0.0),
            f0=0.0,
            deltaT=deltaT,
            length=working_length,
            sampleUnits=lal.Unit("strain"),
        )
        kernel_tseries.data.data = linear_phase_kernel

        absX = lal.CreateCOMPLEX16FrequencySeries(
            name="absX",
            epoch=LIGOTimeGPS(0),
            f0=0.0,
            deltaF=deltaF,
            length=working_length,
            sampleUnits=lal.Unit("strain s"),
        )

        logabsX = lal.CreateCOMPLEX16FrequencySeries(
            name="absX",
            epoch=LIGOTimeGPS(0),
            f0=0.0,
            deltaF=deltaF,
            length=working_length,
            sampleUnits=lal.Unit("strain s"),
        )

        cepstrum = lal.CreateCOMPLEX16TimeSeries(
            name="cepstrum",
            epoch=LIGOTimeGPS(0.0),
            f0=0.0,
            deltaT=deltaT,
            length=working_length,
            sampleUnits=lal.Unit("strain"),
        )

        theta = lal.CreateCOMPLEX16FrequencySeries(
            name="theta",
            epoch=LIGOTimeGPS(0),
            f0=0.0,
            deltaF=deltaF,
            length=working_length,
            sampleUnits=lal.Unit("strain s"),
        )

        min_phase_kernel = lal.CreateCOMPLEX16TimeSeries(
            name="min phase kernel",
            epoch=LIGOTimeGPS(0.0),
            f0=0.0,
            deltaT=deltaT,
            length=working_length,
            sampleUnits=lal.Unit("strain"),
        )

        lal.COMPLEX16TimeFreqFFT(absX, kernel_tseries, self.fwdplan)
        absX.data.data[:] = abs(absX.data.data)

        #
        # compute the cepstrum of the kernel (i.e., the iFFT of the
        # log of the abs of the FFT of the kernel)
        #

        logabsX.data.data[:] = numpy.log(absX.data.data)
        lal.COMPLEX16FreqTimeFFT(cepstrum, logabsX, self.revplan)

        #
        # multiply cepstrum by sgn
        #

        cepstrum.data.data[0] = 0.0
        cepstrum.data.data[working_length // 2] = 0.0
        cepstrum.data.data[working_length // 2 + 1 :] = -cepstrum.data.data[
            working_length // 2 + 1 :
        ]

        #
        # compute theta
        #

        lal.COMPLEX16TimeFreqFFT(theta, cepstrum, self.fwdplan)

        #
        # compute the gain and phase of the zero-phase
        # approximation relative to the original linear-phase
        # filter
        #

        theta_data = theta.data.data[working_length // 2 :]
        # gain = numpy.exp(theta_data.real)
        phase = -theta_data.imag

        #
        # apply optional masked phase adjustment
        #

        if self.target_phase is not None:
            # compute phase adjustment for +ve frequencies
            phase_adjustment = (self.target_phase - phase) * self.target_phase_mask

            # combine with phase adjustment for -ve frequencies
            phase_adjustment = numpy.concatenate(
                (phase_adjustment[1:][-1::-1].conj(), phase_adjustment)
            )

            # apply adjustment.  phase adjustment is what we
            # wish to add to the phase.  theta's imaginary
            # component contains the negative of the phase, so
            # we need to add -phase to theta's imaginary
            # component
            theta.data.data += -1.0j * phase_adjustment

            # report adjusted phase
            # phase = -theta.data.data[working_length // 2:].imag

        #
        # compute minimum phase kernel
        #

        absX.data.data *= numpy.exp(theta.data.data)
        lal.COMPLEX16FreqTimeFFT(min_phase_kernel, absX, self.revplan)

        kernel = min_phase_kernel.data.data.real

        #
        # this kernel needs to be reversed to follow conventions
        # used with the audiofirfilter and lal_firbank elements
        #

        kernel = kernel[-1::-1]

        #
        # done
        #

        return kernel, phase


def fir_whitener_kernel(
    length: int, duration: float, sample_rate: int, psd: lal.REAL8FrequencySeries
) -> lal.COMPLEX16FrequencySeries:
    """Create an FIR whitener kernel."""
    assert psd
    #
    # Add another COMPLEX16TimeSeries and COMPLEX16FrequencySeries for kernel's FFT
    # (Leo)
    #

    # Add another FFT plan for kernel FFT (Leo)
    fwdplan_kernel = lal.CreateForwardCOMPLEX16FFTPlan(length, 1)
    kernel_tseries = lal.CreateCOMPLEX16TimeSeries(
        name="timeseries of whitening kernel",
        epoch=LIGOTimeGPS(0.0),
        f0=0.0,
        deltaT=1.0 / sample_rate,
        length=length,
        sampleUnits=lal.Unit("strain"),
    )
    kernel_fseries = lal.CreateCOMPLEX16FrequencySeries(
        name="freqseries of whitening kernel",
        epoch=LIGOTimeGPS(0),
        f0=0.0,
        deltaF=1.0 / duration,
        length=length,
        sampleUnits=lal.Unit("strain s"),
    )

    #
    # Obtain a kernel of zero-latency whitening filter and
    # adjust its length (Leo)
    #

    psd_fir_kernel = PSDFirKernel()
    (kernel, latency, fir_rate) = (
        psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(
            psd, nyquist=sample_rate / 2.0
        )
    )
    (kernel, theta) = (
        psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(
            kernel, fir_rate
        )
    )
    kernel = kernel[-1::-1]
    # FIXME this is off by one sample, but shouldn't be. Look at the miminum phase
    # function
    # assert len(kernel) == length
    if len(kernel) < length:
        kernel = numpy.append(kernel, numpy.zeros(length - len(kernel)))
    else:
        kernel = kernel[:length]

    kernel_tseries.data.data = kernel

    #
    # FFT of the kernel
    #

    lal.COMPLEX16TimeFreqFFT(kernel_fseries, kernel_tseries, fwdplan_kernel)  # FIXME

    return kernel_fseries
