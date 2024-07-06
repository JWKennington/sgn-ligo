from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import torch
from torch.masked import masked_tensor, as_masked_tensor

from sgnts.base import Offset
from ..base import SeriesBuffer, TSFrame, TSTransform, AdapterConfig, Time 
from ..math import Math
import math

import lal

def index_select(tensor, dim, index):
    return tensor.gather(dim, index.unsqueeze(dim)).squeeze(dim)

def light_travel_time(ifo1, ifo2):
    """
    Compute and return the time required for light to travel through
    free space the distance separating the two ifos. The result is
    returned in seconds.

    Arguments:
    ----------
    ifo1: str
        prefix of the first ifo (e.g., "H1")
    ifo2: str
        prefix of the first ifo (e.g., "L1")
    """
    dx = lal.cached_detector_by_prefix[ifo1].location - lal.cached_detector_by_prefix[ifo2].location
    return math.sqrt((dx * dx).sum()) / lal.C_SI

@dataclass
class Itacacac(TSTransform):
    """
    An inspiral trigger, autocorrelation chisq, and coincidence, and clustering element
    """
    trigger_finding_length: int = None
    autocorrelation_banks: Sequence[Any] = None
    template_ids: Sequence[Any] = None
    end_times: Sequence[Any] = None
    device: str = "cpu"

    def __post_init__(self):

        self.ifos = list(self.autocorrelation_banks.keys())
        self.nifo = len(self.ifos)

        (
            self.nbank,
            self.ntempmax,
            self.autocorrelation_length,
        ) = self.autocorrelation_banks[self.ifos[0]].shape

        self.padding = self.autocorrelation_length // 2
        self.adapter_config = AdapterConfig(
            stride = self.trigger_finding_length,
            overlap = (self.padding, self.padding),
            lib=Math
        )
        self.ifos_number_map = {ifo: i+1 for i, ifo in enumerate(self.ifos)}
        self.template_ids = self.template_ids.to(self.device)

        # Denominator Eq 28 from arXiv:1604.04324
        # self.autocorrelation_norms = torch.sum(
        #    2 - 2 * abs(self.autocorrelation_banks) ** 2.0, dim=-1
        # )
        # FIXME: Dropping the factor of 2 in front of abs to match the norm in
        #        gstlal_autocorrelation_chi2.c

        self.autocorrelation_norms = {}
        for ifo in self.ifos:
            self.autocorrelation_norms[ifo] = torch.sum(
                2 - abs(self.autocorrelation_banks[ifo]) ** 2, dim=-1
            )

        self.snr_time_series_indices = torch.arange(
            self.autocorrelation_length, device=self.device
        ).expand(self.nbank, self.ntempmax, -1)

        super().__post_init__()

    def find_peaks_and_calculate_chisqs(self, snrs):
        """
        Find snr peaks in a given snr time series window, and obtain peak time,
           phase, and chisq

        Arguments:
        ----------
        snrs: dict[str, torch.tensor]
            a dictionary of torch.tensors, with ifo as keys
            only contains snrs for ifos with nongap tensors
        """

        # Assume data at this point has already been vetted,
        #   i.e. non-gap samples only and there's enough padding
        padding = self.padding
        idi = padding
        idf = padding + self.trigger_finding_length
        triggers = {}

        for (ifo, snr) in snrs.items():
            # NOTE Assumes snr[nbank, channel index, real or imaginary index,
            #   time index]
            shape = snr.shape
            snr = snr.view(shape[0], shape[1] // 2, 2, shape[2])
            real = snr[..., 0, :]
            imag = snr[..., 1, :]
            # complex_snr = torch.complex(real, imag)

            # Get peaks
            # peaks, peak_locations = torch.max(abs(complex_snr[...,
            #   initial_padding_slice:final_padding_slice]), dim=-1)
            peaks, peak_locations = torch.max(
                torch.sqrt(real[..., idi:idf] ** 2 + imag[..., idi:idf] ** 2), dim=-1
            )
            peak_locations += idi

            # angles = torch.angle(snr[..., 0, peak_locations] + 1j*snr[..., 1,
            #   peak_locations]).transpose(0,1)
            angles = torch.angle(
                index_select(real, 2, peak_locations)
                + 1j * index_select(imag, 2, peak_locations)
            )

            # above_threshold = torch.ge(peaks, 4) # TODO Explore dropping threshold
            # peaks = peaks[above_threshold]
            # peak_locations = peak_locations[above_threshold]

            # Save the SNR which we'll use to compute the autocorrelation chisq and
            # that we'll save for use in localization
            # autocorrelation_snr_time_series = snr[...,
            #   (peak_locations - self.padding):(peak_locations + self.padding)]
            time_series_indices = self.snr_time_series_indices + (
                peak_locations - self.padding
            ).unsqueeze(2)
            # Gather is not implemented for complex half type,
            #   so do it for real and imag separately
            autocorrelation_snr_time_series = (
                real.gather(2, time_series_indices)
                + imag.gather(2, time_series_indices) * 1j
            )
            snr_ts_shape = autocorrelation_snr_time_series.shape

            # Make a copy that we can rotate
            # Defer till later
            # rotated_autocorrelation_snr_time_series =
            #   autocorrelation_snr_time_series.detach().clone()
            # rotated_snr = autocorrelation_snr_time_series.detach().clone()
            rotated_snr = autocorrelation_snr_time_series

            # Rotate the snr time series such that the SNR peak is real
            # rotated_autocorrelation_snr_time_series[..., 0, :]
            #   *= torch.cos(angles)
            # rotated_autocorrelation_snr_time_series[..., 1, :]
            #   *= (-1)*torch.sin(angles)
            rotated_snr *= torch.exp(-angles * 1j).unsqueeze(2).expand(snr_ts_shape)

            # Eq 28 from arXiv:1604.04324 reduces down to (norm)^-1 *
            #   sum_t((Re[z(t)] - z(0)*R(t))**2. + Im[z(t)]**2.)
            #   if z(t) has been rotated so z(0) is real
            # autocorrelation_chisq = torch.sum((
            #   rotated_autocorrelation_snr_time_series[:,0,:]
            #   - peaks * self.autocorrelations)**2. +
            #   rotated_autocorrelation_snr_time_series[:,1,:]**2., dim=1)
            expanded_peaks = peaks.unsqueeze(2).expand(snr_ts_shape)
            autocorrelation_chisq = torch.sum(
                abs(rotated_snr - expanded_peaks * self.autocorrelation_banks[ifo]) ** 2,
                dim=-1,
            )
            autocorrelation_chisq /= self.autocorrelation_norms[ifo]

            # Eq blah from Prathamesh paper will go here
            # bank_chisq =

            # Construct the tensor of trigger information
            triggers[ifo] = [
                peak_locations,
                torch.dstack([peaks, angles, autocorrelation_chisq]),
            ]
        return triggers

    def make_coincs(self, triggers):
        on_ifos = list(triggers.keys())
        nifo = len(on_ifos)
        snr_chisq_hist_index = {}
        single_masks = {} # for snr chisq histogram

        if nifo == 1:
            # return the single ifo snrs
            all_network_snr = [t[1] for t in triggers.values()][0]
            ifo_combs = torch.zeros_like(all_network_snr, dtype=torch.int) * self.ifos_number_map[on_ifos[0]]

        elif nifo == 2:
            times = [t[0] for t in triggers.values()]
            snrs = [t[1] for t in triggers.values()]
            chisqs = [t[3] for t in triggers.values()]
            coinc2_mask, single_mask1, single_mask2, all_network_snr = self.coinc2(snrs, times, on_ifos)

            # convert ifo combination masks to numbers
            ifo_numbers = [self.ifos_number_map[ifo] for ifo in on_ifos]
            ifo_combs = torch.zeros_like(all_network_snr, dtype=torch.int)
            ifo_combs.masked_fill_(coinc2_mask, ifo_numbers[0]*10 + ifo_numbers[1])
            ifo_combs.masked_fill_(single_mask1, ifo_numbers[0])
            ifo_combs.masked_fill_(single_mask2, ifo_numbers[1])

            smasks = [single_mask1, single_mask2]

            for i, ifo in enumerate(on_ifos):
                single_masks[ifo] = smasks[i]
                #snr_chisq_hist_index[ifo] = self.hist_index(snrs[i], chisq[i], single_masks[i])

        elif nifo == 3:
            coinc3_mask, coinc2_mask12, coinc2_mask23, coinc2_mask31, single_mask1, single_mask2, single_mask3, all_network_snr = self.coinc3(triggers)

            # convert ifo combination masks to numbers
            ifo_numbers = list(self.ifos_number_map.values())
            ifo_combs = torch.zeros_like(all_network_snr, dtype=torch.int)
            ifo_combs.masked_fill_(coinc3_mask, ifo_numbers[0]*100 + ifo_numbers[1]*10 + ifo_numbers[2])
            ifo_combs.masked_fill_(coinc2_mask12, ifo_numbers[0]*10 + ifo_numbers[1])
            ifo_combs.masked_fill_(coinc2_mask23, ifo_numbers[1]*10 + ifo_numbers[2])
            ifo_combs.masked_fill_(coinc2_mask31, ifo_numbers[0]*10 + ifo_numbers[2])
            ifo_combs.masked_fill_(single_mask1, ifo_numbers[0])
            ifo_combs.masked_fill_(single_mask2, ifo_numbers[1])
            ifo_combs.masked_fill_(single_mask3, ifo_numbers[2])

            smasks = [single_mask1, single_mask2, single_mask3]
            for i, ifo in enumerate(on_ifos):
                single_masks[ifo] = smasks[i]
                #snr_chisq_hist_index[ifo] = self.hist_index(snrs[i], chisq[i], single_masks[i])
            
        else:
            raise ValueError("nifo > 3 is not tested")

        return ifo_combs, all_network_snr, single_masks

    def coinc3(self, triggers):
        ifos = list(triggers.keys())
        snrs = [t[1][...,0] for t in triggers.values()]
        times = [t[0] for t in triggers.values()]

        snr1 = snrs[0]
        snr2 = snrs[1]
        snr3 = snrs[2]

        # all combinations
        coinc2_mask12, _, _, _ = self.coinc2([snr1, snr2], [times[0], times[1]], [ifos[0], ifos[1]])
        coinc2_mask23, _, _, _ = self.coinc2([snr2, snr3], [times[1], times[2]], [ifos[1], ifos[2]])
        coinc2_mask31, _, _, _ = self.coinc2([snr1, snr3], [times[0], times[2]], [ifos[0], ifos[2]])

        # 3 ifo coincs
        coinc3_mask = coinc2_mask12 & coinc2_mask23 & coinc2_mask31
        network_snr123 = (snr1.masked_fill(~coinc3_mask,0)**2
                      + snr2.masked_fill(~coinc3_mask,0)**2
                      + snr3.masked_fill(~coinc3_mask,0)**2)**.5

        # 2 ifo coincs
        # update coinc masks: filter out 3 ifo coincs
        coinc2_mask12 = coinc2_mask12 & ~coinc3_mask
        coinc2_mask23 = coinc2_mask23 & ~coinc3_mask
        coinc2_mask31 = coinc2_mask31 & ~coinc3_mask

        network_snr12 = (snr1.masked_fill(~coinc2_mask12,0)**2
                        +snr2.masked_fill(~coinc2_mask12,0)**2)**.5
        network_snr23 = (snr2.masked_fill(~coinc2_mask23,0)**2
                        +snr3.masked_fill(~coinc2_mask23,0)**2)**.5
        network_snr31 = (snr1.masked_fill(~coinc2_mask31,0)**2
                        +snr3.masked_fill(~coinc2_mask31,0)**2)**.5

        # update coinc masks: there may be cases where a template has 
        # two coincs, (e.g., HV coinc and LV coinc, but not HL coinc), 
        # in this case, compare HV, LV coinc network snrs and choose 
        # the larger one
        # FIXME: what to do when snrs are equal?
        coinc2_mask12 = coinc2_mask12 & (network_snr12 > network_snr23) & (network_snr12 >= network_snr31)
        coinc2_mask23 = coinc2_mask23 & (network_snr23 >= network_snr12) & (network_snr23 > network_snr31)
        coinc2_mask31 = coinc2_mask31 & (network_snr31 > network_snr12) & (network_snr31 >= network_snr23)

        # update 2 ifo network snrs
        network_snr12 = (snr1.masked_fill(~coinc2_mask12,0)**2
                        +snr2.masked_fill(~coinc2_mask12,0)**2)**.5
        network_snr23 = (snr2.masked_fill(~coinc2_mask23,0)**2
                        +snr3.masked_fill(~coinc2_mask23,0)**2)**.5
        network_snr31 = (snr1.masked_fill(~coinc2_mask31,0)**2
                        +snr3.masked_fill(~coinc2_mask31,0)**2)**.5

        # 1 ifo
        # FIXME: what to do when snrs are equal?
        single_mask1 = ~coinc3_mask & ~coinc2_mask12 & ~coinc2_mask23 & ~coinc2_mask31 & (snr1 > snr2) & (snr1 >= snr3)
        single_mask2 = ~coinc3_mask & ~coinc2_mask12 & ~coinc2_mask23 & ~coinc2_mask31 & (snr2 >= snr1) & (snr2 > snr3)
        single_mask3 = ~coinc3_mask & ~coinc2_mask12 & ~coinc2_mask23 & ~coinc2_mask31 & (snr3 > snr1) & (snr3 >= snr2)

        single_snr1 = snr1.masked_fill(~single_mask1,0)
        single_snr2 = snr2.masked_fill(~single_mask2,0)
        single_snr3 = snr3.masked_fill(~single_mask3,0)

        all_network_snrs = network_snr123 + network_snr12 + network_snr23 + network_snr31 + single_snr1  + single_snr2  + single_snr3 

        return coinc3_mask, coinc2_mask12, coinc2_mask23, coinc2_mask31, single_mask1, single_mask2, single_mask3, all_network_snrs


    def coinc2(self, snrs, times, ifos):
        dt = Offset.fromsec(light_travel_time(*ifos))
        snr1 = snrs[0]
        snr2 = snrs[1]
        time1 = times[0]
        time2 = times[1]
        coinc_mask = abs(time1-time2) < dt
        single_mask1 = (snr1 > snr2) & ~coinc_mask
        single_mask2 = ~single_mask1 & ~coinc_mask

        snr_masked1 = masked_tensor(snr1, coinc_mask)
        snr_masked2 = masked_tensor(snr2, coinc_mask)
        coinc_network_snr = (snr_masked1 ** 2 + snr_masked2 ** 2)**.5

        single1 = masked_tensor(snr1, single_mask1)
        single2 = masked_tensor(snr2, single_mask2)

        all_network_snr = coinc_network_snr.to_tensor(0) + single1.to_tensor(0) + single2.to_tensor(0)

        return coinc_mask, single_mask1, single_mask2, all_network_snr,

    def cluster_coincs(self, ifo_combs, all_network_snr, template_ids, triggers):
        clustered_snr, max_locations = torch.max(all_network_snr, dim=-1)
        clustered_ifo_combs = index_select(ifo_combs, 1, max_locations)
        clustered_template_ids = index_select(template_ids, 1, max_locations)
        sngls = {}
        for ifo, trig in triggers.items():
            sngls[ifo] = {}
            time = index_select(triggers[ifo][0], 1, max_locations)
            trig1  = index_select(triggers[ifo][1], 1, max_locations.unsqueeze(1).expand(self.nbank, 3))
            if self.device != "cpu":
                time = time.to("cpu")
                trig1 = trig1.to("cpu")
            sngls[ifo]["time"] = time

            sngls[ifo]["snrs"] = trig1[...,0]
            sngls[ifo]["phase"] = trig1[...,1]
            sngls[ifo]["chisq"] = trig1[...,2]

        #clustered_phases = index_select(phases, 1, max_locations)
        #clustered_chisq = index_select(phase, 1, max_locations)


        # FIXME: is stacking then index_select faster?
        # FIXME: how is the time, phase, chisq defined?
        # FIXME: add end_time correction

        return [torch.dstack([clustered_template_ids, clustered_ifo_combs]),
        torch.dstack([clustered_snr]), sngls]
        #torch.dstack([clustered_snr, clustered_phases, clustered_chisq])]

    def transform(self, pad):
        frames = self.preparedframes
        self.preparedframes = {}

        snrs = {}

        for sink_pad in self.sink_pads:
            # FIXME: consider multiple buffers
            frame = frames[sink_pad]
            assert len(frame.buffers) == 1
            buf = frame.buffers[0]
            if not buf.is_gap: 
                snrs[sink_pad.name.split(":")[-1]] = buf.data

        if len(snrs.keys()) >= 1:
            triggers = self.find_peaks_and_calculate_chisqs(snrs)

            # FIXME: consider edge effects
            ifo_combs, all_network_snr, single_masks = self.make_coincs(triggers)

            clustered_coinc = self.cluster_coincs(ifo_combs, all_network_snr, self.template_ids, triggers)

            if self.device != "cpu":
                for ifo in triggers.keys():
                    for i in range(len(triggers[ifo])):
                        triggers[ifo][i] = triggers[ifo][i].to("cpu").numpy()
                    single_masks[ifo]=single_masks[ifo].to("cpu").numpy()
                clustered_coinc[0] = clustered_coinc[0].to("cpu").numpy()
                clustered_coinc[1] = clustered_coinc[1].to("cpu").numpy()

            # FIXME: is stacking then copying to cpu faster?
            # FIXME: do we only need snr chisq for singles?
            background = {}
            for ifo in triggers.keys():
                background[ifo] = {}
                background[ifo]["snrs"] = triggers[ifo][1][...,0]
                background[ifo]["chisqs"] = triggers[ifo][1][...,2]
                background[ifo]["single_masks"] = single_masks[ifo]

            metadata= frame.metadata
            metadata["coincs"] = {
                    "template_ids":clustered_coinc[0][...,0],
                    "ifo_combs":clustered_coinc[0][...,1],
                    "snrs":clustered_coinc[1][...,0],
                    "ifos_number_map": self.ifos_number_map,
                    "time":None, # FIXME: how is time defined?
                    "sngl":clustered_coinc[2],
                    }
            metadata["background"] = background
        else:
            metadata = frame.metadata

        outbuf = SeriesBuffer(
            offset=self.preparedoutoffsets[self.sink_pads[0]][0]["offset"],
            sample_rate=frame.sample_rate,
            data=None,
            shape= (self.nbank, Offset.tosamples(self.preparedoutoffsets[self.sink_pads[0]][0]["noffset"], frame.sample_rate),),
        )

        return TSFrame(buffers=[outbuf], EOS=frame.EOS, metadata=metadata)
