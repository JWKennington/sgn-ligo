from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import torch

from sgnts.base import Offset
from ..base import SeriesBuffer, TSFrame, TSTransform, AdapterConfig, ArrayOps, now
import math

import lal
import numpy as np


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
    dx = (
        lal.cached_detector_by_prefix[ifo1].location
        - lal.cached_detector_by_prefix[ifo2].location
    )
    return math.sqrt((dx * dx).sum()) / lal.C_SI


@dataclass
class Itacacac(TSTransform):
    """
    An inspiral trigger, autocorrelation chisq, and coincidence, and clustering element
    """

    trigger_finding_length: int = None
    autocorrelation_banks: Sequence[Any] = None
    template_ids: Sequence[Any] = None
    bankids_map: Sequence[Any] = None
    end_times: Sequence[Any] = None
    device: str = "cpu"
    kafka: bool = False

    def __post_init__(self):

        self.ifos = list(self.autocorrelation_banks.keys())
        self.nifo = len(self.ifos)

        (
            self.nsubbank,
            self.ntempmax,
            self.autocorrelation_length,
        ) = self.autocorrelation_banks[self.ifos[0]].shape
        self.autocorrelation_banks_real = {}
        self.autocorrelation_banks_imag = {}
        self.ifos_number_map = {ifo: i + 1 for i, ifo in enumerate(self.ifos)}
        for ifo in self.ifos:
            self.autocorrelation_banks_real[ifo] = self.autocorrelation_banks[ifo].real
            self.autocorrelation_banks_imag[ifo] = self.autocorrelation_banks[ifo].imag

        self.padding = self.autocorrelation_length // 2
        self.adapter_config = AdapterConfig(
            stride=self.trigger_finding_length,
            overlap=(self.padding, self.padding),
            lib=ArrayOps,
        )
        self.template_ids = self.template_ids.to(self.device)
        self.template_ids_np = self.template_ids.to("cpu").numpy()
        self.end_times = self.end_times.numpy()

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
        ).expand(self.nsubbank, self.ntempmax, -1)

        super().__post_init__()

    def find_peaks_and_calculate_chisqs(self, snrs):
        """
        find snr peaks in a given snr time series window, and obtain peak time,
           phase, and chisq

        arguments:
        ----------
        snrs: dict[str, torch.tensor]
            a dictionary of torch.tensors, with ifo as keys
            only contains snrs for ifos with nongap tensors
        """

        padding = self.padding
        idi = padding
        idf = padding + self.trigger_finding_length
        triggers = {}
        for ifo, snr in snrs.items():
            shape = snr.shape
            snr = snr.view(shape[0], shape[1] // 2, 2, shape[2])
            real = snr[..., 0, :]
            imag = snr[..., 1, :]
            peaks, peak_locations = torch.max(
                (real[..., idi:idf] ** 2 + imag[..., idi:idf] ** 2), dim=-1
            )
            peaks **= 0.5
            peak_locations += idi
            time_series_indices = self.snr_time_series_indices + (
                peak_locations - self.padding
            ).unsqueeze(2)
            real_imag_time_series = snr.gather(
                3,
                time_series_indices.unsqueeze(2).expand(
                    shape[0], shape[1] // 2, 2, self.autocorrelation_length
                ),
            )
            real_time_series = real_imag_time_series[..., 0, :]
            imag_time_series = real_imag_time_series[..., 1, :]
            snr_ts_shape = real_time_series.shape

            real_peak = real_time_series[..., padding].unsqueeze(2).expand(snr_ts_shape)
            imag_peak = imag_time_series[..., padding].unsqueeze(2).expand(snr_ts_shape)

            # complex operations are slow with torch compile, make them real
            autocorrelation_chisq = torch.sum(
                (
                    real_time_series
                    - real_peak * self.autocorrelation_banks_real[ifo]
                    + imag_peak * self.autocorrelation_banks_imag[ifo]
                )
                ** 2
                + (
                    imag_time_series
                    - real_peak * self.autocorrelation_banks_imag[ifo]
                    - imag_peak * self.autocorrelation_banks_real[ifo]
                )
                ** 2,
                dim=-1,
            )
            autocorrelation_chisq /= self.autocorrelation_norms[ifo]
            triggers[ifo] = [
                peak_locations,
                peaks,
                autocorrelation_chisq,
            ]
        return triggers

    def make_coincs(self, triggers):
        on_ifos = list(triggers.keys())
        nifo = len(on_ifos)
        snr_chisq_hist_index = {}
        single_masks = {}  # for snr chisq histogram

        if nifo == 1:
            # return the single ifo snrs
            all_network_snr = [t[1] for t in triggers.values()][0]
            ifo_combs = (
                torch.ones_like(all_network_snr, dtype=torch.int)
                * self.ifos_number_map[on_ifos[0]]
            )

        elif nifo == 2:
            times = [t[0] for t in triggers.values()]
            snrs = [t[1] for t in triggers.values()]
            coinc2_mask, single_mask1, single_mask2, all_network_snr = self.coinc2(
                snrs, times, on_ifos
            )

            # convert ifo combination masks to numbers
            ifo_numbers = [self.ifos_number_map[ifo] for ifo in on_ifos]
            ifo_combs = (
                coinc2_mask * (ifo_numbers[0] * 10 + ifo_numbers[1])
                + single_mask1 * ifo_numbers[0]
                + single_mask2 * ifo_numbers[1]
            )

            smasks = [single_mask1, single_mask2]
            for i, ifo in enumerate(on_ifos):
                single_masks[ifo] = smasks[i]

        elif nifo == 3:
            (
                coinc3_mask,
                coinc2_mask12,
                coinc2_mask23,
                coinc2_mask31,
                single_mask1,
                single_mask2,
                single_mask3,
                all_network_snr,
            ) = self.coinc3(triggers)

            # convert ifo combination masks to numbers
            ifo_numbers = list(self.ifos_number_map.values())

            ifo_combs = (
                coinc3_mask
                * (ifo_numbers[0] * 100 + ifo_numbers[1] * 10 + ifo_numbers[2])
                + coinc2_mask12 * (ifo_numbers[0] * 10 + ifo_numbers[1])
                + coinc2_mask23 * (ifo_numbers[1] * 10 + ifo_numbers[2])
                + coinc2_mask31 * (ifo_numbers[0] * 10 + ifo_numbers[2])
                + single_mask1 * ifo_numbers[0]
                + single_mask2 * ifo_numbers[1]
                + single_mask3 * ifo_numbers[2]
            )

            smasks = [single_mask1, single_mask2, single_mask3]
            for i, ifo in enumerate(on_ifos):
                single_masks[ifo] = smasks[i]
        else:
            raise ValueError("nifo > 3 is not implemented")

        return ifo_combs, all_network_snr, single_masks

    def coinc3(self, triggers):
        ifos = list(triggers.keys())
        snrs = [t[1] for t in triggers.values()]
        times = [t[0] for t in triggers.values()]

        snr1 = snrs[0]
        snr2 = snrs[1]
        snr3 = snrs[2]

        # all combinations
        coinc2_mask12, _, _, _ = self.coinc2(
            [snr1, snr2], [times[0], times[1]], [ifos[0], ifos[1]]
        )
        coinc2_mask23, _, _, _ = self.coinc2(
            [snr2, snr3], [times[1], times[2]], [ifos[1], ifos[2]]
        )
        coinc2_mask31, _, _, _ = self.coinc2(
            [snr1, snr3], [times[0], times[2]], [ifos[0], ifos[2]]
        )

        # 3 ifo coincs
        coinc3_mask = coinc2_mask12 & coinc2_mask23 & coinc2_mask31
        network_snr123 = (
            (snr1 * coinc3_mask) ** 2
            + (snr2 * coinc3_mask) ** 2
            + (snr3 * coinc3_mask) ** 2
        ) ** 0.5

        # 2 ifo coincs
        # update coinc masks: filter out 3 ifo coincs
        coinc2_mask12 = coinc2_mask12 & ~coinc3_mask
        coinc2_mask23 = coinc2_mask23 & ~coinc3_mask
        coinc2_mask31 = coinc2_mask31 & ~coinc3_mask

        network_snr12 = (
            (snr1 * coinc2_mask12) ** 2 + (snr2 * coinc2_mask12) ** 2
        ) ** 0.5
        network_snr23 = (
            (snr2 * coinc2_mask23) ** 2 + (snr3 * coinc2_mask23) ** 2
        ) ** 0.5
        network_snr31 = (
            (snr1 * coinc2_mask31) ** 2 + (snr3 * coinc2_mask31) ** 2
        ) ** 0.5

        # update coinc masks: there may be cases where a template has
        # two coincs, (e.g., HV coinc and LV coinc, but not HL coinc),
        # in this case, compare HV, LV coinc network snrs and choose
        # the larger one
        # FIXME: what to do when snrs are equal?
        coinc2_mask12 = (
            coinc2_mask12
            & (network_snr12 > network_snr23)
            & (network_snr12 >= network_snr31)
        )
        coinc2_mask23 = (
            coinc2_mask23
            & (network_snr23 >= network_snr12)
            & (network_snr23 > network_snr31)
        )
        coinc2_mask31 = (
            coinc2_mask31
            & (network_snr31 > network_snr12)
            & (network_snr31 >= network_snr23)
        )

        # update 2 ifo network snrs
        network_snr12 = (
            (snr1 * coinc2_mask12) ** 2 + (snr2 * coinc2_mask12) ** 2
        ) ** 0.5
        network_snr23 = (
            (snr2 * coinc2_mask23) ** 2 + (snr3 * coinc2_mask23) ** 2
        ) ** 0.5
        network_snr31 = (
            (snr1 * coinc2_mask31) ** 2 + (snr3 * coinc2_mask31) ** 2
        ) ** 0.5

        # 1 ifo
        # FIXME: what to do when snrs are equal?
        single_mask1 = (
            ~coinc3_mask
            & ~coinc2_mask12
            & ~coinc2_mask23
            & ~coinc2_mask31
            & (snr1 > snr2)
            & (snr1 >= snr3)
        )
        single_mask2 = (
            ~coinc3_mask
            & ~coinc2_mask12
            & ~coinc2_mask23
            & ~coinc2_mask31
            & (snr2 >= snr1)
            & (snr2 > snr3)
        )
        single_mask3 = (
            ~coinc3_mask
            & ~coinc2_mask12
            & ~coinc2_mask23
            & ~coinc2_mask31
            & (snr3 > snr1)
            & (snr3 >= snr2)
        )

        single_snr1 = snr1 * single_mask1
        single_snr2 = snr2 * single_mask2
        single_snr3 = snr3 * single_mask3

        all_network_snrs = (
            network_snr123
            + network_snr12
            + network_snr23
            + network_snr31
            + single_snr1
            + single_snr2
            + single_snr3
        )

        return (
            coinc3_mask,
            coinc2_mask12,
            coinc2_mask23,
            coinc2_mask31,
            single_mask1,
            single_mask2,
            single_mask3,
            all_network_snrs,
        )

    def coinc2(self, snrs, times, ifos):
        dt = Offset.fromsec(light_travel_time(*ifos))
        snr1 = snrs[0]
        snr2 = snrs[1]
        time1 = times[0]
        time2 = times[1]
        coinc_mask = abs(time1 - time2) < dt
        single_mask1 = (snr1 > snr2) & ~coinc_mask
        single_mask2 = ~single_mask1 & ~coinc_mask

        snr_masked1 = snr1 * coinc_mask
        snr_masked2 = snr2 * coinc_mask
        coinc_network_snr = (snr_masked1**2 + snr_masked2**2) ** 0.5

        single1 = snr1 * single_mask1
        single2 = snr2 * single_mask2

        all_network_snr = coinc_network_snr + single1 + single2

        return (
            coinc_mask,
            single_mask1,
            single_mask2,
            all_network_snr,
        )

    def cluster_coincs(self, ifo_combs, all_network_snr, template_ids, triggers, snrs):
        clustered_snr, max_locations = torch.max(all_network_snr, dim=-1)
        clustered_ifo_combs = ifo_combs.gather(1,max_locations.unsqueeze(1)).squeeze()
        max_locations = max_locations.to("cpu").numpy()
        clustered_template_ids = template_ids[range(self.nsubbank), max_locations]
        sngls = {}
        for ifo, trig in triggers.items():
            sngls[ifo] = {}
            peak_locations = triggers[ifo][0][range(self.nsubbank), max_locations]
            sngl_snr = triggers[ifo][1][range(self.nsubbank), max_locations]
            sngl_chisq = triggers[ifo][2][range(self.nsubbank), max_locations]

            sngls[ifo]["time"] = (
                np.round(
                    (Offset.fromsamples(peak_locations, self.rate) + self.offset)
                    / Offset.OFFSET_RATE
                    * 1_000_000_000
                ).astype(int)
                + Offset.offset_ref_t0
                + self.end_times
            )
            sngls[ifo]["snr"] = sngl_snr
            sngls[ifo]["chisq"] = sngl_chisq

            # go back and find the phase only for the clustered coincs
            # FIXME: find the snr snippet
            snrs0 = snrs[ifo]
            snrs0 = snrs0.view(snrs0.shape[0], snrs0.shape[1] // 2, 2, snrs0.shape[2])
            snr_pairs = snrs0[range(snrs0.shape[0]), max_locations]
            sngl_peaks = snr_pairs[range(snr_pairs.shape[0]), :, peak_locations]
            real = sngl_peaks[:, 0]
            imag = sngl_peaks[:, 1]
            phase = torch.atan2(imag, real)
            sngls[ifo]["phase"] = phase.to("cpu").numpy()

        # FIXME: is stacking then index_select faster?
        # FIXME: is stacking then copying to cpu faster?
        return [clustered_template_ids, clustered_ifo_combs, clustered_snr, sngls]

    #@torch.compile
    def itacacac(self, snrs):
        triggers = self.find_peaks_and_calculate_chisqs(snrs)

        # FIXME: consider edge effects
        ifo_combs, all_network_snr, single_masks = self.make_coincs(triggers)

        # FIXME: this part and clustered_coinc is lowering the GPU utilization
        for ifo in triggers.keys():
            for i in range(len(triggers[ifo])):
                triggers[ifo][i] = triggers[ifo][i].to("cpu").numpy()

        clustered_coinc = self.cluster_coincs(
            ifo_combs, all_network_snr, self.template_ids_np, triggers, snrs
        )

        return triggers, ifo_combs, all_network_snr, single_masks, clustered_coinc

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
        self.rate = frame.sample_rate
        self.offset = frame.offset

        metadata = frame.metadata
        if len(snrs.keys()) >= 1:

            triggers, ifo_combs, all_network_snr, single_masks, clustered_coinc = (
                self.itacacac(snrs)
            )
            ifos = triggers.keys()

            if self.kafka:
                maxsnrs = {}
                maxlatency = {"time": None, "data": None}
                mincoinc_time = min(float(np.min(clustered_coinc[3][ifo]["time"])) for ifo in ifos)
                for ifo in ifos:
                    snrs = triggers[ifo][1]
                    maxsnr_id = np.unravel_index(np.argmax(snrs), snrs.shape)
                    maxsnrs[ifo+"_snr_history"] = {"time":
                                [(np.round(
                                    (Offset.fromsamples(triggers[ifo][0][maxsnr_id], self.rate) + self.offset)
                                    / Offset.OFFSET_RATE
                                    * 1_000_000_000
                                ).astype(int)
                                + Offset.offset_ref_t0
                                + self.end_times[maxsnr_id[0]])/1_000_000_000,]
                            , "data": [triggers[ifo][1][maxsnr_id].item(),]}
                maxlatency["time"] = [mincoinc_time/1_000_000_000]
                maxlatency["data"] = [(now().ns() - mincoinc_time)/1_000_000_000]

            for j in range(1, len(clustered_coinc) - 1):
                clustered_coinc[j] = clustered_coinc[j].to("cpu").numpy()

            # Populate background snr, chisq, time for each bank, ifo
            # FIXME: is stacking then copying to cpu faster?
            # FIXME: do we only need snr chisq for singles?
            background = {bankid: {ifo: None} for bankid, ids in self.bankids_map.items() for ifo in ifos}
            # loop over banks
            for bankid, ids in self.bankids_map.items():
                # loop over ifos
                for ifo in ifos:
                    times = []
                    snrs = []
                    chisqs = []
                    template_ids = []

                    if ifo in single_masks:
                        if True in single_masks[ifo]:
                            smask0 = single_masks[ifo].to("cpu").numpy()
                            # loop over subbank ids in this bank
                            for i in ids:
                                smask = smask0[i]
                                time = triggers[ifo][0][i][smask]
                                time = (
                                    np.round(
                                        (Offset.fromsamples(time, self.rate) + self.offset)
                                        / Offset.OFFSET_RATE
                                        * 1_000_000_000
                                    ).astype(int)
                                    + Offset.offset_ref_t0
                                    + self.end_times[i]
                                )
                                times.append(time)
                                snrs.append(triggers[ifo][1][i][smask])
                                chisqs.append(triggers[ifo][2][i][smask])
                                template_ids.append(self.template_ids_np[i][smask])

                    background[bankid][ifo] = {"time": times, "snrs": snrs, "chisqs": chisqs, "template_ids": template_ids}

            metadata["coincs"] = {
                "template_ids": clustered_coinc[0],
                "ifo_combs": clustered_coinc[1],
                "network_snrs": clustered_coinc[2],
                "ifos_number_map": self.ifos_number_map,
                "time": None,  # FIXME: how is time defined?
                "sngl": clustered_coinc[3],
            }
            metadata["background"] = background
            if self.kafka:
                metadata["kafka"] = maxsnrs
                metadata["kafka"]["latency_history"] = maxlatency

        outbuf = SeriesBuffer(
            offset=self.preparedoutoffsets[self.sink_pads[0]][0]["offset"],
            sample_rate=frame.sample_rate,
            data=None,
            shape=(
                self.nsubbank,
                Offset.tosamples(
                    self.preparedoutoffsets[self.sink_pads[0]][0]["noffset"],
                    frame.sample_rate,
                ),
            ),
        )

        return TSFrame(buffers=[outbuf], EOS=frame.EOS, metadata=metadata)
