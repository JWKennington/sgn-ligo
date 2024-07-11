from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sgn.sinks import *
from sgnts.sinks import *
from sgnts.base import Offset
from ..base import ArrayOps
import h5py
import torch
from strike.stats import likelihood_ratio

@dataclass
class event_dummy(object):
    ifo: list[str]
    end: float
    snr: float
    chisq: float
    combochisq: float
    template_id: int


@dataclass
class StrikeSink(SinkElement):

    ifos: list[str] = None
    template_ids: Sequence[Any] = None
    bankids_map: dict[str, list] = None
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.sink_pads}
        assert isinstance(self.ifos, list)
        self.rs = {}
        for bankid in self.bankids_map:
            self.rs[bankid] = likelihood_ratio.LnLikelihoodRatio(
                template_ids=self.template_ids,
                instruments=self.ifos,
                )

    def pull(self, pad, bufs):
        self.cnt[pad] += 1
        if bufs.EOS:
            self.mark_eos(pad)
            for bankid in self.bankids_map:
                self.rs[bankid].save(''.join(self.ifos)+"-"+"{:04d}".format(int(bankid))+"_LnLikelihoodRatio_dummy.xml.gz")
        if self.verbose is True:
            print(self.cnt[pad], bufs.metadata)

        metadata = bufs.metadata
        if "background" in metadata:
            background = metadata["background"]
            # form events
            for ifo in self.ifos:
                for bankid in self.bankids_map:
                    if ifo in background[bankid]:
                        trigs = background[bankid][ifo]
                        time = trigs["time"]
                        snr = trigs["snrs"]
                        chisq = trigs["chisqs"]
                        template_id = trigs["template_ids"]

                        # loop over subbanks
                        for time0, snr0, chisq0, templateid0 in zip(time, snr, chisq, template_id):
                            # loop over triggers in subbanks
                            for t, s, c, tid in zip(time0, snr0, chisq0, templateid0):
                                event = event_dummy(ifo=ifo, end=t, snr=s, chisq=c, combochisq=c, template_id=tid)
                                self.rs[bankid].train_noise(event)

            #ifo_combs = metadata["coincs"]["ifo_combs"]
            #ifos_number_map = metadata["coincs"]["ifos_number_map"]
            #reverse_map = {v: k for k, v in ifos_number_map.items()}
            #sngl = metadata["coincs"]["sngl"]
            #template_ids = metadata["coincs"]["template_ids"]

            #for i, ifo_comb in enumerate(ifo_combs):
            #    for j in str(ifo_comb):
            #        ifo = reverse_map[int(j)]
            #        single = sngl[ifo]
            #        event = event_dummy(ifo=ifo, end=single["time"][i], snr=single["snr"][i], chisq=single["chisq"][i], combochisq=single["chisq"][i], template_id=template_ids[i])

            #        self.rs.train_noise(event)



    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())

