import time
from collections.abc import Sequence
from typing import Any

from lal import UTCToGPS
from ligo.lw import ligolw, lsctables, utils
from strike.stats import likelihood_ratio

from sgn.sinks import *
from sgnts.sinks import *


class DefaultContentHandler(lsctables.ligolw.LIGOLWContentHandler):
    pass


lsctables.use_in(DefaultContentHandler)


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
    all_template_ids: Sequence[Any] = None
    bankids_map: dict[str, list] = None
    subbankids: Sequence[Any] = None
    template_sngls: list[dict] = None
    verbose: bool = False
    trigger_output: list[str] = None
    ranking_stat_output: list[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.sink_pads}
        assert isinstance(self.ifos, list)
        self.ranking_stats = {}
        sngl_inspiral_columns = (
            "process:process_id",
            "ifo",
            "end_time",
            "end_time_ns",
            "eff_distance",
            "coa_phase",
            "mass1",
            "mass2",
            "snr",
            "chisq",
            "chisq_dof",
            "bank_chisq",
            "bank_chisq_dof",
            "sigmasq",
            "spin1x",
            "spin1y",
            "spin1z",
            "spin2x",
            "spin2y",
            "spin2z",
            "template_duration",
            "event_id",
            "Gamma0",
            "Gamma1",
            "Gamma2",
        )
        coinc_inspiral_columns = (
            "coinc_event:coinc_event_id",
            "combined_far",
            "end_time",
            "end_time_ns",
            "false_alarm_rate",
            "ifos",
            "mass",
            "mchirp",
            "minimum_duration",
            "snr",
        )
        coinc_event_map_columns = (
            "coinc_event:coinc_event_id",
            "event_id",
            "table_name",
        )
        self.sngl_tables = {}
        self.coinc_tables = {}
        self.coinc_event_map_tables = {}
        self.coinc_outdocs = {}
        for bankid, ids in self.bankids_map.items():
            bank_template_ids = self.all_template_ids[ids]
            bank_template_ids = tuple(bank_template_ids[bank_template_ids != -1])
            # Ranking stat output
            self.ranking_stats[bankid] = likelihood_ratio.LnLikelihoodRatio(
                template_ids=bank_template_ids,
                instruments=self.ifos,
            )

            # Coincs Document
            self.sngl_tables[bankid] = lsctables.New(
                lsctables.SnglInspiralTable, columns=sngl_inspiral_columns
            )
            self.coinc_tables[bankid] = lsctables.New(
                lsctables.CoincInspiralTable, columns=coinc_inspiral_columns
            )
            self.coinc_event_map_tables[bankid] = lsctables.New(
                lsctables.CoincMapTable, columns=coinc_event_map_columns
            )
            self.coinc_outdocs[bankid] = ligolw.Document()
            self.coinc_outdocs[bankid].appendChild(ligolw.LIGO_LW())

            # add sngl_inspiral table to output XML document
            self.coinc_outdocs[bankid].childNodes[0].appendChild(
                self.sngl_tables[bankid]
            )
            self.coinc_outdocs[bankid].childNodes[0].appendChild(
                self.coinc_event_map_tables[bankid]
            )
            self.coinc_outdocs[bankid].childNodes[0].appendChild(
                self.coinc_tables[bankid]
            )

        self.coinc_event_id_counter = 0
        self.event_id_counter = 0

    def pull(self, pad, bufs):
        self.cnt[pad] += 1
        if bufs.EOS:
            self.mark_eos(pad)

            for i, bankid in enumerate(self.bankids_map):
                # FIXME correct file name assignment
                # write ranking stats file
                self.ranking_stats[bankid].save(self.ranking_stat_output[i])

                # write coincs file
                output_file = self.trigger_output[i]
                utils.write_filename(self.coinc_outdocs[bankid], output_file)

        if self.verbose is True:
            print(self.cnt[pad], bufs)

        metadata = bufs.metadata
        if "background" in metadata:
            background = metadata["background"]
            #
            # Background triggers
            #
            # form events
            for ifo in self.ifos:
                for bankid in self.bankids_map:
                    if ifo in background[bankid]:
                        trigs = background[bankid][ifo]
                        bgtime = trigs["time"]
                        snr = trigs["snrs"]
                        chisq = trigs["chisqs"]
                        template_id = trigs["template_ids"]

                        # loop over subbanks
                        for time0, snr0, chisq0, templateid0 in zip(
                            bgtime, snr, chisq, template_id
                        ):
                            # loop over triggers in subbanks
                            for t, s, c, tid in zip(time0, snr0, chisq0, templateid0):
                                # FIXME: is end time in seconds??
                                event = event_dummy(
                                    ifo=ifo,
                                    end=t/1_000_000_000,
                                    snr=s,
                                    chisq=c,
                                    combochisq=c,
                                    template_id=tid,
                                )
                                self.ranking_stats[bankid].train_noise(event)
            #
            # Coinc triggers
            #

            ifo_combs = metadata["coincs"]["ifo_combs"]
            ifos_number_map = metadata["coincs"]["ifos_number_map"]
            reverse_map = {v: k for k, v in ifos_number_map.items()}
            sngl = metadata["coincs"]["sngl"]
            template_ids = metadata["coincs"]["template_ids"]
            network_snrs = metadata["coincs"]["network_snrs"]

            # loop over svd banks
            for bankid, ids in self.bankids_map.items():
                # loop over subbanks in this svd bank
                for i in ids:
                    ifo_comb = ifo_combs[i]
                    ifo_str = ""
                    coinc_end = None
                    template_id = template_ids[i]
                    template_row = self.template_sngls[i][template_id]

                    # loop over ifos
                    for j in str(ifo_comb):
                        ifo = reverse_map[int(j)]
                        if ifo_str == "":
                            ifo_str += ifo
                        else:
                            ifo_str += "," + ifo

                        single = sngl[ifo]
                        sngl_row = lsctables.SnglInspiral()
                        sngl_row.ifo = ifo

                        snr = sngl[ifo]["snr"][i]
                        sngl_row.snr = snr

                        chisq = sngl[ifo]["chisq"][i]
                        sngl_row.chisq = chisq
                        sngl_row.coa_phase = sngl[ifo]["phase"][i]

                        sngl_row.Gamma0 = template_id
                        sngl_row.Gamma1 = int(self.subbankids[i].split("_")[0])

                        # FIXME calculate a chisq weighted SNR and store it in the Gamma2 column
                        sngl_row.Gamma2 = snr / ((1 + max(1.0, chisq) ** 3) / 2.0) ** (
                            1.0 / 5.0
                        )
                        sngl_row.bank_chisq = 0
                        sngl_row.bank_chisq_dof = 0
                        sngl_row.chisq_dof = 1

                        end = sngl[ifo]["time"][i]
                        # coinc end time is the time of the first ifo in alphabetical order
                        if coinc_end is None:
                            coinc_end = end

                        sngl_row.end_time = end // Time.SECONDS
                        sngl_row.end_time_ns = end % Time.SECONDS
                        sngl_row.eta = template_row.eta
                        sngl_row.f_final = template_row.f_final
                        sngl_row.ifo = ifo
                        sngl_row.mass1 = template_row.mass1
                        sngl_row.mass2 = template_row.mass2
                        sngl_row.mchirp = template_row.mchirp
                        sngl_row.mtotal = template_row.mtotal
                        sngl_row.search = template_row.search
                        # sngl.search = "signal_model"# signal_model?
                        sngl_row.sigmasq = template_row.sigmasq
                        sngl_row.spin1 = template_row.spin1
                        sngl_row.spin2 = template_row.spin2
                        sngl_row.tau0 = template_row.tau0
                        sngl_row.tau3 = template_row.tau3
                        sngl_row.template_duration = template_row.template_duration
                        sngl_row.template_id = template_id
                        sngl_row.process_id = 0
                        sngl_row.eff_distance = float('nan')
                        sngl_row.event_id = self.event_id_counter
                        self.event_id_counter += 1
                        self.sngl_tables[bankid].append(sngl_row)

                        # coincs event map
                        coinc_map_row = lsctables.CoincMap()
                        coinc_map_row.coinc_event_id = self.coinc_event_id_counter
                        coinc_map_row.event_id = sngl_row.event_id
                        coinc_map_row.table_name = "sngl_inspiral"
                        self.coinc_event_map_tables[bankid].append(coinc_map_row)

                    # coincs inspiral
                    coinc_row = lsctables.CoincInspiral()
                    coinc_row.coinc_event_id = self.coinc_event_id_counter
                    self.coinc_event_id_counter += 1
                    coinc_row.end_time = coinc_end // Time.SECONDS
                    coinc_row.end_time_ns = coinc_end % Time.SECONDS
                    coinc_row.ifos = ifo_str
                    coinc_row.mass = template_row.mass1 + template_row.mass2
                    coinc_row.mchirp = template_row.mchirp
                    coinc_row.minimum_duration = (
                        float(UTCToGPS(time.gmtime())) - coinc_row.end
                    )
                    coinc_row.snr = network_snrs[i]
                    coinc_row.combined_far = None
                    coinc_row.false_alarm_rate = None
                    self.coinc_tables[bankid].append(coinc_row)

    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())
