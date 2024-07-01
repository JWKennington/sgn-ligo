"""
Sort svd banks and time slices by same rate
"""

from collections import defaultdict, Counter
from lal.utils import CacheEntry
import itertools
import operator
import sys
from .svd_bank import parse_bank_files

import numpy as np

import torch
import torch.utils.benchmark as benchmark

from ..base import Offset
from sgnts.transforms.resampler import UP_HALF_LENGTH, DOWN_HALF_LENGTH


def group_and_read_banks(svd_bank, nbank_pretend=0, nslice=-1, verbose=False):
    """
    Read a list of svd banks file names into bank objects
    """
    # Read SVD banks
    svd_bank_cache = [CacheEntry.from_T050017(path) for path in svd_bank]
    svd_bank_cache.sort(key=lambda cache_entry: cache_entry.description)
    svd_banks = []

    for key, seq in itertools.groupby(
        svd_bank_cache, key=lambda cache_entry: cache_entry.description
    ):
        svd_group = dict(
            (cache_entry.observatory, cache_entry.url) for cache_entry in seq
        )
        svd_banks.append(svd_group)

    ifoset = set([tuple(s.keys()) for s in svd_banks])
    if len(ifoset) != 1:
        raise ValueError("The ifos have different sets of svd bank files provided.")

    ifos = list(ifoset)[0]

    banks = {ifo: [] for ifo in ifos}
    if nbank_pretend:
        # Pretend we are filtering multiple banks by copying the first bank
        #   many times
        svd_bank_url_dict = svd_banks[0]
        banks_per_svd = parse_bank_files(svd_bank_url_dict, verbose=verbose)
        for i in range(nbank_pretend):
            for ifo in ifos:
                banks[ifo].extend([banks_per_svd[ifo][0]])
    else:
        for svd_bank_url_dict in svd_banks:
            if verbose:
                print(svd_bank_url_dict, flush=True, file=sys.stderr)
            banks_per_svd = parse_bank_files(svd_bank_url_dict, verbose=verbose)
            for ifo in ifos:
                banks[ifo].extend(banks_per_svd[ifo])

    # TODO: fix nslice reading
    if nslice > 0:
        for ifo in ifos:
            bifo = banks[ifo]
            for sub in bifo:
                sub.bank_fragments = sub.bank_fragments[:nslice]

    if verbose:
        print("Using nsubbanks", len(banks[ifo]), flush=True, file=sys.stderr)

    return banks


class SortedBank:
    """
    Sort svd banks and time slices by same rate
    """

    def __init__(
        self,
        banks,
        copy_block,
        device="cpu",
        dtype=torch.float16,
        memory_format=torch.contiguous_format,
        nbank_pretend=0,
        nslice=-1,
        verbose=False,
    ):
        self.copy_block = copy_block
        self.device = device
        self.dtype = dtype
        self.memory_format = memory_format
        self.verbose = verbose

        temp = torch.empty(1, dtype=dtype)
        self.cdtype = torch.complex(temp, temp).dtype
        self.nbank_pretend = nbank_pretend

        self.bank_metadata, self.reordered_bank = self.prepare_metadata(banks)
        # Prepare tensors for LLOID methods
        (
            self.coeff_sv_cat,
            self.bases_cat,
            self.template_ids,
            self.end_times,
            self.bankids,
            self.bankids_map,
            self.sngls,
            self.autocorrelation_banks,
            self.processed_psd,
        ) = self.prepare_tensors(self.bank_metadata, self.reordered_bank)

    def prepare_metadata(self, bank):
        """
        Determine rates and template properties across subbanks

        bank_metadata.keys() = ['ifos', 'nifo', 'nbank', 'maxrate', 'unique_rates',
        'nrates', 'nfilter_samples', 'ntempmax', 'delay_per_rate', 'sorted_rates']

        Arguments:
        ----------
        bank:
            gstlal Bank class
        """
        bank_metadata = {}
        ifos = list(bank.keys())
        bank_metadata["ifos"] = ifos
        bank_metadata["nifo"] = len(ifos)

        # assume all ifos have same banks
        bank0 = bank[ifos[0]]

        nbank = len(bank0)  # number of subbanks in each ifo
        bank_metadata["nbank"] = nbank

        # determine some properties across banks
        # determine rates
        maxrate = bank0[0].sample_rate_max
        bank_metadata["maxrate"] = maxrate  # assume same for all banks
        unique_rates = dict(
            sorted(
                Counter([bi.rate for b in bank0 for bi in b.bank_fragments]).items(),
                reverse=True,
            )
        )
        bank_metadata["unique_rates"] = unique_rates
        bank_metadata["nrates"] = len(unique_rates)

        # number of samples in each filter, assume it's the same for all templates
        bank_metadata["nfilter_samples"] = (
            bank0[0].bank_fragments[0].orthogonal_template_bank.shape[1]
        )

        # maximum number of templates across all subbanks
        bank_metadata["ntempmax"] = max(
            sub.bank_fragments[0].mix_matrix.shape[1] for sub in bank0
        )
        delay_per_rate = dict.fromkeys(unique_rates.keys())
        for rate in unique_rates:
            delay_per_rate[rate] = max(
                int(bf.start * rate)
                for banki in bank.values()
                for sub in banki
                for bf in sub.bank_fragments
                if rate == bf.rate
            )
        bank_metadata["delay_per_rate"] = delay_per_rate

        sorted_rates, reordered_bank = self.sort_bank_by_rates(bank, bank_metadata)
        bank_metadata["sorted_rates"] = sorted_rates

        return bank_metadata, reordered_bank

    def sort_bank_by_rates(self, bank, bank_metadata):
        """
        Determine the ids to upsample from rate1 to rate2
        This places templates that need to be upsampled from and to
            the same rate next to each other, which allows the templates
            to be upsampled together and allows a simpler slicing method
        The following example shows id placement
        slice  | 0  | 1   |  2 |  3 |
        -----------------------------
        bank0: [2048, 1024, 512, 512]
        bank1: [2048, 1024, 512, 256]
        bank2: [2048,  512, 256]

        bases[sample_rate][upsample_rate]:

        bases[2048][()]
            = [bank0slice0, bank1slice0, bank2slice0]
        bases[1024][(2048,)]
            = [bank0slice1, bank1slice1]
        bases[512][(2048, 1024)]
            = [bank0slice2, bank0slice3, bank1slice2]
            # templates in the same bank will be placed next to each other,
                to allow more convenient summing in sum_same_rates()

        bases[512][(2048,)]
            = [bank2slice1]
        bases[256][(2048, 1024, 512)]
            = [bank1slice3]
        bases[256][(2048, 512)]
            = [bank2slice2]

        * The reason that bases[512][(2048, 1024)] is in a different group than
            bases[512][(2048,)] is because they will go into different upsamplers
        * The reason that bases[256][(2048, 1024, 512)] is in a different group
            than bases[256][(2048, 512)], even though they are upsampling to the
            same rate (512), is because due to the different upsampling path
            256->512->1024->2048 vs. 256->512->2048, the time stamps of the buffers
            will be different (to account for the upsampling padding)

        ----------

        sorted_rates = {
            from_rate: {
                to_rate: {
                    'ids_counter':,'upids_counter':,'addto_counter':,'addids':,
                    'counts':,'upcounts':,'nbmax':, 'ntempmax':,'sum_same_rate':,
                    'same_data':,'unique_segments':,'segments_map':,'uppad':,
                    'addslice':,'upslice':,'rescale':,'start':,
                    'metadata': {
                         bank_order: {
                             'ids':,'nslice':,'sum_same_rate':,
                         }
                    },
                }
            }
        }

        """

        ifos = bank_metadata["ifos"]
        bank0 = bank[ifos[0]]
        nbank = len(bank0)  # number of subbanks in each ifo
        unique_rates = bank_metadata["unique_rates"]
        maxrate = bank_metadata["maxrate"]

        # Sort the unique rates, this might alter the order of the template banks
        sorted_unique_rates = []
        for j in range(nbank):
            bk = bank0[j]
            bankid = bk.bank_id
            unique_rates0 = dict(
                sorted(
                    Counter([bi.rate for bi in bk.bank_fragments]).items(), reverse=True
                )
            )
            for key in unique_rates:
                if key not in unique_rates0:
                    unique_rates0[key] = 0
            unique_rates0["bankid"] = bankid
            sorted_unique_rates.append(unique_rates0)

        sorted_unique_rates = sorted(
            # sorted_unique_rates, key=lambda x: x.keys(), reverse=True
            sorted_unique_rates,
            key=operator.itemgetter(*list(unique_rates.keys())),
            reverse=True,
        )
        if self.verbose:
            print("sorted_unique_rates", flush=True, file=sys.stderr)
            for a in sorted_unique_rates:
                print(a, flush=True, file=sys.stderr)

        # check if nbank_pretend is true
        bankids = [a["bankid"] for a in sorted_unique_rates]
        nbank_pretend = False
        if len(bankids) > 1 and len(set(bankids)) == 1:
            if self.verbose:
                print("nbank_pretend", flush=True, file=sys.stderr)
            nbank_pretend = True

        # reorder bankid
        bankid_order = {}
        for i, a in enumerate(sorted_unique_rates):
            bankid = a["bankid"]
            bankid_order[bankid] = i

        reorder = {}
        # reordered bank
        if nbank_pretend is False:
            for ifo in ifos:
                reorder[ifo] = {}
                for j in range(nbank):
                    bk = bank[ifo][j]
                    bankid = bk.bank_id
                    order = bankid_order[bankid]
                    reorder[ifo][order] = bk
        else:
            for ifo in ifos:
                reorder[ifo] = {i: b for i, b in enumerate(bank[ifo])}

        #
        # Construct sorted_rates, determine id placement of timeslices
        #
        sorted_rates = defaultdict(dict)
        for from_rate in unique_rates:
            from_bank = sorted_rates[from_rate]
            for a in sorted_unique_rates:
                k = list(a.keys())
                if from_rate in k and a[from_rate] > 0:
                    to_rate = tuple(
                        ki
                        for ki in k
                        if type(ki) is int and ki > from_rate and a[ki] > 0
                    )
                    if to_rate not in from_bank:
                        from_bank[to_rate] = {
                            "ids_counter": 0,
                            "upids_counter": 0,
                            "addto_counter": 0,
                            "addids": [],
                            "counts": 0,
                            "upcounts": 0,
                            "metadata": {},
                            "nbmax": [],
                            "ntempmax": [],
                            "sum_same_rate": False,
                            "same_data": False,
                            "unique_segments": [],
                            "segments_map": [],
                            "sum_same_rate_slices": [],
                        }
                    if from_rate != maxrate:
                        to_bank = from_bank[to_rate]
                        to_bank["counts"] += a[from_rate]
                        to_bank["upcounts"] += 1
                        # shift starting point of ids counter,
                        # the slices that are added are placed
                        # after slices that are going to be upsampled
                        # to_bank["ids_counter"] += 1 # preallocate
                        to_bank["addids"].append(
                            sorted_rates[to_rate[-1]][to_rate[:-1]]["addto_counter"]
                        )
                        sorted_rates[to_rate[-1]][to_rate[:-1]]["addto_counter"] += 1

        sorted_rates[maxrate][()]["counts"] = nbank

        urates = list(unique_rates.keys())
        downpads = {r: None for r in urates}
        downpad = 0
        downpads[2048] = 0
        for urate in urates[1:]:
            downpad += Offset.fromsamples(DOWN_HALF_LENGTH, urate)
            downpads[urate] = downpad

        # loop over all subbanks and update metadata in each rate group
        for j in range(nbank):
            bk = reorder[ifos[0]][j]
            rates = [bf.rate for bf in bk.bank_fragments]
            # urates = np.array(sorted(set(rates), reverse=True))
            urates = list(sorted(set(rates), reverse=True))
            urs = urates[1:]
            # uppad = {
            #    #r0: sum(Offset.fromsamples(UP_HALF_LENGTH, ri) for ri in urs[urs >= r0])
            #    r0: sum([Offset.fromsamples(UP_HALF_LENGTH, ri) for ri in urs[urs >= r0]])
            #    for r0 in urs
            # }
            uppads = {r: None for r in urates}
            uppad = 0
            uppads[maxrate] = 0
            for ri in urs:
                uppad += Offset.fromsamples(UP_HALF_LENGTH, ri)
                uppads[ri] = uppad

            for bi, bf in enumerate(bk.bank_fragments):
                rate = bf.rate
                to_rate = tuple(r for r in urates if r > rate)
                rb = sorted_rates[rate][to_rate]
                if j not in rb["metadata"]:
                    rb["metadata"][j] = {
                        # "starts": [],
                        # "ends": [],
                        "ids": [],
                        "nslice": [],
                        "sum_same_rate": False,
                    }
                mdata = rb["metadata"][j]
                mdata["bankid"] = bk.bank_id
                # mdata["starts"].append(bf.start)
                # mdata["ends"].append(bf.end)
                mdata["nslice"].append(bi)
                # rb["uppad"] = uppad[rate]
                rb["shift"] = uppads[rate] + downpads[rate]
                rb["segments_map"].append((bf.start, bf.end))
                if rate != maxrate:
                    if rate == rates[bi - 1]:
                        mdata["ids"].append(rb["ids_counter"])
                        rb["ids_counter"] += 1
                        if len(mdata["ids"]) > 1:
                            rb["sum_same_rate"] = True
                            mdata["sum_same_rate"] = True
                    else:
                        # mdata["ids"].append(rb["upids_counter"]) # preallocate
                        # rb["upids_counter"] += 1 # preallocate
                        mdata["ids"].append(rb["ids_counter"])
                        rb["ids_counter"] += 1
                else:
                    mdata["ids"] = [j]

        for ifo in ifos:
            for j in range(nbank):
                bk = reorder[ifo][j]
                rates = [bf.rate for bf in bk.bank_fragments]
                for bf in bk.bank_fragments:
                    rate = bf.rate
                    urates = np.array(sorted(set(rates), reverse=True))
                    to_rate = tuple(r for r in urates if r > rate)
                    rb = sorted_rates[rate][to_rate]
                    rb["nbmax"].append(bf.mix_matrix.shape[0])
                    rb["ntempmax"].append(bf.mix_matrix.shape[1])

        for from_rate, v in sorted_rates.items():
            for to_rate, v2 in v.items():
                if from_rate != maxrate:
                    addids = v2["addids"]
                    v2["addslice"] = slice(addids[0], addids[-1] + 1)
                    v2["upslice"] = slice(0, v2["upcounts"])
                    v2["rescale"] = (to_rate[-1] / from_rate) ** 0.5
                v2["nbmax"] = max(v2["nbmax"])
                v2["ntempmax"] = max(v2["ntempmax"])
                # starts = [s for b in v2["metadata"].values() for s in b["starts"]]
                # sst = set(starts)
                # if len(sst) == 1:
                #    v2["same_data"] = True
                #    v2["start"] = list(sst)[0]
                v2["unique_segments"] = sorted(set([seg for seg in v2["segments_map"]]))
                unique_segments2 = sorted(set(v2["segments_map"]))
                assert v2["unique_segments"] == unique_segments2
                mdata = v2["metadata"]
                if v2["sum_same_rate"] is True:
                    for md in mdata.values():
                        ids = md["ids"]
                        v2["sum_same_rate_slices"].append(slice(ids[0], ids[-1] + 1))

        return sorted_rates, reorder

    def prepare_tensors(self, bank_metadata, reordered_bank):
        """
        Prepare large tensors to store input and output of methods in LLOID
        coeff_sv:      all the coeff_sv from different banks
        bases:         all the bases from different banks
        """
        print(
            "Preparing tensors for LLOID methods...",
            end="",
            flush=True,
            file=sys.stderr,
        )

        dtype = self.dtype
        device = self.device
        copy_block = self.copy_block

        nifo = bank_metadata["nifo"]
        ifos = bank_metadata["ifos"]
        nfilter_samples = bank_metadata["nfilter_samples"]
        nbank = bank_metadata["nbank"]
        sorted_rates = bank_metadata["sorted_rates"]

        # outputs
        data_by_rate = defaultdict(dict)
        bases_by_rate = defaultdict(dict)
        coeff_sv_by_rate = defaultdict(dict)

        # construct big tensors of data, bases, and coeff_sv, grouped by sample rates
        for from_rate, rbr in sorted_rates.items():
            for to_rate, rb in rbr.items():
                count = rb["counts"]
                mdata = rb["metadata"]
                nbm = rb["nbmax"]
                ntempmax = rb["ntempmax"]
                same_data = rb["same_data"]

                # data
                if same_data:
                    data_by_rate[from_rate][to_rate] = torch.zeros(
                        size=(nifo, nfilter_samples - 1 + int(from_rate * copy_block)),
                        device=device,
                        dtype=dtype,
                    )
                else:
                    data_by_rate[from_rate][to_rate] = torch.zeros(
                        size=(
                            nifo * count,
                            nfilter_samples - 1 + int(from_rate * copy_block),
                        ),
                        device=device,
                        dtype=dtype,
                    )

                # group the bases by sample rate
                bases_by_rate[from_rate][to_rate] = torch.zeros(
                    size=(nifo, count, nbm, nfilter_samples),
                    device=device,
                    dtype=dtype,
                )

                # group the coeff by sample rate
                coeff_sv_by_rate[from_rate][to_rate] = torch.zeros(
                    size=(nifo, count, ntempmax, nbm),
                    device=device,
                    dtype=dtype,
                )

                # fill in the bases and coeff tensors!
                for k, ifo in enumerate(ifos):
                    bifo = reordered_bank[ifo]
                    for bank_order, md in mdata.items():
                        bifo_order = bifo[bank_order]
                        ids = md["ids"]
                        nslices = md["nslice"]
                        for id0, nslice in zip(ids, nslices):
                            this_slice = bifo_order.bank_fragments[nslice]
                            assert this_slice.rate == from_rate
                            b = this_slice.orthogonal_template_bank
                            c = this_slice.mix_matrix.T
                            bases_by_rate[from_rate][to_rate][
                                k, id0, : b.shape[0], :
                            ] = torch.tensor(b, device=device, dtype=dtype)
                            coeff_sv_by_rate[from_rate][to_rate][
                                k, id0, : c.shape[0], : c.shape[1]
                            ] = torch.tensor(c, device=device, dtype=dtype)
                bases = bases_by_rate[from_rate][to_rate].view(
                    -1,
                    nfilter_samples,
                )
                bases = (
                    bases.unsqueeze(1).unsqueeze(0).to(memory_format=self.memory_format)
                )
                bases_by_rate[from_rate][to_rate] = bases.view(
                    nifo, count, nbm, nfilter_samples
                )
                # benchmark conv methods
                rb["conv_group"] = True
                """
                if self.device == "cpu":
                    rb["conv_group"] = True
                else:
                    if self.verbose:
                        print("Benchmarking conv...", flush=True, file=sys.stderr)
                    tgroup = benchmark.Timer(
                        stmt="SNRSlices.conv_group(mask, data, nbm, basesr)",
                        setup="from greg.filtering.snr_slices import SNRSlices",
                        globals={
                            "mask": [True] * nifo,
                            "data": data_by_rate[from_rate][to_rate],
                            "nbm": nbm,
                            "basesr": bases_by_rate[from_rate][to_rate],
                        },
                    )

                    tgroupm = tgroup.timeit(500).median

                    tloop = benchmark.Timer(
                        stmt="SNRSlices.conv_loop(mask, data, nbm, basesr)",
                        setup="from greg.filtering.snr_slices import SNRSlices",
                        globals={
                            "mask": [True] * nifo,
                            "data": data_by_rate[from_rate][to_rate],
                            "nbm": nbm,
                            "basesr": bases_by_rate[from_rate][to_rate],
                        },
                    )

                    tloopm = tloop.timeit(500).median

                    if self.verbose:
                        print(
                            f"{from_rate=} {to_rate=}",
                            "conv_group",
                            tgroupm,
                            "conv_loop",
                            tloopm,
                            flush=True,
                            file=sys.stderr,
                        )
                    if tgroupm < tloopm:
                        rb["conv_group"] = True
                    else:
                        rb["conv_group"] = False
                """

        # Get template ids
        #   Assume same ids for all ifos for the same bank
        #   Init template ids as -1 for banks with ntemp < ntempmax,
        #   the template id for empty entries will be -1
        ntempmax = bank_metadata["ntempmax"]
        template_ids = torch.ones(size=(nbank, ntempmax // 2), dtype=torch.int32) * -1
        bankids = []
        sngls = []
        end_times = torch.zeros(size=(nbank,), dtype=torch.long)
        bankids_map = defaultdict(list)
        for j in range(nbank):
            sngl = reordered_bank[ifos[0]][j].sngl_inspiral_table
            template_ids0 = torch.tensor([row.template_id for row in sngl])
            template_ids[j, : template_ids0.shape[0]] = template_ids0

            ends0 = [row.end.ns() for row in sngl]
            assert len(set(ends0)) == 1, "there are different end times in a subbank"
            end_times[j] = list(set(ends0))[0]

            subbank_id = reordered_bank[ifos[0]][j].bank_id
            if self.nbank_pretend:
                bank_id = subbank_id.split("_")[0] + "_" + str(j)
            else:
                bank_id = subbank_id.split("_")[0]
            bankids.append(subbank_id)
            bankids_map[bank_id].append(j)
            sngls.append(sngl)

        # Write out single inspiral table
        # sngl0 = reordered_bank[ifos[0]][0].sngl_inspiral_table
        # row = sngl0[0]
        # keys = [a for a in dir(row) if not a.startswith('__') and not
        #           callable(getattr(row, a))]
        # import h5py
        # with h5py.File('sngl_inspiral_table.h5', "a") as f:
        #    for j in range(nbank):
        #        sngl = reordered_bank[ifos[0]][j].sngl_inspiral_table
        #        for row in sngl:
        #            group = str(row.template_id)
        #            f.create_group(group)
        #            for k in keys:
        #                v = getattr(row, k)
        #                if k == 'end':
        #                    v = float(v)
        #                f[group][k] = v

        # Trigger generator
        # Get the autocorrelation_bank
        max_acl = max(
            reordered_bank[ifo][j].autocorrelation_bank.shape[1]
            for i, ifo in enumerate(ifos)
            for j in range(nbank)
        )
        autocorrelation_banks = torch.zeros(
            size=(nifo, nbank, ntempmax // 2, max_acl), device=device, dtype=self.cdtype
        )
        for i, ifo in enumerate(ifos):
            for j in range(nbank):
                acorr = reordered_bank[ifo][j].autocorrelation_bank

                # this is for adjusting to the bank used for impulse test
                if acorr.shape[0] > ntempmax // 2:
                    acorr = acorr[: ntempmax // 2]
                autocorrelation_banks[i, j, : acorr.shape[0], : acorr.shape[1]] = (
                    torch.tensor(acorr)
                )

        processed_psd = dict(
            [(ifo, b[0].processed_psd) for ifo, b in reordered_bank.items()]
        )

        print(" Done.", flush=True, file=sys.stderr)

        return (
            coeff_sv_by_rate,
            bases_by_rate,
            template_ids,
            end_times,
            bankids,
            bankids_map,
            sngls,
            autocorrelation_banks,
            processed_psd,
        )
