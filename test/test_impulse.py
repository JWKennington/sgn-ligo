from sgnligo.cbc.sort_bank import group_and_read_banks, SortedBank
import torch

from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink
from sgnts.base import AdapterConfig, Offset

from sgnligo.sinks import ImpulseSink

from sgnligo.transforms import (
    Converter,
    TorchResampler,
    LLOIDCorrelate,
    TorchMatmul,
    SumIndex,
    Adder,
    Align,
)
from sgnligo import math

# Read in the svd banks!
svd_bank = [
    "H1-0250_GSTLAL_SVD_BANK_half-0-0.xml.gz",
]
original_templates = "full_templates_bin0250_tol999_1024.hdf5"
nbank_pretend = 0
nslice = -1
verbose = True

copy_block = 1
device = "cpu"
dtype = torch.float32
verbose = False

banks = group_and_read_banks(
    svd_bank=svd_bank, nbank_pretend=nbank_pretend, nslice=nslice, verbose=verbose
)

sorted_bank = SortedBank(
    banks=banks,
    copy_block=copy_block,
    device=device,
    dtype=dtype,
    nbank_pretend=nbank_pretend,
    nslice=nslice,
    verbose=verbose,
)
bases = sorted_bank.bases_cat
coeff = sorted_bank.coeff_sv_cat

bank_metadata = sorted_bank.bank_metadata
unique_rates = list(bank_metadata["unique_rates"].keys())
maxrate = bank_metadata["maxrate"]

num_samples = 2048

pipeline = Pipeline()

# Build pipeline
pipeline.insert(
    FakeSeriesSrc(
        name="src1",
        source_pad_names=("H1","H1im"),
        num_buffers=143,
        rate=2048,
        num_samples=2048,
        signal_type="impulse",
        impulse_position=4 * 2048 - 64,
        verbose=verbose,
    ),
    Converter(
        name="converter1",
        sink_pad_names=("H1",),
        source_pad_names=(
            "H1down",
            "H1shift",
        ),
        adapter_config=AdapterConfig(
            stride=num_samples
        ),
        backend="torch",
        dtype="float32",
        device=device,
    ),
    link_map={
        "converter1:sink:H1": "src1:src:H1",
        }
    )
prev_source_pad = "converter1:src:H1down"

# Multi-band
sorted_rates = bank_metadata["sorted_rates"]
multiband_source_pad_names = {r: {} for r in unique_rates}
multiband_source_pad_names[maxrate][()] = "converter1:src:H1shift"
for i, rate in enumerate(unique_rates[:-1]):
    rate_down = unique_rates[i+1]
    name=f"down_SR{rate}_SR{rate_down}"
    sink_pad = "H1"
    sink_pad_full = name+":sink:"+sink_pad

    source_pad = "H1down"
    source_pad_full = name+":src:"+source_pad

    to_rates = sorted_rates[rate_down].keys()
    source_pads = [source_pad]
    for to_rate in to_rates:
        source_pad = "H1shift_"+str(to_rate)
        source_pads.append(source_pad)
        multiband_source_pad_names[rate_down][to_rate] = name+":src:"+source_pad

    pipeline.insert(
        TorchResampler(
            name=name,
            sink_pad_names=(sink_pad,),
            source_pad_names=tuple(source_pads),
            dtype=dtype,
            device=device,
            adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
            inrate=rate,
            outrate=rate_down,
        ),
        link_map={sink_pad_full: prev_source_pad}
    )
    prev_source_pad = source_pad_full

# time segment shift
nfilter_samples = bank_metadata["nfilter_samples"]

for from_rate in reversed(unique_rates):
    for to_rate, rate_group in sorted_rates[from_rate].items():
        segments = rate_group["segments_map"]
        shift=rate_group["shift"]
        uppad=rate_group["uppad"]
        downpad=rate_group["downpad"]
        delays =[]
        for segment in segments:
            delays.append(Offset.fromsec(segment[0]))

        # Correlate
        corrname=f"corr_{from_rate}_{to_rate}"
        pipeline.insert(
            LLOIDCorrelate(
                name=corrname,
                sink_pad_names=("H1",),
                source_pad_names=("H1",),
                filters=bases[from_rate][to_rate],
                lib=math,
                uppad=uppad,
                downpad=downpad,
                delays=delays,
            ),
            link_map = {corrname+":sink:H1": multiband_source_pad_names[from_rate][to_rate]},
        )

        # matmul
        mmname=f"mm_{from_rate}_{to_rate}"
        pipeline.insert(
            TorchMatmul(
                name=mmname,
                sink_pad_names=("H1",),
                source_pad_names=("H1",),
                matrix=coeff[from_rate][to_rate],
            ),
            link_map = {mmname+":sink:H1": corrname+":src:H1"},
        )

        # sum same rate
        sumname = None
        if rate_group["sum_same_rate"] is True:
            sl = rate_group["sum_same_rate_slices"]
            sumname=f"sumindex_{from_rate}_{to_rate}"
            pipeline.insert(
                SumIndex(
                    name=sumname,
                    sink_pad_names=("H1",),
                    source_pad_names=("H1",),
                    sl=sl,
                ),
                link_map = {sumname+":sink:H1": mmname+":src:H1"}
            )

        # link to previous adder
        if from_rate != min(unique_rates):
            pipeline.insert(
                link_map= {addname+":sink:H1": (sumname or mmname)+":src:H1"}
            )

        # upsample
        if from_rate != maxrate:
            upname=f"up_{from_rate}_{to_rate}"
            pipeline.insert(
                TorchResampler(
                    name=upname,
                    sink_pad_names=("H1",),
                    source_pad_names=("H1",),
                    dtype=dtype,
                    device=device,
                    adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
                    inrate=from_rate,
                    outrate=to_rate[-1],
                ),
            )
            if from_rate == min(unique_rates):
                pipeline.insert(
                    link_map = {upname+":sink:H1": (sumname or mmname)+":src:H1"}
                )
            else:
                # link the previous adder to this upsampler
                pipeline.insert(
                    link_map = {upname+":sink:H1": addname+":src:H1"}
                )

            # add
            addname = f"add_{from_rate}_{to_rate}"
            pipeline.insert(
                Adder(
                    name=addname,
                    sink_pad_names=("H1", "H1up"),
                    source_pad_names=("H1",),
                    lib=math,
                    coeff_map={"H1": 1, "H1up": (to_rate[-1]/ from_rate) ** .5},
                ),
                link_map = {addname+":sink:H1up": upname+":src:H1"},
            )
        else:
            pipeline.insert(
                ImpulseSink(
                    name="sink0",
                    sink_pad_names=("H1","H1src"),
                    original_templates=original_templates,
                    template_duration=143,
                    plotname="plots/response",
                    impulse_pad="H1src",
                    verbose=verbose
                ),
                link_map = {"sink0:sink:H1": addname+":src:H1",
                "sink0:sink:H1src": "src1:src:H1im"}
            )

# Plot pipeline
pipeline.visualize("plots/graph.svg")

# Run pipeline
pipeline.run()
