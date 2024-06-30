from sgnligo.cbc.sort_bank import group_and_read_banks, SortedBank
import torch

from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink
from sgnts.base import AdapterConfig, Offset

from sgnligo.transforms import (
    Converter,
    TorchResampler,
    TimeShifter,
    TorchCorrelateValid,
    TorchMatmul,
    SumIndex,
    Adder,
)
from sgnligo import math

# Read in the svd banks!
svd_bank = [
    "H1-0250_GSTLAL_SVD_BANK_half-0-0.xml.gz",
]
nbank_pretend = 0
nslice = -1
verbose = True

copy_block = 1
device = "cpu"
dtype = torch.float32

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

pipeline = Pipeline()

# Build pipeline
pipeline.insert(
    FakeSeriesSrc(
        name="src1",
        source_pad_names=("H1",),
        num_buffers=143,
        rate=2048,
        num_samples=2048,
        signal_type="impulse",
        impulse_position=2 * 2048 - 64,
    ),
    Converter(
        name="converter1",
        sink_pad_names=("H1",),
        source_pad_names=(
            "H1down",
            "H1shift",
        ),
        backend="torch",
        dtype="float32",
        device=device,
    ),
    TorchResampler(
        name="down1",
        sink_pad_names=("H1",),
        source_pad_names=("H1down", "H1shift"),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=2048,
        outrate=512,
    ),
    TorchResampler(
        name="down2",
        sink_pad_names=("H1",),
        source_pad_names=("H1down", "H1shift"),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=512,
        outrate=256,
    ),
    TorchResampler(
        name="down3",
        sink_pad_names=("H1",),
        source_pad_names=("H1down", "H1shift"),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=256,
        outrate=128,
    ),
    TorchResampler(
        name="down4",
        sink_pad_names=("H1",),
        source_pad_names=("H1down", "H1shift"),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=128,
        outrate=64,
    ),
    TimeShifter(
        name="timeshift0",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        offset_segments=[
            (-16384 - Offset.fromsamples(2048 - 1, 2048), 0),
        ],
        shift=0,
        lib=math,
    ),
    TimeShifter(
        name="timeshift1",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        offset_segments=[
            (
                -Offset.fromsec(1)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 512),
                -Offset.fromsec(1)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 512),
            ),
        ],
        shift=Offset.fromsamples(32, 512) + Offset.fromsamples(8, 512),
        lib=math,
    ),
    TimeShifter(
        name="timeshift2",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        offset_segments=[
            (
                -Offset.fromsec(5)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 256),
                -Offset.fromsec(5)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512),
            ),
        ],
        shift=Offset.fromsamples(32, 256)
        + Offset.fromsamples(32, 512)
        + Offset.fromsamples(8, 256)
        + Offset.fromsamples(8, 512),
        lib=math,
    ),
    TimeShifter(
        name="timeshift3",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        offset_segments=[
            (
                -Offset.fromsec(13)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 128),
                -Offset.fromsec(13)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512),
            ),
            (
                -Offset.fromsec(29)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 128),
                -Offset.fromsec(29)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512),
            ),
            (
                -Offset.fromsec(45)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 128),
                -Offset.fromsec(45)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512),
            ),
            (
                -Offset.fromsec(61)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 128),
                -Offset.fromsec(61)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512),
            ),
        ],
        shift=Offset.fromsamples(32, 128)
        + Offset.fromsamples(32, 256)
        + Offset.fromsamples(32, 512)
        + Offset.fromsamples(8, 128)
        + Offset.fromsamples(8, 256)
        + Offset.fromsamples(8, 512),
        lib=math,
    ),
    TimeShifter(
        name="timeshift4",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        offset_segments=[
            (
                -Offset.fromsec(77)
                + Offset.fromsamples(32, 64)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 64)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 64),
                -Offset.fromsec(77)
                + Offset.fromsamples(32, 64)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 64)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512),
            ),
            (
                -Offset.fromsec(109)
                + Offset.fromsamples(32, 64)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 64)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512)
                - 16384
                - Offset.fromsamples(2048 - 1, 64),
                -Offset.fromsec(109)
                + Offset.fromsamples(32, 64)
                + Offset.fromsamples(32, 128)
                + Offset.fromsamples(32, 256)
                + Offset.fromsamples(32, 512)
                + Offset.fromsamples(8, 64)
                + Offset.fromsamples(8, 128)
                + Offset.fromsamples(8, 256)
                + Offset.fromsamples(8, 512),
            ),
        ],
        shift=Offset.fromsamples(32, 64)
        + Offset.fromsamples(32, 128)
        + Offset.fromsamples(32, 256)
        + Offset.fromsamples(32, 512)
        + Offset.fromsamples(8, 64)
        + Offset.fromsamples(8, 128)
        + Offset.fromsamples(8, 256)
        + Offset.fromsamples(8, 512),
        lib=math,
    ),
    TorchCorrelateValid(
        name="corr0",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        filters=bases[2048][()],
    ),
    TorchCorrelateValid(
        name="corr1",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        filters=bases[512][(2048,)],
    ),
    TorchCorrelateValid(
        name="corr2",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        filters=bases[256][(2048, 512)],
    ),
    TorchCorrelateValid(
        name="corr3",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        filters=bases[128][(2048, 512, 256)],
    ),
    TorchCorrelateValid(
        name="corr4",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        filters=bases[64][(2048, 512, 256, 128)],
    ),
    TorchMatmul(
        name="mm0",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        matrix=coeff[2048][()],
    ),
    TorchMatmul(
        name="mm1",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        matrix=coeff[512][(2048,)],
    ),
    TorchMatmul(
        name="mm2",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        matrix=coeff[256][(2048, 512)],
    ),
    TorchMatmul(
        name="mm3",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        matrix=coeff[128][(2048, 512, 256)],
    ),
    TorchMatmul(
        name="mm4",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        matrix=coeff[64][(2048, 512, 256, 128)],
    ),
    TorchResampler(
        name="up1",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=512,
        outrate=2048,
    ),
    TorchResampler(
        name="up2",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=256,
        outrate=512,
    ),
    TorchResampler(
        name="up3",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=128,
        outrate=256,
    ),
    TorchResampler(
        name="up4",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        dtype=dtype,
        device=device,
        adapter_config=AdapterConfig(pad_zeros_startup=True, lib=math),
        inrate=64,
        outrate=128,
    ),
    SumIndex(
        name="sumindex3",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        sl=[
            slice(0, 4),
        ],
    ),
    SumIndex(
        name="sumindex4",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        sl=[
            slice(0, 2),
        ],
    ),
    Adder(
        name="add0",
        sink_pad_names=("H1", "H1up"),
        source_pad_names=("H1",),
        lib=math,
        coeff_map={"H1": 1, "H1up": (2048 / 512) ** 0.5},
    ),
    Adder(
        name="add1",
        sink_pad_names=("H1", "H1up"),
        source_pad_names=("H1",),
        lib=math,
        coeff_map={"H1": 1, "H1up": (512 / 256) ** 0.5},
    ),
    Adder(
        name="add2",
        sink_pad_names=("H1", "H1up"),
        source_pad_names=("H1",),
        lib=math,
        coeff_map={"H1": 1, "H1up": (256 / 128) ** 0.5},
    ),
    Adder(
        name="add3",
        sink_pad_names=("H1", "H1up"),
        source_pad_names=("H1",),
        lib=math,
        coeff_map={"H1": 1, "H1up": (128 / 64) ** 0.5},
    ),
    DumpSeriesSink(
        name="sink0",
        sink_pad_names=("H1",),
        fname="impulse.out",
    ),
    link_map={
        "converter1:sink:H1": "src1:src:H1",
        # "converter2:sink:H1": "converter1:src:H1",
        "down1:sink:H1": "converter1:src:H1down",
        "down2:sink:H1": "down1:src:H1down",
        "down3:sink:H1": "down2:src:H1down",
        "down4:sink:H1": "down3:src:H1down",
        "timeshift0:sink:H1": "converter1:src:H1shift",
        "timeshift1:sink:H1": "down1:src:H1shift",
        "timeshift2:sink:H1": "down2:src:H1shift",
        "timeshift3:sink:H1": "down3:src:H1shift",
        "timeshift4:sink:H1": "down4:src:H1shift",
        "corr0:sink:H1": "timeshift0:src:H1",
        "corr1:sink:H1": "timeshift1:src:H1",
        "corr2:sink:H1": "timeshift2:src:H1",
        "corr3:sink:H1": "timeshift3:src:H1",
        "corr4:sink:H1": "timeshift4:src:H1",
        "mm0:sink:H1": "corr0:src:H1",
        "mm1:sink:H1": "corr1:src:H1",
        "mm2:sink:H1": "corr2:src:H1",
        "mm3:sink:H1": "corr3:src:H1",
        "mm4:sink:H1": "corr4:src:H1",
        "sumindex3:sink:H1": "mm3:src:H1",
        "sumindex4:sink:H1": "mm4:src:H1",
        "up4:sink:H1": "sumindex4:src:H1",
        "add3:sink:H1up": "up4:src:H1",
        "add3:sink:H1": "sumindex3:src:H1",
        "up3:sink:H1": "add3:src:H1",
        "add2:sink:H1up": "up3:src:H1",
        "add2:sink:H1": "mm2:src:H1",
        "up2:sink:H1": "add2:src:H1",
        "add1:sink:H1up": "up2:src:H1",
        "add1:sink:H1": "mm1:src:H1",
        "up1:sink:H1": "add1:src:H1",
        "add0:sink:H1up": "up1:src:H1",
        "add0:sink:H1": "mm0:src:H1",
        "sink0:sink:H1": "add0:src:H1",
    },
)

# Plot pipeline
pipeline.visualize("plots/graph.svg")

# Run pipeline
pipeline.run()

"""
"""
