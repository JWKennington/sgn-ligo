import torch

from sgn.apps import Pipeline
from sgn.sinks import FakeSink

from sgnts.base import AdapterConfig, Offset
from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink, FakeSeriesSink
from sgnts.transforms import Threshold

from sgnligo.cbc.sort_bank import group_and_read_banks, SortedBank
from sgnligo.base import ArrayOps
from sgnligo.sources import FrameReader
from sgnligo.sinks import ImpulseSink, StrikeSink
from sgnligo.transforms import (
    lloid,
    Whiten,
    Itacacac,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Read in the svd banks!
svd_bank = [
    # "H1-0750_GSTLAL_SVD_BANK-0-0.xml.gz",
    # "L1-0750_GSTLAL_SVD_BANK-0-0.xml.gz",
    # "V1-0750_GSTLAL_SVD_BANK-0-0.xml.gz",
    "H1-0250_GSTLAL_SVD_BANK_half-0-0.xml.gz",
    "L1-0250_GSTLAL_SVD_BANK_half-0-0.xml.gz",
    "V1-0250_GSTLAL_SVD_BANK_half-0-0.xml.gz",
    # "/home/yun-jing.huang/phd/o3mdc-filters/mdc11/svd_bank/H1-0000_GSTLAL_SVD_BANK-0-0.xml.gz",
    # "/home/yun-jing.huang/phd/o3mdc-filters/mdc11/svd_bank/H1-0101_GSTLAL_SVD_BANK-0-0.xml.gz",
    # "/home/yun-jing.huang/phd/o3mdc-filters/mdc11/svd_bank/H1-0102_GSTLAL_SVD_BANK-0-0.xml.gz",
    # "/home/yun-jing.huang/phd/o3mdc-filters/mdc11/svd_bank/L1-0000_GSTLAL_SVD_BANK-0-0.xml.gz",
    # "/home/yun-jing.huang/phd/o3mdc-filters/mdc11/svd_bank/V1-0000_GSTLAL_SVD_BANK-0-0.xml.gz",
]
impulse_bankno = 0
original_templates = "full_templates_bin0250_tol999_1024.hdf5"
nbank_pretend = 0
nslice = -1
verbose = True

copy_block = 1
device = "cpu"
dtype = torch.float32
impulse = True

trigger_finding_length = 2048

ArrayOps.DEVICE = device
ArrayOps.DTYPE = dtype

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
bank_metadata = sorted_bank.bank_metadata

num_samples = 2048
verbose = False

ifos = bank_metadata["ifos"]

pipeline = Pipeline()

lloid_input_source_link = {}
for ifo in ifos:
    # Build pipeline
    if impulse:
        source_pad_names = (ifo,)
        signal_type = "impulse"
        impulse_position = 2048 + 8 * 10
    else:
        source_pad_names = (ifo,)
        signal_type = "white"
        impulse_position = None
    pipeline.insert(
        FakeSeriesSrc(
            name=ifo + "_src",
            source_pad_names=source_pad_names,
            num_buffers=10,
            rate=2048,
            num_samples=num_samples,
            signal_type=signal_type,
            impulse_position=impulse_position,
            verbose=verbose,
        ),
    )
    lloid_input_source_link[ifo] = ifo + "_src:src:" + ifo

# connect LLOID
lloid_output_source_link = lloid(
    pipeline, sorted_bank, lloid_input_source_link, num_samples, nslice, device, dtype
)

# make the sink
impulse_ifo = "H1"
if impulse:
    pipeline.insert(
        ImpulseSink(
            name="imsink0",
            sink_pad_names=tuple(ifos) + (impulse_ifo + "_src",),
            original_templates=original_templates,
            template_duration=141,
            plotname="plots/response",
            impulse_pad=impulse_ifo + "_src",
            data_pad=impulse_ifo,
            bankno=impulse_bankno,
            verbose=verbose,
        ),
    )
else:
    # connect itacacac
    pipeline.insert(
        Itacacac(
            name="itacacac",
            sink_pad_names=tuple(ifos),
            source_pad_names=("trigs",),
            trigger_finding_length=trigger_finding_length,
            autocorrelation_banks=sorted_bank.autocorrelation_banks,
            template_ids=sorted_bank.template_ids,
            bankids_map=sorted_bank.bankids_map,
            end_times=sorted_bank.end_times,
            device=device,
        ),
        StrikeSink(
            name="sink0",
            sink_pad_names=("trigs",),
            ifos=ifos,
            verbose=False,
            all_template_ids=sorted_bank.template_ids.numpy(),
            bankids_map=sorted_bank.bankids_map,
            subbankids=sorted_bank.subbankids,
            template_sngls=sorted_bank.sngls,
        ),
        link_map={"sink0:sink:trigs": "itacacac:src:trigs"},
    )


# link output of lloid
for ifo, link in lloid_output_source_link.items():
    if impulse:
        pipeline.insert(
            link_map={
                "imsink0:sink:" + ifo: link,
            }
        )
        if ifo == impulse_ifo:
            pipeline.insert(
                link_map={"imsink0:sink:" + ifo + "_src": ifo + "_src:src:" + ifo}
            )
    else:
        pipeline.insert(
            link_map={
                "itacacac:sink:" + ifo: link,
            }
        )

# Plot pipeline
pipeline.visualize("plots/graph.png")

# Run pipeline
pipeline.run()
