from sgnligo.cbc.sort_bank import group_and_read_banks, SortedBank
import torch

from sgn.apps import Pipeline
from sgn.sinks import FakeSink

from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink, FakeSeriesSink
from sgnts.base import AdapterConfig, Offset

from sgnligo.sinks import ImpulseSink, StrikeSink

from sgnligo.transforms import (
    Converter,
    TorchResampler,
    LLOIDCorrelate,
    TorchMatmul,
    SumIndex,
    Adder,
    Align,
    Itacacac,
)
from sgnligo.base import ArrayOps

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Read in the svd banks!
svd_bank = [
    "H1-0750_GSTLAL_SVD_BANK-0-0.xml.gz",
    "L1-0750_GSTLAL_SVD_BANK-0-0.xml.gz",
    "V1-0750_GSTLAL_SVD_BANK-0-0.xml.gz",
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
impulse = False

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
bases = sorted_bank.bases_cat
coeff = sorted_bank.coeff_sv_cat

bank_metadata = sorted_bank.bank_metadata
unique_rates = list(bank_metadata["unique_rates"].keys())
maxrate = bank_metadata["maxrate"]

num_samples = 2048
verbose = False

ifos = bank_metadata["ifos"]

pipeline = Pipeline()

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
            num_samples=2048,
            signal_type=signal_type,
            impulse_position=impulse_position,
            verbose=verbose,
        ),
    )

pipeline.insert(
    Converter(
        name="converter1",
        sink_pad_names=tuple(ifos),
        source_pad_names=tuple(ifos),
        adapter_config=AdapterConfig(stride=num_samples),
        backend="torch",
        dtype="float32",
        device=device,
    ),
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
    template_ids_np = sorted_bank.template_ids.numpy().flatten()
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
            template_ids=tuple(template_ids_np[template_ids_np != -1]),
            bankids_map=sorted_bank.bankids_map,
        ),
        link_map={"sink0:sink:trigs": "itacacac:src:trigs"},
    )

# Multi-band
sorted_rates = bank_metadata["sorted_rates"]
# multiband_source_pad_names = {r: {} for r in unique_rates}

for ifo in ifos:
    pipeline.insert(
        link_map={
            "converter1:sink:" + ifo: ifo + "_src:src:" + ifo,
        }
    )
    prev_source_pad = "converter1:src:" + ifo

    # multiband_source_pad_names[maxrate][()] = "converter1:src:"+ifo
    for i, rate in enumerate(unique_rates[:-1]):
        rate_down = unique_rates[i + 1]
        name = f"{ifo}_down_{rate_down}"
        sink_pad_full = name + ":sink:" + ifo

        source_pad_full = name + ":src:" + ifo

        to_rates = sorted_rates[rate_down].keys()
        # source_pads = [ifo]
        # for to_rate in to_rates:
        # source_pad = ifo+"shift_"+str(to_rate)
        # source_pads.append(source_pad)
        # multiband_source_pad_names[rate_down][to_rate] = name+":src:"+source_pad
        # multiband_source_pad_names[rate_down][to_rate] = name+":src:"+ifo

        pipeline.insert(
            TorchResampler(
                name=name,
                sink_pad_names=(ifo,),
                source_pad_names=(ifo,),
                dtype=dtype,
                device=device,
                adapter_config=AdapterConfig(pad_zeros_startup=True, lib=ArrayOps),
                inrate=rate,
                outrate=rate_down,
            ),
            link_map={sink_pad_full: prev_source_pad},
        )
        prev_source_pad = source_pad_full

# time segment shift
nfilter_samples = bank_metadata["nfilter_samples"]


for ifo in ifos:
    snr_slices = {r1: {} for r1 in reversed(unique_rates)}
    final_adder_coeff_map = {}  # sinkname: scale
    final_adder_addslices_map = {}  # sinkname: scale

    for from_rate in reversed(unique_rates):
        for to_rate, rate_group in sorted_rates[from_rate].items():
            segments = rate_group["segments"]
            shift = rate_group["shift"]
            uppad = rate_group["uppad"]
            downpad = rate_group["downpad"]
            delays = []
            for segment in segments:
                delays.append(Offset.fromsec(segment[0]))

            # Correlate
            corrname = f"{ifo}_corr_{from_rate}_{to_rate}"
            pipeline.insert(
                LLOIDCorrelate(
                    name=corrname,
                    sink_pad_names=(ifo,),
                    source_pad_names=(ifo,),
                    filters=bases[from_rate][to_rate][ifo],
                    lib=ArrayOps,
                    uppad=uppad,
                    downpad=downpad,
                    delays=delays,
                ),
            )
            if from_rate != maxrate:
                pipeline.insert(
                    link_map={
                        corrname + ":sink:" + ifo: f"{ifo}_down_{from_rate}:src:" + ifo
                    },
                )
            else:
                pipeline.insert(
                    link_map={corrname + ":sink:" + ifo: "converter1:src:" + ifo},
                )

            # matmul
            mmname = f"{ifo}_mm_{from_rate}_{to_rate}"
            pipeline.insert(
                TorchMatmul(
                    name=mmname,
                    sink_pad_names=(ifo,),
                    source_pad_names=(ifo,),
                    matrix=coeff[from_rate][to_rate][ifo],
                ),
                link_map={mmname + ":sink:" + ifo: corrname + ":src:" + ifo},
            )

            # sum same rate
            sumname = None
            if rate_group["sum_same_rate_slices"] is not None:
                sl = rate_group["sum_same_rate_slices"]
                sumname = f"{ifo}_sumindex_{from_rate}_{to_rate}"
                pipeline.insert(
                    SumIndex(
                        name=sumname,
                        sink_pad_names=(ifo,),
                        source_pad_names=(ifo,),
                        sl=sl,
                    ),
                    link_map={sumname + ":sink:" + ifo: mmname + ":src:" + ifo},
                )
                snr_slices[from_rate][to_rate] = sumname + ":src:" + ifo
            else:
                snr_slices[from_rate][to_rate] = mmname + ":src:" + ifo

            ## link to previous adder
            # if from_rate != min(unique_rates):
            #    pipeline.insert(
            #        link_map= {addname+":sink:H1": (sumname or mmname)+":src:H1"}
            #    )

            if from_rate != maxrate:
                upname = f"{ifo}_up_{from_rate}_{to_rate}"

                # upsample
                pipeline.insert(
                    TorchResampler(
                        name=upname,
                        sink_pad_names=(ifo,),
                        source_pad_names=(ifo,),
                        dtype=dtype,
                        device=device,
                        adapter_config=AdapterConfig(
                            pad_zeros_startup=True, lib=ArrayOps
                        ),
                        inrate=from_rate,
                        outrate=to_rate[-1],
                    ),
                )

                # else:
                #    # link the previous adder to this upsampler
                #    pipeline.insert(
                #        link_map = {upname+":sink:H1": addname+":src:H1"}
                #    )

                # add
                addname = f"{ifo}_add_{from_rate}_{to_rate}"
                sink_name = f"{ifo}_up_{from_rate}_{to_rate}"

                if to_rate[-1] != maxrate:
                    pipeline.insert(
                        Adder(
                            name=addname,
                            sink_pad_names=(ifo, sink_name),
                            source_pad_names=(ifo,),
                            lib=ArrayOps,
                            coeff_map={
                                ifo: 1,
                                sink_name: (to_rate[-1] / from_rate) ** 0.5,
                            },
                            addslices_map={
                                sink_name: (
                                    rate_group["addslice"],
                                    slice(rate_group["ntempmax"]),
                                )
                            },
                        ),
                    )
                else:
                    final_adder_coeff_map[sink_name] = (to_rate[-1] / from_rate) ** 0.5
                    final_adder_addslices_map[sink_name] = (
                        rate_group["addslice"],
                        slice(rate_group["ntempmax"]),
                    )

    if nslice != 1:
        # final adder
        pipeline.insert(
            Adder(
                name=f"{ifo}_add_{maxrate}",
                sink_pad_names=(ifo,) + tuple(k for k in final_adder_coeff_map.keys()),
                source_pad_names=(ifo,),
                lib=ArrayOps,
                coeff_map=dict(
                    {
                        ifo: 1,
                    },
                    **final_adder_coeff_map,
                ),
                addslices_map=final_adder_addslices_map,
            ),
        )
        if impulse:
            pipeline.insert(
                link_map={
                    "imsink0:sink:" + ifo: f"{ifo}_add_{maxrate}:src:" + ifo,
                }
            )
            if ifo == impulse_ifo:
                pipeline.insert(
                    link_map={"imsink0:sink:" + ifo + "_src": ifo + "_src:src:" + ifo}
                )
        else:
            pipeline.insert(
                link_map={
                    "itacacac:sink:" + ifo: f"{ifo}_add_{maxrate}:src:" + ifo,
                }
            )
    else:
        if impulse:
            pipeline.insert(
                link_map={
                    "imsink0:sink:" + ifo: mmname + ":src:" + ifo,
                }
            )
            if ifo == impulse_ifo:
                pipeline.insert(
                    link_map={"imsink0:sink:" + ifo + "_src": ifo + "_src:src:" + ifo}
                )
        else:
            pipeline.insert(
                link_map={
                    "itacacac:sink:" + ifo: mmname + ":src:" + ifo,
                }
            )

    connected = []
    # links for upsampler and adder
    for from_rate, v in snr_slices.items():
        for to_rate, snr_link in v.items():
            if from_rate != maxrate:
                if to_rate[-1] != maxrate:
                    upname = f"{ifo}_up_{to_rate[-1]}_{to_rate[:-1]}:sink:" + ifo
                    pipeline.insert(
                        link_map={
                            upname: f"{ifo}_add_{from_rate}_{to_rate}:src:" + ifo,
                        }
                    )
                    pipeline.insert(
                        link_map={
                            f"{ifo}_add_{from_rate}_{to_rate}:sink:"
                            + ifo
                            + f"_up_{from_rate}_{to_rate}": f"{ifo}_up_{from_rate}_{to_rate}:src:"
                            + ifo,
                        }
                    )
                    pipeline.insert(
                        link_map={
                            f"{ifo}_add_{from_rate}_{to_rate}:sink:"
                            + ifo: snr_slices[to_rate[-1]][to_rate[:-1]]
                        }
                    )
                else:
                    pipeline.insert(
                        link_map={
                            f"{ifo}_add_{maxrate}:sink:"
                            + ifo
                            + f"_up_{from_rate}_{to_rate}": f"{ifo}_up_{from_rate}_{to_rate}:src:"
                            + ifo,
                        }
                    )
                    pipeline.insert(
                        link_map={
                            f"{ifo}_add_{maxrate}:sink:"
                            + ifo: snr_slices[to_rate[-1]][to_rate[:-1]]
                        }
                    )
                connected.append(snr_slices[to_rate[-1]][to_rate[:-1]])

    # link the rest
    # FIXME: find a better way
    for from_rate, v in snr_slices.items():
        for to_rate, snr_link in v.items():
            if from_rate != maxrate:
                if snr_link not in connected:
                    upname = f"{ifo}_up_{from_rate}_{to_rate}"
                    pipeline.insert(
                        link_map={
                            f"{ifo}_up_{from_rate}_{to_rate}:sink:" + ifo: snr_link
                        }
                    )

# Plot pipeline
pipeline.visualize("plots/graph.png")

# Run pipeline
pipeline.run()
