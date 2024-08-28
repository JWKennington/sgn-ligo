import torch

from . import (
    Converter,
    TorchResampler,
    LLOIDCorrelate,
    TorchMatmul,
    SumIndex,
    Adder,
)

from sgnts.base import AdapterConfig, Offset

from sgnligo.base import ArrayOps

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def lloid(
    pipeline,
    sorted_bank,
    input_source_links: dict[str, str],
    num_samples: int,
    nslice: int,
    device,
    dtype,
):
    ArrayOps.DEVICE = device
    ArrayOps.DTYPE = dtype

    output_source_links = {}

    bank_metadata = sorted_bank.bank_metadata
    ifos = bank_metadata["ifos"]
    unique_rates = list(bank_metadata["unique_rates"].keys())
    maxrate = bank_metadata["maxrate"]
    bases = sorted_bank.bases_cat
    coeff = sorted_bank.coeff_sv_cat

    pipeline.insert(
        Converter(
            name="converter1",
            sink_pad_names=tuple(ifos),
            source_pad_names=tuple(ifos),
            adapter_config=AdapterConfig(stride=num_samples),
            backend="torch",
            dtype=dtype,
            device=device,
        ),
    )

    # Multi-band
    sorted_rates = bank_metadata["sorted_rates"]
    for ifo in ifos:
        pipeline.insert(
            link_map={
                "converter1:sink:" + ifo: input_source_links[ifo],
            }
        )
        prev_source_pad = "converter1:src:" + ifo

        for i, rate in enumerate(unique_rates[:-1]):
            rate_down = unique_rates[i + 1]
            name = f"{ifo}_down_{rate_down}"
            sink_pad_full = name + ":sink:" + ifo

            source_pad_full = name + ":src:" + ifo

            to_rates = sorted_rates[rate_down].keys()

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
                            corrname
                            + ":sink:"
                            + ifo: f"{ifo}_down_{from_rate}:src:"
                            + ifo
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
                        final_adder_coeff_map[sink_name] = (
                            to_rate[-1] / from_rate
                        ) ** 0.5
                        final_adder_addslices_map[sink_name] = (
                            rate_group["addslice"],
                            slice(rate_group["ntempmax"]),
                        )

        if nslice != 1:
            # final adder
            pipeline.insert(
                Adder(
                    name=f"{ifo}_add_{maxrate}",
                    sink_pad_names=(ifo,)
                    + tuple(k for k in final_adder_coeff_map.keys()),
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
            output_source_links[ifo] = f"{ifo}_add_{maxrate}:src:" + ifo
        else:
            output_source_links[ifo] = mmname + ":src:" + ifo

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
    return output_source_links
