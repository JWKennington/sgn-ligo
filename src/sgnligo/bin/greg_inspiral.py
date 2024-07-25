import torch

from optparse import OptionParser

from sgn.apps import Pipeline
from sgn.sinks import FakeSink

from sgnts.base import AdapterConfig, Offset
from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink, FakeSeriesSink
from sgnts.transforms import Threshold

from sgnligo.cbc.sort_bank import group_and_read_banks, SortedBank
from sgnligo.sources import FrameReader, DevShmSrc
from sgnligo.sinks import ImpulseSink, StrikeSink, KafkaSink
from sgnligo.transforms import (
    lloid,
    Whiten,
    Resampler,
    Itacacac,
)


def parse_command_line():
    parser = OptionParser()

    parser.add_option(
        "--data-source",
        action="store",
        default="frames",
        help="The type of the input source.",
    )
    parser.add_option(
        "--shared-memory-dir",
        metavar="directory",
        help="Set the name of the shared memory directory.",
    )
    parser.add_option(
        "--wait-time",
        metavar="seconds",
        type=int,
        default=60,
        help="Time to wait for new files in seconds before throwing an error. In online mode, new files should always arrive every second, unless there are problems. Default wait time is 60 seconds.",
    )
    parser.add_option(
        "--num-buffers",
        type="int",
        action="store",
        default=10,
        help="Number of buffers the source element should produce when source is fake source",
    )
    parser.add_option(
        "--gps-start-time",
        metavar="seconds",
        help="Set the start time of the segment to analyze in GPS seconds.",
    )
    parser.add_option(
        "--gps-end-time",
        metavar="seconds",
        help="Set the end time of the segment to analyze in GPS seconds.",
    )
    parser.add_option(
        "--sample-rate",
        metavar="Hz",
        type=int,
        default=16384,
        help="Requested sampling rate of the data.",
    )
    parser.add_option(
        "--frame-cache",
        metavar="file",
        help="Set the path to the frame cache file to analyze.",
    )
    parser.add_option(
        "--channel-name", metavar="channel", help="Name of the data channel to analyze."
    )
    parser.add_option(
        "--whitening-method",
        metavar="algorithm",
        default="gstlal",
        help="Algorithm to use for whitening the data. Supported options are 'gwpy' or 'gstlal'. Default is gstlal.",
    )
    parser.add_option(
        "--reference-psd",
        metavar="file",
        help="load the spectrum from this LIGO light-weight XML file (optional).",
    )
    parser.add_option(
        "--track-psd",
        action="store_true",
        help="Enable dynamic PSD tracking.  Always enabled if --reference-psd is not given.",
    )
    parser.add_option(
        "--psd-fft-length",
        action="store",
        type=int,
        help="The fft length for psd estimation.",
    )
    parser.add_option(
        "--ht-gate-threshold",
        action="store",
        type=float,
        default=float("+inf"),
        help="The gating threshold. Data above this value will be gated out.",
    )
    parser.add_option(
        "--svd-bank",
        metavar="filename",
        action="append",
        default=[],
        help="Set the name of the LIGO light-weight XML file from which to load the "
        "svd bank for a given instrument.  To analyze multiple instruments, "
        "--svd-bank can be called multiple times for svd banks corresponding "
        "to different instruments.  If --data-source is lvshm or framexmit, "
        "then only svd banks corresponding to a single bin must be given. "
        "If given multiple times, the banks will be processed one-by-one, in "
        "order.  At least one svd bank for at least 2 detectors is required, "
        "but see also --svd-bank-cache.",
    )
    parser.add_option(
        "--nbank-pretend",
        type="int",
        action="store",
        default=0,
        help="Pretend we have this many subbanks by copying the first subbank "
        "this many times",
    )
    parser.add_option(
        "--nslice",
        type="int",
        action="store",
        default=-1,
        help="Only filter this many timeslices. Default: -1, filter all timeslices.",
    )
    parser.add_option(
        "-v", "--verbose", action="store_true", help="Be verbose (optional)."
    )
    parser.add_option(
        "--torch-device",
        action="store",
        default="cpu",
        help="The device to run LLOID on.",
    )
    parser.add_option(
        "--torch-dtype",
        action="store",
        type="str",
        default="float32",
        help="The data type to run LLOID with.",
    )
    parser.add_option(
        "--trigger-finding-length",
        type="int",
        metavar="samples",
        action="store",
        default=2048,
        help="Produce triggers in blocks of this many samples.",
    )
    parser.add_option(
        "--num-samples",
        type="int",
        metavar="samples",
        action="store",
        default=2048,
        help="Source elements will produce buffers in strides of this many samples.",
    )
    parser.add_option(
        "--impulse-bank",
        metavar="filename",
        action="store",
        default=None,
        help="The full original templates to compare the impulse response test with.",
    )
    parser.add_option(
        "--impulse-position",
        action="store",
        default=-1,
        help="The sample point to put the impulse at. If -1, place randomly.",
    )
    parser.add_option(
        "--output-kafka-server",
        metavar="addr",
        help="Set the server address and port number for output data. Optional",
    )
    parser.add_option(
        "--analysis-tag",
        metavar="tag",
        default="test",
        help='Set the string to identify the analysis in which this job is part of. Used when --output-kafka-server is set. May not contain "." nor "-". Default is test.',
    )

    options, args = parser.parse_args()

    return options, args


def main():
    # parse arguments
    options, args = parse_command_line()

    dtype = options.torch_dtype
    if dtype == "float64":
        dtype = torch.float64
    elif dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError("Unknown data type")

    if options.impulse_bank is not None:
        impulse = True
    else:
        impulse = False

    # read in the svd banks
    banks = group_and_read_banks(
        svd_bank=options.svd_bank,
        nbank_pretend=options.nbank_pretend,
        nslice=options.nslice,
        verbose=True,
    )

    # sort and group the svd banks by sample rate
    sorted_bank = SortedBank(
        banks=banks,
        device=options.torch_device,
        dtype=dtype,
        nbank_pretend=options.nbank_pretend,
        nslice=options.nslice,
        verbose=True,
    )
    bank_metadata = sorted_bank.bank_metadata

    ifos = bank_metadata["ifos"]
    maxrate = bank_metadata["maxrate"]

    pipeline = Pipeline()

    lloid_input_source_link = {}
    for ifo in ifos:
        # Build pipeline
        if options.data_source == "frames" or options.data_source == "devshm":
            if options.data_source == "frames":
                pipeline.insert(
                    FrameReader(
                        name=ifo + "_Source",
                        source_pad_names=(ifo,),
                        rate=options.sample_rate,
                        num_samples=options.num_samples,
                        framecache=options.frame_cache,
                        channel_name=options.channel_name,
                        instrument=ifo,
                        gps_start_time=options.gps_start_time,
                        gps_end_time=options.gps_end_time,
                    ),
                )
            else:
                pipeline.insert(
                    DevShmSrc(
                        name=ifo + "_Source",
                        source_pad_names=(ifo,),
                        rate=16384,
                        num_samples=16384,
                        channel_name=options.channel_name,
                        instrument=ifo,
                        shared_memory_dir=options.shared_memory_dir,
                        wait_time=options.wait_time,
                    ),
                )
            pipeline.insert(
                Resampler(
                    name=ifo + "_SourceResampler",
                    sink_pad_names=(ifo,),
                    source_pad_names=(ifo,),
                    inrate=options.sample_rate,
                    outrate=maxrate,
                ),
                Whiten(
                    name=ifo + "_Whitener",
                    sink_pad_names=(ifo,),
                    source_pad_names=(ifo,),
                    instrument=ifo,
                    sample_rate=maxrate,
                    fft_length=options.psd_fft_length,
                    whitening_method=options.whitening_method,
                    reference_psd=options.reference_psd,
                    # psd_pad_name = "Whitener:src:spectrum"
                ),
                Threshold(
                    name=ifo + "_Threshold",
                    source_pad_names=(ifo,),
                    sink_pad_names=(ifo,),
                    threshold=options.ht_gate_threshold,
                    startwn=maxrate // 2,
                    stopwn=maxrate // 2,
                    invert=True,
                ),
            )
            pipeline.insert(
                link_map={
                    ifo
                    + "_SourceResampler:sink:"
                    + ifo: ifo
                    + "_Source:src:"
                    + ifo,
                    ifo + "_Whitener:sink:" + ifo: ifo + "_SourceResampler:src:" + ifo,
                    ifo + "_Threshold:sink:" + ifo: ifo + "_Whitener:src:" + ifo,
                }
            )
            lloid_input_source_link[ifo] = ifo + "_Threshold:src:" + ifo
        else:
            if options.data_source == "impulse":
                source_pad_names = (ifo,)
                signal_type = "impulse"
                impulse_position = options.impulse_position
            elif options.data_source == "white":
                source_pad_names = (ifo,)
                signal_type = "white"
                impulse_position = None
            pipeline.insert(
                FakeSeriesSrc(
                    name=ifo + "_src",
                    source_pad_names=source_pad_names,
                    num_buffers=options.num_buffers,
                    rate=options.sample_rate,
                    num_samples=options.num_samples,
                    signal_type=signal_type,
                    impulse_position=options.impulse_position,
                    verbose=options.verbose,
                ),
            )
            lloid_input_source_link[ifo] = ifo + "_src:src:" + ifo

    # connect LLOID
    lloid_output_source_link = lloid(
        pipeline,
        sorted_bank,
        lloid_input_source_link,
        options.num_samples,
        options.nslice,
        options.torch_device,
        dtype,
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
                verbose=options.verbose,
            ),
        )
    else:
        # connect itacacac
        pipeline.insert(
            Itacacac(
                name="itacacac",
                sink_pad_names=tuple(ifos),
                source_pad_names=("trigs",),
                trigger_finding_length=options.trigger_finding_length,
                autocorrelation_banks=sorted_bank.autocorrelation_banks,
                template_ids=sorted_bank.template_ids,
                bankids_map=sorted_bank.bankids_map,
                end_times=sorted_bank.end_times,
                kafka=True,
                device=options.torch_device,
            ),
            StrikeSink(
                name="StrikeSnk",
                sink_pad_names=("trigs",),
                ifos=ifos,
                verbose=options.verbose,
                all_template_ids=sorted_bank.template_ids.numpy(),
                bankids_map=sorted_bank.bankids_map,
                subbankids=sorted_bank.subbankids,
                template_sngls=sorted_bank.sngls,
            ),
            KafkaSink(
                name="KafkaSnk",
                sink_pad_names=("kafka",),
                output_kafka_server=options.output_kafka_server,
                #topic="gstlal."+options.analysis_tag+"."+options.instrument+"_range_history",
                topic="gstlal.greg_test.H1_snr_history",
                routes="H1_snr_history",
                metadata_key="H1_snr_history",
                #tags=[options.instrument,],
                #reduce_time=options.kafka_reduce_time,
                reduce_time=1,
                verbose=True
            ),
            link_map={"StrikeSnk:sink:trigs": "itacacac:src:trigs",
            "KafkaSnk:sink:kafka": "itacacac:src:trigs"},
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


if __name__ == "__main__":
    main()
