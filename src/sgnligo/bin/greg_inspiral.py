import torch
import os
import sys

from optparse import OptionGroup, OptionParser

from sgn.apps import Pipeline
from sgn.sinks import FakeSink

from sgnts.base import AdapterConfig, Offset
from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink, FakeSeriesSink
from sgnts.transforms import Threshold

from sgnligo.cbc.sort_bank import group_and_read_banks, SortedBank
from sgnligo.base.utils import parse_list_to_dict
from sgnligo.sources import datasource
from sgnligo.sinks import ImpulseSink, StrikeSink, KafkaSink
from sgnligo.transforms import (
    condition,
    lloid,
    Itacacac,
    Latency,
)


def parse_command_line():
    parser = OptionParser()

    group = OptionGroup(parser, "Data source", "Options for data source.")
    group.add_option(
        "--data-source",
        action="store",
        default="frames",
        help="The type of the input source. Supported sources: 'white', 'sin', "
        "'impulse', 'frames', 'devshm'",
    )
    group.add_option(
        "--channel-name",
        metavar="channel",
        action="append",
        help="Name of the data channel to analyze. Can be given multiple times as "
        "--channel-name=IFO=CHANNEL-NAME",
    )
    group.add_option(
        "--sample-rate",
        metavar="Hz",
        type=int,
        help="Requested sampling rate of the data.",
    )
    group.add_option(
        "--source-buffer-duration",
        type=float,
        metavar="seconds",
        action="store",
        help="Source elements will produce buffers in strides of this duration.",
    )
    group.add_option(
        "--frame-cache",
        metavar="file",
        help="Set the path to the frame cache file to analyze.",
    )
    group.add_option(
        "--gps-start-time",
        metavar="seconds",
        help="Set the start time of the segment to analyze in GPS seconds. "
        "For frame cache data source",
    )
    group.add_option(
        "--gps-end-time",
        metavar="seconds",
        help="Set the end time of the segment to analyze in GPS seconds. "
        "For frame cache data source",
    )
    group.add_option(
        "--shared-memory-dir",
        metavar="ifo=directory",
        action="append",
        help="Set the name of the shared memory directory. "
        "Can be given multiple times as --shared-memory-dir=IFO=DIR-NAME",
    )
    group.add_option(
        "--state-vector-on-bits",
        metavar="ifo=number",
        action="append",
        help="Set the state vector on bits. "
        "Can be given multiple times as --state-vector-on-bits=IFO=NUMBER",
    )
    group.add_option(
        "--wait-time",
        metavar="seconds",
        type=int,
        default=60,
        help="Time to wait for new files in seconds before throwing an error. "
        "In online mode, new files should always arrive every second, unless "
        "there are problems. Default wait time is 60 seconds.",
    )
    group.add_option(
        "--num-buffers",
        type="int",
        action="store",
        default=10,
        help="Number of buffers the source element should produce when source is "
        "fake source",
    )
    group.add_option(
        "--impulse-position",
        action="store",
        default=-1,
        help="The sample point to put the impulse at. If -1, place randomly.",
    )
    group.add_option(
        "--impulse-ifo",
        action="store",
        help="Only do impulse test on data from this ifo.",
    )
    parser.add_option_group(group)

    group = OptionGroup(
        parser, "PSD Options", "Adjust noise spectrum estimation parameters"
    )
    group.add_option(
        "--whitening-method",
        metavar="algorithm",
        default="gstlal",
        help="Algorithm to use for whitening the data. Supported options are 'gwpy' "
        "or 'gstlal'. Default is gstlal.",
    )
    group.add_option(
        "--psd-fft-length",
        action="store",
        type=int,
        help="The fft length for psd estimation.",
    )
    group.add_option(
        "--reference-psd",
        metavar="file",
        help="load the spectrum from this LIGO light-weight XML file (optional).",
    )
    group.add_option(
        "--track-psd",
        action="store_true",
        help="Enable dynamic PSD tracking.  Always enabled if --reference-psd is not "
        "given.",
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Data Qualtiy", "Adjust data quality handling")
    group.add_option(
        "--ht-gate-threshold",
        action="store",
        type=float,
        default=float("+inf"),
        help="The gating threshold. Data above this value will be gated out.",
    )
    parser.add_option_group(group)

    group = OptionGroup(
        parser, "Trigger Generator", "Adjust trigger generator behaviour"
    )
    group.add_option(
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
    group.add_option(
        "--nbank-pretend",
        type="int",
        action="store",
        default=0,
        help="Pretend we have this many subbanks by copying the first subbank "
        "this many times",
    )
    group.add_option(
        "--nslice",
        type="int",
        action="store",
        default=-1,
        help="Only filter this many timeslices. Default: -1, filter all timeslices.",
    )
    group.add_option(
        "--trigger-finding-length",
        type="int",
        metavar="samples",
        action="store",
        default=2048,
        help="Produce triggers in blocks of this many samples.",
    )
    group.add_option(
        "--impulse-bank",
        metavar="filename",
        action="store",
        default=None,
        help="The full original templates to compare the impulse response test with.",
    )
    group.add_option(
        "--impulse-bankno",
        metavar="index",
        action="store",
        help="The template bank index to perform the impulse test on.",
    )
    group.add_option(
        "--trigger-output",
        metavar="filename",
        action="append",
        default=[],
        help="Set the name of the LIGO light-weight XML output file *.{xml,xml.gz} "
        "or an SQLite database *.sqlite (required).  Can be given multiple times.  "
        "Exactly as many output files must be specified as svd-bank files will be "
        "processed (see --svd-bank).",
    )
    parser.add_option_group(group)

    group = OptionGroup(
        parser, "Ranking Statistic Options", "Adjust ranking statistic behaviour"
    )
    group.add_option(
        "--ranking-stat-output",
        metavar="filename",
        action="append",
        default=[],
        help="Set the name of the file to which to write ranking statistic data "
        "collected from triggers (optional).  Can be given more than once.  If given, "
        "exactly as many must be provided as there are --svd-bank options and they will"
        " be writen to in order.",
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Program Behaviour")
    group.add_option(
        "--torch-device",
        action="store",
        default="cpu",
        help="The device to run LLOID and Trigger generation on.",
    )
    group.add_option(
        "--torch-dtype",
        action="store",
        type="str",
        default="float32",
        help="The data type to run LLOID and Trigger generation with.",
    )
    group.add_option(
        "-v", "--verbose", action="store_true", help="Be verbose (optional)."
    )
    group.add_option(
        "--output-kafka-server",
        metavar="addr",
        help="Set the server address and port number for output data. Optional",
    )
    group.add_option(
        "--analysis-tag",
        metavar="tag",
        default="test",
        help="Set the string to identify the analysis in which this job is part of. "
        'Used when --output-kafka-server is set. May not contain "." nor "-". Default '
        "is test.",
    )
    group.add_option(
        "--graph-name", metavar="filename", help="Plot pipieline graph to graph_name."
    )
    group.add_option(
        "--fake-sink", action="store_true", help="Connect to a FakeSeriesSink"
    )
    parser.add_option_group(group)

    options, args = parser.parse_args()

    return options, args


def main():
    # parse arguments
    options, args = parse_command_line()

    # sanity check options
    known_datasources = ["white", "sin", "impulse", "frames", "devshm"]
    if options.data_source in ["white", "sin", "impulse"]:
        if not options.num_buffers:
            raise ValueError(
                "Must specify num_buffers when data_source is one of 'white',"
                "'sin', 'impulse'"
            )
        elif not options.sample_rate:
            raise ValueError(
                "Must specify sample_rate when data_source is one of 'white',"
                "'sin', 'impulse'"
            )
        elif not options.source_buffer_duration:
            raise ValueError(
                "Must specify source_buffer_duration when data_source is one "
                "of 'white', 'sin', 'impulse'"
            )

        if options.data_source == "impulse":
            if not options.impulse_bank:
                raise ValueError("Must specify impulse_bank when data_source='impulse'")
            elif not options.impulse_bankno:
                raise ValueError(
                    "Must specify impulse_bankno when data_source='impulse'"
                )
            elif not options.impulse_ifo:
                raise ValueError("Must specify impulse_ifo when data_source='impulse'")

        source_ifos = None
    elif options.data_source == "frames":
        if not options.frame_cache:
            raise ValueError("Must specify frame_cache when data_source='frames'")
        elif not options.gps_start_time or not options.gps_end_time:
            raise ValueError(
                "Must specify gps_start_time and gps_end_time when "
                "data_source='frames'"
            )
        elif not options.channel_name:
            raise ValueError("Must specify channel_name when data_source='frames'")
        elif not options.sample_rate:
            # FIXME: shoud we just determine the sample rate from the gwf file?
            raise ValueError("Must specify sample_rate when data_source='frames'")
        elif not options.source_buffer_duration:
            raise ValueError("Must specify source_buffer_duration")
        channel_dict = parse_list_to_dict(options.channel_name)
        source_ifos = list(channel_dict.keys())
    elif options.data_source == "devshm":
        if not options.shared_memory_dir:
            raise ValueError("Must specify shared_memory_dir when data_source='devshm'")
        elif not options.channel_name:
            raise ValueError("Must specify channel_name when data_source='devshm'")
        options.source_buffer_duration = 1
        options.sample_rate = 16384
        channel_dict = parse_list_to_dict(options.channel_name)
        source_ifos = list(channel_dict.keys())
    else:
        raise ValueError(f"Unknown data source, must be one of {known_datasources}")

    # FIXME: currently track psd is always enabled in whitener
    if not options.reference_psd:
        options.track_psd = True

    # check pytorch data type
    dtype = options.torch_dtype
    if dtype == "float64":
        dtype = torch.float64
    elif dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError("Unknown data type")

    # read in the svd banks
    banks = group_and_read_banks(
        svd_bank=options.svd_bank,
        source_ifos=source_ifos,
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

    ifos = source_ifos
    maxrate = bank_metadata["maxrate"]

    #
    # Build pipeline
    #
    pipeline = Pipeline()

    # Data source
    source_out_links = datasource(pipeline, options, ifos)
    if options.data_source != "impulse":
        source_out_links, horizon_out_links, whiten_latency_out_links = condition(
            pipeline, options, ifos, maxrate, source_out_links
        )
    else:
        horizon_out_links = None

    num_samples = int(options.source_buffer_duration * maxrate)

    # connect LLOID
    lloid_output_source_link = lloid(
        pipeline,
        sorted_bank,
        source_out_links,
        num_samples,
        options.nslice,
        options.torch_device,
        dtype,
    )

    # make the sink
    if options.data_source == "impulse":
        pipeline.insert(
            ImpulseSink(
                name="imsink0",
                sink_pad_names=tuple(ifos) + (options.impulse_ifo + "_src",),
                original_templates=original_templates,
                template_duration=141,
                plotname="plots/response",
                impulse_pad=options.impulse_ifo + "_src",
                data_pad=options.impulse_ifo,
                bankno=options.impulse_bankno,
                verbose=options.verbose,
            ),
        )
        # link output of lloid
        for ifo, link in lloid_output_source_link.items():
            pipeline.insert(
                link_map={
                    "imsink0:sink:" + ifo: link,
                }
            )
            if ifo == impulse_ifo:
                pipeline.insert(
                    link_map={"imsink0:sink:" + ifo + "_src": source_out_links[ifo]}
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
                kafka=options.data_source == "devshm",
                device=options.torch_device,
            ),
        )
        for ifo in ifos:
            pipeline.insert(
                link_map={
                    "itacacac:sink:" + ifo: lloid_output_source_link[ifo],
                }
            )
        if options.data_source == "devshm":
            pipeline.insert(
                Latency(
                    name="itacacac_latency",
                    sink_pad_names=("data",),
                    source_pad_names=("latency",),
                    route="all_itacacac_latency",
                ),
                link_map={"itacacac_latency:sink:data": "itacacac:src:trigs"},
            )

        # Connect sink
        if options.fake_sink:
            pipeline.insert(
                FakeSeriesSink(
                    name="Sink",
                    sink_pad_names=("sink",),
                    verbose=options.verbose,
                ),
                link_map={"Sink:sink:sink": "itacacac:src:trigs"},
            )
        elif options.data_source == "devshm":
            pipeline.insert(
                KafkaSink(
                    name="KafkaSnk",
                    sink_pad_names=(
                        "kafka",
                        "itacacac_latency",
                    )
                    + tuple(ifo + "_whiten_latency" for ifo in ifos),
                    output_kafka_server=options.output_kafka_server,
                    topics=[
                        "gstlal." + options.analysis_tag + "." + ifo + "_snr_history"
                        for ifo in ifos
                    ]
                    + [
                        "gstlal." + options.analysis_tag + ".latency_history",
                    ]
                    + [
                        "gstlal." + options.analysis_tag + ".all_itacacac_latency",
                    ]
                    + [
                        "gstlal."
                        + options.analysis_tag
                        + "."
                        + ifo
                        + "_whitening_latency"
                        for ifo in ifos
                    ],
                    # topic="gstlal.greg_test.H1_snr_history",
                    routes=[ifo + "_snr_history" for ifo in ifos],
                    reduce_time=1,
                    verbose=options.verbose,
                ),
                link_map={
                    "KafkaSnk:sink:kafka": "itacacac:src:trigs",
                    "KafkaSnk:sink:itacacac_latency": "itacacac_latency:src:latency",
                },
            )
            for ifo in ifos:
                pipeline.insert(
                    link_map={
                        "KafkaSnk:sink:"
                        + ifo
                        + "_whiten_latency": whiten_latency_out_links[ifo],
                    }
                )
        else:
            pipeline.insert(
                StrikeSink(
                    name="StrikeSnk",
                    sink_pad_names=("trigs",)
                    + tuple(["horizon_" + ifo for ifo in ifos]),
                    ifos=ifos,
                    verbose=options.verbose,
                    all_template_ids=sorted_bank.template_ids.numpy(),
                    bankids_map=sorted_bank.bankids_map,
                    subbankids=sorted_bank.subbankids,
                    template_sngls=sorted_bank.sngls,
                    trigger_output=options.trigger_output,
                    ranking_stat_output=options.ranking_stat_output,
                ),
                link_map={
                    "StrikeSnk:sink:trigs": "itacacac:src:trigs",
                },
            )
            for ifo in ifos:
                pipeline.insert(
                    link_map={
                        "StrikeSnk:sink:horizon_" + ifo: horizon_out_links[ifo],
                    }
                )

    # Plot pipeline
    if options.graph_name:
        pipeline.visualize(options.graph_name)

    # Run pipeline
    pipeline.run()

    #
    # Cleanup template bank temp files
    #

    print("shutdown: template bank cleanup", flush=True, file=sys.stderr)
    for ifo in banks:
        for bank in banks[ifo]:
            if options.verbose:
                print("removing file: ", bank.template_bank_filename, file=sys.stderr)
            os.remove(bank.template_bank_filename)

    print("shutdown: del bank", flush=True, file=sys.stderr)
    del bank


if __name__ == "__main__":
    main()
