from optparse import OptionParser

from sgn.apps import Pipeline
from sgn.sinks import FakeSink

from sgnts.base import AdapterConfig, Offset
from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink, FakeSeriesSink
from sgnts.transforms import Threshold

from sgnligo.sources import DevShmSrc
from sgnligo.sinks import FakeSeriesSink, KafkaSink, InfluxSink
from sgnligo.transforms import (
    Whiten,
    Resampler,
    HorizonDistance,
)


def parse_command_line():
    parser = OptionParser(description=__doc__)

    # add our own options
    parser.add_option(
        "--sample-rate",
        metavar="Hz",
        default=2048,
        type="int",
        help="Sample rate at which to generate the PSD, default 2048 Hz",
    )
    parser.add_option(
        "--psd-fft-length",
        metavar="s",
        default=8,
        type="int",
        help="FFT length, default 8s",
    )
    parser.add_option(
        "--scald-config",
        metavar="path",
        help="sets ligo-scald options based on yaml configuration.",
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
    parser.add_option(
        "--reference-psd",
        metavar="filename",
        help="Load spectrum from this LIGO light-weight XML file. The noise spectrum will be measured and tracked starting from this reference. (optional).",
    )
    parser.add_option(
        "--horizon-approximant",
        type="string",
        default="IMRPhenomD",
        help="Specify a waveform approximant to use while calculating the horizon distance and range. Default is IMRPhenomD.",
    )
    parser.add_option(
        "--horizon-f-min",
        metavar="Hz",
        type="float",
        default=15.0,
        help="Set the frequency at which the waveform model is to begin for the horizon distance and range calculation. Default is 15 Hz.",
    )
    parser.add_option(
        "--horizon-f-max",
        metavar="Hz",
        type="float",
        default=900.0,
        help="Set the upper frequency cut off for the waveform model used in the horizon distance and range calculation. Default is 900 Hz.",
    )
    parser.add_option(
        "--whitening-method",
        metavar="algorithm",
        default="gstlal",
        help="Algorithm to use for whitening the data. Supported options are 'gwpy' or 'gstlal'. Default is gstlal.",
    )
    parser.add_option(
        "-v", "--verbose", action="store_true", help="Be verbose (optional)."
    )

    parser.add_option(
        "--instrument", metavar="ifo", help="Instrument to analyze. H1, L1, or V1."
    )
    parser.add_option(
        "--channel-name", metavar="channel", help="Name of the data channel to analyze."
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
        "--kafka-reduce-time",
        metavar="seconds",
        type=int,
        default=1,
        help="Time to reduce data to send to kafka.",
    )

    options, filenames = parser.parse_args()

    return options, filenames


def main():
    # parse arguments
    options, args = parse_command_line()

    source_sample_rate = 16384

    num_samples = source_sample_rate * 1

    # sanity check the whitening method given
    if options.whitening_method not in ("gwpy", "gstlal"):
        raise ValueError("Unknown whitening method, exiting.")

    if options.reference_psd is None:
        options.track_psd = True  # FIXME not implemented

    pipeline = Pipeline()

    #
    #          -----------
    #         | DevShmSrc |
    #          -----------
    #         /
    #     H1 /
    #   ------------
    #  |  Resampler |
    #   ------------
    #       |
    #   ------------  hoft ----------
    #  |  Whiten    | --- | FakeSink |
    #   ------------       ----------
    #          |psd
    #   ------------
    #  |  Horizon   |
    #   ------------
    #          \
    #       H1  \
    #           -----------
    #          | KafkaSink |
    #           -----------
    #

    pipeline.insert(
        DevShmSrc(
            name="src1",
            source_pad_names=("frsrc",),
            rate=16384,
            num_samples=16384,
            channel_name=options.channel_name,
            instrument=options.instrument,
            shared_memory_dir=options.shared_memory_dir,
            wait_time=options.wait_time,
        ),
        Resampler(
            name="Resampler",
            source_pad_names=("resamp",),
            sink_pad_names=("frsrc",),
            inrate=source_sample_rate,
            outrate=options.sample_rate,
        ),
        Whiten(
            name="Whitener",
            source_pad_names=("hoft", "spectrum"),
            sink_pad_names=("resamp",),
            instrument=options.instrument,
            sample_rate=options.sample_rate,
            fft_length=options.psd_fft_length,
            whitening_method=options.whitening_method,
            reference_psd=options.reference_psd,
            psd_pad_name="Whitener:src:spectrum",
        ),
        HorizonDistance(
            name="Horizon",
            source_pad_names=("horizon",),
            sink_pad_names=("spectrum",),
            m1=1.4,
            m2=1.4,
            fmin=options.horizon_f_min,
            fmax=options.horizon_f_max,
            range=True,
            delta_f=1 / 16.0,
        ),
        FakeSeriesSink(
            name="HoftSnk",
            sink_pad_names=("hoft",),
            verbose=True,
        ),
        KafkaSink(
            name="HorizonSnk",
            sink_pad_names=("horizon",),
            output_kafka_server=options.output_kafka_server,
            topic="gstlal."+options.analysis_tag+"."+options.instrument+"_range_history",
            route='range_history',
            metadata_key='range',
            tags=[options.instrument,],
            reduce_time=options.kafka_reduce_time,
            verbose=True
        ),
    )

    pipeline.insert(
        link_map={
            "Resampler:sink:frsrc": "src1:src:frsrc",
            "Whitener:sink:resamp": "Resampler:src:resamp",
            "Horizon:sink:spectrum": "Whitener:src:spectrum",
            "HorizonSnk:sink:horizon": "Horizon:src:horizon",
            "HoftSnk:sink:hoft": "Whitener:src:hoft",
        }
    )

    pipeline.run()


if __name__ == "__main__":
    main()
