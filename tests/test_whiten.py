#!/usr/bin/env python3

import os
from optparse import OptionParser

import pytest
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink
from sgnts.transforms import Resampler

from sgnligo.sources import FrameReader
from sgnligo.transforms import Whiten


def parse_command_line():
    parser = OptionParser()

    parser.add_option(
        "--instrument", metavar="ifo", help="Instrument to analyze. H1, L1, or V1."
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
        "--output-dir", metavar="path", help="Directory to write output data into."
    )
    parser.add_option(
        "--sample-rate",
        metavar="Hz",
        type=int,
        default=16384,
        help="Requested sampling rate of the data.",
    )
    parser.add_option(
        "--buffer-duration",
        metavar="seconds",
        type=int,
        default=1,
        help="Length of output buffers in seconds. Default is 1 second.",
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
        "--reference-psd",
        metavar="file",
        help="load the spectrum from this LIGO light-weight XML file (optional).",
    )
    parser.add_option(
        "--track-psd",
        action="store_true",
        help="Enable dynamic PSD tracking.  Always enabled if --reference-psd is not"
        " given.",
    )

    options, args = parser.parse_args()

    return options, args


@pytest.mark.skip(reason="Not currently pytest compatible")
def test_whitengraph(capsys):

    # parse arguments
    options, args = parse_command_line()

    os.makedirs(options.output_dir, exist_ok=True)

    num_samples = options.sample_rate * options.buffer_duration

    if not (options.gps_start_time and options.gps_end_time):
        raise ValueError("Must provide both --gps-start-time and --gps-end-time.")

    if options.reference_psd is None:
        options.track_psd = True  # FIXME not implemented

    pipeline = Pipeline()

    #
    #          ------   H1   -------
    #         | src1 | ---- | snk2  |
    #          ------   SR1  -------
    #         /
    #     H1 /
    #   ----------
    #  |  whiten  |
    #   ----------
    #          \
    #       H1  \
    #           ------
    #          | snk1 |
    #           ------
    #

    pipeline.insert(
        FrameReader(
            name="FrameReader",
            source_pad_names=("frsrc",),
            rate=options.sample_rate,
            num_samples=num_samples,
            framecache=options.frame_cache,
            channel_name=options.channel_name,
            instrument=options.instrument,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
        ),
        Resampler(
            name="Resampler",
            source_pad_names=("resamp",),
            sink_pad_names=("frsrc",),
            inrate=options.sample_rate,
            outrate=2048,
        ),
        Whiten(
            name="Whitener",
            source_pad_names=("hoft", "spectrum"),
            sink_pad_names=("resamp",),
            instrument=options.instrument,
            sample_rate=2048,
            fft_length=4,
            reference_psd=options.reference_psd,
            psd_pad_name="spectrum",
            whiten_pad_name="hoft",
        ),
        DumpSeriesSink(
            name="HoftSnk",
            sink_pad_names=("hoft",),
            fname=os.path.join(options.output_dir, "out.txt"),
        ),
        DumpSeriesSink(
            name="SpectrumSnk",
            sink_pad_names=("spectrum",),
            fname=os.path.join(options.output_dir, "psd_out.txt"),
        ),
    )

    pipeline.insert(
        DumpSeriesSink(name="RawSnk", sink_pad_names=("frsrc",), fname="in.txt")
    )
    pipeline.insert(
        link_map={
            "Resampler:snk:frsrc": "FrameReader:src:frsrc",
            "Whitener:snk:resamp": "Resampler:src:resamp",
            "HoftSnk:snk:hoft": "Whitener:src:hoft",
            "SpectrumSnk:snk:spectrum": "Whitener:src:spectrum",
            "RawSnk:snk:frsrc": "FrameReader:src:frsrc",
        }
    )

    pipeline.run()


if __name__ == "__main__":
    test_whitengraph(None)
