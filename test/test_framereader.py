#!/usr/bin/env python3
from optparse import OptionParser

from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink
from sgnligo.sources import FrameReader

from sgnts.transforms import Resampler

def parse_command_line():
    parser = OptionParser()

    parser.add_option("--instrument", metavar = "ifo", help = "Instrument to analyze. H1, L1, or V1.")
    parser.add_option("--channel-name", metavar = "channel", help = "Name of the data channel to analyze.")
    parser.add_option("--gps-start-time", metavar = "seconds", help="Set the start time of the segment to analyze in GPS seconds.")
    parser.add_option("--gps-end-time", metavar = "seconds", help="Set the end time of the segment to analyze in GPS seconds.")
    parser.add_option("--frame-cache", metavar = "file", help="Set the path to the frame cache file to analyze.")
    parser.add_option("--sample-rate", metavar = "Hz", type = int, default=16384, help="Requested sampling rate of the data.")
    parser.add_option("--buffer-duration", metavar = "seconds", type = int, default = 1, help = "Length of output buffers in seconds. Default is 1 second.")

    options, args = parser.parse_args()

    return options, args

def test_framereader(capsys): 

    # parse arguments
    options, args = parse_command_line()

    if not (options.gps_start_time and options.gps_end_time):
        raise ValueError("Must provide both --gps-start-time and --gps-end-time.")

    num_samples = options.sample_rate * options.buffer_duration

    pipeline = Pipeline()

    #
    #       ---------- 
    #      | src1     |
    #       ---------- 
    #              \
    #           H1  \ SR1
    #           ------------
    #          | Resampler  |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

    pipeline.insert(FrameReader(
               name = "src1",
               source_pad_names = ("H1",),
               rate=options.sample_rate,
               num_samples=num_samples,
               framecache = options.frame_cache,
               channel_name = options.channel_name,
               instrument = options.instrument,
               gps_start_time = options.gps_start_time,
               gps_end_time = options.gps_end_time,
             ),
             Resampler(
               name="trans1",
               source_pad_names=("H1",),
               sink_pad_names=("H1",),
               inrate=options.sample_rate,
               outrate=2048,
             ),
             DumpSeriesSink(
               name = "snk1",
               sink_pad_names = ("H1",),
               fname = 'out.txt'
             )
    )

    pipeline.insert(link_map={
                              "trans1:sink:H1": "src1:src:H1",
                              "snk1:sink:H1": "trans1:src:H1"
                              })

    pipeline.run()

if __name__ == "__main__":
    test_framereader(None)



