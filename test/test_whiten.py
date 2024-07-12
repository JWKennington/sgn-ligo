#!/usr/bin/env python3

import os
from optparse import OptionParser

from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink
from sgnligo.transforms import Whiten
from sgnligo.sources import FrameReader
from sgnts.transforms import Resampler
import os

def parse_command_line():
    parser = OptionParser()

    parser.add_option("--instrument", metavar = "ifo", help = "Instrument to analyze. H1, L1, or V1.")
    parser.add_option("--output-dir", metavar = "path", help = "Directory to write output data into.")
    parser.add_option("--sample-rate", metavar = "Hz", type = int, default=16384, help="Requested sampling rate of the data.")
    parser.add_option("--buffer-duration", metavar = "seconds", type = int, default = 1, help = "Length of output buffers in seconds. Default is 1 second.")
    parser.add_option("--frame-cache", metavar = "file", help="Set the path to the frame cache file to analyze.")
    parser.add_option("--channel-name", metavar = "channel", help = "Name of the data channel to analyze.")
    parser.add_option("--whitening-method", metavar = "algorithm", default = "gstlal", help = "Algorithm to use for whitening the data. Supported options are 'gwpy' or 'gstlal'. Default is gstlal.")
    parser.add_option("--reference-psd", metavar = "file", help = "load the spectrum from this LIGO light-weight XML file (optional).")
    parser.add_option("--track-psd", action = "store_true", help = "Enable dynamic PSD tracking.  Always enabled if --reference-psd is not given.")

    options, args = parser.parse_args()

    return options, args

def test_whitengraph(capsys): 

    # parse arguments
    options, args = parse_command_line()

    os.makedirs(options.output_dir, exist_ok=True)

    num_samples = options.sample_rate * options.buffer_duration

    # sanity check the whitening method given
    if options.whitening_method not in ("gwpy", "gstlal"):
        raise ValueError("Unknown whitening method, exiting.")

    if options.reference_psd is None:
        options.track_psd = True # FIXME not implemented

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

    pipeline.insert(FrameReader(
               name = "src1",
               source_pad_names = ("H1",),
               rate=options.sample_rate,
               num_samples=num_samples,
               framecache=options.frame_cache,
               channel_name = (options.channel_name,), # FIXME why is this a tuple
               instruments = (options.instrument,), # FIXME why  is this a tuple
             ),
             Resampler(
               name="trans1",
               source_pad_names=("H1",),
               sink_pad_names=("H1",),
               inrate=options.sample_rate,
               outrate=2048,
             ),
             Whiten(
               name = "whiten",
               source_pad_names = ("H1","H2"),
               sink_pad_names = ("H1",),
               whitening_method = options.whitening_method,
               ref_psd = options.reference_psd,
               psd_pad_name = "whiten:src:H2"
             ),
             DumpSeriesSink(
               name = "snk1",
               sink_pad_names = ("H1",),
               fname = os.path.join(options.output_dir, 'out.txt'),
             ),
             DumpSeriesSink(
                 name = "snk3",
                 sink_pad_names = ("H2",),
                 fname = os.path.join(options.output_dir, 'psd_out.txt'),
             )
    )

    pipeline.insert(DumpSeriesSink(
           name = "snk2",
           sink_pad_names = ("H1",),
           fname = 'in.txt'
         ))
    pipeline.insert(link_map={
                              "trans1:sink:H1": "src1:src:H1",
                              "whiten:sink:H1":"trans1:src:H1",
                              "snk1:sink:H1":"whiten:src:H1",
                              "snk2:sink:H1":"src1:src:H1",
                              "snk3:sink:H2":"whiten:src:H2"
                              })

    pipeline.run()

if __name__ == "__main__":
    test_whitengraph(None)

