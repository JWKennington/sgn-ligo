#!/usr/bin/env python3
import os
from optparse import OptionParser

from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink
from sgnligo.sources import DevShmSrc

from sgnts.transforms import Resampler

def parse_command_line():
    parser = OptionParser()

    parser.add_option("--instrument", metavar = "ifo", help = "Instrument to analyze. H1, L1, or V1.")
    parser.add_option("--channel-name", metavar = "channel", help = "Name of the data channel to analyze.")
    parser.add_option("--shared-memory-dir", metavar = "directory", help = "Set the name of the shared memory directory.")
    parser.add_option("--wait-time", metavar = "seconds", type = int, default = 60, help = "Time to wait for new files in seconds before throwing an error. In online mode, new files should always arrive every second, unless there are problems. Default wait time is 60 seconds.")
    parser.add_option("--state-vector-on-bits", metavar = "bits", type = int, help = "Set the state vector on bits to process.")

    options, args = parser.parse_args()

    return options, args

def test_devshmsrc(capsys): 

    # parse arguments
    options, args = parse_command_line()

    if not os.path.exists(options.shared_memory_dir):
        raise ValueError(f"{options.shared_memory_dir} directory not found, exiting.")

    pipeline = Pipeline()

    #
    #       ---------- 
    #      | src1     |
    #       ---------- 
    #              \
    #           H1  \ SR1
    #             ---------
    #            | snk1    |
    #             ---------

    
    pipeline.insert(DevShmSrc(
               name = "src1",
               source_pad_names = ("H1",),
               rate=16384,
               num_samples=16384,
               channel_name = options.channel_name,
               instrument = options.instrument,
               shared_memory_dir = options.shared_memory_dir,
               wait_time = options.wait_time,
               state_vector_on_bits = options.state_vector_on_bits,
             ),
             DumpSeriesSink(
               name = "snk1",
               sink_pad_names = ("H1",),
               fname = 'out.txt'
             )
    )

    pipeline.insert(link_map={
                              "snk1:sink:H1": "src1:src:H1"
                              })

    pipeline.run()

if __name__ == "__main__":
    test_devshmsrc(None)



