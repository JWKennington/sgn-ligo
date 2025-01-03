#!/usr/bin/env python3
import os
from optparse import OptionParser

import pytest
from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgnts.transforms import Gate

from sgnligo.sources import DevShmSource
from sgnligo.transforms import BitMask


def parse_command_line():
    parser = OptionParser()

    parser.add_option(
        "--instrument", metavar="ifo", help="Instrument to analyze. H1, L1, or V1."
    )
    parser.add_option(
        "--channel-name", metavar="channel", help="Name of the data channel to analyze."
    )
    parser.add_option(
        "--state-channel-name",
        metavar="channel",
        help="Name of the state vector channel to analyze.",
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
        help="Time to wait for new files in seconds before throwing an error. In online"
        " mode, new files should always arrive every second, unless there are problems."
        " Default wait time is 60 seconds.",
    )
    parser.add_option(
        "--state-vector-on-bits",
        metavar="bits",
        type=int,
        help="Set the state vector on bits to process.",
    )

    options, args = parser.parse_args()

    return options, args


@pytest.mark.skip(reason="Not currently pytest compatible")
def test_devshmsrc(capsys):

    # parse arguments
    options, args = parse_command_line()

    if not os.path.exists(options.shared_memory_dir):
        raise ValueError(f"{options.shared_memory_dir} directory not found, exiting.")

    pipeline = Pipeline()

    #
    #       -----------
    #      | DevShmSource |
    #       -----------
    #  state |       |
    #  vector|       |
    #  ---------     | strain
    # | BitMask |    |
    #  ---------     |
    #        \       |
    #         \      |
    #       ------------
    #      |   Gate     |
    #       ------------
    #             |
    #             |
    #       ------------
    #      |   NullSink |
    #       ------------

    pipeline.insert(
        DevShmSource(
            name="src1",
            # source_pad_names=("H1",),
            channel_name=options.channel_name,
            state_channel_name=options.state_channel_name,
            instrument=options.instrument,
            shared_memory_dir=options.shared_memory_dir,
            wait_time=options.wait_time,
            verbose=True,
        ),
        BitMask(
            name="mask",
            sink_pad_names=(options.instrument,),
            source_pad_names=(options.instrument,),
            bit_mask=options.state_vector_on_bits,
        ),
        Gate(
            name="gate",
            sink_pad_names=("strain", "state_vector"),
            control="state_vector",
            source_pad_names=(options.instrument,),
        ),
        NullSink(
            name="snk2",
            sink_pad_names=(options.state_channel_name,),
        ),
    )

    pipeline.insert(
        link_map={
            "mask:sink:" + options.instrument: "src1:src:" + options.state_channel_name,
            "gate:sink:strain": "src1:src:" + options.channel_name,
            "gate:sink:state_vector": "mask:src:" + options.instrument,
            "snk2:sink:" + options.state_channel_name: "gate:src:" + options.instrument,
        }
    )

    pipeline.run()


if __name__ == "__main__":
    test_devshmsrc(None)
