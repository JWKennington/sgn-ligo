#!/usr/bin/env python3
import pytest
from sgn.apps import Pipeline
from sgnts.sinks import FakeSeriesSink
from sgnts.transforms import Align

from sgnligo.sources import datasource_from_options, parse_command_line_datasource


@pytest.mark.skip(reason="Not currently pytest compatible")
def test_devshmsrc(capsys):
    parser = parse_command_line_datasource()
    options = parser.parse_args()

    pipeline = Pipeline()

    #
    #       -----------
    #      | DevShmSrc |
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

    source_out_links, _ = datasource_from_options(pipeline, options)

    ifos = tuple(source_out_links.keys())

    pipeline.insert(
        Align(
            name="trans1",
            sink_pad_names=ifos,
            source_pad_names=ifos,
        ),
        FakeSeriesSink(
            name="snk2",
            sink_pad_names=ifos,
        ),
        link_map={"trans1:sink:" + ifo: source_out_links[ifo] for ifo in ifos},
    )
    pipeline.insert(
        link_map={"snk2:sink:" + ifo: "trans1:src:" + ifo for ifo in ifos},
    )

    pipeline.run()


if __name__ == "__main__":
    test_devshmsrc(None)
