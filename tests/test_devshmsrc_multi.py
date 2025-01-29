#!/usr/bin/env python3
from argparse import ArgumentParser

import pytest
from sgn.apps import Pipeline
from sgn.sources import SignalEOS
from sgnts.sinks import FakeSeriesSink
from sgnts.transforms import Align

from sgnligo.sources import DataSourceInfo, datasource


@pytest.mark.skip(reason="Not currently pytest compatible")
def test_devshmsrc_multi(capsys):
    parser = ArgumentParser()
    DataSourceInfo.append_options(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    options = parser.parse_args()

    data_source_info = DataSourceInfo.from_options(options)

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

    source_out_links, _ = datasource(pipeline, data_source_info, False, options.verbose)

    ifos = data_source_info.ifos

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
        link_map={"trans1:snk:" + ifo: source_out_links[ifo] for ifo in ifos},
    )
    pipeline.insert(
        link_map={"snk2:snk:" + ifo: "trans1:src:" + ifo for ifo in ifos},
    )

    with SignalEOS():
        pipeline.run()


if __name__ == "__main__":
    test_devshmsrc_multi(None)
