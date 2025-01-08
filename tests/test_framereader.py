#!/usr/bin/env python3
import pathlib

from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgnts.sinks import DumpSeriesSink
from sgnts.transforms import Resampler

from sgnligo.sources import FrameReader

PATH_DATA = pathlib.Path(__file__).parent / "data"


def test_framereader():

    gps_start_time = 1240215487
    gps_end_time = 1240215519
    frame_cache = PATH_DATA / "gw190425.cache"
    channel_names = ["L1:GWOSC-16KHZ_R1_STRAIN", "L1:GWOSC-16KHZ_R1_DQMASK"]
    instrument = "L1"

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #       /      \
    #   DQ /        \ Strain
    #  ------   ------------
    # | Null | | Resampler  |
    #  ------   ------------
    #                 \
    #                  \
    #             ---------
    #            | snk1    |
    #             ---------

    src = FrameReader(
        name="src1",
        framecache=frame_cache.as_posix(),
        channel_names=channel_names,
        instrument=instrument,
        t0=gps_start_time,
        end=gps_end_time,
    )
    sample_rate = src.rates[channel_names[0]]
    pipeline.insert(
        src,
        Resampler(
            name="trans1",
            source_pad_names=(instrument,),
            sink_pad_names=(instrument,),
            inrate=sample_rate,
            outrate=2048,
        ),
        DumpSeriesSink(name="snk1", sink_pad_names=(instrument,), fname="strain.txt"),
        NullSink(name="snk2", sink_pad_names=("DQ",)),
    )

    pipeline.insert(
        link_map={
            "trans1:snk:" + instrument: "src1:src:L1:GWOSC-16KHZ_R1_STRAIN",
            "snk1:snk:" + instrument: "trans1:src:" + instrument,
            "snk2:snk:DQ": "src1:src:L1:GWOSC-16KHZ_R1_DQMASK",
        }
    )

    pipeline.run()


if __name__ == "__main__":
    test_framereader()
