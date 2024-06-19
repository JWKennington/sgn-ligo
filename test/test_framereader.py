#!/usr/bin/env python3
from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink
from sgnligo.sources import FrameReader

from sgnts.transforms import Resampler

def test_framereader(capsys): 

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
               rate=16384,
               num_samples=16384,
               framecache="test/gw190425.cache",
               channel_name = ("GWOSC-16KHZ_R1_STRAIN",),
               instruments = ("L1",),
             ),
             Resampler(
               name="trans1",
               source_pad_names=("H1",),
               sink_pad_names=("H1",),
               inrate=16384,
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



