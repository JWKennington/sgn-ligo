#!/usr/bin/env python3
from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink
from sgnligo.transforms import Whiten
from sgnligo.sources import FrameReader
from sgnts.transforms import Resampler
import os

def test_whitengraph(capsys): 

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

    output_dir = '/home/joshua.gonsalves/sgn-ligo/seg_data'
    os.makedirs(output_dir, exist_ok=True)
    
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
             Whiten(
               name = "whiten",
               source_pad_names = ("H1","H2"),
               sink_pad_names = ("H1",),
               whitening_method = "gstlal",
               ref_psd = "/home/joshua.gonsalves/sgn-ligo/psd_geo.txt",
               psd_pad_name = "whiten:src:H2"
             ),
             DumpSeriesSink(
               name = "snk1",
               sink_pad_names = ("H1",),
               fname = 'out.txt'
             ),
             DumpSeriesSink(
                 name = "snk3",
                 sink_pad_names = ("H2",),
                 fname = 'psd_out.txt'
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

