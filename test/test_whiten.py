#!/usr/bin/env python3
from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSrc
from sgnts.sinks import DumpSeriesSink
from sgnligo.transforms import Whiten

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

    
    
    pipeline.insert(FakeSeriesSrc(
               name = "src1",
               source_pad_names = ("H1",),
               num_buffers = 32,
               duration = 1,
               signal_type = 'white',
             ),
             Whiten(
               name = "whiten",
               source_pad_names = ("H1",),
               sink_pad_names = ("H1",),
               whitening_method = "gwpy",
             ),
             DumpSeriesSink(
               name = "snk1",
               sink_pad_names = ("H1",),
               fname = 'out.txt'
             )
    )

    pipeline.insert(DumpSeriesSink(
           name = "snk2",
           sink_pad_names = ("H1",),
           fname = 'in.txt'
         ))
    pipeline.insert(link_map={
                              "whiten:sink:H1":"src1:src:H1",
                              "snk1:sink:H1":"whiten:src:H1",
                              "snk2:sink:H1":"src1:src:H1"
                              })

    pipeline.run()

if __name__ == "__main__":
    test_whitengraph(None)

