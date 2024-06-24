#!/usr/bin/env python3
from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink
from sgnligo.sources import DevShmSrc

from sgnts.transforms import Resampler

def test_devshmsrc(capsys): 

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
               channel_name = "GDS-CALIB_STRAIN_O3Replay",
               instrument = "L1",
               shared_memory_dir = "/dev/shm/kafka/L1_O3ReplayMDC",
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



