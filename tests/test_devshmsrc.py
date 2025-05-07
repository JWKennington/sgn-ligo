import shutil
import time
from pathlib import Path
from threading import Thread

import pytest
from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgnts.transforms import Gate

from sgnligo.sources import DevShmSource
from sgnligo.transforms import BitMask


@pytest.mark.freeze_time("2019-04-25 08:17:49", tick=True)
def test_devshmsrc(tmp_path):
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

    datadir = Path(__file__).parent / "data"
    test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

    # add frame to "shmdir" to determine sampling rates
    dest_path = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215465-32.gwf"
    shutil.copyfile(datadir / test_frame, dest_path)

    # create pipeline
    hoft = "L1:GWOSC-16KHZ_R1_STRAIN"
    dqmask = "L1:GWOSC-16KHZ_R1_DQMASK"

    src = DevShmSource(
        name="src",
        channel_names=[hoft, dqmask],
        shared_memory_dirs=str(tmp_path),
        duration=32,
    )
    mask = BitMask(
        name="mask",
        sink_pad_names=(dqmask,),
        source_pad_names=("dqmask",),
        bit_mask=0,
    )
    gate = Gate(
        name="gate",
        source_pad_names=("strain",),
        sink_pad_names=(hoft, "dqmask"),
        control="dqmask",
    )
    sink = NullSink(
        name="sink",
        sink_pad_names=("strain",),
    )

    pipeline.insert(
        src,
        mask,
        gate,
        sink,
        link_map={
            mask.snks[dqmask]: src.srcs[dqmask],
            gate.snks[hoft]: src.srcs[hoft],
            gate.snks["dqmask"]: mask.srcs["dqmask"],
            sink.snks["strain"]: gate.srcs["strain"],
        },
    )

    # add frame to simulate live data after a short period of time
    def populate_frame():
        time.sleep(3)
        shutil.copyfile(datadir / test_frame, tmp_path / test_frame)

    thread = Thread(target=populate_frame)
    thread.start()

    # start pipeline
    pipeline.run()
    thread.join()
