"""Test for the FrameSink class"""

import pathlib
from tempfile import TemporaryDirectory
from unittest import mock

import numpy

from sgnligo.sinks import FrameSink
from sgnts.base import AdapterConfig, Offset
from sgnts.sources import RealTimeWhiteNoiseSrc

from sgn.apps import Pipeline


_count1 = 0
MOCK_TIMES = list(numpy.arange(0.0, 20.0, 0.5))


def mock_time_now1(self):
    """Mock time_now method for RealTimeWhiteNoiseSrc
    ONLY to be used for TestFrameSink.test_frame_sink,
    since it counts the number of calls to time_now
    """
    global _count1
    idx = _count1 % len(MOCK_TIMES)
    res = MOCK_TIMES[idx]
    _count1 += 1
    return res


class TestFrameSink:
    """Test group for framesink class"""

    def test_frame_sink(self):
        """Test the frame sink with two different rate sources


        The pipeline is as follows:
              ---------------------------        --------------------------
              | RealTimeWhiteNoiseSrc	|        | RealTimeWhiteNoiseSrc  |
              ---------------------------        --------------------------
                                      |            |
                                     ----------------
                                     | FrameWriter  |
                                     ----------------
        """

        pipeline = Pipeline()
        t0 = 0.0
        duration = 3  # seconds
        duration_offsets = Offset.fromsec(duration)

        with TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir)
            path_format = path / "{channels}-{gps_start_time}-{duration}.gwf"
            out1 = path / "H1L1-3000000000-3.gwf"
            out2 = path / "H1L1-0-3.gwf"

            # Verify the files do not exist
            assert not out1.exists()
            assert not out2.exists()

            # Mock the time_now method of RealTimeWhiteNoiseSrc for reproducibility
            with mock.patch(
                "sgnligo.sources.fake_realtime.RealTimeWhiteNoiseSrc.time_now",
                mock_time_now1,
            ):

                # Run pipeline
                pipeline.insert(
                    RealTimeWhiteNoiseSrc(
                        name="src_H1",
                        source_pad_names=("H1",),
                        rate=256,
                        t0=t0,
                        duration=2 * duration,
                    ),
                    RealTimeWhiteNoiseSrc(
                        name="src_L1",
                        source_pad_names=("L1",),
                        rate=512,
                        t0=t0,
                        duration=2 * duration,
                    ),
                    FrameSink(
                        name="snk",
                        channels=(
                            "H1",
                            "L1",
                        ),
                        duration=duration,
                        path=path_format.as_posix(),
                        adapter_config=AdapterConfig(stride=duration_offsets),
                    ),
                    link_map={
                        "snk:sink:H1": "src_H1:src:H1",
                        "snk:sink:L1": "src_L1:src:L1",
                    },
                )
                pipeline.run()

            # Verify the files exist
            assert out1.exists()
            assert out2.exists()
