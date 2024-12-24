"""Test for the FrameSink class"""

import pathlib
from tempfile import TemporaryDirectory
from unittest import mock

import numpy
from sgn.apps import Pipeline
from sgnts.base import AdapterConfig, Offset
from sgnts.sources import FakeSeriesSource

from sgnligo.sinks import FrameSink

_count1 = 0
MOCK_TIMES = list(numpy.arange(0.0, 20.0, 0.5))


def mock_gpsnow():
    """Mock time_now method for RealTimeWhiteNoiseSource
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
              --------------------        --------------------
              | FakeSeriesSource |        | FakeSeriesSource |
              --------------------        --------------------
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

            # Mock the gpsnow function for FakeSeriesSource for reproducibility
            with mock.patch(
                "sgnts.sources.fake_series.gpsnow",
                mock_gpsnow,
            ):

                # Run pipeline
                pipeline.insert(
                    FakeSeriesSource(
                        name="src_H1",
                        source_pad_names=("H1",),
                        rate=256,
                        t0=t0,
                        end=2 * duration,
                        real_time=True,
                    ),
                    FakeSeriesSource(
                        name="src_L1",
                        source_pad_names=("L1",),
                        rate=512,
                        t0=t0,
                        end=2 * duration,
                        real_time=True,
                    ),
                    FrameSink(
                        name="snk",
                        channels=(
                            "H1:FOO-BAR",
                            "L1:BAZ-QUX_0",
                        ),
                        duration=duration,
                        path=path_format.as_posix(),
                    ),
                    link_map={
                        "snk:snk:H1:FOO-BAR": "src_H1:src:H1",
                        "snk:snk:L1:BAZ-QUX_0": "src_L1:src:L1",
                    },
                )
                pipeline.run()

            # Verify the files exist
            assert out1.exists()
            assert out2.exists()
