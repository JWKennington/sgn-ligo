"""Test for the FrameSink class"""

import pathlib
from tempfile import TemporaryDirectory

import pytest
from sgn.apps import Pipeline
from sgnts.sources import FakeSeriesSource

from sgnligo.sinks import FrameSink


def test_frame_sink():
    with pytest.raises(RuntimeError):
        FrameSink(
            adapter_config="config",
            name="snk",
            channels=(
                "H1:FOO-BAR",
                "L1:BAZ-QUX_0",
            ),
            duration=256,
            description="testing",
        )

    with pytest.raises(ValueError):
        FrameSink(
            name="snk",
            channels=(
                "H1:FOO-BAR",
                "L1:BAZ-QUX_0",
            ),
            duration=256.0,
            description="testing",
        )

    with pytest.raises(ValueError):
        FrameSink(
            name="snk",
            channels=(
                "H1:FOO-BAR",
                "L1:BAZ-QUX_0",
            ),
            duration=256,
            description="testing",
            path="",
        )


class TestFrameSink:
    """Test group for framesink class"""

    @pytest.mark.freeze_time("1980-01-06 00:00:00", auto_tick_seconds=0.5)
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

        with TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir)
            path_format = path / (
                "{instruments}-{description}-{gps_start_time}-{" "duration}.gwf"
            )
            out1 = path / "H1L1-testing-0000000003-3.gwf"
            out2 = path / "H1L1-testing-0000000000-3.gwf"

            # Verify the files do not exist
            assert not out1.exists()
            assert not out2.exists()

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
                    description="testing",
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

    # There is a bug in the framewriter so this hangs
    @pytest.mark.skip
    def test_frame_sink_not_enough_data(self):
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

        with TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir)
            path_format = path / (
                "{instruments}-{description}-{gps_start_time}-{" "duration}.gwf"
            )
            out1 = path / "H1L1-testing-0000000003-3.gwf"
            out2 = path / "H1L1-testing-0000000000-3.gwf"

            # Verify the files do not exist
            assert not out1.exists()
            assert not out2.exists()

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
                    duration=duration * 10,
                    path=path_format.as_posix(),
                    description="testing",
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

    @pytest.mark.freeze_time("1980-01-06 00:00:00", auto_tick_seconds=0.5)
    def test_frame_sink_path_exists_force(self):
        r"""Test the frame sink with two different rate sources


        The pipeline is as follows:
              --------------------        --------------------
              | FakeSeriesSource |        | FakeSeriesSource |
              --------------------        --------------------
                        |         \       /        |
                        |          \     /         |
                        |           \   /          |
                        |            \ /           |
                        |             \            |
                        |            / \           |
                        |           /   \          |
                      ----------------  ---------------
                      | FrameWriter  |  | FrameWriter |
                      ----------------  ---------------
        """

        pipeline = Pipeline()
        t0 = 0.0
        duration = 3  # seconds

        if True:  # with pytest.raises(FileExistsError):
            with TemporaryDirectory() as tmpdir:
                path = pathlib.Path(tmpdir)
                path_format = path / (
                    "{instruments}-{description}-{gps_start_time}-{" "duration}.gwf"
                )
                out1 = path / "H1L1-testing-0000000003-3.gwf"
                out2 = path / "H1L1-testing-0000000000-3.gwf"

                # Verify the files do not exist
                assert not out1.exists()
                assert not out2.exists()

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
                        description="testing",
                        force=True,
                    ),
                    FrameSink(
                        name="snk2",
                        channels=(
                            "H1:FOO-BAR",
                            "L1:BAZ-QUX_0",
                        ),
                        duration=duration,
                        path=path_format.as_posix(),
                        description="testing",
                        force=True,
                    ),
                    link_map={
                        "snk:snk:H1:FOO-BAR": "src_H1:src:H1",
                        "snk:snk:L1:BAZ-QUX_0": "src_L1:src:L1",
                        "snk2:snk:H1:FOO-BAR": "src_H1:src:H1",
                        "snk2:snk:L1:BAZ-QUX_0": "src_L1:src:L1",
                    },
                )
                pipeline.run()

                # Verify the files exist
                assert out1.exists()
                assert out2.exists()

    @pytest.mark.freeze_time("1980-01-06 00:00:00", auto_tick_seconds=0.5)
    def test_frame_sink_path_exists(self):
        r"""Test the frame sink with two different rate sources


        The pipeline is as follows:
              --------------------        --------------------
              | FakeSeriesSource |        | FakeSeriesSource |
              --------------------        --------------------
                        |         \       /        |
                        |          \     /         |
                        |           \   /          |
                        |            \ /           |
                        |             \            |
                        |            / \           |
                        |           /   \          |
                      ----------------  ---------------
                      | FrameWriter  |  | FrameWriter |
                      ----------------  ---------------
        """

        pipeline = Pipeline()
        t0 = 0.0
        duration = 3  # seconds

        with pytest.raises(FileExistsError):
            with TemporaryDirectory() as tmpdir:
                path = pathlib.Path(tmpdir)
                path_format = path / (
                    "{instruments}-{description}-{gps_start_time}-{" "duration}.gwf"
                )
                out1 = path / "H1L1-testing-0000000003-3.gwf"
                out2 = path / "H1L1-testing-0000000000-3.gwf"

                # Verify the files do not exist
                assert not out1.exists()
                assert not out2.exists()

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
                        description="testing",
                    ),
                    FrameSink(
                        name="snk2",
                        channels=(
                            "H1:FOO-BAR",
                            "L1:BAZ-QUX_0",
                        ),
                        duration=duration,
                        path=path_format.as_posix(),
                        description="testing",
                    ),
                    link_map={
                        "snk:snk:H1:FOO-BAR": "src_H1:src:H1",
                        "snk:snk:L1:BAZ-QUX_0": "src_L1:src:L1",
                        "snk2:snk:H1:FOO-BAR": "src_H1:src:H1",
                        "snk2:snk:L1:BAZ-QUX_0": "src_L1:src:L1",
                    },
                )
                pipeline.run()

                # Verify the files exist
                assert out1.exists()
                assert out2.exists()
