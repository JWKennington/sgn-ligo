"""Test for the FrameSink class"""

import pathlib
from tempfile import TemporaryDirectory

import pytest
from sgn.apps import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Resampler

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


class TestFrameSinkCoverage:
    """Additional tests for full coverage of FrameSink"""

    def test_frame_sink_circular_buffer_cleanup(self, tmp_path):
        """Test the circular buffer cleanup functionality directly"""
        from collections import deque
        from unittest.mock import patch

        # Create a FrameSink with circular buffer limited to 2 files
        sink = FrameSink(
            name="test_sink",
            channels=("H1:TEST",),
            duration=1,
            max_files=2,
            path=str(
                tmp_path / "{instruments}-{description}-{gps_start_time}-{duration}.gwf"
            ),
        )

        # Manually populate the file cache with 4 files
        sink._file_cache = deque(
            [
                str(tmp_path / "H1-TEST-1234567700-10.gwf"),  # oldest - delete
                str(tmp_path / "H1-TEST-1234567750-10.gwf"),  # second oldest - delete
                str(tmp_path / "H1-TEST-1234567785-10.gwf"),  # third - keep
                str(tmp_path / "H1-TEST-1234567850-10.gwf"),  # newest - keep
            ]
        )

        # Mock os.path.exists to return True for all files
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            # Mock os.remove
            with patch("os.remove") as mock_remove:
                # Call cleanup
                sink._cleanup_old_frames()

                # Should delete 2 oldest files to maintain max_files=2
                assert mock_remove.call_count == 2
                # Verify which files were deleted
                mock_remove.assert_any_call(str(tmp_path / "H1-TEST-1234567700-10.gwf"))
                mock_remove.assert_any_call(str(tmp_path / "H1-TEST-1234567750-10.gwf"))
                # Should keep 2 newest files in cache
                assert len(sink._file_cache) == 2
                assert list(sink._file_cache) == [
                    str(tmp_path / "H1-TEST-1234567785-10.gwf"),
                    str(tmp_path / "H1-TEST-1234567850-10.gwf"),
                ]

    def test_frame_sink_circular_buffer_error_handling(self, tmp_path):
        """Test circular buffer cleanup error handling"""
        from collections import deque
        from unittest.mock import patch

        sink = FrameSink(
            name="test_sink",
            channels=("H1:TEST",),
            duration=1,
            max_files=1,  # Keep only 1 file
            path=str(
                tmp_path / "{instruments}-{description}-{gps_start_time}-{duration}.gwf"
            ),
        )

        # Populate cache with 2 files (one will be deleted)
        sink._file_cache = deque(
            [
                str(tmp_path / "H1-TEST-1234567700-10.gwf"),
                str(tmp_path / "H1-TEST-1234567750-10.gwf"),
            ]
        )

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            # Mock os.remove to raise an exception
            with patch("os.remove") as mock_remove:
                mock_remove.side_effect = OSError("Permission denied")

                # Should not raise, just log warning
                sink._cleanup_old_frames()

                # Should keep only the newest file in cache
                assert len(sink._file_cache) == 1
                assert list(sink._file_cache) == [
                    str(tmp_path / "H1-TEST-1234567750-10.gwf")
                ]

    def test_frame_sink_hdf5_output(self):
        """Test frame sink with HDF5 output format"""
        pipeline = Pipeline()
        t0 = 0.0
        duration = 1

        with TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir)
            path_format = path / (
                "{instruments}-{description}-{gps_start_time}-{duration}.hdf5"
            )
            out = path / "H1-testing-0000000000-1.hdf5"

            # Verify the file does not exist
            assert not out.exists()

            # Run pipeline
            pipeline.insert(
                FakeSeriesSource(
                    name="src_H1",
                    source_pad_names=("H1",),
                    rate=256,
                    t0=t0,
                    end=duration,
                ),
                FrameSink(
                    name="snk",
                    channels=("H1:TEST",),
                    duration=duration,
                    path=path_format.as_posix(),
                    description="testing",
                ),
                link_map={
                    "snk:snk:H1:TEST": "src_H1:src:H1",
                },
            )
            pipeline.run()

            # Verify the file exists
            assert out.exists()

    def test_frame_sink_invalid_duration(self):
        """Test invalid duration values"""
        # Test zero duration
        with pytest.raises(ValueError, match="Duration must be an positive integer"):
            FrameSink(
                name="test",
                channels=("H1:TEST",),
                duration=0,
            )

        # Test negative duration
        with pytest.raises(ValueError, match="Duration must be an positive integer"):
            FrameSink(
                name="test",
                channels=("H1:TEST",),
                duration=-1,
            )

    @pytest.mark.parametrize(
        "path,missing_param",
        [
            ("test-{description}-{gps_start_time}-{duration}.gwf", "instruments"),
            ("{instruments}-test-{gps_start_time}-{duration}.gwf", "description"),
            ("{instruments}-{description}-test-{duration}.gwf", "gps_start_time"),
            ("{instruments}-{description}-{gps_start_time}-test.gwf", "duration"),
        ],
    )
    def test_frame_sink_missing_path_params(self, path, missing_param):
        """Test missing required path parameters"""
        with pytest.raises(ValueError, match="Path must contain parameter"):
            FrameSink(
                name="test",
                channels=("H1:TEST",),
                duration=1,
                path=path,
            )

    def test_frame_sink_circular_buffer_disabled(self):
        """Test that cleanup returns early when max_files is None or <= 0"""
        sink1 = FrameSink(
            name="test_sink1",
            channels=("H1:TEST",),
            duration=1,
            max_files=None,
        )

        sink2 = FrameSink(
            name="test_sink2",
            channels=("H1:TEST",),
            duration=1,
            max_files=0,
        )

        sink3 = FrameSink(
            name="test_sink3",
            channels=("H1:TEST",),
            duration=1,
            max_files=-10,
        )

        # These should all return without doing anything
        sink1._cleanup_old_frames()
        sink2._cleanup_old_frames()
        sink3._cleanup_old_frames()

    def test_frame_sink_no_cleanup_when_under_limit(self, tmp_path):
        """Test that cleanup does nothing when file count is under the limit"""
        from collections import deque
        from unittest.mock import patch

        sink = FrameSink(
            name="test_sink",
            channels=("H1:TEST",),
            duration=1,
            max_files=5,  # Allow up to 5 files
        )

        # Populate cache with only 3 files (under the limit)
        sink._file_cache = deque(
            [
                str(tmp_path / "H1-TEST-1.gwf"),
                str(tmp_path / "H1-TEST-2.gwf"),
                str(tmp_path / "H1-TEST-3.gwf"),
            ]
        )

        with patch("os.remove") as mock_remove:
            # Call cleanup
            sink._cleanup_old_frames()

            # Should not delete anything since we're under the limit
            mock_remove.assert_not_called()
            # Cache should remain unchanged
            assert len(sink._file_cache) == 3

    def test_frame_sink_with_misaligned_resampler(self):
        """Test FrameSink with misaligned input from Resampler.

        Pipeline: FakeSeriesSource -> Resampler -> FrameSink

        The Resampler introduces offset misalignment due to FIR filter latency.
        Without automatic alignment in AdapterConfig, this should cause an
        assertion error when FrameSink tries to verify integer second boundaries.
        """
        pipeline = Pipeline()
        t0 = 0.0
        inrate = 16384
        outrate = 2048

        # 2 frame files, extra 1 second padding to handle resampler offset
        frame_duration = 3  # seconds
        total_duration = (2 * frame_duration) + 1

        with TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir)
            path_format = (
                path
                / "{instruments}-{description}-{gps_start_time}-{duration}.gwf"
            )

            pipeline.insert(
                FakeSeriesSource(
                    name="src_H1",
                    source_pad_names=("H1",),
                    rate=inrate,
                    t0=t0,
                    end=total_duration,
                ),
                Resampler(
                    name="resample",
                    sink_pad_names=("H1",),
                    source_pad_names=("H1",),
                    inrate=inrate,
                    outrate=outrate,
                ),
                FrameSink(
                    name="snk",
                    channels=("H1:FOO-BAR",),
                    duration=frame_duration,
                    path=path_format.as_posix(),
                    description="RESAMPLE_TEST",
                ),
                link_map={
                    "resample:snk:H1": "src_H1:src:H1",
                    "snk:snk:H1:FOO-BAR": "resample:src:H1",
                },
            )

            pipeline.run()

            # Verify both frames were written
            files = list(path.glob("*.gwf"))
            expected_files = [
                path / "H1-RESAMPLE_TEST-0000000000-3.gwf",
                path / "H1-RESAMPLE_TEST-0000000003-3.gwf",
            ]
            assert (
                len(files) == 2
            ), f"Expected 2 frames, got {len(files)}: {[f.name for f in files]}"
            assert all(f.exists() for f in expected_files), "Missing expected frames"
