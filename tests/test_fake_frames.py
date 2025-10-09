#!/usr/bin/env python3
"""Test coverage for sgnligo.bin.fake_frames module."""

from argparse import Namespace
from unittest.mock import Mock, patch

import pytest

# Import the module to test
from sgnligo.base import read_segments_and_values_from_file
from sgnligo.bin.fake_frames import generate_fake_frames, main, parse_command_line


class TestParseCommandLine:
    """Test command line parsing."""

    def test_parse_minimal_args(self):
        """Test parsing with minimal required arguments."""
        with patch(
            "sys.argv",
            ["fake_frames", "--state-file", "segments.txt"],
        ):
            args = parse_command_line()
            assert args.state_file == "segments.txt"
            assert args.state_channel == "L1:FAKE-STATE_VECTOR"  # default
            assert args.strain_channel == "L1:FAKE-STRAIN"  # default
            assert args.state_sample_rate == 16  # default
            assert args.strain_sample_rate == 16384  # default
            assert args.frame_duration == 16  # default
            assert args.cleanup_interval == 300  # default

    def test_parse_all_args(self):
        """Test parsing with all arguments."""
        with patch(
            "sys.argv",
            [
                "fake_frames",
                "--state-file",
                "segments.txt",
                "--state-channel",
                "H1:TEST-STATE",
                "--strain-channel",
                "H1:TEST-STRAIN",
                "--state-sample-rate",
                "32",
                "--strain-sample-rate",
                "4096",
                "--frame-duration",
                "8",
                "--gps-start-time",
                "1000000000",
                "--gps-end-time",
                "1000000100",
                "--duration",
                "100",
                "--output-path",
                "test/{instruments}-{description}-{gps_start_time}-{duration}.gwf",
                "--description",
                "TEST",
                "--verbose",
                "--real-time",
                "--history",
                "600",
                "--cleanup-interval",
                "60",
            ],
        ):
            args = parse_command_line()
            assert args.state_file == "segments.txt"
            assert args.state_channel == "H1:TEST-STATE"
            assert args.strain_channel == "H1:TEST-STRAIN"
            assert args.state_sample_rate == 32
            assert args.strain_sample_rate == 4096
            assert args.frame_duration == 8
            assert args.gps_start_time == 1000000000
            assert args.gps_end_time == 1000000100
            assert args.duration == 100
            assert (
                args.output_path
                == "test/{instruments}-{description}-{gps_start_time}-{duration}.gwf"
            )
            assert args.description == "TEST"
            assert args.verbose is True
            assert args.real_time is True
            assert args.history == 600
            assert args.cleanup_interval == 60

    def test_required_state_file(self):
        """Test that segment file is required."""
        with patch("sys.argv", ["fake_frames"]):
            with pytest.raises(SystemExit):
                parse_command_line()


class TestReadSegmentsAndValuesFromFile:
    """Test segment and value file reading."""

    def test_read_single_segment(self, tmp_path):
        """Test reading a file with a single segment."""
        state_file = tmp_path / "segments.txt"
        state_file.write_text("1000000000 1000000010 3\n")

        segments, values = read_segments_and_values_from_file(str(state_file))
        assert len(segments) == 1
        assert len(values) == 1
        assert segments[0] == (1000000000000000000, 1000000010000000000)  # nanoseconds
        assert values[0] == 3

    def test_read_multiple_segments(self, tmp_path):
        """Test reading a file with multiple segments."""
        state_file = tmp_path / "segments.txt"
        state_file.write_text(
            "1000000000 1000000010 1\n"
            "1000000010 1000000020 2\n"
            "1000000020 1000000030 3\n"
        )

        segments, values = read_segments_and_values_from_file(str(state_file))
        assert len(segments) == 3
        assert len(values) == 3
        assert segments[0] == (1000000000000000000, 1000000010000000000)
        assert segments[1] == (1000000010000000000, 1000000020000000000)
        assert segments[2] == (1000000020000000000, 1000000030000000000)
        assert values == (1, 2, 3)

    def test_read_segments_verbose(self, tmp_path, capsys):
        """Test verbose output when reading segments."""
        state_file = tmp_path / "segments.txt"
        state_file.write_text("1000000000 1000000010 5\n")

        segments, values = read_segments_and_values_from_file(
            str(state_file), verbose=True
        )
        captured = capsys.readouterr()
        assert f"Reading segments and values from {state_file}" in captured.out
        assert "Segment 1: 1000000000.0s - 1000000010.0s, Value: 5" in captured.out

    def test_read_segments_single_row(self, tmp_path):
        """Test reading a file with a single row (edge case)."""
        state_file = tmp_path / "segments.txt"
        state_file.write_text("1000000000 1000000010 7")  # No newline

        segments, values = read_segments_and_values_from_file(str(state_file))
        assert len(segments) == 1
        assert values[0] == 7

    def test_read_segments_wrong_columns(self, tmp_path):
        """Test error when file has wrong number of columns."""
        state_file = tmp_path / "segments.txt"
        state_file.write_text("1000000000 1000000010\n")  # Only 2 columns

        with pytest.raises(ValueError, match="must have 3 columns"):
            read_segments_and_values_from_file(str(state_file))

    def test_read_segments_with_comments(self, tmp_path):
        """Test reading a file with comments."""
        state_file = tmp_path / "segments.txt"
        state_file.write_text(
            "# This is a comment\n"
            "1000000000 1000000010 1\n"
            "# Another comment\n"
            "1000000010 1000000020 2\n"
        )

        segments, values = read_segments_and_values_from_file(str(state_file))
        assert len(segments) == 2
        assert values == (1, 2)


class TestGenerateFakeFrames:
    """Test frame generation."""

    @patch("sgnligo.bin.fake_frames.Pipeline")
    def test_generate_fake_frames_basic(self, mock_pipeline):
        """Test basic frame generation."""
        options = Namespace(
            ifo="L1",
            state_channel="L1:FAKE-STATE_VECTOR",
            strain_channel="L1:FAKE-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            gps_start_time=1000000000,
            gps_end_time=1000000016,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="TEST",
            verbose=False,
            real_time=False,
            history=None,
            cleanup_interval=300,
        )

        segments = ((1000000000000000000, 1000000016000000000),)
        values = (3,)

        # Mock pipeline instance
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance

        generate_fake_frames(
            state_channel=options.state_channel,
            strain_channel=options.strain_channel,
            state_sample_rate=options.state_sample_rate,
            strain_sample_rate=options.strain_sample_rate,
            frame_duration=options.frame_duration,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
            output_path=options.output_path,
            description=options.description,
            segments=segments,
            values=values,
            verbose=options.verbose,
            real_time=options.real_time,
            history=options.history,
            cleanup_interval=options.cleanup_interval,
        )

        # Check pipeline was created and run
        mock_pipeline.assert_called_once()
        mock_pipe_instance.insert.assert_called()
        mock_pipe_instance.run.assert_called_once()

    @patch("sgnligo.bin.fake_frames.Pipeline")
    @patch("sgnligo.bin.fake_frames.SegmentSource")
    @patch("sgnligo.bin.fake_frames.GWDataNoiseSource")
    @patch("sgnligo.bin.fake_frames.FrameSink")
    def test_generate_fake_frames_with_defaults(
        self, mock_frame_sink, mock_noise_source, mock_segment_source, mock_pipeline
    ):
        """Test frame generation with default channel names."""
        options = Namespace(
            ifo="H1",
            state_channel="H1:FAKE-STATE_VECTOR",
            strain_channel="H1:FAKE-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            gps_start_time=1000000000,
            gps_end_time=1000000080,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="TEST",
            verbose=False,
            real_time=False,
            history=None,
            cleanup_interval=300,
        )

        segments = ((1000000000000000000, 1000000080000000000),)
        values = (1,)

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_segment_instance = Mock()
        mock_segment_source.return_value = mock_segment_instance
        mock_noise_instance = Mock()
        mock_noise_source.return_value = mock_noise_instance
        mock_sink_instance = Mock()
        mock_frame_sink.return_value = mock_sink_instance

        generate_fake_frames(
            state_channel=options.state_channel,
            strain_channel=options.strain_channel,
            state_sample_rate=options.state_sample_rate,
            strain_sample_rate=options.strain_sample_rate,
            frame_duration=options.frame_duration,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
            output_path=options.output_path,
            description=options.description,
            segments=segments,
            values=values,
            verbose=options.verbose,
            real_time=options.real_time,
            history=options.history,
            cleanup_interval=options.cleanup_interval,
        )

        # Check components were created with correct parameters
        mock_segment_source.assert_called_once()
        assert mock_segment_source.call_args[1]["rate"] == 16
        assert mock_segment_source.call_args[1]["segments"] == segments
        assert mock_segment_source.call_args[1]["values"] == values

        mock_noise_source.assert_called_once()
        assert mock_noise_source.call_args[1]["channel_dict"] == {
            "H1": "H1:FAKE-STRAIN"
        }
        assert mock_noise_source.call_args[1]["t0"] == 1000000000
        assert mock_noise_source.call_args[1]["end"] == 1000000080

        mock_frame_sink.assert_called_once()
        assert mock_frame_sink.call_args[1]["channels"] == [
            "H1:FAKE-STATE_VECTOR",
            "H1:FAKE-STRAIN",
        ]

    @patch("sgnligo.bin.fake_frames.Pipeline")
    @patch("sgnligo.bin.fake_frames.Resampler")
    def test_generate_fake_frames_with_resampler(self, mock_resampler, mock_pipeline):
        """Test frame generation with resampler for non-16384 Hz strain."""
        options = Namespace(
            ifo="L1",
            state_channel="L1:TEST-STATE",
            strain_channel="L1:TEST-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=4096,  # Requires resampling
            frame_duration=16,
            gps_start_time=1000000000,
            gps_end_time=1000000016,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="TEST",
            verbose=False,
            real_time=False,
            history=None,
            cleanup_interval=300,
        )

        segments = ((1000000000000000000, 1000000016000000000),)
        values = (1,)

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_resampler_instance = Mock()
        mock_resampler.return_value = mock_resampler_instance

        generate_fake_frames(
            state_channel=options.state_channel,
            strain_channel=options.strain_channel,
            state_sample_rate=options.state_sample_rate,
            strain_sample_rate=options.strain_sample_rate,
            frame_duration=options.frame_duration,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
            output_path=options.output_path,
            description=options.description,
            segments=segments,
            values=values,
            verbose=options.verbose,
            real_time=options.real_time,
            history=options.history,
            cleanup_interval=options.cleanup_interval,
        )

        # Check resampler was created
        mock_resampler.assert_called_once()
        assert mock_resampler.call_args[1]["inrate"] == 16384
        assert mock_resampler.call_args[1]["outrate"] == 4096

    @patch("sgnligo.bin.fake_frames.Pipeline")
    def test_generate_fake_frames_verbose(self, mock_pipeline, capsys):
        """Test verbose output during frame generation."""
        options = Namespace(
            ifo="V1",
            state_channel="V1:TEST-STATE",
            strain_channel="V1:TEST-STRAIN",
            state_sample_rate=32,
            strain_sample_rate=8192,
            frame_duration=8,
            gps_start_time=1000000000,
            gps_end_time=1000000016,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="TEST",
            verbose=True,
            real_time=False,
            history=None,
            cleanup_interval=300,
        )

        segments = ((1000000000000000000, 1000000016000000000),)
        values = (7,)

        # Mock pipeline instance
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance

        generate_fake_frames(
            state_channel=options.state_channel,
            strain_channel=options.strain_channel,
            state_sample_rate=options.state_sample_rate,
            strain_sample_rate=options.strain_sample_rate,
            frame_duration=options.frame_duration,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
            output_path=options.output_path,
            description=options.description,
            segments=segments,
            values=values,
            verbose=options.verbose,
            real_time=options.real_time,
            history=options.history,
            cleanup_interval=options.cleanup_interval,
        )

        captured = capsys.readouterr()
        assert "Creating pipeline with:" in captured.out
        assert "IFO: V1" in captured.out
        assert "State channel: V1:TEST-STATE" in captured.out
        assert "Strain channel: V1:TEST-STRAIN" in captured.out
        assert "State sample rate: 32 Hz" in captured.out
        assert "Strain sample rate: 8192 Hz" in captured.out
        assert "Frame duration: 8 seconds" in captured.out
        assert "Running pipeline..." in captured.out
        assert "Pipeline execution completed." in captured.out
        assert "Expected number of frames: 2" in captured.out

    @patch("sgnligo.bin.fake_frames.Pipeline")
    def test_generate_fake_frames_real_time(self, mock_pipeline, capsys):
        """Test real-time mode settings."""
        options = Namespace(
            ifo="L1",
            state_channel="L1:TEST-STATE",
            strain_channel="L1:TEST-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=1,
            gps_start_time=1000000000,
            gps_end_time=None,  # Real-time mode
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="RT",
            verbose=True,
            real_time=True,
            history=300,
            cleanup_interval=300,
        )

        segments = ((1000000000000000000, 2000000000000000000),)
        values = (3,)

        # Mock pipeline instance
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance

        generate_fake_frames(
            state_channel=options.state_channel,
            strain_channel=options.strain_channel,
            state_sample_rate=options.state_sample_rate,
            strain_sample_rate=options.strain_sample_rate,
            frame_duration=options.frame_duration,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
            output_path=options.output_path,
            description=options.description,
            segments=segments,
            values=values,
            verbose=options.verbose,
            real_time=options.real_time,
            history=options.history,
            cleanup_interval=options.cleanup_interval,
        )

        captured = capsys.readouterr()
        assert "Real-time mode: will run indefinitely" in captured.out

    @patch("sgnligo.bin.fake_frames.Pipeline")
    def test_generate_fake_frames_with_segments(self, mock_pipeline, capsys):
        """Test frame generation with multiple segments."""
        options = Namespace(
            ifo="L1",
            state_channel="L1:TEST-STATE",
            strain_channel="L1:TEST-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            gps_start_time=1000000000,
            gps_end_time=1000000048,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="TEST",
            verbose=True,
            real_time=False,
            history=None,
            cleanup_interval=300,
        )

        segments = (
            (1000000000000000000, 1000000016000000000),
            (1000000016000000000, 1000000032000000000),
            (1000000032000000000, 1000000048000000000),
        )
        values = (1, 2, 3)

        # Mock pipeline instance
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance

        generate_fake_frames(
            state_channel=options.state_channel,
            strain_channel=options.strain_channel,
            state_sample_rate=options.state_sample_rate,
            strain_sample_rate=options.strain_sample_rate,
            frame_duration=options.frame_duration,
            gps_start_time=options.gps_start_time,
            gps_end_time=options.gps_end_time,
            output_path=options.output_path,
            description=options.description,
            segments=segments,
            values=values,
            verbose=options.verbose,
            real_time=options.real_time,
            history=options.history,
            cleanup_interval=options.cleanup_interval,
        )

        captured = capsys.readouterr()
        assert "Segments provided: 3" in captured.out
        assert "First segment: GPS 1000000000.0 - 1000000016.0" in captured.out

    def test_generate_fake_frames_ifo_mismatch(self):
        """Test IFO mismatch between state and strain channels raises ValueError."""
        options = Namespace(
            ifo="L1",
            state_channel="L1:FAKE-STATE_VECTOR",
            strain_channel="H1:FAKE-STRAIN",  # Different IFO
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            gps_start_time=1000000000,
            gps_end_time=1000000016,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="TEST",
            verbose=False,
            real_time=False,
            history=None,
            cleanup_interval=300,
        )

        segments = ((1000000000000000000, 1000000016000000000),)
        values = (1,)

        with pytest.raises(
            ValueError,
            match="IFO mismatch: state channel uses L1, strain channel uses H1",
        ):
            generate_fake_frames(
                state_channel=options.state_channel,
                strain_channel=options.strain_channel,
                state_sample_rate=options.state_sample_rate,
                strain_sample_rate=options.strain_sample_rate,
                frame_duration=options.frame_duration,
                gps_start_time=options.gps_start_time,
                gps_end_time=options.gps_end_time,
                output_path=options.output_path,
                description=options.description,
                segments=segments,
                values=values,
                verbose=options.verbose,
                real_time=options.real_time,
                history=options.history,
                cleanup_interval=options.cleanup_interval,
            )


class TestMain:
    """Test main entry point."""

    @patch("sgnligo.bin.fake_frames.read_segments_and_values_from_file")
    @patch("sgnligo.bin.fake_frames.generate_fake_frames")
    @patch("sgnligo.bin.fake_frames.parse_command_line")
    def test_main_basic(self, mock_parse, mock_generate, mock_read_segments):
        """Test main function basic flow."""
        # Mock command line arguments
        mock_options = Namespace(
            state_file="segments.txt",
            verbose=False,
            real_time=False,
            gps_start_time=None,
            gps_end_time=None,
            duration=80,
            history=3600,
            cleanup_interval=300,
            state_channel="L1:FAKE-STATE_VECTOR",
            strain_channel="L1:FAKE-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="BITMASK_TEST",
        )
        mock_parse.return_value = mock_options

        # Mock segment reading
        mock_read_segments.return_value = (
            ((1000000000000000000, 1000000080000000000),),
            (1,),
        )

        main()

        mock_parse.assert_called_once()
        mock_read_segments.assert_called_once_with("segments.txt", False)
        mock_generate.assert_called_once()

    @patch("sgnligo.bin.fake_frames.read_segments_and_values_from_file")
    @patch("sgnligo.bin.fake_frames.generate_fake_frames")
    @patch("sgnligo.bin.fake_frames.parse_command_line")
    @patch("sgnligo.bin.fake_frames.now")
    def test_main_real_time_mode(
        self, mock_now, mock_parse, mock_generate, mock_read_segments, capsys
    ):
        """Test main function in real-time mode."""
        mock_now.return_value = 1234567890

        # Mock command line arguments for real-time mode
        mock_options = Namespace(
            state_file="segments.txt",
            sample_rate=None,
            verbose=True,
            real_time=True,
            gps_start_time=None,  # Should be set to current GPS time
            gps_end_time=None,  # Should remain None
            duration=80,
            history=600,
            cleanup_interval=300,
            state_channel="L1:FAKE-STATE_VECTOR",
            strain_channel="L1:FAKE-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="BITMASK_TEST",
        )
        mock_parse.return_value = mock_options

        # Mock segment reading
        mock_read_segments.return_value = (
            ((1000000000000000000, 2000000000000000000),),
            (3,),
        )

        main()

        # Check that GPS start time was set
        assert mock_options.gps_start_time == 1234567890.0
        assert mock_options.gps_end_time is None

        # Check verbose output
        captured = capsys.readouterr()
        assert "Real-time mode starting at GPS time: 1234567890" in captured.out
        assert "History retention: 600 seconds" in captured.out
        assert "Running in real-time mode" in captured.out

    @patch("sgnligo.bin.fake_frames.read_segments_and_values_from_file")
    @patch("sgnligo.bin.fake_frames.generate_fake_frames")
    @patch("sgnligo.bin.fake_frames.parse_command_line")
    @patch("sgnligo.bin.fake_frames.now")
    def test_main_batch_mode_no_times(
        self, mock_now, mock_parse, mock_generate, mock_read_segments, capsys
    ):
        """Test main function in batch mode without specified times."""
        mock_now.return_value = 1234567890

        # Mock command line arguments for batch mode
        mock_options = Namespace(
            state_file="segments.txt",
            sample_rate=None,
            verbose=True,
            real_time=False,
            gps_start_time=None,  # Should be set to current GPS time
            gps_end_time=None,  # Should be calculated from duration
            duration=100,
            history=3600,
            cleanup_interval=300,
            state_channel="L1:FAKE-STATE_VECTOR",
            strain_channel="L1:FAKE-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="BITMASK_TEST",
        )
        mock_parse.return_value = mock_options

        # Mock segment reading
        mock_read_segments.return_value = (
            ((1234567890000000000, 1234567990000000000),),
            (1,),
        )

        main()

        # Check that times were set correctly
        assert mock_options.gps_start_time == 1234567890.0
        assert mock_options.gps_end_time == 1234567990.0

        # Check verbose output
        captured = capsys.readouterr()
        assert "Using current GPS time: 1234567890" in captured.out
        assert "Calculated end time: 1234567990" in captured.out
        assert "(duration: 100s)" in captured.out

    @patch("sgnligo.bin.fake_frames.read_segments_and_values_from_file")
    @patch("sgnligo.bin.fake_frames.generate_fake_frames")
    @patch("sgnligo.bin.fake_frames.parse_command_line")
    def test_main_batch_mode_with_times(
        self, mock_parse, mock_generate, mock_read_segments
    ):
        """Test main function in batch mode with specified times."""
        # Mock command line arguments with explicit times
        mock_options = Namespace(
            state_file="segments.txt",
            verbose=False,
            real_time=False,
            gps_start_time=1000000000.0,
            gps_end_time=1000000100.0,
            duration=80,  # Ignored when end time is specified
            history=3600,
            cleanup_interval=300,
            state_channel="L1:FAKE-STATE_VECTOR",
            strain_channel="L1:FAKE-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="BITMASK_TEST",
        )
        mock_parse.return_value = mock_options

        # Mock segment reading
        mock_read_segments.return_value = (
            ((1000000000000000000, 1000000100000000000),),
            (1,),
        )

        main()

        # Check that times were not modified
        assert mock_options.gps_start_time == 1000000000.0
        assert mock_options.gps_end_time == 1000000100.0

    @patch("sgnligo.bin.fake_frames.read_segments_and_values_from_file")
    @patch("sgnligo.bin.fake_frames.parse_command_line")
    def test_main_segment_loading_verbose(self, mock_parse, mock_read_segments, capsys):
        """Test verbose output during segment loading."""
        # Mock command line arguments
        mock_options = Namespace(
            state_file="test_segments.txt",
            sample_rate=None,
            verbose=True,
            real_time=False,
            gps_start_time=1000000000.0,
            gps_end_time=1000000016.0,
            duration=16,
            history=3600,
            cleanup_interval=300,
            state_channel="L1:FAKE-STATE_VECTOR",
            strain_channel="L1:FAKE-STRAIN",
            state_sample_rate=16,
            strain_sample_rate=16384,
            frame_duration=16,
            output_path="{instruments}-{description}-{gps_start_time}-{duration}.gwf",
            description="BITMASK_TEST",
        )
        mock_parse.return_value = mock_options

        # Mock segment reading
        segments = (
            (1000000000000000000, 1000000008000000000),
            (1000000008000000000, 1000000016000000000),
        )
        values = (1, 2)
        mock_read_segments.return_value = (segments, values)

        # Need to mock generate_fake_frames to avoid running the pipeline
        with patch("sgnligo.bin.fake_frames.generate_fake_frames"):
            main()

        # Check verbose output
        captured = capsys.readouterr()
        assert "Segments loaded successfully: 2 segments" in captured.out


class TestMainEntryPoint:
    """Test the main entry point execution."""

    def test_main_block_execution(self):
        """Test the if __name__ == '__main__' block."""
        import subprocess  # noqa: S404
        import sys

        # Run the module as a script with --help to avoid actual execution
        # This will cover the if __name__ == "__main__": block
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--append",
                "-m",
                "sgnligo.bin.fake_frames",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        # Check it ran successfully (--help exits with 0)
        assert result.returncode == 0
        assert "Generate fake frame files" in result.stdout


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
