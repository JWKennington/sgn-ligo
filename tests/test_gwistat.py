#!/usr/bin/env python3
"""Test coverage for sgnligo.bin.gwistat module."""

import json
from argparse import Namespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sgn.base import SourcePad
from sgnts.base import EventBuffer, EventFrame, SeriesBuffer, TSFrame

# Import the module to test
from sgnligo.bin.gwistat import (
    BitMaskInterpreter,
    bitmask_interpreter_pipeline,
    main,
    parse_command_line,
)


class TestParseCommandLine:
    """Test command line parsing."""

    def test_parse_minimal_args(self):
        """Test parsing with minimal required arguments."""
        with patch(
            "sys.argv",
            [
                "gwistat",
                "--channel-name",
                "L1:GDS-CALIB_STATE_VECTOR",
                "--mapping-file",
                "mapping.json",
            ],
        ):
            args = parse_command_line()
            assert args.channel_name == "L1:GDS-CALIB_STATE_VECTOR"
            assert args.mapping_file == "mapping.json"
            assert args.data_source == "devshm"  # default
            assert args.kafka_topic == "gwistat"  # default

    def test_parse_all_args(self):
        """Test parsing with all arguments."""
        with patch(
            "sys.argv",
            [
                "gwistat",
                "--data-source",
                "frames",
                "--shared-memory-dir",
                "/dev/shm/kafka/L1",  # noqa: S108
                "--frame-cache",
                "frames.cache",
                "--channel-name",
                "L1:GDS-CALIB_STATE_VECTOR",
                "--mapping-file",
                "mapping.json",
                "--gps-start-time",
                "1234567890",
                "--gps-end-time",
                "1234567900",
                "--discont-wait-time",
                "30",
                "--queue-timeout",
                "5",
                "--verbose",
                "--output-kafka-server",
                "localhost:9092",
                "--kafka-topic",
                "test_topic",
                "--kafka-tag",
                "L1",
            ],
        ):
            args = parse_command_line()
            assert args.data_source == "frames"
            assert args.shared_memory_dir == "/dev/shm/kafka/L1"  # noqa: S108
            assert args.frame_cache == "frames.cache"
            assert args.channel_name == "L1:GDS-CALIB_STATE_VECTOR"
            assert args.mapping_file == "mapping.json"
            assert args.gps_start_time == 1234567890
            assert args.gps_end_time == 1234567900
            assert args.discont_wait_time == 30
            assert args.queue_timeout == 5
            assert args.verbose is True
            assert args.output_kafka_server == "localhost:9092"
            assert args.kafka_topic == "test_topic"
            assert args.kafka_tag == "L1"

    def test_parse_data_source_choices(self):
        """Test data source choices."""
        # Test devshm
        with patch(
            "sys.argv",
            [
                "gwistat",
                "--data-source",
                "devshm",
                "--channel-name",
                "L1:TEST",
                "--mapping-file",
                "map.json",
            ],
        ):
            args = parse_command_line()
            assert args.data_source == "devshm"

        # Test frames
        with patch(
            "sys.argv",
            [
                "gwistat",
                "--data-source",
                "frames",
                "--channel-name",
                "L1:TEST",
                "--mapping-file",
                "map.json",
            ],
        ):
            args = parse_command_line()
            assert args.data_source == "frames"


class TestBitMaskInterpreter:
    """Test BitMaskInterpreter transform."""

    def create_mock_buffer(self, data, offset=0, sample_rate=16):
        """Create a mock SeriesBuffer."""
        buffer = Mock(spec=SeriesBuffer)
        buffer.data = np.array(data)
        buffer.is_gap = False
        buffer.offset = offset
        buffer.sample_rate = sample_rate
        return buffer

    def create_mock_frame(self, buffers, EOS=False, start=0, end=16384):
        """Create a mock TSFrame."""
        frame = Mock(spec=TSFrame)
        frame.__iter__ = Mock(return_value=iter(buffers))
        frame.EOS = EOS
        frame.offset = start
        frame.end_offset = end
        return frame

    def test_init(self, tmp_path):
        """Test BitMaskInterpreter initialization."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0", "1": "BIT_1", "2": "BIT_2"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=True,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        assert interpreter.mapping_file == str(mapping_file)
        assert interpreter.kafka_topic == "test_topic"
        assert interpreter.verbose is True
        assert interpreter.bit_mappings == mapping

    def test_init_no_mapping_file(self):
        """Test initialization without mapping file."""
        with pytest.raises(AssertionError):
            BitMaskInterpreter(
                name="TestInterpreter",
                mapping_file=None,
                source_pad_names=("output",),
                sink_pad_names=("input",),
            )

    def test_new_single_value(self, tmp_path, capsys):
        """Test processing a single bitmask value."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "HOFT_OK", "1": "OBS_INTENT"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=True,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with data
        buffer = self.create_mock_buffer([3])  # Binary: 11 (both bits set)
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check result
        assert isinstance(result, EventFrame)
        assert len(result.data) == 1
        event_buffer = result.data[0]
        assert isinstance(event_buffer, EventBuffer)

        # Check data
        data = event_buffer.data[0]["test_topic"]
        assert len(data["time"]) == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["value"] == 3
        assert data["data"][0]["active_bits"] == [0, 1]
        assert data["data"][0]["bit_meanings"] == ["HOFT_OK", "OBS_INTENT"]

        # Check verbose output
        captured = capsys.readouterr()
        assert "value=3 -> ['HOFT_OK', 'OBS_INTENT']" in captured.out

    def test_new_multiple_values(self, tmp_path):
        """Test processing multiple bitmask values."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0", "1": "BIT_1", "2": "BIT_2", "3": "BIT_3"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with multiple values
        buffer = self.create_mock_buffer([0, 1, 2, 4, 7, 15])
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check data
        data = result.data[0].data[0]["test_topic"]
        assert len(data["data"]) == 6

        # Check each value
        assert data["data"][0]["value"] == 0
        assert data["data"][0]["active_bits"] == []
        assert data["data"][0]["bit_meanings"] == []

        assert data["data"][1]["value"] == 1
        assert data["data"][1]["active_bits"] == [0]
        assert data["data"][1]["bit_meanings"] == ["BIT_0"]

        assert data["data"][2]["value"] == 2
        assert data["data"][2]["active_bits"] == [1]
        assert data["data"][2]["bit_meanings"] == ["BIT_1"]

        assert data["data"][3]["value"] == 4
        assert data["data"][3]["active_bits"] == [2]
        assert data["data"][3]["bit_meanings"] == ["BIT_2"]

        assert data["data"][4]["value"] == 7
        assert data["data"][4]["active_bits"] == [0, 1, 2]
        assert data["data"][4]["bit_meanings"] == ["BIT_0", "BIT_1", "BIT_2"]

        assert data["data"][5]["value"] == 15
        assert data["data"][5]["active_bits"] == [0, 1, 2, 3]
        assert data["data"][5]["bit_meanings"] == ["BIT_0", "BIT_1", "BIT_2", "BIT_3"]

    def test_new_with_unmapped_bits(self, tmp_path):
        """Test processing values with unmapped bits."""
        # Create mapping file with only some bits mapped
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0", "2": "BIT_2"}  # Skip bit 1
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame
        buffer = self.create_mock_buffer([7])  # Binary: 111 (all 3 bits set)
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check data
        data = result.data[0].data[0]["test_topic"]
        assert data["data"][0]["value"] == 7
        assert data["data"][0]["active_bits"] == [0, 1, 2]
        # Only mapped bits should have meanings
        assert data["data"][0]["bit_meanings"] == ["BIT_0", "BIT_2"]

    def test_new_with_gap_buffer(self, tmp_path):
        """Test handling gap buffers."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with gap buffer
        buffer = Mock(spec=SeriesBuffer)
        buffer.is_gap = True
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check that no data was processed
        data = result.data[0].data[0]["test_topic"]
        assert len(data["data"]) == 0

    def test_new_with_eos(self, tmp_path):
        """Test handling End-of-Stream."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with EOS
        buffer = self.create_mock_buffer([1])
        frame = self.create_mock_frame([buffer], EOS=True)

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check that EOS is propagated
        assert result.EOS is True

    def test_timestamp_calculation(self, tmp_path):
        """Test correct timestamp calculation."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with specific offset
        buffer = self.create_mock_buffer([1, 2, 3], offset=1000000000, sample_rate=16)
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check timestamps
        data = result.data[0].data[0]["test_topic"]
        times = data["time"]
        assert len(times) == 3

        # Verify timestamps are incrementing correctly
        # With sample rate 16, each sample is 1/16 second apart
        assert times[1] - times[0] == pytest.approx(1.0 / 16)
        assert times[2] - times[1] == pytest.approx(1.0 / 16)

    def test_verbose_output_limit(self, tmp_path, capsys):
        """Test that verbose output is limited to first 5 samples."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter with verbose=True
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=True,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with many values
        buffer = self.create_mock_buffer([1] * 10)  # 10 values
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        interpreter.new(source_pad)

        # Check verbose output
        captured = capsys.readouterr()
        # Should only print first 5
        assert captured.out.count("value=1") == 5

    def test_event_buffer_timestamps(self, tmp_path):
        """Test that EventBuffer timestamps are set from frame start/end."""
        # Create mapping file
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with specific start and end times
        # Using offset values that represent specific times
        frame_start_offset = 1234567890000000000  # Large offset value
        frame_end_offset = 1234567891000000000  # 1 second later at MAX_RATE
        buffer = self.create_mock_buffer(
            [1, 2, 3], offset=frame_start_offset, sample_rate=16
        )
        frame = self.create_mock_frame(
            [buffer], start=frame_start_offset, end=frame_end_offset
        )

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check EventBuffer timestamps
        event_buffer = result.data[0]

        # The timestamps should be based on the frame start/end in nanoseconds
        # EventBuffer expects nanosecond timestamps (integers)
        # Since Offset.offset_ref_t0 = 0, the timestamps are just frame.start
        # and frame.end
        expected_start_ns = (
            frame_start_offset  # frame.start + Offset.offset_ref_t0 (which is 0)
        )
        expected_end_ns = (
            frame_end_offset  # frame.end + Offset.offset_ref_t0 (which is 0)
        )

        assert event_buffer.start == expected_start_ns
        assert event_buffer.end == expected_end_ns

    def test_backward_compatibility_old_format(self, tmp_path):
        """Test that old mapping format still works."""
        # Create mapping file with old format (just bit mappings)
        mapping_file = tmp_path / "mapping.json"
        mapping = {"0": "BIT_0", "1": "BIT_1", "2": "BIT_2"}
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Verify mappings loaded correctly
        assert interpreter.bit_mappings == {"0": "BIT_0", "1": "BIT_1", "2": "BIT_2"}
        assert interpreter.value_mappings == {}

        # Create mock sink pad and frame
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]
        buffer = self.create_mock_buffer([7])  # Binary: 111
        frame = self.create_mock_frame([buffer])
        interpreter.preparedframes = {sink_pad: frame}

        # Process frame
        source_pad = Mock(spec=SourcePad)
        result = interpreter.new(source_pad)

        # Check data - should work as before
        data = result.data[0].data[0]["test_topic"]
        assert data["data"][0]["value"] == 7
        assert data["data"][0]["active_bits"] == [0, 1, 2]
        assert data["data"][0]["bit_meanings"] == ["BIT_0", "BIT_1", "BIT_2"]
        assert "value_meaning" not in data["data"][0]  # No value meanings in old format

    def test_new_format_with_value_mappings(self, tmp_path, capsys):
        """Test the new mapping format with both bit and value mappings."""
        # Create mapping file with new format
        mapping_file = tmp_path / "mapping.json"
        mapping = {
            "bits": {"0": "HOFT_OK", "1": "OBS_INTENT", "2": "SCIENCE_INTENT"},
            "values": {
                "0": "NO_DATA",
                "1": "HOFT_OK_ONLY",
                "3": "OBSERVING_READY",
                "7": "SCIENCE_MODE",
            },
        }
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=True,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]

        # Create mock frame with data
        buffer = self.create_mock_buffer([0, 1, 3, 7])
        frame = self.create_mock_frame([buffer])

        # Mock preparedframes
        interpreter.preparedframes = {sink_pad: frame}

        # Create source pad
        source_pad = Mock(spec=SourcePad)

        # Process frame
        result = interpreter.new(source_pad)

        # Check data
        data = result.data[0].data[0]["test_topic"]
        assert len(data["data"]) == 4

        # Check value 0 - NO_DATA
        assert data["data"][0]["value"] == 0
        assert data["data"][0]["active_bits"] == []
        assert data["data"][0]["bit_meanings"] == []
        assert data["data"][0]["value_meaning"] == "NO_DATA"

        # Check value 1 - HOFT_OK_ONLY
        assert data["data"][1]["value"] == 1
        assert data["data"][1]["active_bits"] == [0]
        assert data["data"][1]["bit_meanings"] == ["HOFT_OK"]
        assert data["data"][1]["value_meaning"] == "HOFT_OK_ONLY"

        # Check value 3 - OBSERVING_READY
        assert data["data"][2]["value"] == 3
        assert data["data"][2]["active_bits"] == [0, 1]
        assert data["data"][2]["bit_meanings"] == ["HOFT_OK", "OBS_INTENT"]
        assert data["data"][2]["value_meaning"] == "OBSERVING_READY"

        # Check value 7 - SCIENCE_MODE
        assert data["data"][3]["value"] == 7
        assert data["data"][3]["active_bits"] == [0, 1, 2]
        assert data["data"][3]["bit_meanings"] == [
            "HOFT_OK",
            "OBS_INTENT",
            "SCIENCE_INTENT",
        ]
        assert data["data"][3]["value_meaning"] == "SCIENCE_MODE"

        # Check verbose output
        captured = capsys.readouterr()
        assert "value=0 -> NO_DATA (bits: [])" in captured.out
        assert "value=1 -> HOFT_OK_ONLY (bits: ['HOFT_OK'])" in captured.out
        assert (
            "value=3 -> OBSERVING_READY (bits: ['HOFT_OK', 'OBS_INTENT'])"
            in captured.out
        )
        assert (
            "value=7 -> SCIENCE_MODE "
            "(bits: ['HOFT_OK', 'OBS_INTENT', 'SCIENCE_INTENT'])" in captured.out
        )

    def test_new_format_only_values(self, tmp_path):
        """Test mapping with only value meanings, no bit meanings."""
        # Create mapping file with only value mappings
        mapping_file = tmp_path / "mapping.json"
        mapping = {
            "values": {
                "0": "SYSTEM_DOWN",
                "15": "PARTIAL_OPERATION",
                "255": "ALL_SYSTEMS_GO",
            }
        }
        mapping_file.write_text(json.dumps(mapping))

        # Create interpreter
        interpreter = BitMaskInterpreter(
            name="TestInterpreter",
            mapping_file=str(mapping_file),
            kafka_topic="test_topic",
            verbose=False,
            source_pad_names=("output",),
            sink_pad_names=("input",),
        )

        # Create mock sink pad and frame
        sink_pad = Mock()
        interpreter.sink_pads = [sink_pad]
        buffer = self.create_mock_buffer([0, 15, 100, 255])
        frame = self.create_mock_frame([buffer])
        interpreter.preparedframes = {sink_pad: frame}

        # Process frame
        source_pad = Mock(spec=SourcePad)
        result = interpreter.new(source_pad)

        # Check data
        data = result.data[0].data[0]["test_topic"]

        # Value 0 - has value meaning
        assert data["data"][0]["value"] == 0
        assert data["data"][0]["value_meaning"] == "SYSTEM_DOWN"
        assert data["data"][0]["bit_meanings"] == []

        # Value 15 - has value meaning
        assert data["data"][1]["value"] == 15
        assert data["data"][1]["value_meaning"] == "PARTIAL_OPERATION"
        assert data["data"][1]["active_bits"] == [0, 1, 2, 3]

        # Value 100 - no value meaning
        assert data["data"][2]["value"] == 100
        assert "value_meaning" not in data["data"][2]
        assert data["data"][2]["active_bits"] == [2, 5, 6]

        # Value 255 - has value meaning
        assert data["data"][3]["value"] == 255
        assert data["data"][3]["value_meaning"] == "ALL_SYSTEMS_GO"
        assert data["data"][3]["active_bits"] == [0, 1, 2, 3, 4, 5, 6, 7]


class TestBitmaskInterpreterPipeline:
    """Test bitmask_interpreter_pipeline function."""

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnligo.bin.gwistat.DevShmSource")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_devshm_source(
        self, mock_kafka_sink, mock_interpreter, mock_devshm_source, mock_pipeline
    ):
        """Test pipeline creation with DevShm source."""
        options = Namespace(
            data_source="devshm",
            shared_memory_dir="/dev/shm/kafka/L1",  # noqa: S108
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            discont_wait_time=60,
            queue_timeout=1,
            verbose=True,
            frame_cache=None,
            gps_start_time=None,
            gps_end_time=None,
            output_kafka_server="localhost:9092",
            kafka_topic="test_topic",
            kafka_tag="L1",
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_devshm_instance = Mock()
        mock_devshm_source.return_value = mock_devshm_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check DevShmSource was created correctly
        mock_devshm_source.assert_called_once_with(
            name="DataSrc",
            shared_memory_dirs="/dev/shm/kafka/L1",  # noqa: S108
            channel_names=["L1:GDS-CALIB_STATE_VECTOR"],
            discont_wait_time=60,
            queue_timeout=1,
            verbose=True,
        )

        # Check pipeline was built correctly
        mock_pipeline.assert_called_once()
        assert mock_pipe_instance.insert.call_count == 2
        mock_pipe_instance.run.assert_called_once()

    def test_pipeline_devshm_missing_dir(self):
        """Test error when shared memory dir not provided for devshm."""
        options = Namespace(
            data_source="devshm",
            shared_memory_dir=None,
            channel_name="L1:TEST",
            mapping_file="mapping.json",
        )

        with pytest.raises(ValueError, match="--shared-memory-dir is required"):
            bitmask_interpreter_pipeline(options)

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnligo.bin.gwistat.FrameReader")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_frames_source_with_cache_file(
        self, mock_kafka_sink, mock_interpreter, mock_frame_reader, mock_pipeline
    ):
        """Test pipeline creation with frames source using cache file."""
        options = Namespace(
            data_source="frames",
            frame_cache="test.cache",
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            gps_start_time=1234567890,
            gps_end_time=1234567900,
            shared_memory_dir=None,
            verbose=False,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_frame_instance = Mock()
        mock_frame_reader.return_value = mock_frame_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check FrameReader was created correctly
        mock_frame_reader.assert_called_once_with(
            name="DataSrc",
            framecache="test.cache",
            channel_names=["L1:GDS-CALIB_STATE_VECTOR"],
            instrument="L1",
            t0=1234567890,
            end=1234567900,
        )

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnligo.bin.gwistat.FrameReader")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_frames_source_with_glob(
        self,
        mock_kafka_sink,
        mock_interpreter,
        mock_frame_reader,
        mock_pipeline,
    ):
        """Test pipeline creation with frames source using glob pattern."""
        # Create a temporary directory with test files
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test gwf files
            gwf1 = os.path.join(tmpdir, "L1-TEST-1234567890-10.gwf")
            gwf2 = os.path.join(tmpdir, "L1-TEST-1234567900-10.gwf")
            open(gwf1, "a").close()
            open(gwf2, "a").close()

            options = Namespace(
                data_source="frames",
                frame_cache=os.path.join(tmpdir, "*.gwf"),
                channel_name="L1:GDS-CALIB_STATE_VECTOR",
                mapping_file="mapping.json",
                gps_start_time=None,
                gps_end_time=None,
                shared_memory_dir=None,
                verbose=True,
                output_kafka_server=None,
                kafka_topic="test_topic",
                kafka_tag=None,
            )

            # Mock instances
            mock_pipe_instance = Mock()
            mock_pipeline.return_value = mock_pipe_instance
            mock_frame_instance = Mock()
            mock_frame_reader.return_value = mock_frame_instance
            mock_interpreter_instance = Mock()
            mock_interpreter.return_value = mock_interpreter_instance
            mock_kafka_instance = Mock()
            mock_kafka_sink.return_value = mock_kafka_instance

            # Run pipeline
            bitmask_interpreter_pipeline(options)

            # Check FrameReader was created
            mock_frame_reader.assert_called_once()
            call_args = mock_frame_reader.call_args[1]
            # Cache file should have been created
            assert call_args["framecache"].endswith(".cache")

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnligo.bin.gwistat.FrameReader")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_frames_source_single_gwf(
        self, mock_kafka_sink, mock_interpreter, mock_frame_reader, mock_pipeline
    ):
        """Test pipeline with single gwf file."""
        options = Namespace(
            data_source="frames",
            frame_cache="/path/to/L1-TEST-1234567890-10.gwf",
            channel_name="L1:TEST",
            mapping_file="mapping.json",
            gps_start_time=None,
            gps_end_time=None,
            shared_memory_dir=None,
            verbose=False,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_frame_instance = Mock()
        mock_frame_reader.return_value = mock_frame_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/test.cache"  # noqa: S108
            mock_temp.return_value.__enter__.return_value = mock_file

            bitmask_interpreter_pipeline(options)

            # Check that single file was processed
            mock_file.write.assert_called_once()

    def test_pipeline_frames_missing_cache(self):
        """Test error when frame cache not provided for frames source."""
        options = Namespace(
            data_source="frames",
            frame_cache=None,
            channel_name="L1:TEST",
            mapping_file="mapping.json",
        )

        with pytest.raises(ValueError, match="--frame-cache is required"):
            bitmask_interpreter_pipeline(options)

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("glob.glob")
    def test_pipeline_frames_no_files_found(self, mock_glob, mock_pipeline):
        """Test error when glob pattern matches no files."""
        mock_glob.return_value = []

        options = Namespace(
            data_source="frames",
            frame_cache="/path/to/*.gwf",
            channel_name="L1:TEST",
            mapping_file="mapping.json",
            shared_memory_dir=None,
        )

        with pytest.raises(ValueError, match="No files found matching"):
            bitmask_interpreter_pipeline(options)

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnligo.bin.gwistat.glob.glob")
    def test_pipeline_frames_malformed_filename(self, mock_glob, mock_pipeline):
        """Test error when filename doesn't follow LAL cache convention."""
        # Return a malformed filename
        mock_glob.return_value = ["/path/to/malformed.gwf"]

        options = Namespace(
            data_source="frames",
            frame_cache="/path/to/*.gwf",
            channel_name="L1:TEST",
            mapping_file="mapping.json",
            shared_memory_dir=None,
        )

        with pytest.raises(
            ValueError, match="Could not parse malformed.gwf into LAL Cache convention"
        ):
            bitmask_interpreter_pipeline(options)

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnligo.bin.gwistat.DevShmSource")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_link_map(
        self, mock_kafka_sink, mock_interpreter, mock_devshm_source, mock_pipeline
    ):
        """Test pipeline link map construction."""
        options = Namespace(
            data_source="devshm",
            shared_memory_dir="/dev/shm/kafka/L1",  # noqa: S108
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            discont_wait_time=60,
            queue_timeout=1,
            verbose=False,
            frame_cache=None,
            gps_start_time=None,
            gps_end_time=None,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check link map
        link_map_call = mock_pipe_instance.insert.call_args_list[1][1]["link_map"]
        assert link_map_call == {
            "BitMaskInt:snk:input": "DataSrc:src:L1:GDS-CALIB_STATE_VECTOR",
            "KafkaSink:snk:data": "BitMaskInt:src:output",
        }


class TestGWDataNoiseRealtimeSource:
    """Test gwdata-noise-realtime data source in pipeline."""

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnts.sources.SegmentSource")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_gwdata_noise_realtime_with_times(
        self, mock_kafka_sink, mock_interpreter, mock_segment_source, mock_pipeline
    ):
        """Test gwdata-noise-realtime source with start and end times."""
        options = Namespace(
            data_source="gwdata-noise-realtime",
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            gps_start_time=1234567890.0,
            gps_end_time=1234567900.0,
            state_segments_file=None,
            verbose=False,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
            shared_memory_dir=None,
            frame_cache=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_segment_instance = Mock()
        mock_segment_source.return_value = mock_segment_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check SegmentSource was created correctly
        mock_segment_source.assert_called_once()
        call_kwargs = mock_segment_source.call_args[1]
        assert call_kwargs["name"] == "DataSrc"
        assert call_kwargs["source_pad_names"] == ("L1:GDS-CALIB_STATE_VECTOR",)
        assert call_kwargs["rate"] == 16
        assert call_kwargs["t0"] == 1234567890.0
        assert call_kwargs["end"] == 1234567900.0

        # Check pipeline was built
        mock_pipeline.assert_called_once()
        assert mock_pipe_instance.insert.call_count == 2
        mock_pipe_instance.run.assert_called_once()

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnts.sources.SegmentSource")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_gwdata_noise_realtime_no_end_time(
        self, mock_kafka_sink, mock_interpreter, mock_segment_source, mock_pipeline
    ):
        """Test gwdata-noise-realtime source with start time but no end time."""
        import numpy as np

        options = Namespace(
            data_source="gwdata-noise-realtime",
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            gps_start_time=1234567890.0,
            gps_end_time=None,
            state_segments_file=None,
            verbose=False,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
            shared_memory_dir=None,
            frame_cache=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_segment_instance = Mock()
        mock_segment_source.return_value = mock_segment_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check SegmentSource was created with max int32 as end time
        mock_segment_source.assert_called_once()
        call_kwargs = mock_segment_source.call_args[1]
        assert call_kwargs["t0"] == 1234567890.0
        assert call_kwargs["end"] == float(np.iinfo(np.int32).max)

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnts.sources.SegmentSource")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    @patch("sgnligo.base.now")
    def test_pipeline_gwdata_noise_realtime_no_start_time(
        self,
        mock_now,
        mock_kafka_sink,
        mock_interpreter,
        mock_segment_source,
        mock_pipeline,
    ):
        """Test gwdata-noise-realtime source without start time (uses current time)."""
        mock_now.return_value = 1234567890.5  # Current GPS time

        options = Namespace(
            data_source="gwdata-noise-realtime",
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            gps_start_time=None,
            gps_end_time=1234567900.0,
            state_segments_file=None,
            verbose=False,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
            shared_memory_dir=None,
            frame_cache=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_segment_instance = Mock()
        mock_segment_source.return_value = mock_segment_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check SegmentSource was created with now() as start time
        mock_segment_source.assert_called_once()
        call_kwargs = mock_segment_source.call_args[1]
        # Start time should be int(now()) = 1234567890
        assert call_kwargs["t0"] == 1234567890.0
        assert call_kwargs["end"] == 1234567900.0

    def test_pipeline_gwdata_noise_realtime_no_times_error(self):
        """Test error when neither start nor end time provided."""
        options = Namespace(
            data_source="gwdata-noise-realtime",
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            gps_start_time=None,
            gps_end_time=None,
            state_segments_file=None,
            verbose=False,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
            shared_memory_dir=None,
            frame_cache=None,
        )

        with pytest.raises(
            ValueError,
            match="--gps-end-time is required when --gps-start-time is not specified",
        ):
            bitmask_interpreter_pipeline(options)

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnts.sources.SegmentSource")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    @patch("sgnligo.base.read_segments_and_values_from_file")
    def test_pipeline_gwdata_noise_realtime_with_segments_file(
        self,
        mock_read_segments,
        mock_kafka_sink,
        mock_interpreter,
        mock_segment_source,
        mock_pipeline,
    ):
        """Test gwdata-noise-realtime source with state segments file."""
        # Mock the segments file reading
        mock_read_segments.return_value = (
            ((1234567890000000000, 1234567900000000000),),  # segments in nanoseconds
            (7,),  # values
        )

        options = Namespace(
            data_source="gwdata-noise-realtime",
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            gps_start_time=1234567890.0,
            gps_end_time=1234567900.0,
            state_segments_file="/path/to/segments.txt",
            verbose=True,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
            shared_memory_dir=None,
            frame_cache=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_segment_instance = Mock()
        mock_segment_source.return_value = mock_segment_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check that read_segments_and_values_from_file was called
        mock_read_segments.assert_called_once_with("/path/to/segments.txt", True)

        # Check SegmentSource was created with the segments from file
        mock_segment_source.assert_called_once()
        call_kwargs = mock_segment_source.call_args[1]
        assert call_kwargs["segments"] == ((1234567890000000000, 1234567900000000000),)
        assert call_kwargs["values"] == (7,)

    @patch("sgnligo.bin.gwistat.Pipeline")
    @patch("sgnts.sources.SegmentSource")
    @patch("sgnligo.bin.gwistat.BitMaskInterpreter")
    @patch("sgnligo.bin.gwistat.KafkaSink")
    def test_pipeline_gwdata_noise_realtime_verbose(
        self,
        mock_kafka_sink,
        mock_interpreter,
        mock_segment_source,
        mock_pipeline,
        capsys,
    ):
        """Test gwdata-noise-realtime source with verbose output."""
        options = Namespace(
            data_source="gwdata-noise-realtime",
            channel_name="L1:GDS-CALIB_STATE_VECTOR",
            mapping_file="mapping.json",
            gps_start_time=1234567890.0,
            gps_end_time=1234567900.0,
            state_segments_file=None,
            verbose=True,
            output_kafka_server=None,
            kafka_topic="test_topic",
            kafka_tag=None,
            shared_memory_dir=None,
            frame_cache=None,
        )

        # Mock instances
        mock_pipe_instance = Mock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_segment_instance = Mock()
        mock_segment_source.return_value = mock_segment_instance
        mock_interpreter_instance = Mock()
        mock_interpreter.return_value = mock_interpreter_instance
        mock_kafka_instance = Mock()
        mock_kafka_sink.return_value = mock_kafka_instance

        # Run pipeline
        bitmask_interpreter_pipeline(options)

        # Check verbose output
        captured = capsys.readouterr()
        assert (
            "Created simulated state vector source for L1:GDS-CALIB_STATE_VECTOR"
            in captured.out
        )
        assert "Time range: 1234567890.0 - 1234567900.0" in captured.out
        assert "State segments:" in captured.out
        assert "State values:" in captured.out


class TestMain:
    """Test main entry point."""

    @patch("sgnligo.bin.gwistat.bitmask_interpreter_pipeline")
    @patch("sgnligo.bin.gwistat.parse_command_line")
    def test_main_basic(self, mock_parse, mock_pipeline):
        """Test main function basic flow."""
        # Mock command line arguments
        mock_options = Namespace(
            data_source="devshm",
            shared_memory_dir="/dev/shm/kafka/L1",  # noqa: S108
            channel_name="L1:TEST",
            mapping_file="mapping.json",
        )
        mock_parse.return_value = mock_options

        main()

        mock_parse.assert_called_once()
        mock_pipeline.assert_called_once_with(mock_options)


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
                "sgnligo.bin.gwistat",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        # Check it ran successfully (--help exits with 0)
        assert result.returncode == 0
        assert "Read GW Frame data" in result.stdout


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
