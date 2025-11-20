"""Comprehensive tests for KafkaSink to achieve 100% coverage."""

import json
from unittest.mock import Mock, patch

import pytest
from lal import LIGOTimeGPS
from sgnts.base import EventBuffer, EventFrame

from sgnligo.sinks.kafka_sink import KafkaSink, LIGOJSONEncoder


class TestLIGOJSONEncoder:
    """Test cases for the custom JSON encoder."""

    def test_encode_ligotimegps(self):
        """Test encoding LIGOTimeGPS objects."""
        encoder = LIGOJSONEncoder()
        gps_time = LIGOTimeGPS(1234567890.5)
        result = encoder.default(gps_time)
        assert result == 1234567890.5

    def test_encode_other_types_raises(self):
        """Test that other types raise TypeError."""
        encoder = LIGOJSONEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())

    def test_encode_in_json_dumps(self):
        """Test using encoder with json.dumps."""
        data = {"time": LIGOTimeGPS(1000), "value": 42}
        result = json.dumps(data, cls=LIGOJSONEncoder)
        assert result == '{"time": 1000.0, "value": 42}'


class TestKafkaSinkInit:
    """Test cases for KafkaSink initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"])
        assert sink.client is None
        assert sink.tag == []
        assert sink.time_series_data is None
        assert sink.trigger_data is None

    def test_init_with_kafka_server(self):
        """Test initialization with kafka server."""
        with patch("sgnligo.sinks.kafka_sink.kafka.Client") as mock_client:
            sink = KafkaSink(
                name="TestSink",
                sink_pad_names=["data"],
                output_kafka_server="localhost:9092",
            )
            mock_client.assert_called_once_with("kafka://localhost:9092")
            assert sink.client is not None

    def test_init_with_none_string_server(self):
        """Test initialization with 'None' string as server."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], output_kafka_server="None"
        )
        assert sink.client is None

    def test_init_with_time_series_topics(self):
        """Test initialization with time series topics."""
        topics = ["topic1", "topic2"]
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], time_series_topics=topics
        )
        assert sink.time_series_data == {
            "topic1": {"time": [], "data": []},
            "topic2": {"time": [], "data": []},
        }

    def test_init_with_trigger_topics(self):
        """Test initialization with trigger topics."""
        topics = ["trigger1", "trigger2"]
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], trigger_topics=topics
        )
        assert sink.trigger_data == {"trigger1": [], "trigger2": []}

    def test_init_with_tags(self):
        """Test initialization with tags."""
        tags = ["tag1", "tag2"]
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"], tag=tags)
        assert sink.tag == tags

    @patch("sgnligo.sinks.kafka_sink.now")
    def test_init_sets_last_sent(self, mock_now):
        """Test that last_sent is initialized with current time."""
        mock_now.return_value = 1234567890
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"])
        assert sink.last_sent == 1234567890


class TestKafkaSinkPrettyPrint:
    """Test cases for pretty printing functionality."""

    @patch("sgnligo.sinks.kafka_sink.now")
    def test_pretty_print_time_series(self, mock_now, capsys):
        """Test pretty printing time series data."""
        mock_now.return_value = 1234567890
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], prefix="test_", tag=["tag1"]
        )

        data = {"time": [1, 2, 3], "data": [10, 20, 30]}
        sink._pretty_print("topic1", data, "time_series")

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["topic"] == "test_topic1"
        assert output["tags"] == ["tag1"]
        assert output["data_type"] == "time_series"
        assert output["timestamp"] == 1234567890
        assert output["data"] == data

    def test_pretty_print_with_ligotimegps(self, capsys):
        """Test pretty printing with LIGOTimeGPS objects."""
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"])
        data = {"time": LIGOTimeGPS(1000), "value": 42}
        sink._pretty_print("topic", data)

        captured = capsys.readouterr()
        assert "1000.0" in captured.out


class TestKafkaSinkWrite:
    """Test cases for write functionality."""

    def test_write_empty_data(self):
        """Test writing when no data is present."""
        sink = KafkaSink(
            name="TestSink",
            sink_pad_names=["data"],
            time_series_topics=["topic1"],
            trigger_topics=["trigger1"],
        )
        sink.write()  # Should not raise any errors

    def test_write_time_series_to_kafka(self):
        """Test writing time series data to kafka."""
        mock_client = Mock()
        sink = KafkaSink(
            name="TestSink",
            sink_pad_names=["data"],
            output_kafka_server="localhost:9092",
            time_series_topics=["topic1"],
            prefix="prefix_",
            tag=["tag1"],
        )
        sink.client = mock_client

        # Add some data
        sink.time_series_data["topic1"]["time"] = [1, 2, 3]
        sink.time_series_data["topic1"]["data"] = [10, 20, 30]

        sink.write()

        # Verify client.write was called
        mock_client.write.assert_called_once_with(
            "prefix_topic1",
            {"time": [1, 2, 3], "data": [10, 20, 30]},
            tags=["tag1"],
        )

        # Verify data was cleared
        assert sink.time_series_data["topic1"]["time"] == []
        assert sink.time_series_data["topic1"]["data"] == []

    def test_write_time_series_to_stdout(self, capsys):
        """Test writing time series data to stdout."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], time_series_topics=["topic1"]
        )

        sink.time_series_data["topic1"]["time"] = [1, 2]
        sink.time_series_data["topic1"]["data"] = [10, 20]

        sink.write()

        captured = capsys.readouterr()
        assert "topic1" in captured.out
        assert "time_series" in captured.out

    def test_write_triggers_to_kafka(self):
        """Test writing trigger data to kafka."""
        mock_client = Mock()
        sink = KafkaSink(
            name="TestSink",
            sink_pad_names=["data"],
            output_kafka_server="localhost:9092",
            trigger_topics=["trigger1"],
        )
        sink.client = mock_client

        # Add trigger data
        sink.trigger_data["trigger1"] = [
            {"time": 1, "snr": 10},
            {"time": 2, "snr": 20},
        ]

        sink.write()

        mock_client.write.assert_called_once()
        assert sink.trigger_data["trigger1"] == []

    def test_write_triggers_to_stdout(self, capsys):
        """Test writing trigger data to stdout."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], trigger_topics=["trigger1"]
        )

        sink.trigger_data["trigger1"] = [{"time": 1, "snr": 10}]

        sink.write()

        captured = capsys.readouterr()
        assert "trigger1" in captured.out
        assert "trigger" in captured.out


class TestKafkaSinkPull:
    """Test cases for pull method."""

    def test_pull_time_series_data(self):
        """Test pulling time series data from frame."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], time_series_topics=["channel1"]
        )

        # Create EventFrame with EventBuffer
        frame_data = {
            "channel1": {
                "time": [1, 2, 3],
                "data": [10, 20, 30],
            }
        }
        event_buffer = EventBuffer.from_span(1000000000, 2000000000, [frame_data])
        frame = EventFrame(data=[event_buffer], EOS=False)

        pad = Mock()
        sink.pull(pad, frame)

        # Verify data was accumulated
        assert sink.time_series_data["channel1"]["time"] == [1, 2, 3]
        assert sink.time_series_data["channel1"]["data"] == [10, 20, 30]

    def test_pull_trigger_data(self):
        """Test pulling trigger data from frame."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], trigger_topics=["trigger1"]
        )

        # Create EventFrame with EventBuffer
        frame_data = {
            "trigger1": [
                {"time": 1, "snr": 10},
                {"time": 2, "snr": 20},
            ]
        }
        event_buffer = EventBuffer.from_span(1000000000, 2000000000, [frame_data])
        frame = EventFrame(data=[event_buffer], EOS=False)

        pad = Mock()
        sink.pull(pad, frame)

        # Verify triggers were accumulated
        assert len(sink.trigger_data["trigger1"]) == 2

    def test_pull_with_none_data(self):
        """Test pulling when data is None."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], time_series_topics=["channel1"]
        )

        event_buffer = EventBuffer.from_span(1000000000, 2000000000, None)
        frame = EventFrame(data=[event_buffer], EOS=False)

        pad = Mock()
        sink.pull(pad, frame)  # Should not raise

    def test_pull_with_bad_data_type(self):
        """Test pulling when data is None."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], time_series_topics=["channel1"]
        )

        frame_data = {
            "known_topic": {"time": [2], "data": [20]},
        }

        event_buffer = EventBuffer.from_span(1000000000, 2000000000, frame_data)
        frame = EventFrame(data=[event_buffer], EOS=False)

        pad = Mock()
        with pytest.raises(ValueError, match="Unknown data type"):
            sink.pull(pad, frame)

    def test_pull_unknown_topic(self):
        """Test pulling data for unknown topic."""
        sink = KafkaSink(
            name="TestSink", sink_pad_names=["data"], time_series_topics=["known_topic"]
        )

        frame_data = {
            "unknown_topic": {"time": [1], "data": [10]},
            "known_topic": {"time": [2], "data": [20]},
        }
        event_buffer = EventBuffer.from_span(1000000000, 2000000000, [frame_data])
        frame = EventFrame(data=[event_buffer], EOS=False)

        pad = Mock()
        with pytest.raises(ValueError, match="Unknwon topic"):
            sink.pull(pad, frame)

    def test_pull_with_eos(self):
        """Test pulling with EOS frame."""
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"])
        sink.mark_eos = Mock()

        event_buffer = EventBuffer.from_span(1000000000, 2000000000, None)
        frame = EventFrame(data=[event_buffer], EOS=True)

        pad = Mock()
        sink.pull(pad, frame)

        sink.mark_eos.assert_called_once_with(pad)


class TestKafkaSinkInternal:
    """Test cases for internal method."""

    def test_internal_no_interval(self):
        """Test internal when interval is None."""
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"], interval=None)
        sink.write = Mock()

        sink.internal()

        sink.write.assert_called_once()

    @patch("sgnligo.sinks.kafka_sink.now")
    def test_internal_with_interval_not_ready(self, mock_now):
        """Test internal when interval hasn't elapsed."""
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"], interval=10.0)
        sink.write = Mock()
        sink.last_sent = 100

        mock_now.return_value = 105  # Only 5 seconds elapsed

        sink.internal()

        sink.write.assert_not_called()

    @patch("sgnligo.sinks.kafka_sink.now")
    def test_internal_with_interval_ready(self, mock_now):
        """Test internal when interval has elapsed."""
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"], interval=10.0)
        sink.write = Mock()
        sink.last_sent = 100

        mock_now.return_value = 111  # 11 seconds elapsed

        sink.internal()

        sink.write.assert_called_once()
        assert sink.last_sent == 111

    def test_internal_at_eos_with_client(self, capsys):
        """Test internal when at EOS with kafka client."""
        mock_client = Mock()
        sink = KafkaSink(
            name="TestSink",
            sink_pad_names=["data"],
            output_kafka_server="localhost:9092",
        )
        sink.client = mock_client

        # Mark EOS on the pad
        pad = sink.snks["data"]
        sink.mark_eos(pad)

        sink.internal()

        captured = capsys.readouterr()
        assert "shutdown: KafkaSink: close" in captured.err
        mock_client.close.assert_called_once()

    def test_internal_at_eos_no_client(self, capsys):
        """Test internal when at EOS without kafka client."""
        sink = KafkaSink(name="TestSink", sink_pad_names=["data"])

        # Mark EOS on the pad
        pad = sink.snks["data"]
        sink.mark_eos(pad)

        sink.internal()

        captured = capsys.readouterr()
        assert "shutdown: KafkaSink: close" in captured.err


class TestKafkaSinkIntegration:
    """Integration tests for KafkaSink."""

    @patch("sgnligo.sinks.kafka_sink.now")
    def test_full_workflow_with_kafka(self, mock_now):
        """Test complete workflow with kafka client."""
        mock_now.return_value = 1000

        with patch("sgnligo.sinks.kafka_sink.kafka.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            sink = KafkaSink(
                name="TestSink",
                sink_pad_names=["data"],
                output_kafka_server="localhost:9092",
                time_series_topics=["ts1"],
                trigger_topics=["trig1"],
                tag=["test"],
                prefix="test_",
                interval=5.0,
            )

            # Pull some data
            frame_data = {
                "ts1": {"time": [1, 2], "data": [10, 20]},
                "trig1": [{"snr": 15}],
            }
            event_buffer = EventBuffer.from_span(1000000000, 2000000000, [frame_data])
            frame = EventFrame(data=[event_buffer], EOS=False)

            pad = Mock()
            sink.pull(pad, frame)

            # First internal call - not enough time elapsed
            mock_now.return_value = 1003
            sink.internal()
            mock_client.write.assert_not_called()

            # Second internal call - enough time elapsed
            mock_now.return_value = 1006
            sink.internal()
            assert mock_client.write.call_count == 2  # One for each topic

            # Pull EOS frame
            eos_event_buffer = EventBuffer.from_span(1000000000, 2000000000, None)
            eos_frame = EventFrame(data=[eos_event_buffer], EOS=True)
            sink.pull(pad, eos_frame)

            # Internal at EOS
            sink.mark_eos(pad)
            sink.internal()
            mock_client.close.assert_called_once()

    def test_full_workflow_stdout(self, capsys):
        """Test complete workflow with stdout output."""
        sink = KafkaSink(
            name="TestSink",
            sink_pad_names=["data"],
            time_series_topics=["channel1"],
            trigger_topics=["trigger1"],
        )

        # Pull and write data
        frame_data = {
            "channel1": {"time": [1], "data": [10]},
            "trigger1": [{"snr": 20}],
        }
        event_buffer = EventBuffer.from_span(1000000000, 2000000000, [frame_data])
        frame = EventFrame(data=[event_buffer], EOS=False)

        pad = Mock()
        sink.pull(pad, frame)
        sink.internal()

        captured = capsys.readouterr()
        assert "channel1" in captured.out
        assert "trigger1" in captured.out
