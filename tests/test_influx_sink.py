"""Test coverage for sgnligo.sinks.influx_sink module."""

from collections import deque
from unittest.mock import Mock, patch

import pytest


class TestInfluxSinkInit:
    """Test InfluxSink initialization."""

    def test_post_init(self, tmp_path):
        """Test __post_init__ sets up aggregator correctly."""
        # Create a mock scald config file
        config_content = """
backends:
  default:
    host: localhost
    port: 8086
    database: test_db
"""
        config_file = tmp_path / "scald_config.yaml"
        config_file.write_text(config_content)

        with patch("sgnligo.sinks.influx_sink.influx.Aggregator") as mock_agg:
            mock_agg_instance = Mock()
            mock_agg.return_value = mock_agg_instance

            from sgnligo.sinks.influx_sink import InfluxSink

            sink = InfluxSink(
                name="test_sink",
                sink_pad_names=("input",),
                instrument="H1",
                metadata_key="horizon",
                scald_config=str(config_file),
                route="test_route",
                verbose=False,
            )

            # Verify aggregator was created with config
            mock_agg.assert_called_once_with(
                host="localhost",
                port=8086,
                database="test_db",
            )

            # Verify load was called
            mock_agg_instance.load.assert_called_once_with(path=str(config_file))

            # Verify internal state was initialized
            assert sink.last_reduce_time is None
            assert sink.last_t0 is None
            assert isinstance(sink.timedeq, deque)
            assert sink.timedeq.maxlen == 100
            assert "test_route" in sink.datadeq
            assert isinstance(sink.datadeq["test_route"], deque)


class TestInfluxSinkPull:
    """Test InfluxSink pull method."""

    def _create_sink_with_mocks(self, tmp_path, verbose=False):
        """Helper to create a sink with mocked dependencies."""
        config_content = """
backends:
  default:
    host: localhost
    port: 8086
    database: test_db
"""
        config_file = tmp_path / "scald_config.yaml"
        config_file.write_text(config_content)

        with patch("sgnligo.sinks.influx_sink.influx.Aggregator") as mock_agg:
            mock_agg_instance = Mock()
            mock_agg.return_value = mock_agg_instance

            from sgnligo.sinks.influx_sink import InfluxSink

            sink = InfluxSink(
                name="test_sink",
                sink_pad_names=("input",),
                instrument="H1",
                metadata_key="horizon",
                scald_config=str(config_file),
                route="test_route",
                verbose=verbose,
                wait_time=2,
            )

            return sink, mock_agg_instance

    def test_pull_first_call_sets_last_t0(self, tmp_path):
        """Test that first pull call sets last_t0."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path)

        # Create mock pad and buffer
        mock_pad = sink.sink_pads[0]
        mock_buf = Mock()
        mock_buf.t0 = 1_000_000_000  # 1 second in nanoseconds (1e9)
        mock_buf.EOS = False

        mock_bufs = Mock()
        mock_bufs.__getitem__ = Mock(return_value=mock_buf)
        mock_bufs.metadata = {}
        mock_bufs.EOS = False

        # Verify last_t0 is None before pull
        assert sink.last_t0 is None

        # Call pull
        sink.pull(mock_pad, mock_bufs)

        # Verify last_t0 was set
        assert sink.last_t0 == 1_000_000_000

    def test_pull_with_metadata_key(self, tmp_path):
        """Test pull when metadata_key is in buffer metadata."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path)

        mock_pad = sink.sink_pads[0]
        mock_buf = Mock()
        mock_buf.t0 = 1_000_000_000  # 1 second in nanoseconds (1e9)

        mock_bufs = Mock()
        mock_bufs.__getitem__ = Mock(return_value=mock_buf)
        mock_bufs.metadata = {"horizon": 150}  # metadata_key is "horizon"
        mock_bufs.EOS = False

        # Call pull
        sink.pull(mock_pad, mock_bufs)

        # Verify data was added to deques
        assert len(sink.timedeq) == 1
        assert sink.timedeq[0] == 1  # 1_000_000_000 / 1e9 = 1
        assert len(sink.datadeq["test_route"]) == 1
        assert sink.datadeq["test_route"][0] == 150

    def test_pull_without_metadata_key(self, tmp_path):
        """Test pull when metadata_key is NOT in buffer metadata."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path)

        mock_pad = sink.sink_pads[0]
        mock_buf = Mock()
        mock_buf.t0 = 1_000_000_000  # 1 second in nanoseconds

        mock_bufs = Mock()
        mock_bufs.__getitem__ = Mock(return_value=mock_buf)
        mock_bufs.metadata = {}  # No horizon key
        mock_bufs.EOS = False

        # Call pull
        sink.pull(mock_pad, mock_bufs)

        # Verify deques are still empty (no data added)
        assert len(sink.timedeq) == 0
        assert len(sink.datadeq["test_route"]) == 0

    def test_pull_triggers_write_after_wait_time(self, tmp_path, capsys):
        """Test pull triggers influx write after wait_time elapsed."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path)

        mock_pad = sink.sink_pads[0]

        # First buffer at t=0
        mock_buf1 = Mock()
        mock_buf1.t0 = 0
        mock_bufs1 = Mock()
        mock_bufs1.__getitem__ = Mock(return_value=mock_buf1)
        mock_bufs1.metadata = {"horizon": 100}
        mock_bufs1.EOS = False

        sink.pull(mock_pad, mock_bufs1)

        # Second buffer at t=3 seconds (> wait_time of 2 seconds)
        mock_buf2 = Mock()
        mock_buf2.t0 = 3_000_000_000  # 3 seconds in nanoseconds (3e9)
        mock_bufs2 = Mock()
        mock_bufs2.__getitem__ = Mock(return_value=mock_buf2)
        mock_bufs2.metadata = {"horizon": 200}
        mock_bufs2.EOS = False

        sink.pull(mock_pad, mock_bufs2)

        # Verify store_columns was called
        mock_agg.store_columns.assert_called_once()
        call_args = mock_agg.store_columns.call_args
        assert call_args[0][0] == "test_route"
        assert call_args[1]["aggregate"] == "max"

        # Verify "Writing out to influx" was printed
        captured = capsys.readouterr()
        assert "Writing out to influx" in captured.out

        # Verify deques were cleared after write
        assert len(sink.timedeq) == 0
        assert len(sink.datadeq["test_route"]) == 0

    def test_pull_no_write_before_wait_time(self, tmp_path):
        """Test pull does NOT trigger write before wait_time elapsed."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path)

        mock_pad = sink.sink_pads[0]

        # First buffer at t=0
        mock_buf1 = Mock()
        mock_buf1.t0 = 0
        mock_bufs1 = Mock()
        mock_bufs1.__getitem__ = Mock(return_value=mock_buf1)
        mock_bufs1.metadata = {"horizon": 100}
        mock_bufs1.EOS = False

        sink.pull(mock_pad, mock_bufs1)

        # Second buffer at t=1 second (< wait_time of 2 seconds)
        mock_buf2 = Mock()
        mock_buf2.t0 = 1_000_000_000  # 1 second in nanoseconds (1e9)
        mock_bufs2 = Mock()
        mock_bufs2.__getitem__ = Mock(return_value=mock_buf2)
        mock_bufs2.metadata = {"horizon": 200}
        mock_bufs2.EOS = False

        sink.pull(mock_pad, mock_bufs2)

        # Verify store_columns was NOT called
        mock_agg.store_columns.assert_not_called()

        # Verify data is still in deques
        assert len(sink.timedeq) == 2
        assert len(sink.datadeq["test_route"]) == 2

    def test_pull_with_eos(self, tmp_path):
        """Test pull handles EOS correctly."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path)

        mock_pad = sink.sink_pads[0]

        # Mock mark_eos
        sink.mark_eos = Mock()

        mock_buf = Mock()
        mock_buf.t0 = 1_000_000_000  # 1 second in nanoseconds
        mock_bufs = Mock()
        mock_bufs.__getitem__ = Mock(return_value=mock_buf)
        mock_bufs.metadata = {}
        mock_bufs.EOS = True  # EOS flag set

        sink.pull(mock_pad, mock_bufs)

        # Verify mark_eos was called
        sink.mark_eos.assert_called_once_with(mock_pad)

    def test_pull_with_verbose(self, tmp_path, capsys):
        """Test pull with verbose=True prints buffer info."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path, verbose=True)

        mock_pad = sink.sink_pads[0]
        mock_buf = Mock()
        mock_buf.t0 = 1_000_000_000  # 1 second in nanoseconds
        mock_bufs = Mock()
        mock_bufs.__getitem__ = Mock(return_value=mock_buf)
        mock_bufs.metadata = {}
        mock_bufs.EOS = False
        mock_bufs.__repr__ = Mock(return_value="MockBuffer")

        sink.pull(mock_pad, mock_bufs)

        # Verify verbose output
        captured = capsys.readouterr()
        assert "1" in captured.out  # cnt value
        assert "MockBuffer" in captured.out

    def test_pull_increments_counter(self, tmp_path):
        """Test that pull increments the counter for the pad."""
        sink, mock_agg = self._create_sink_with_mocks(tmp_path)

        mock_pad = sink.sink_pads[0]
        mock_buf = Mock()
        mock_buf.t0 = 1_000_000_000  # 1 second in nanoseconds
        mock_bufs = Mock()
        mock_bufs.__getitem__ = Mock(return_value=mock_buf)
        mock_bufs.metadata = {}
        mock_bufs.EOS = False

        # Verify counter starts at 0
        assert sink.cnt[mock_pad] == 0

        # First pull
        sink.pull(mock_pad, mock_bufs)
        assert sink.cnt[mock_pad] == 1

        # Second pull
        sink.pull(mock_pad, mock_bufs)
        assert sink.cnt[mock_pad] == 2


class TestInfluxSinkIntegration:
    """Integration tests for InfluxSink."""

    def test_multiple_pulls_accumulate_data(self, tmp_path):
        """Test that multiple pulls accumulate data in deques."""
        config_content = """
backends:
  default:
    host: localhost
    port: 8086
    database: test_db
"""
        config_file = tmp_path / "scald_config.yaml"
        config_file.write_text(config_content)

        with patch("sgnligo.sinks.influx_sink.influx.Aggregator") as mock_agg:
            mock_agg_instance = Mock()
            mock_agg.return_value = mock_agg_instance

            from sgnligo.sinks.influx_sink import InfluxSink

            sink = InfluxSink(
                name="test_sink",
                sink_pad_names=("input",),
                instrument="H1",
                metadata_key="horizon",
                scald_config=str(config_file),
                route="test_route",
                wait_time=10,  # Long wait time so we don't trigger write
            )

            mock_pad = sink.sink_pads[0]

            # Pull multiple times with incrementing timestamps (0.1 second increments)
            for i in range(5):
                mock_buf = Mock()
                mock_buf.t0 = i * 100_000_000  # 0.1 second increments (100ms in ns)
                mock_bufs = Mock()
                mock_bufs.__getitem__ = Mock(return_value=mock_buf)
                mock_bufs.metadata = {"horizon": 100 + i}
                mock_bufs.EOS = False

                sink.pull(mock_pad, mock_bufs)

            # Verify all data accumulated
            assert len(sink.timedeq) == 5
            assert len(sink.datadeq["test_route"]) == 5
            assert list(sink.datadeq["test_route"]) == [100, 101, 102, 103, 104]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
