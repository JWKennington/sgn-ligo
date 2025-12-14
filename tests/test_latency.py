#!/usr/bin/env python3
"""Test coverage for sgnligo.transforms.latency module."""

from unittest.mock import Mock, patch

import pytest
from sgnts.base import EventFrame, TSFrame

from sgnligo.transforms.latency import Latency


class TestLatencyInit:
    """Test Latency initialization."""

    def test_init_without_interval(self):
        """Test initialization without interval (lines 31-35)."""
        latency = Latency(
            name="test_latency",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            route="test_route",
            interval=None,
        )
        assert latency.route == "test_route"
        assert latency.interval is None
        assert latency.frame is None
        assert not hasattr(latency, "last_time")
        assert not hasattr(latency, "latencies")

    def test_init_with_interval(self):
        """Test initialization with interval (lines 37-39)."""
        with patch("sgnligo.transforms.latency.now") as mock_now:
            mock_now.return_value = 1000.0
            latency = Latency(
                name="test_latency",
                source_pad_names=("out",),
                sink_pad_names=("in",),
                route="test_route",
                interval=1.0,
            )
        assert latency.route == "test_route"
        assert latency.interval == 1.0
        assert latency.frame is None
        assert latency.last_time == 1000.0
        assert latency.latencies == []

    def test_init_requires_route(self):
        """Test that route must be a string."""
        with pytest.raises(AssertionError):
            Latency(
                name="test_latency",
                source_pad_names=("out",),
                sink_pad_names=("in",),
                route=None,  # Should fail assertion
            )

    def test_init_requires_single_sink_pad(self):
        """Test that exactly one sink pad is required."""
        with pytest.raises(AssertionError):
            Latency(
                name="test_latency",
                source_pad_names=("out",),
                sink_pad_names=("in1", "in2"),  # Two pads - should fail
                route="test_route",
            )


class TestLatencyPull:
    """Test Latency pull method."""

    def test_pull_stores_frame(self):
        """Test pull() stores the frame (line 42)."""
        latency = Latency(
            name="test_latency",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            route="test_route",
        )

        mock_frame = Mock(spec=TSFrame)
        mock_pad = latency.sink_pads[0]

        latency.pull(mock_pad, mock_frame)

        assert latency.frame is mock_frame


class TestLatencyNew:
    """Test Latency new method."""

    def _create_mock_tsframe(self, start_ns=1000000000000, end_ns=1001000000000):
        """Helper to create a mock TSFrame."""
        mock_frame = Mock(spec=TSFrame)
        mock_frame.start = start_ns
        mock_frame.end = end_ns
        mock_frame.EOS = False
        return mock_frame

    def _create_mock_eventframe(self, start_ns=1000000000000, end_ns=1001000000000):
        """Helper to create a mock EventFrame."""
        mock_frame = Mock(spec=EventFrame)
        mock_frame.start = start_ns
        mock_frame.end = end_ns
        mock_frame.EOS = False
        return mock_frame

    def test_new_without_interval(self):
        """Test new() without interval (lines 48-63)."""
        latency = Latency(
            name="test_latency",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            route="test_route",
            interval=None,
        )

        # Frame starts at 1000 seconds (in nanoseconds)
        mock_frame = self._create_mock_tsframe(
            start_ns=1000_000_000_000, end_ns=1001_000_000_000
        )
        latency.frame = mock_frame

        # Mock now() to return 1002 seconds (in nanoseconds)
        mock_now_obj = Mock()
        mock_now_obj.ns.return_value = 1002_000_000_000

        with patch("sgnligo.transforms.latency.now", return_value=mock_now_obj):
            result = latency.new(latency.source_pads[0])

        assert isinstance(result, EventFrame)
        assert result.EOS is False
        event_data = result.data[0].data[0]
        assert "test_route" in event_data
        assert event_data["test_route"]["time"] == [1000.0]
        # Latency should be 2 seconds (1002 - 1000)
        assert event_data["test_route"]["data"] == [2.0]

    def test_new_with_eventframe(self):
        """Test new() with EventFrame input."""
        latency = Latency(
            name="test_latency",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            route="test_route",
            interval=None,
        )

        mock_frame = self._create_mock_eventframe(
            start_ns=1000_000_000_000, end_ns=1001_000_000_000
        )
        latency.frame = mock_frame

        mock_now_obj = Mock()
        mock_now_obj.ns.return_value = 1001_500_000_000  # 1.5 seconds latency

        with patch("sgnligo.transforms.latency.now", return_value=mock_now_obj):
            result = latency.new(latency.source_pads[0])

        assert isinstance(result, EventFrame)
        event_data = result.data[0].data[0]
        assert event_data["test_route"]["data"] == [1.5]

    def test_new_with_interval_before_elapsed(self):
        """Test new() with interval before interval has elapsed (lines 64-66, 79-80)."""
        with patch("sgnligo.transforms.latency.now") as mock_now_init:
            mock_now_init.return_value = 1000.0  # Initial last_time
            latency = Latency(
                name="test_latency",
                source_pad_names=("out",),
                sink_pad_names=("in",),
                route="test_route",
                interval=5.0,  # 5 second interval
            )

        mock_frame = self._create_mock_tsframe(
            start_ns=1000_000_000_000, end_ns=1001_000_000_000
        )
        latency.frame = mock_frame

        # Current time is only 2 seconds after last_time (less than 5s interval)
        mock_now_obj = Mock()
        mock_now_obj.ns.return_value = 1002_000_000_000  # 1002 seconds

        with patch("sgnligo.transforms.latency.now", return_value=mock_now_obj):
            result = latency.new(latency.source_pads[0])

        # Should return empty event_data since interval hasn't elapsed
        event_data = result.data[0].data[0]
        assert event_data == {}
        # Latency should have been appended
        assert len(latency.latencies) == 1

    def test_new_with_interval_after_elapsed(self):
        """Test new() with interval after interval has elapsed (lines 64-78)."""
        with patch("sgnligo.transforms.latency.now") as mock_now_init:
            mock_now_init.return_value = 1000.0  # Initial last_time
            latency = Latency(
                name="test_latency",
                source_pad_names=("out",),
                sink_pad_names=("in",),
                route="test_route",
                interval=5.0,  # 5 second interval
            )

        # Add some latencies
        latency.latencies = [1.0, 2.0, 3.0]

        mock_frame = self._create_mock_tsframe(
            start_ns=1000_000_000_000, end_ns=1001_000_000_000
        )
        latency.frame = mock_frame

        # Current time is 6 seconds after last_time (more than 5s interval)
        mock_now_obj = Mock()
        mock_now_obj.ns.return_value = 1006_000_000_000  # 1006 seconds

        with patch("sgnligo.transforms.latency.now", return_value=mock_now_obj):
            result = latency.new(latency.source_pads[0])

        event_data = result.data[0].data[0]
        assert "test_route" in event_data
        # Should report max latency (3.0 from existing + 6.0 from current)
        assert event_data["test_route"]["data"] == [6.0]
        # Latencies should be reset
        assert latency.latencies == []
        # last_time should be updated
        assert latency.last_time == 1006.0

    def test_new_with_eos(self):
        """Test new() passes through EOS flag."""
        latency = Latency(
            name="test_latency",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            route="test_route",
        )

        mock_frame = self._create_mock_tsframe()
        mock_frame.EOS = True
        latency.frame = mock_frame

        mock_now_obj = Mock()
        mock_now_obj.ns.return_value = 1002_000_000_000

        with patch("sgnligo.transforms.latency.now", return_value=mock_now_obj):
            result = latency.new(latency.source_pads[0])

        assert result.EOS is True


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
