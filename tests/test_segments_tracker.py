"""Comprehensive tests for GapTransitionSink and SegmentsTrackerSink."""

from unittest.mock import MagicMock

import numpy as np
from igwn_segments import segment, segmentlist
from sgnts.base import Offset, SeriesBuffer, TSFrame

from sgnligo.sinks.segments_tracker import (
    GapTracker,
    GapTransitionSink,
    SegmentsTrackerSink,
)


class TestGapTracker:
    """Test cases for the standalone GapTracker helper class."""

    def test_init_basic(self):
        """Test GapTracker initialization."""
        tracker = GapTracker(["ch1", "ch2"])

        assert "ch1" in tracker.segments
        assert "ch2" in tracker.segments
        assert len(tracker.segments["ch1"]) == 0

    def test_init_with_tuple(self):
        """Test GapTracker with tuple of pad names."""
        tracker = GapTracker(("H1", "L1"))

        assert "H1" in tracker.segments
        assert "L1" in tracker.segments

    def test_track_creates_segment(self):
        """Test that tracking an ON buffer creates a segment."""
        tracker = GapTracker(["ch1"])

        buf = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        tracker.track("ch1", buf)

        assert len(tracker.segments["ch1"]) == 1
        assert tracker.segments["ch1"][0] == segment(1000.0, 1001.0)

    def test_on_transition_callback(self):
        """Test that on_transition callback is called on state changes."""
        transitions = []

        def callback(pad_name, timestamp, is_on):
            transitions.append((pad_name, timestamp, is_on))

        tracker = GapTracker(["ch1"], on_transition=callback)

        # ON buffer
        buf1 = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        tracker.track("ch1", buf1)

        # OFF buffer
        buf2 = SeriesBuffer(
            offset=Offset.fromsec(1001.0),
            sample_rate=16,
            data=None,
            shape=(16,),
        )
        tracker.track("ch1", buf2)

        assert transitions == [
            ("ch1", 1000.0, True),
            ("ch1", 1001.0, False),
        ]

    def test_add_pad(self):
        """Test adding a new pad dynamically."""
        tracker = GapTracker(["ch1"])

        assert "ch2" not in tracker.segments

        tracker.add_pad("ch2")

        assert "ch2" in tracker.segments
        assert tracker._prev_state["ch2"] is None

    def test_get_state_at_time(self):
        """Test get_state_at_time method."""
        tracker = GapTracker(["ch1"])

        # Add a segment manually
        tracker.segments["ch1"].append(segment(1000.0, 2000.0))

        assert tracker.get_state_at_time("ch1", 1500.0) is True
        assert tracker.get_state_at_time("ch1", 500.0) is False
        assert tracker.get_state_at_time("ch1", 2500.0) is False

    def test_get_state_unknown_pad(self):
        """Test get_state_at_time with unknown pad."""
        tracker = GapTracker(["ch1"])

        assert tracker.get_state_at_time("unknown", 1000.0) is None


class TestGapTransitionSinkInit:
    """Test cases for GapTransitionSink initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        assert list(tracker.sink_pad_names) == ["H1"]

    def test_init_with_multiple_pads(self):
        """Test initialization with multiple pads."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1", "L1"),
        )
        assert list(tracker.sink_pad_names) == ["H1", "L1"]


class TestGapTransitionSinkConfigure:
    """Test cases for configure method."""

    def test_configure_initializes_segments(self):
        """Test that configure initializes segments structure."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1", "L1"),
        )
        tracker.configure()

        assert "H1" in tracker.segments
        assert "L1" in tracker.segments
        assert isinstance(tracker.segments["H1"], segmentlist)
        assert len(tracker.segments["H1"]) == 0

    def test_configure_initializes_gap_tracker(self):
        """Test that configure initializes the gap tracker."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1", "L1", "V1"),
        )
        tracker.configure()

        assert tracker._gap_tracker is not None
        assert tracker._gap_tracker._prev_state == {"H1": None, "L1": None, "V1": None}


class TestGapTransitionSinkTrack:
    """Test cases for gap tracking via the helper."""

    def test_track_gap_buffer(self):
        """Test that gap buffer does not create segment."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        buf = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=None,  # Gap buffer
            shape=(16,),
        )

        tracker._gap_tracker.track("H1", buf)

        assert len(tracker.segments["H1"]) == 0

    def test_track_non_gap_creates_segment(self):
        """Test that non-gap buffer creates segment."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        buf = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )

        tracker._gap_tracker.track("H1", buf)

        assert len(tracker.segments["H1"]) == 1
        seg = tracker.segments["H1"][0]
        assert seg[0] == 1000.0
        assert seg[1] == 1001.0  # 16 samples at 16 Hz = 1 second

    def test_track_extends_segment(self):
        """Test that consecutive ON buffers extend segment."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        # First ON buffer
        buf1 = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        tracker._gap_tracker.track("H1", buf1)

        # Second ON buffer (consecutive)
        buf2 = SeriesBuffer(
            offset=Offset.fromsec(1001.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        tracker._gap_tracker.track("H1", buf2)

        # Should still be one segment, but extended
        assert len(tracker.segments["H1"]) == 1
        seg = tracker.segments["H1"][0]
        assert seg[0] == 1000.0
        assert seg[1] == 1002.0

    def test_track_gap_closes_segment(self):
        """Test that gap after ON closes the segment."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        # ON buffer
        buf1 = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        tracker._gap_tracker.track("H1", buf1)

        # Gap buffer
        buf2 = SeriesBuffer(
            offset=Offset.fromsec(1001.0),
            sample_rate=16,
            data=None,
            shape=(16,),
        )
        tracker._gap_tracker.track("H1", buf2)

        # Segment should be [1000, 1001)
        assert len(tracker.segments["H1"]) == 1
        seg = tracker.segments["H1"][0]
        assert seg[0] == 1000.0
        assert seg[1] == 1001.0

    def test_track_creates_new_segment_after_gap(self):
        """Test that ON after gap creates new segment."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        # ON -> OFF -> ON sequence
        bufs = [
            SeriesBuffer(
                offset=Offset.fromsec(1000.0),
                sample_rate=16,
                data=np.ones(16, dtype=np.float32),
                shape=(16,),
            ),
            SeriesBuffer(
                offset=Offset.fromsec(1001.0), sample_rate=16, data=None, shape=(16,)
            ),
            SeriesBuffer(
                offset=Offset.fromsec(1002.0),
                sample_rate=16,
                data=np.ones(16, dtype=np.float32),
                shape=(16,),
            ),
        ]

        for buf in bufs:
            tracker._gap_tracker.track("H1", buf)

        # Should have two segments
        assert len(tracker.segments["H1"]) == 2
        assert tracker.segments["H1"][0] == segment(1000.0, 1001.0)
        assert tracker.segments["H1"][1] == segment(1002.0, 1003.0)


class TestGapTransitionSinkProcess:
    """Test cases for process method."""

    def test_process_single_frame(self):
        """Test processing a single frame."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        pad = MagicMock()
        pad.pad_name = "H1"

        buf = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        frame = MagicMock(spec=TSFrame)
        frame.__iter__ = lambda self: iter([buf])
        frame.EOS = False

        tracker.process({pad: frame})

        assert len(tracker.segments["H1"]) == 1

    def test_process_with_eos(self):
        """Test processing a frame with EOS."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        pad = MagicMock()
        pad.pad_name = "H1"

        buf = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        frame = MagicMock(spec=TSFrame)
        frame.__iter__ = lambda self: iter([buf])
        frame.EOS = True

        tracker.mark_eos = MagicMock()

        tracker.process({pad: frame})

        tracker.mark_eos.assert_called_once_with(pad)

    def test_process_multiple_pads(self):
        """Test processing frames from multiple pads."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1", "L1"),
        )
        tracker.configure()

        pad_h1 = MagicMock()
        pad_h1.pad_name = "H1"
        pad_l1 = MagicMock()
        pad_l1.pad_name = "L1"

        buf_h1 = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=np.ones(16, dtype=np.float32),
            shape=(16,),
        )
        buf_l1 = SeriesBuffer(
            offset=Offset.fromsec(1000.0),
            sample_rate=16,
            data=None,  # Gap
            shape=(16,),
        )

        frame_h1 = MagicMock(spec=TSFrame)
        frame_h1.__iter__ = lambda self: iter([buf_h1])
        frame_h1.EOS = False

        frame_l1 = MagicMock(spec=TSFrame)
        frame_l1.__iter__ = lambda self: iter([buf_l1])
        frame_l1.EOS = False

        tracker.process({pad_h1: frame_h1, pad_l1: frame_l1})

        assert len(tracker.segments["H1"]) == 1  # ON
        assert len(tracker.segments["L1"]) == 0  # OFF (gap)


class TestGapTransitionSinkGetStateAtTime:
    """Test cases for get_state_at_time method."""

    def test_get_state_no_segments(self):
        """Test get_state_at_time with no recorded segments."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        result = tracker.get_state_at_time("H1", 1000.0)
        assert result is None

    def test_get_state_unknown_pad(self):
        """Test get_state_at_time with unknown pad."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        result = tracker.get_state_at_time("L1", 1000.0)
        assert result is None

    def test_get_state_in_segment(self):
        """Test get_state_at_time for time within a segment."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        # Add a segment manually
        tracker.segments["H1"].append(segment(1000.0, 2000.0))

        assert tracker.get_state_at_time("H1", 1500.0) is True

    def test_get_state_outside_segment(self):
        """Test get_state_at_time for time outside segments."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        tracker.segments["H1"].append(segment(1000.0, 2000.0))

        assert tracker.get_state_at_time("H1", 500.0) is False
        assert tracker.get_state_at_time("H1", 2500.0) is False

    def test_get_state_at_segment_boundary(self):
        """Test get_state_at_time at segment boundaries."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        tracker.segments["H1"].append(segment(1000.0, 2000.0))

        # Start is inclusive
        assert tracker.get_state_at_time("H1", 1000.0) is True
        # End is exclusive (half-open interval)
        assert tracker.get_state_at_time("H1", 2000.0) is False


class TestGapTransitionSinkIntegration:
    """Integration tests for GapTransitionSink."""

    def test_segment_operations(self):
        """Test that segments support IGWN segment operations."""
        tracker = GapTransitionSink(
            name="Tracker",
            sink_pad_names=("H1", "L1"),
        )
        tracker.configure()

        # Add segments
        tracker.segments["H1"].append(segment(100.0, 200.0))
        tracker.segments["H1"].append(segment(300.0, 400.0))
        tracker.segments["L1"].append(segment(150.0, 350.0))

        # Test intersection (when both are ON)
        both_on = tracker.segments["H1"] & tracker.segments["L1"]
        assert len(both_on) == 2
        assert segment(150.0, 200.0) in both_on
        assert segment(300.0, 350.0) in both_on

    def test_multiple_transitions(self):
        """Test multiple state transitions are recorded correctly."""
        tracker = GapTransitionSink(
            name="Tracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        # Simulate: ON(1000-1002) -> OFF(1002-1003) -> ON(1003-1005)
        bufs = [
            SeriesBuffer(
                offset=Offset.fromsec(1000.0),
                sample_rate=16,
                data=np.ones(16),
                shape=(16,),
            ),
            SeriesBuffer(
                offset=Offset.fromsec(1001.0),
                sample_rate=16,
                data=np.ones(16),
                shape=(16,),
            ),
            SeriesBuffer(
                offset=Offset.fromsec(1002.0), sample_rate=16, data=None, shape=(16,)
            ),
            SeriesBuffer(
                offset=Offset.fromsec(1003.0),
                sample_rate=16,
                data=np.ones(16),
                shape=(16,),
            ),
            SeriesBuffer(
                offset=Offset.fromsec(1004.0),
                sample_rate=16,
                data=np.ones(16),
                shape=(16,),
            ),
        ]

        for buf in bufs:
            tracker._gap_tracker.track("H1", buf)

        assert len(tracker.segments["H1"]) == 2
        assert tracker.segments["H1"][0] == segment(1000.0, 1002.0)
        assert tracker.segments["H1"][1] == segment(1003.0, 1005.0)

        # Query states
        assert tracker.get_state_at_time("H1", 1001.5) is True
        assert tracker.get_state_at_time("H1", 1002.5) is False
        assert tracker.get_state_at_time("H1", 1004.0) is True


class TestGapTransitionSinkBeforeConfigure:
    """Test cases for accessing properties before configure is called."""

    def test_segments_before_configure(self):
        """Test segments property returns empty dict when _gap_tracker is None."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        # Force _gap_tracker to be None to test the early return path
        tracker._gap_tracker = None

        result = tracker.segments
        assert len(result) == 0

    def test_get_state_at_time_before_configure(self):
        """Test get_state_at_time returns None when _gap_tracker is None."""
        tracker = GapTransitionSink(
            name="TestTracker",
            sink_pad_names=("H1",),
        )
        # Force _gap_tracker to be None to test the early return path
        tracker._gap_tracker = None

        result = tracker.get_state_at_time("H1", 1000.0)
        assert result is None


class TestSegmentsTrackerSink:
    """Test cases for backward-compatible SegmentsTrackerSink."""

    def test_transitions_property(self):
        """Test that transitions property converts segments to events."""
        tracker = SegmentsTrackerSink(
            name="Tracker",
            sink_pad_names=("H1",),
        )
        tracker.configure()

        tracker.segments["H1"].append(segment(1000.0, 2000.0))
        tracker.segments["H1"].append(segment(3000.0, 4000.0))

        transitions = tracker.transitions["H1"]
        assert transitions == [
            (1000.0, 1.0),  # ON
            (2000.0, 0.0),  # OFF
            (3000.0, 1.0),  # ON
            (4000.0, 0.0),  # OFF
        ]

    def test_gate_history_property(self):
        """Test that gate_history provides gstlal-compatible structure."""
        tracker = SegmentsTrackerSink(
            name="Tracker",
            sink_pad_names=("H1", "L1"),
        )
        tracker.configure()

        tracker.segments["H1"].append(segment(1000.0, 2000.0))

        gate_history = tracker.gate_history
        assert "statevectorsegments" in gate_history
        assert "H1" in gate_history["statevectorsegments"]
        assert gate_history["statevectorsegments"]["H1"] == [
            (1000.0, 1.0),
            (2000.0, 0.0),
        ]
