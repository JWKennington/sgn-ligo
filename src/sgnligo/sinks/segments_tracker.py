"""A sink element to track gap/non-gap transitions on input pads using IGWN segments."""

# Copyright (C) 2024

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from igwn_segments import segment, segmentlist, segmentlistdict
from sgn.base import SinkPad
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSink

logger = logging.getLogger("sgn")


class GapTracker:
    """Standalone helper class to track gap/non-gap transitions using IGWN segments.

    This class can be composed into any SGN element (transform, sink, etc.) to
    track contiguous ON periods (non-gap buffers) as segments.

    Args:
        pad_names:
            Iterable of pad names to track (e.g., ["channel1", "channel2"])
        name:
            str, Name for logging (typically the element name)
        on_transition:
            Optional callback function called on state transitions.
            Signature: on_transition(pad_name: str, timestamp: float, is_on: bool)

    Attributes:
        segments:
            segmentlistdict mapping pad names to segmentlists of ON periods.
            Each segment represents a contiguous period where data was present.

    Example:
        >>> # Usage in a custom transform:
        >>> class MyTransform(TSTransform):
        ...     def configure(self):
        ...         super().configure()
        ...         self.gap_tracker = GapTracker(
        ...             self.sink_pad_names,
        ...             name=self.name,
        ...         )
        ...
        ...     def process(self, input_frames, output_frames):
        ...         for pad, frame in input_frames.items():
        ...             for buf in frame:
        ...                 self.gap_tracker.track(pad.pad_name, buf)
        ...         # ... rest of processing
        ...
        ...     @property
        ...     def segments(self):
        ...         return self.gap_tracker.segments
    """

    def __init__(
        self,
        pad_names: list[str] | tuple[str, ...],
        name: str = "GapTracker",
        on_transition: Optional[Callable[[str, float, bool], None]] = None,
    ):
        self.name = name
        self.on_transition = on_transition
        self._logger = logger.getChild(name)

        pad_names_list = list(pad_names)
        self.segments: segmentlistdict = segmentlistdict(
            (name, segmentlist()) for name in pad_names_list
        )
        self._prev_state: dict[str, Optional[bool]] = {
            name: None for name in pad_names_list
        }

        self._logger.info("Tracking pads %s", pad_names_list)

    def track(self, pad_name: str, buf: SeriesBuffer) -> None:
        """Track a buffer and update segments if needed.

        Args:
            pad_name:
                str, The pad name
            buf:
                SeriesBuffer, The incoming buffer to track
        """
        buf_start = float(Offset.tosec(buf.offset))
        buf_end = float(Offset.tosec(buf.end_offset))

        is_on = not buf.is_gap
        was_on = self._prev_state[pad_name]
        seglist = self.segments[pad_name]

        if is_on:
            if was_on:
                # Continuing ON - extend current segment's end
                last_seg = seglist.pop()
                seglist.append(segment(last_seg[0], buf_end))
            else:
                # Transition to ON - create new segment
                seglist.append(segment(buf_start, buf_end))
                self._logger.info("%s transition: ON @ %s", pad_name, buf_start)
                if self.on_transition:
                    self.on_transition(pad_name, buf_start, True)
        else:
            # OFF - segment already closed at correct end time
            if was_on:
                self._logger.info("%s transition: OFF @ %s", pad_name, buf_start)
                if self.on_transition:
                    self.on_transition(pad_name, buf_start, False)

        self._prev_state[pad_name] = is_on

    def get_state_at_time(self, pad_name: str, timestamp: float) -> Optional[bool]:
        """Get the state of a pad at a specific time.

        Args:
            pad_name:
                str, The pad name
            timestamp:
                float, Timestamp to query

        Returns:
            bool or None: True if on, False if off, None if no data available
        """
        seglist = self.segments.get(pad_name)
        if seglist is None:
            return None

        if timestamp in seglist:
            return True

        # If we have segments, we know the state; if not, return None
        return False if seglist else None

    def add_pad(self, pad_name: str) -> None:
        """Add a new pad to track.

        Args:
            pad_name:
                str, The pad name to add
        """
        if pad_name not in self.segments:
            self.segments[pad_name] = segmentlist()
            self._prev_state[pad_name] = None


@dataclass
class GapTransitionSink(TSSink):
    """Sink element that tracks gap/non-gap transitions on input pads.

    Monitors incoming buffers and records on/off segments based on gap status.
    A gap buffer (data is None) represents an "off" state, while a non-gap buffer
    represents an "on" state. Contiguous ON periods are stored as segments.

    Uses IGWN segments infrastructure for efficient segment operations.

    Args:
        sink_pad_names:
            Tuple of pad names to track (e.g., ("channel1", "channel2"))

    Attributes:
        segments:
            segmentlistdict mapping pad names to segmentlists of ON periods.
            Each segment represents a contiguous period where data was present.

    Example:
        >>> tracker = GapTransitionSink(
        ...     name="Tracker",
        ...     sink_pad_names=("ch1", "ch2"),
        ... )
        >>> pipeline.insert(tracker, link_map={
        ...     "Tracker:snk:ch1": "Source:src:ch1",
        ...     "Tracker:snk:ch2": "Source:src:ch2",
        ... })
        >>> # After running, access segment history:
        >>> tracker.segments["ch1"]
        [segment(100.0, 200.0), segment(300.0, 400.0)]
        >>> # Check if a time is in an ON period:
        >>> 150.0 in tracker.segments["ch1"]
        True
    """

    # The gap tracker helper instance
    _gap_tracker: Optional[GapTracker] = field(default=None, init=False, repr=False)

    def configure(self) -> None:
        """Initialize the gap tracker."""
        super().configure()

        self._gap_tracker = GapTracker(
            pad_names=list(self.sink_pad_names),
            name=self.name,
        )

    @property
    def segments(self) -> segmentlistdict:
        """Segment lists per pad: {pad_name: segmentlist of ON periods}."""
        if self._gap_tracker is None:
            return segmentlistdict()
        return self._gap_tracker.segments

    def process(self, input_frames: dict[SinkPad, TSFrame]) -> None:
        """Process incoming frames and record any gap transitions.

        Args:
            input_frames:
                Dictionary mapping sink pads to their input TSFrames
        """
        assert self._gap_tracker is not None

        for pad, frame in input_frames.items():
            pad_name = pad.pad_name

            for buf in frame:
                self._gap_tracker.track(pad_name, buf)

            if frame.EOS:
                self.mark_eos(pad)

    def get_state_at_time(self, pad_name: str, timestamp: float) -> Optional[bool]:
        """Get the state of a pad at a specific time.

        Args:
            pad_name:
                str, The pad name
            timestamp:
                float, Timestamp to query

        Returns:
            bool or None: True if on, False if off, None if no data available
        """
        if self._gap_tracker is None:
            return None
        return self._gap_tracker.get_state_at_time(pad_name, timestamp)


# Backward-compatible alias with gstlal-compatible interface
@dataclass
class SegmentsTrackerSink(GapTransitionSink):
    """Gap transition tracker with gstlal-compatible interface.

    This is a convenience subclass that provides a gate_history property
    matching the gstlal SegmentsTracker interface for LIGO applications.

    The gate_history structure is:
        {"statevectorsegments": {instrument: deque of (timestamp, on_off)}}

    Example:
        >>> tracker = SegmentsTrackerSink(
        ...     name="StateTracker",
        ...     sink_pad_names=("H1", "L1"),
        ... )
        >>> # After running:
        >>> tracker.gate_history["statevectorsegments"]["H1"]
        [(1234567890.0, 1.0), (1234567900.0, 0.0), ...]
    """

    @property
    def transitions(self) -> dict[str, list[tuple[float, float]]]:
        """Convert segments to transition events for backward compatibility.

        Returns:
            dict mapping pad names to lists of (timestamp, on_off) tuples.
        """
        result = {}
        for pad_name, seglist in self.segments.items():
            transitions = []
            for seg in seglist:
                transitions.append((float(seg[0]), 1.0))  # ON
                transitions.append((float(seg[1]), 0.0))  # OFF
            result[pad_name] = transitions
        return result

    @property
    def gate_history(self) -> dict[str, dict[str, list[tuple[float, float]]]]:
        """gstlal-compatible gate history structure."""
        return {"statevectorsegments": self.transitions}
