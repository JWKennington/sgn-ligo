#!/usr/bin/env python3
"""Test coverage for sgnligo.transforms.horizon module."""

from unittest.mock import Mock

import lal
import pytest
from sgnts.base import Offset

from sgnligo.transforms.horizon import HorizonDistanceTracker


class TestHorizonDistanceTrackerInit:
    """Test HorizonDistanceTracker initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
        )
        assert tracker.horizon_distance_funcs is None
        assert tracker.ifo is None
        assert tracker.range is False

    def test_init_with_single_func(self):
        """Test initialization with a single horizon distance function."""

        def mock_func(psd, snr):
            return (100.0, None)

        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs=mock_func,
            ifo="H1",
        )
        assert tracker.horizon_distance_funcs == mock_func
        assert tracker.ifo == "H1"

    def test_init_with_dict_funcs(self):
        """Test initialization with dict of horizon distance functions."""

        def mock_func1(psd, snr):
            return (100.0, None)

        def mock_func2(psd, snr):
            return (150.0, None)

        funcs = {"bank1": mock_func1, "bank2": mock_func2}
        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs=funcs,
            ifo="L1",
            range=True,
        )
        assert tracker.horizon_distance_funcs == funcs
        assert tracker.ifo == "L1"
        assert tracker.range is True


class TestHorizonDistanceTrackerNew:
    """Test HorizonDistanceTracker new() method."""

    def _create_mock_frame(self, psd, epoch=1000000000):
        """Helper to create a mock frame with metadata."""
        mock_frame = Mock()
        mock_frame.EOS = False
        mock_frame.shape = (2048,)
        mock_frame.offset = Offset.fromsec(epoch)
        mock_frame.sample_rate = 2048
        mock_frame.metadata = {
            "psd": psd,
            "navg": 10,
            "n_samples": 16384,
            "epoch": epoch,
        }
        return mock_frame

    def _create_mock_psd(self):
        """Helper to create a mock PSD that passes isinstance check."""
        # Create a real REAL8FrequencySeries with proper unit
        psd = lal.CreateREAL8FrequencySeries(
            "psd",
            lal.LIGOTimeGPS(0),
            0.0,
            1.0,
            lal.StrainUnit,
            1024,
        )
        return psd

    def test_new_with_single_func_range_false(self):
        """Test new() with single func and range=False (lines 53-54, 62-68)."""

        def mock_horizon_func(psd, snr):
            return (200.0, None)

        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs=mock_horizon_func,
            ifo="H1",
            range=False,
        )

        mock_psd = self._create_mock_psd()
        mock_frame = self._create_mock_frame(mock_psd)
        tracker.preparedframes = {tracker.sink_pads[0]: mock_frame}

        result = tracker.new(tracker.source_pads[0])

        assert result.EOS is False
        assert len(result.data) == 1
        event_data = result.data[0].data[0]
        assert event_data["horizon"] == 200.0
        assert event_data["ifo"] == "H1"
        assert event_data["navg"] == 10
        assert event_data["n_samples"] == 16384

    def test_new_with_dict_funcs_range_false(self):
        """Test new() with dict of funcs and range=False (lines 48-52, 62-68)."""

        def mock_func1(psd, snr):
            return (100.0, None)

        def mock_func2(psd, snr):
            return (150.0, None)

        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs={"bank1": mock_func1, "bank2": mock_func2},
            ifo="L1",
            range=False,
        )

        mock_psd = self._create_mock_psd()
        mock_frame = self._create_mock_frame(mock_psd)
        tracker.preparedframes = {tracker.sink_pads[0]: mock_frame}

        result = tracker.new(tracker.source_pads[0])

        assert result.EOS is False
        event_data = result.data[0].data[0]
        assert event_data["horizon"] == {"bank1": 100.0, "bank2": 150.0}
        assert event_data["ifo"] == "L1"

    def test_new_with_single_func_range_true(self):
        """Test new() with single func and range=True (lines 53-54, 57-61)."""

        def mock_horizon_func(psd, snr):
            return (225.0, None)  # Will be divided by 2.25 to get 100.0

        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs=mock_horizon_func,
            ifo="H1",
            range=True,
        )

        mock_psd = self._create_mock_psd()
        mock_frame = self._create_mock_frame(mock_psd, epoch=1000000000)
        tracker.preparedframes = {tracker.sink_pads[0]: mock_frame}

        result = tracker.new(tracker.source_pads[0])

        assert result.EOS is False
        event_data = result.data[0].data[0]
        assert "range_history" in event_data
        assert event_data["range_history"]["data"] == [100.0]
        assert event_data["range_history"]["time"] == [1000000000.0]

    def test_new_with_psd_none_range_false(self):
        """Test new() with psd=None and range=False (lines 72-79)."""
        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs=lambda psd, snr: (100.0, None),
            ifo="H1",
            range=False,
        )

        mock_frame = self._create_mock_frame(psd=None)
        tracker.preparedframes = {tracker.sink_pads[0]: mock_frame}

        result = tracker.new(tracker.source_pads[0])

        assert result.EOS is False
        event_data = result.data[0].data[0]
        assert event_data["horizon"] is None
        assert event_data["ifo"] == "H1"
        assert event_data["navg"] is None
        assert event_data["n_samples"] is None

    def test_new_with_psd_none_range_true(self):
        """Test new() with psd=None and range=True (lines 70-71)."""
        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs=lambda psd, snr: (100.0, None),
            ifo="H1",
            range=True,
        )

        mock_frame = self._create_mock_frame(psd=None)
        tracker.preparedframes = {tracker.sink_pads[0]: mock_frame}

        result = tracker.new(tracker.source_pads[0])

        assert result.EOS is False
        event_data = result.data[0].data[0]
        assert event_data is None

    def test_new_with_eos(self):
        """Test new() passes through EOS flag."""

        def mock_horizon_func(psd, snr):
            return (200.0, None)

        tracker = HorizonDistanceTracker(
            name="test_tracker",
            source_pad_names=("out",),
            sink_pad_names=("in",),
            horizon_distance_funcs=mock_horizon_func,
            ifo="H1",
        )

        mock_psd = self._create_mock_psd()
        mock_frame = self._create_mock_frame(mock_psd)
        mock_frame.EOS = True
        tracker.preparedframes = {tracker.sink_pads[0]: mock_frame}

        result = tracker.new(tracker.source_pads[0])

        assert result.EOS is True


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
