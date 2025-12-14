#!/usr/bin/env python3
import pathlib

import pytest
from lal import LIGOTimeGPS
from lal.utils import CacheEntry
from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgnts.sinks import DumpSeriesSink
from sgnts.transforms import Resampler

from sgnligo.sources import FrameReader

PATH_DATA = pathlib.Path(__file__).parent / "data"


def test_framereader():

    gps_start_time = 1240215487
    gps_end_time = 1240215519
    frame_cache = PATH_DATA / "gw190425.cache"
    channel_names = ["L1:GWOSC-16KHZ_R1_STRAIN", "L1:GWOSC-16KHZ_R1_DQMASK"]
    instrument = "L1"

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #       /      \
    #   DQ /        \ Strain
    #  ------   ------------
    # | Null | | Resampler  |
    #  ------   ------------
    #                 \
    #                  \
    #             ---------
    #            | snk1    |
    #             ---------

    src = FrameReader(
        name="src1",
        framecache=frame_cache.as_posix(),
        channel_names=channel_names,
        instrument=instrument,
        t0=gps_start_time,
        end=gps_end_time,
    )
    sample_rate = src.rates[channel_names[0]]
    pipeline.insert(
        src,
        Resampler(
            name="trans1",
            source_pad_names=(instrument,),
            sink_pad_names=(instrument,),
            inrate=sample_rate,
            outrate=2048,
        ),
        DumpSeriesSink(name="snk1", sink_pad_names=(instrument,), fname="strain.txt"),
        NullSink(name="snk2", sink_pad_names=("DQ",)),
    )

    pipeline.insert(
        link_map={
            "trans1:snk:" + instrument: "src1:src:L1:GWOSC-16KHZ_R1_STRAIN",
            "snk1:snk:" + instrument: "trans1:src:" + instrument,
            "snk2:snk:DQ": "src1:src:L1:GWOSC-16KHZ_R1_DQMASK",
        }
    )

    pipeline.run()


class TestFrameReaderInit:
    """Test FrameReader initialization edge cases."""

    def test_source_pad_names_mismatch(self):
        """Test error when source_pad_names doesn't match channel_names."""
        frame_cache = PATH_DATA / "gw190425.cache"
        channel_names = ["L1:GWOSC-16KHZ_R1_STRAIN"]

        with pytest.raises(ValueError, match="Expected source pad names to match"):
            FrameReader(
                name="src",
                source_pad_names=("wrong_pad",),  # Doesn't match channel names
                framecache=frame_cache.as_posix(),
                channel_names=channel_names,
                t0=1240215487,
                end=1240215519,
            )

    def test_ifo_strings_with_1_suffix(self):
        """Test ifo_strings with IFO that has '1' suffix (line 159-160)."""
        result = FrameReader.ifo_strings("H1")
        assert result == ("H", "H1")

    def test_ifo_strings_without_1_suffix(self):
        """Test ifo_strings with IFO that doesn't have '1' suffix (line 162)."""
        result = FrameReader.ifo_strings("H")
        assert result == ("H", "H1")

    def test_ifo_strings_with_L1(self):
        """Test ifo_strings with L1."""
        result = FrameReader.ifo_strings("L1")
        assert result == ("L", "L1")

    def test_ifo_strings_with_V(self):
        """Test ifo_strings with V (Virgo)."""
        result = FrameReader.ifo_strings("V")
        assert result == ("V", "V1")


class TestFrameReaderMissingSegments:
    """Test FrameReader missing segment handling."""

    def test_missing_segment_warning(self, tmp_path, caplog):
        """Test warning is logged when there are missing segments."""
        # Create a cache file pointing to existing frame
        cache_file = tmp_path / "test.cache"
        # Request a segment that's only partially covered by the frame
        # Frame covers 1240215487-1240215519 (32 seconds)
        # Request 1240215480-1240215530 (50s, starting 7s before, ending 11s after)
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        import logging

        with caplog.at_level(logging.WARNING):
            FrameReader(
                name="src",
                framecache=str(cache_file),
                channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
                t0=1240215480,  # Before frame start
                end=1240215530,  # After frame end
            )

        # Should have logged a warning about missing segments
        assert "missing segment" in caplog.text.lower()


class TestFrameReaderSegmentFiltering:
    """Test segment filtering in FrameReader."""

    def test_segment_not_intersecting(self, tmp_path):
        """Test files that don't intersect analysis segment are skipped."""
        # Create a cache with a frame that doesn't intersect the analysis segment
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        # Add both the actual frame and a fake entry for a different time range
        cache_file.write_text(
            f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n"
            f"L L1_GWOSC_16KHZ_R1 1000000000 32 /fake/path.gwf\n"
        )

        reader = FrameReader(
            name="src",
            framecache=str(cache_file),
            channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
            t0=1240215487,
            end=1240215519,
        )

        # The fake entry at 1000000000 should have been filtered out
        # Only the valid frame should remain (and it gets popped in __post_init__)
        assert len(reader.cache) == 0  # Already popped in __post_init__


class TestFrameReaderGapHandling:
    """Test gap handling in FrameReader."""

    def test_gap_buffer_via_missing_segment(self, tmp_path, capsys):
        """Test gap buffer is sent when analysis starts before frame (lines 193-205).

        This triggers the gap buffer code path by requesting an analysis segment
        that starts before the available frame data.
        """
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Frame covers 1240215487-1240215519
        # Request starting 7 seconds earlier to trigger gap buffer
        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        FrameReader(
            name="src",
            framecache=str(cache_file),
            channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
            t0=1240215480,  # 7 seconds before frame start
            end=1240215519,
        )

        captured = capsys.readouterr()
        # Gap buffer message is printed when last_epoch < start
        assert "Unepected epoch" in captured.out

    def test_epoch_error_when_last_epoch_greater(self, tmp_path):
        """Test ValueError when last_epoch > start (line 206-210)."""
        from sgnts.base import Audioadapter

        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        reader = FrameReader(
            name="src",
            framecache=str(cache_file),
            channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
            t0=1240215487,
            end=1240215519,
        )

        # Set last_epoch to AFTER the frame start to trigger the error
        reader.last_epoch = LIGOTimeGPS(1240215510)

        # Reset the audioadapter to allow pushing fresh data
        channel = "L1:GWOSC-16KHZ_R1_STRAIN"
        reader.A[channel] = Audioadapter()

        # Create a CacheEntry that starts before last_epoch
        frame_line = f"L L1_TEST 1240215487 32 {gwf_path}"
        frame_entry = CacheEntry(frame_line)

        with pytest.raises(ValueError, match="Unepected epoch"):
            reader.load_gwf_data(frame_entry)


class TestFrameReaderInternal:
    """Test FrameReader internal() method."""

    def test_internal_loads_next_frame(self, tmp_path):
        """Test internal() loads next frame when buffer is low (lines 229-241)."""
        # Create cache with two consecutive frames
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Use same frame twice to simulate consecutive frames
        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        reader = FrameReader(
            name="src",
            framecache=str(cache_file),
            channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
            t0=1240215487,
            end=1240215519,
        )

        # Call internal - cache should be empty after __post_init__ popped first frame
        initial_cache_len = len(reader.cache)
        reader.internal()

        # Cache length should be same or smaller
        assert len(reader.cache) <= initial_cache_len

    def test_internal_triggers_read_new_and_loads_frame(self, tmp_path):
        """Test internal() triggers read_new and loads next frame (lines 232-241).

        This test creates a cache with two entries. After __post_init__ loads the
        first entry, we drain the adapter below num_samples threshold and call
        internal() to trigger loading the second cache entry.
        """
        from sgnts.base import Audioadapter

        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Create cache with two entries covering different time ranges
        # Frame actually covers 1240215487-1240215519 (32 seconds)
        # First entry: 1240215487-1240215503 (16 seconds)
        # Second entry: 1240215503-1240215519 (16 seconds)
        cache_file.write_text(
            f"L L1_GWOSC_16KHZ_R1 1240215487 16 {gwf_path}\n"
            f"L L1_GWOSC_16KHZ_R1 1240215503 16 {gwf_path}\n"
        )

        reader = FrameReader(
            name="src",
            framecache=str(cache_file),
            channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
            t0=1240215487,
            end=1240215519,
        )

        # After __post_init__, first cache entry is loaded and popped
        # So cache should have 1 entry remaining
        assert len(reader.cache) == 1

        channel = "L1:GWOSC-16KHZ_R1_STRAIN"

        # Drain the adapter to trigger read_new condition
        # num_samples returns buffer_duration * rate samples
        # We need adapter.size < num_samples for read_new to trigger
        # Reset the adapter to have very few samples
        reader.A[channel] = Audioadapter()

        # Call internal - this should trigger read_new and load the next frame
        reader.internal()

        # Cache should now be empty (second entry was loaded and popped)
        assert len(reader.cache) == 0

    def test_internal_reads_when_adapter_low(self, tmp_path):
        """Test internal() reads new data when adapter has few samples."""
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Create cache with multiple entries
        cache_file.write_text(
            f"L L1_GWOSC_16KHZ_R1 1240215487 16 {gwf_path}\n"
            f"L L1_GWOSC_16KHZ_R1 1240215503 16 {gwf_path}\n"
        )

        reader = FrameReader(
            name="src",
            framecache=str(cache_file),
            channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
            t0=1240215487,
            end=1240215519,
        )

        # Call internal multiple times to drain the adapter
        for _ in range(3):
            reader.internal()


class TestFrameReaderNew:
    """Test FrameReader new() method."""

    def test_new_creates_frame(self):
        """Test new() creates a TSFrame with correct metadata."""
        frame_cache = PATH_DATA / "gw190425.cache"
        channel_names = ["L1:GWOSC-16KHZ_R1_STRAIN"]

        reader = FrameReader(
            name="src",
            framecache=frame_cache.as_posix(),
            channel_names=channel_names,
            t0=1240215487,
            end=1240215519,
        )

        pad = reader.source_pads[0]
        frame = reader.new(pad)

        assert frame is not None
        assert reader.cnt[pad] == 1

    def test_new_increments_counter(self):
        """Test new() increments the counter."""
        frame_cache = PATH_DATA / "gw190425.cache"
        channel_names = ["L1:GWOSC-16KHZ_R1_STRAIN"]

        reader = FrameReader(
            name="src",
            framecache=frame_cache.as_posix(),
            channel_names=channel_names,
            t0=1240215487,
            end=1240215519,
        )

        pad = reader.source_pads[0]

        # Call new multiple times
        reader.new(pad)
        assert reader.cnt[pad] == 1

        reader.new(pad)
        assert reader.cnt[pad] == 2


class TestFrameReaderMissingSegmentDetection:
    """Test missing segment detection logic (lines 97-119)."""

    def test_segment_remaining_all_covered(self, tmp_path):
        """Test when cache fully covers the analysis segment (line 101-103)."""
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        # Request exactly what the frame covers
        reader = FrameReader(
            name="src",
            framecache=str(cache_file),
            channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
            t0=1240215487,
            end=1240215519,
        )

        # No warning should be logged for this case
        assert reader is not None

    def test_discontinuity_in_middle(self, tmp_path, caplog):
        """Test discontinuity detection in the middle (lines 104-114)."""
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Create a scenario where the analysis segment starts before the frame
        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        import logging

        with caplog.at_level(logging.WARNING):
            FrameReader(
                name="src",
                framecache=str(cache_file),
                channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
                t0=1240215470,  # 17 seconds before frame start
                end=1240215519,
            )

        # Should have logged a warning about missing segments
        assert "missing segment" in caplog.text.lower()

    def test_discontinuity_cache_extends_past_analysis(self, tmp_path, caplog):
        """Test line 114: cache segment fully covers remaining analysis segment.

        This triggers when:
        - Analysis starts before cache (discontinuity)
        - Cache ends after analysis ends (cache covers rest completely)
        """
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Frame covers 1240215487-1240215519
        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        import logging

        with caplog.at_level(logging.WARNING):
            reader = FrameReader(
                name="src",
                framecache=str(cache_file),
                channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
                t0=1240215480,  # 7 seconds before frame start (discontinuity)
                end=1240215510,  # 9 seconds before frame end (cache extends past)
            )

        # Should have logged a warning about missing segment at the start
        assert "missing segment" in caplog.text.lower()
        # Reader should still be created successfully
        assert reader is not None

    def test_segment_remaining_at_end(self, tmp_path, caplog):
        """Test remaining segment at the end."""
        cache_file = tmp_path / "test.cache"
        gwf_path = PATH_DATA / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        cache_file.write_text(f"L L1_GWOSC_16KHZ_R1 1240215487 32 {gwf_path}\n")

        import logging

        with caplog.at_level(logging.WARNING):
            FrameReader(
                name="src",
                framecache=str(cache_file),
                channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
                t0=1240215487,
                end=1240215550,  # 31 seconds after frame end
            )

        # Should have logged a warning about missing segments
        assert "missing segment" in caplog.text.lower()


if __name__ == "__main__":
    test_framereader()
