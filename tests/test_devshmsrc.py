import queue
import shutil
import time
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch

import pytest
from gwpy.timeseries import TimeSeriesDict
from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgnts.transforms import Gate

from sgnligo.sources import DevShmSource
from sgnligo.sources.devshmsrc import _FrameFileEventHandler
from sgnligo.transforms import BitMask


@pytest.mark.freeze_time("2019-04-25 08:17:49", tick=True)
def test_devshmsrc(tmp_path):
    pipeline = Pipeline()

    #
    #       -----------
    #      | DevShmSource |
    #       -----------
    #  state |       |
    #  vector|       |
    #  ---------     | strain
    # | BitMask |    |
    #  ---------     |
    #        \       |
    #         \      |
    #       ------------
    #      |   Gate     |
    #       ------------
    #             |
    #             |
    #       ------------
    #      |   NullSink |
    #       ------------

    datadir = Path(__file__).parent / "data"
    test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

    # add frame to "shmdir" to determine sampling rates
    dest_path = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215465-32.gwf"
    shutil.copyfile(datadir / test_frame, dest_path)

    # create pipeline
    hoft = "L1:GWOSC-16KHZ_R1_STRAIN"
    dqmask = "L1:GWOSC-16KHZ_R1_DQMASK"

    src = DevShmSource(
        name="src",
        channel_names=[hoft, dqmask],
        shared_memory_dirs=str(tmp_path),
        duration=32,
    )
    mask = BitMask(
        name="mask",
        sink_pad_names=(dqmask,),
        source_pad_names=("dqmask",),
        bit_mask=0,
    )
    gate = Gate(
        name="gate",
        source_pad_names=("strain",),
        sink_pad_names=(hoft, "dqmask"),
        control="dqmask",
    )
    sink = NullSink(
        name="sink",
        sink_pad_names=("strain",),
    )

    pipeline.insert(
        src,
        mask,
        gate,
        sink,
        link_map={
            mask.snks[dqmask]: src.srcs[dqmask],
            gate.snks[hoft]: src.srcs[hoft],
            gate.snks["dqmask"]: mask.srcs["dqmask"],
            sink.snks["strain"]: gate.srcs["strain"],
        },
    )

    # add frame to simulate live data after a short period of time
    def populate_frame():
        time.sleep(3)
        shutil.copyfile(datadir / test_frame, tmp_path / test_frame)

    thread = Thread(target=populate_frame)
    thread.start()

    # start pipeline
    pipeline.run()
    thread.join()


class TestFrameFileEventHandler:
    """Test _FrameFileEventHandler class (lines 24-51)."""

    def test_on_closed_directory(self):
        """Test on_closed skips directories (lines 34-36)."""
        q = queue.Queue()
        handler = _FrameFileEventHandler(q, ".gwf", 1000)

        # Create mock event for a directory
        event = Mock()
        event.is_directory = True
        event.src_path = "/dev/shm/test"  # noqa: S108

        handler.on_closed(event)

        # Queue should be empty since directories are skipped
        assert q.empty()

    def test_on_closed_file(self, tmp_path):
        """Test on_closed handles files (lines 34-36)."""
        q = queue.Queue()
        handler = _FrameFileEventHandler(q, ".gwf", 1000)

        # Create a real temp file with T050017 compliant name
        test_file = tmp_path / "H-H1_TEST-1500-32.gwf"
        test_file.touch()

        event = Mock()
        event.is_directory = False
        event.src_path = str(test_file)

        handler.on_closed(event)

        # File should be in queue since t0 (1500) >= watermark (1000)
        assert not q.empty()
        path, t0 = q.get()
        assert path == str(test_file)
        assert t0 == 1500

    def test_on_moved_directory(self):
        """Test on_moved skips directories (lines 39-41)."""
        q = queue.Queue()
        handler = _FrameFileEventHandler(q, ".gwf", 1000)

        event = Mock()
        event.is_directory = True
        event.dest_path = "/dev/shm/test"  # noqa: S108

        handler.on_moved(event)

        assert q.empty()

    def test_on_moved_file(self, tmp_path):
        """Test on_moved handles files (lines 39-41)."""
        q = queue.Queue()
        handler = _FrameFileEventHandler(q, ".gwf", 1000)

        test_file = tmp_path / "L-L1_TEST-2000-32.gwf"
        test_file.touch()

        event = Mock()
        event.is_directory = False
        event.dest_path = str(test_file)

        handler.on_moved(event)

        assert not q.empty()
        path, t0 = q.get()
        assert t0 == 2000

    def test_handle_event_wrong_extension(self, tmp_path):
        """Test _handle_event skips files with wrong extension (lines 44-46)."""
        q = queue.Queue()
        handler = _FrameFileEventHandler(q, ".gwf", 1000)

        # File with wrong extension
        test_file = tmp_path / "H-H1_TEST-1500-32.txt"
        test_file.touch()

        handler._handle_event(str(test_file))

        assert q.empty()

    def test_handle_event_before_watermark(self, tmp_path):
        """Test _handle_event skips files before watermark (lines 48-50)."""
        q = queue.Queue()
        # Set watermark to 2000, file is at 1500 (before watermark)
        handler = _FrameFileEventHandler(q, ".gwf", 2000)

        test_file = tmp_path / "H-H1_TEST-1500-32.gwf"
        test_file.touch()

        handler._handle_event(str(test_file))

        # File should be skipped since t0 (1500) < watermark (2000)
        assert q.empty()

    def test_handle_event_at_watermark(self, tmp_path):
        """Test _handle_event accepts files at watermark (line 51)."""
        q = queue.Queue()
        handler = _FrameFileEventHandler(q, ".gwf", 1500)

        test_file = tmp_path / "H-H1_TEST-1500-32.gwf"
        test_file.touch()

        handler._handle_event(str(test_file))

        # File should be added since t0 (1500) >= watermark (1500)
        assert not q.empty()


class TestDevShmSourceInit:
    """Test DevShmSource initialization edge cases."""

    def test_source_pad_names_mismatch(self, tmp_path):
        """Test error when source_pad_names doesn't match channel_names."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        dest_path = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215465-32.gwf"
        shutil.copyfile(datadir / test_frame, dest_path)

        hoft = "L1:GWOSC-16KHZ_R1_STRAIN"

        with pytest.raises(ValueError, match="Expected source pad names to match"):
            DevShmSource(
                name="src",
                source_pad_names=("wrong_pad",),  # Doesn't match channel names
                channel_names=[hoft],
                shared_memory_dirs=str(tmp_path),
                duration=32,
            )

    @pytest.mark.freeze_time("2019-04-25 08:17:49", tick=True)
    def test_verbose_init(self, tmp_path, capsys):
        """Test verbose output in __post_init__ (line 127)."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        dest_path = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215465-32.gwf"
        shutil.copyfile(datadir / test_frame, dest_path)

        hoft = "L1:GWOSC-16KHZ_R1_STRAIN"

        src = DevShmSource(
            name="src",
            channel_names=[hoft],
            shared_memory_dirs=str(tmp_path),
            duration=32,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Start up t0:" in captured.err

        # Clean up observers
        for observer in src.observer.values():
            observer.unschedule_all()
            observer.stop()
            observer.join()


class TestDevShmSourceInternal:
    """Test DevShmSource internal() method edge cases."""

    @pytest.fixture
    def mock_source(self, tmp_path):
        """Create a DevShmSource with mocked components."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        dest_path = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215465-32.gwf"
        shutil.copyfile(datadir / test_frame, dest_path)

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215465

            src = DevShmSource(
                name="src",
                channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
                shared_memory_dirs=str(tmp_path),
                duration=32,
            )

        yield src

        # Clean up
        for observer in src.observer.values():
            observer.unschedule_all()
            observer.stop()
            observer.join()

    def test_old_data_continuous(self, mock_source, capsys):
        """Test old data handling with continuous data (lines 201-210)."""
        ifo = "L1"
        channel = "L1:GWOSC-16KHZ_R1_STRAIN"

        # Set up old data scenario - reset_start must be False to preserve times
        mock_source.reset_start = False
        mock_source.data_dict[ifo][channel] = Mock()  # Non-None = old data exists
        # file_t0 should match next_buffer_end (which becomes next_buffer_t0)
        mock_source.file_t0[ifo] = 1240215465
        mock_source.next_buffer_end[ifo] = 1240215465  # This becomes next_buffer_t0

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215470
            mock_source.internal()

        captured = capsys.readouterr()
        assert "old data cont." in captured.err

    def test_old_data_discontinuous(self, mock_source, capsys):
        """Test old data handling with discontinuous data (lines 211-221)."""
        ifo = "L1"
        channel = "L1:GWOSC-16KHZ_R1_STRAIN"

        # Set up discontinuous data scenario - file_t0 > next_buffer_t0
        mock_source.reset_start = False
        mock_source.data_dict[ifo][channel] = Mock()  # Non-None = old data exists
        mock_source.file_t0[ifo] = 1240215500  # Later than next_buffer_end
        mock_source.next_buffer_end[ifo] = 1240215465  # This becomes next_buffer_t0

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215470
            mock_source.internal()

        captured = capsys.readouterr()
        assert "old data discont." in captured.err
        assert mock_source.discont[ifo] is True
        assert mock_source.send_gap[ifo] is True

    def test_old_data_wrong_t0(self, mock_source):
        """Test old data handling with file_t0 < next_buffer_t0 (line 221)."""
        ifo = "L1"
        channel = "L1:GWOSC-16KHZ_R1_STRAIN"

        # Set up scenario where file_t0 < next_buffer_t0 (wrong t0)
        mock_source.reset_start = False
        mock_source.data_dict[ifo][channel] = Mock()  # Non-None = old data exists
        mock_source.file_t0[ifo] = 1240215400  # Earlier than next_buffer_end
        mock_source.next_buffer_end[ifo] = 1240215465  # This becomes next_buffer_t0

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215470
            with pytest.raises(ValueError, match="wrong t0"):
                mock_source.internal()

    def test_skip_reading_when_old_data_exists(self, mock_source, capsys):
        """Test skip reading new file when old data exists (lines 238-243)."""
        ifo = "L1"
        channel = "L1:GWOSC-16KHZ_R1_STRAIN"

        # Set up old data - file_t0 matches next_buffer_end (becomes next_buffer_t0)
        mock_source.reset_start = False
        mock_source.data_dict[ifo][channel] = Mock()
        mock_source.file_t0[ifo] = 1240215465
        mock_source.next_buffer_end[ifo] = 1240215465

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215470
            mock_source.internal()

        captured = capsys.readouterr()
        assert "skip reading new file" in captured.err

    def test_file_does_not_exist(self, mock_source, tmp_path, capsys):
        """Test handling when file doesn't exist anymore (lines 263-269)."""
        ifo = "L1"

        # Put a non-existent file in the queue
        mock_source.reset_start = False
        mock_source.queues[ifo].put(("/nonexistent/file.gwf", 1240215465))
        mock_source.next_buffer_end[ifo] = 1240215465

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215470
            mock_source.internal()

        captured = capsys.readouterr()
        assert "File does not exist anymore" in captured.err

    def test_verbose_queue_output(self, mock_source, tmp_path, capsys):
        """Test verbose output during queue processing (lines 270-272)."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        file_path = tmp_path / test_frame
        shutil.copyfile(datadir / test_frame, file_path)

        ifo = "L1"
        mock_source.reset_start = False
        mock_source.verbose = True

        # Put file in queue with matching t0
        mock_source.queues[ifo].put((str(file_path), 1240215487))
        mock_source.next_buffer_end[ifo] = 1240215487

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215490
            mock_source.internal()

        captured = capsys.readouterr()
        assert str(file_path) in captured.err

    def test_queue_t0_less_than_expected(self, mock_source, tmp_path):
        """Test skipping files with t0 < next_buffer_t0 (lines 273-274)."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        file_path = tmp_path / test_frame
        shutil.copyfile(datadir / test_frame, file_path)

        ifo = "L1"
        mock_source.reset_start = False
        # Put old file first, then correct file
        mock_source.queues[ifo].put((str(file_path), 1240215400))  # Old file
        mock_source.queues[ifo].put((str(file_path), 1240215487))  # Correct file
        mock_source.next_buffer_end[ifo] = 1240215487

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215490
            mock_source.internal()

        # Should have processed the second file, skipping the first
        assert mock_source.file_t0[ifo] == 1240215487

    def test_queue_discontinuity(self, mock_source, tmp_path, capsys):
        """Test discontinuity when t0 > next_buffer_t0 (lines 282-284, 310-318)."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
        file_path = tmp_path / test_frame
        shutil.copyfile(datadir / test_frame, file_path)

        ifo = "L1"
        mock_source.reset_start = False
        # File t0 is later than expected
        mock_source.queues[ifo].put((str(file_path), 1240215487))
        mock_source.next_buffer_end[ifo] = 1240215455  # Earlier than file t0

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215490
            mock_source.internal()

        captured = capsys.readouterr()
        assert "discont t0" in captured.err
        assert mock_source.discont[ifo] is True
        assert mock_source.send_gap[ifo] is True

    def test_file_read_error(self, mock_source, tmp_path, capsys):
        """Test handling RuntimeError when reading file (lines 329-336)."""
        ifo = "L1"
        mock_source.reset_start = False
        # Create a corrupt/invalid gwf file
        corrupt_file = tmp_path / "L-L1_CORRUPT-1240215487-32.gwf"
        corrupt_file.write_bytes(b"not a valid gwf file")

        mock_source.queues[ifo].put((str(corrupt_file), 1240215487))
        mock_source.next_buffer_end[ifo] = 1240215487

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215490
            mock_source.internal()

        captured = capsys.readouterr()
        assert "Could not read file" in captured.err
        assert mock_source.send_gap[ifo] is True


class TestDevShmSourceNew:
    """Test DevShmSource new() method."""

    @pytest.fixture
    def source_with_data(self, tmp_path):
        """Create DevShmSource with actual data loaded."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Copy frame for initialization
        init_frame = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215455-32.gwf"
        shutil.copyfile(datadir / test_frame, init_frame)

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215487

            src = DevShmSource(
                name="src",
                channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
                shared_memory_dirs=str(tmp_path),
                duration=32,
                verbose=True,
            )

        yield src, datadir, test_frame

        # Clean up
        for observer in src.observer.values():
            observer.unschedule_all()
            observer.stop()
            observer.join()

    def test_new_gap_verbose(self, source_with_data, capsys):
        """Test new() gap buffer with verbose output (line 357)."""
        src, datadir, test_frame = source_with_data

        ifo = "L1"
        pad = src.source_pads[0]

        # Set up for gap buffer
        src.send_gap[ifo] = True
        src.send_gap_duration[ifo] = 32
        src.next_buffer_t0[ifo] = 1240215487

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215490
            frame = src.new(pad)

        captured = capsys.readouterr()
        assert "Queue is empty, sending a gap buffer" in captured.err
        assert frame.buffers[0].is_gap

    def test_new_data_buffer(self, source_with_data, tmp_path, capsys):
        """Test new() with actual data buffer (lines 374-408)."""
        src, datadir, test_frame = source_with_data

        ifo = "L1"
        channel = "L1:GWOSC-16KHZ_R1_STRAIN"
        pad = src.source_pads[0]

        # Load actual data into data_dict
        file_path = datadir / test_frame
        tsd = TimeSeriesDict.read(str(file_path), [channel])
        src.data_dict[ifo][channel] = tsd[channel]

        # Set up for data buffer (not gap)
        src.send_gap[ifo] = False
        src.next_buffer_t0[ifo] = 1240215487  # Match file t0

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215520
            frame = src.new(pad)

        captured = capsys.readouterr()
        assert "Buffer t0:" in captured.err
        assert not frame.buffers[0].is_gap
        assert frame.buffers[0].data is not None

        # Verify data_dict was cleared for this channel
        assert src.data_dict[ifo][channel] is None


class TestDevShmSourceSendGapSync:
    """Test send_gap_sync branch (lines 248-253, 276-279)."""

    def test_send_gap_sync_direct(self, tmp_path, capsys):
        """Test send_gap_sync sets up correctly when IFO catches up."""
        datadir = Path(__file__).parent / "data"
        test_frame = "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"

        # Set up for single IFO first
        init_frame = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215455-32.gwf"
        shutil.copyfile(datadir / test_frame, init_frame)

        with patch("sgnligo.sources.devshmsrc.now") as mock_now:
            mock_now.return_value = 1240215487

            src = DevShmSource(
                name="src",
                channel_names=["L1:GWOSC-16KHZ_R1_STRAIN"],
                shared_memory_dirs=str(tmp_path),
                duration=32,
                verbose=True,
            )

        try:
            # Manually add another IFO to simulate multi-IFO setup
            src.ifos = {"L1", "H1"}
            src.queues["H1"] = queue.Queue()
            src.next_buffer_t0["H1"] = 1240215519  # H1 is ahead
            src.next_buffer_end["H1"] = 1240215519
            src.data_dict["H1"] = {"H1:FAKE": None}
            src.discont["H1"] = False
            src.send_gap["H1"] = False
            src.send_gap_duration["H1"] = 0

            # L1 is behind
            src.reset_start = False
            src.next_buffer_end["L1"] = 1240215487

            # Put a file in L1 queue that matches expected t0
            # When L1 catches up with valid data, H1 should get send_gap_sync
            l1_frame = tmp_path / "L-L1_GWOSC_16KHZ_R1-1240215487-32.gwf"
            shutil.copyfile(datadir / test_frame, l1_frame)
            src.queues["L1"].put((str(l1_frame), 1240215487))

            with patch("sgnligo.sources.devshmsrc.now") as mock_now:
                mock_now.return_value = 1240215520
                src.internal()

            captured = capsys.readouterr()
            # H1 should have send_gap_sync set
            assert src.send_gap["H1"] is True
            assert "send_gap_sync" in captured.err
        finally:
            for observer in src.observer.values():
                observer.unschedule_all()
                observer.stop()
                observer.join()
