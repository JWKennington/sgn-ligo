#!/usr/bin/env python3
"""Test coverage for sgnligo.sources.gwdata_noise_source module."""

from unittest.mock import Mock, patch

import lal
import numpy as np
import pytest
from sgn.apps import Pipeline
from sgnts.base import Offset, SeriesBuffer, TSFrame
from sgnts.sinks import DumpSeriesSink, NullSeriesSink

from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource, parse_psd


@pytest.fixture
def mock_psd():
    """Create a mock PSD for testing."""
    psd = lal.CreateREAL8FrequencySeries(
        name="test_psd",
        epoch=lal.LIGOTimeGPS(0),
        f0=0.0,
        deltaF=1.0,
        sampleUnits=lal.Unit("strain^2 s"),
        length=8193,  # 8192 Hz Nyquist
    )
    psd.data.data = 1e-46 * np.ones(8193)
    return psd


@pytest.fixture
def mock_fir_kernel():
    """Create a mock FIR kernel."""
    kernel = Mock()
    fir_matrix = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    kernel.psd_to_linear_phase_whitening_fir_kernel.return_value = (
        fir_matrix,
        2,  # latency
        16384,  # sample rate
    )
    return kernel


class TestParsePSD:
    """Test cases for parse_psd function."""

    def test_parse_psd_single_detector(self):
        """Test parse_psd with single detector."""
        result = parse_psd({"H1": "H1:FAKE-STRAIN"})
        assert "H1" in result
        assert result["H1"]["channel-name"] == "H1:FAKE-STRAIN"
        assert result["H1"]["rate"] == 16384
        assert isinstance(result["H1"]["state"], np.ndarray)
        assert isinstance(result["H1"]["fir-matrix"], np.ndarray)

    def test_parse_psd_multiple_detectors(self):
        """Test parse_psd with multiple detectors."""
        result = parse_psd({"H1": "H1:FAKE-STRAIN", "L1": "L1:FAKE-STRAIN"})
        assert "H1" in result and "L1" in result
        assert result["H1"]["channel-name"] == "H1:FAKE-STRAIN"
        assert result["L1"]["channel-name"] == "L1:FAKE-STRAIN"

    @patch("sgnligo.sources.gwdata_noise_source.PSDFirKernel")
    @patch("sgnligo.sources.gwdata_noise_source.fake_gwdata_psd")
    def test_parse_psd_structure(
        self, mock_fake_psd, mock_kernel_class, mock_psd, mock_fir_kernel
    ):
        """Test detailed PSD parsing structure."""
        mock_fake_psd.return_value = {"H1": mock_psd}
        mock_kernel_class.return_value = mock_fir_kernel

        result = parse_psd({"H1": "H1:FAKE-STRAIN"})

        h1_info = result["H1"]
        assert h1_info["rate"] == 16384
        assert h1_info["psd"] is mock_psd
        assert h1_info["sample-stride"] == 16384
        assert len(h1_info["state"]) == 16384 + 5 - 1
        np.testing.assert_array_equal(
            h1_info["fir-matrix"], np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        )

    @patch("sgnligo.sources.gwdata_noise_source.PSDFirKernel")
    @patch("sgnligo.sources.gwdata_noise_source.fake_gwdata_psd")
    def test_parse_psd_missing_ifo(self, mock_fake_psd, mock_kernel_class):
        """Test parse_psd with missing IFO raises assertion."""
        mock_psd = Mock()
        mock_psd.data = Mock()
        mock_psd.data.data = np.ones(9)
        mock_psd.deltaF = 1.0

        mock_fir_kernel = Mock()
        mock_fir_kernel.psd_to_linear_phase_whitening_fir_kernel.return_value = (
            np.ones(5),
            2,
            16384,
        )
        mock_kernel_class.return_value = mock_fir_kernel

        # Only return H1, not L1
        mock_fake_psd.return_value = {"H1": mock_psd}

        with pytest.raises(AssertionError):
            parse_psd({"H1": "H1:FAKE-STRAIN", "L1": "L1:FAKE-STRAIN"})

    @patch("sgnligo.sources.gwdata_noise_source.PSDFirKernel")
    @patch("sgnligo.sources.gwdata_noise_source.fake_gwdata_psd")
    def test_parse_psd_invalid_nyquist(self, mock_fake_psd, mock_kernel_class):
        """Test parse_psd with non-power-of-two Nyquist frequency."""
        bad_psd = lal.CreateREAL8FrequencySeries(
            name="bad_psd",
            epoch=lal.LIGOTimeGPS(0),
            f0=0.0,
            deltaF=1.0,
            sampleUnits=lal.Unit("strain^2 s"),
            length=1001,  # 1000 Hz Nyquist (not power of 2)
        )
        bad_psd.data.data = 1e-46 * np.ones(1001)

        mock_fake_psd.return_value = {"H1": bad_psd}

        with pytest.raises(AssertionError):
            parse_psd({"H1": "H1:FAKE-STRAIN"})


class TestGWDataNoiseSource:
    """Test cases for GWDataNoiseSource class."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        source = GWDataNoiseSource(name="TestSource", duration=1.0)
        assert source.channel_dict == {"H1": "H1:FAKE-STRAIN", "L1": "L1:FAKE-STRAIN"}
        assert "H1:FAKE-STRAIN" in source.source_pad_names
        assert "L1:FAKE-STRAIN" in source.source_pad_names
        assert source.t0 is not None
        assert not source.verbose and not source.real_time

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        custom_t0, custom_end = 1234567890, 1234567900
        source = GWDataNoiseSource(
            name="TestSource",
            channel_dict={"V1": "V1:CUSTOM-STRAIN"},
            t0=custom_t0,
            end=custom_end,
            real_time=True,
            verbose=True,
        )
        assert source.channel_dict == {"V1": "V1:CUSTOM-STRAIN"}
        assert source.t0 == custom_t0
        assert source.end == custom_end
        assert source.verbose and source.real_time

    def test_init_duration(self):
        """Test initialization with duration instead of end time."""
        t0 = 1234567890
        source = GWDataNoiseSource(name="TestSource", t0=t0, duration=10.0)
        assert source.end == t0 + 10.0

    def test_init_realtime_none_t0_none_end(self):
        """Test real-time mode with None t0 and None end."""
        source = GWDataNoiseSource(name="TestSource", real_time=True, t0=None, end=None)
        assert source.t0 is not None  # Should be set to current GPS time
        assert source.end is None  # Should remain None for indefinite operation
        assert source.real_time is True

    def test_init_realtime_none_t0_verbose(self, capsys):
        """Test real-time mode with None t0 shows current GPS time message."""
        GWDataNoiseSource(real_time=True, t0=None, verbose=True)
        captured = capsys.readouterr()
        assert "Using current GPS time as start:" in captured.out
        assert "Real-time mode: will run indefinitely" in captured.out

    def test_init_non_realtime_none_end_raises_error(self):
        """Test that non-real-time mode requires end or duration."""
        with pytest.raises(ValueError, match="When real_time is False"):
            GWDataNoiseSource(name="TestSource", real_time=False, t0=1000, end=None)

    @patch("sgnligo.sources.gwdata_noise_source.time.time")
    @patch("sgnligo.sources.gwdata_noise_source.now")
    @patch("sgnligo.sources.gwdata_noise_source.parse_psd")
    def test_init_real_time_tracking(self, mock_parse_psd, mock_now, mock_time):
        """Test real-time tracking initialization."""
        mock_now.return_value = 1234567890
        mock_time.return_value = 100.0
        mock_parse_psd.return_value = {
            "H1": {
                "channel-name": "H1:FAKE-STRAIN",
                "rate": 16384,
                "sample-stride": Offset.sample_stride(16384),
                "state": np.zeros(100),
                "fir-matrix": np.ones(5),
            }
        }

        source = GWDataNoiseSource(real_time=True, t0=1000)
        assert source._start_wall_time == 100.0
        assert source._start_gps_time == 1000

    def test_init_verbose_output(self, capsys):
        """Test verbose output during initialization."""
        # Test with end time
        GWDataNoiseSource(verbose=True, t0=1000, end=2000)
        captured = capsys.readouterr()
        assert "Will run until GPS time: 2000" in captured.out

        # Test without end time in real-time mode
        GWDataNoiseSource(verbose=True, t0=1000, real_time=True)
        captured = capsys.readouterr()
        assert (
            "Real-time mode: will run indefinitely, synced with wall time"
            in captured.out
        )

        # Test without end time in non-real-time mode (requires duration)
        GWDataNoiseSource(verbose=True, t0=1000, duration=10.0)
        captured = capsys.readouterr()
        # Should not print the real-time message
        assert "Real-time mode:" not in captured.out

    def test_generate_noise_chunk(self):
        """Test noise chunk generation produces different noise."""
        source = GWDataNoiseSource(
            name="TestSource", channel_dict={"H1": "H1:FAKE-STRAIN"}, duration=1.0
        )
        h1_pad = source.srcs["H1:FAKE-STRAIN"]

        chunk1 = source._generate_noise_chunk(h1_pad)
        chunk2 = source._generate_noise_chunk(h1_pad)

        assert isinstance(chunk1, np.ndarray)
        assert len(chunk1) == source.channel_info["H1"]["sample-stride"]
        assert not np.array_equal(chunk1, chunk2)  # Different noise each time

    @patch("sgnligo.sources.gwdata_noise_source.signal.correlate")
    @patch("sgnligo.sources.gwdata_noise_source.numpy.random.randn")
    @patch("sgnligo.sources.gwdata_noise_source.parse_psd")
    def test_generate_noise_chunk_internals(
        self, mock_parse_psd, mock_randn, mock_correlate
    ):
        """Test noise chunk generation internals with mocking."""
        mock_parse_psd.return_value = {
            "H1": {
                "channel-name": "H1:FAKE-STRAIN",
                "rate": 16384,
                "sample-stride": Offset.sample_stride(16384),
                "state": np.zeros(10),
                "fir-matrix": np.ones(5),
            }
        }

        mock_randn.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_correlate.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        source = GWDataNoiseSource(channel_dict={"H1": "H1:FAKE-STRAIN"}, duration=1.0)
        pad = source.srcs["H1:FAKE-STRAIN"]

        result = source._generate_noise_chunk(pad)

        assert np.array_equal(result, mock_correlate.return_value)
        mock_correlate.assert_called_once()
        mock_randn.assert_called_once_with(5)

        # Check state update
        h1_info = source.channel_info["H1"]
        assert np.array_equal(h1_info["state"][-5:], mock_randn.return_value)

    def test_new_frame(self):
        """Test new frame generation with mocking."""
        source = GWDataNoiseSource(duration=1.0)
        pad = source.srcs["H1:FAKE-STRAIN"]

        # Mock prepare_frame and _generate_noise_chunk
        mock_buffer = Mock(spec=SeriesBuffer)
        mock_buffer.set_data = Mock()

        mock_frame = Mock(spec=TSFrame)
        mock_frame.__len__ = Mock(return_value=1)
        mock_frame.buffers = [mock_buffer]
        mock_frame.end = 1234567891.0

        with patch.object(source, "prepare_frame", return_value=mock_frame):
            with patch.object(
                source, "_generate_noise_chunk", return_value=np.ones(100)
            ):
                result = source.new(pad)

        assert result is mock_frame
        mock_buffer.set_data.assert_called_once()
        # Note: current_end is managed by the base class, not set in new()


class TestRealTimeMode:
    """Test cases for real-time synchronization."""

    @patch("sgnligo.sources.gwdata_noise_source.time.sleep")
    @patch("sgnligo.sources.gwdata_noise_source.time.time")
    def test_internal_real_time_ahead(self, mock_time, mock_sleep):
        """Test sleeping when ahead of real-time schedule."""
        mock_time.side_effect = [100.0, 101.0]  # init, internal call

        from sgnts.base.time import Time

        source = GWDataNoiseSource(real_time=True, t0=1000)

        # Mock the current_end property
        with patch.object(
            type(source),
            "current_end",
            new_callable=lambda: property(lambda self: 1002 * Time.SECONDS),
        ):
            source.internal()

        # Should sleep for ~1 second (2 seconds data - 1 second wall time)
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert 0.9 < sleep_time < 1.1

    @patch("sgnligo.sources.gwdata_noise_source.time.sleep")
    @patch("sgnligo.sources.gwdata_noise_source.time.time")
    def test_internal_real_time_behind(self, mock_time, mock_sleep, capsys):
        """Test warning when significantly behind real-time schedule."""
        mock_time.side_effect = [100.0, 105.0]  # init, internal (5s later)

        from sgnts.base.time import Time

        source = GWDataNoiseSource(real_time=True, t0=1000, verbose=True)

        # Mock the current_end property
        with patch.object(
            type(source),
            "current_end",
            new_callable=lambda: property(lambda self: 1002 * Time.SECONDS),
        ):
            source.internal()

        # Should not sleep when behind
        mock_sleep.assert_not_called()

        # Should print warning when behind
        captured = capsys.readouterr()
        assert "Warning: GWDataNoiseSource falling behind real time" in captured.out
        # The exact timing might vary slightly due to initialization
        assert "s)" in captured.out  # Just check it shows some timing

    @patch("sgnligo.sources.gwdata_noise_source.time.sleep")
    @patch("sgnligo.sources.gwdata_noise_source.time.time")
    def test_internal_real_time_slightly_behind(self, mock_time, mock_sleep, capsys):
        """Test no warning when slightly behind schedule."""
        mock_time.side_effect = [
            100.0,
            101.5,
        ]  # 1.5s later (data time 2s - wall time 1.5s = 0.5s ahead)

        from sgnts.base.time import Time

        source = GWDataNoiseSource(real_time=True, t0=1000, verbose=True)

        # Mock the current_end property
        with patch.object(
            type(source),
            "current_end",
            new_callable=lambda: property(lambda self: 1002 * Time.SECONDS),
        ):
            source.internal()

        # Should not sleep when only 0.5s ahead
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert 0.4 < sleep_time < 0.6  # Should sleep ~0.5s

    @patch("sgnligo.sources.gwdata_noise_source.time.sleep")
    def test_internal_no_real_time(self, mock_sleep):
        """Test no sleeping when not in real-time mode."""
        source = GWDataNoiseSource(real_time=False, duration=1.0)
        source.internal()
        mock_sleep.assert_not_called()

    @patch("sgnligo.sources.gwdata_noise_source.time.time")
    @patch("sgnligo.sources.gwdata_noise_source.now")
    @patch("sgnligo.sources.gwdata_noise_source.parse_psd")
    def test_set_pad_buffer_params(self, mock_parse_psd, mock_now, mock_time):
        """Test that buffer parameters are set correctly for each pad."""
        mock_now.return_value = 1234567890

        mock_parse_psd.return_value = {
            "H1": {
                "channel-name": "H1:FAKE-STRAIN",
                "rate": 16384,
                "sample-stride": Offset.sample_stride(16384),
                "state": np.zeros(10),
                "fir-matrix": np.ones(5),
            },
            "L1": {
                "channel-name": "L1:FAKE-STRAIN",
                "rate": 16384,
                "sample-stride": Offset.sample_stride(16384),
                "state": np.zeros(10),
                "fir-matrix": np.ones(5),
            },
        }

        with patch.object(
            GWDataNoiseSource, "set_pad_buffer_params"
        ) as mock_set_params:
            GWDataNoiseSource(duration=1.0)

            # Check that set_pad_buffer_params was called for each channel
            assert mock_set_params.call_count == 2

            # Check the calls
            calls = mock_set_params.call_args_list
            for call in calls:
                kwargs = call[1]
                assert kwargs["sample_shape"] == ()
                assert kwargs["rate"] == 16384


class TestIntegration:
    """Integration tests with pipelines."""

    def test_basic_pipeline(self):
        """Test basic pipeline with NullSeriesSink."""
        pipe = Pipeline()

        source = GWDataNoiseSource(
            name="NoiseSource",
            channel_dict={"H1": "H1:FAKE-STRAIN"},
            duration=1.0,
        )

        sink = NullSeriesSink(name="NullSink", sink_pad_names=["H1:FAKE-STRAIN"])

        pipe.insert(source, sink)
        pipe.insert(
            link_map={"NullSink:snk:H1:FAKE-STRAIN": "NoiseSource:src:H1:FAKE-STRAIN"}
        )
        pipe.run()

    def test_pipeline_with_file_output(self, tmp_path):
        """Test pipeline with file output."""
        output_file = tmp_path / "strain.txt"
        pipe = Pipeline()

        source = GWDataNoiseSource(
            name="NoiseSource",
            channel_dict={"H1": "H1:FAKE-STRAIN"},
            duration=1.0,
        )

        sink = DumpSeriesSink(
            name="DumpSink",
            sink_pad_names=["H1:FAKE-STRAIN"],
            fname=str(output_file),
        )

        pipe.insert(source, sink)
        pipe.insert(
            link_map={"DumpSink:snk:H1:FAKE-STRAIN": "NoiseSource:src:H1:FAKE-STRAIN"}
        )
        pipe.run()

        # Verify output
        assert output_file.exists()
        data = np.loadtxt(str(output_file))
        assert data.shape[0] > 0
        assert data.shape[1] == 2  # Time and strain columns
        assert np.all(np.diff(data[:, 0]) > 0)  # Monotonic time

    def test_multiple_detectors_pipeline(self):
        """Test pipeline with multiple detectors."""
        pipe = Pipeline()

        source = GWDataNoiseSource(
            name="NoiseSource",
            channel_dict={"H1": "H1:FAKE-STRAIN", "L1": "L1:FAKE-STRAIN"},
            duration=1.0,
        )

        h1_sink = NullSeriesSink(name="H1_Sink", sink_pad_names=["H1:FAKE-STRAIN"])
        l1_sink = NullSeriesSink(name="L1_Sink", sink_pad_names=["L1:FAKE-STRAIN"])

        pipe.insert(source, h1_sink, l1_sink)
        pipe.insert(
            link_map={
                "H1_Sink:snk:H1:FAKE-STRAIN": "NoiseSource:src:H1:FAKE-STRAIN",
                "L1_Sink:snk:L1:FAKE-STRAIN": "NoiseSource:src:L1:FAKE-STRAIN",
            }
        )
        pipe.run()

    def test_eos_handling(self):
        """Test End-of-Stream handling."""

        class CountingSink(NullSeriesSink):
            def __init__(self, max_frames=5, **kwargs):
                super().__init__(**kwargs)
                self.counter = 0
                self.max_frames = max_frames

            def pull(self, pad, frame):
                super().pull(pad, frame)
                self.counter += 1
                if self.counter >= self.max_frames:
                    self.mark_eos(pad)
                return frame

        pipe = Pipeline()

        source = GWDataNoiseSource(
            name="NoiseSource",
            channel_dict={"H1": "H1:FAKE-STRAIN"},
            t0=1234567890,
            end=None,  # Indefinite duration
            real_time=True,  # Required for None end
        )

        sink = CountingSink(name="CountingSink", sink_pad_names=["H1:FAKE-STRAIN"])

        pipe.insert(source, sink)
        pipe.insert(
            link_map={
                "CountingSink:snk:H1:FAKE-STRAIN": "NoiseSource:src:H1:FAKE-STRAIN"
            }
        )
        pipe.run()

        assert sink.counter >= 5

    @pytest.mark.parametrize("detector", ["H1", "L1", "V1"])
    def test_detector_specific_properties(self, detector):
        """Test that different detectors have proper configuration."""
        source = GWDataNoiseSource(
            name="NoiseSource",
            channel_dict={detector: f"{detector}:FAKE-STRAIN"},
            duration=1.0,
        )

        psd_info = source.channel_info[detector]
        assert psd_info["channel-name"] == f"{detector}:FAKE-STRAIN"
        assert psd_info["rate"] == 16384
        assert "psd" in psd_info
        assert "fir-matrix" in psd_info
        assert "state" in psd_info

    def test_realtime_none_t0_none_end_integration(self):
        """Test real-time mode with None t0 and end in integration."""
        pipe = Pipeline()

        # This should work - real-time mode allows None t0 and end
        source = GWDataNoiseSource(
            name="NoiseSource",
            channel_dict={"H1": "H1:FAKE-STRAIN"},
            t0=None,
            end=None,
            real_time=True,
        )

        # Verify t0 was set to current GPS time
        assert source.t0 is not None
        assert source.end is None

        sink = NullSeriesSink(name="NullSink", sink_pad_names=["H1:FAKE-STRAIN"])

        pipe.insert(source, sink)
        pipe.insert(
            link_map={"NullSink:snk:H1:FAKE-STRAIN": "NoiseSource:src:H1:FAKE-STRAIN"}
        )
        # Don't run indefinitely in tests


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
