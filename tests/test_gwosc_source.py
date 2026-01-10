"""Tests for the GWOSCSource element."""

from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from gwpy.timeseries import TimeSeries
from sgn.apps import Pipeline
from sgn.sinks import CollectSink

from sgnligo.sources.gwosc import GWOSCSource


@pytest.fixture
def mock_fetch_real_object():
    """Mock fetch_open_data but return a REAL TimeSeries object."""
    with patch("sgnligo.sources.gwosc.TimeSeries") as mock_ts_cls:

        def side_effect(det, start, end, **kwargs):
            duration = end - start
            rate = 4096 * u.Hz
            samples = int(duration * 4096)
            data = np.arange(samples, dtype=float)
            if det == "L1":
                data += 1000
            ts = TimeSeries(data, t0=start, sample_rate=rate, name=det)
            return ts

        mock_ts_cls.fetch_open_data.side_effect = side_effect
        yield mock_ts_cls


class TestGwoscBasic:
    def test_init_defaults(self):
        src = GWOSCSource(start=1000, end=1010)
        assert src.detectors == ["H1"]
        assert src.source_pad_names == ["H1"]
        assert src.t0 == 1000
        assert src.block_duration == 1.0

    def test_multi_channel_init(self):
        src = GWOSCSource(start=0, end=10, detectors=["H1", "L1"])
        assert src.source_pad_names == ["H1", "L1"]
        assert "H1" in src.srcs and "L1" in src.srcs


class TestGwoscRateSafety:
    def test_rate_mismatch_raises(self, mock_fetch_real_object):
        """Ensure Source raises ValueError if GWOSC returns wrong rate."""
        src = GWOSCSource(start=0, end=10, sample_rate=2048, batch_duration=10.0)
        with pytest.raises(RuntimeError) as excinfo:
            src.internal()
        assert "4096 Hz" in str(excinfo.value)

    def test_rate_match_ok(self, mock_fetch_real_object):
        src = GWOSCSource(start=0, end=10, sample_rate=4096)
        src.internal()
        assert src._adapters["H1"].size > 0


class TestGwoscData:
    def test_fetch_batching_logic(self, mock_fetch_real_object):
        start = 1000
        end = 1100
        batch = 50.0
        src = GWOSCSource(
            start=start, end=end, batch_duration=batch, min_request_interval=0.0
        )

        # 1. Trigger first fetch
        src.internal()
        assert mock_fetch_real_object.fetch_open_data.call_count == 1

        # 2. Flush buffer to simulate consumption
        src._adapters["H1"].flush_samples(src._adapters["H1"].size)

        # 3. Trigger second fetch
        src.internal()
        assert mock_fetch_real_object.fetch_open_data.call_count == 2
        args, _ = mock_fetch_real_object.fetch_open_data.call_args_list[1]
        assert args[1] == 1050

    def test_rate_limiting(self, mock_fetch_real_object):
        src = GWOSCSource(
            start=0, end=100, batch_duration=1.0, min_request_interval=0.1
        )
        with patch("time.sleep") as mock_sleep:
            src.internal()
            mock_sleep.assert_not_called()
            src._adapters["H1"].flush_samples(src._adapters["H1"].size)
            src.internal()
            assert mock_sleep.called


class TestGwoscIntegration:
    @pytest.mark.integration
    def test_live_gwosc_fetch(self):
        """Hits GWOSC for 4s of GW150914. Requires internet."""
        start = 1126259460
        duration = 4
        end = start + duration

        src = GWOSCSource(
            detectors="H1",
            start=start,
            end=end,
            sample_rate=4096,
            batch_duration=4.0,
            verbose=True,
        )
        src.internal()

        adapter = src._adapters["H1"]
        assert adapter.size == duration * 4096

        # Use copy_samples to peek without removing
        data = adapter.copy_samples(adapter.size)
        assert np.mean(np.abs(data)) > 0


class TestGwoscPipeline:

    @pytest.mark.integration
    def test_pipeline_live_data(self):
        """Run a real SGN pipeline with 1.0s blocks."""
        start = 1126259460
        duration = 4
        end = start + duration

        pipeline = Pipeline()
        src = GWOSCSource(
            detectors="H1",
            start=start,
            end=end,
            sample_rate=4096,
            batch_duration=4.0,
            block_duration=1.0,  # Explicitly set 1s blocks
        )
        sink = CollectSink(
            name="sink",
            sink_pad_names=["H1"],
        )

        pipeline.insert(src, sink)
        pipeline.link(
            {
                sink.snks["H1"]: src.srcs["H1"],
            }
        )

        pipeline.run()

        frames = sink.collects["H1"]
        assert len(frames) >= 4

        # Extract all the data from the deques
        data = [dq[0] for dq in frames]
        total_samples = sum(len(buf.data) for buf in data)
        assert total_samples == duration * 4096


class TestGwoscValidation:
    """Test input validation."""

    def test_end_before_start_raises(self):
        """Line 67: ValueError when end < start."""
        with pytest.raises(ValueError) as excinfo:
            GWOSCSource(start=1000, end=500)
        assert "must be after start time" in str(excinfo.value)

    def test_end_equals_start_raises(self):
        """Line 67: ValueError when end == start."""
        with pytest.raises(ValueError) as excinfo:
            GWOSCSource(start=1000, end=1000)
        assert "must be after start time" in str(excinfo.value)


class TestGwoscVerbose:
    """Test verbose output paths."""

    def test_verbose_rate_limiting(self, mock_fetch_real_object):
        """Line 102: verbose print during rate limiting."""
        from sgnligo.sources.gwosc import _GWOSC_RAM_CACHE

        _GWOSC_RAM_CACHE.clear()

        src = GWOSCSource(
            start=0, end=100, batch_duration=1.0, min_request_interval=0.1, verbose=True
        )
        with (
            patch("builtins.print") as mock_print,
            patch("sgnligo.sources.gwosc.time.sleep"),
        ):
            src.internal()
            src._adapters["H1"].flush_samples(src._adapters["H1"].size)
            src.internal()
            rate_limit_calls = [
                c for c in mock_print.call_args_list if "Rate limiting" in str(c)
            ]
            assert len(rate_limit_calls) > 0

        _GWOSC_RAM_CACHE.clear()

    def test_verbose_cache_hit(self, mock_fetch_real_object):
        """Line 119: verbose print on cache hit."""
        from sgnligo.sources.gwosc import _GWOSC_RAM_CACHE

        _GWOSC_RAM_CACHE.clear()

        # First source populates cache
        src1 = GWOSCSource(
            start=0,
            end=10,
            batch_duration=10.0,
            min_request_interval=0.0,
            cache_data=True,
            verbose=False,
        )
        src1.internal()

        # Second source with same params should hit cache
        src2 = GWOSCSource(
            start=0,
            end=10,
            batch_duration=10.0,
            min_request_interval=0.0,
            cache_data=True,
            verbose=True,
        )
        with patch("builtins.print") as mock_print:
            src2.internal()
            cache_hit_calls = [
                c for c in mock_print.call_args_list if "Cache Hit" in str(c)
            ]
            assert len(cache_hit_calls) > 0

        _GWOSC_RAM_CACHE.clear()


class TestGwoscFrameType:
    """Test frame_type parameter."""

    def test_frame_type_passed_to_fetch(self, mock_fetch_real_object):
        """Line 136: frame_type is passed to fetch_open_data."""
        from sgnligo.sources.gwosc import _GWOSC_RAM_CACHE

        _GWOSC_RAM_CACHE.clear()

        src = GWOSCSource(
            start=0,
            end=10,
            frame_type="H1_HOFT_C01",
            min_request_interval=0.0,
        )
        src.internal()
        assert mock_fetch_real_object.fetch_open_data.called
        call_kwargs = mock_fetch_real_object.fetch_open_data.call_args[1]
        assert call_kwargs.get("frametype") == "H1_HOFT_C01"

        _GWOSC_RAM_CACHE.clear()


class TestGwoscEdgeCases:
    """Test edge cases."""

    def test_fetch_at_end_returns_early(self, mock_fetch_real_object):
        """Line 112: _fetch_next_batch returns early when cursor is at end."""
        src = GWOSCSource(
            start=0, end=10, batch_duration=10.0, min_request_interval=0.0
        )
        src.internal()
        initial_call_count = mock_fetch_real_object.fetch_open_data.call_count
        src._fetch_next_batch("H1")
        assert mock_fetch_real_object.fetch_open_data.call_count == initial_call_count


class TestGwoscVisuals:
    @pytest.mark.skip(reason="Manual verification plot.")
    def test_plot_gwosc_spectrum(self):
        import matplotlib.pyplot as plt
        from scipy import signal

        start = 1126259460
        duration = 32
        end = start + duration
        rate = 4096

        src = GWOSCSource(
            detectors="H1",
            start=start,
            end=end,
            sample_rate=rate,
            batch_duration=32.0,
            verbose=True,
        )
        src.internal()

        data = src._adapters["H1"].copy_samples(duration * rate)
        f, Pxx = signal.welch(data, fs=rate, nperseg=rate)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(data)
        ax1.set_title("H1 Strain Time Series")
        ax2.loglog(f, np.sqrt(Pxx))
        ax2.set_title("ASD")
        ax2.set_xlim(10, 2000)
        plt.tight_layout()
        plt.savefig("gwosc_verification.png")
