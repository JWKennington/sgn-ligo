"""Tests for the InjectedNoiseSource composed element."""

import numpy as np
import pytest
from sgn.apps import Pipeline
from sgn.sinks import CollectSink

from sgnligo.sources import create_injected_noise_source


class TestInjectedNoiseSourceValidation:
    """Test input validation."""

    def test_requires_injection_source(self):
        """Must specify either injection_file or test_mode."""
        with pytest.raises(ValueError, match="Must specify either"):
            create_injected_noise_source(
                name="test",
                ifos=["H1"],
                t0=1000,
                duration=10,
            )

    def test_cannot_specify_both_injection_sources(self):
        """Cannot specify both injection_file and test_mode."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_injected_noise_source(
                name="test",
                ifos=["H1"],
                t0=1000,
                duration=10,
                injection_file="test.xml",
                test_mode="bbh",
            )

    def test_requires_duration_or_end_when_not_realtime(self):
        """Must specify duration or end when real_time=False."""
        with pytest.raises(ValueError, match="Must specify either duration or end"):
            create_injected_noise_source(
                name="test",
                ifos=["H1"],
                t0=1000,
                test_mode="bbh",
                real_time=False,
            )

    def test_realtime_mode_allows_no_duration(self):
        """Real-time mode can omit duration/end for indefinite operation."""
        # Should not raise - t0 is still required by SimInspiralSource
        source = create_injected_noise_source(
            name="test",
            ifos=["H1"],
            t0=1000,  # t0 required by SimInspiralSource
            test_mode="bbh",
            real_time=True,
        )
        assert source is not None


class TestInjectedNoiseSourceCreation:
    """Test source creation and structure."""

    def test_creates_source_with_single_ifo(self):
        """Create source with single detector."""
        source = create_injected_noise_source(
            name="test",
            ifos=["H1"],
            t0=1000,
            duration=10,
            test_mode="bbh",
        )
        assert "H1:STRAIN" in source.srcs
        assert len(source.srcs) == 1

    def test_creates_source_with_multiple_ifos(self):
        """Create source with multiple detectors."""
        source = create_injected_noise_source(
            name="test",
            ifos=["H1", "L1"],
            t0=1000,
            duration=10,
            test_mode="bbh",
        )
        assert "H1:STRAIN" in source.srcs
        assert "L1:STRAIN" in source.srcs
        assert len(source.srcs) == 2

    def test_custom_output_channel_pattern(self):
        """Test custom output channel naming."""
        source = create_injected_noise_source(
            name="test",
            ifos=["H1"],
            t0=1000,
            duration=10,
            test_mode="bbh",
            output_channel_pattern="{ifo}:GDS-CALIB_STRAIN",
        )
        assert "H1:GDS-CALIB_STRAIN" in source.srcs

    def test_all_test_modes(self):
        """Test all supported test modes."""
        for mode in ["bns", "nsbh", "bbh"]:
            source = create_injected_noise_source(
                name=f"test_{mode}",
                ifos=["H1"],
                t0=1000,
                duration=10,
                test_mode=mode,
            )
            assert source is not None


class TestInjectedNoiseSourcePipeline:
    """Test running the source in a pipeline."""

    def test_produces_data_single_ifo(self):
        """Test that source produces data for single IFO."""
        source = create_injected_noise_source(
            name="test",
            ifos=["H1"],
            t0=1000,
            duration=4,
            test_mode="bbh",
        )

        sink = CollectSink(name="sink", sink_pad_names=["H1:STRAIN"])

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        frames = sink.collects["H1:STRAIN"]
        assert len(frames) > 0

        # Extract data from the deques (each entry is a deque with buffer objects)
        data_list = [dq[0] for dq in frames if len(dq) > 0]
        all_data = np.concatenate([buf.data for buf in data_list])

        # Should have data (4 seconds at 16384 Hz = 65536 samples)
        assert len(all_data) == 4 * 16384

        # Data should have variance (noise + possibly injection)
        assert np.std(all_data) > 0

    def test_produces_data_multiple_ifos(self):
        """Test that source produces data for multiple IFOs."""
        source = create_injected_noise_source(
            name="test",
            ifos=["H1", "L1"],
            t0=1000,
            duration=2,
            test_mode="bbh",
        )

        sink = CollectSink(name="sink", sink_pad_names=["H1:STRAIN", "L1:STRAIN"])

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Check both detectors have data
        for ifo in ["H1", "L1"]:
            frames = sink.collects[f"{ifo}:STRAIN"]
            assert len(frames) > 0

            data_list = [dq[0] for dq in frames if len(dq) > 0]
            all_data = np.concatenate([buf.data for buf in data_list])
            assert len(all_data) == 2 * 16384
            assert np.std(all_data) > 0

    def test_different_detectors_have_different_noise(self):
        """Verify H1 and L1 have independent noise realizations."""
        source = create_injected_noise_source(
            name="test",
            ifos=["H1", "L1"],
            t0=1000,
            duration=2,
            test_mode="bbh",
        )

        sink = CollectSink(name="sink", sink_pad_names=["H1:STRAIN", "L1:STRAIN"])

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        h1_frames = sink.collects["H1:STRAIN"]
        l1_frames = sink.collects["L1:STRAIN"]

        h1_data = np.concatenate([dq[0].data for dq in h1_frames if len(dq) > 0])
        l1_data = np.concatenate([dq[0].data for dq in l1_frames if len(dq) > 0])

        # Data should NOT be identical (different noise realizations)
        # Use array_equal for exact comparison (allclose treats tiny values as close)
        assert not np.array_equal(h1_data, l1_data)
