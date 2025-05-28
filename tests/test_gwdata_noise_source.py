#!/usr/bin/env python3
"""Tests for the GWDataNoiseSource class."""

import numpy as np
import pytest
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink, NullSeriesSink

from sgnligo.sources.gwdata_noise_source import GWDataNoiseSource, parse_psd


def test_parse_psd():
    """Test parse_psd function with single and multiple detectors."""
    # Single detector
    result = parse_psd({"H1": "H1:FAKE-STRAIN"})
    assert "H1" in result
    assert result["H1"]["channel-name"] == "H1:FAKE-STRAIN"
    assert result["H1"]["rate"] == 16384
    assert isinstance(result["H1"]["state"], np.ndarray)
    assert isinstance(result["H1"]["fir-matrix"], np.ndarray)

    # Multiple detectors
    result = parse_psd({"H1": "H1:FAKE-STRAIN", "L1": "L1:FAKE-STRAIN"})
    assert "H1" in result and "L1" in result


def test_gwdata_noise_source_init():
    """Test GWDataNoiseSource initialization with various parameters."""
    # Default initialization
    source = GWDataNoiseSource(name="TestSource")
    assert source.channel_dict == {"H1": "H1:FAKE-STRAIN", "L1": "L1:FAKE-STRAIN"}
    assert "H1:FAKE-STRAIN" in source.source_pad_names
    assert source.t0 is not None
    assert not source.verbose and not source.real_time

    # Custom initialization
    custom_t0, custom_end = 1234567890, 1234567900
    source = GWDataNoiseSource(
        name="TestSource",
        channel_dict={"H1": "H1:CUSTOM-STRAIN"},
        t0=custom_t0,
        end=custom_end,
        real_time=True,
        verbose=True,
    )
    assert source.t0 == custom_t0
    assert source.end == custom_end
    assert source.verbose and source.real_time

    # Duration instead of end
    source = GWDataNoiseSource(name="TestSource", t0=custom_t0, duration=10.0)
    assert source.end == custom_t0 + 10.0


def test_generate_noise_chunk():
    """Test noise chunk generation."""
    source = GWDataNoiseSource(name="TestSource", channel_dict={"H1": "H1:FAKE-STRAIN"})

    # Get the actual source pad for H1:FAKE-STRAIN
    h1_pad = source.srcs["H1:FAKE-STRAIN"]

    chunk1 = source._generate_noise_chunk(h1_pad)
    chunk2 = source._generate_noise_chunk(h1_pad)

    assert isinstance(chunk1, np.ndarray)
    assert len(chunk1) == source.channel_info["H1"]["sample-stride"]
    assert not np.array_equal(chunk1, chunk2)  # Different noise each time


def test_basic_pipeline():
    """Test basic pipeline functionality with NullSeriesSink."""
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


def test_pipeline_with_file_output(tmp_path):
    """Test pipeline with file output and data validation."""
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


def test_multiple_detectors():
    """Test source with multiple detectors."""
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


def test_eos_handling():
    """Test End-of-Stream handling with indefinite duration."""

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
    )

    sink = CountingSink(name="CountingSink", sink_pad_names=["H1:FAKE-STRAIN"])

    pipe.insert(source, sink)
    pipe.insert(
        link_map={"CountingSink:snk:H1:FAKE-STRAIN": "NoiseSource:src:H1:FAKE-STRAIN"}
    )
    pipe.run()

    assert sink.counter >= 5


@pytest.mark.parametrize("detector", ["H1", "L1", "V1"])
def test_detector_specific_noise(detector):
    """Test that different detectors have proper configuration."""
    source = GWDataNoiseSource(
        name="NoiseSource",
        channel_dict={detector: f"{detector}:FAKE-STRAIN"},
        duration=1.0,
    )

    # Verify detector-specific configuration exists
    psd_info = source.channel_info[detector]
    assert "psd" in psd_info
    assert "channel-name" in psd_info
    assert psd_info["channel-name"] == f"{detector}:FAKE-STRAIN"
    assert psd_info["rate"] == 16384


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
